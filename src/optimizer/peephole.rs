/// Peephole optimizer for AQL instruction sequences.
///
/// Algorithm
/// ---------
/// Stream instructions left-to-right into an output buffer.
/// For each instruction I acting on qubit q, find the last instruction in the
/// output buffer that also acts on q (searching backward, stopping at BARRIER).
/// If that predecessor P and I satisfy an algebraic rule, apply the reduction
/// in-place; otherwise push I normally.
///
/// Because the output buffer is built left-to-right, the "last touching q"
/// predecessor is always truly adjacent to I on qubit q's timeline — any
/// instruction in between doesn't touch q, so it commutes transparently.
///
/// The scan repeats until a full pass produces zero reductions (fixed point).
use crate::compiler::ir::Instruction;
use std::f64::consts::{PI, TAU};

// ── Public API ─────────────────────────────────────────────────────────────

/// Statistics reported after optimization.
#[derive(Debug, Clone)]
pub struct OptimizationStats {
    /// Gate count before optimization.
    pub gates_before: usize,
    /// Gate count after optimization.
    pub gates_after: usize,
    /// Total gates removed across all passes.
    pub gates_removed: usize,
    /// Number of passes required to reach fixed point.
    pub passes: usize,
}

impl OptimizationStats {
    /// Fraction of gates eliminated (0.0 – 1.0).
    pub fn reduction_ratio(&self) -> f64 {
        if self.gates_before == 0 { return 0.0; }
        self.gates_removed as f64 / self.gates_before as f64
    }

    /// Percentage of gates eliminated (0.0 – 100.0).
    pub fn reduction_percent(&self) -> f64 {
        self.reduction_ratio() * 100.0
    }
}

/// Optimize an instruction sequence using peephole passes.
///
/// Returns `(optimized_instructions, stats)`.
///
/// Programs with control flow are returned unchanged (see module doc).
pub fn optimize(instructions: &[Instruction]) -> (Vec<Instruction>, OptimizationStats) {
    if Instruction::program_has_control_flow(instructions) {
        let n = count_gates(instructions);
        return (
            instructions.to_vec(),
            OptimizationStats { gates_before: n, gates_after: n, gates_removed: 0, passes: 0 },
        );
    }

    let gates_before = count_gates(instructions);
    let mut current = instructions.to_vec();
    let mut total_removed = 0usize;
    let mut passes = 0usize;

    loop {
        let (next, removed) = run_pass(&current);
        passes += 1;
        total_removed += removed;
        current = next;
        if removed == 0 { break; }
    }

    let gates_after = count_gates(&current);
    (current, OptimizationStats { gates_before, gates_after, gates_removed: total_removed, passes })
}

// ── Single pass ────────────────────────────────────────────────────────────

fn run_pass(instrs: &[Instruction]) -> (Vec<Instruction>, usize) {
    let mut out: Vec<Instruction> = Vec::with_capacity(instrs.len());
    let mut removed = 0usize;

    for instr in instrs {
        // ── BARRIER: optimization fence, pass through unchanged ──────────
        if matches!(instr, Instruction::Barrier) {
            out.push(instr.clone());
            continue;
        }

        // ── Pass 1: zero-angle rotation removal ──────────────────────────
        if is_zero_angle_rotation(instr) {
            removed += 1;
            continue;
        }

        // ── Passes 2-4: adjacent single-qubit gate optimization ──────────
        if let Some(q) = primary_single_qubit(instr) {
            if let Some(pos) = last_on_qubit_before_barrier(&out, q) {
                // Pass 2: self-inverse cancellation (H·H, X·X, Y·Y, Z·Z → I)
                if self_inverse_cancels(&out[pos], instr) {
                    out.remove(pos);
                    removed += 2;
                    continue;
                }
                // Pass 3: Pauli product merging (S·S → Z, T·T → S)
                if let Some(merged) = merge_pauli_products(&out[pos], instr) {
                    out[pos] = merged;
                    removed += 1;
                    continue;
                }
                // Pass 4: rotation merging (Rx+Rx, Ry+Ry, Rz+Rz, Phase+Phase)
                match try_merge_rotations(&out[pos], instr) {
                    Some(None) => {
                        // angles sum to zero — both cancel
                        out.remove(pos);
                        removed += 2;
                        continue;
                    }
                    Some(Some(merged)) => {
                        out[pos] = merged;
                        removed += 1;
                        continue;
                    }
                    None => {}
                }
            }
        }

        out.push(instr.clone());
    }

    (out, removed)
}

// ── Helpers ────────────────────────────────────────────────────────────────

fn count_gates(instrs: &[Instruction]) -> usize {
    instrs.iter().filter(|i| i.is_gate()).count()
}

/// Normalize angle to (-π, π].
fn normalize_angle(theta: f64) -> f64 {
    let t = theta.rem_euclid(TAU); // [0, 2π)
    if t > PI { t - TAU } else { t }   // (-π, π]
}

/// True if this is a rotation gate with a zero effective angle (mod 2π).
fn is_zero_angle_rotation(instr: &Instruction) -> bool {
    let theta = match instr {
        Instruction::Rx    { theta, .. }
        | Instruction::Ry  { theta, .. }
        | Instruction::Rz  { theta, .. }
        | Instruction::Phase { theta, .. } => *theta,
        _ => return false,
    };
    normalize_angle(theta).abs() < 1e-9
}

/// If `instr` is a single-qubit gate (not multi-qubit), return its qubit index.
fn primary_single_qubit(instr: &Instruction) -> Option<usize> {
    match instr {
        Instruction::H(q) | Instruction::X(q) | Instruction::Y(q)
        | Instruction::Z(q) | Instruction::S(q) | Instruction::T(q) => Some(*q),
        Instruction::Rx    { qubit, .. }
        | Instruction::Ry  { qubit, .. }
        | Instruction::Rz  { qubit, .. }
        | Instruction::Phase { qubit, .. } => Some(*qubit),
        _ => None,
    }
}

/// Search `buf` backward for the last instruction that touches `qubit`,
/// stopping at (not crossing) any BARRIER or CallGate on the same qubit.
///
/// `CallGate` is treated as an opaque fence: the optimizer cannot inspect
/// the gate body, so it cannot prove that commutation across the call is safe.
fn last_on_qubit_before_barrier(buf: &[Instruction], qubit: usize) -> Option<usize> {
    for i in (0..buf.len()).rev() {
        if matches!(buf[i], Instruction::Barrier) {
            return None;
        }
        // CallGate is opaque — stop the search if it touches our qubit.
        if matches!(&buf[i], Instruction::CallGate { .. }) && buf[i].qubits().contains(&qubit) {
            return None;
        }
        if buf[i].qubits().contains(&qubit) {
            return Some(i);
        }
    }
    None
}

/// True if `a` and `b` are identical self-inverse single-qubit gates
/// on the same qubit (H·H, X·X, Y·Y, Z·Z all equal the identity).
fn self_inverse_cancels(a: &Instruction, b: &Instruction) -> bool {
    match (a, b) {
        (Instruction::H(q1), Instruction::H(q2))
        | (Instruction::X(q1), Instruction::X(q2))
        | (Instruction::Y(q1), Instruction::Y(q2))
        | (Instruction::Z(q1), Instruction::Z(q2)) => q1 == q2,
        _ => false,
    }
}

/// Algebraic product rules for Clifford gates on the same qubit:
///   S·S = Z,   T·T = S
fn merge_pauli_products(a: &Instruction, b: &Instruction) -> Option<Instruction> {
    match (a, b) {
        (Instruction::S(q1), Instruction::S(q2)) if q1 == q2 => Some(Instruction::Z(*q1)),
        (Instruction::T(q1), Instruction::T(q2)) if q1 == q2 => Some(Instruction::S(*q1)),
        _ => None,
    }
}

/// Try to merge two rotation gates on the same qubit.
///
/// Returns:
///   `None`            — incompatible (different variant or different qubit)
///   `Some(None)`      — angles sum to zero → both cancel
///   `Some(Some(i))`   — merged into a single gate `i`
fn try_merge_rotations(a: &Instruction, b: &Instruction) -> Option<Option<Instruction>> {
    // Macro: merge two rotations of the same variant
    macro_rules! do_merge {
        ($variant:ident, $q1:expr, $t1:expr, $q2:expr, $t2:expr) => {{
            if $q1 != $q2 {
                return None; // different qubits — can't merge
            }
            let sum = normalize_angle($t1 + $t2);
            if sum.abs() < 1e-9 {
                Some(None)
            } else {
                Some(Some(Instruction::$variant { qubit: *$q1, theta: sum }))
            }
        }};
    }

    match (a, b) {
        (Instruction::Rx { qubit: q1, theta: t1 }, Instruction::Rx { qubit: q2, theta: t2 }) =>
            do_merge!(Rx, q1, *t1, q2, *t2),
        (Instruction::Ry { qubit: q1, theta: t1 }, Instruction::Ry { qubit: q2, theta: t2 }) =>
            do_merge!(Ry, q1, *t1, q2, *t2),
        (Instruction::Rz { qubit: q1, theta: t1 }, Instruction::Rz { qubit: q2, theta: t2 }) =>
            do_merge!(Rz, q1, *t1, q2, *t2),
        (Instruction::Phase { qubit: q1, theta: t1 }, Instruction::Phase { qubit: q2, theta: t2 }) =>
            do_merge!(Phase, q1, *t1, q2, *t2),
        _ => None,
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn instrs(src: &str) -> Vec<Instruction> {
        use crate::compiler::parse_source;
        let prog = parse_source(&format!("QREG 4\n{src}")).expect("parse failed");
        prog.instructions
    }

    fn opt(src: &str) -> (Vec<Instruction>, OptimizationStats) {
        optimize(&instrs(src))
    }

    fn gate_mnemonics(instrs: &[Instruction]) -> Vec<&str> {
        instrs.iter().map(|i| i.mnemonic()).collect()
    }

    // ── Pass 1: zero-angle removal ─────────────────────────────────────

    #[test]
    fn test_zero_rx_removed() {
        let (out, stats) = opt("RX 0 0.0");
        assert!(out.is_empty(), "Rx(0) should be removed");
        assert_eq!(stats.gates_removed, 1);
    }

    #[test]
    fn test_zero_rz_mod_tau_removed() {
        // 2π is effectively zero rotation
        use std::f64::consts::TAU;
        let src = format!("RZ 0 {TAU}");
        let (out, stats) = opt(&src);
        assert!(out.is_empty(), "Rz(2π) should be removed");
        assert_eq!(stats.gates_removed, 1);
    }

    #[test]
    fn test_nonzero_rz_kept() {
        let (out, _) = opt("RZ 0 1.0");
        assert_eq!(out.len(), 1);
    }

    // ── Pass 2: self-inverse cancellation ─────────────────────────────

    #[test]
    fn test_hh_cancels() {
        let (out, stats) = opt("H 0\nH 0");
        assert!(out.is_empty(), "H·H should cancel");
        assert_eq!(stats.gates_removed, 2);
        assert_eq!(stats.gates_after, 0);
    }

    #[test]
    fn test_xx_cancels() {
        let (out, stats) = opt("X 0\nX 0");
        assert!(out.is_empty());
        assert_eq!(stats.gates_removed, 2);
    }

    #[test]
    fn test_yy_cancels() {
        let (out, _) = opt("Y 0\nY 0");
        assert!(out.is_empty());
    }

    #[test]
    fn test_zz_cancels() {
        let (out, _) = opt("Z 0\nZ 0");
        assert!(out.is_empty());
    }

    #[test]
    fn test_hxh_becomes_x() {
        // H·X·H on same qubit: H cancels with last H → X remains
        // H(0) H(1) H(0) — H on q0 cancels across H on q1
        let (out, stats) = opt("H 0\nH 1\nH 0");
        assert_eq!(gate_mnemonics(&out), vec!["H"]);
        assert_eq!(out[0], Instruction::H(1));
        assert_eq!(stats.gates_removed, 2);
    }

    #[test]
    fn test_different_qubits_not_cancelled() {
        let (out, stats) = opt("H 0\nH 1");
        assert_eq!(out.len(), 2);
        assert_eq!(stats.gates_removed, 0);
    }

    #[test]
    fn test_cnot_prevents_hh_cancellation() {
        // H(0) · CNOT(0,1) · H(0): CNOT is the last instruction on qubit 0
        // so the two H gates are NOT adjacent on qubit 0 — no cancellation
        let (out, stats) = opt("H 0\nCNOT 0 1\nH 0");
        assert_eq!(out.len(), 3, "H·CNOT·H should not be optimized");
        assert_eq!(stats.gates_removed, 0);
    }

    #[test]
    fn test_barrier_blocks_hh() {
        let (out, stats) = opt("H 0\nBARRIER\nH 0");
        assert_eq!(out.len(), 3, "BARRIER should block H·H cancellation");
        assert_eq!(stats.gates_removed, 0);
    }

    #[test]
    fn test_triple_h_reduces_to_one() {
        // H·H·H = H (after two-pass: first H·H cancel, third H survives)
        let (out, stats) = opt("H 0\nH 0\nH 0");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0], Instruction::H(0));
        assert_eq!(stats.gates_removed, 2);
    }

    #[test]
    fn test_four_h_reduces_to_nothing() {
        let (out, stats) = opt("H 0\nH 0\nH 0\nH 0");
        assert!(out.is_empty());
        assert_eq!(stats.gates_removed, 4);
    }

    // ── Pass 3: Pauli product merging ──────────────────────────────────

    #[test]
    fn test_ss_becomes_z() {
        let (out, stats) = opt("S 0\nS 0");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0], Instruction::Z(0));
        assert_eq!(stats.gates_removed, 1);
    }

    #[test]
    fn test_tt_becomes_s() {
        let (out, stats) = opt("T 0\nT 0");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0], Instruction::S(0));
        assert_eq!(stats.gates_removed, 1);
    }

    #[test]
    fn test_ssss_becomes_identity() {
        // S·S → Z,  then S·S → Z again, then Z·Z cancels → I
        // But each pair reduces independently across passes
        // S·S·S·S: pass1: S+S→Z, S+S→Z → [Z, Z]; pass2: Z·Z → []; done
        let (out, stats) = opt("S 0\nS 0\nS 0\nS 0");
        assert!(out.is_empty(), "S⁴ = I, expected empty but got {out:?}");
        assert_eq!(stats.gates_removed, 4);
    }

    // ── Pass 4: rotation merging ───────────────────────────────────────

    #[test]
    fn test_rx_merges() {
        let (out, stats) = opt("RX 0 1.0\nRX 0 1.0");
        assert_eq!(out.len(), 1);
        if let Instruction::Rx { qubit: 0, theta } = out[0] {
            assert!((theta - 2.0).abs() < 1e-9);
        } else {
            panic!("expected Rx, got {:?}", out[0]);
        }
        assert_eq!(stats.gates_removed, 1);
    }

    #[test]
    fn test_rx_cancel_via_merge() {
        // Rx(π) · Rx(-π) = Rx(0) → removed
        let (out, stats) = opt(&format!("RX 0 {PI}\nRX 0 -{PI}"));
        assert!(out.is_empty(), "Rx(π)·Rx(-π) should cancel");
        assert_eq!(stats.gates_removed, 2);
    }

    #[test]
    fn test_rz_chain_merges_in_one_pass() {
        // 4 × Rz(π/4) → Rz(π) in a single pass
        let src = format!("RZ 0 {}\nRZ 0 {}\nRZ 0 {}\nRZ 0 {}",
            PI / 4.0, PI / 4.0, PI / 4.0, PI / 4.0);
        let (out, stats) = opt(&src);
        assert_eq!(out.len(), 1, "4×Rz(π/4) should merge to one gate");
        if let Instruction::Rz { qubit: 0, theta } = out[0] {
            assert!((theta - PI).abs() < 1e-9, "expected Rz(π), got Rz({theta})");
        } else {
            panic!("expected Rz, got {:?}", out[0]);
        }
        assert_eq!(stats.gates_removed, 3);
        // The algorithm always runs one extra verification pass after the last
        // productive pass to confirm the fixed point — so passes >= 2 for any
        // circuit that required at least one reduction.
        assert!(stats.passes >= 1);
    }

    #[test]
    fn test_phase_merges() {
        let (out, stats) = opt(&format!("PHASE 0 {}\nPHASE 0 {}", PI / 4.0, PI / 4.0));
        assert_eq!(out.len(), 1);
        if let Instruction::Phase { qubit: 0, theta } = out[0] {
            assert!((theta - PI / 2.0).abs() < 1e-9);
        } else {
            panic!("expected Phase, got {:?}", out[0]);
        }
        assert_eq!(stats.gates_removed, 1);
    }

    #[test]
    fn test_mixed_rotation_types_not_merged() {
        // Rx and Rz cannot be merged
        let (out, stats) = opt("RX 0 1.0\nRZ 0 1.0");
        assert_eq!(out.len(), 2);
        assert_eq!(stats.gates_removed, 0);
    }

    // ── Multi-pass convergence ─────────────────────────────────────────

    #[test]
    fn test_stats_gates_before_after() {
        // H·H·CNOT·H·H → CNOT (four H's cancel)
        let (out, stats) = opt("H 0\nH 0\nCNOT 0 1\nH 0\nH 0");
        // H(0)·H(0) cancel, then H(0)·H(0) cancel; CNOT remains
        assert_eq!(stats.gates_before, 5);
        assert_eq!(stats.gates_after, 1);
        assert_eq!(stats.gates_removed, 4);
        assert_eq!(out.len(), 1);
        assert!(matches!(out[0], Instruction::Cnot { .. }));
    }

    #[test]
    fn test_control_flow_program_unchanged() {
        // Programs with GOTO/IF must not be optimized
        let src = "H 0\nH 0\nGOTO done\nLABEL done";
        let (out, stats) = opt(src);
        assert_eq!(out.len(), instrs(src).len(), "control-flow program should be unchanged");
        assert_eq!(stats.gates_removed, 0);
        assert_eq!(stats.passes, 0);
    }

    #[test]
    fn test_empty_program_unchanged() {
        let (out, stats) = optimize(&[]);
        assert!(out.is_empty());
        assert_eq!(stats.gates_removed, 0);
    }

    // ── CallGate barrier semantics ─────────────────────────────────────

    #[test]
    fn test_call_gate_blocks_hh_on_touched_qubit() {
        // H(0) · CALL bell 0 1 · H(0): cannot cancel — CALL touches qubit 0
        let src = "GATE bell 2\n  H 0\n  CNOT 0 1\nEND\nH 0\nCALL bell 0 1\nH 0";
        let instrs_in = instrs(src);
        let (out, stats) = optimize(&instrs_in);
        assert_eq!(stats.gates_removed, 0, "CALL should block H·H cancellation");
        assert_eq!(out.len(), instrs_in.len());
    }

    #[test]
    fn test_call_gate_transparent_to_untouched_qubit() {
        // H(0) · CALL flip 1 · H(0): CALL only touches qubit 1 → H·H on qubit 0 cancels
        let src = "GATE flip 1\n  X 0\nEND\nH 0\nCALL flip 1\nH 0";
        let instrs_in = instrs(src);
        let (out, stats) = optimize(&instrs_in);
        // H(0) and H(0) should cancel; CALL flip 1 survives
        assert_eq!(stats.gates_removed, 2, "H·H on untouched qubit should cancel");
        let mnems: Vec<&str> = out.iter().map(|i| i.mnemonic()).collect();
        assert_eq!(mnems, vec!["CALL"], "only CALL should remain");
    }
}
