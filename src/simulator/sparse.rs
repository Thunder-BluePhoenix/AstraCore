/// Sparse statevector simulator for AstraCore.
///
/// Stores only the non-zero amplitudes of the quantum state as a
/// `HashMap<u64, Complex>` keyed by basis-state index.  For circuits
/// that begin as product states and have few entangling gates (or many
/// mid-circuit measurements that collapse branches), this can be orders
/// of magnitude faster and more memory-efficient than the dense statevector.
///
/// ## Scaling
/// - Memory: O(k) where k = number of non-zero amplitudes (k ≤ 2ⁿ)
/// - Single-qubit gate: O(k) — iterate over entries, update in pairs
/// - CNOT / CZ / SWAP: O(k) — permute or phase-flip entries
/// - Excellent for: product states, circuits with early measurements,
///   Grover with few iterations, circuits with mid-circuit resets
///
/// ## Limitations
/// - Worst case (fully entangled state) = same as dense: 2ⁿ entries
/// - No AVX2 acceleration (scalar ops on HashMap entries)
/// - Sparse → dense threshold not auto-applied (backend chosen by user)
use std::collections::HashMap;
use crate::core::{Complex, gates};
use crate::compiler::{AqlError, ir::{Instruction, Program}};
use crate::runtime::{ExecutionResult, MeasurementRecord};

// ── SparseState ───────────────────────────────────────────────────────────

/// Sparse quantum state: `HashMap<basis_index, amplitude>`.
///
/// Invariant: only amplitudes with `|a|² > PRUNE_THRESHOLD` are stored.
pub struct SparseState {
    /// Non-zero amplitudes keyed by basis state index.
    pub amplitudes: HashMap<u64, Complex>,
    /// Number of qubits in the register.
    pub n_qubits: usize,
}

/// Amplitudes smaller than this are pruned to keep the map compact.
const PRUNE_THRESHOLD: f64 = 1e-15;

impl SparseState {
    /// Create a new sparse state initialized to |0…0⟩.
    pub fn new(n_qubits: usize) -> Self {
        let mut amplitudes = HashMap::with_capacity(4);
        amplitudes.insert(0u64, Complex::one());
        Self { amplitudes, n_qubits }
    }

    /// Apply a single-qubit 2×2 unitary to `target`.
    ///
    /// For each pair (|…0…⟩, |…1…⟩) differing only in the target bit,
    /// mixes the two amplitudes using the gate matrix.
    pub fn apply_single_qubit(&mut self, target: usize, mat: &gates::Matrix2x2) {
        let bit = 1u64 << target;

        // Collect the set of "base keys" (target bit = 0) that have at least
        // one non-zero partner.
        let base_keys: Vec<u64> = self.amplitudes.keys()
            .map(|k| k & !bit)
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        let mut new_amps: HashMap<u64, Complex> =
            HashMap::with_capacity(self.amplitudes.len() * 2);

        for base in base_keys {
            let key0 = base;         // target = 0
            let key1 = base | bit;   // target = 1

            let a0 = self.amplitudes.get(&key0).copied().unwrap_or(Complex::zero());
            let a1 = self.amplitudes.get(&key1).copied().unwrap_or(Complex::zero());

            let new0 = mat[0][0] * a0 + mat[0][1] * a1;
            let new1 = mat[1][0] * a0 + mat[1][1] * a1;

            if new0.norm_sq() > PRUNE_THRESHOLD { new_amps.insert(key0, new0); }
            if new1.norm_sq() > PRUNE_THRESHOLD { new_amps.insert(key1, new1); }
        }

        self.amplitudes = new_amps;
    }

    /// Apply a CNOT gate: flip `target` when `control` is |1⟩.
    pub fn apply_cnot(&mut self, control: usize, target: usize) {
        let ctrl_bit = 1u64 << control;
        let tgt_bit  = 1u64 << target;
        let mut new_amps: HashMap<u64, Complex> =
            HashMap::with_capacity(self.amplitudes.len());
        for (&key, &amp) in &self.amplitudes {
            let new_key = if key & ctrl_bit != 0 { key ^ tgt_bit } else { key };
            new_amps.insert(new_key, amp);
        }
        self.amplitudes = new_amps;
    }

    /// Apply a CZ gate: phase-flip |11⟩ component.
    pub fn apply_cz(&mut self, control: usize, target: usize) {
        let ctrl_bit = 1u64 << control;
        let tgt_bit  = 1u64 << target;
        for (&key, amp) in self.amplitudes.iter_mut() {
            if key & ctrl_bit != 0 && key & tgt_bit != 0 {
                amp.re = -amp.re;
                amp.im = -amp.im;
            }
        }
    }

    /// Apply a SWAP gate: exchange the values of qubit `qa` and qubit `qb`.
    pub fn apply_swap(&mut self, qa: usize, qb: usize) {
        let bit_a = 1u64 << qa;
        let bit_b = 1u64 << qb;
        let mut new_amps: HashMap<u64, Complex> =
            HashMap::with_capacity(self.amplitudes.len());
        for (&key, &amp) in &self.amplitudes {
            let a_val = (key >> qa) & 1;
            let b_val = (key >> qb) & 1;
            let new_key = if a_val != b_val {
                (key & !(bit_a | bit_b)) | (a_val << qb) | (b_val << qa)
            } else {
                key
            };
            new_amps.insert(new_key, amp);
        }
        self.amplitudes = new_amps;
    }

    /// Apply a Toffoli (CCX) gate: flip `target` when both controls are |1⟩.
    pub fn apply_toffoli(&mut self, c0: usize, c1: usize, target: usize) {
        let c0_bit  = 1u64 << c0;
        let c1_bit  = 1u64 << c1;
        let tgt_bit = 1u64 << target;
        let mut new_amps: HashMap<u64, Complex> =
            HashMap::with_capacity(self.amplitudes.len());
        for (&key, &amp) in &self.amplitudes {
            let new_key = if key & c0_bit != 0 && key & c1_bit != 0 {
                key ^ tgt_bit
            } else {
                key
            };
            new_amps.insert(new_key, amp);
        }
        self.amplitudes = new_amps;
    }

    /// Measure `qubit`, collapsing the state.  `rng` ∈ [0, 1) is the random
    /// draw (use `rand::random::<f64>()` in callers).
    ///
    /// Returns the measurement outcome (`false` = |0⟩, `true` = |1⟩).
    pub fn measure(&mut self, qubit: usize, rng: f64) -> bool {
        let bit = 1u64 << qubit;

        let prob_one: f64 = self.amplitudes.iter()
            .filter(|(k, _)| *k & bit != 0)
            .map(|(_, a)| a.norm_sq())
            .sum();

        let outcome = rng < prob_one;
        let keep_mask = if outcome { bit } else { 0 };

        // Collapse: retain only amplitudes consistent with the outcome.
        self.amplitudes.retain(|k, _| k & bit == keep_mask);

        // Re-normalize.
        let norm_sq: f64 = self.amplitudes.values().map(|a| a.norm_sq()).sum();
        let norm = norm_sq.sqrt();
        if norm > 1e-15 {
            let inv = 1.0 / norm;
            for a in self.amplitudes.values_mut() {
                a.re *= inv;
                a.im *= inv;
            }
        }

        outcome
    }

    /// Return a dense probability vector of length 2^n.
    pub fn probabilities(&self) -> Vec<f64> {
        let n_states = 1usize << self.n_qubits;
        let mut probs = vec![0.0f64; n_states];
        for (&key, a) in &self.amplitudes {
            probs[key as usize] = a.norm_sq();
        }
        probs
    }

    /// Number of non-zero amplitude entries (sparsity metric).
    pub fn nnz(&self) -> usize {
        self.amplitudes.len()
    }
}

// ── Executor ──────────────────────────────────────────────────────────────

/// Execute an AQL `Program` using the sparse statevector backend.
///
/// # When to prefer this backend
/// - Circuits that start in |0…0⟩ and apply only a few entangling gates
/// - Circuits with many mid-circuit measurements (each collapses branches)
/// - Large circuits that stay sparse (product-state layers, etc.)
///
/// # Limitations
/// - Fully-entangled circuits degrade to O(2ⁿ) memory, same as dense
/// - No noise model support in this implementation
/// - CallGate with user-defined bodies expands inline (fully supported)
pub fn execute_sparse(program: &Program) -> Result<ExecutionResult, AqlError> {
    const MAX_STEPS: usize = 1_000_000;

    // Build label table
    let mut label_table: HashMap<String, usize> = HashMap::new();
    for (i, instr) in program.instructions.iter().enumerate() {
        if let Instruction::Label(name) = instr {
            label_table.insert(name.clone(), i + 1);
        }
    }

    let mut state = SparseState::new(program.num_qubits);
    let mut classical: Vec<Option<bool>> = vec![None; program.num_qubits];
    let mut measurements: Vec<MeasurementRecord> = Vec::new();
    let mut pre_measurement_probs: Option<Vec<f64>> = None;
    let mut gate_count    = 0usize;
    let mut branch_count  = 0usize;
    let mut steps_executed = 0usize;
    let mut first_measure  = true;

    let mut pc: usize = 0;

    while pc < program.instructions.len() {
        if steps_executed >= MAX_STEPS {
            return Err(AqlError::Runtime {
                msg: format!(
                    "execution exceeded {MAX_STEPS} steps — possible infinite loop"
                ),
            });
        }
        steps_executed += 1;

        let instr = &program.instructions[pc];

        // Snapshot before first measurement
        if instr.is_measurement() && first_measure {
            pre_measurement_probs = Some(state.probabilities());
            first_measure = false;
        }

        match instr {
            // ── Single-qubit gates ──────────────────────────────────────
            Instruction::H(q) => {
                state.apply_single_qubit(*q, &gates::hadamard());
                gate_count += 1;
            }
            Instruction::X(q) => {
                state.apply_single_qubit(*q, &gates::pauli_x());
                gate_count += 1;
            }
            Instruction::Y(q) => {
                state.apply_single_qubit(*q, &gates::pauli_y());
                gate_count += 1;
            }
            Instruction::Z(q) => {
                state.apply_single_qubit(*q, &gates::pauli_z());
                gate_count += 1;
            }
            Instruction::S(q) => {
                state.apply_single_qubit(*q, &gates::s_gate());
                gate_count += 1;
            }
            Instruction::T(q) => {
                state.apply_single_qubit(*q, &gates::t_gate());
                gate_count += 1;
            }
            Instruction::Rx { qubit, theta } => {
                state.apply_single_qubit(*qubit, &gates::rx(*theta));
                gate_count += 1;
            }
            Instruction::Ry { qubit, theta } => {
                state.apply_single_qubit(*qubit, &gates::ry(*theta));
                gate_count += 1;
            }
            Instruction::Rz { qubit, theta } => {
                state.apply_single_qubit(*qubit, &gates::rz(*theta));
                gate_count += 1;
            }
            Instruction::Phase { qubit, theta } => {
                state.apply_single_qubit(*qubit, &gates::phase_gate(*theta));
                gate_count += 1;
            }

            // ── Multi-qubit gates ───────────────────────────────────────
            Instruction::Cnot { control, target } => {
                state.apply_cnot(*control, *target);
                gate_count += 1;
            }
            Instruction::Cz { control, target } => {
                state.apply_cz(*control, *target);
                gate_count += 1;
            }
            Instruction::Swap { qubit_a, qubit_b } => {
                state.apply_swap(*qubit_a, *qubit_b);
                gate_count += 1;
            }
            Instruction::Toffoli { control0, control1, target } => {
                state.apply_toffoli(*control0, *control1, *target);
                gate_count += 1;
            }

            // ── Measurement ─────────────────────────────────────────────
            Instruction::Measure(q) => {
                let outcome = state.measure(*q, rand::random());
                classical[*q] = Some(outcome);
                measurements.push(MeasurementRecord { qubit: *q, outcome, step: pc });
            }
            Instruction::MeasureAll => {
                for q in 0..program.num_qubits {
                    let outcome = state.measure(q, rand::random());
                    classical[q] = Some(outcome);
                    measurements.push(MeasurementRecord { qubit: q, outcome, step: pc });
                }
            }

            // ── Structural ──────────────────────────────────────────────
            Instruction::Barrier => {}
            Instruction::Label(_) => {}

            // ── Custom gate call ────────────────────────────────────────
            Instruction::CallGate { name, qubits: arg_qubits } => {
                let def = program.gate_defs.get(name.as_str()).ok_or_else(|| {
                    AqlError::Runtime {
                        msg: format!("undefined gate '{name}' at runtime"),
                    }
                })?;
                for body_instr in &def.body.clone() {
                    apply_remapped_sparse(body_instr, arg_qubits, &mut state)?;
                    gate_count += 1;
                }
            }

            // ── Control flow ─────────────────────────────────────────────
            Instruction::Goto { label } => {
                let &target = label_table.get(label.as_str()).ok_or_else(|| {
                    AqlError::Runtime { msg: format!("undefined label '{label}'") }
                })?;
                pc = target;
                continue;
            }
            Instruction::GotoIf { qubit, label } => {
                let measured = classical[*qubit].ok_or_else(|| AqlError::Runtime {
                    msg: format!("qubit {qubit} used in IF before measurement"),
                })?;
                if measured {
                    let &target = label_table.get(label.as_str()).ok_or_else(|| {
                        AqlError::Runtime { msg: format!("undefined label '{label}'") }
                    })?;
                    pc = target;
                    branch_count += 1;
                    continue;
                }
            }
            Instruction::GotoIfNot { qubit, label } => {
                let measured = classical[*qubit].ok_or_else(|| AqlError::Runtime {
                    msg: format!("qubit {qubit} used in IFNOT before measurement"),
                })?;
                if !measured {
                    let &target = label_table.get(label.as_str()).ok_or_else(|| {
                        AqlError::Runtime { msg: format!("undefined label '{label}'") }
                    })?;
                    pc = target;
                    branch_count += 1;
                    continue;
                }
            }
        }

        pc += 1;
    }

    let final_probabilities = state.probabilities();

    Ok(ExecutionResult {
        num_qubits: program.num_qubits,
        measurements,
        pre_measurement_probs,
        final_probabilities,
        gate_count,
        branch_count,
        steps_executed,
    })
}

// ── Helpers ───────────────────────────────────────────────────────────────

/// Apply a gate body instruction to the sparse state, remapping local qubit
/// indices to global indices via `qubit_map`.
fn apply_remapped_sparse(
    instr: &Instruction,
    qubit_map: &[usize],
    state: &mut SparseState,
) -> Result<(), AqlError> {
    let q = |local: usize| -> usize { qubit_map[local] };
    match instr {
        Instruction::H(lq)     => state.apply_single_qubit(q(*lq), &gates::hadamard()),
        Instruction::X(lq)     => state.apply_single_qubit(q(*lq), &gates::pauli_x()),
        Instruction::Y(lq)     => state.apply_single_qubit(q(*lq), &gates::pauli_y()),
        Instruction::Z(lq)     => state.apply_single_qubit(q(*lq), &gates::pauli_z()),
        Instruction::S(lq)     => state.apply_single_qubit(q(*lq), &gates::s_gate()),
        Instruction::T(lq)     => state.apply_single_qubit(q(*lq), &gates::t_gate()),
        Instruction::Rx { qubit, theta } => state.apply_single_qubit(q(*qubit), &gates::rx(*theta)),
        Instruction::Ry { qubit, theta } => state.apply_single_qubit(q(*qubit), &gates::ry(*theta)),
        Instruction::Rz { qubit, theta } => state.apply_single_qubit(q(*qubit), &gates::rz(*theta)),
        Instruction::Phase { qubit, theta } => state.apply_single_qubit(q(*qubit), &gates::phase_gate(*theta)),
        Instruction::Cnot { control, target } => state.apply_cnot(q(*control), q(*target)),
        Instruction::Cz   { control, target } => state.apply_cz(q(*control), q(*target)),
        Instruction::Swap { qubit_a, qubit_b } => state.apply_swap(q(*qubit_a), q(*qubit_b)),
        Instruction::Toffoli { control0, control1, target } =>
            state.apply_toffoli(q(*control0), q(*control1), q(*target)),
        Instruction::Barrier | Instruction::Label(_) => {}
        _ => return Err(AqlError::Runtime {
            msg: "unsupported instruction in custom gate body (sparse backend)".into(),
        }),
    }
    Ok(())
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::parse_source;

    fn nearly_eq(a: f64, b: f64) -> bool { (a - b).abs() < 1e-9 }

    #[test]
    fn test_sparse_initial_state_is_zero() {
        let s = SparseState::new(3);
        let probs = s.probabilities();
        assert!(nearly_eq(probs[0], 1.0), "P(|000⟩) should be 1.0");
        assert!(probs[1..].iter().all(|&p| p < 1e-15));
        assert_eq!(s.nnz(), 1);
    }

    #[test]
    fn test_sparse_h_gate_creates_superposition() {
        let mut s = SparseState::new(1);
        s.apply_single_qubit(0, &gates::hadamard());
        let probs = s.probabilities();
        assert!(nearly_eq(probs[0], 0.5));
        assert!(nearly_eq(probs[1], 0.5));
        assert_eq!(s.nnz(), 2);
    }

    #[test]
    fn test_sparse_x_gate_flips_qubit() {
        let mut s = SparseState::new(2);
        s.apply_single_qubit(0, &gates::pauli_x());
        let probs = s.probabilities();
        // |00⟩ → |01⟩ (qubit 0 is LSB, so state index 1)
        assert!(nearly_eq(probs[1], 1.0));
        assert!(nearly_eq(probs[0], 0.0));
    }

    #[test]
    fn test_sparse_cnot_bell_state() {
        // H(0) + CNOT(0,1) → Bell state (|00⟩+|11⟩)/√2
        let mut s = SparseState::new(2);
        s.apply_single_qubit(0, &gates::hadamard());
        s.apply_cnot(0, 1);
        let probs = s.probabilities();
        assert!(nearly_eq(probs[0], 0.5), "P(|00⟩)={}", probs[0]);
        assert!(nearly_eq(probs[3], 0.5), "P(|11⟩)={}", probs[3]);
        assert!(nearly_eq(probs[1], 0.0));
        assert!(nearly_eq(probs[2], 0.0));
    }

    #[test]
    fn test_sparse_cz_phase() {
        // |11⟩ → -|11⟩ under CZ
        let mut s = SparseState::new(2);
        s.apply_single_qubit(0, &gates::pauli_x()); // |01⟩
        s.apply_single_qubit(1, &gates::pauli_x()); // |11⟩
        s.apply_cz(0, 1);
        // Probability is invariant under global phase
        let probs = s.probabilities();
        assert!(nearly_eq(probs[3], 1.0));
    }

    #[test]
    fn test_sparse_swap() {
        // |01⟩ (q0=1, q1=0) → swap → |10⟩ (q0=0, q1=1)
        let mut s = SparseState::new(2);
        s.apply_single_qubit(0, &gates::pauli_x()); // index 1 = |01⟩
        s.apply_swap(0, 1);
        let probs = s.probabilities();
        assert!(nearly_eq(probs[2], 1.0), "P(|10⟩)={}", probs[2]);
    }

    #[test]
    fn test_sparse_toffoli() {
        // |110⟩ → CCX(0,1,2) → |111⟩
        let mut s = SparseState::new(3);
        s.apply_single_qubit(0, &gates::pauli_x());
        s.apply_single_qubit(1, &gates::pauli_x());
        s.apply_toffoli(0, 1, 2);
        let probs = s.probabilities();
        // q0=1,q1=1,q2=1 → index = 0b111 = 7
        assert!(nearly_eq(probs[7], 1.0), "P(|111⟩)={}", probs[7]);
    }

    #[test]
    fn test_sparse_hh_cancels() {
        let mut s = SparseState::new(1);
        s.apply_single_qubit(0, &gates::hadamard());
        s.apply_single_qubit(0, &gates::hadamard());
        let probs = s.probabilities();
        assert!(nearly_eq(probs[0], 1.0)); // back to |0⟩
        assert!(nearly_eq(probs[1], 0.0));
    }

    #[test]
    fn test_execute_sparse_bell_circuit() {
        let prog = parse_source("QREG 2\nH 0\nCNOT 0 1\nMEASURE_ALL").unwrap();
        let result = execute_sparse(&prog).unwrap();
        assert_eq!(result.num_qubits, 2);
        assert_eq!(result.measurements.len(), 2);
        // Both qubits must agree (both 0 or both 1)
        let q0 = result.measurements[0].outcome;
        let q1 = result.measurements[1].outcome;
        assert_eq!(q0, q1, "Bell state: q0 and q1 must agree");
    }

    #[test]
    fn test_execute_sparse_x_measure_gives_one() {
        let prog = parse_source("QREG 1\nX 0\nMEASURE 0").unwrap();
        for _ in 0..20 {
            let result = execute_sparse(&prog).unwrap();
            assert!(result.measurements[0].outcome, "X|0⟩ must measure as |1⟩");
        }
    }

    #[test]
    fn test_execute_sparse_ghz_state() {
        let prog = parse_source(
            "QREG 3\nH 0\nCNOT 0 1\nCNOT 0 2\nMEASURE_ALL"
        ).unwrap();
        for _ in 0..30 {
            let result = execute_sparse(&prog).unwrap();
            let outcomes: Vec<bool> = result.measurements.iter().map(|m| m.outcome).collect();
            assert!(
                outcomes.iter().all(|&o| o == outcomes[0]),
                "GHZ: all qubits must agree, got {outcomes:?}"
            );
        }
    }

    #[test]
    fn test_execute_sparse_matches_dense_probabilities() {
        // Compare sparse probabilities with dense state for a small circuit.
        use crate::core::Simulator;
        let src = "QREG 3\nH 0\nH 1\nCNOT 0 2\nCNOT 1 2";
        let prog = parse_source(src).unwrap();
        let sparse_result = execute_sparse(&prog).unwrap();

        let mut sim = Simulator::new(3);
        sim.h(0); sim.h(1); sim.cnot(0, 2); sim.cnot(1, 2);
        let dense_probs = sim.probabilities();

        for (i, (&sp, &dp)) in sparse_result.final_probabilities.iter()
            .zip(dense_probs.iter()).enumerate()
        {
            assert!(
                (sp - dp).abs() < 1e-9,
                "state {i}: sparse={sp:.6} dense={dp:.6}"
            );
        }
    }

    #[test]
    fn test_execute_sparse_custom_gate() {
        // Custom gate: GATE bell 2 / H 0 / CNOT 0 1 / END
        let prog = parse_source(
            "GATE bell 2\n  H 0\n  CNOT 0 1\nEND\nQREG 2\nCALL bell 0 1\nMEASURE_ALL"
        ).unwrap();
        for _ in 0..20 {
            let result = execute_sparse(&prog).unwrap();
            let q0 = result.measurements[0].outcome;
            let q1 = result.measurements[1].outcome;
            assert_eq!(q0, q1, "Custom Bell gate: q0 and q1 must agree");
        }
    }
}
