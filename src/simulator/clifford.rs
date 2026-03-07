/// Stabilizer / Clifford-circuit simulator.
///
/// Uses the Aaronson-Gottesman CHP algorithm (2004).
/// Binary symplectic tableau: O(n²) memory, O(n) per gate.
///
/// # Supported gates
/// H, S, X, Y, Z, CNOT, CZ, SWAP
///
/// # Unsupported (non-Clifford)
/// T, Rx, Ry, Rz, Phase → `AqlError::Runtime`
///
/// # Tableau layout
/// 2n rows × n columns.
/// - Rows 0..n:    destabilizer generators
/// - Rows n..2n:   stabilizer generators
///
/// Each row encodes a Pauli string  (-1)^r · ∏_j (i^{x_j z_j} X_j^{x_j} Z_j^{z_j}).
/// With this convention (1,1) encodes Y directly (i·XZ = Y).
use std::collections::HashMap;
use crate::compiler::{AqlError, ir::{Instruction, Program}};
use crate::runtime::{ExecutionResult, MeasurementRecord};

// ── Clifford state ─────────────────────────────────────────────────────────

pub struct CliffordState {
    pub n: usize,
    x: Vec<Vec<bool>>,  // x[row][qubit]
    z: Vec<Vec<bool>>,  // z[row][qubit]
    r: Vec<bool>,       // r[row]: false = +1, true = -1
}

impl CliffordState {
    /// Initialise to |0…0⟩.
    /// Destabilizers: X_i for i in 0..n
    /// Stabilizers:   Z_i for i in 0..n
    pub fn new(n: usize) -> Self {
        assert!(n >= 1, "need at least one qubit");
        let mut x = vec![vec![false; n]; 2 * n];
        let mut z = vec![vec![false; n]; 2 * n];
        for i in 0..n { x[i][i] = true; }       // destabilizers: X_i
        for i in 0..n { z[n + i][i] = true; }   // stabilizers:   Z_i
        Self { n, x, z, r: vec![false; 2 * n] }
    }

    // ── Phase contribution ────────────────────────────────────────────────

    /// Phase contribution (in units of i) from multiplying single-qubit Paulis
    /// P1·P2 where P_k = i^{x_k z_k} X^{x_k} Z^{z_k}.
    /// Returns a value in {-1, 0, 1}.
    #[inline(always)]
    fn g(x1: bool, z1: bool, x2: bool, z2: bool) -> i32 {
        match (x1, z1, x2, z2) {
            (false, false, _, _) | (_, _, false, false) => 0, // I·anything or anything·I
            (true,  false, true,  false) => 0,  // X·X = I
            (true,  false, false, true)  => -1, // X·Z = −iY
            (true,  false, true,  true)  => 1,  // X·Y = iZ
            (false, true,  true,  false) => 1,  // Z·X = iY
            (false, true,  false, true)  => 0,  // Z·Z = I
            (false, true,  true,  true)  => -1, // Z·Y = −iX
            (true,  true,  true,  false) => -1, // Y·X = −iZ
            (true,  true,  false, true)  => 1,  // Y·Z = iX
            (true,  true,  true,  true)  => 0,  // Y·Y = I
        }
    }

    // ── Row multiply ──────────────────────────────────────────────────────

    /// Multiply row h (in-place) by row i on the left: row_h ← row_i · row_h.
    fn rowmul(&mut self, h: usize, i: usize) {
        let n = self.n;
        // Accumulated exponent of i (in units of i^1)
        let mut exp: i32 = 2 * (self.r[i] as i32) + 2 * (self.r[h] as i32);
        for j in 0..n {
            exp += Self::g(self.x[i][j], self.z[i][j], self.x[h][j], self.z[h][j]);
        }
        // XOR the bits (after accumulating g with OLD values)
        for j in 0..n {
            self.x[h][j] ^= self.x[i][j];
            self.z[h][j] ^= self.z[i][j];
        }
        // exp mod 4: 0 → +1 (r=false), 2 → −1 (r=true)
        self.r[h] = ((exp % 4 + 4) % 4) == 2;
    }

    // ── Gate operations ───────────────────────────────────────────────────

    /// Hadamard: H X H = Z, H Z H = X, H Y H = −Y.
    pub fn apply_h(&mut self, q: usize) {
        for h in 0..2 * self.n {
            let xq = self.x[h][q];
            let zq = self.z[h][q];
            self.r[h] ^= xq & zq;
            self.x[h][q] = zq;
            self.z[h][q] = xq;
        }
    }

    /// S gate (phase π/2): S X S† = Y, S Y S† = −X, S Z S† = Z.
    pub fn apply_s(&mut self, q: usize) {
        for h in 0..2 * self.n {
            let xq = self.x[h][q];
            let zq = self.z[h][q];
            self.r[h] ^= xq & zq;
            self.z[h][q] = zq ^ xq;
        }
    }

    /// Pauli-X: X X X = X, X Y X = −Y, X Z X = −Z.
    pub fn apply_x(&mut self, q: usize) {
        for h in 0..2 * self.n {
            self.r[h] ^= self.z[h][q];
        }
    }

    /// Pauli-Y: Y X Y = −X, Y Y Y = Y, Y Z Y = −Z.
    pub fn apply_y(&mut self, q: usize) {
        for h in 0..2 * self.n {
            self.r[h] ^= self.x[h][q] ^ self.z[h][q];
        }
    }

    /// Pauli-Z: Z X Z = −X, Z Y Z = −Y, Z Z Z = Z.
    pub fn apply_z(&mut self, q: usize) {
        for h in 0..2 * self.n {
            self.r[h] ^= self.x[h][q];
        }
    }

    /// CNOT: control=c, target=t.
    ///
    /// CNOT conjugation rules:
    ///   X_c → X_c X_t,  Z_t → Z_c Z_t,  X_t → X_t,  Z_c → Z_c
    pub fn apply_cnot(&mut self, c: usize, t: usize) {
        for h in 0..2 * self.n {
            let xc = self.x[h][c];
            let zt = self.z[h][t];
            let xt = self.x[h][t];
            let zc = self.z[h][c];
            // Phase change when x[c]=1, z[t]=1, and x[t] == z[c] (both same parity)
            self.r[h] ^= xc & zt & (xt ^ zc ^ true);
            self.x[h][t] ^= xc;
            self.z[h][c] ^= zt;
        }
    }

    /// CZ gate: CZ = (I⊗H) CNOT (I⊗H).
    pub fn apply_cz(&mut self, c: usize, t: usize) {
        self.apply_h(t);
        self.apply_cnot(c, t);
        self.apply_h(t);
    }

    /// SWAP gate = CNOT(a,b) · CNOT(b,a) · CNOT(a,b).
    pub fn apply_swap(&mut self, a: usize, b: usize) {
        self.apply_cnot(a, b);
        self.apply_cnot(b, a);
        self.apply_cnot(a, b);
    }

    // ── Measurement ───────────────────────────────────────────────────────

    /// Measure qubit `q` in the Z basis.  `rng` ∈ [0, 1).
    ///
    /// Returns `(outcome, was_random)`.
    /// `outcome` = true means |1⟩, false means |0⟩.
    pub fn measure(&mut self, q: usize, rng: f64) -> (bool, bool) {
        let n = self.n;

        // Is the measurement random? Find first stabilizer row (h ∈ [n, 2n)) with x[h][q] = 1.
        let p = (n..2 * n).find(|&h| self.x[h][q]);

        if let Some(p) = p {
            // ── Random outcome ────────────────────────────────────────────
            let outcome = rng >= 0.5;

            // For every row (destabilizer or stabilizer) other than p that has X on q,
            // multiply it by row p (clears the X on q for that row).
            for h in 0..2 * n {
                if h != p && self.x[h][q] {
                    self.rowmul(h, p);
                }
            }

            // Promote: destabilizer (p-n) ← old stabilizer p
            let old_x = self.x[p].clone();
            let old_z = self.z[p].clone();
            let old_r = self.r[p];
            self.x[p - n] = old_x;
            self.z[p - n] = old_z;
            self.r[p - n] = old_r;

            // New stabilizer p = ±Z_q (eigenvalue determined by outcome)
            self.x[p] = vec![false; n];
            self.z[p] = vec![false; n];
            self.z[p][q] = true;
            self.r[p] = outcome;   // false → +Z_q (outcome 0), true → −Z_q (outcome 1)

            (outcome, true)
        } else {
            // ── Deterministic outcome ─────────────────────────────────────
            // Find the stabilizer expression that equals ±Z_q.
            // For each destabilizer i (row i in [0,n)) that has X on qubit q,
            // multiply the scratch row by the corresponding stabilizer (row n+i).
            let mut sx = vec![false; n];
            let mut sz = vec![false; n];
            let mut sr = false;

            for i in 0..n {
                if self.x[i][q] {
                    // rowmul scratch ← row_(n+i) · scratch
                    let mut exp: i32 = 2 * (self.r[n + i] as i32) + 2 * (sr as i32);
                    for j in 0..n {
                        exp += Self::g(self.x[n + i][j], self.z[n + i][j], sx[j], sz[j]);
                    }
                    for j in 0..n {
                        sx[j] ^= self.x[n + i][j];
                        sz[j] ^= self.z[n + i][j];
                    }
                    sr = ((exp % 4 + 4) % 4) == 2;
                }
            }

            // sr encodes the sign of the stabilizer Z_q: false → +1 (outcome 0), true → −1 (outcome 1)
            (sr, false)
        }
    }
}

// ── Clifford Executor ──────────────────────────────────────────────────────

/// Execute an AQL Program using the Clifford (stabilizer) backend.
///
/// Memory: O(n²) — supports thousands of qubits for Clifford circuits.
/// Non-Clifford gates (T, Rx, Ry, Rz, Phase, Toffoli) return a `Runtime` error.
pub fn execute_clifford(program: &Program) -> Result<ExecutionResult, AqlError> {
    let n = program.num_qubits;
    let mut state = CliffordState::new(n);
    let mut measurements: Vec<MeasurementRecord> = Vec::new();
    let mut classical: HashMap<usize, bool> = HashMap::new();
    let mut gate_count = 0usize;
    let mut branch_count = 0usize;
    let mut steps = 0usize;

    // Build label → PC table
    let mut labels: HashMap<String, usize> = HashMap::new();
    for (pc, instr) in program.instructions.iter().enumerate() {
        if let Instruction::Label(name) = instr {
            labels.insert(name.clone(), pc);
        }
    }

    let instrs = &program.instructions;
    let mut pc = 0usize;
    const MAX_STEPS: usize = 1_000_000;

    while pc < instrs.len() {
        if steps > MAX_STEPS {
            return Err(AqlError::Runtime {
                msg: format!("exceeded MAX_STEPS ({MAX_STEPS}) — possible infinite loop"),
            });
        }
        steps += 1;

        match &instrs[pc] {
            Instruction::H(q)    => { state.apply_h(*q);  gate_count += 1; }
            Instruction::X(q)    => { state.apply_x(*q);  gate_count += 1; }
            Instruction::Y(q)    => { state.apply_y(*q);  gate_count += 1; }
            Instruction::Z(q)    => { state.apply_z(*q);  gate_count += 1; }
            Instruction::S(q)    => { state.apply_s(*q);  gate_count += 1; }
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
            Instruction::T(_)
            | Instruction::Rx { .. }
            | Instruction::Ry { .. }
            | Instruction::Rz { .. }
            | Instruction::Phase { .. } => {
                return Err(AqlError::Runtime {
                    msg: "Clifford backend does not support non-Clifford gates \
                          (T, Rx, Ry, Rz, Phase). Use --backend mps or statevector."
                        .to_string(),
                });
            }
            Instruction::Toffoli { .. } => {
                return Err(AqlError::Runtime {
                    msg: "Clifford backend: Toffoli is not a Clifford gate.".to_string(),
                });
            }
            Instruction::Measure(q) => {
                let (outcome, _) = state.measure(*q, rand::random::<f64>());
                classical.insert(*q, outcome);
                measurements.push(MeasurementRecord { qubit: *q, outcome, step: pc });
            }
            Instruction::MeasureAll => {
                for q in 0..n {
                    let (outcome, _) = state.measure(q, rand::random::<f64>());
                    classical.insert(q, outcome);
                    measurements.push(MeasurementRecord { qubit: q, outcome, step: pc });
                }
            }
            Instruction::Label(_) | Instruction::Barrier => {}
            Instruction::Goto { label } => {
                if let Some(&tgt) = labels.get(label) { pc = tgt; continue; }
                return Err(AqlError::Runtime { msg: format!("undefined label '{label}'") });
            }
            Instruction::GotoIf { qubit, label } => {
                let val = classical.get(qubit).copied().ok_or_else(|| AqlError::Runtime {
                    msg: format!("IF: qubit {qubit} has not been measured"),
                })?;
                if val {
                    branch_count += 1;
                    if let Some(&tgt) = labels.get(label) { pc = tgt; continue; }
                    return Err(AqlError::Runtime { msg: format!("undefined label '{label}'") });
                }
            }
            Instruction::GotoIfNot { qubit, label } => {
                let val = classical.get(qubit).copied().ok_or_else(|| AqlError::Runtime {
                    msg: format!("IFNOT: qubit {qubit} has not been measured"),
                })?;
                if !val {
                    branch_count += 1;
                    if let Some(&tgt) = labels.get(label) { pc = tgt; continue; }
                    return Err(AqlError::Runtime { msg: format!("undefined label '{label}'") });
                }
            }
            Instruction::CallGate { name, qubits } => {
                if let Some(gate_def) = program.gate_defs.get(name) {
                    apply_gate_def_clifford(&mut state, gate_def, qubits, &program.gate_defs)?;
                    gate_count += 1;
                } else {
                    return Err(AqlError::Runtime {
                        msg: format!("undefined gate '{name}'"),
                    });
                }
            }
            Instruction::MeasureInto { .. } | Instruction::GotoIfCreg { .. }
            | Instruction::GotoIfNotCreg { .. } => {
                return Err(AqlError::Runtime {
                    msg: "CREG instructions not supported in Clifford backend; use --backend statevector".to_string(),
                });
            }
        }
        pc += 1;
    }

    Ok(ExecutionResult {
        num_qubits: n,
        measurements,
        pre_measurement_probs: None,
        pre_measurement_amplitudes: None,
        final_probabilities: vec![], // Clifford backend: no full amplitude vector
        final_amplitudes:    vec![],
        gate_count,
        branch_count,
        steps_executed: steps,
    })
}

fn apply_gate_def_clifford(
    state:     &mut CliffordState,
    gate_def:  &crate::compiler::ir::GateDef,
    qubits:    &[usize],
    gate_defs: &HashMap<String, crate::compiler::ir::GateDef>,
) -> Result<(), AqlError> {
    let remap = |local: usize| qubits[local];
    for instr in &gate_def.body {
        match instr {
            Instruction::H(q)    => state.apply_h(remap(*q)),
            Instruction::X(q)    => state.apply_x(remap(*q)),
            Instruction::Y(q)    => state.apply_y(remap(*q)),
            Instruction::Z(q)    => state.apply_z(remap(*q)),
            Instruction::S(q)    => state.apply_s(remap(*q)),
            Instruction::Cnot { control, target } =>
                state.apply_cnot(remap(*control), remap(*target)),
            Instruction::Cz { control, target } =>
                state.apply_cz(remap(*control), remap(*target)),
            Instruction::Swap { qubit_a, qubit_b } =>
                state.apply_swap(remap(*qubit_a), remap(*qubit_b)),
            Instruction::T(_)
            | Instruction::Rx { .. }
            | Instruction::Ry { .. }
            | Instruction::Rz { .. }
            | Instruction::Phase { .. }
            | Instruction::Toffoli { .. } => {
                return Err(AqlError::Runtime {
                    msg: "Clifford backend does not support non-Clifford gates.".to_string(),
                });
            }
            Instruction::CallGate { name, qubits: sub_q } => {
                let sub_qubits: Vec<usize> = sub_q.iter().map(|&l| remap(l)).collect();
                if let Some(gd) = gate_defs.get(name) {
                    apply_gate_def_clifford(state, gd, &sub_qubits, gate_defs)?;
                } else {
                    return Err(AqlError::Runtime {
                        msg: format!("undefined gate '{name}'"),
                    });
                }
            }
            _ => {} // Barrier, Label — no-op
        }
    }
    Ok(())
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Gate verification ──────────────────────────────────────────────────

    #[test]
    fn test_initial_state_all_zero() {
        // All qubits start in |0⟩: deterministic measurement → 0
        let mut state = CliffordState::new(3);
        for q in 0..3 {
            let (outcome, was_random) = state.measure(q, 0.3);
            assert!(!outcome,     "q{q}: expected |0⟩");
            assert!(!was_random,  "q{q}: initial |0⟩ must be deterministic");
        }
    }

    #[test]
    fn test_x_gate_flips_to_one() {
        let mut state = CliffordState::new(1);
        state.apply_x(0);
        let (outcome, was_random) = state.measure(0, 0.3);
        assert!(outcome,     "X|0⟩ should give |1⟩");
        assert!(!was_random, "X|0⟩ measurement is deterministic");
    }

    #[test]
    fn test_z_gate_keeps_zero() {
        // Z|0⟩ = |0⟩ (Z only changes the phase of |1⟩)
        let mut state = CliffordState::new(1);
        state.apply_z(0);
        let (outcome, was_random) = state.measure(0, 0.3);
        assert!(!outcome,    "Z|0⟩ should still be |0⟩");
        assert!(!was_random, "Z|0⟩ measurement is deterministic");
    }

    #[test]
    fn test_h_gate_gives_superposition() {
        // H|0⟩ = |+⟩; measurement is random
        let mut state = CliffordState::new(1);
        state.apply_h(0);
        let (_, was_random) = state.measure(0, 0.3);
        assert!(was_random, "H|0⟩ must produce a random measurement");
    }

    #[test]
    fn test_hxh_equals_z() {
        // H X H = Z, so H X H |+⟩ = Z |+⟩ = |−⟩ (still random but phase-flipped)
        // Simpler check: H X H |0⟩ should give deterministic |0⟩ (since HXH = Z, Z|0⟩=|0⟩)
        let mut state = CliffordState::new(1);
        state.apply_h(0);
        state.apply_x(0);
        state.apply_h(0);
        // HXH = Z, so state is Z|0⟩=|0⟩
        let (outcome, was_random) = state.measure(0, 0.3);
        assert!(!outcome,    "HXH|0⟩ = Z|0⟩ = |0⟩");
        assert!(!was_random, "deterministic");
    }

    #[test]
    fn test_double_x_is_identity() {
        let mut state = CliffordState::new(2);
        state.apply_x(0);
        state.apply_x(0);
        let (o0, r0) = state.measure(0, 0.3);
        let (o1, r1) = state.measure(1, 0.3);
        assert!(!o0 && !r0, "X²|0⟩ = |0⟩");
        assert!(!o1 && !r1, "q1 untouched");
    }

    #[test]
    fn test_s_gate_phase() {
        // S|0⟩ = |0⟩ (S only affects the |1⟩ component)
        let mut state = CliffordState::new(1);
        state.apply_s(0);
        let (outcome, was_random) = state.measure(0, 0.3);
        assert!(!outcome,    "S|0⟩ = |0⟩");
        assert!(!was_random, "deterministic");
    }

    #[test]
    fn test_y_gate_flips_with_phase() {
        // Y = iXZ.  Y|0⟩ = i|1⟩; measurement gives |1⟩.
        let mut state = CliffordState::new(1);
        state.apply_y(0);
        let (outcome, was_random) = state.measure(0, 0.3);
        assert!(outcome,     "Y|0⟩ should give |1⟩");
        assert!(!was_random, "deterministic");
    }

    // ── Entanglement ───────────────────────────────────────────────────────

    #[test]
    fn test_bell_state_correlation() {
        // |Φ+⟩ = (|00⟩ + |11⟩)/√2: both qubits always agree
        for trial in 0..20 {
            let mut state = CliffordState::new(2);
            state.apply_h(0);
            state.apply_cnot(0, 1);
            let rng = (trial as f64) * 0.05; // vary rng across [0, 1)
            let (o0, _) = state.measure(0, rng);
            let (o1, _) = state.measure(1, 0.3);
            assert_eq!(o0, o1, "Bell state qubits must agree (trial {trial})");
        }
    }

    #[test]
    fn test_ghz_state_3qubits() {
        // GHZ: H(0) CNOT(0,1) CNOT(0,2) → (|000⟩+|111⟩)/√2
        let mut state = CliffordState::new(3);
        state.apply_h(0);
        state.apply_cnot(0, 1);
        state.apply_cnot(0, 2);
        // All qubits must agree
        let (o0, r0) = state.measure(0, 0.3);
        let (o1, _)  = state.measure(1, 0.3);
        let (o2, _)  = state.measure(2, 0.3);
        assert!(r0, "q0 measurement of GHZ should be random");
        assert_eq!(o0, o1, "GHZ: q0 == q1");
        assert_eq!(o0, o2, "GHZ: q0 == q2");
    }

    #[test]
    fn test_cz_gate() {
        // CZ|11⟩ = -|11⟩.  The phase is unobservable by measurement alone.
        // Verify via: H(0) CZ(0,1) H(0) on |00⟩ = CNOT-like but with Z on control.
        // H(0) → |+0⟩; CZ(0,1) leaves |+0⟩ = |+0⟩ (|0⟩ part of |+⟩ unaffected,
        //   |1⟩ part: CZ|10⟩ = |10⟩ → only |11⟩ gets phase, not |10⟩).
        // Just check: CZ on |00⟩ gives |00⟩ (no phase on |00⟩).
        let mut state = CliffordState::new(2);
        state.apply_cz(0, 1);
        let (o0, r0) = state.measure(0, 0.3);
        let (o1, r1) = state.measure(1, 0.3);
        assert!(!o0 && !r0 && !o1 && !r1, "CZ|00⟩ = |00⟩");
    }

    #[test]
    fn test_swap_gate() {
        // X|0⟩ ⊗ |0⟩ = |10⟩; SWAP → |01⟩
        let mut state = CliffordState::new(2);
        state.apply_x(0);             // |10⟩
        state.apply_swap(0, 1);       // |01⟩
        let (o0, _) = state.measure(0, 0.3);
        let (o1, _) = state.measure(1, 0.3);
        assert!(!o0, "after SWAP: q0 should be |0⟩");
        assert!(o1,  "after SWAP: q1 should be |1⟩");
    }

    // ── Non-Clifford error ─────────────────────────────────────────────────

    #[test]
    fn test_t_gate_returns_error() {
        let src = "QREG 1\nT 0\nMEASURE 0";
        let prog = crate::compiler::parse_source(src).unwrap();
        let result = execute_clifford(&prog);
        assert!(result.is_err(), "T gate must fail on Clifford backend");
        assert!(result.unwrap_err().to_string().contains("Clifford"));
    }

    // ── AQL integration ────────────────────────────────────────────────────

    #[test]
    fn test_aql_bell_clifford() {
        let src = "QREG 2\nH 0\nCNOT 0 1\nMEASURE_ALL";
        let prog = crate::compiler::parse_source(src).unwrap();
        let result = execute_clifford(&prog).unwrap();
        assert_eq!(result.measurements.len(), 2);
        let o0 = result.measurements[0].outcome;
        let o1 = result.measurements[1].outcome;
        assert_eq!(o0, o1, "AQL Bell state: both qubits must agree");
    }

    #[test]
    fn test_aql_all_zeros() {
        let src = "QREG 4\nMEASURE_ALL";
        let prog = crate::compiler::parse_source(src).unwrap();
        let result = execute_clifford(&prog).unwrap();
        assert!(result.measurements.iter().all(|m| !m.outcome), "all qubits start at |0⟩");
    }

    // ── Large circuit ──────────────────────────────────────────────────────

    #[test]
    fn test_large_clifford_100_qubits() {
        // 100-qubit GHZ circuit: should run in O(n²) time
        let n = 100usize;
        let mut src = format!("QREG {n}\nH 0\n");
        for i in 1..n { src.push_str(&format!("CNOT 0 {i}\n")); }
        src.push_str("MEASURE_ALL");
        let prog = crate::compiler::parse_source(&src).unwrap();
        let result = execute_clifford(&prog).unwrap();
        // All qubits must agree
        let o0 = result.measurements[0].outcome;
        assert!(
            result.measurements.iter().all(|m| m.outcome == o0),
            "all qubits of 100-qubit GHZ must agree"
        );
    }
}
