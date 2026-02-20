/// Quantum gate definitions and application logic.
///
/// Gates are represented as unitary matrices.
/// Single-qubit gates are 2×2 complex matrices.
/// Multi-qubit gates operate on tensor product spaces.
///
/// Application strategy: iterate over all 2^n basis states,
/// pair up states that differ only in the target qubit, then
/// apply the 2×2 gate matrix to each pair — O(2^n) per gate.
use super::complex::Complex;
use super::state::StateVector;
use std::f64::consts::{FRAC_1_SQRT_2, PI};

/// A 2×2 complex unitary matrix representing a single-qubit gate.
/// Row-major: matrix[row][col]
pub type Matrix2x2 = [[Complex; 2]; 2];

// ── Standard Gate Matrices ─────────────────────────────────────────────────

/// Hadamard gate — creates superposition from a basis state.
/// H = (1/√2) * [[1, 1], [1, -1]]
pub fn hadamard() -> Matrix2x2 {
    let h = Complex::new(FRAC_1_SQRT_2, 0.0);
    let neg_h = Complex::new(-FRAC_1_SQRT_2, 0.0);
    [
        [h, h],
        [h, neg_h],
    ]
}

/// Pauli-X gate — quantum NOT, flips |0⟩ ↔ |1⟩.
/// X = [[0, 1], [1, 0]]
pub fn pauli_x() -> Matrix2x2 {
    [
        [Complex::zero(), Complex::one()],
        [Complex::one(),  Complex::zero()],
    ]
}

/// Pauli-Y gate — bit + phase flip.
/// Y = [[0, -i], [i, 0]]
pub fn pauli_y() -> Matrix2x2 {
    [
        [Complex::zero(), -Complex::i()],
        [Complex::i(),    Complex::zero()],
    ]
}

/// Pauli-Z gate — phase flip, |1⟩ → -|1⟩.
/// Z = [[1, 0], [0, -1]]
pub fn pauli_z() -> Matrix2x2 {
    [
        [Complex::one(),  Complex::zero()],
        [Complex::zero(), Complex::new(-1.0, 0.0)],
    ]
}

/// S gate — π/2 phase gate.
/// S = [[1, 0], [0, i]]
pub fn s_gate() -> Matrix2x2 {
    [
        [Complex::one(),  Complex::zero()],
        [Complex::zero(), Complex::i()],
    ]
}

/// T gate — π/4 phase gate.
/// T = [[1, 0], [0, e^(iπ/4)]]
pub fn t_gate() -> Matrix2x2 {
    let phase = Complex::from_polar(1.0, PI / 4.0);
    [
        [Complex::one(),  Complex::zero()],
        [Complex::zero(), phase],
    ]
}

/// Rotation around X axis by angle θ.
/// Rx(θ) = [[cos(θ/2), -i·sin(θ/2)], [-i·sin(θ/2), cos(θ/2)]]
pub fn rx(theta: f64) -> Matrix2x2 {
    let cos = Complex::new((theta / 2.0).cos(), 0.0);
    let i_sin = Complex::new(0.0, -(theta / 2.0).sin());
    [
        [cos,   i_sin],
        [i_sin, cos],
    ]
}

/// Rotation around Y axis by angle θ.
/// Ry(θ) = [[cos(θ/2), -sin(θ/2)], [sin(θ/2), cos(θ/2)]]
pub fn ry(theta: f64) -> Matrix2x2 {
    let cos = Complex::new((theta / 2.0).cos(), 0.0);
    let sin = Complex::new((theta / 2.0).sin(), 0.0);
    let neg_sin = Complex::new(-(theta / 2.0).sin(), 0.0);
    [
        [cos,     neg_sin],
        [sin,     cos],
    ]
}

/// Rotation around Z axis by angle θ.
/// Rz(θ) = [[e^(-iθ/2), 0], [0, e^(iθ/2)]]
pub fn rz(theta: f64) -> Matrix2x2 {
    [
        [Complex::from_polar(1.0, -theta / 2.0), Complex::zero()],
        [Complex::zero(), Complex::from_polar(1.0, theta / 2.0)],
    ]
}

/// Phase gate — applies a phase e^(iθ) to |1⟩.
/// P(θ) = [[1, 0], [0, e^(iθ)]]
pub fn phase_gate(theta: f64) -> Matrix2x2 {
    [
        [Complex::one(),  Complex::zero()],
        [Complex::zero(), Complex::from_polar(1.0, theta)],
    ]
}

/// Identity gate — no-op, useful for circuit padding.
pub fn identity() -> Matrix2x2 {
    [
        [Complex::one(),  Complex::zero()],
        [Complex::zero(), Complex::one()],
    ]
}

// ── Gate Application ───────────────────────────────────────────────────────

/// Apply a single-qubit gate to `target` qubit in the state vector.
///
/// Dispatches to the fastest available backend at runtime:
/// - **AVX2** (256-bit): used when `target == 0` and the CPU supports AVX2.
///   Processes adjacent amplitude pairs in one YMM-register-wide pass.
/// - **Scalar**: used for all other targets or when AVX2 is unavailable.
///
/// Both paths produce bit-identical results (within floating-point rounding).
pub fn apply_single_qubit_gate(state: &mut StateVector, gate: &Matrix2x2, target: usize) {
    assert!(target < state.num_qubits, "target qubit out of range");
    super::simd::apply_gate_simd(state, gate, target);
}

/// Apply CNOT (Controlled-NOT) gate.
///
/// Flips `target` qubit when `control` qubit is |1⟩.
/// Implements quantum entanglement when combined with Hadamard.
pub fn apply_cnot(state: &mut StateVector, control: usize, target: usize) {
    assert!(control < state.num_qubits, "control qubit out of range");
    assert!(target < state.num_qubits, "target qubit out of range");
    assert_ne!(control, target, "control and target must be different qubits");

    let dim = state.dim();
    let control_mask = 1 << control;
    let target_mask = 1 << target;

    for i in 0..dim {
        // Only act on basis states where control = 1 and target = 0
        if (i & control_mask != 0) && (i & target_mask == 0) {
            let j = i | target_mask; // flip the target bit
            state.amplitudes.swap(i, j);
        }
    }
}

/// Apply CZ (Controlled-Z) gate.
///
/// Applies Z to `target` when `control` is |1⟩.
/// Equivalent to a phase flip on |11⟩.
pub fn apply_cz(state: &mut StateVector, control: usize, target: usize) {
    assert!(control < state.num_qubits, "control qubit out of range");
    assert!(target < state.num_qubits, "target qubit out of range");
    assert_ne!(control, target, "control and target must be different qubits");

    let control_mask = 1 << control;
    let target_mask = 1 << target;

    for i in 0..state.dim() {
        if (i & control_mask != 0) && (i & target_mask != 0) {
            state.amplitudes[i] = -state.amplitudes[i];
        }
    }
}

/// Apply SWAP gate — exchanges the states of two qubits.
pub fn apply_swap(state: &mut StateVector, qubit_a: usize, qubit_b: usize) {
    assert!(qubit_a < state.num_qubits, "qubit_a out of range");
    assert!(qubit_b < state.num_qubits, "qubit_b out of range");
    assert_ne!(qubit_a, qubit_b, "SWAP requires two different qubits");

    let dim = state.dim();
    let mask_a = 1 << qubit_a;
    let mask_b = 1 << qubit_b;

    for i in 0..dim {
        // Only process pairs where the two bits differ (avoid double-swap)
        let bit_a = (i >> qubit_a) & 1;
        let bit_b = (i >> qubit_b) & 1;
        if bit_a == 1 && bit_b == 0 {
            let j = (i & !(mask_a | mask_b)) | (bit_b << qubit_a) | (bit_a << qubit_b);
            state.amplitudes.swap(i, j);
        }
    }
}

/// Apply Toffoli gate (CCNOT) — flips `target` when both controls are |1⟩.
pub fn apply_toffoli(state: &mut StateVector, control0: usize, control1: usize, target: usize) {
    assert!(control0 < state.num_qubits);
    assert!(control1 < state.num_qubits);
    assert!(target < state.num_qubits);
    assert!(control0 != control1 && control0 != target && control1 != target);

    let dim = state.dim();
    let c0_mask = 1 << control0;
    let c1_mask = 1 << control1;
    let t_mask = 1 << target;

    for i in 0..dim {
        if (i & c0_mask != 0) && (i & c1_mask != 0) && (i & t_mask == 0) {
            let j = i | t_mask;
            state.amplitudes.swap(i, j);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn nearly_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-10
    }

    #[test]
    fn test_x_gate_flips_qubit() {
        let mut state = StateVector::new(1);
        // Start: |0⟩, apply X → |1⟩
        apply_single_qubit_gate(&mut state, &pauli_x(), 0);
        assert!(nearly_eq(state.amplitudes[0].norm_sq(), 0.0));
        assert!(nearly_eq(state.amplitudes[1].norm_sq(), 1.0));
    }

    #[test]
    fn test_x_gate_twice_is_identity() {
        let mut state = StateVector::new(1);
        let x = pauli_x();
        apply_single_qubit_gate(&mut state, &x, 0);
        apply_single_qubit_gate(&mut state, &x, 0);
        // Should be back to |0⟩
        assert!(nearly_eq(state.amplitudes[0].norm_sq(), 1.0));
        assert!(nearly_eq(state.amplitudes[1].norm_sq(), 0.0));
    }

    #[test]
    fn test_hadamard_creates_superposition() {
        let mut state = StateVector::new(1);
        apply_single_qubit_gate(&mut state, &hadamard(), 0);
        let p0 = state.amplitudes[0].norm_sq();
        let p1 = state.amplitudes[1].norm_sq();
        assert!(nearly_eq(p0, 0.5));
        assert!(nearly_eq(p1, 0.5));
    }

    #[test]
    fn test_hadamard_twice_is_identity() {
        let mut state = StateVector::new(1);
        let h = hadamard();
        apply_single_qubit_gate(&mut state, &h, 0);
        apply_single_qubit_gate(&mut state, &h, 0);
        assert!(nearly_eq(state.amplitudes[0].norm_sq(), 1.0));
        assert!(nearly_eq(state.amplitudes[1].norm_sq(), 0.0));
    }

    #[test]
    fn test_bell_state_creation() {
        // |Φ+⟩ = (|00⟩ + |11⟩) / √2
        // Circuit: H on qubit 0, then CNOT(0, 1)
        let mut state = StateVector::new(2);
        apply_single_qubit_gate(&mut state, &hadamard(), 0);
        apply_cnot(&mut state, 0, 1);

        let p00 = state.amplitudes[0].norm_sq(); // |00⟩
        let p01 = state.amplitudes[1].norm_sq(); // |01⟩
        let p10 = state.amplitudes[2].norm_sq(); // |10⟩
        let p11 = state.amplitudes[3].norm_sq(); // |11⟩

        assert!(nearly_eq(p00, 0.5));
        assert!(nearly_eq(p01, 0.0));
        assert!(nearly_eq(p10, 0.0));
        assert!(nearly_eq(p11, 0.5));
    }

    #[test]
    fn test_z_gate_phase_flip() {
        let mut state = StateVector::new(1);
        // Put into |1⟩ first
        apply_single_qubit_gate(&mut state, &pauli_x(), 0);
        // Z|1⟩ = -|1⟩
        apply_single_qubit_gate(&mut state, &pauli_z(), 0);
        assert!(nearly_eq(state.amplitudes[1].re, -1.0));
    }

    #[test]
    fn test_rx_pi_equals_x() {
        let mut state = StateVector::new(1);
        // Rx(π) ≈ -iX (up to global phase, probabilities same as X)
        apply_single_qubit_gate(&mut state, &rx(PI), 0);
        // |0⟩ → Rx(π) → -i|1⟩, probability is still 1 for |1⟩
        assert!(nearly_eq(state.amplitudes[0].norm_sq(), 0.0));
        assert!(nearly_eq(state.amplitudes[1].norm_sq(), 1.0));
    }

    #[test]
    fn test_swap_gate() {
        let mut state = StateVector::new(2);
        // Start: |00⟩, put into |01⟩ (qubit 0 = 1)
        apply_single_qubit_gate(&mut state, &pauli_x(), 0);
        // SWAP(0, 1): should give |10⟩ (qubit 1 = 1)
        apply_swap(&mut state, 0, 1);
        // |10⟩ = basis index 2
        assert!(nearly_eq(state.amplitudes[2].norm_sq(), 1.0));
        assert!(nearly_eq(state.amplitudes[1].norm_sq(), 0.0));
    }

    #[test]
    fn test_toffoli_gate() {
        let mut state = StateVector::new(3);
        // Set control0 = 1, control1 = 1, target = 0
        apply_single_qubit_gate(&mut state, &pauli_x(), 1);
        apply_single_qubit_gate(&mut state, &pauli_x(), 2);
        // Toffoli(1, 2, 0): should flip qubit 0 → |111⟩
        apply_toffoli(&mut state, 1, 2, 0);
        // |111⟩ = basis index 7
        assert!(nearly_eq(state.amplitudes[7].norm_sq(), 1.0));
    }
}
