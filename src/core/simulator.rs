/// High-level quantum simulator interface.
///
/// `Simulator` wraps the state vector and gate operations behind
/// a clean, ergonomic API. It handles:
///   - Gate application (H, X, Y, Z, S, T, Rx, Ry, Rz, CNOT, CZ, SWAP, Toffoli)
///   - Measurement with wavefunction collapse
///   - Deterministic mode (fixed seed) for reproducible research
///   - State inspection and pretty-printing
///   - Optional noise model applied after each single-qubit gate (Phase 5)
use super::gates::{
    self, apply_cnot, apply_cz, apply_single_qubit_gate, apply_swap, apply_toffoli,
};
use super::noise::NoiseChannel;
use super::state::StateVector;
use rand::Rng;
use std::fmt;

pub struct Simulator {
    pub state: StateVector,
    /// Measurement results per qubit (None = not yet measured)
    pub measurements: Vec<Option<bool>>,
    /// Optional fixed seed for deterministic measurement
    rng_seed: Option<u64>,
    /// Internal RNG state for deterministic mode
    det_counter: u64,
    /// Optional noise channel applied after every gate (Phase 5)
    noise_model: Option<NoiseChannel>,
}

impl Simulator {
    /// Create a new simulator for `num_qubits` qubits, initialized to |0...0⟩.
    pub fn new(num_qubits: usize) -> Self {
        Self {
            state: StateVector::new(num_qubits),
            measurements: vec![None; num_qubits],
            rng_seed: None,
            det_counter: 0,
            noise_model: None,
        }
    }

    /// Create a simulator with a fixed RNG seed for deterministic measurement outcomes.
    /// Useful for reproducible research and unit testing.
    pub fn with_seed(num_qubits: usize, seed: u64) -> Self {
        Self {
            state: StateVector::new(num_qubits),
            measurements: vec![None; num_qubits],
            rng_seed: Some(seed),
            det_counter: seed,
            noise_model: None,
        }
    }

    /// Create a noisy simulator. The `channel` is applied after every gate
    /// on the affected qubit(s).
    pub fn with_noise(num_qubits: usize, channel: NoiseChannel) -> Self {
        let mut sim = Self::new(num_qubits);
        sim.noise_model = Some(channel);
        sim
    }

    /// Attach or replace the noise model on an existing simulator.
    pub fn set_noise(&mut self, channel: NoiseChannel) {
        self.noise_model = Some(channel);
    }

    /// Remove the noise model (revert to ideal simulation).
    pub fn clear_noise(&mut self) {
        self.noise_model = None;
    }

    /// Number of qubits in this simulator.
    pub fn num_qubits(&self) -> usize {
        self.state.num_qubits
    }

    // ── Single-Qubit Gates ────────────────────────────────────────────────

    pub fn h(&mut self, qubit: usize) -> &mut Self {
        apply_single_qubit_gate(&mut self.state, &gates::hadamard(), qubit);
        self.apply_noise_if_set(qubit);
        self
    }

    pub fn x(&mut self, qubit: usize) -> &mut Self {
        apply_single_qubit_gate(&mut self.state, &gates::pauli_x(), qubit);
        self.apply_noise_if_set(qubit);
        self
    }

    pub fn y(&mut self, qubit: usize) -> &mut Self {
        apply_single_qubit_gate(&mut self.state, &gates::pauli_y(), qubit);
        self.apply_noise_if_set(qubit);
        self
    }

    pub fn z(&mut self, qubit: usize) -> &mut Self {
        apply_single_qubit_gate(&mut self.state, &gates::pauli_z(), qubit);
        self.apply_noise_if_set(qubit);
        self
    }

    pub fn s(&mut self, qubit: usize) -> &mut Self {
        apply_single_qubit_gate(&mut self.state, &gates::s_gate(), qubit);
        self.apply_noise_if_set(qubit);
        self
    }

    pub fn t(&mut self, qubit: usize) -> &mut Self {
        apply_single_qubit_gate(&mut self.state, &gates::t_gate(), qubit);
        self.apply_noise_if_set(qubit);
        self
    }

    pub fn rx(&mut self, qubit: usize, theta: f64) -> &mut Self {
        apply_single_qubit_gate(&mut self.state, &gates::rx(theta), qubit);
        self.apply_noise_if_set(qubit);
        self
    }

    pub fn ry(&mut self, qubit: usize, theta: f64) -> &mut Self {
        apply_single_qubit_gate(&mut self.state, &gates::ry(theta), qubit);
        self.apply_noise_if_set(qubit);
        self
    }

    pub fn rz(&mut self, qubit: usize, theta: f64) -> &mut Self {
        apply_single_qubit_gate(&mut self.state, &gates::rz(theta), qubit);
        self.apply_noise_if_set(qubit);
        self
    }

    pub fn phase(&mut self, qubit: usize, theta: f64) -> &mut Self {
        apply_single_qubit_gate(&mut self.state, &gates::phase_gate(theta), qubit);
        self.apply_noise_if_set(qubit);
        self
    }

    // ── Multi-Qubit Gates ─────────────────────────────────────────────────

    pub fn cnot(&mut self, control: usize, target: usize) -> &mut Self {
        apply_cnot(&mut self.state, control, target);
        self.apply_noise_if_set(control);
        self.apply_noise_if_set(target);
        self
    }

    pub fn cz(&mut self, control: usize, target: usize) -> &mut Self {
        apply_cz(&mut self.state, control, target);
        self.apply_noise_if_set(control);
        self.apply_noise_if_set(target);
        self
    }

    pub fn swap(&mut self, qubit_a: usize, qubit_b: usize) -> &mut Self {
        apply_swap(&mut self.state, qubit_a, qubit_b);
        self.apply_noise_if_set(qubit_a);
        self.apply_noise_if_set(qubit_b);
        self
    }

    pub fn toffoli(&mut self, control0: usize, control1: usize, target: usize) -> &mut Self {
        apply_toffoli(&mut self.state, control0, control1, target);
        self.apply_noise_if_set(control0);
        self.apply_noise_if_set(control1);
        self.apply_noise_if_set(target);
        self
    }

    // ── Custom Gate ───────────────────────────────────────────────────────

    pub fn apply(&mut self, gate: &gates::Matrix2x2, qubit: usize) -> &mut Self {
        apply_single_qubit_gate(&mut self.state, gate, qubit);
        self.apply_noise_if_set(qubit);
        self
    }

    // ── Measurement ───────────────────────────────────────────────────────

    /// Measure a single qubit. Collapses the state vector.
    /// Returns `true` for |1⟩, `false` for |0⟩.
    pub fn measure(&mut self, qubit: usize) -> bool {
        let r = self.sample_random();
        let result = self.state.collapse(qubit, r);
        self.measurements[qubit] = Some(result);
        result
    }

    /// Measure all qubits in order. Returns results as a Vec<bool>.
    pub fn measure_all(&mut self) -> Vec<bool> {
        (0..self.num_qubits()).map(|q| self.measure(q)).collect()
    }

    /// Measure all qubits and return as a bit-string (e.g. "01101").
    pub fn measure_all_string(&mut self) -> String {
        self.measure_all()
            .iter()
            .map(|&b| if b { '1' } else { '0' })
            .collect()
    }

    /// Return probabilities of each basis state without collapsing.
    /// Purely classical readout of the current amplitudes.
    pub fn probabilities(&self) -> Vec<f64> {
        self.state.amplitudes.iter().map(|a| a.norm_sq()).collect()
    }

    /// Return the probability of a specific qubit measuring as |1⟩.
    pub fn qubit_probability_one(&self, qubit: usize) -> f64 {
        self.state.marginal_probability_one(qubit)
    }

    // ── State Control ─────────────────────────────────────────────────────

    /// Reset state to |0...0⟩ and clear all measurements.
    pub fn reset(&mut self) {
        self.state.reset();
        for m in self.measurements.iter_mut() {
            *m = None;
        }
        self.det_counter = self.rng_seed.unwrap_or(0);
    }

    /// Print the full state vector to stdout.
    pub fn print_state(&self) {
        print!("{}", self.state);
    }

    // ── Internal noise application ────────────────────────────────────────

    /// Apply the noise model to `qubit` (if one is configured).
    fn apply_noise_if_set(&mut self, qubit: usize) {
        if self.noise_model.is_none() { return; }
        let rng = self.sample_random();
        // Clone to satisfy borrow checker (noise_model: &self, state: &mut self)
        let channel = self.noise_model.clone().unwrap();
        channel.apply(&mut self.state, qubit, rng);
    }

    // ── Internal RNG ──────────────────────────────────────────────────────

    fn sample_random(&mut self) -> f64 {
        match self.rng_seed {
            Some(_) => {
                // Simple deterministic LCG for reproducibility
                self.det_counter = self
                    .det_counter
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                // Map to [0, 1)
                (self.det_counter >> 33) as f64 / (1u64 << 31) as f64
            }
            None => rand::thread_rng().gen::<f64>(),
        }
    }
}

impl fmt::Display for Simulator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.state)?;
        write!(f, "Measurements: [")?;
        for (i, m) in self.measurements.iter().enumerate() {
            match m {
                Some(true)  => write!(f, "q{}=1", i)?,
                Some(false) => write!(f, "q{}=0", i)?,
                None        => write!(f, "q{}=?", i)?,
            }
            if i + 1 < self.measurements.len() {
                write!(f, ", ")?;
            }
        }
        writeln!(f, "]")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_basic_x_measurement() {
        let mut sim = Simulator::with_seed(1, 42);
        sim.x(0);
        let result = sim.measure(0);
        // X flips |0⟩ to |1⟩, so measurement must be 1
        assert!(result);
    }

    #[test]
    fn test_x_twice_back_to_zero() {
        let mut sim = Simulator::with_seed(1, 42);
        sim.x(0).x(0);
        let result = sim.measure(0);
        assert!(!result); // back to |0⟩
    }

    #[test]
    fn test_hadamard_probabilities() {
        let sim_ref = {
            let mut s = Simulator::new(1);
            s.h(0);
            s
        };
        let p1 = sim_ref.qubit_probability_one(0);
        assert!((p1 - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_bell_state_probabilities() {
        let mut sim = Simulator::new(2);
        sim.h(0).cnot(0, 1);
        let probs = sim.probabilities();
        // |Φ+⟩: equal probability for |00⟩ and |11⟩ only
        assert!((probs[0] - 0.5).abs() < 1e-10); // |00⟩
        assert!(probs[1].abs() < 1e-10);           // |01⟩
        assert!(probs[2].abs() < 1e-10);           // |10⟩
        assert!((probs[3] - 0.5).abs() < 1e-10); // |11⟩
    }

    #[test]
    fn test_ghz_state() {
        // GHZ = (|000⟩ + |111⟩) / √2
        // Circuit: H(0), CNOT(0,1), CNOT(0,2)
        let mut sim = Simulator::new(3);
        sim.h(0).cnot(0, 1).cnot(0, 2);
        let probs = sim.probabilities();
        assert!((probs[0] - 0.5).abs() < 1e-10); // |000⟩
        assert!((probs[7] - 0.5).abs() < 1e-10); // |111⟩
        // All others should be ~0
        for i in 1..7 {
            assert!(probs[i].abs() < 1e-10, "unexpected prob at index {}", i);
        }
    }

    #[test]
    fn test_reset() {
        let mut sim = Simulator::with_seed(1, 0);
        sim.x(0);
        sim.measure(0);
        sim.reset();
        assert!(sim.measurements.iter().all(|m| m.is_none()));
        assert!((sim.state.amplitudes[0].norm_sq() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_chain_api() {
        // Builder-style chaining
        let mut sim = Simulator::new(2);
        sim.h(0).h(1).cnot(0, 1).z(0);
        // Just check normalization after chained ops
        let total: f64 = sim.probabilities().iter().sum();
        assert!((total - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_rz_pi_equals_z_up_to_phase() {
        let mut sim = Simulator::new(1);
        sim.h(0).rz(0, PI).h(0);
        // H·Rz(π)·H = X (up to global phase) — qubit should flip
        let p1 = sim.qubit_probability_one(0);
        assert!((p1 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_measure_all_string() {
        let mut sim = Simulator::with_seed(2, 0);
        sim.x(0).x(1); // set both qubits to |1⟩
        let s = sim.measure_all_string();
        assert_eq!(s, "11");
    }
}
