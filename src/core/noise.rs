/// Quantum noise channels for state-vector simulation.
///
/// Noise is modeled using the quantum-trajectory (Monte Carlo wave-function)
/// method: at each noise application site a random number selects one of the
/// Kraus operators, which is then applied to the pure state and renormalized.
/// For unitary channels (Pauli errors) renormalization is unnecessary.
///
/// Available channels
/// ------------------
/// | Channel           | Model                                         |
/// |-------------------|-----------------------------------------------|
/// | BitFlip(p)        | X applied with probability p                  |
/// | PhaseFlip(p)      | Z applied with probability p                  |
/// | Depolarizing(p)   | X, Y, or Z each with probability p/3          |
/// | AmplitudeDamping(γ) | |1⟩→|0⟩ decay with probability γ·P(|1⟩)  |
///
/// Usage
/// -----
/// Pass a `NoiseChannel` to `Simulator::with_noise` or `Simulator::set_noise`.
/// The channel is applied automatically after every single-qubit gate on the
/// affected qubit(s).
use super::complex::Complex;
use super::gates::{apply_single_qubit_gate, pauli_x, pauli_y, pauli_z};
use super::state::StateVector;

// ── NoiseChannel ───────────────────────────────────────────────────────────

/// A quantum noise channel applied after each gate.
#[derive(Debug, Clone)]
pub enum NoiseChannel {
    /// **Bit-flip** — Pauli X applied with probability `prob`.
    ///
    /// `prob` must be in [0, 1].
    BitFlip { prob: f64 },

    /// **Phase-flip** — Pauli Z applied with probability `prob`.
    ///
    /// `prob` must be in [0, 1].
    PhaseFlip { prob: f64 },

    /// **Depolarizing** — random Pauli (X, Y, or Z) each with probability `prob/3`.
    ///
    /// Total error probability is `prob`. `prob` must be in [0, 1].
    Depolarizing { prob: f64 },

    /// **Amplitude damping** — models energy relaxation (|1⟩ → |0⟩).
    ///
    /// `gamma` is the decay probability per gate; must be in [0, 1].
    /// Applied via quantum-trajectory Kraus operators.
    AmplitudeDamping { gamma: f64 },
}

impl NoiseChannel {
    /// Apply this noise channel to `qubit` in `state`.
    ///
    /// `rng` must be a uniform random value in [0, 1).
    pub fn apply(&self, state: &mut StateVector, qubit: usize, rng: f64) {
        match self {
            // ── Bit-flip: X with probability p ──────────────────────────
            NoiseChannel::BitFlip { prob } => {
                if rng < *prob {
                    apply_single_qubit_gate(state, &pauli_x(), qubit);
                }
            }

            // ── Phase-flip: Z with probability p ────────────────────────
            NoiseChannel::PhaseFlip { prob } => {
                if rng < *prob {
                    apply_single_qubit_gate(state, &pauli_z(), qubit);
                }
            }

            // ── Depolarizing: X/Y/Z each with probability p/3 ───────────
            NoiseChannel::Depolarizing { prob } => {
                let p3 = prob / 3.0;
                if rng < p3 {
                    apply_single_qubit_gate(state, &pauli_x(), qubit);
                } else if rng < 2.0 * p3 {
                    apply_single_qubit_gate(state, &pauli_y(), qubit);
                } else if rng < *prob {
                    apply_single_qubit_gate(state, &pauli_z(), qubit);
                }
                // rng ≥ prob: no error
            }

            // ── Amplitude damping: |1⟩ → |0⟩ decay ─────────────────────
            //
            // Kraus operators (quantum trajectory):
            //   K0 = [[1, 0], [0, √(1−γ)]]   (no decay)
            //   K1 = [[0, √γ], [0, 0]]         (decay)
            //
            // Selection probabilities:
            //   P(K1) = γ · P(|1⟩) = γ · ⟨ψ|Π₁|ψ⟩
            //   P(K0) = 1 − P(K1)
            NoiseChannel::AmplitudeDamping { gamma } => {
                let p1 = state.marginal_probability_one(qubit);
                let p_decay = gamma * p1;
                if rng < p_decay {
                    // Apply decay operator K1 then renormalize
                    let k1 = [
                        [Complex::zero(), Complex::new(gamma.sqrt(), 0.0)],
                        [Complex::zero(), Complex::zero()],
                    ];
                    apply_single_qubit_gate(state, &k1, qubit);
                    state.normalize();
                } else {
                    // Apply no-decay operator K0 then renormalize
                    let k0 = [
                        [Complex::one(),  Complex::zero()],
                        [Complex::zero(), Complex::new((1.0 - gamma).sqrt(), 0.0)],
                    ];
                    apply_single_qubit_gate(state, &k0, qubit);
                    state.normalize();
                }
            }
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::state::StateVector;

    fn qubit_p1(state: &StateVector, q: usize) -> f64 {
        state.marginal_probability_one(q)
    }

    // ── BitFlip ───────────────────────────────────────────────────────

    #[test]
    fn test_bitflip_p0_no_error() {
        let mut state = StateVector::new(1);
        // |0⟩, rng=0.0, p=0 → no flip
        NoiseChannel::BitFlip { prob: 0.0 }.apply(&mut state, 0, 0.0);
        assert!(qubit_p1(&state, 0).abs() < 1e-10, "P(|1⟩) should remain 0");
    }

    #[test]
    fn test_bitflip_p1_always_flips() {
        let mut state = StateVector::new(1);
        // |0⟩, rng=0.0, p=1 → always flip to |1⟩
        NoiseChannel::BitFlip { prob: 1.0 }.apply(&mut state, 0, 0.0);
        assert!((qubit_p1(&state, 0) - 1.0).abs() < 1e-10, "P(|1⟩) should be 1");
    }

    #[test]
    fn test_bitflip_p1_from_one_to_zero() {
        let mut state = StateVector::new(1);
        // Start in |1⟩: apply X manually (flip state)
        NoiseChannel::BitFlip { prob: 1.0 }.apply(&mut state, 0, 0.0); // |0⟩ → |1⟩
        assert!((qubit_p1(&state, 0) - 1.0).abs() < 1e-10);
        NoiseChannel::BitFlip { prob: 1.0 }.apply(&mut state, 0, 0.0); // |1⟩ → |0⟩
        assert!(qubit_p1(&state, 0).abs() < 1e-10);
    }

    // ── PhaseFlip ─────────────────────────────────────────────────────

    #[test]
    fn test_phaseflip_p0_no_change() {
        let mut state = StateVector::new(1);
        // |0⟩ is a Z eigenstate — Z|0⟩ = |0⟩ so phase flip has no probability effect
        NoiseChannel::PhaseFlip { prob: 0.0 }.apply(&mut state, 0, 0.0);
        assert!(qubit_p1(&state, 0).abs() < 1e-10);
    }

    #[test]
    fn test_phaseflip_on_superposition() {
        // H|0⟩ = (|0⟩+|1⟩)/√2, PhaseFlip(p=1) applies Z → (|0⟩-|1⟩)/√2
        // Probabilities unchanged (Z only flips phase, not |amplitude|²)
        use crate::core::gates::{apply_single_qubit_gate, hadamard};
        let mut state = StateVector::new(1);
        apply_single_qubit_gate(&mut state, &hadamard(), 0);
        let p1_before = qubit_p1(&state, 0);
        NoiseChannel::PhaseFlip { prob: 1.0 }.apply(&mut state, 0, 0.0);
        let p1_after = qubit_p1(&state, 0);
        assert!((p1_before - p1_after).abs() < 1e-10,
            "PhaseFlip doesn't change probabilities");
    }

    // ── Depolarizing ──────────────────────────────────────────────────

    #[test]
    fn test_depolarizing_p0_no_error() {
        let mut state = StateVector::new(1);
        NoiseChannel::Depolarizing { prob: 0.0 }.apply(&mut state, 0, 0.5);
        assert!(qubit_p1(&state, 0).abs() < 1e-10);
    }

    #[test]
    fn test_depolarizing_applies_x_in_first_third() {
        // rng = 0.0 < p/3 = 0.1 → applies X
        let mut state = StateVector::new(1);
        NoiseChannel::Depolarizing { prob: 0.3 }.apply(&mut state, 0, 0.0);
        assert!((qubit_p1(&state, 0) - 1.0).abs() < 1e-10,
            "X should flip |0⟩ to |1⟩");
    }

    #[test]
    fn test_depolarizing_no_error_outside_range() {
        // rng = 0.99 ≥ prob=0.3 → no error
        let mut state = StateVector::new(1);
        NoiseChannel::Depolarizing { prob: 0.3 }.apply(&mut state, 0, 0.99);
        assert!(qubit_p1(&state, 0).abs() < 1e-10, "no error expected");
    }

    // ── AmplitudeDamping ──────────────────────────────────────────────

    #[test]
    fn test_amplitude_damping_zero_from_ground_state() {
        // |0⟩ has P(|1⟩)=0, so p_decay=0 regardless of gamma — K0 applied
        let mut state = StateVector::new(1);
        NoiseChannel::AmplitudeDamping { gamma: 1.0 }.apply(&mut state, 0, 0.0);
        // K0 applied: [[1,0],[0,0]] → |0⟩ → |0⟩ (after normalize)
        assert!(qubit_p1(&state, 0).abs() < 1e-10, "ground state stays ground");
    }

    #[test]
    fn test_amplitude_damping_decays_excited_state() {
        // Start in |1⟩: apply BitFlip(p=1) to go from |0⟩ to |1⟩
        let mut state = StateVector::new(1);
        NoiseChannel::BitFlip { prob: 1.0 }.apply(&mut state, 0, 0.0); // → |1⟩
        assert!((qubit_p1(&state, 0) - 1.0).abs() < 1e-10);

        // gamma=1.0, P(|1⟩)=1.0 → p_decay=1.0 → K1 always applied
        // K1|1⟩ = [[0,√1],[0,0]]|1⟩ = √1·|0⟩ = |0⟩ after normalize
        NoiseChannel::AmplitudeDamping { gamma: 1.0 }.apply(&mut state, 0, 0.0);
        assert!(qubit_p1(&state, 0).abs() < 1e-10,
            "γ=1 should fully decay |1⟩ to |0⟩");
    }

    #[test]
    fn test_amplitude_damping_gamma0_no_change() {
        // gamma=0 → no decay, K0 = I
        let mut state = StateVector::new(1);
        NoiseChannel::BitFlip { prob: 1.0 }.apply(&mut state, 0, 0.0); // → |1⟩
        NoiseChannel::AmplitudeDamping { gamma: 0.0 }.apply(&mut state, 0, 0.5);
        assert!((qubit_p1(&state, 0) - 1.0).abs() < 1e-10,
            "γ=0 should leave state unchanged");
    }
}
