/// Quantum state vector representation.
///
/// An n-qubit system has 2^n basis states.
/// The state vector holds one complex amplitude per basis state.
/// The vector must satisfy the normalization constraint: Σ|αᵢ|² = 1
use super::complex::Complex;
use std::fmt;

pub struct StateVector {
    pub num_qubits: usize,
    pub amplitudes: Vec<Complex>,
}

impl StateVector {
    /// Create a new state vector initialized to |0...0⟩.
    /// All amplitude is concentrated in the all-zeros basis state.
    pub fn new(num_qubits: usize) -> Self {
        assert!(num_qubits >= 1, "at least one qubit required");
        assert!(num_qubits <= 30, "num_qubits > 30 would require >8 GB RAM");

        let dim = 1 << num_qubits; // 2^n
        let mut amplitudes = vec![Complex::zero(); dim];
        amplitudes[0] = Complex::one(); // |0...0⟩ state

        Self {
            num_qubits,
            amplitudes,
        }
    }

    /// Dimension of the state space: 2^n
    #[inline(always)]
    pub fn dim(&self) -> usize {
        self.amplitudes.len()
    }

    /// Probability of measuring basis state at index `i`: |αᵢ|²
    #[inline(always)]
    pub fn probability(&self, index: usize) -> f64 {
        self.amplitudes[index].norm_sq()
    }

    /// Total probability (should be ≈ 1.0 after normalization)
    pub fn total_probability(&self) -> f64 {
        self.amplitudes.iter().map(|a| a.norm_sq()).sum()
    }

    /// Re-normalize the state vector to unit length.
    /// Used after numerical drift in long simulation chains.
    pub fn normalize(&mut self) {
        let total = self.total_probability();
        if total < 1e-12 {
            panic!("State vector collapsed to zero — unphysical state");
        }
        let inv_norm = 1.0 / total.sqrt();
        for amp in self.amplitudes.iter_mut() {
            *amp = amp.scale(inv_norm);
        }
    }

    /// Reset to |0...0⟩
    pub fn reset(&mut self) {
        for amp in self.amplitudes.iter_mut() {
            *amp = Complex::zero();
        }
        self.amplitudes[0] = Complex::one();
    }

    /// Check if this qubit's bit is set in basis state index `basis_idx`.
    /// Qubit 0 is the least-significant bit.
    #[inline(always)]
    pub fn qubit_bit(basis_idx: usize, qubit: usize) -> bool {
        (basis_idx >> qubit) & 1 == 1
    }

    /// Return a string representation of basis state `index` as a ket |010...⟩.
    /// Qubit 0 is rightmost (LSB convention).
    pub fn basis_label(&self, index: usize) -> String {
        let mut s = String::with_capacity(self.num_qubits);
        for q in (0..self.num_qubits).rev() {
            s.push(if Self::qubit_bit(index, q) { '1' } else { '0' });
        }
        s
    }

    /// Return the probability of measuring qubit `q` in state |1⟩,
    /// marginalized over all other qubits.
    pub fn marginal_probability_one(&self, qubit: usize) -> f64 {
        self.amplitudes
            .iter()
            .enumerate()
            .filter(|(i, _)| Self::qubit_bit(*i, qubit))
            .map(|(_, a)| a.norm_sq())
            .sum()
    }

    /// Perform projective measurement collapse on qubit `q`.
    ///
    /// Returns `true` if the qubit measured as |1⟩, `false` for |0⟩.
    /// The state vector is collapsed and re-normalized in-place.
    pub fn collapse(&mut self, qubit: usize, rng: f64) -> bool {
        let prob_one = self.marginal_probability_one(qubit);
        let outcome = rng < prob_one;

        // Zero out amplitudes inconsistent with measurement outcome
        for i in 0..self.dim() {
            if Self::qubit_bit(i, qubit) != outcome {
                self.amplitudes[i] = Complex::zero();
            }
        }

        self.normalize();
        outcome
    }
}

impl fmt::Display for StateVector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "StateVector ({} qubits, dim={}):", self.num_qubits, self.dim())?;
        for (i, amp) in self.amplitudes.iter().enumerate() {
            let prob = amp.norm_sq();
            if prob > 1e-12 {
                writeln!(
                    f,
                    "  |{}⟩  amplitude: {}  probability: {:.4}",
                    self.basis_label(i),
                    amp,
                    prob
                )?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_state() {
        let sv = StateVector::new(2);
        assert_eq!(sv.dim(), 4);
        assert_eq!(sv.amplitudes[0], Complex::one());
        for i in 1..4 {
            assert_eq!(sv.amplitudes[i], Complex::zero());
        }
    }

    #[test]
    fn test_total_probability_initial() {
        let sv = StateVector::new(3);
        assert!((sv.total_probability() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_normalize() {
        let mut sv = StateVector::new(1);
        // Manually scale to break normalization
        sv.amplitudes[0] = Complex::new(2.0, 0.0);
        sv.amplitudes[1] = Complex::new(0.0, 0.0);
        sv.normalize();
        assert!((sv.total_probability() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_qubit_bit() {
        // basis index 5 = 0b101: qubit 0 = 1, qubit 1 = 0, qubit 2 = 1
        assert!(StateVector::qubit_bit(5, 0));
        assert!(!StateVector::qubit_bit(5, 1));
        assert!(StateVector::qubit_bit(5, 2));
    }

    #[test]
    fn test_basis_label() {
        let sv = StateVector::new(3);
        assert_eq!(sv.basis_label(0), "000");
        assert_eq!(sv.basis_label(1), "001");
        assert_eq!(sv.basis_label(5), "101");
        assert_eq!(sv.basis_label(7), "111");
    }

    #[test]
    fn test_marginal_probability() {
        let mut sv = StateVector::new(1);
        // Equal superposition: |0⟩ and |1⟩ with equal amplitude
        let amp = Complex::new(1.0 / 2.0_f64.sqrt(), 0.0);
        sv.amplitudes[0] = amp;
        sv.amplitudes[1] = amp;
        let p = sv.marginal_probability_one(0);
        assert!((p - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_reset() {
        let mut sv = StateVector::new(2);
        sv.amplitudes[0] = Complex::zero();
        sv.amplitudes[3] = Complex::one();
        sv.reset();
        assert_eq!(sv.amplitudes[0], Complex::one());
        assert_eq!(sv.amplitudes[3], Complex::zero());
    }
}
