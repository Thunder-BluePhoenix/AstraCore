/// Simulation Backend Plugin — swap-in execution engines.
///
/// `SimulationBackend` abstracts the layer between the executor and the physical
/// compute resource (CPU, GPU, distributed cluster). Today only `CpuBackend`
/// exists, but the trait establishes the interface that future backends must
/// implement.
///
/// # Design note
/// The trait is intentionally minimal: single-qubit gate dispatch is the only
/// required method. Multi-qubit gates have default implementations backed by the
/// existing scalar `apply_*` functions from `crate::core::gates`. A GPU backend
/// would override those defaults with device-resident kernels.
use crate::compiler::AqlError;
use crate::core::{Matrix2x2, StateVector};

// ── SimulationBackend trait ───────────────────────────────────────────────

/// A simulation backend that applies quantum gates to a `StateVector`.
///
/// Used by `execute_with_plugins` to dispatch every gate instruction. Each gate
/// is forwarded to the backend rather than being hardcoded to a specific path.
///
/// # Required method
/// Only `apply_single_qubit_gate` must be implemented. All multi-qubit gates
/// have default implementations that call the CPU scalar path.
///
/// # Thread safety
/// All backends must be `Send + Sync` so they can be stored in a `PluginRegistry`
/// and shared across threads.
pub trait SimulationBackend: Send + Sync {
    /// Descriptive name for logging and introspection (e.g. `"cpu"`, `"gpu"`).
    fn name(&self) -> &str;

    /// Apply a single-qubit unitary gate to `state`.
    ///
    /// `matrix` is a `Matrix2x2` — a 2×2 complex unitary (row-major).
    /// `target` is the global qubit index (0-based).
    ///
    /// Called for: H, X, Y, Z, S, T, Rx, Ry, Rz, Phase.
    fn apply_single_qubit_gate(
        &self,
        state: &mut StateVector,
        matrix: &Matrix2x2,
        target: usize,
    ) -> Result<(), AqlError>;

    /// Apply a CNOT gate (`control` → `target`).
    ///
    /// Default: delegates to `crate::core::gates::apply_cnot` (scalar path).
    /// Override for hardware-accelerated two-qubit entangling operations.
    fn apply_cnot(
        &self,
        state: &mut StateVector,
        control: usize,
        target: usize,
    ) -> Result<(), AqlError> {
        crate::core::gates::apply_cnot(state, control, target);
        Ok(())
    }

    /// Apply a CZ gate.
    ///
    /// Default: delegates to `crate::core::gates::apply_cz`.
    fn apply_cz(
        &self,
        state: &mut StateVector,
        control: usize,
        target: usize,
    ) -> Result<(), AqlError> {
        crate::core::gates::apply_cz(state, control, target);
        Ok(())
    }

    /// Apply a SWAP gate.
    ///
    /// Default: delegates to `crate::core::gates::apply_swap`.
    fn apply_swap(
        &self,
        state: &mut StateVector,
        qubit_a: usize,
        qubit_b: usize,
    ) -> Result<(), AqlError> {
        crate::core::gates::apply_swap(state, qubit_a, qubit_b);
        Ok(())
    }

    /// Apply a Toffoli (CCX) gate.
    ///
    /// Default: delegates to `crate::core::gates::apply_toffoli`.
    fn apply_toffoli(
        &self,
        state: &mut StateVector,
        control0: usize,
        control1: usize,
        target: usize,
    ) -> Result<(), AqlError> {
        crate::core::gates::apply_toffoli(state, control0, control1, target);
        Ok(())
    }
}

// ── CpuBackend ────────────────────────────────────────────────────────────

/// CPU backend: routes single-qubit gates through `apply_gate_simd`.
///
/// At runtime, `apply_gate_simd` selects AVX2 (for x86-64 with AVX2 support)
/// or the scalar path — the same dispatch used by `Simulator::h()`, `Simulator::x()`,
/// etc. Multi-qubit gates fall through to the `SimulationBackend` default methods.
pub struct CpuBackend;

impl SimulationBackend for CpuBackend {
    fn name(&self) -> &str {
        "cpu"
    }

    fn apply_single_qubit_gate(
        &self,
        state: &mut StateVector,
        matrix: &Matrix2x2,
        target: usize,
    ) -> Result<(), AqlError> {
        crate::core::simd::apply_gate_simd(state, matrix, target);
        Ok(())
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{StateVector, gates::{hadamard, pauli_x}};
    use crate::core::simd::apply_gate_simd;

    #[test]
    fn cpu_backend_name() {
        assert_eq!(CpuBackend.name(), "cpu");
    }

    #[test]
    fn cpu_backend_single_qubit_matches_simd_direct() {
        let gate = hadamard();
        let mut s1 = StateVector::new(2);
        let mut s2 = StateVector::new(2);

        CpuBackend.apply_single_qubit_gate(&mut s1, &gate, 0).unwrap();
        apply_gate_simd(&mut s2, &gate, 0);

        for (a, b) in s1.amplitudes.iter().zip(s2.amplitudes.iter()) {
            assert!((a.re - b.re).abs() < 1e-12);
            assert!((a.im - b.im).abs() < 1e-12);
        }
    }

    #[test]
    fn cpu_backend_applies_x_correctly() {
        let mut state = StateVector::new(1);
        CpuBackend.apply_single_qubit_gate(&mut state, &pauli_x(), 0).unwrap();
        assert!((state.amplitudes[1].re - 1.0).abs() < 1e-10);
        assert!(state.amplitudes[0].re.abs() < 1e-10);
    }

    #[test]
    fn cpu_backend_cnot_default() {
        let mut state = StateVector::new(2);
        // Put q0 into |1⟩
        CpuBackend.apply_single_qubit_gate(&mut state, &pauli_x(), 0).unwrap();
        // CNOT(0→1) should flip q1
        CpuBackend.apply_cnot(&mut state, 0, 1).unwrap();
        // State should be |11⟩ = index 3
        assert!((state.amplitudes[3].re - 1.0).abs() < 1e-10);
    }
}
