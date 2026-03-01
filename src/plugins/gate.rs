/// Gate Plugin trait — Rust-level custom gates callable from AQL `CALL` instructions
/// or directly from Rust code.
///
/// # Lookup precedence (in `execute_with_plugins`)
/// 1. `program.gate_defs` — AQL `GATE…END` definitions (existing path, highest priority)
/// 2. `PluginRegistry` — registered `GatePlugin` implementations (this module)
/// 3. `AqlError::Runtime` — undefined gate error
///
/// Existing AQL programs with `GATE…END` blocks are completely unaffected; the
/// registry is only consulted when `program.gate_defs` does not contain the name.
use crate::core::StateVector;

// ── GatePlugin trait ──────────────────────────────────────────────────────

/// A Rust-level gate plugin that operates directly on the state vector.
///
/// Implementors receive:
/// - A mutable reference to the live `StateVector` mid-execution.
/// - The global qubit indices the gate should act on (already remapped by caller).
///
/// # Contract
/// - `apply` must preserve normalization: Σ|αᵢ|² = 1 after the call.
/// - `apply` must only touch qubits listed in `qubits`.
/// - All implementations must be `Send + Sync`.
///
/// # Example
/// ```rust
/// use astracore::plugins::{FnGatePlugin, GatePlugin};
/// use astracore::core::gates::{apply_single_qubit_gate, pauli_x};
///
/// let flip = FnGatePlugin::new("flip", 1, |state, qubits| {
///     apply_single_qubit_gate(state, &pauli_x(), qubits[0]);
/// });
/// assert_eq!(flip.name(), "flip");
/// assert_eq!(flip.num_qubits(), 1);
/// ```
pub trait GatePlugin: Send + Sync {
    /// Unique name this plugin registers under (lowercase recommended).
    /// Must match the `CALL` instruction name in AQL source.
    fn name(&self) -> &str;

    /// Number of qubits this gate acts on.
    /// `execute_with_plugins` validates the call-site arity before calling `apply`.
    fn num_qubits(&self) -> usize;

    /// Apply the gate transformation to `state`.
    ///
    /// `qubits` contains exactly `self.num_qubits()` global qubit indices,
    /// already remapped from any enclosing `GATE…END` body.
    fn apply(&self, state: &mut StateVector, qubits: &[usize]);

    /// Optional human-readable description for diagnostics and introspection.
    fn description(&self) -> &str {
        ""
    }
}

// ── FnGatePlugin ──────────────────────────────────────────────────────────

/// Convenience wrapper: construct a `GatePlugin` from a closure.
///
/// # Example
/// ```rust
/// use astracore::plugins::{FnGatePlugin, GatePlugin};
/// use astracore::core::gates::{apply_single_qubit_gate, hadamard, apply_cnot};
///
/// // A Bell-pair factory: applies H then CNOT on two qubits.
/// let bell = FnGatePlugin::new("rust_bell", 2, |state, qubits| {
///     apply_single_qubit_gate(state, &hadamard(), qubits[0]);
///     apply_cnot(state, qubits[0], qubits[1]);
/// });
/// assert_eq!(bell.name(), "rust_bell");
/// assert_eq!(bell.num_qubits(), 2);
/// ```
pub struct FnGatePlugin {
    name: String,
    num_qubits: usize,
    description: String,
    func: Box<dyn Fn(&mut StateVector, &[usize]) + Send + Sync>,
}

impl FnGatePlugin {
    /// Create a new closure-based gate plugin.
    ///
    /// - `name` — the CALL-site name; stored lowercased.
    /// - `num_qubits` — how many qubit arguments this gate accepts.
    /// - `func` — closure implementing the gate transformation.
    pub fn new<F>(name: &str, num_qubits: usize, func: F) -> Self
    where
        F: Fn(&mut StateVector, &[usize]) + Send + Sync + 'static,
    {
        Self {
            name: name.to_lowercase(),
            num_qubits,
            description: String::new(),
            func: Box::new(func),
        }
    }

    /// Attach an optional description string (builder-style).
    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = desc.to_owned();
        self
    }
}

impl GatePlugin for FnGatePlugin {
    fn name(&self) -> &str {
        &self.name
    }

    fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    fn apply(&self, state: &mut StateVector, qubits: &[usize]) {
        (self.func)(state, qubits);
    }

    fn description(&self) -> &str {
        &self.description
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::gates::{apply_single_qubit_gate, pauli_x};
    use crate::core::StateVector;

    #[test]
    fn fn_gate_plugin_name_lowercased() {
        let p = FnGatePlugin::new("MyGate", 1, |_, _| {});
        assert_eq!(p.name(), "mygate");
    }

    #[test]
    fn fn_gate_plugin_num_qubits() {
        let p = FnGatePlugin::new("g", 3, |_, _| {});
        assert_eq!(p.num_qubits(), 3);
    }

    #[test]
    fn fn_gate_plugin_description_default_empty() {
        let p = FnGatePlugin::new("g", 1, |_, _| {});
        assert_eq!(p.description(), "");
    }

    #[test]
    fn fn_gate_plugin_with_description() {
        let p = FnGatePlugin::new("g", 1, |_, _| {}).with_description("test gate");
        assert_eq!(p.description(), "test gate");
    }

    #[test]
    fn fn_gate_plugin_applies_x() {
        let p = FnGatePlugin::new("flip", 1, |state, qubits| {
            apply_single_qubit_gate(state, &pauli_x(), qubits[0]);
        });
        let mut state = StateVector::new(1);
        p.apply(&mut state, &[0]);
        // After X on |0⟩, amplitude of |1⟩ should be 1
        assert!((state.amplitudes[1].re - 1.0).abs() < 1e-10);
        assert!(state.amplitudes[0].re.abs() < 1e-10);
    }
}
