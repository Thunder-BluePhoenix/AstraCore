/// AstraCore Plugin System
///
/// Extension points for custom gates, optimizer passes, and simulation backends.
/// This is the v1.0 plugin architecture that enables swapping any layer of the
/// AstraCore execution pipeline from Rust code.
///
/// # Quick start
///
/// ```rust
/// use astracore::plugins::{PluginRegistry, FnGatePlugin, run_with_plugins};
/// use astracore::core::gates::{apply_single_qubit_gate, pauli_x};
///
/// // Register a custom Rust-level gate
/// let mut registry = PluginRegistry::default();
/// registry.register_gate(Box::new(FnGatePlugin::new(
///     "my_x",
///     1,
///     |state, qubits| {
///         apply_single_qubit_gate(state, &pauli_x(), qubits[0]);
///     },
/// )));
///
/// // Execute AQL that calls the plugin gate via CALL
/// let result = run_with_plugins("QREG 1\nCALL my_x 0\nMEASURE 0", &registry).unwrap();
/// assert_eq!(result.outcome(0), Some(true));
/// ```
///
/// # Extension axes
///
/// | Axis | Trait | Built-in |
/// |------|-------|---------|
/// | Custom gates | [`GatePlugin`] | [`FnGatePlugin`] (closure wrapper) |
/// | Optimizer passes | [`OptimizerPass`] | [`PeepholePass`] (wraps existing optimizer) |
/// | Simulation backend | [`SimulationBackend`] | [`CpuBackend`] (AVX2/scalar dispatch) |
///
/// # Registry
/// Build a [`PluginRegistry`] and pass it to [`execute_with_plugins`] or
/// [`run_with_plugins`]. Use `PluginRegistry::default()` to get started with
/// the CPU backend and peephole optimizer pre-loaded.
pub mod backend;
pub mod gate;
pub mod integration;
pub mod optimizer;
pub mod registry;

// ── Public re-exports ──────────────────────────────────────────────────────
pub use backend::{CpuBackend, SimulationBackend};
pub use gate::{FnGatePlugin, GatePlugin};
pub use integration::{execute_with_plugins, run_with_plugins};
pub use optimizer::{OptimizerPass, PeepholePass};
pub use registry::PluginRegistry;
