/// Plugin Registry — central store for gate plugins, optimizer passes, and the backend.
///
/// Build a registry with the builder-style methods, then pass it to
/// `execute_with_plugins` or `run_with_plugins`.
///
/// # Defaults
/// `PluginRegistry::default()` pre-loads:
/// - `CpuBackend` as the active backend.
/// - `PeepholePass` as the only optimizer pass (same behaviour as `run_optimized`).
/// - No gate plugins.
///
/// `PluginRegistry::new()` creates a clean registry with only `CpuBackend` and no passes.
use std::collections::HashMap;
use crate::compiler::ir::Instruction;
use crate::optimizer::OptimizationStats;
use super::gate::GatePlugin;
use super::optimizer::{OptimizerPass, PeepholePass};
use super::backend::{SimulationBackend, CpuBackend};

// ── PluginRegistry ────────────────────────────────────────────────────────

pub struct PluginRegistry {
    /// Gate plugins keyed by lowercase name.
    pub(crate) gates: HashMap<String, Box<dyn GatePlugin>>,
    /// Optimizer passes, run in registration order.
    pub(crate) optimizer_passes: Vec<Box<dyn OptimizerPass>>,
    /// Active simulation backend.
    pub(crate) backend: Box<dyn SimulationBackend>,
}

impl Default for PluginRegistry {
    /// Registry with `CpuBackend` + `PeepholePass` and no gate plugins.
    fn default() -> Self {
        Self {
            gates: HashMap::new(),
            optimizer_passes: vec![Box::new(PeepholePass)],
            backend: Box::new(CpuBackend),
        }
    }
}

impl PluginRegistry {
    // ── Construction ──────────────────────────────────────────────────────

    /// Create an empty registry with `CpuBackend` and no optimizer passes.
    pub fn new() -> Self {
        Self {
            gates: HashMap::new(),
            optimizer_passes: Vec::new(),
            backend: Box::new(CpuBackend),
        }
    }

    // ── Gate plugins ──────────────────────────────────────────────────────

    /// Register a gate plugin. The plugin's `name()` (lowercased) is the lookup key.
    ///
    /// Silently replaces any previously registered plugin with the same name.
    /// Returns `&mut Self` for method chaining.
    pub fn register_gate(&mut self, plugin: Box<dyn GatePlugin>) -> &mut Self {
        self.gates.insert(plugin.name().to_lowercase(), plugin);
        self
    }

    /// Look up a registered gate plugin by name (case-insensitive).
    ///
    /// Returns `None` if no plugin with that name is registered.
    pub fn get_gate(&self, name: &str) -> Option<&dyn GatePlugin> {
        self.gates.get(&name.to_lowercase()).map(|b| b.as_ref())
    }

    /// Number of registered gate plugins.
    pub fn gate_count(&self) -> usize {
        self.gates.len()
    }

    /// Names of all registered gate plugins (lowercased, arbitrary order).
    pub fn gate_names(&self) -> Vec<&str> {
        self.gates.keys().map(|s| s.as_str()).collect()
    }

    // ── Optimizer passes ──────────────────────────────────────────────────

    /// Append an optimizer pass to the end of the pipeline.
    ///
    /// Passes run in the order they were added. Returns `&mut Self`.
    pub fn add_optimizer_pass(&mut self, pass: Box<dyn OptimizerPass>) -> &mut Self {
        self.optimizer_passes.push(pass);
        self
    }

    /// Remove all registered optimizer passes.
    pub fn clear_optimizer_passes(&mut self) -> &mut Self {
        self.optimizer_passes.clear();
        self
    }

    /// Number of registered optimizer passes.
    pub fn optimizer_pass_count(&self) -> usize {
        self.optimizer_passes.len()
    }

    /// Run the full optimizer pipeline over `instructions`.
    ///
    /// Each pass receives the output of the previous pass. If no passes are
    /// registered the instructions are returned unchanged.
    ///
    /// The returned `OptimizationStats` aggregates `gates_removed` and `passes`
    /// across the entire pipeline.
    pub fn run_optimizer_pipeline(
        &self,
        instructions: &[Instruction],
    ) -> (Vec<Instruction>, OptimizationStats) {
        if self.optimizer_passes.is_empty() {
            let n = instructions.iter().filter(|i| i.is_gate()).count();
            return (
                instructions.to_vec(),
                OptimizationStats {
                    gates_before: n,
                    gates_after:  n,
                    gates_removed: 0,
                    passes: 0,
                },
            );
        }

        let gates_before = instructions.iter().filter(|i| i.is_gate()).count();
        let mut current = instructions.to_vec();
        let mut total_removed = 0usize;
        let mut total_passes  = 0usize;

        for pass in &self.optimizer_passes {
            let (next, stats) = pass.run(&current);
            total_removed += stats.gates_removed;
            total_passes  += stats.passes;
            current = next;
        }

        let gates_after = current.iter().filter(|i| i.is_gate()).count();
        (
            current,
            OptimizationStats {
                gates_before,
                gates_after,
                gates_removed: total_removed,
                passes: total_passes,
            },
        )
    }

    // ── Backend ───────────────────────────────────────────────────────────

    /// Replace the active simulation backend.
    pub fn set_backend(&mut self, backend: Box<dyn SimulationBackend>) -> &mut Self {
        self.backend = backend;
        self
    }

    /// Reference to the active backend.
    pub fn backend(&self) -> &dyn SimulationBackend {
        self.backend.as_ref()
    }

    /// Name of the active backend.
    pub fn backend_name(&self) -> &str {
        self.backend.name()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plugins::gate::FnGatePlugin;
    use crate::plugins::optimizer::PeepholePass;
    use crate::compiler::ir::Instruction;

    #[test]
    fn default_registry_zero_gates() {
        let reg = PluginRegistry::default();
        assert_eq!(reg.gate_count(), 0);
    }

    #[test]
    fn default_registry_backend_is_cpu() {
        let reg = PluginRegistry::default();
        assert_eq!(reg.backend_name(), "cpu");
    }

    #[test]
    fn default_registry_has_one_pass() {
        let reg = PluginRegistry::default();
        assert_eq!(reg.optimizer_pass_count(), 1);
    }

    #[test]
    fn new_registry_no_passes() {
        let reg = PluginRegistry::new();
        assert_eq!(reg.optimizer_pass_count(), 0);
    }

    #[test]
    fn register_gate_lookup() {
        let mut reg = PluginRegistry::new();
        reg.register_gate(Box::new(FnGatePlugin::new("mygate", 1, |_, _| {})));
        assert!(reg.get_gate("mygate").is_some());
        assert!(reg.get_gate("MYGATE").is_some()); // case-insensitive
        assert!(reg.get_gate("other").is_none());
    }

    #[test]
    fn register_gate_count() {
        let mut reg = PluginRegistry::new();
        reg.register_gate(Box::new(FnGatePlugin::new("a", 1, |_, _| {})));
        reg.register_gate(Box::new(FnGatePlugin::new("b", 1, |_, _| {})));
        assert_eq!(reg.gate_count(), 2);
    }

    #[test]
    fn register_gate_overwrites_same_name() {
        let mut reg = PluginRegistry::new();
        reg.register_gate(Box::new(FnGatePlugin::new("g", 1, |_, _| {})));
        reg.register_gate(Box::new(FnGatePlugin::new("g", 2, |_, _| {})));
        assert_eq!(reg.gate_count(), 1);
        assert_eq!(reg.get_gate("g").unwrap().num_qubits(), 2);
    }

    #[test]
    fn add_and_clear_optimizer_passes() {
        let mut reg = PluginRegistry::new();
        reg.add_optimizer_pass(Box::new(PeepholePass));
        reg.add_optimizer_pass(Box::new(PeepholePass));
        assert_eq!(reg.optimizer_pass_count(), 2);
        reg.clear_optimizer_passes();
        assert_eq!(reg.optimizer_pass_count(), 0);
    }

    #[test]
    fn run_optimizer_pipeline_empty_passes() {
        let reg = PluginRegistry::new(); // no passes
        let instrs = vec![Instruction::H(0), Instruction::H(0)];
        let (out, stats) = reg.run_optimizer_pipeline(&instrs);
        assert_eq!(out.len(), 2, "no passes → instructions unchanged");
        assert_eq!(stats.gates_removed, 0);
        assert_eq!(stats.passes, 0);
    }

    #[test]
    fn run_optimizer_pipeline_peephole_cancels() {
        let mut reg = PluginRegistry::new();
        reg.add_optimizer_pass(Box::new(PeepholePass));
        let instrs = vec![Instruction::H(0), Instruction::H(0)];
        let (out, stats) = reg.run_optimizer_pipeline(&instrs);
        assert!(out.is_empty());
        assert_eq!(stats.gates_removed, 2);
    }

    #[test]
    fn run_optimizer_pipeline_two_passes_aggregate() {
        // Two passes, each cancels a different pair
        let mut reg = PluginRegistry::new();
        reg.add_optimizer_pass(Box::new(PeepholePass));
        reg.add_optimizer_pass(Box::new(PeepholePass));
        // H·H·X·X — after first pass both pairs cancelled
        let instrs = vec![
            Instruction::H(0), Instruction::H(0),
            Instruction::X(1), Instruction::X(1),
        ];
        let (out, stats) = reg.run_optimizer_pipeline(&instrs);
        assert!(out.is_empty());
        assert_eq!(stats.gates_removed, 4);
    }
}
