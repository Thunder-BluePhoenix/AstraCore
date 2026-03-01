/// Optimizer Pass Plugin — composable instruction-level optimization passes.
///
/// Any type implementing `OptimizerPass` can be added to a `PluginRegistry`
/// and will be run in registration order by `PluginRegistry::run_optimizer_pipeline`.
///
/// The existing peephole optimizer is wrapped as `PeepholePass` so it participates
/// in the same pipeline without any changes to its implementation.
use crate::compiler::ir::Instruction;
use crate::optimizer::OptimizationStats;

// ── OptimizerPass trait ───────────────────────────────────────────────────

/// A single optimization pass over an AQL instruction sequence.
///
/// # Contract
/// - The pass must preserve the quantum semantics of the instruction sequence.
/// - The returned `Vec<Instruction>` replaces the input; it may be shorter, equal,
///   or (in rare cases) longer.
/// - Passes are composable: each pass receives the output of the previous pass.
/// - All implementations must be `Send + Sync`.
///
/// # Example — inline identity pass
/// ```rust
/// use astracore::plugins::OptimizerPass;
/// use astracore::compiler::ir::Instruction;
/// use astracore::optimizer::OptimizationStats;
///
/// struct IdentityPass;
/// impl OptimizerPass for IdentityPass {
///     fn name(&self) -> &str { "identity" }
///     fn run(&self, instructions: &[Instruction]) -> (Vec<Instruction>, OptimizationStats) {
///         let n = instructions.len();
///         (
///             instructions.to_vec(),
///             OptimizationStats { gates_before: n, gates_after: n, gates_removed: 0, passes: 1 },
///         )
///     }
/// }
/// ```
pub trait OptimizerPass: Send + Sync {
    /// Human-readable name of this pass, used in pipeline diagnostics.
    fn name(&self) -> &str;

    /// Run the pass over `instructions`.
    ///
    /// Returns `(transformed_instructions, stats)`.
    /// `stats.gates_removed` must accurately reflect actual reductions so the
    /// pipeline can aggregate totals correctly.
    fn run(&self, instructions: &[Instruction]) -> (Vec<Instruction>, OptimizationStats);
}

// ── PeepholePass ──────────────────────────────────────────────────────────

/// Wraps the built-in peephole optimizer as an `OptimizerPass`.
///
/// Cancels adjacent inverse gate pairs (H·H, X·X, Z·Z, CNOT·CNOT, S^4, T^8, etc.)
/// in a fixed-point loop. See `crate::optimizer::peephole` for full rule set.
///
/// This is the default pass in `PluginRegistry::default()`.
pub struct PeepholePass;

impl OptimizerPass for PeepholePass {
    fn name(&self) -> &str {
        "peephole"
    }

    fn run(&self, instructions: &[Instruction]) -> (Vec<Instruction>, OptimizationStats) {
        crate::optimizer::peephole::optimize(instructions)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::ir::Instruction;

    #[test]
    fn peephole_pass_name() {
        assert_eq!(PeepholePass.name(), "peephole");
    }

    #[test]
    fn peephole_pass_cancels_hh() {
        let instrs = vec![Instruction::H(0), Instruction::H(0)];
        let (out, stats) = PeepholePass.run(&instrs);
        assert!(out.is_empty(), "H·H should cancel");
        assert_eq!(stats.gates_removed, 2);
    }

    #[test]
    fn peephole_pass_cancels_xx() {
        let instrs = vec![Instruction::X(1), Instruction::X(1)];
        let (out, stats) = PeepholePass.run(&instrs);
        assert!(out.is_empty());
        assert_eq!(stats.gates_removed, 2);
    }

    #[test]
    fn peephole_pass_noop_on_different_qubits() {
        let instrs = vec![Instruction::H(0), Instruction::H(1)];
        let (out, stats) = PeepholePass.run(&instrs);
        assert_eq!(out.len(), 2, "different qubits must not cancel");
        assert_eq!(stats.gates_removed, 0);
    }
}
