/// AQL Instruction Scheduler — Phase 3/5.
///
/// The scheduler transforms a parsed Program before execution:
///   1. Peephole optimization (Phase 5) — gate cancellation and rotation merging
///   2. Instruction reordering (future)  — commutativity-based depth reduction
///   3. SIMD batching hints (future)     — group single-qubit gates by qubit
///
/// Control-flow programs (LABEL/GOTO/IF) skip peephole optimization because
/// branching invalidates qubit-adjacency analysis.
use crate::compiler::ir::{Instruction, Program};
use crate::optimizer::peephole;

/// A scheduled (and optimized) program ready for execution.
#[derive(Debug, Clone)]
pub struct ScheduledProgram {
    pub num_qubits: usize,
    pub instructions: Vec<Instruction>,
    pub metadata: SchedulerMetadata,
}

#[derive(Debug, Clone, Default)]
pub struct SchedulerMetadata {
    /// Gate merges / cancellations performed by the peephole optimizer.
    pub gates_merged: usize,
    /// Gate count after optimization (circuit depth proxy).
    pub circuit_depth: usize,
    /// Number of optimizer passes to reach fixed point.
    pub optimizer_passes: usize,
}

/// Schedule and optimize a program for execution.
///
/// Runs the peephole optimizer then returns the transformed program with
/// accompanying metadata.  Control-flow programs pass through unchanged.
pub fn schedule(program: &Program) -> ScheduledProgram {
    let (optimized, stats) = peephole::optimize(&program.instructions);
    ScheduledProgram {
        num_qubits: program.num_qubits,
        instructions: optimized,
        metadata: SchedulerMetadata {
            gates_merged: stats.gates_removed,
            circuit_depth: stats.gates_after,
            optimizer_passes: stats.passes,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::ir::{Instruction, Program};

    #[test]
    fn test_schedule_passthrough_no_optimizable_gates() {
        // Bell circuit has no cancellable pairs — should be unchanged
        let prog = Program::new(2, vec![
            Instruction::H(0),
            Instruction::Cnot { control: 0, target: 1 },
            Instruction::MeasureAll,
        ]);
        let scheduled = schedule(&prog);
        assert_eq!(scheduled.num_qubits, prog.num_qubits);
        assert_eq!(scheduled.instructions.len(), prog.instructions.len());
        assert_eq!(scheduled.metadata.gates_merged, 0);
        assert_eq!(scheduled.metadata.circuit_depth, prog.gate_count);
    }

    #[test]
    fn test_schedule_cancels_hh() {
        // H·H·CNOT: H pair should be eliminated
        let prog = Program::new(2, vec![
            Instruction::H(0),
            Instruction::H(0),
            Instruction::Cnot { control: 0, target: 1 },
            Instruction::MeasureAll,
        ]);
        let scheduled = schedule(&prog);
        assert_eq!(scheduled.metadata.gates_merged, 2);
        // Remaining: CNOT + MeasureAll
        assert_eq!(
            scheduled.instructions.len(), 2,
            "H·H should cancel, leaving CNOT + MeasureAll"
        );
    }

    #[test]
    fn test_schedule_merges_rotations() {
        // Rz(π/4) · Rz(π/4) → Rz(π/2)
        use std::f64::consts::PI;
        let prog = Program::new(1, vec![
            Instruction::Rz { qubit: 0, theta: PI / 4.0 },
            Instruction::Rz { qubit: 0, theta: PI / 4.0 },
        ]);
        let scheduled = schedule(&prog);
        assert_eq!(scheduled.metadata.gates_merged, 1);
        assert_eq!(scheduled.instructions.len(), 1);
        if let Instruction::Rz { qubit: 0, theta } = scheduled.instructions[0] {
            assert!((theta - PI / 2.0).abs() < 1e-9);
        } else {
            panic!("expected merged Rz");
        }
    }
}
