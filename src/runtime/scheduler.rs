/// AQL Instruction Scheduler — Phase 3 stub.
///
/// The scheduler's future role is to reorder commuting quantum gates to:
///   - Minimize circuit depth
///   - Improve qubit locality (reduce crosstalk on real hardware)
///   - Enable SIMD-friendly batching of single-qubit gates
///   - Detect and merge adjacent canceling gates (H·H = I, X·X = I)
///
/// Current implementation: identity pass (no reordering).
/// This maintains correctness while the optimizer is developed in Phase 3/5.
use crate::compiler::ir::{Instruction, Program};

/// A scheduled program ready for execution by the runtime executor.
/// Currently identical to the source program (no-op pass-through).
#[derive(Debug, Clone)]
pub struct ScheduledProgram {
    pub num_qubits: usize,
    pub instructions: Vec<Instruction>,
    /// Annotations from the scheduler (empty for now).
    pub metadata: SchedulerMetadata,
}

#[derive(Debug, Clone, Default)]
pub struct SchedulerMetadata {
    /// How many gate merges the scheduler performed (0 = no-op pass).
    pub gates_merged: usize,
    /// Estimated circuit depth after scheduling.
    pub circuit_depth: usize,
}

impl From<&Program> for ScheduledProgram {
    fn from(program: &Program) -> Self {
        Self {
            num_qubits: program.num_qubits,
            instructions: program.instructions.clone(),
            metadata: SchedulerMetadata {
                gates_merged: 0,
                circuit_depth: program.gate_count,
            },
        }
    }
}

/// Schedule a program for execution.
///
/// Phase 3 stub: passes the program through unchanged.
/// Future: gate cancellation, commutativity reordering, SIMD batching.
pub fn schedule(program: &Program) -> ScheduledProgram {
    ScheduledProgram::from(program)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::ir::{Instruction, Program};

    #[test]
    fn test_schedule_passthrough() {
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
}
