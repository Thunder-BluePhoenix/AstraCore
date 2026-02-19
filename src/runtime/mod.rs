/// AstraCore Hybrid Runtime — Phase 4
///
/// The runtime layer sits above the compiler pipeline and handles:
///   - PC-based execution with classical control flow
///   - Classical register file (measurement outcomes)
///   - Conditional branching (IF/GOTO)
///   - Instruction scheduling (stub → Phase 5)
///
/// Architecture:
///   AQL source → Compiler (lexer+parser+IR) → Runtime Executor → Result
///
/// The hybrid executor is the definitive execution engine.
/// It handles both simple circuits (no control flow) and full hybrid programs.
pub mod executor;
pub mod scheduler;

pub use executor::{execute, ExecutionResult, MeasurementRecord};
pub use scheduler::schedule;
