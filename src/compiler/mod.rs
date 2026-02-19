/// AQL — Astra Quantum Language compiler pipeline.
///
/// Pipeline: source text → Lexer → Parser → IR → Executor → Result
///
/// Each stage is a separate module with a clean boundary.
/// Errors carry source line numbers for precise diagnostics.
pub mod executor;
pub mod ir;
pub mod lexer;
pub mod parser;

pub use executor::{execute, ExecutionResult, MeasurementRecord};
pub use ir::{Instruction, Program};
pub use parser::parse;

/// Error type shared across all compiler stages.
#[derive(Debug, Clone)]
pub enum AqlError {
    /// Unrecognized token in source
    Lex { line: usize, msg: String },
    /// Syntactically invalid instruction
    Parse { line: usize, msg: String },
    /// Semantically invalid program (e.g. out-of-range qubit)
    Validation { msg: String },
    /// Runtime execution failure
    Runtime { msg: String },
}

impl std::fmt::Display for AqlError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Lex { line, msg }        => write!(f, "Lex error   line {}: {}", line, msg),
            Self::Parse { line, msg }      => write!(f, "Parse error line {}: {}", line, msg),
            Self::Validation { msg }       => write!(f, "Validation error: {}", msg),
            Self::Runtime { msg }          => write!(f, "Runtime error: {}", msg),
        }
    }
}

impl std::error::Error for AqlError {}

/// One-shot: lex + parse an AQL source string, return the validated Program.
pub fn parse_source(source: &str) -> Result<Program, AqlError> {
    let tokens = lexer::tokenize(source)?;
    parser::parse(tokens)
}

/// One-shot: lex + parse + execute. Returns the execution result.
pub fn run(source: &str) -> Result<ExecutionResult, AqlError> {
    let program = parse_source(source)?;
    executor::execute(&program)
}
