/// AQL — Astra Quantum Language compiler pipeline.
///
/// Pipeline: source text → Lexer → Parser → IR → Runtime Executor → Result
///
/// Each stage is a separate module with a clean boundary.
/// Errors carry source line numbers for precise diagnostics.
///
/// # AQL v2 additions
/// - `REPEAT N … END` loops (unrolled at parse time)
/// - `INCLUDE filename.aql` (file-level source inclusion)
/// - `parse_source_file(path)` — reads, preprocesses, and compiles `.aql` files
/// - `QREG name[n]` named qubit registers (resolved to absolute indices)
/// - `IFMEASURED q THEN … END` / `IFNOTMEASURED q THEN … END` conditional sugar
/// - `error::Diagnostic` — rich error output with source context and "did you mean?"
pub mod analysis;
pub mod error;
pub mod ir;
pub mod lexer;
pub mod parser;
pub mod qasm_export;
pub mod qasm_import;

pub use analysis::{analyze, CircuitAnalysis};
pub use crate::runtime::{execute, ExecutionResult, MeasurementRecord};
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

/// One-shot: lex + parse + analyze. Returns the `CircuitAnalysis` for the program.
pub fn analyze_source(source: &str) -> Result<CircuitAnalysis, AqlError> {
    let program = parse_source(source)?;
    Ok(analysis::analyze(&program))
}

/// One-shot: lex + parse + execute. Returns the execution result.
pub fn run(source: &str) -> Result<ExecutionResult, AqlError> {
    let program = parse_source(source)?;
    crate::runtime::execute(&program)
}

/// One-shot: lex + parse + **optimize** + execute.
///
/// Runs the peephole optimizer before execution.  Equivalent quantum semantics
/// as `run()` but the instruction stream may be shorter.
/// Custom gate definitions are preserved so CALL instructions still resolve.
pub fn run_optimized(source: &str) -> Result<ExecutionResult, AqlError> {
    let program = parse_source(source)?;
    let (opt_instrs, _stats) = crate::optimizer::peephole::optimize(&program.instructions);
    let opt_program = ir::Program::with_gate_defs(program.num_qubits, opt_instrs, program.gate_defs);
    crate::runtime::execute(&opt_program)
}

/// One-shot: lex + parse + optimize. Returns the optimized `Program` and stats.
///
/// Use this when you want to inspect what the optimizer did before running.
/// Custom gate definitions are preserved in the returned program.
pub fn optimize(source: &str)
    -> Result<(Program, crate::optimizer::OptimizationStats), AqlError>
{
    let program = parse_source(source)?;
    let (opt_instrs, stats) = crate::optimizer::peephole::optimize(&program.instructions);
    let opt_program = ir::Program::with_gate_defs(program.num_qubits, opt_instrs, program.gate_defs);
    Ok((opt_program, stats))
}

// ── AQL v2: file-based compilation ────────────────────────────────────────

/// Preprocess `INCLUDE filename` directives in AQL source text.
///
/// Replaces each `INCLUDE <path>` line with the content of the referenced
/// file, resolved relative to `base_dir`.  Include depth is capped at 16
/// to prevent infinite recursion.
///
/// Syntax: `INCLUDE path/to/file.aql` (no quotes; path ends at end of line).
pub fn preprocess_includes(source: &str, base_dir: &std::path::Path) -> Result<String, AqlError> {
    preprocess_includes_depth(source, base_dir, 0)
}

fn preprocess_includes_depth(
    source: &str,
    base_dir: &std::path::Path,
    depth: usize,
) -> Result<String, AqlError> {
    const MAX_INCLUDE_DEPTH: usize = 16;
    if depth > MAX_INCLUDE_DEPTH {
        return Err(AqlError::Validation {
            msg: format!("INCLUDE nesting exceeds {MAX_INCLUDE_DEPTH} — possible circular include"),
        });
    }

    let mut out = String::with_capacity(source.len());
    for (idx, line) in source.lines().enumerate() {
        let trimmed = line.trim_start();
        if trimmed.to_ascii_uppercase().starts_with("INCLUDE") {
            let rest = trimmed["INCLUDE".len()..].trim();
            if rest.is_empty() {
                return Err(AqlError::Parse {
                    line: idx + 1,
                    msg: "'INCLUDE' expects a filename argument".into(),
                });
            }
            // Strip optional surrounding quotes
            let path_str = rest.trim_matches('"').trim_matches('\'');
            let full_path = base_dir.join(path_str);
            let included = std::fs::read_to_string(&full_path).map_err(|e| {
                AqlError::Validation {
                    msg: format!("INCLUDE '{}': {e}", full_path.display()),
                }
            })?;
            // Recurse for nested includes
            let included_dir = full_path.parent().unwrap_or(base_dir);
            let expanded = preprocess_includes_depth(&included, included_dir, depth + 1)?;
            out.push_str(&expanded);
            out.push('\n');
        } else {
            out.push_str(line);
            out.push('\n');
        }
    }
    Ok(out)
}

/// Compile an AQL source file at `path`.
///
/// Handles `INCLUDE` directives relative to the file's directory.
/// Returns a fully validated `Program` ready for execution.
pub fn parse_source_file(path: &std::path::Path) -> Result<Program, AqlError> {
    let source = std::fs::read_to_string(path).map_err(|e| AqlError::Validation {
        msg: format!("cannot read '{}': {e}", path.display()),
    })?;
    let base_dir = path.parent().unwrap_or(std::path::Path::new("."));
    let preprocessed = preprocess_includes(&source, base_dir)?;
    parse_source(&preprocessed)
}

/// One-shot: read file, preprocess, compile, and execute.
pub fn run_file(path: &std::path::Path) -> Result<ExecutionResult, AqlError> {
    let program = parse_source_file(path)?;
    crate::runtime::execute(&program)
}
