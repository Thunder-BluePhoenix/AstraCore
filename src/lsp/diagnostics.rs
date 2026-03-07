/// AQL diagnostics for the Language Server — maps compile errors to LSP Diagnostic positions.

use crate::compiler::{self, AqlError};

/// A diagnostic location (0-indexed line, character range).
#[derive(Debug, Clone, PartialEq)]
pub struct DiagnosticSpan {
    /// 0-indexed line number.
    pub line: u32,
    /// Start character (0 = whole line).
    pub col_start: u32,
    /// End character (u32::MAX = end of line).
    pub col_end: u32,
}

/// A single diagnostic message returned by the language server.
#[derive(Debug, Clone, PartialEq)]
pub struct LspDiagnostic {
    pub span:    DiagnosticSpan,
    pub message: String,
    pub severity: DiagnosticSeverity,
}

/// Severity level (mirrors LSP DiagnosticSeverity).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DiagnosticSeverity { Error, Warning }

/// Parse `source` and return LSP diagnostics (empty = no errors).
///
/// Handles all AqlError variants and converts 1-indexed source lines
/// to 0-indexed LSP positions.
pub fn diagnostics_for(source: &str) -> Vec<LspDiagnostic> {
    match compiler::parse_source(source) {
        Ok(_) => vec![],
        Err(e) => vec![aql_error_to_diagnostic(&e)],
    }
}

fn aql_error_to_diagnostic(err: &AqlError) -> LspDiagnostic {
    match err {
        AqlError::Lex { line, msg } => LspDiagnostic {
            span:     line_span(*line),
            message:  format!("Lex error: {msg}"),
            severity: DiagnosticSeverity::Error,
        },
        AqlError::Parse { line, msg } => LspDiagnostic {
            span:     line_span(*line),
            message:  format!("Parse error: {msg}"),
            severity: DiagnosticSeverity::Error,
        },
        AqlError::Validation { msg } => LspDiagnostic {
            span:     line_span(1),
            message:  format!("Validation error: {msg}"),
            severity: DiagnosticSeverity::Error,
        },
        AqlError::Runtime { msg } => LspDiagnostic {
            span:     line_span(1),
            message:  format!("Runtime error: {msg}"),
            severity: DiagnosticSeverity::Warning,
        },
    }
}

/// Convert a 1-indexed source line to a 0-indexed LSP span (whole line).
fn line_span(one_indexed_line: usize) -> DiagnosticSpan {
    let line = if one_indexed_line > 0 { (one_indexed_line - 1) as u32 } else { 0 };
    DiagnosticSpan { line, col_start: 0, col_end: u32::MAX }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diagnostics_clean_source() {
        let diags = diagnostics_for("QREG 2\nH 0\nCNOT 0 1\nMEASURE_ALL");
        assert!(diags.is_empty(), "valid AQL should produce no diagnostics");
    }

    #[test]
    fn test_diagnostics_parse_error() {
        let diags = diagnostics_for("QREG 2\nCNTO 0 1");
        assert_eq!(diags.len(), 1);
        assert_eq!(diags[0].severity, DiagnosticSeverity::Error);
        // Error is on line 2 (1-indexed) → span.line = 1 (0-indexed)
        assert_eq!(diags[0].span.line, 1);
        assert!(diags[0].message.contains("CNTO") || diags[0].message.contains("parse") ||
                diags[0].message.to_lowercase().contains("unknown"));
    }

    #[test]
    fn test_diagnostics_lex_error() {
        let diags = diagnostics_for("QREG 2\n@@@");
        assert_eq!(diags.len(), 1);
        assert_eq!(diags[0].severity, DiagnosticSeverity::Error);
    }
}
