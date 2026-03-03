/// AQL Rich Diagnostic Engine.
///
/// Wraps an `AqlError` with the original source text to produce
/// human-friendly error output with:
///   - Source context: the offending line highlighted with a caret
///   - "Did you mean?" suggestions for unknown mnemonics (Levenshtein ≤ 2)
///
/// # Example
/// ```
/// use astracore::compiler::{AqlError, error::Diagnostic};
///
/// let src = "QREG 2\nCNTO 0 1";
/// let err = AqlError::Parse { line: 2, msg: "unknown mnemonic 'CNTO'".into() };
/// let diag = Diagnostic::new(err, src);
/// println!("{}", diag.display());
/// // ✗ Parse error at line 2: unknown mnemonic 'CNTO'
/// //   2 │ CNTO 0 1
/// //     │ ^^^^
/// //   did you mean: CNOT?
/// ```
use super::AqlError;

// ── All known AQL mnemonics for "did you mean?" suggestions ───────────────

static KNOWN_MNEMONICS: &[&str] = &[
    "QREG", "H", "X", "Y", "Z", "S", "T",
    "RX", "RY", "RZ", "PHASE",
    "CNOT", "CX", "CZ", "SWAP", "CCX", "TOFFOLI",
    "MEASURE", "MEASURE_ALL",
    "BARRIER",
    "LABEL", "GOTO", "IF", "IFNOT",
    "GATE", "END", "CALL",
    "REPEAT", "INCLUDE",
    "IFMEASURED", "IFNOTMEASURED", "THEN",
];

// ── Public API ────────────────────────────────────────────────────────────

/// An `AqlError` annotated with the source text for rich output.
pub struct Diagnostic {
    pub error: AqlError,
    lines: Vec<String>,
}

impl Diagnostic {
    /// Wrap an error with the original source text.
    pub fn new(error: AqlError, source: &str) -> Self {
        Self {
            error,
            lines: source.lines().map(|l| l.to_owned()).collect(),
        }
    }

    /// Render a rich, multi-line diagnostic string.
    pub fn display(&self) -> String {
        let mut out = String::new();

        match &self.error {
            AqlError::Lex { line, msg } => {
                out.push_str(&format!("✗ Lex error at line {line}: {msg}\n"));
                self.push_context(&mut out, *line);
            }
            AqlError::Parse { line, msg } => {
                out.push_str(&format!("✗ Parse error at line {line}: {msg}\n"));
                self.push_context(&mut out, *line);
                self.push_suggestion(&mut out, msg);
            }
            AqlError::Validation { msg } => {
                out.push_str(&format!("✗ Validation error: {msg}\n"));
            }
            AqlError::Runtime { msg } => {
                out.push_str(&format!("✗ Runtime error: {msg}\n"));
            }
        }

        out
    }

    fn push_context(&self, out: &mut String, line: usize) {
        if line == 0 || line > self.lines.len() {
            return;
        }
        let text = &self.lines[line - 1];
        let prefix = format!("  {line} │ ");
        out.push_str(&prefix);
        out.push_str(text);
        out.push('\n');
        // Caret under the first non-whitespace token
        let indent = " ".repeat(prefix.len());
        let token_len = text.trim_start()
            .split_whitespace()
            .next()
            .map_or(1, |w| w.len());
        out.push_str(&indent);
        out.push_str(&"^".repeat(token_len.max(1)));
        out.push('\n');
    }

    fn push_suggestion(&self, out: &mut String, msg: &str) {
        // Extract the unknown word from messages like:
        //   "unknown mnemonic 'CNTO'"
        //   "unknown instruction 'FOO'"
        let word = extract_quoted(msg);
        if let Some(w) = word {
            if let Some(suggestion) = closest_mnemonic(w) {
                out.push_str(&format!("  → did you mean: {suggestion}?\n"));
            }
        }
    }
}

/// Convenience: produce a rich diagnostic string directly from error + source.
pub fn diagnose(error: &AqlError, source: &str) -> String {
    Diagnostic::new(error.clone(), source).display()
}

// ── Internal helpers ──────────────────────────────────────────────────────

/// Extract the first single-quoted word from a message string.
fn extract_quoted(msg: &str) -> Option<&str> {
    let start = msg.find('\'')?;
    let rest = &msg[start + 1..];
    let end = rest.find('\'')?;
    Some(&rest[..end])
}

/// Find the mnemonic with minimum Levenshtein distance ≤ 2 to `name`.
fn closest_mnemonic(name: &str) -> Option<&'static str> {
    let upper = name.to_ascii_uppercase();
    KNOWN_MNEMONICS
        .iter()
        .filter_map(|&m| {
            let d = levenshtein(&upper, m);
            if d <= 2 { Some((d, m)) } else { None }
        })
        .min_by_key(|&(d, _)| d)
        .map(|(_, m)| m)
}

/// Standard Levenshtein edit distance (small strings only).
fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let m = a.len();
    let n = b.len();

    // Early exit for short strings
    if m == 0 { return n; }
    if n == 0 { return m; }

    let mut row: Vec<usize> = (0..=n).collect();
    for i in 1..=m {
        let mut prev = row[0];
        row[0] = i;
        for j in 1..=n {
            let old = row[j];
            row[j] = if a[i - 1] == b[j - 1] {
                prev
            } else {
                1 + prev.min(row[j]).min(row[j - 1])
            };
            prev = old;
        }
    }
    row[n]
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_levenshtein_identical() {
        assert_eq!(levenshtein("CNOT", "CNOT"), 0);
    }

    #[test]
    fn test_levenshtein_one_substitution() {
        assert_eq!(levenshtein("CNTO", "CNOT"), 2); // transposition = 2 edits
    }

    #[test]
    fn test_levenshtein_empty() {
        assert_eq!(levenshtein("", "CNOT"), 4);
        assert_eq!(levenshtein("CNOT", ""), 4);
    }

    #[test]
    fn test_closest_mnemonic_cnto() {
        // CNTO is 2 edits from CNOT (swap O↔T)
        let m = closest_mnemonic("CNTO");
        assert_eq!(m, Some("CNOT"));
    }

    #[test]
    fn test_closest_mnemonic_h_gate() {
        assert_eq!(closest_mnemonic("H"), Some("H"));
    }

    #[test]
    fn test_closest_mnemonic_no_match() {
        // 'ZZZZZ' is far from all known mnemonics
        let m = closest_mnemonic("ZZZZZ");
        assert!(m.is_none());
    }

    #[test]
    fn test_diagnostic_parse_error_output() {
        let src = "QREG 2\nCNTO 0 1";
        let err = AqlError::Parse {
            line: 2,
            msg: "unknown mnemonic 'CNTO' — did you mean a gate?".into(),
        };
        let out = Diagnostic::new(err, src).display();
        assert!(out.contains("Parse error at line 2"), "header missing");
        assert!(out.contains("CNTO 0 1"), "source context missing");
        assert!(out.contains("^"), "caret missing");
        assert!(out.contains("CNOT"), "suggestion missing");
    }

    #[test]
    fn test_diagnostic_lex_error_output() {
        let src = "QREG 2\n@bad";
        let err = AqlError::Lex { line: 2, msg: "invalid token '@bad'".into() };
        let out = Diagnostic::new(err, src).display();
        assert!(out.contains("Lex error at line 2"));
        assert!(out.contains("@bad"));
    }

    #[test]
    fn test_diagnostic_validation_no_context() {
        let err = AqlError::Validation { msg: "qubit 5 out of range".into() };
        let out = Diagnostic::new(err, "QREG 2\nH 5").display();
        assert!(out.contains("Validation error"));
        assert!(out.contains("qubit 5 out of range"));
    }

    #[test]
    fn test_diagnose_convenience() {
        let src = "QREG 1\nFLIB 0";
        let err = AqlError::Parse { line: 2, msg: "unknown mnemonic 'FLIB'".into() };
        let out = diagnose(&err, src);
        assert!(out.contains("FLIB 0"));
    }
}
