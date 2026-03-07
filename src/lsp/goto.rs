/// Go-to-definition support for AQL — finds LABEL and GATE definitions.

/// A source position (0-indexed line and column).
#[derive(Debug, Clone, PartialEq)]
pub struct SourcePos {
    pub line: u32,
    pub col:  u32,
}

/// Find the 0-indexed line where `word` is defined as a `LABEL` or `GATE`.
///
/// Searches for:
/// - `LABEL <word>` (any case)
/// - `GATE <word> ...` (any case)
///
/// Returns `None` if no definition is found.
pub fn find_definition(source: &str, word: &str) -> Option<SourcePos> {
    let needle = word.to_ascii_uppercase();
    for (line_idx, line) in source.lines().enumerate() {
        let upper = line.trim().to_ascii_uppercase();
        // LABEL <name>
        if let Some(rest) = upper.strip_prefix("LABEL") {
            let name = rest.trim();
            if name == needle || name.starts_with(&(needle.clone() + " ")) {
                return Some(SourcePos { line: line_idx as u32, col: 0 });
            }
        }
        // GATE <name> <n_qubits>
        if let Some(rest) = upper.strip_prefix("GATE") {
            let mut parts = rest.split_ascii_whitespace();
            if let Some(gate_name) = parts.next() {
                if gate_name == needle {
                    return Some(SourcePos { line: line_idx as u32, col: 0 });
                }
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_goto_label_definition() {
        let src = "QREG 2\nH 0\nLABEL done\nMEASURE_ALL";
        let pos = find_definition(src, "done");
        assert!(pos.is_some(), "should find LABEL done");
        assert_eq!(pos.unwrap().line, 2); // 0-indexed: line 2
    }

    #[test]
    fn test_goto_gate_definition() {
        let src = "GATE my_gate 2\n  H 0\n  CNOT 0 1\nEND\nQREG 2\nCALL my_gate 0 1";
        let pos = find_definition(src, "my_gate");
        assert!(pos.is_some(), "should find GATE my_gate");
        assert_eq!(pos.unwrap().line, 0); // 0-indexed: line 0
    }

    #[test]
    fn test_goto_not_found() {
        let src = "QREG 2\nH 0\nMEASURE_ALL";
        assert!(find_definition(src, "nonexistent").is_none());
    }
}
