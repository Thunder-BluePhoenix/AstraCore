/// AQL code completion items for the Language Server.

/// A simple completion item (label + detail).
#[derive(Debug, Clone)]
pub struct CompletionItem {
    /// The text inserted when selected.
    pub label: String,
    /// Short description shown next to the label.
    pub detail: String,
    /// Full insert text (may differ from label, e.g. includes snippet placeholders).
    pub insert_text: String,
}

impl CompletionItem {
    fn new(label: &str, detail: &str, insert_text: &str) -> Self {
        Self {
            label: label.to_string(),
            detail: detail.to_string(),
            insert_text: insert_text.to_string(),
        }
    }
}

/// Returns all AQL keyword and gate completion items, sorted alphabetically.
pub fn completions() -> Vec<CompletionItem> {
    let mut items = vec![
        // ── Directives ──────────────────────────────────────────────────────
        CompletionItem::new("BARRIER",        "Optimization boundary",              "BARRIER"),
        CompletionItem::new("CALL",           "Invoke user-defined gate",           "CALL "),
        CompletionItem::new("CREG",           "Classical register declaration",     "CREG ["),
        CompletionItem::new("DIFFUSION",      "Grover diffusion operator",          "DIFFUSION "),
        CompletionItem::new("END",            "Close GATE/REPEAT/ORACLE block",     "END"),
        CompletionItem::new("GATE",           "Define a reusable gate",             "GATE  \n\nEND"),
        CompletionItem::new("GOTO",           "Unconditional jump",                 "GOTO "),
        CompletionItem::new("IF",             "Conditional jump (if |1⟩)",          "IF  GOTO "),
        CompletionItem::new("IFMEASURED",     "Structured conditional (if |1⟩)",    "IFMEASURED  == 1 THEN\n\nEND"),
        CompletionItem::new("IFNOT",          "Conditional jump (if |0⟩)",          "IFNOT  GOTO "),
        CompletionItem::new("IFNOTMEASURED",  "Structured conditional (if |0⟩)",    "IFNOTMEASURED  == 1 THEN\n\nEND"),
        CompletionItem::new("INCLUDE",        "Include gate library file",          "INCLUDE \"\""),
        CompletionItem::new("LABEL",          "Define a jump target",               "LABEL "),
        CompletionItem::new("MEASURE",        "Measure a single qubit",             "MEASURE "),
        CompletionItem::new("MEASURE_ALL",    "Measure all qubits",                 "MEASURE_ALL"),
        CompletionItem::new("ORACLE",         "Grover phase-flip oracle block",     "ORACLE\n  PATTERN \nEND"),
        CompletionItem::new("PATTERN",        "Target bit-string inside ORACLE",    "PATTERN "),
        CompletionItem::new("QREG",           "Quantum register declaration",       "QREG "),
        CompletionItem::new("REPEAT",         "Compile-time loop unrolling",        "REPEAT \n\nEND"),
        // ── Single-qubit gates ───────────────────────────────────────────────
        CompletionItem::new("CCX",    "Toffoli gate",                  "CCX   "),
        CompletionItem::new("CNOT",   "Controlled-NOT gate",           "CNOT  "),
        CompletionItem::new("CX",     "Controlled-X (CNOT alias)",     "CX  "),
        CompletionItem::new("CZ",     "Controlled-Z gate",             "CZ  "),
        CompletionItem::new("H",      "Hadamard gate",                 "H "),
        CompletionItem::new("PHASE",  "Phase gate (angle in radians)", "PHASE  "),
        CompletionItem::new("RX",     "X-rotation gate",               "RX  "),
        CompletionItem::new("RY",     "Y-rotation gate",               "RY  "),
        CompletionItem::new("RZ",     "Z-rotation gate",               "RZ  "),
        CompletionItem::new("S",      "S gate (π/2 phase)",            "S "),
        CompletionItem::new("SWAP",   "SWAP two qubits",               "SWAP  "),
        CompletionItem::new("T",      "T gate (π/4 phase)",            "T "),
        CompletionItem::new("TOFFOLI","Toffoli gate",                   "TOFFOLI   "),
        CompletionItem::new("X",      "Pauli-X gate (NOT)",            "X "),
        CompletionItem::new("Y",      "Pauli-Y gate",                  "Y "),
        CompletionItem::new("Z",      "Pauli-Z gate",                  "Z "),
        // ── Constants ────────────────────────────────────────────────────────
        CompletionItem::new("PI",    "π ≈ 3.14159265", "PI"),
        CompletionItem::new("PI_2",  "π/2 ≈ 1.5707963", "PI_2"),
        CompletionItem::new("PI_4",  "π/4 ≈ 0.7853981", "PI_4"),
        CompletionItem::new("TAU",   "2π ≈ 6.2831853",  "TAU"),
    ];
    items.sort_by(|a, b| a.label.cmp(&b.label));
    items
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_completion_list_not_empty() {
        let items = completions();
        assert!(items.len() >= 20, "should have at least 20 completion items, got {}", items.len());
    }

    #[test]
    fn test_completion_includes_core_gates() {
        let items = completions();
        let labels: Vec<&str> = items.iter().map(|i| i.label.as_str()).collect();
        assert!(labels.contains(&"H"),    "H missing from completions");
        assert!(labels.contains(&"CNOT"), "CNOT missing");
        assert!(labels.contains(&"QREG"), "QREG missing");
        assert!(labels.contains(&"ORACLE"), "ORACLE missing");
    }
}
