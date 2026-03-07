/// Gate hover documentation for the AQL Language Server.

use std::collections::HashMap;

/// Returns a markdown documentation string for the given word (case-insensitive),
/// or `None` if the word is not a recognized AQL gate or keyword.
pub fn hover_for_word(word: &str) -> Option<String> {
    GATE_DOCS.get(word.to_ascii_uppercase().as_str()).map(|s| s.to_string())
}

static GATE_DOCS: std::sync::LazyLock<HashMap<&'static str, &'static str>> =
    std::sync::LazyLock::new(|| {
        let mut m = HashMap::new();
        // ── Single-qubit gates ──────────────────────────────────────────────
        m.insert("H",    "**Hadamard gate** — creates equal superposition.\n\n`H|0⟩ = (|0⟩+|1⟩)/√2`\n\nSyntax: `H <qubit>`");
        m.insert("X",    "**Pauli-X gate** (NOT gate) — bit flip.\n\n`X|0⟩ = |1⟩`, `X|1⟩ = |0⟩`\n\nSyntax: `X <qubit>`");
        m.insert("Y",    "**Pauli-Y gate** — bit+phase flip.\n\nSyntax: `Y <qubit>`");
        m.insert("Z",    "**Pauli-Z gate** — phase flip.\n\n`Z|1⟩ = -|1⟩`\n\nSyntax: `Z <qubit>`");
        m.insert("S",    "**S gate** (phase gate, T²) — π/2 phase on |1⟩.\n\nSyntax: `S <qubit>`");
        m.insert("T",    "**T gate** (π/8 gate) — π/4 phase on |1⟩.\n\nSyntax: `T <qubit>`");
        m.insert("RX",   "**Rx rotation** — rotate around X-axis by theta radians.\n\nSyntax: `RX <qubit> <theta>`");
        m.insert("RY",   "**Ry rotation** — rotate around Y-axis by theta radians.\n\nSyntax: `RY <qubit> <theta>`");
        m.insert("RZ",   "**Rz rotation** — rotate around Z-axis by theta radians.\n\nSyntax: `RZ <qubit> <theta>`");
        m.insert("PHASE","**Phase gate** — apply e^(i·theta) phase to |1⟩.\n\nSyntax: `PHASE <qubit> <theta>`");
        // ── Two-qubit gates ─────────────────────────────────────────────────
        m.insert("CNOT", "**Controlled-NOT** — flips target if control is |1⟩.\n\nSyntax: `CNOT <control> <target>`");
        m.insert("CX",   "**Controlled-X** (alias for CNOT).\n\nSyntax: `CX <control> <target>`");
        m.insert("CZ",   "**Controlled-Z** — applies Z to target if control is |1⟩.\n\nSyntax: `CZ <control> <target>`");
        m.insert("SWAP", "**SWAP gate** — swaps two qubit states.\n\nSyntax: `SWAP <qubit_a> <qubit_b>`");
        // ── Three-qubit gates ───────────────────────────────────────────────
        m.insert("CCX",     "**Toffoli gate** (CCX, CCNOT) — flips target if both controls are |1⟩.\n\nSyntax: `CCX <ctrl0> <ctrl1> <target>`");
        m.insert("TOFFOLI", "**Toffoli gate** — flips target if both controls are |1⟩.\n\nSyntax: `TOFFOLI <ctrl0> <ctrl1> <target>`");
        // ── Measurement ─────────────────────────────────────────────────────
        m.insert("MEASURE",     "**Measure** — collapses a single qubit to |0⟩ or |1⟩.\n\nSyntax: `MEASURE <qubit>`");
        m.insert("MEASURE_ALL", "**Measure all** — collapses all qubits.\n\nSyntax: `MEASURE_ALL`");
        // ── Control flow ────────────────────────────────────────────────────
        m.insert("LABEL",  "**Label** — defines a jump target for GOTO/IF.\n\nSyntax: `LABEL <name>`");
        m.insert("GOTO",   "**Goto** — unconditional jump to a label.\n\nSyntax: `GOTO <label>`");
        m.insert("IF",     "**If** — conditional jump if qubit was measured as |1⟩.\n\nSyntax: `IF <qubit> GOTO <label>`");
        m.insert("IFNOT",  "**IfNot** — conditional jump if qubit was measured as |0⟩.\n\nSyntax: `IFNOT <qubit> GOTO <label>`");
        m.insert("IFMEASURED",    "**IfMeasured** — structured conditional (sugar for IF/GOTO/LABEL).\n\nSyntax: `IFMEASURED <qubit> == 1 THEN\\n  ...\\nEND`");
        m.insert("IFNOTMEASURED", "**IfNotMeasured** — structured conditional (sugar for IFNOT/GOTO/LABEL).\n\nSyntax: `IFNOTMEASURED <qubit> == 1 THEN\\n  ...\\nEND`");
        // ── Structural ──────────────────────────────────────────────────────
        m.insert("QREG",    "**Quantum register declaration** — allocate N qubits.\n\nSyntax: `QREG <n>` or `QREG name[n]`");
        m.insert("CREG",    "**Classical register declaration** — allocate N classical bits.\n\nSyntax: `CREG name[n]`");
        m.insert("BARRIER", "**Barrier** — optimization boundary (no quantum effect).\n\nSyntax: `BARRIER`");
        m.insert("GATE",    "**Gate definition** — define a reusable gate.\n\nSyntax: `GATE <name> <n_qubits>\\n  ...body...\\nEND`");
        m.insert("CALL",    "**Call gate** — invoke a user-defined gate.\n\nSyntax: `CALL <name> <qubit0> <qubit1> ...`");
        m.insert("REPEAT",  "**Repeat** — compile-time loop unrolling.\n\nSyntax: `REPEAT <n>\\n  ...body...\\nEND`");
        m.insert("INCLUDE", "**Include** — import gate definitions from another file.\n\nSyntax: `INCLUDE \"path/to/file.aql\"`");
        // ── Oracle/Diffusion ────────────────────────────────────────────────
        m.insert("ORACLE",    "**Oracle block** — Grover phase-flip oracle.\n\nSyntax: `ORACLE\\n  PATTERN b0 b1 ...\\nEND`\n\nExpands at compile time to a multi-controlled Z gate circuit.");
        m.insert("DIFFUSION", "**Diffusion operator** — Grover diffusion (Hadamard + reflect).\n\nSyntax: `DIFFUSION <n_qubits>`\n\nExpands at compile time to H·X·CZ·X·H sequence.");
        // ── Constants ───────────────────────────────────────────────────────
        m.insert("PI",   "**π** — mathematical constant ≈ 3.14159265...\n\nUse in angle arguments: `RX 0 PI`");
        m.insert("PI_2", "**π/2** — mathematical constant ≈ 1.5707963...\n\nUse in angle arguments: `RY 0 PI_2`");
        m.insert("PI_4", "**π/4** — mathematical constant ≈ 0.7853981...");
        m.insert("TAU",  "**2π** — mathematical constant ≈ 6.2831853...");
        m
    });

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hover_h_gate() {
        let doc = hover_for_word("H");
        assert!(doc.is_some(), "H gate should have documentation");
        assert!(doc.unwrap().contains("Hadamard"));
    }

    #[test]
    fn test_hover_case_insensitive() {
        assert!(hover_for_word("h").is_some());
        assert!(hover_for_word("cnot").is_some());
        assert!(hover_for_word("CNOT").is_some());
    }

    #[test]
    fn test_hover_unknown_word() {
        assert!(hover_for_word("FOOBAR").is_none());
        assert!(hover_for_word("").is_none());
    }
}
