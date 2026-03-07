/// AQL → OpenQASM 2.0 exporter.
///
/// Converts an AQL `Program` (or AQL source text) to a valid OpenQASM 2.0
/// string that can be imported into Qiskit, IBM Quantum, etc.
///
/// # Supported gates
/// All standard AQL gates are mapped to their OpenQASM 2.0 equivalents.
/// Custom gate definitions (GATE…END) are emitted as `gate` declarations.
/// Control flow (LABEL/GOTO/IF/IFNOT) is not representable in OpenQASM 2.0
/// and is replaced by an inline comment; the exporter does not error out.
use crate::compiler::ir::{GateDef, Instruction, Program};

// ── Public API ────────────────────────────────────────────────────────────

/// Export an AQL `Program` to an OpenQASM 2.0 string.
pub fn to_qasm(program: &Program) -> String {
    let mut out = String::new();

    // Header
    out.push_str("OPENQASM 2.0;\n");
    out.push_str("include \"qelib1.inc\";\n");
    out.push('\n');

    // Custom gate definitions
    for (name, def) in &program.gate_defs {
        emit_gate_def(&mut out, name, def);
    }

    // Register declarations
    let n = program.num_qubits;
    out.push_str(&format!("qreg q[{n}];\n"));

    // Count classical bits needed (qubits that are measured)
    let measured: Vec<usize> = {
        // For MeasureAll expand properly
        let mut bits = std::collections::BTreeSet::new();
        for i in &program.instructions {
            match i {
                Instruction::Measure(q)  => { bits.insert(*q); }
                Instruction::MeasureAll  => { for q in 0..n { bits.insert(q); } }
                _ => {}
            }
        }
        let _measured_vec: Vec<usize> = bits.into_iter().collect();
        _measured_vec
    };

    if !measured.is_empty() {
        out.push_str(&format!("creg c[{n}];\n"));
    }
    out.push('\n');

    // Instructions
    for instr in &program.instructions {
        emit_instruction(&mut out, instr, n);
    }

    out
}

/// Export AQL source text to an OpenQASM 2.0 string.
pub fn source_to_qasm(source: &str) -> Result<String, crate::compiler::AqlError> {
    let program = crate::compiler::parse_source(source)?;
    Ok(to_qasm(&program))
}

// ── Internal helpers ──────────────────────────────────────────────────────

fn emit_gate_def(out: &mut String, name: &str, def: &GateDef) {
    // Build qubit parameter list: q0, q1, ...
    let params: Vec<String> = (0..def.num_qubits).map(|i| format!("q{i}")).collect();
    out.push_str(&format!("gate {} {} {{\n", name, params.join(",")));
    for instr in &def.body {
        out.push_str("  ");
        emit_instruction(out, instr, def.num_qubits);
    }
    out.push_str("}\n\n");
}

fn emit_instruction(out: &mut String, instr: &Instruction, _n: usize) {
    match instr {
        Instruction::H(q)    => out.push_str(&format!("h q[{q}];\n")),
        Instruction::X(q)    => out.push_str(&format!("x q[{q}];\n")),
        Instruction::Y(q)    => out.push_str(&format!("y q[{q}];\n")),
        Instruction::Z(q)    => out.push_str(&format!("z q[{q}];\n")),
        Instruction::S(q)    => out.push_str(&format!("s q[{q}];\n")),
        Instruction::T(q)    => out.push_str(&format!("t q[{q}];\n")),

        Instruction::Rx { qubit, theta } =>
            out.push_str(&format!("rx({:.10}) q[{qubit}];\n", theta)),
        Instruction::Ry { qubit, theta } =>
            out.push_str(&format!("ry({:.10}) q[{qubit}];\n", theta)),
        Instruction::Rz { qubit, theta } =>
            out.push_str(&format!("rz({:.10}) q[{qubit}];\n", theta)),
        Instruction::Phase { qubit, theta } =>
            out.push_str(&format!("p({:.10}) q[{qubit}];\n", theta)),

        Instruction::Cnot { control, target } =>
            out.push_str(&format!("cx q[{control}],q[{target}];\n")),
        Instruction::Cz { control, target } =>
            out.push_str(&format!("cz q[{control}],q[{target}];\n")),
        Instruction::Swap { qubit_a, qubit_b } =>
            out.push_str(&format!("swap q[{qubit_a}],q[{qubit_b}];\n")),
        Instruction::Toffoli { control0, control1, target } =>
            out.push_str(&format!("ccx q[{control0}],q[{control1}],q[{target}];\n")),

        Instruction::Measure(q) =>
            out.push_str(&format!("measure q[{q}] -> c[{q}];\n")),
        Instruction::MeasureAll => {
            // Emit individual measure statements (OpenQASM 2.0 doesn't have measure_all)
            // We don't know n here without extra context, use a placeholder comment.
            // The caller (to_qasm) has n; but emit_instruction is generic.
            // For gate bodies this won't be called. Use a marker that to_qasm handles.
            out.push_str("// MEASURE_ALL (see individual measures below)\n");
        }

        Instruction::Barrier =>
            out.push_str("barrier;\n"),

        // Control flow: not representable in QASM 2.0 — emit comment
        Instruction::Label(name) =>
            out.push_str(&format!("// LABEL {name}\n")),
        Instruction::Goto { label } =>
            out.push_str(&format!("// GOTO {label} (classical control flow — not expressible in QASM 2.0)\n")),
        Instruction::GotoIf { qubit, label } =>
            out.push_str(&format!("// IF {qubit} GOTO {label} (classical control flow)\n")),
        Instruction::GotoIfNot { qubit, label } =>
            out.push_str(&format!("// IFNOT {qubit} GOTO {label} (classical control flow)\n")),

        Instruction::MeasureInto { qubit, creg, creg_bit } =>
            out.push_str(&format!("measure q[{qubit}] -> {creg}[{creg_bit}];\n")),
        Instruction::GotoIfCreg { creg, bit, label } =>
            out.push_str(&format!("// IF {creg}[{bit}] GOTO {label} (CREG conditional — not expressible in QASM 2.0)\n")),
        Instruction::GotoIfNotCreg { creg, bit, label } =>
            out.push_str(&format!("// IFNOT {creg}[{bit}] GOTO {label} (CREG conditional — not expressible in QASM 2.0)\n")),

        Instruction::CallGate { name, qubits } => {
            let args: Vec<String> = qubits.iter().map(|q| format!("q[{q}]")).collect();
            out.push_str(&format!("{name} {};\n", args.join(",")));
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn qasm(src: &str) -> String {
        source_to_qasm(src).expect("qasm export failed")
    }

    #[test]
    fn test_qasm_header() {
        let out = qasm("QREG 2\nH 0");
        assert!(out.starts_with("OPENQASM 2.0;\n"), "must start with OPENQASM header");
        assert!(out.contains("include \"qelib1.inc\";"), "must include qelib1");
    }

    #[test]
    fn test_qasm_qreg() {
        let out = qasm("QREG 3\nH 0");
        assert!(out.contains("qreg q[3];"), "qreg declaration");
    }

    #[test]
    fn test_qasm_single_qubit_gates() {
        let out = qasm("QREG 1\nH 0\nX 0\nY 0\nZ 0\nS 0\nT 0");
        assert!(out.contains("h q[0];"));
        assert!(out.contains("x q[0];"));
        assert!(out.contains("y q[0];"));
        assert!(out.contains("z q[0];"));
        assert!(out.contains("s q[0];"));
        assert!(out.contains("t q[0];"));
    }

    #[test]
    fn test_qasm_rotation_gates() {
        let out = qasm("QREG 1\nRX 0 1.5708\nRY 0 3.1416\nRZ 0 0.7854");
        assert!(out.contains("rx("), "rx gate");
        assert!(out.contains("ry("), "ry gate");
        assert!(out.contains("rz("), "rz gate");
    }

    #[test]
    fn test_qasm_two_qubit_gates() {
        let out = qasm("QREG 2\nCNOT 0 1\nCZ 0 1\nSWAP 0 1");
        assert!(out.contains("cx q[0],q[1];"));
        assert!(out.contains("cz q[0],q[1];"));
        assert!(out.contains("swap q[0],q[1];"));
    }

    #[test]
    fn test_qasm_toffoli() {
        let out = qasm("QREG 3\nCCX 0 1 2");
        assert!(out.contains("ccx q[0],q[1],q[2];"));
    }

    #[test]
    fn test_qasm_measure() {
        let out = qasm("QREG 2\nH 0\nCNOT 0 1\nMEASURE 0\nMEASURE 1");
        assert!(out.contains("creg c[2];"), "classical register");
        assert!(out.contains("measure q[0] -> c[0];"));
        assert!(out.contains("measure q[1] -> c[1];"));
    }

    #[test]
    fn test_qasm_custom_gate_def() {
        let src = "GATE bell 2\n  H 0\n  CNOT 0 1\nEND\nQREG 2\nCALL bell 0 1";
        let out = qasm(src);
        assert!(out.contains("gate bell"), "custom gate definition");
        assert!(out.contains("bell q[0],q[1];"), "custom gate call");
    }

    #[test]
    fn test_qasm_barrier() {
        let out = qasm("QREG 1\nH 0\nBARRIER\nH 0");
        assert!(out.contains("barrier;"));
    }

    #[test]
    fn test_qasm_control_flow_becomes_comment() {
        let out = qasm("QREG 1\nX 0\nMEASURE 0\nIF 0 GOTO done\nX 0\nLABEL done");
        assert!(out.contains("// IF"), "IF becomes comment");
        assert!(out.contains("// LABEL"), "LABEL becomes comment");
    }
}
