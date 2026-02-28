/// AQL Circuit Analysis — static profiling of AQL programs.
///
/// Computes structural metrics **without** executing the circuit:
///
/// | Metric                 | Description                                              |
/// |------------------------|----------------------------------------------------------|
/// | `gate_count`           | Number of gate instructions (top-level, CALL counts as 1)|
/// | `circuit_depth`        | Critical-path length assuming unbounded parallelism       |
/// | `two_qubit_gate_count` | Multi-qubit gates — proxy for hardware entanglement cost  |
/// | `gate_histogram`       | Per-mnemonic gate counts                                  |
/// | `qubit_utilization`    | Number of gates touching each qubit                       |
///
/// Circuit depth for `CALL` instructions is computed by expanding the gate body
/// locally, so depth reflects the actual sequential structure of user-defined gates.
use std::collections::HashMap;
use crate::compiler::ir::{Instruction, Program};

// ── Public types ──────────────────────────────────────────────────────────

/// Static analysis result for an AQL program.
#[derive(Debug, Clone)]
pub struct CircuitAnalysis {
    /// Number of qubits declared by `QREG`.
    pub num_qubits: usize,

    /// Number of top-level gate operations.
    /// `CALL` counts as 1 regardless of body length; see `expanded_gate_count`.
    pub gate_count: usize,

    /// Number of top-level gate operations after expanding all `CALL` bodies.
    /// Equals `gate_count` for programs without custom gate calls.
    pub expanded_gate_count: usize,

    /// Number of measurement operations.
    pub measure_count: usize,

    /// Critical-path depth: the minimum number of sequential time steps
    /// required to execute the circuit assuming unlimited qubit parallelism.
    ///
    /// `CALL` bodies are expanded to give an accurate depth.
    pub circuit_depth: usize,

    /// Number of multi-qubit gate operations (CNOT, CZ, SWAP, Toffoli, CALL).
    /// A proxy for the hardware entanglement cost of the circuit.
    pub two_qubit_gate_count: usize,

    /// Count of each gate mnemonic (top-level instructions only).
    pub gate_histogram: HashMap<String, usize>,

    /// Number of times each qubit is referenced by a gate instruction.
    /// Index = qubit index; does **not** count measurements.
    pub qubit_utilization: Vec<usize>,

    /// True if the program contains `LABEL`/`GOTO`/`IF`/`IFNOT`.
    pub has_control_flow: bool,

    /// True if the program declares any `GATE` definitions.
    pub has_custom_gates: bool,

    /// Number of distinct user-defined gate types declared.
    pub custom_gate_defs: usize,
}

impl CircuitAnalysis {
    /// Average number of gates per qubit (top-level `gate_count`).
    pub fn avg_gates_per_qubit(&self) -> f64 {
        if self.num_qubits == 0 { return 0.0; }
        self.gate_count as f64 / self.num_qubits as f64
    }

    /// Fraction of top-level gates that are multi-qubit (entangling).
    pub fn entanglement_ratio(&self) -> f64 {
        if self.gate_count == 0 { return 0.0; }
        self.two_qubit_gate_count as f64 / self.gate_count as f64
    }

    /// Human-readable profiling report.
    pub fn report(&self) -> String {
        let mut out = String::new();

        out.push_str(&format!("  Qubits         : {}\n", self.num_qubits));
        out.push_str(&format!("  Gate count     : {}", self.gate_count));
        if self.expanded_gate_count != self.gate_count {
            out.push_str(&format!("  (expanded: {})", self.expanded_gate_count));
        }
        out.push('\n');
        out.push_str(&format!("  Circuit depth  : {}\n", self.circuit_depth));
        out.push_str(&format!(
            "  2-qubit gates  : {}  ({:.1}% entangling)\n",
            self.two_qubit_gate_count,
            self.entanglement_ratio() * 100.0
        ));
        out.push_str(&format!("  Measurements   : {}\n", self.measure_count));
        out.push_str(&format!("  Control flow   : {}\n",
            if self.has_control_flow { "yes" } else { "no" }));
        if self.has_custom_gates {
            out.push_str(&format!(
                "  Custom gates   : {} definition(s) declared\n",
                self.custom_gate_defs
            ));
        }

        out.push_str("\n  Gate breakdown:\n");
        let mut hist: Vec<(&String, &usize)> = self.gate_histogram.iter().collect();
        hist.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0))); // sort by count desc, then name
        for (mnemonic, count) in hist {
            out.push_str(&format!("    {:8}  {count}\n", mnemonic));
        }

        out.push_str("\n  Qubit utilization (gate touches per qubit):\n");
        let max_uses = self.qubit_utilization.iter().copied().max().unwrap_or(1).max(1);
        for (q, &uses) in self.qubit_utilization.iter().enumerate() {
            let bar_len = (uses * 20 / max_uses).min(20);
            let bar = "█".repeat(bar_len);
            let pad = " ".repeat(20 - bar_len);
            out.push_str(&format!("    q{q:<2}  {bar}{pad}  {uses}\n"));
        }

        out
    }
}

// ── Public API ────────────────────────────────────────────────────────────

/// Analyze an AQL `Program`, returning a rich set of structural metrics.
///
/// Runs in O(instructions + expanded_gates) time. Does not execute the circuit.
///
/// # Example
/// ```rust
/// use astracore::compiler::{parse_source, analysis::analyze};
///
/// // H(0) then CNOT(0,1): sequential on q0 → depth 2
/// let prog = parse_source("QREG 2\nH 0\nCNOT 0 1").unwrap();
/// let a = analyze(&prog);
/// assert_eq!(a.circuit_depth, 2);
/// assert_eq!(a.two_qubit_gate_count, 1);
/// ```
pub fn analyze(program: &Program) -> CircuitAnalysis {
    // `qubit_time[q]` = depth of the "frontier" on qubit `q`.
    let mut qubit_time = vec![0usize; program.num_qubits];
    let mut gate_count = 0usize;
    let mut expanded_gate_count = 0usize;
    let mut measure_count = 0usize;
    let mut two_qubit_gate_count = 0usize;
    let mut gate_histogram: HashMap<String, usize> = HashMap::new();
    let mut qubit_utilization = vec![0usize; program.num_qubits];
    let mut has_control_flow = false;

    for instr in &program.instructions {
        // ── Control flow ──────────────────────────────────────────────────
        if instr.is_control_flow() {
            has_control_flow = true;
            continue;
        }

        // ── Measurements ──────────────────────────────────────────────────
        if instr.is_measurement() {
            measure_count += 1;
            let qs = instr.qubits();
            if qs.is_empty() {
                // MEASURE_ALL — advances all qubits to a common frontier
                let max_t = qubit_time.iter().copied().max().unwrap_or(0);
                for t in &mut qubit_time { *t = max_t + 1; }
            } else {
                let start = qs.iter().map(|&q| qubit_time[q]).max().unwrap_or(0);
                for &q in &qs { qubit_time[q] = start + 1; }
            }
            continue;
        }

        // ── Non-gate structural instructions (BARRIER, LABEL, GOTO) ───────
        if !instr.is_gate() {
            continue;
        }

        // ── Gate instruction ───────────────────────────────────────────────
        gate_count += 1;
        *gate_histogram.entry(instr.mnemonic().to_string()).or_insert(0) += 1;

        let q_list = instr.qubits();

        // Per-qubit utilization
        for &q in &q_list {
            qubit_utilization[q] += 1;
        }

        // Multi-qubit gate?
        if q_list.len() >= 2 {
            two_qubit_gate_count += 1;
        }

        // ── Depth update ───────────────────────────────────────────────────
        if let Instruction::CallGate { name, qubits: arg_qubits } = instr {
            // Expand the gate body to get an accurate depth contribution.
            let call_depth = program.gate_defs
                .get(name.as_str())
                .map(|def| {
                    // Count expanded gates
                    let body_gates = def.body.iter().filter(|i| i.is_gate()).count();
                    expanded_gate_count += body_gates;
                    body_depth(&def.body)
                })
                .unwrap_or(1); // unknown gate body → conservative depth of 1

            let start = arg_qubits.iter().map(|&q| qubit_time[q]).max().unwrap_or(0);
            for &q in arg_qubits { qubit_time[q] = start + call_depth; }
        } else {
            // Standard gate: depth contribution = 1
            expanded_gate_count += 1;
            if !q_list.is_empty() {
                let start = q_list.iter().map(|&q| qubit_time[q]).max().unwrap_or(0);
                let end = start + 1;
                for &q in &q_list { qubit_time[q] = end; }
            }
        }
    }

    let circuit_depth = qubit_time.iter().copied().max().unwrap_or(0);

    CircuitAnalysis {
        num_qubits: program.num_qubits,
        gate_count,
        expanded_gate_count,
        measure_count,
        circuit_depth,
        two_qubit_gate_count,
        gate_histogram,
        qubit_utilization,
        has_control_flow,
        has_custom_gates: !program.gate_defs.is_empty(),
        custom_gate_defs: program.gate_defs.len(),
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────

/// Compute the critical-path depth of a gate body using local qubit indices.
///
/// Only counts `is_gate()` instructions; barriers and no-ops are ignored.
fn body_depth(body: &[Instruction]) -> usize {
    // Find the number of local qubits referenced in the body.
    let max_q = body.iter()
        .flat_map(|i| i.qubits())
        .max()
        .map(|q| q + 1)
        .unwrap_or(0);

    if max_q == 0 { return 0; }

    let mut local_time = vec![0usize; max_q];
    for instr in body {
        if !instr.is_gate() { continue; }
        let qs = instr.qubits();
        if qs.is_empty() { continue; }
        let start = qs.iter().map(|&q| local_time[q]).max().unwrap_or(0);
        let end = start + 1;
        for &q in &qs { local_time[q] = end; }
    }
    local_time.iter().copied().max().unwrap_or(0)
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::parse_source;

    fn prog(src: &str) -> Program {
        parse_source(src).expect("parse failed")
    }

    // ── Depth computation ─────────────────────────────────────────────

    #[test]
    fn test_depth_empty_circuit() {
        let a = analyze(&prog("QREG 1"));
        assert_eq!(a.circuit_depth, 0);
    }

    #[test]
    fn test_depth_single_gate() {
        let a = analyze(&prog("QREG 1\nH 0"));
        assert_eq!(a.circuit_depth, 1);
    }

    #[test]
    fn test_depth_sequential_on_same_qubit() {
        // H then X on same qubit: depth = 2
        let a = analyze(&prog("QREG 1\nH 0\nX 0"));
        assert_eq!(a.circuit_depth, 2);
    }

    #[test]
    fn test_depth_parallel_on_different_qubits() {
        // H(0) and H(1) are independent → depth = 1
        let a = analyze(&prog("QREG 2\nH 0\nH 1"));
        assert_eq!(a.circuit_depth, 1);
    }

    #[test]
    fn test_depth_bell_circuit() {
        // H(0) → CNOT(0,1): H finishes at t=1 for q0; CNOT starts at max(1,0)=1 → ends t=2
        let a = analyze(&prog("QREG 2\nH 0\nCNOT 0 1"));
        assert_eq!(a.circuit_depth, 2);
    }

    #[test]
    fn test_depth_cnot_after_two_h_gates() {
        // H(0) and H(1) in parallel (both at t=1), then CNOT(0,1) at t=2
        let a = analyze(&prog("QREG 2\nH 0\nH 1\nCNOT 0 1"));
        assert_eq!(a.circuit_depth, 2);
    }

    #[test]
    fn test_depth_ghz_three_qubits() {
        // H(0) at t=1; CNOT(0,1) at t=2; CNOT(0,2) at t=3
        let a = analyze(&prog("QREG 3\nH 0\nCNOT 0 1\nCNOT 0 2"));
        assert_eq!(a.circuit_depth, 3);
    }

    #[test]
    fn test_depth_call_gate_expands_body() {
        // GATE bell 2: H(0) → CNOT(0,1) → body_depth = 2
        // CALL bell 0 1 contributes depth 2
        let a = analyze(&prog(
            "GATE bell 2\n  H 0\n  CNOT 0 1\nEND\nQREG 2\nCALL bell 0 1"
        ));
        assert_eq!(a.circuit_depth, 2, "CALL should expand body for accurate depth");
    }

    #[test]
    fn test_depth_barrier_is_ignored() {
        // BARRIER is structural, not a gate — depth unchanged
        let a = analyze(&prog("QREG 1\nH 0\nBARRIER\nX 0"));
        assert_eq!(a.circuit_depth, 2);
    }

    // ── Gate count & histogram ────────────────────────────────────────

    #[test]
    fn test_gate_count_basic() {
        let a = analyze(&prog("QREG 2\nH 0\nCNOT 0 1\nMEASURE_ALL"));
        assert_eq!(a.gate_count, 2);
        assert_eq!(a.measure_count, 1);
    }

    #[test]
    fn test_gate_histogram() {
        let a = analyze(&prog("QREG 2\nH 0\nH 1\nCNOT 0 1"));
        assert_eq!(a.gate_histogram.get("H"), Some(&2));
        assert_eq!(a.gate_histogram.get("CNOT"), Some(&1));
        assert_eq!(a.gate_histogram.len(), 2);
    }

    #[test]
    fn test_expanded_gate_count_call() {
        // CALL bell 0 1 expands to H + CNOT = 2 body gates
        let a = analyze(&prog(
            "GATE bell 2\n  H 0\n  CNOT 0 1\nEND\nQREG 2\nCALL bell 0 1"
        ));
        assert_eq!(a.gate_count, 1, "top-level: CALL counts as 1");
        assert_eq!(a.expanded_gate_count, 2, "expanded: body has 2 gates");
    }

    // ── Two-qubit gate count ───────────────────────────────────────────

    #[test]
    fn test_two_qubit_gate_count() {
        let a = analyze(&prog("QREG 3\nH 0\nCNOT 0 1\nCZ 1 2\nSWAP 0 2"));
        assert_eq!(a.two_qubit_gate_count, 3); // CNOT, CZ, SWAP
        assert_eq!(a.gate_count, 4);           // H + 3 two-qubit
    }

    #[test]
    fn test_entanglement_ratio() {
        let a = analyze(&prog("QREG 2\nH 0\nCNOT 0 1"));
        // 1 of 2 gates is two-qubit → 50%
        assert!((a.entanglement_ratio() - 0.5).abs() < 1e-9);
    }

    // ── Qubit utilization ─────────────────────────────────────────────

    #[test]
    fn test_qubit_utilization() {
        let a = analyze(&prog("QREG 3\nH 0\nCNOT 0 1"));
        // H touches q0 → 1; CNOT touches q0, q1 → q0=2, q1=1, q2=0
        assert_eq!(a.qubit_utilization[0], 2);
        assert_eq!(a.qubit_utilization[1], 1);
        assert_eq!(a.qubit_utilization[2], 0);
    }

    // ── Flags ─────────────────────────────────────────────────────────

    #[test]
    fn test_has_control_flow_flag() {
        let without = analyze(&prog("QREG 1\nH 0"));
        let with_cf = analyze(&prog("QREG 1\nH 0\nMEASURE 0\nIF 0 GOTO done\nLABEL done"));
        assert!(!without.has_control_flow);
        assert!(with_cf.has_control_flow);
    }

    #[test]
    fn test_has_custom_gates_flag() {
        let without = analyze(&prog("QREG 1\nH 0"));
        let with_gate = analyze(&prog("GATE flip 1\n  X 0\nEND\nQREG 1\nCALL flip 0"));
        assert!(!without.has_custom_gates);
        assert!(with_gate.has_custom_gates);
        assert_eq!(with_gate.custom_gate_defs, 1);
    }

    // ── avg_gates_per_qubit ───────────────────────────────────────────

    #[test]
    fn test_avg_gates_per_qubit() {
        let a = analyze(&prog("QREG 4\nH 0\nH 1\nH 2\nH 3"));
        assert!((a.avg_gates_per_qubit() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_avg_gates_per_qubit_zero_qubits() {
        let dummy = CircuitAnalysis {
            num_qubits: 0, gate_count: 0, expanded_gate_count: 0,
            measure_count: 0, circuit_depth: 0, two_qubit_gate_count: 0,
            gate_histogram: HashMap::new(), qubit_utilization: vec![],
            has_control_flow: false, has_custom_gates: false, custom_gate_defs: 0,
        };
        assert_eq!(dummy.avg_gates_per_qubit(), 0.0);
    }

    // ── body_depth helper ─────────────────────────────────────────────

    #[test]
    fn test_body_depth_empty() {
        assert_eq!(body_depth(&[]), 0);
    }

    #[test]
    fn test_body_depth_sequential() {
        let body = vec![
            Instruction::H(0),
            Instruction::X(0),
        ];
        assert_eq!(body_depth(&body), 2);
    }

    #[test]
    fn test_body_depth_parallel() {
        let body = vec![
            Instruction::H(0),
            Instruction::H(1),
        ];
        assert_eq!(body_depth(&body), 1);
    }

    #[test]
    fn test_body_depth_bell() {
        let body = vec![
            Instruction::H(0),
            Instruction::Cnot { control: 0, target: 1 },
        ];
        assert_eq!(body_depth(&body), 2);
    }
}
