/// AstraCore Hybrid Executor — Phase 4
///
/// Executes an AQL Program using a program counter (PC) and a classical
/// register file. This enables full hybrid classical-quantum computation:
///
///   - Gates and measurements execute as in the simple executor
///   - LABEL defines named jump targets (no-op at runtime)
///   - GOTO sets PC unconditionally
///   - IF  <q> GOTO <label>   — jump if qubit q last measured as |1⟩
///   - IFNOT <q> GOTO <label> — jump if qubit q last measured as |0⟩
///
/// Classical register file:
///   One slot per qubit. Set by MEASURE. Read by IF/IFNOT.
///   Reading an unmeasured qubit causes a runtime error.
///
/// Infinite-loop guard:
///   Execution is capped at `MAX_STEPS` total instructions.
///   Programs exceeding this return `AqlError::Runtime`.
use std::collections::HashMap;
use crate::compiler::{AqlError, ir::{Instruction, Program}};
use crate::core::Simulator;

/// Hard upper bound on total instructions executed.
/// Prevents unbounded loops while allowing large circuits.
const MAX_STEPS: usize = 1_000_000;

// ── Result types ──────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct MeasurementRecord {
    pub qubit:   usize,
    pub outcome: bool,
    /// PC value (instruction index) at which this measurement occurred.
    pub step:    usize,
}

#[derive(Debug)]
pub struct ExecutionResult {
    pub num_qubits: usize,
    /// All measurement outcomes in execution order.
    pub measurements: Vec<MeasurementRecord>,
    /// Probability snapshot taken before the first measurement (if any).
    pub pre_measurement_probs: Option<Vec<f64>>,
    /// Final probability distribution (post-collapse if measured).
    pub final_probabilities: Vec<f64>,
    /// Total gate operations applied.
    pub gate_count: usize,
    /// Number of conditional branches taken (GotoIf/GotoIfNot that jumped).
    pub branch_count: usize,
    /// Total instruction steps executed (accounts for loops/branches).
    pub steps_executed: usize,
}

impl ExecutionResult {
    /// Last measured outcome of qubit `q`. Returns `None` if not yet measured.
    pub fn outcome(&self, qubit: usize) -> Option<bool> {
        self.measurements.iter().rfind(|m| m.qubit == qubit).map(|m| m.outcome)
    }

    /// True if every qubit has at least one measurement record.
    pub fn fully_collapsed(&self) -> bool {
        (0..self.num_qubits).all(|q| self.outcome(q).is_some())
    }

    /// Measurement outcomes as a bit-string, qubit 0 first.
    /// Returns `None` if not all qubits were measured.
    pub fn bitstring(&self) -> Option<String> {
        if !self.fully_collapsed() { return None; }
        Some(
            (0..self.num_qubits)
                .map(|q| if self.outcome(q).unwrap() { '1' } else { '0' })
                .collect()
        )
    }

    /// Basis states with probability above `threshold`.
    /// Returns `(binary_label, probability)` pairs sorted by index.
    pub fn significant_states(&self, probs: &[f64], threshold: f64) -> Vec<(String, f64)> {
        let n = self.num_qubits;
        probs.iter().enumerate()
            .filter(|(_, &p)| p > threshold)
            .map(|(i, &p)| (format!("{i:0>n$b}"), p))
            .collect()
    }
}

// ── Execution ─────────────────────────────────────────────────────────────

/// Build the label table: label name → index of first instruction after the label.
fn build_label_table(instructions: &[Instruction]) -> HashMap<String, usize> {
    let mut table = HashMap::new();
    for (i, instr) in instructions.iter().enumerate() {
        if let Instruction::Label(name) = instr {
            // Jump target = instruction immediately after LABEL
            table.insert(name.clone(), i + 1);
        }
    }
    table
}

/// Execute an AQL Program on a fresh Simulator instance.
///
/// Handles both simple circuits (no control flow) and hybrid programs
/// with LABEL/GOTO/IF/IFNOT instructions.
pub fn execute(program: &Program) -> Result<ExecutionResult, AqlError> {
    let label_table = build_label_table(&program.instructions);

    let mut sim       = Simulator::new(program.num_qubits);
    let mut classical = vec![None::<bool>; program.num_qubits]; // per-qubit register file
    let mut measurements: Vec<MeasurementRecord> = Vec::new();
    let mut pre_measurement_probs: Option<Vec<f64>> = None;
    let mut gate_count    = 0usize;
    let mut branch_count  = 0usize;
    let mut steps_executed = 0usize;
    let mut first_measure = true;

    let mut pc: usize = 0;

    while pc < program.instructions.len() {
        if steps_executed >= MAX_STEPS {
            return Err(AqlError::Runtime {
                msg: format!(
                    "execution exceeded {MAX_STEPS} steps — possible infinite loop in program"
                ),
            });
        }
        steps_executed += 1;

        let instr = &program.instructions[pc];

        // Snapshot state before first measurement
        if instr.is_measurement() && first_measure {
            pre_measurement_probs = Some(sim.probabilities());
            first_measure = false;
        }

        match instr {
            // ── Single-qubit gates ──────────────────────────────────────
            Instruction::H(q)                  => { sim.h(*q);                gate_count += 1; }
            Instruction::X(q)                  => { sim.x(*q);                gate_count += 1; }
            Instruction::Y(q)                  => { sim.y(*q);                gate_count += 1; }
            Instruction::Z(q)                  => { sim.z(*q);                gate_count += 1; }
            Instruction::S(q)                  => { sim.s(*q);                gate_count += 1; }
            Instruction::T(q)                  => { sim.t(*q);                gate_count += 1; }
            Instruction::Rx { qubit, theta }   => { sim.rx(*qubit, *theta);   gate_count += 1; }
            Instruction::Ry { qubit, theta }   => { sim.ry(*qubit, *theta);   gate_count += 1; }
            Instruction::Rz { qubit, theta }   => { sim.rz(*qubit, *theta);   gate_count += 1; }
            Instruction::Phase { qubit, theta }=> { sim.phase(*qubit, *theta); gate_count += 1; }

            // ── Multi-qubit gates ───────────────────────────────────────
            Instruction::Cnot { control, target } => { sim.cnot(*control, *target); gate_count += 1; }
            Instruction::Cz   { control, target } => { sim.cz(*control, *target);   gate_count += 1; }
            Instruction::Swap { qubit_a, qubit_b }=> { sim.swap(*qubit_a, *qubit_b); gate_count += 1; }
            Instruction::Toffoli { control0, control1, target } => {
                sim.toffoli(*control0, *control1, *target);
                gate_count += 1;
            }

            // ── Measurement ─────────────────────────────────────────────
            Instruction::Measure(q) => {
                let outcome = sim.measure(*q);
                classical[*q] = Some(outcome);
                measurements.push(MeasurementRecord { qubit: *q, outcome, step: pc });
            }
            Instruction::MeasureAll => {
                for q in 0..program.num_qubits {
                    let outcome = sim.measure(q);
                    classical[q] = Some(outcome);
                    measurements.push(MeasurementRecord { qubit: q, outcome, step: pc });
                }
            }

            // ── Structural ──────────────────────────────────────────────
            Instruction::Barrier => {}
            Instruction::Label(_) => {} // no-op; jump targets indexed by label_table

            // ── Control flow ────────────────────────────────────────────
            Instruction::Goto { label } => {
                let &target = label_table.get(label.as_str()).ok_or_else(|| {
                    AqlError::Runtime { msg: format!("undefined label '{label}' at runtime") }
                })?;
                pc = target;
                continue; // skip pc += 1 below
            }

            Instruction::GotoIf { qubit, label } => {
                let reg = classical[*qubit].ok_or_else(|| AqlError::Runtime {
                    msg: format!(
                        "IF on qubit {qubit}: qubit has not been measured yet — \
                         classical register is unset"
                    ),
                })?;
                if reg {
                    let &target = label_table.get(label.as_str()).ok_or_else(|| {
                        AqlError::Runtime { msg: format!("undefined label '{label}' at runtime") }
                    })?;
                    pc = target;
                    branch_count += 1;
                    continue;
                }
            }

            Instruction::GotoIfNot { qubit, label } => {
                let reg = classical[*qubit].ok_or_else(|| AqlError::Runtime {
                    msg: format!(
                        "IFNOT on qubit {qubit}: qubit has not been measured yet — \
                         classical register is unset"
                    ),
                })?;
                if !reg {
                    let &target = label_table.get(label.as_str()).ok_or_else(|| {
                        AqlError::Runtime { msg: format!("undefined label '{label}' at runtime") }
                    })?;
                    pc = target;
                    branch_count += 1;
                    continue;
                }
            }
        }

        pc += 1;
    }

    Ok(ExecutionResult {
        num_qubits: program.num_qubits,
        measurements,
        pre_measurement_probs,
        final_probabilities: sim.probabilities(),
        gate_count,
        branch_count,
        steps_executed,
    })
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::parse_source;

    fn run(src: &str) -> ExecutionResult {
        execute(&parse_source(src).expect("parse failed")).expect("execute failed")
    }

    // ── Basic gate execution ───────────────────────────────────────────

    #[test]
    fn test_x_gate_measures_one() {
        let r = run("QREG 1\nX 0\nMEASURE 0");
        assert!(r.outcome(0).unwrap());
    }

    #[test]
    fn test_double_x_returns_to_zero() {
        let r = run("QREG 1\nX 0\nX 0\nMEASURE 0");
        assert!(!r.outcome(0).unwrap());
    }

    #[test]
    fn test_bell_pre_measurement_probs() {
        let r = run("QREG 2\nH 0\nCNOT 0 1\nMEASURE_ALL");
        let probs = r.pre_measurement_probs.as_ref().unwrap();
        assert!((probs[0] - 0.5).abs() < 1e-10);
        assert!(probs[1].abs() < 1e-10);
        assert!(probs[2].abs() < 1e-10);
        assert!((probs[3] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_bell_qubits_always_agree() {
        for _ in 0..200 {
            let r = run("QREG 2\nH 0\nCNOT 0 1\nMEASURE_ALL");
            assert_eq!(r.outcome(0), r.outcome(1));
        }
    }

    #[test]
    fn test_gate_count() {
        let r = run("QREG 2\nH 0\nCNOT 0 1\nBARRIER\nMEASURE_ALL");
        assert_eq!(r.gate_count, 2);
    }

    #[test]
    fn test_no_measurement_program() {
        let r = run("QREG 2\nH 0\nCNOT 0 1");
        assert!(r.pre_measurement_probs.is_none());
        assert!(r.measurements.is_empty());
        // State should be Bell state
        assert!((r.final_probabilities[0] - 0.5).abs() < 1e-10);
        assert!((r.final_probabilities[3] - 0.5).abs() < 1e-10);
    }

    // ── Control flow ───────────────────────────────────────────────────

    #[test]
    fn test_goto_skips_instructions() {
        // GOTO done skips X 0 — q0 should stay |0⟩
        let r = run("QREG 1\nGOTO done\nX 0\nLABEL done\nMEASURE 0");
        assert!(!r.outcome(0).unwrap(), "X should have been skipped");
    }

    #[test]
    fn test_if_taken_when_qubit_is_one() {
        // X sets q0=1, then IF 0 GOTO skip skips the second X
        let r = run(
            "QREG 1\n\
             X 0\n\
             MEASURE 0\n\
             IF 0 GOTO skip\n\
             X 0\n\
             LABEL skip\n\
             MEASURE 0"
        );
        assert_eq!(r.measurements.len(), 2);
        // First measure: 1, branch taken, X skipped → second measure: 1
        assert!(r.measurements[0].outcome);
        assert!(r.measurements[1].outcome);
        assert_eq!(r.branch_count, 1);
    }

    #[test]
    fn test_if_not_taken_when_qubit_is_zero() {
        // q0 = 0 (no X), IF 0 GOTO skip is NOT taken
        let r = run(
            "QREG 1\n\
             MEASURE 0\n\
             IF 0 GOTO skip\n\
             X 0\n\
             LABEL skip\n\
             MEASURE 0"
        );
        // IF not taken → X applied → second measure: 1
        assert!(!r.measurements[0].outcome);
        assert!(r.measurements[1].outcome);
        assert_eq!(r.branch_count, 0);
    }

    #[test]
    fn test_ifnot_taken_when_qubit_is_zero() {
        // q0 = 0, IFNOT 0 GOTO skip → taken, X skipped
        let r = run(
            "QREG 1\n\
             MEASURE 0\n\
             IFNOT 0 GOTO skip\n\
             X 0\n\
             LABEL skip\n\
             MEASURE 0"
        );
        // IFNOT taken → X skipped → second measure: 0
        assert!(!r.measurements[0].outcome);
        assert!(!r.measurements[1].outcome);
        assert_eq!(r.branch_count, 1);
    }

    #[test]
    fn test_teleportation_with_conditional_corrections() {
        // Teleport |+⟩ (H|0⟩) from q0 to q2 with proper classical corrections
        // P(q2 = 1) should be 0.5 regardless of alice's measurement results
        let src = "\
QREG 3
H 0
H 1
CNOT 1 2
CNOT 0 1
H 0
MEASURE 0
MEASURE 1
IF 1 GOTO apply_x
GOTO skip_x
LABEL apply_x
X 2
LABEL skip_x
IF 0 GOTO apply_z
GOTO done
LABEL apply_z
Z 2
LABEL done
MEASURE 2";
        let mut p1_count = 0u32;
        let total = 400u32;
        for _ in 0..total {
            let r = run(src);
            if r.outcome(2).unwrap() { p1_count += 1; }
        }
        let p1 = p1_count as f64 / total as f64;
        assert!(
            (p1 - 0.5).abs() < 0.08,
            "P(q2=1) should be ~0.5, got {p1:.3}"
        );
    }

    #[test]
    fn test_grover_2qubits_finds_target() {
        // Grover on 2 qubits searching for |11⟩ — after 1 iteration P(|11⟩) = 1
        let r = run("\
QREG 2
H 0
H 1
CZ 0 1
H 0
H 1
X 0
X 1
CZ 0 1
X 0
X 1
H 0
H 1
MEASURE_ALL");
        // After one Grover iteration on 2 qubits, |11⟩ is certain
        let probs = r.pre_measurement_probs.as_ref().unwrap();
        assert!((probs[3] - 1.0).abs() < 1e-10, "P(|11⟩) should be 1.0, got {}", probs[3]);
        assert!(r.outcome(0).unwrap() && r.outcome(1).unwrap());
    }

    #[test]
    fn test_label_is_noop() {
        // Program with LABEL but no jumps — executes linearly
        let r = run("QREG 1\nLABEL start\nX 0\nLABEL end\nMEASURE 0");
        assert!(r.outcome(0).unwrap());
        assert_eq!(r.branch_count, 0);
    }

    // ── Error cases ────────────────────────────────────────────────────

    #[test]
    fn test_error_if_on_unmeasured_qubit() {
        // IF before MEASURE — classical register is unset
        let prog = parse_source("QREG 1\nIF 0 GOTO end\nLABEL end").unwrap();
        let err = execute(&prog).unwrap_err();
        assert!(matches!(err, AqlError::Runtime { .. }));
    }

    #[test]
    fn test_bitstring_full_collapse() {
        let r = run("QREG 2\nX 0\nX 1\nMEASURE_ALL");
        assert_eq!(r.bitstring(), Some("11".into()));
    }

    #[test]
    fn test_bitstring_partial_is_none() {
        let r = run("QREG 2\nX 0\nMEASURE 0");
        assert!(r.bitstring().is_none());
    }

    #[test]
    fn test_significant_states_bell() {
        let r = run("QREG 2\nH 0\nCNOT 0 1\nMEASURE_ALL");
        let probs = r.pre_measurement_probs.as_ref().unwrap();
        let states = r.significant_states(probs, 1e-6);
        assert_eq!(states.len(), 2);
        let labels: Vec<&str> = states.iter().map(|(l, _)| l.as_str()).collect();
        assert!(labels.contains(&"00"));
        assert!(labels.contains(&"11"));
    }

    #[test]
    fn test_steps_executed_is_tracked() {
        let r = run("QREG 1\nH 0\nH 0\nMEASURE 0");
        assert!(r.steps_executed >= 3);
    }

    #[test]
    fn test_ghz_state_probs() {
        let r = run("QREG 3\nH 0\nCNOT 0 1\nCNOT 0 2\nMEASURE_ALL");
        let probs = r.pre_measurement_probs.as_ref().unwrap();
        assert!((probs[0] - 0.5).abs() < 1e-10, "|000⟩ should be 0.5");
        assert!((probs[7] - 0.5).abs() < 1e-10, "|111⟩ should be 0.5");
        for i in 1..7 {
            assert!(probs[i].abs() < 1e-10, "unexpected amplitude at index {i}");
        }
        // All three qubits agree
        assert_eq!(r.outcome(0), r.outcome(1));
        assert_eq!(r.outcome(1), r.outcome(2));
    }

    #[test]
    fn test_outcome_last_measurement_wins() {
        // Measure q0 twice — rfind should return the last outcome
        let r = run("QREG 1\nX 0\nMEASURE 0\nX 0\nMEASURE 0");
        assert_eq!(r.measurements.len(), 2);
        // First MEASURE: X|0⟩=|1⟩ → true. Second MEASURE: X|1⟩=|0⟩ → false.
        assert!(r.measurements[0].outcome);
        assert!(!r.measurements[1].outcome);
        assert!(!r.outcome(0).unwrap(), "rfind should return last outcome = 0");
    }

    #[test]
    fn test_rx_pi_flips_qubit() {
        use std::f64::consts::PI;
        let r = run(&format!("QREG 1\nRX 0 {PI}\nMEASURE 0"));
        let probs = r.pre_measurement_probs.as_ref().unwrap();
        // Rx(π)|0⟩ = -i|1⟩ — P(|1⟩) = 1
        assert!(probs[0].abs() < 1e-10, "P(|0⟩) should be 0");
        assert!((probs[1] - 1.0).abs() < 1e-10, "P(|1⟩) should be 1");
        assert!(r.outcome(0).unwrap());
    }

    #[test]
    fn test_deutsch_constant_gives_zero() {
        // Constant oracle (identity) → q0 must measure 0
        let r = run("QREG 2\nX 1\nH 0\nH 1\nH 0\nMEASURE 0");
        assert!(!r.outcome(0).unwrap());
    }

    #[test]
    fn test_deutsch_balanced_gives_one() {
        // Balanced oracle (CNOT) → q0 must measure 1
        let r = run("QREG 2\nX 1\nH 0\nH 1\nCNOT 0 1\nH 0\nMEASURE 0");
        assert!(r.outcome(0).unwrap());
    }
}
