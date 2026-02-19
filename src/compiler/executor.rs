/// AQL Executor — maps an IR Program onto the quantum Simulator.
///
/// Execution model:
///   1. A fresh `Simulator` is created for the declared qubit count.
///   2. Instructions are executed in order.
///   3. Before the first measurement, a probability snapshot is captured
///      so the caller can display the theoretical distribution.
///   4. Measurements collapse the state and record the outcome.
///
/// The executor is intentionally stateless — it creates a new Simulator
/// per call. The Hybrid Runtime (Phase 4) will extend this with classical
/// control flow and persistent state across measurement-branch paths.
use super::{AqlError, ir::{Instruction, Program}};
use crate::core::Simulator;

// ── Result types ──────────────────────────────────────────────────────────

/// A single qubit measurement outcome.
#[derive(Debug, Clone)]
pub struct MeasurementRecord {
    /// Which qubit was measured.
    pub qubit: usize,
    /// Outcome: true = |1⟩, false = |0⟩.
    pub outcome: bool,
    /// Index into Program.instructions where this measurement occurred.
    pub step: usize,
}

/// The full result of executing an AQL program.
#[derive(Debug)]
pub struct ExecutionResult {
    /// Number of qubits in the program.
    pub num_qubits: usize,
    /// All measurement outcomes in the order they were performed.
    pub measurements: Vec<MeasurementRecord>,
    /// Probability distribution captured just before the first measurement.
    /// `None` if the program contained no measurement instructions.
    pub pre_measurement_probs: Option<Vec<f64>>,
    /// Final probability distribution after all operations.
    /// If measurements occurred, most amplitudes will be zero (collapsed).
    pub final_probabilities: Vec<f64>,
    /// Total gate operations applied (BARRIER and MEASURE not counted).
    pub gate_count: usize,
}

impl ExecutionResult {
    /// Return the measured outcome for qubit `q`.
    /// If a qubit was measured multiple times, returns the last outcome.
    /// Returns `None` if the qubit was never measured.
    pub fn outcome(&self, qubit: usize) -> Option<bool> {
        self.measurements
            .iter()
            .rfind(|m| m.qubit == qubit)
            .map(|m| m.outcome)
    }

    /// True if every qubit has at least one measurement record.
    pub fn fully_collapsed(&self) -> bool {
        (0..self.num_qubits).all(|q| self.outcome(q).is_some())
    }

    /// Measurement outcomes as a bit-string with qubit 0 at index 0 (left).
    /// Returns `None` if not all qubits were measured.
    pub fn bitstring(&self) -> Option<String> {
        if !self.fully_collapsed() {
            return None;
        }
        Some(
            (0..self.num_qubits)
                .map(|q| if self.outcome(q).unwrap() { '1' } else { '0' })
                .collect(),
        )
    }

    /// Format probabilities for display.
    /// Returns (basis_label, probability) pairs for states with p > threshold.
    pub fn significant_states(&self, probs: &[f64], threshold: f64) -> Vec<(String, f64)> {
        let n = self.num_qubits;
        probs
            .iter()
            .enumerate()
            .filter(|(_, &p)| p > threshold)
            .map(|(i, &p)| (format!("{:0>width$b}", i, width = n), p))
            .collect()
    }
}

// ── Execution ─────────────────────────────────────────────────────────────

/// Execute an AQL Program on a fresh Simulator instance.
pub fn execute(program: &Program) -> Result<ExecutionResult, AqlError> {
    let mut sim = Simulator::new(program.num_qubits);
    let mut measurements: Vec<MeasurementRecord> = Vec::new();
    let mut pre_measurement_probs: Option<Vec<f64>> = None;
    let mut gate_count: usize = 0;
    let mut first_measure = true;

    for (step, instr) in program.instructions.iter().enumerate() {
        // Capture quantum state snapshot before the first measurement
        if instr.is_measurement() && first_measure {
            pre_measurement_probs = Some(sim.probabilities());
            first_measure = false;
        }

        match instr {
            // ── Single-qubit gates ──────────────────────────────────────
            Instruction::H(q)               => { sim.h(*q);                gate_count += 1; }
            Instruction::X(q)               => { sim.x(*q);                gate_count += 1; }
            Instruction::Y(q)               => { sim.y(*q);                gate_count += 1; }
            Instruction::Z(q)               => { sim.z(*q);                gate_count += 1; }
            Instruction::S(q)               => { sim.s(*q);                gate_count += 1; }
            Instruction::T(q)               => { sim.t(*q);                gate_count += 1; }
            Instruction::Rx { qubit, theta } => { sim.rx(*qubit, *theta);  gate_count += 1; }
            Instruction::Ry { qubit, theta } => { sim.ry(*qubit, *theta);  gate_count += 1; }
            Instruction::Rz { qubit, theta } => { sim.rz(*qubit, *theta);  gate_count += 1; }
            Instruction::Phase { qubit, theta } => { sim.phase(*qubit, *theta); gate_count += 1; }

            // ── Multi-qubit gates ───────────────────────────────────────
            Instruction::Cnot { control, target } => { sim.cnot(*control, *target); gate_count += 1; }
            Instruction::Cz   { control, target } => { sim.cz(*control, *target);   gate_count += 1; }
            Instruction::Swap { qubit_a, qubit_b } => { sim.swap(*qubit_a, *qubit_b); gate_count += 1; }
            Instruction::Toffoli { control0, control1, target } => {
                sim.toffoli(*control0, *control1, *target);
                gate_count += 1;
            }

            // ── Measurement ─────────────────────────────────────────────
            Instruction::Measure(q) => {
                let outcome = sim.measure(*q);
                measurements.push(MeasurementRecord { qubit: *q, outcome, step });
            }
            Instruction::MeasureAll => {
                for q in 0..program.num_qubits {
                    let outcome = sim.measure(q);
                    measurements.push(MeasurementRecord { qubit: q, outcome, step });
                }
            }

            // ── Structural (no quantum effect) ──────────────────────────
            Instruction::Barrier => {}
        }
    }

    Ok(ExecutionResult {
        num_qubits: program.num_qubits,
        measurements,
        pre_measurement_probs,
        final_probabilities: sim.probabilities(),
        gate_count,
    })
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::{parse_source, AqlError};
    use super::execute;

    fn run(src: &str) -> Result<super::ExecutionResult, AqlError> {
        let prog = parse_source(src)?;
        execute(&prog)
    }

    #[test]
    fn test_x_gate_measures_one() {
        let result = run("QREG 1\nX 0\nMEASURE 0").unwrap();
        assert_eq!(result.measurements.len(), 1);
        assert!(result.measurements[0].outcome); // must be |1⟩
    }

    #[test]
    fn test_double_x_measures_zero() {
        let result = run("QREG 1\nX 0\nX 0\nMEASURE 0").unwrap();
        assert!(!result.measurements[0].outcome); // back to |0⟩
    }

    #[test]
    fn test_bell_pre_measurement_probs() {
        let result = run("QREG 2\nH 0\nCNOT 0 1\nMEASURE_ALL").unwrap();
        let probs = result.pre_measurement_probs.as_ref().unwrap();
        // Bell state: equal probability |00⟩ and |11⟩
        assert!((probs[0] - 0.5).abs() < 1e-10); // |00⟩
        assert!(probs[1].abs() < 1e-10);           // |01⟩
        assert!(probs[2].abs() < 1e-10);           // |10⟩
        assert!((probs[3] - 0.5).abs() < 1e-10); // |11⟩
    }

    #[test]
    fn test_bell_measurement_correlated() {
        // Bell state: q0 and q1 must always agree
        let mut agree = 0u32;
        let mut total = 0u32;
        for _ in 0..200 {
            let r = run("QREG 2\nH 0\nCNOT 0 1\nMEASURE_ALL").unwrap();
            let q0 = r.outcome(0).unwrap();
            let q1 = r.outcome(1).unwrap();
            if q0 == q1 { agree += 1; }
            total += 1;
        }
        assert_eq!(agree, total, "Bell state: qubits must always agree");
    }

    #[test]
    fn test_ghz_state() {
        let result = run("QREG 3\nH 0\nCNOT 0 1\nCNOT 0 2\nMEASURE_ALL").unwrap();
        let probs = result.pre_measurement_probs.as_ref().unwrap();
        assert!((probs[0] - 0.5).abs() < 1e-10); // |000⟩
        assert!((probs[7] - 0.5).abs() < 1e-10); // |111⟩
        for i in 1..7 {
            assert!(probs[i].abs() < 1e-10, "unexpected prob at index {i}");
        }
        // All 3 qubits must agree
        let q0 = result.outcome(0).unwrap();
        let q1 = result.outcome(1).unwrap();
        let q2 = result.outcome(2).unwrap();
        assert_eq!(q0, q1);
        assert_eq!(q1, q2);
    }

    #[test]
    fn test_no_measurement_returns_none_pre_probs() {
        let result = run("QREG 2\nH 0\nCNOT 0 1").unwrap();
        assert!(result.pre_measurement_probs.is_none());
        assert!(result.measurements.is_empty());
        // Final probs should reflect Bell state
        let probs = &result.final_probabilities;
        assert!((probs[0] - 0.5).abs() < 1e-10);
        assert!((probs[3] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_gate_count() {
        let prog = super::super::parse_source("QREG 2\nH 0\nCNOT 0 1\nBARRIER\nMEASURE_ALL")
            .unwrap();
        let result = execute(&prog).unwrap();
        // H + CNOT = 2 gates; BARRIER and MEASURE_ALL are not counted
        assert_eq!(result.gate_count, 2);
    }

    #[test]
    fn test_bitstring_all_ones() {
        let result = run("QREG 2\nX 0\nX 1\nMEASURE_ALL").unwrap();
        assert_eq!(result.bitstring(), Some("11".into()));
    }

    #[test]
    fn test_bitstring_partial_measurement_is_none() {
        let result = run("QREG 2\nX 0\nMEASURE 0").unwrap();
        // q1 not measured → bitstring unavailable
        assert!(result.bitstring().is_none());
    }

    #[test]
    fn test_outcome_last_measurement_wins() {
        // Measure q0 twice — only last result stored via rfind
        let result = run("QREG 1\nX 0\nMEASURE 0\nX 0\nMEASURE 0").unwrap();
        assert_eq!(result.measurements.len(), 2);
        // First: X→|1⟩ → 1. Second: X→|0⟩ → 0. Last outcome = 0.
        assert!(!result.outcome(0).unwrap());
    }

    #[test]
    fn test_significant_states() {
        let result = run("QREG 2\nH 0\nCNOT 0 1\nMEASURE_ALL").unwrap();
        let probs = result.pre_measurement_probs.as_ref().unwrap();
        let states = result.significant_states(probs, 1e-6);
        // Only |00⟩ and |11⟩ should be significant
        assert_eq!(states.len(), 2);
        let labels: Vec<&str> = states.iter().map(|(l, _)| l.as_str()).collect();
        assert!(labels.contains(&"00"));
        assert!(labels.contains(&"11"));
    }

    #[test]
    fn test_rx_pi_flips_qubit() {
        use std::f64::consts::PI;
        let result = run(&format!("QREG 1\nRX 0 {PI}\nMEASURE 0")).unwrap();
        // Rx(π)|0⟩ = -i|1⟩ — probability of |1⟩ is 1
        let probs = result.pre_measurement_probs.as_ref().unwrap();
        assert!(probs[0].abs() < 1e-10);
        assert!((probs[1] - 1.0).abs() < 1e-10);
        assert!(result.outcome(0).unwrap()); // must be |1⟩
    }

    #[test]
    fn test_deutsch_constant_gives_zero() {
        // Constant oracle (identity) → measurement of q0 must be 0
        let result = run(
            "QREG 2\n\
             X 1\n\
             H 0\nH 1\n\
             H 0\n\
             MEASURE 0"
        ).unwrap();
        assert!(!result.outcome(0).unwrap());
    }

    #[test]
    fn test_deutsch_balanced_gives_one() {
        // Balanced oracle (CNOT) → measurement of q0 must be 1
        let result = run(
            "QREG 2\n\
             X 1\n\
             H 0\nH 1\n\
             CNOT 0 1\n\
             H 0\n\
             MEASURE 0"
        ).unwrap();
        assert!(result.outcome(0).unwrap());
    }
}
