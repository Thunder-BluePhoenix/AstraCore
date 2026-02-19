/// AQL Intermediate Representation.
///
/// After parsing, an AQL program is a flat, ordered sequence of `Instruction`s
/// wrapped in a validated `Program`. The IR is the canonical form that the
/// executor and future optimizer operate on.
///
/// Design principles:
///   - One enum variant per instruction — no string dispatch at runtime
///   - Angles stored as f64 radians — no unit ambiguity
///   - Qubit indices as usize — validated against QREG at parse time
///   - Display impl produces valid AQL output — IR is round-trippable

// ── Instruction ───────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum Instruction {
    // ── Single-qubit gates ──────────────────────────────────────────────
    H(usize),
    X(usize),
    Y(usize),
    Z(usize),
    S(usize),
    T(usize),
    Rx     { qubit: usize, theta: f64 },
    Ry     { qubit: usize, theta: f64 },
    Rz     { qubit: usize, theta: f64 },
    Phase  { qubit: usize, theta: f64 },

    // ── Multi-qubit gates ───────────────────────────────────────────────
    Cnot    { control: usize, target: usize },
    Cz      { control: usize, target: usize },
    Swap    { qubit_a: usize, qubit_b: usize },
    Toffoli { control0: usize, control1: usize, target: usize },

    // ── Measurement ─────────────────────────────────────────────────────
    Measure(usize),
    MeasureAll,

    // ── Structural (no quantum effect) ──────────────────────────────────
    Barrier,

    // ── Phase 4: Hybrid control flow ─────────────────────────────────────
    /// Define a named jump target. No quantum effect.
    Label(String),
    /// Unconditional jump to a label.
    Goto { label: String },
    /// Jump to `label` if qubit `qubit` last measured as |1⟩.
    GotoIf { qubit: usize, label: String },
    /// Jump to `label` if qubit `qubit` last measured as |0⟩.
    GotoIfNot { qubit: usize, label: String },
}

impl Instruction {
    /// Mnemonic name used in AQL source and diagnostics.
    pub fn mnemonic(&self) -> &'static str {
        match self {
            Self::H(_)              => "H",
            Self::X(_)              => "X",
            Self::Y(_)              => "Y",
            Self::Z(_)              => "Z",
            Self::S(_)              => "S",
            Self::T(_)              => "T",
            Self::Rx { .. }         => "RX",
            Self::Ry { .. }         => "RY",
            Self::Rz { .. }         => "RZ",
            Self::Phase { .. }      => "PHASE",
            Self::Cnot { .. }       => "CNOT",
            Self::Cz { .. }         => "CZ",
            Self::Swap { .. }       => "SWAP",
            Self::Toffoli { .. }    => "CCX",
            Self::Measure(_)        => "MEASURE",
            Self::MeasureAll        => "MEASURE_ALL",
            Self::Barrier           => "BARRIER",
            Self::Label(_)          => "LABEL",
            Self::Goto { .. }       => "GOTO",
            Self::GotoIf { .. }     => "IF",
            Self::GotoIfNot { .. }  => "IFNOT",
        }
    }

    /// True if this instruction applies a quantum gate.
    pub fn is_gate(&self) -> bool {
        matches!(self,
            Self::H(_) | Self::X(_) | Self::Y(_) | Self::Z(_) | Self::S(_) | Self::T(_)
            | Self::Rx { .. } | Self::Ry { .. } | Self::Rz { .. } | Self::Phase { .. }
            | Self::Cnot { .. } | Self::Cz { .. } | Self::Swap { .. } | Self::Toffoli { .. }
        )
    }

    /// True if this instruction is a measurement.
    pub fn is_measurement(&self) -> bool {
        matches!(self, Self::Measure(_) | Self::MeasureAll)
    }

    /// True if this instruction is a control-flow directive.
    pub fn is_control_flow(&self) -> bool {
        matches!(self, Self::Label(_) | Self::Goto { .. } | Self::GotoIf { .. } | Self::GotoIfNot { .. })
    }

    /// Qubit indices referenced by this instruction.
    /// Includes the condition qubit in GotoIf/GotoIfNot (data dependency).
    pub fn qubits(&self) -> Vec<usize> {
        match self {
            Self::H(q) | Self::X(q) | Self::Y(q) | Self::Z(q)
            | Self::S(q) | Self::T(q) | Self::Measure(q)           => vec![*q],
            Self::Rx { qubit, .. } | Self::Ry { qubit, .. }
            | Self::Rz { qubit, .. } | Self::Phase { qubit, .. }   => vec![*qubit],
            Self::Cnot { control, target }
            | Self::Cz  { control, target }                         => vec![*control, *target],
            Self::Swap { qubit_a, qubit_b }                         => vec![*qubit_a, *qubit_b],
            Self::Toffoli { control0, control1, target }            => vec![*control0, *control1, *target],
            Self::GotoIf { qubit, .. } | Self::GotoIfNot { qubit, .. } => vec![*qubit],
            Self::MeasureAll | Self::Barrier
            | Self::Label(_) | Self::Goto { .. }                    => vec![],
        }
    }

    /// True if the program has any control-flow instructions.
    pub fn program_has_control_flow(instructions: &[Self]) -> bool {
        instructions.iter().any(|i| i.is_control_flow())
    }
}

impl std::fmt::Display for Instruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::H(q)                                    => write!(f, "H {q}"),
            Self::X(q)                                    => write!(f, "X {q}"),
            Self::Y(q)                                    => write!(f, "Y {q}"),
            Self::Z(q)                                    => write!(f, "Z {q}"),
            Self::S(q)                                    => write!(f, "S {q}"),
            Self::T(q)                                    => write!(f, "T {q}"),
            Self::Rx     { qubit, theta }                 => write!(f, "RX {qubit} {theta:.6}"),
            Self::Ry     { qubit, theta }                 => write!(f, "RY {qubit} {theta:.6}"),
            Self::Rz     { qubit, theta }                 => write!(f, "RZ {qubit} {theta:.6}"),
            Self::Phase  { qubit, theta }                 => write!(f, "PHASE {qubit} {theta:.6}"),
            Self::Cnot   { control, target }              => write!(f, "CNOT {control} {target}"),
            Self::Cz     { control, target }              => write!(f, "CZ {control} {target}"),
            Self::Swap   { qubit_a, qubit_b }             => write!(f, "SWAP {qubit_a} {qubit_b}"),
            Self::Toffoli{ control0, control1, target }   => write!(f, "CCX {control0} {control1} {target}"),
            Self::Measure(q)                              => write!(f, "MEASURE {q}"),
            Self::MeasureAll                              => write!(f, "MEASURE_ALL"),
            Self::Barrier                                 => write!(f, "BARRIER"),
            Self::Label(name)                             => write!(f, "LABEL {name}"),
            Self::Goto { label }                          => write!(f, "GOTO {label}"),
            Self::GotoIf { qubit, label }                 => write!(f, "IF {qubit} GOTO {label}"),
            Self::GotoIfNot { qubit, label }              => write!(f, "IFNOT {qubit} GOTO {label}"),
        }
    }
}

// ── Program ───────────────────────────────────────────────────────────────

/// A validated, ready-to-execute AQL program.
#[derive(Debug, Clone)]
pub struct Program {
    /// Qubit count declared by `QREG n`.
    pub num_qubits: usize,
    /// Flat instruction sequence (QREG directive excluded).
    pub instructions: Vec<Instruction>,
    /// Number of gate operations (excludes BARRIER, MEASURE).
    pub gate_count: usize,
    /// Number of measurement operations.
    pub measure_count: usize,
}

impl Program {
    pub(crate) fn new(num_qubits: usize, instructions: Vec<Instruction>) -> Self {
        let gate_count    = instructions.iter().filter(|i| i.is_gate()).count();
        let measure_count = instructions.iter().filter(|i| i.is_measurement()).count();
        Self { num_qubits, instructions, gate_count, measure_count }
    }
}

impl std::fmt::Display for Program {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "QREG {}", self.num_qubits)?;
        for instr in &self.instructions {
            writeln!(f, "{}", instr)?;
        }
        Ok(())
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_instruction_mnemonic() {
        assert_eq!(Instruction::H(0).mnemonic(), "H");
        assert_eq!(Instruction::Cnot { control: 0, target: 1 }.mnemonic(), "CNOT");
        assert_eq!(Instruction::MeasureAll.mnemonic(), "MEASURE_ALL");
    }

    #[test]
    fn test_instruction_is_gate() {
        assert!(Instruction::H(0).is_gate());
        assert!(Instruction::Cnot { control: 0, target: 1 }.is_gate());
        assert!(!Instruction::Measure(0).is_gate());
        assert!(!Instruction::MeasureAll.is_gate());
        assert!(!Instruction::Barrier.is_gate());
    }

    #[test]
    fn test_instruction_qubits() {
        assert_eq!(Instruction::H(2).qubits(), vec![2]);
        assert_eq!(
            Instruction::Cnot { control: 0, target: 3 }.qubits(),
            vec![0, 3]
        );
        assert_eq!(
            Instruction::Toffoli { control0: 0, control1: 1, target: 2 }.qubits(),
            vec![0, 1, 2]
        );
        assert_eq!(Instruction::MeasureAll.qubits(), vec![]);
    }

    #[test]
    fn test_instruction_display() {
        assert_eq!(format!("{}", Instruction::H(0)), "H 0");
        assert_eq!(format!("{}", Instruction::Cnot { control: 0, target: 1 }), "CNOT 0 1");
        assert_eq!(
            format!("{}", Instruction::Rx { qubit: 2, theta: PI }),
            format!("RX 2 {:.6}", PI)
        );
        assert_eq!(format!("{}", Instruction::MeasureAll), "MEASURE_ALL");
    }

    #[test]
    fn test_program_gate_count() {
        let prog = Program::new(2, vec![
            Instruction::H(0),
            Instruction::Cnot { control: 0, target: 1 },
            Instruction::Barrier,
            Instruction::Measure(0),
            Instruction::Measure(1),
        ]);
        assert_eq!(prog.gate_count, 2);
        assert_eq!(prog.measure_count, 2);
    }

    #[test]
    fn test_control_flow_instructions() {
        let label = Instruction::Label("start".into());
        let goto  = Instruction::Goto { label: "start".into() };
        let gif   = Instruction::GotoIf    { qubit: 0, label: "branch".into() };
        let gifn  = Instruction::GotoIfNot { qubit: 1, label: "skip".into() };

        assert!(label.is_control_flow());
        assert!(goto.is_control_flow());
        assert!(!label.is_gate());
        assert!(!goto.is_gate());
        assert_eq!(gif.qubits(), vec![0]);
        assert_eq!(gifn.qubits(), vec![1]);
        assert_eq!(format!("{label}"), "LABEL start");
        assert_eq!(format!("{goto}"),  "GOTO start");
        assert_eq!(format!("{gif}"),   "IF 0 GOTO branch");
        assert_eq!(format!("{gifn}"),  "IFNOT 1 GOTO skip");
    }

    #[test]
    fn test_program_has_control_flow() {
        let no_cf = vec![Instruction::H(0), Instruction::MeasureAll];
        let with_cf = vec![Instruction::H(0), Instruction::Label("x".into())];
        assert!(!Instruction::program_has_control_flow(&no_cf));
        assert!(Instruction::program_has_control_flow(&with_cf));
    }

    #[test]
    fn test_program_display_round_trip() {
        let prog = Program::new(2, vec![
            Instruction::H(0),
            Instruction::Cnot { control: 0, target: 1 },
            Instruction::MeasureAll,
        ]);
        let s = prog.to_string();
        assert!(s.contains("QREG 2"));
        assert!(s.contains("H 0"));
        assert!(s.contains("CNOT 0 1"));
        assert!(s.contains("MEASURE_ALL"));
    }
}
