/// AQL Parser — converts a token stream into a validated Program.
///
/// Grammar (simplified BNF):
///   program     := 'QREG' INT instruction*
///   instruction := single_gate | rotation_gate | two_qubit_gate | toffoli
///                | measure | 'MEASURE_ALL' | 'BARRIER'
///   single_gate := ('H'|'X'|'Y'|'Z'|'S'|'T') INT
///   rot_gate    := ('RX'|'RY'|'RZ'|'PHASE') INT FLOAT
///   two_qubit   := ('CNOT'|'CZ'|'SWAP') INT INT
///   toffoli     := 'CCX' INT INT INT
///   measure     := 'MEASURE' INT
///
/// Validation:
///   - First statement must be QREG
///   - QREG declares 1–30 qubits
///   - No duplicate QREG after the first
///   - All qubit indices in [0, num_qubits)
use super::{
    AqlError,
    ir::{Instruction, Program},
    lexer::{Spanned, Token},
};

// Internal sentinel that can represent either a QREG directive or a body instruction.
enum Item {
    Qreg(usize),
    Instr(Instruction),
}

// ── Public API ────────────────────────────────────────────────────────────

/// Parse a token stream (output of the lexer) into a validated Program.
pub fn parse(statements: Vec<Vec<Spanned>>) -> Result<Program, AqlError> {
    if statements.is_empty() {
        return Err(AqlError::Validation {
            msg: "empty program — expected 'QREG <n>' as first statement".into(),
        });
    }

    // Parse every line
    let mut items: Vec<Item> = Vec::with_capacity(statements.len());
    for stmt in &statements {
        items.push(parse_statement(stmt)?);
    }

    // First item must be QREG
    let num_qubits = match items.first() {
        Some(Item::Qreg(n)) => *n,
        Some(Item::Instr(_)) => return Err(AqlError::Validation {
            msg: "first instruction must be 'QREG <n>'".into(),
        }),
        None => unreachable!(),
    };

    if num_qubits == 0 {
        return Err(AqlError::Validation {
            msg: "QREG must declare at least 1 qubit".into(),
        });
    }
    if num_qubits > 30 {
        return Err(AqlError::Validation {
            msg: format!(
                "QREG {num_qubits} exceeds maximum of 30 qubits (would require >8 GB RAM)"
            ),
        });
    }

    // Collect body — reject any additional QREG
    let mut body: Vec<Instruction> = Vec::with_capacity(items.len().saturating_sub(1));
    for item in items.into_iter().skip(1) {
        match item {
            Item::Qreg(_) => return Err(AqlError::Validation {
                msg: "only one QREG declaration is allowed per program".into(),
            }),
            Item::Instr(instr) => body.push(instr),
        }
    }

    // Validate qubit indices are in range
    validate_bounds(&body, num_qubits)?;

    Ok(Program::new(num_qubits, body))
}

// ── Statement parser ──────────────────────────────────────────────────────

fn parse_statement(tokens: &[Spanned]) -> Result<Item, AqlError> {
    debug_assert!(!tokens.is_empty());
    let line = tokens[0].line;

    // Enforce exact argument count.
    let check_argc = |expected: usize| -> Result<(), AqlError> {
        if tokens.len() != expected {
            Err(AqlError::Parse {
                line,
                msg: format!(
                    "'{}' expects {} argument(s), got {}",
                    tokens[0].token.display(),
                    expected - 1,
                    tokens.len() - 1
                ),
            })
        } else {
            Ok(())
        }
    };

    Ok(match &tokens[0].token {
        // ── Directive ──────────────────────────────────────────────────
        Token::Qreg => {
            check_argc(2)?;
            Item::Qreg(int_arg(&tokens[1])?)
        }

        // ── Single-qubit gates ─────────────────────────────────────────
        Token::H => { check_argc(2)?; Item::Instr(Instruction::H(int_arg(&tokens[1])?)) }
        Token::X => { check_argc(2)?; Item::Instr(Instruction::X(int_arg(&tokens[1])?)) }
        Token::Y => { check_argc(2)?; Item::Instr(Instruction::Y(int_arg(&tokens[1])?)) }
        Token::Z => { check_argc(2)?; Item::Instr(Instruction::Z(int_arg(&tokens[1])?)) }
        Token::S => { check_argc(2)?; Item::Instr(Instruction::S(int_arg(&tokens[1])?)) }
        Token::T => { check_argc(2)?; Item::Instr(Instruction::T(int_arg(&tokens[1])?)) }

        // ── Rotation gates ─────────────────────────────────────────────
        Token::Rx => {
            check_argc(3)?;
            Item::Instr(Instruction::Rx {
                qubit: int_arg(&tokens[1])?,
                theta: flt_arg(&tokens[2])?,
            })
        }
        Token::Ry => {
            check_argc(3)?;
            Item::Instr(Instruction::Ry {
                qubit: int_arg(&tokens[1])?,
                theta: flt_arg(&tokens[2])?,
            })
        }
        Token::Rz => {
            check_argc(3)?;
            Item::Instr(Instruction::Rz {
                qubit: int_arg(&tokens[1])?,
                theta: flt_arg(&tokens[2])?,
            })
        }
        Token::Phase => {
            check_argc(3)?;
            Item::Instr(Instruction::Phase {
                qubit: int_arg(&tokens[1])?,
                theta: flt_arg(&tokens[2])?,
            })
        }

        // ── Two-qubit gates ────────────────────────────────────────────
        Token::Cnot => {
            check_argc(3)?;
            Item::Instr(Instruction::Cnot {
                control: int_arg(&tokens[1])?,
                target:  int_arg(&tokens[2])?,
            })
        }
        Token::Cz => {
            check_argc(3)?;
            Item::Instr(Instruction::Cz {
                control: int_arg(&tokens[1])?,
                target:  int_arg(&tokens[2])?,
            })
        }
        Token::Swap => {
            check_argc(3)?;
            Item::Instr(Instruction::Swap {
                qubit_a: int_arg(&tokens[1])?,
                qubit_b: int_arg(&tokens[2])?,
            })
        }

        // ── Toffoli ────────────────────────────────────────────────────
        Token::Ccx => {
            check_argc(4)?;
            Item::Instr(Instruction::Toffoli {
                control0: int_arg(&tokens[1])?,
                control1: int_arg(&tokens[2])?,
                target:   int_arg(&tokens[3])?,
            })
        }

        // ── Measurement ────────────────────────────────────────────────
        Token::Measure => {
            check_argc(2)?;
            Item::Instr(Instruction::Measure(int_arg(&tokens[1])?))
        }
        Token::MeasureAll => {
            check_argc(1)?;
            Item::Instr(Instruction::MeasureAll)
        }

        // ── Structural ─────────────────────────────────────────────────
        Token::Barrier => Item::Instr(Instruction::Barrier),

        // ── Literals at start of line are errors ───────────────────────
        Token::Int(n) => return Err(AqlError::Parse {
            line,
            msg: format!("unexpected integer '{n}' at start of statement — expected a gate mnemonic"),
        }),
        Token::Float(f) => return Err(AqlError::Parse {
            line,
            msg: format!("unexpected float '{f}' at start of statement — expected a gate mnemonic"),
        }),
    })
}

// ── Argument extractors ───────────────────────────────────────────────────

/// Require a qubit index: a non-negative integer token.
fn int_arg(s: &Spanned) -> Result<usize, AqlError> {
    match s.token {
        Token::Int(n) => Ok(n),
        _ => Err(AqlError::Parse {
            line: s.line,
            msg: format!(
                "expected qubit index (non-negative integer), got '{}'",
                s.token.display()
            ),
        }),
    }
}

/// Require a rotation angle: a float or integer token (integer promoted to f64).
fn flt_arg(s: &Spanned) -> Result<f64, AqlError> {
    match s.token {
        Token::Float(f) => Ok(f),
        Token::Int(n)   => Ok(n as f64),
        _ => Err(AqlError::Parse {
            line: s.line,
            msg: format!(
                "expected rotation angle (float), got '{}'",
                s.token.display()
            ),
        }),
    }
}

// ── Qubit bound validation ────────────────────────────────────────────────

fn validate_bounds(body: &[Instruction], num_qubits: usize) -> Result<(), AqlError> {
    for instr in body {
        for q in instr.qubits() {
            if q >= num_qubits {
                return Err(AqlError::Validation {
                    msg: format!(
                        "qubit index {q} is out of range — QREG declared {num_qubits} qubit(s)"
                    ),
                });
            }
        }
    }
    Ok(())
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::{lexer, AqlError};
    use super::super::ir::{Instruction, Program};

    fn compile(src: &str) -> Result<Program, AqlError> {
        let tokens = lexer::tokenize(src)?;
        super::parse(tokens)
    }

    #[test]
    fn test_bell_circuit() {
        let prog = compile("QREG 2\nH 0\nCNOT 0 1\nMEASURE_ALL").unwrap();
        assert_eq!(prog.num_qubits, 2);
        assert_eq!(prog.instructions.len(), 3);
        assert_eq!(prog.gate_count, 2);
        assert_eq!(prog.measure_count, 1);
    }

    #[test]
    fn test_ghz_circuit() {
        let prog = compile("QREG 3\nH 0\nCNOT 0 1\nCNOT 0 2\nMEASURE_ALL").unwrap();
        assert_eq!(prog.num_qubits, 3);
        assert_eq!(prog.gate_count, 3);
    }

    #[test]
    fn test_rotation_gates_with_pi() {
        use std::f64::consts::PI;
        let prog = compile("QREG 1\nRX 0 PI\nRY 0 PI_2\nRZ 0 -1.5708").unwrap();
        assert_eq!(prog.gate_count, 3);
        if let Instruction::Rx { qubit, theta } = prog.instructions[0] {
            assert_eq!(qubit, 0);
            assert!((theta - PI).abs() < 1e-10);
        } else {
            panic!("expected Rx instruction");
        }
    }

    #[test]
    fn test_toffoli_gate() {
        let prog = compile("QREG 3\nCCX 0 1 2\nMEASURE_ALL").unwrap();
        assert_eq!(prog.gate_count, 1);
        assert!(matches!(prog.instructions[0], Instruction::Toffoli { .. }));
    }

    #[test]
    fn test_barrier_not_counted_as_gate() {
        let prog = compile("QREG 2\nH 0\nBARRIER\nH 1").unwrap();
        assert_eq!(prog.instructions.len(), 3);
        assert_eq!(prog.gate_count, 2);
    }

    #[test]
    fn test_all_single_qubit_gates() {
        let prog = compile("QREG 1\nH 0\nX 0\nY 0\nZ 0\nS 0\nT 0").unwrap();
        assert_eq!(prog.gate_count, 6);
    }

    #[test]
    fn test_comments_ignored() {
        let prog = compile("QREG 2 // two qubits\nH 0 # put in superposition\nCNOT 0 1").unwrap();
        assert_eq!(prog.gate_count, 2);
    }

    #[test]
    fn test_case_insensitive() {
        let prog = compile("qreg 2\nh 0\ncnot 0 1\nmeasure_all").unwrap();
        assert_eq!(prog.num_qubits, 2);
        assert_eq!(prog.gate_count, 2);
    }

    // ── Error cases ───────────────────────────────────────────────────

    #[test]
    fn test_error_empty_program() {
        assert!(matches!(compile(""), Err(AqlError::Validation { .. })));
    }

    #[test]
    fn test_error_missing_qreg() {
        assert!(matches!(compile("H 0\nCNOT 0 1"), Err(AqlError::Validation { .. })));
    }

    #[test]
    fn test_error_qreg_zero() {
        assert!(matches!(compile("QREG 0\nH 0"), Err(AqlError::Validation { .. })));
    }

    #[test]
    fn test_error_duplicate_qreg() {
        assert!(matches!(compile("QREG 2\nH 0\nQREG 3"), Err(AqlError::Validation { .. })));
    }

    #[test]
    fn test_error_qubit_out_of_range() {
        assert!(matches!(compile("QREG 2\nH 5"), Err(AqlError::Validation { .. })));
    }

    #[test]
    fn test_error_too_many_args() {
        // H takes 1 qubit arg; passing 2 is wrong
        assert!(matches!(compile("QREG 2\nH 0 1"), Err(AqlError::Parse { .. })));
    }

    #[test]
    fn test_error_too_few_args() {
        // CNOT needs 2 qubit args
        assert!(matches!(compile("QREG 2\nCNOT 0"), Err(AqlError::Parse { .. })));
    }

    #[test]
    fn test_error_float_as_qubit_index() {
        // 1.5 cannot be a qubit index
        let result = compile("QREG 2\nH 1.5");
        assert!(result.is_err());
    }

    #[test]
    fn test_error_unknown_token() {
        assert!(matches!(compile("QREG 2\nFLIB 0"), Err(AqlError::Lex { .. })));
    }
}
