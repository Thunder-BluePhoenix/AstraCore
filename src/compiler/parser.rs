/// AQL Parser — converts a token stream into a validated Program.
///
/// Grammar (simplified BNF):
///   program        := gate_def* 'QREG' INT (gate_def | instruction)*
///   gate_def       := 'GATE' IDENT INT gate_body_instr* 'END'
///   instruction    := single_gate | rotation_gate | two_qubit_gate | toffoli
///                   | measure | 'MEASURE_ALL' | 'BARRIER' | call_gate
///                   | label | goto | if_goto | ifnot_goto
///   call_gate      := 'CALL' IDENT INT+
///   single_gate    := ('H'|'X'|'Y'|'Z'|'S'|'T') INT
///   rot_gate       := ('RX'|'RY'|'RZ'|'PHASE') INT FLOAT
///   two_qubit      := ('CNOT'|'CZ'|'SWAP') INT INT
///   toffoli        := 'CCX' INT INT INT
///   measure        := 'MEASURE' INT
///
/// Validation:
///   - QREG declares 1–30 qubits (may be preceded by GATE definitions)
///   - No duplicate QREG
///   - All global qubit indices in [0, num_qubits)
///   - Gate body uses local indices [0, gate.num_qubits)
///   - CALL arity matches gate definition
///   - All jump targets refer to defined labels
use std::collections::HashMap;
use super::{
    AqlError,
    ir::{GateDef, Instruction, Program},
    lexer::{Spanned, Token},
};

// Internal sentinel that can represent a QREG directive or a body instruction.
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

    // ── Phase 1: Extract all GATE...END definitions ────────────────────────
    let (gate_defs, remaining) = extract_gate_defs(statements)?;

    if remaining.is_empty() {
        return Err(AqlError::Validation {
            msg: "empty program body — expected 'QREG <n>'".into(),
        });
    }

    // ── Phase 2: Parse remaining statements ───────────────────────────────
    let mut items: Vec<Item> = Vec::with_capacity(remaining.len());
    for stmt in &remaining {
        items.push(parse_statement(stmt, &gate_defs)?);
    }

    // First item must be QREG
    let num_qubits = match items.first() {
        Some(Item::Qreg(n)) => *n,
        Some(Item::Instr(_)) => return Err(AqlError::Validation {
            msg: "first non-GATE statement must be 'QREG <n>'".into(),
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

    // ── Phase 3: Validate ─────────────────────────────────────────────────
    validate_bounds(&body, num_qubits, &gate_defs)?;
    validate_labels(&body)?;

    Ok(Program::with_gate_defs(num_qubits, body, gate_defs))
}

// ── Gate definition extraction ────────────────────────────────────────────

/// Pre-pass: scan all statements for GATE...END blocks, extracting them into
/// a gate library. Returns the library and the remaining (non-GATE) statements.
///
/// GATE blocks may appear before or after QREG — they are always extracted first.
fn extract_gate_defs(
    statements: Vec<Vec<Spanned>>,
) -> Result<(HashMap<String, GateDef>, Vec<Vec<Spanned>>), AqlError> {
    let mut gate_defs: HashMap<String, GateDef> = HashMap::new();
    let mut remaining: Vec<Vec<Spanned>> = Vec::new();
    let mut iter = statements.into_iter();

    while let Some(stmt) = iter.next() {
        match stmt.first().map(|s| &s.token) {
            Some(Token::Gate) => {
                // ── Header: GATE <name> <num_qubits> ─────────────────────
                let line = stmt[0].line;
                if stmt.len() != 3 {
                    return Err(AqlError::Parse {
                        line,
                        msg: format!(
                            "'GATE' expects exactly 2 arguments (name, qubit-count), got {}",
                            stmt.len().saturating_sub(1)
                        ),
                    });
                }
                let name = ident_arg(&stmt[1])?.to_lowercase();
                let num_qubits = int_arg(&stmt[2])?;
                if num_qubits == 0 {
                    return Err(AqlError::Validation {
                        msg: format!("gate '{name}' must accept at least 1 qubit parameter"),
                    });
                }

                // ── Collect body lines until END ──────────────────────────
                let mut body_stmts: Vec<Vec<Spanned>> = Vec::new();
                let mut found_end = false;
                for body_stmt in iter.by_ref() {
                    if matches!(body_stmt.first().map(|s| &s.token), Some(Token::End)) {
                        found_end = true;
                        break;
                    }
                    body_stmts.push(body_stmt);
                }
                if !found_end {
                    return Err(AqlError::Parse {
                        line,
                        msg: format!("gate '{name}' has no matching END"),
                    });
                }

                // ── Parse body using local qubit indices ──────────────────
                let mut body: Vec<Instruction> = Vec::new();
                for body_stmt in &body_stmts {
                    let instr = parse_gate_body_stmt(body_stmt, &name, num_qubits)?;
                    body.push(instr);
                }

                // ── Register — reject duplicates ──────────────────────────
                if gate_defs.contains_key(&name) {
                    return Err(AqlError::Validation {
                        msg: format!("duplicate gate definition '{name}'"),
                    });
                }
                gate_defs.insert(name.clone(), GateDef::new(name, num_qubits, body));
            }

            Some(Token::End) => {
                return Err(AqlError::Parse {
                    line: stmt[0].line,
                    msg: "'END' without a preceding 'GATE' definition".into(),
                });
            }

            _ => remaining.push(stmt),
        }
    }

    Ok((gate_defs, remaining))
}

/// Parse one instruction line inside a GATE body.
///
/// - Only pure-gate instructions are permitted (no MEASURE, QREG, control flow, CALL).
/// - Qubit indices are validated against [0, gate_qubits).
fn parse_gate_body_stmt(
    tokens: &[Spanned],
    gate_name: &str,
    gate_qubits: usize,
) -> Result<Instruction, AqlError> {
    debug_assert!(!tokens.is_empty());
    let line = tokens[0].line;

    // Reuse the normal statement parser (empty gate_defs: no nested CALL)
    let item = parse_statement(tokens, &HashMap::new())?;
    let instr = match item {
        Item::Instr(i) => i,
        Item::Qreg(_) => return Err(AqlError::Parse {
            line,
            msg: format!("'QREG' is not allowed inside gate '{gate_name}' body"),
        }),
    };

    if instr.is_measurement() {
        return Err(AqlError::Parse {
            line,
            msg: format!("MEASURE is not allowed inside gate '{gate_name}' body"),
        });
    }
    if instr.is_control_flow() {
        return Err(AqlError::Parse {
            line,
            msg: format!(
                "control flow (LABEL/GOTO/IF/IFNOT) is not allowed inside gate '{gate_name}' body"
            ),
        });
    }
    if matches!(instr, Instruction::CallGate { .. }) {
        return Err(AqlError::Parse {
            line,
            msg: format!("nested CALL is not allowed inside gate '{gate_name}' body"),
        });
    }

    // Validate local qubit indices
    for q in instr.qubits() {
        if q >= gate_qubits {
            return Err(AqlError::Validation {
                msg: format!(
                    "local qubit index {q} in gate '{gate_name}' is out of range \
                     (gate has {gate_qubits} qubit parameter(s))"
                ),
            });
        }
    }

    Ok(instr)
}

// ── Statement parser ──────────────────────────────────────────────────────

fn parse_statement(
    tokens: &[Spanned],
    gate_defs: &HashMap<String, GateDef>,
) -> Result<Item, AqlError> {
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

        // ── Control flow ───────────────────────────────────────────────
        Token::Label => {
            check_argc(2)?;
            Item::Instr(Instruction::Label(ident_arg(&tokens[1])?))
        }
        Token::Goto => {
            check_argc(2)?;
            Item::Instr(Instruction::Goto { label: ident_arg(&tokens[1])? })
        }
        Token::If => {
            // IF <q> GOTO <label>  — 4 tokens
            check_argc(4)?;
            let qubit = int_arg(&tokens[1])?;
            if !matches!(tokens[2].token, Token::Goto) {
                return Err(AqlError::Parse {
                    line: tokens[2].line,
                    msg: format!(
                        "expected 'GOTO' after qubit index in 'IF' statement, got '{}'",
                        tokens[2].token.display()
                    ),
                });
            }
            Item::Instr(Instruction::GotoIf { qubit, label: ident_arg(&tokens[3])? })
        }
        Token::IfNot => {
            // IFNOT <q> GOTO <label>  — 4 tokens
            check_argc(4)?;
            let qubit = int_arg(&tokens[1])?;
            if !matches!(tokens[2].token, Token::Goto) {
                return Err(AqlError::Parse {
                    line: tokens[2].line,
                    msg: format!(
                        "expected 'GOTO' after qubit index in 'IFNOT' statement, got '{}'",
                        tokens[2].token.display()
                    ),
                });
            }
            Item::Instr(Instruction::GotoIfNot { qubit, label: ident_arg(&tokens[3])? })
        }

        // ── Custom gate invocation: CALL <name> <q0> <q1> … ───────────
        Token::Call => {
            // Minimum: CALL <name> <q0>  → 3 tokens
            if tokens.len() < 3 {
                return Err(AqlError::Parse {
                    line,
                    msg: format!(
                        "'CALL' expects at least 2 arguments (gate name and qubit index), got {}",
                        tokens.len().saturating_sub(1)
                    ),
                });
            }
            let name = ident_arg(&tokens[1])?.to_lowercase();

            // Collect qubit arguments
            let mut qubits: Vec<usize> = Vec::new();
            for tok in &tokens[2..] {
                qubits.push(int_arg(tok)?);
            }

            // Validate arity against definition (if available in this context)
            if let Some(def) = gate_defs.get(&name) {
                if qubits.len() != def.num_qubits {
                    return Err(AqlError::Validation {
                        msg: format!(
                            "gate '{name}' expects {} qubit argument(s), got {}",
                            def.num_qubits, qubits.len()
                        ),
                    });
                }
            }
            // Note: if gate_defs is empty (gate body context) this is a no-op;
            //       callers that disallow CALL in gate bodies reject it before bounds checks.

            Item::Instr(Instruction::CallGate { name, qubits })
        }

        // ── GATE / END at statement level are handled by the pre-pass ──
        Token::Gate => return Err(AqlError::Parse {
            line,
            msg: "'GATE' definition must not appear nested inside another statement".into(),
        }),
        Token::End => return Err(AqlError::Parse {
            line,
            msg: "'END' without a matching 'GATE' definition".into(),
        }),

        // ── Literals at statement start are errors ─────────────────────
        Token::Int(n) => return Err(AqlError::Parse {
            line,
            msg: format!("unexpected integer '{n}' at start of statement — expected a gate mnemonic"),
        }),
        Token::Float(f) => return Err(AqlError::Parse {
            line,
            msg: format!("unexpected float '{f}' at start of statement — expected a gate mnemonic"),
        }),
        Token::Ident(name) => return Err(AqlError::Parse {
            line,
            msg: format!("unknown mnemonic '{name}' — did you mean a gate, LABEL/GOTO/IF, or CALL?"),
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

/// Require a label name or gate name: an identifier or keyword token.
///
/// The lexer eagerly classifies all known words as keywords, so words like
/// `end`, `gate`, `call`, `goto` would otherwise be rejected as label names.
/// We accept any non-numeric token as a valid identifier (lowercased).
/// Integers and floats are still rejected — they cannot be names.
fn ident_arg(s: &Spanned) -> Result<String, AqlError> {
    match &s.token {
        Token::Ident(name)           => Ok(name.clone()),
        Token::Int(_) | Token::Float(_) => Err(AqlError::Parse {
            line: s.line,
            msg: format!(
                "expected identifier (name), got '{}'",
                s.token.display()
            ),
        }),
        // Keyword used as a label/gate name — accepted, lowercased for
        // case-insensitive label matching.
        kw => Ok(kw.display().to_lowercase()),
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

// ── Validation ────────────────────────────────────────────────────────────

/// Validate that all global qubit indices are in [0, num_qubits).
/// For CallGate, also validates arity against the gate library.
fn validate_bounds(
    body: &[Instruction],
    num_qubits: usize,
    gate_defs: &HashMap<String, GateDef>,
) -> Result<(), AqlError> {
    for instr in body {
        // Validate all referenced global qubit indices
        for q in instr.qubits() {
            if q >= num_qubits {
                return Err(AqlError::Validation {
                    msg: format!(
                        "qubit index {q} is out of range — QREG declared {num_qubits} qubit(s)"
                    ),
                });
            }
        }
        // For CallGate, validate the gate is defined
        if let Instruction::CallGate { name, qubits } = instr {
            if !gate_defs.contains_key(name.as_str()) {
                return Err(AqlError::Validation {
                    msg: format!("call to undefined gate '{name}'"),
                });
            }
            let expected = gate_defs[name.as_str()].num_qubits;
            if qubits.len() != expected {
                return Err(AqlError::Validation {
                    msg: format!(
                        "gate '{name}' expects {expected} qubit argument(s), got {}",
                        qubits.len()
                    ),
                });
            }
        }
    }
    Ok(())
}

/// Validate that every jump target (GOTO/IF/IFNOT) refers to a defined LABEL.
fn validate_labels(body: &[Instruction]) -> Result<(), AqlError> {
    use std::collections::HashSet;
    let defined: HashSet<&str> = body.iter()
        .filter_map(|i| if let Instruction::Label(n) = i { Some(n.as_str()) } else { None })
        .collect();

    for instr in body {
        let target = match instr {
            Instruction::Goto { label }
            | Instruction::GotoIf    { label, .. }
            | Instruction::GotoIfNot { label, .. } => Some(label.as_str()),
            _ => None,
        };
        if let Some(t) = target {
            if !defined.contains(t) {
                return Err(AqlError::Validation {
                    msg: format!("jump to undefined label '{t}'"),
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
        assert!(matches!(compile("QREG 2\nH 0 1"), Err(AqlError::Parse { .. })));
    }

    #[test]
    fn test_error_too_few_args() {
        assert!(matches!(compile("QREG 2\nCNOT 0"), Err(AqlError::Parse { .. })));
    }

    #[test]
    fn test_error_float_as_qubit_index() {
        let result = compile("QREG 2\nH 1.5");
        assert!(result.is_err());
    }

    #[test]
    fn test_error_unknown_mnemonic() {
        assert!(matches!(compile("QREG 2\nFLIB 0"), Err(AqlError::Parse { .. })));
    }

    #[test]
    fn test_error_invalid_character() {
        assert!(matches!(compile("QREG 2\n@foo"), Err(AqlError::Lex { .. })));
    }

    #[test]
    fn test_control_flow_if_goto() {
        let prog = compile(
            "QREG 1\nX 0\nMEASURE 0\nIF 0 GOTO done\nX 0\nLABEL done"
        ).unwrap();
        assert_eq!(prog.instructions.len(), 5);
    }

    #[test]
    fn test_control_flow_ifnot_goto() {
        let prog = compile(
            "QREG 1\nMEASURE 0\nIFNOT 0 GOTO end\nX 0\nLABEL end"
        ).unwrap();
        assert_eq!(prog.instructions.len(), 4);
    }

    #[test]
    fn test_error_undefined_label() {
        assert!(matches!(
            compile("QREG 1\nGOTO nowhere"),
            Err(AqlError::Validation { .. })
        ));
    }

    #[test]
    fn test_error_if_missing_goto_keyword() {
        assert!(matches!(
            compile("QREG 1\nIF 0 LABEL end\nLABEL end"),
            Err(AqlError::Parse { .. })
        ));
    }

    // ── Custom gate definition tests ──────────────────────────────────

    #[test]
    fn test_gate_def_parsed() {
        let prog = compile(
            "GATE bell 2\n  H 0\n  CNOT 0 1\nEND\nQREG 2\nCALL bell 0 1\nMEASURE_ALL"
        ).unwrap();
        assert!(prog.gate_defs.contains_key("bell"));
        let def = &prog.gate_defs["bell"];
        assert_eq!(def.num_qubits, 2);
        assert_eq!(def.body.len(), 2);
    }

    #[test]
    fn test_gate_before_qreg() {
        // GATE definition before QREG is valid
        let prog = compile(
            "GATE flip 1\n  X 0\nEND\nQREG 2\nCALL flip 1"
        ).unwrap();
        assert!(prog.gate_defs.contains_key("flip"));
    }

    #[test]
    fn test_call_gate_in_instructions() {
        let prog = compile(
            "GATE bell 2\n  H 0\n  CNOT 0 1\nEND\nQREG 4\nCALL bell 0 1\nCALL bell 2 3\nMEASURE_ALL"
        ).unwrap();
        assert_eq!(prog.instructions.len(), 3); // 2 CALL + 1 MEASURE_ALL
        assert_eq!(prog.gate_count, 2);         // 2 CallGate (counted as gates)
        assert!(matches!(&prog.instructions[0], Instruction::CallGate { name, qubits }
            if name == "bell" && qubits == &[0, 1]));
        assert!(matches!(&prog.instructions[1], Instruction::CallGate { name, qubits }
            if name == "bell" && qubits == &[2, 3]));
    }

    #[test]
    fn test_gate_name_case_insensitive() {
        // Gate names are lowercased during parsing
        let prog = compile(
            "GATE MyGate 1\n  H 0\nEND\nQREG 1\nCALL mygate 0"
        ).unwrap();
        assert!(prog.gate_defs.contains_key("mygate"));
    }

    #[test]
    fn test_multiple_gate_defs() {
        let prog = compile(
            "GATE prep 1\n  H 0\nEND\nGATE entangle 2\n  CNOT 0 1\nEND\n\
             QREG 3\nCALL prep 0\nCALL entangle 0 1"
        ).unwrap();
        assert_eq!(prog.gate_defs.len(), 2);
        assert!(prog.gate_defs.contains_key("prep"));
        assert!(prog.gate_defs.contains_key("entangle"));
    }

    #[test]
    fn test_error_gate_no_end() {
        assert!(matches!(
            compile("GATE broken 1\n  H 0\nQREG 1"),
            Err(AqlError::Parse { .. })
        ));
    }

    #[test]
    fn test_error_gate_duplicate() {
        assert!(matches!(
            compile("GATE f 1\n  H 0\nEND\nGATE f 1\n  X 0\nEND\nQREG 1\nCALL f 0"),
            Err(AqlError::Validation { .. })
        ));
    }

    #[test]
    fn test_error_call_undefined_gate() {
        assert!(matches!(
            compile("QREG 1\nCALL nosuchgate 0"),
            Err(AqlError::Validation { .. })
        ));
    }

    #[test]
    fn test_error_call_wrong_arity() {
        assert!(matches!(
            compile("GATE f 2\n  CNOT 0 1\nEND\nQREG 3\nCALL f 0"),
            Err(AqlError::Validation { .. })
        ));
    }

    #[test]
    fn test_error_gate_body_has_measure() {
        assert!(matches!(
            compile("GATE bad 1\n  MEASURE 0\nEND\nQREG 1"),
            Err(AqlError::Parse { .. })
        ));
    }

    #[test]
    fn test_error_gate_body_has_control_flow() {
        assert!(matches!(
            compile("GATE bad 1\n  LABEL x\nEND\nQREG 1"),
            Err(AqlError::Parse { .. })
        ));
    }

    #[test]
    fn test_error_gate_body_qubit_out_of_range() {
        // Gate has 1 qubit, body references qubit 1
        assert!(matches!(
            compile("GATE bad 1\n  CNOT 0 1\nEND\nQREG 2"),
            Err(AqlError::Validation { .. })
        ));
    }

    #[test]
    fn test_error_end_without_gate() {
        assert!(matches!(
            compile("QREG 1\nEND"),
            Err(AqlError::Parse { .. })
        ));
    }

    #[test]
    fn test_gate_def_display() {
        // GateDef Display produces round-trippable AQL
        let prog = compile(
            "GATE bell 2\n  H 0\n  CNOT 0 1\nEND\nQREG 2\nCALL bell 0 1"
        ).unwrap();
        let def = &prog.gate_defs["bell"];
        let s = def.to_string();
        assert!(s.contains("GATE bell 2"));
        assert!(s.contains("H 0"));
        assert!(s.contains("CNOT 0 1"));
        assert!(s.contains("END"));
    }
}
