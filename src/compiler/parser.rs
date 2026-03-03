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

    // ── Phase 0: Unroll all REPEAT N … END blocks ─────────────────────────
    let statements = unroll_repeats(statements)?;

    // ── Phase 0b: Resolve named registers (QREG data[4]) ──────────────────
    let statements = resolve_registers(statements)?;

    // ── Phase 0c: Desugar IFMEASURED/IFNOTMEASURED … THEN … END ──────────
    let statements = desugar_if_measured(statements)?;

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
    if num_qubits > 1000 {
        return Err(AqlError::Validation {
            msg: format!(
                "QREG {num_qubits} exceeds maximum of 1000 qubits. \
                 Statevector backend supports ≤30 qubits; MPS supports ≤200; \
                 Clifford supports unlimited Clifford circuits."
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

// ── AQL v2: REPEAT unrolling ──────────────────────────────────────────────

/// Unroll all `REPEAT N … END` blocks by inlining the body N times.
///
/// Nesting is handled correctly: GATE...END and REPEAT...END blocks inside a
/// REPEAT body are collected as-is (their own END is not mistaken for the
/// outer END) via a depth counter.
fn unroll_repeats(
    statements: Vec<Vec<Spanned>>,
) -> Result<Vec<Vec<Spanned>>, AqlError> {
    let mut out: Vec<Vec<Spanned>> = Vec::new();
    let stmts = statements;
    let mut i = 0;
    unroll_slice(&stmts, &mut i, &mut out)?;
    Ok(out)
}

fn unroll_slice(
    stmts: &[Vec<Spanned>],
    i: &mut usize,
    out: &mut Vec<Vec<Spanned>>,
) -> Result<(), AqlError> {
    while *i < stmts.len() {
        let stmt = &stmts[*i];
        match stmt.first().map(|s| &s.token) {
            Some(Token::Repeat) => {
                let line = stmt[0].line;
                if stmt.len() != 2 {
                    return Err(AqlError::Parse {
                        line,
                        msg: format!(
                            "'REPEAT' expects exactly 1 argument (repetition count), got {}",
                            stmt.len().saturating_sub(1)
                        ),
                    });
                }
                let count = match stmt[1].token {
                    Token::Int(n) => n,
                    _ => return Err(AqlError::Parse {
                        line: stmt[1].line,
                        msg: format!(
                            "'REPEAT' expects an integer count, got '{}'",
                            stmt[1].token.display()
                        ),
                    }),
                };
                *i += 1;

                // Collect the body up to the matching END
                let body = collect_repeat_body(stmts, i, line)?;

                // Recursively unroll nested REPEATs inside the body
                let mut unrolled_body: Vec<Vec<Spanned>> = Vec::new();
                let mut j = 0;
                unroll_slice(&body, &mut j, &mut unrolled_body)?;

                // Emit the body `count` times
                for _ in 0..count {
                    out.extend(unrolled_body.clone());
                }
            }
            _ => {
                out.push(stmt.clone());
                *i += 1;
            }
        }
    }
    Ok(())
}

/// Collect statements between the current position and the matching `END`.
/// Tracks nesting depth for nested GATE...END and REPEAT...END pairs.
/// Advances `i` past the closing END.
fn collect_repeat_body(
    stmts: &[Vec<Spanned>],
    i: &mut usize,
    start_line: usize,
) -> Result<Vec<Vec<Spanned>>, AqlError> {
    let mut body: Vec<Vec<Spanned>> = Vec::new();
    let mut depth = 1usize;

    while *i < stmts.len() {
        let stmt = &stmts[*i];
        match stmt.first().map(|s| &s.token) {
            Some(Token::Gate)
            | Some(Token::Repeat)
            | Some(Token::IfMeasured)
            | Some(Token::IfNotMeasured) => {
                depth += 1;
                body.push(stmt.clone());
            }
            Some(Token::End) => {
                depth -= 1;
                if depth == 0 {
                    *i += 1; // consume the closing END
                    return Ok(body);
                }
                body.push(stmt.clone());
            }
            _ => body.push(stmt.clone()),
        }
        *i += 1;
    }

    Err(AqlError::Parse {
        line: start_line,
        msg: "'REPEAT' block has no matching 'END'".into(),
    })
}

// ── AQL v2: Named register resolution ────────────────────────────────────
//
// Transforms `QREG data[4]` + `H data[0]` into `QREG 4` + `H 0`.
//
// Rules:
//   - Multiple `QREG name[n]` declarations accumulate total qubits in order.
//   - Cannot mix `QREG <int>` and `QREG name[n]` in the same program.
//   - All `name[k]` references are resolved to absolute qubit indices.
//   - Duplicate register names and out-of-bounds indices are errors.

fn resolve_registers(
    statements: Vec<Vec<Spanned>>,
) -> Result<Vec<Vec<Spanned>>, AqlError> {
    // Detect whether any named QREGs are present
    let has_named = statements.iter().any(|stmt| {
        matches!(stmt.first().map(|s| &s.token), Some(Token::Qreg))
            && stmt.len() == 2
            && matches!(stmt.get(1).map(|s| &s.token), Some(Token::RegRef { .. }))
    });
    if !has_named {
        return Ok(statements); // fast path — nothing to do
    }

    // Ensure no mixed numeric QREG
    let has_numeric = statements.iter().any(|stmt| {
        matches!(stmt.first().map(|s| &s.token), Some(Token::Qreg))
            && stmt.len() == 2
            && matches!(stmt.get(1).map(|s| &s.token), Some(Token::Int(_)))
    });
    if has_numeric {
        return Err(AqlError::Validation {
            msg: "cannot mix 'QREG <n>' and named 'QREG name[n]' declarations".into(),
        });
    }

    // Build register table: name → (base_qubit, size)
    let mut reg_table: HashMap<String, (usize, usize)> = HashMap::new();
    let mut total = 0usize;
    // Collect in declaration order to assign sequential bases
    for stmt in &statements {
        if matches!(stmt.first().map(|s| &s.token), Some(Token::Qreg)) && stmt.len() == 2 {
            if let Token::RegRef { name, num: size } = &stmt[1].token {
                if *size == 0 {
                    return Err(AqlError::Validation {
                        msg: format!("register '{name}' must have size ≥ 1"),
                    });
                }
                if reg_table.contains_key(name.as_str()) {
                    return Err(AqlError::Validation {
                        msg: format!("duplicate register name '{name}'"),
                    });
                }
                reg_table.insert(name.clone(), (total, *size));
                total += size;
            }
        }
    }

    // Rewrite: collapse all QREG name[n] → single QREG <total>, resolve refs
    let mut out: Vec<Vec<Spanned>> = Vec::with_capacity(statements.len());
    let mut emitted_qreg = false;

    for stmt in statements {
        // Named QREG declaration → emit one collapsed QREG <total>, skip rest
        if matches!(stmt.first().map(|s| &s.token), Some(Token::Qreg))
            && stmt.len() == 2
            && matches!(stmt.get(1).map(|s| &s.token), Some(Token::RegRef { .. }))
        {
            if !emitted_qreg {
                emitted_qreg = true;
                let line = stmt[0].line;
                out.push(vec![
                    Spanned { token: Token::Qreg, line },
                    Spanned { token: Token::Int(total), line },
                ]);
            }
            continue;
        }

        // Resolve all RegRef tokens in this statement
        let resolved: Result<Vec<Spanned>, AqlError> = stmt
            .into_iter()
            .map(|s| {
                if let Token::RegRef { ref name, num } = s.token {
                    let &(base, size) = reg_table.get(name.as_str()).ok_or_else(|| {
                        let known: Vec<_> = reg_table.keys().cloned().collect();
                        AqlError::Validation {
                            msg: format!(
                                "undefined register '{name}' — declared: {}",
                                if known.is_empty() { "none".into() }
                                else { known.join(", ") }
                            ),
                        }
                    })?;
                    if num >= size {
                        return Err(AqlError::Validation {
                            msg: format!(
                                "'{name}[{num}]' is out of bounds \
                                 (register '{name}' has {size} qubit(s), indices 0..{})",
                                size - 1
                            ),
                        });
                    }
                    Ok(Spanned { token: Token::Int(base + num), line: s.line })
                } else {
                    Ok(s)
                }
            })
            .collect();

        out.push(resolved?);
    }

    Ok(out)
}

// ── AQL v2: IFMEASURED / IFNOTMEASURED desugaring ─────────────────────────
//
// `IFMEASURED <q> THEN … END`    → IFNOT q GOTO __im_N ; body ; LABEL __im_N
// `IFNOTMEASURED <q> THEN … END` → IF    q GOTO __im_N ; body ; LABEL __im_N
//
// Generated labels use the prefix `__im_` — user labels must not start with this.

fn desugar_if_measured(
    statements: Vec<Vec<Spanned>>,
) -> Result<Vec<Vec<Spanned>>, AqlError> {
    let has_any = statements.iter().any(|s| {
        matches!(
            s.first().map(|t| &t.token),
            Some(Token::IfMeasured) | Some(Token::IfNotMeasured)
        )
    });
    if !has_any {
        return Ok(statements); // fast path
    }

    let mut out: Vec<Vec<Spanned>> = Vec::new();
    let mut i = 0;
    let mut counter = 0usize;
    desugar_slice(&statements, &mut i, &mut out, &mut counter)?;
    Ok(out)
}

fn desugar_slice(
    stmts: &[Vec<Spanned>],
    i: &mut usize,
    out: &mut Vec<Vec<Spanned>>,
    counter: &mut usize,
) -> Result<(), AqlError> {
    while *i < stmts.len() {
        let stmt = &stmts[*i];
        match stmt.first().map(|s| &s.token) {
            Some(Token::IfMeasured) | Some(Token::IfNotMeasured) => {
                let is_if_measured = matches!(stmt[0].token, Token::IfMeasured);
                let line = stmt[0].line;

                // Syntax: IFMEASURED <q> THEN  (3 tokens)
                if stmt.len() != 3 {
                    return Err(AqlError::Parse {
                        line,
                        msg: format!(
                            "'{}' expects 2 arguments (qubit index and THEN), got {}",
                            stmt[0].token.display(),
                            stmt.len().saturating_sub(1)
                        ),
                    });
                }
                let qubit_tok = stmt[1].clone();
                if !matches!(stmt[2].token, Token::Then) {
                    return Err(AqlError::Parse {
                        line: stmt[2].line,
                        msg: format!(
                            "expected 'THEN' after qubit index in '{}', got '{}'",
                            stmt[0].token.display(),
                            stmt[2].token.display()
                        ),
                    });
                }

                *i += 1;
                // Collect body up to matching END
                let body = collect_repeat_body(stmts, i, line)?;

                // Recursively desugar nested blocks
                let mut body_out: Vec<Vec<Spanned>> = Vec::new();
                let mut j = 0;
                desugar_slice(&body, &mut j, &mut body_out, counter)?;

                // Generate a unique internal label
                let label_name = format!("__im_{counter}");
                *counter += 1;

                // IFMEASURED → branch on NOT (skip body if qubit==0)
                // IFNOTMEASURED → branch on YES (skip body if qubit==1)
                let branch_tok = if is_if_measured { Token::IfNot } else { Token::If };
                out.push(vec![
                    Spanned { token: branch_tok, line },
                    qubit_tok,
                    Spanned { token: Token::Goto, line },
                    Spanned { token: Token::Ident(label_name.clone()), line },
                ]);
                out.extend(body_out);
                out.push(vec![
                    Spanned { token: Token::Label, line },
                    Spanned { token: Token::Ident(label_name), line },
                ]);
            }
            _ => {
                out.push(stmt.clone());
                *i += 1;
            }
        }
    }
    Ok(())
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

        // ── GATE / END / REPEAT / IFMEASURED are handled by pre-passes ─
        Token::Gate => return Err(AqlError::Parse {
            line,
            msg: "'GATE' definition must not appear nested inside another statement".into(),
        }),
        Token::End => return Err(AqlError::Parse {
            line,
            msg: "'END' without a matching 'GATE', 'REPEAT', or 'IFMEASURED' block".into(),
        }),
        Token::Repeat => return Err(AqlError::Parse {
            line,
            msg: "'REPEAT' block was not unrolled — internal compiler error".into(),
        }),
        Token::IfMeasured | Token::IfNotMeasured => return Err(AqlError::Parse {
            line,
            msg: "'IFMEASURED' block was not desugared — internal compiler error".into(),
        }),
        Token::Then => return Err(AqlError::Parse {
            line,
            msg: "'THEN' without a preceding 'IFMEASURED' or 'IFNOTMEASURED'".into(),
        }),
        Token::RegRef { name, num } => return Err(AqlError::Parse {
            line,
            msg: format!(
                "register reference '{name}[{num}]' was not resolved — \
                 did you declare 'QREG {name}[N]'?"
            ),
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

    // ── Named register tests ──────────────────────────────────────────

    #[test]
    fn test_named_register_single() {
        let prog = compile("QREG data[4]\nH data[0]\nMEASURE data[3]").unwrap();
        assert_eq!(prog.num_qubits, 4);
        assert!(matches!(prog.instructions[0], Instruction::H(0)));
        assert!(matches!(prog.instructions[1], Instruction::Measure(3)));
    }

    #[test]
    fn test_named_register_two_registers() {
        // data gets qubits 0-3, ancilla gets qubits 4-5
        let prog = compile(
            "QREG data[4]\nQREG ancilla[2]\nCNOT data[0] ancilla[0]\nMEASURE_ALL"
        ).unwrap();
        assert_eq!(prog.num_qubits, 6);
        assert!(matches!(
            prog.instructions[0],
            Instruction::Cnot { control: 0, target: 4 }
        ));
    }

    #[test]
    fn test_named_register_backwards_compatible() {
        // Unnamed QREG still works exactly as before
        let prog = compile("QREG 3\nH 0\nH 1\nH 2").unwrap();
        assert_eq!(prog.num_qubits, 3);
    }

    #[test]
    fn test_error_named_register_out_of_bounds() {
        // data has 2 qubits (0..1); data[2] is out of range
        assert!(matches!(
            compile("QREG data[2]\nH data[2]"),
            Err(AqlError::Validation { .. })
        ));
    }

    #[test]
    fn test_error_named_register_mixed_with_numeric() {
        assert!(matches!(
            compile("QREG 4\nQREG ancilla[2]\nH 0"),
            Err(AqlError::Validation { .. })
        ));
    }

    #[test]
    fn test_error_named_register_duplicate() {
        assert!(matches!(
            compile("QREG q[2]\nQREG q[2]\nH q[0]"),
            Err(AqlError::Validation { .. })
        ));
    }

    #[test]
    fn test_error_named_register_undefined() {
        // Use a register name that was never declared
        assert!(matches!(
            compile("QREG data[2]\nH ghost[0]"),
            Err(AqlError::Validation { .. })
        ));
    }

    // ── IFMEASURED sugar tests ────────────────────────────────────────

    #[test]
    fn test_ifmeasured_desugars() {
        // IFMEASURED 0 THEN X 1 END should compile without error
        let prog = compile(
            "QREG 2\nH 0\nMEASURE 0\nIFMEASURED 0 THEN\n  X 1\nEND"
        ).unwrap();
        // Desugars to: H 0 / MEASURE 0 / IFNOT 0 GOTO __im_0 / X 1 / LABEL __im_0
        assert_eq!(prog.instructions.len(), 5);
        assert!(matches!(prog.instructions[2], Instruction::GotoIfNot { qubit: 0, .. }));
        assert!(matches!(prog.instructions[3], Instruction::X(1)));
    }

    #[test]
    fn test_ifnotmeasured_desugars() {
        let prog = compile(
            "QREG 2\nMEASURE 0\nIFNOTMEASURED 0 THEN\n  X 1\nEND"
        ).unwrap();
        // Desugars to: MEASURE 0 / IF 0 GOTO __im_0 / X 1 / LABEL __im_0
        assert!(matches!(prog.instructions[1], Instruction::GotoIf { qubit: 0, .. }));
        assert!(matches!(prog.instructions[2], Instruction::X(1)));
    }

    #[test]
    fn test_ifmeasured_with_named_registers() {
        let prog = compile(
            "QREG ctrl[1]\nQREG tgt[1]\nMEASURE ctrl[0]\nIFMEASURED ctrl[0] THEN\n  X tgt[0]\nEND"
        ).unwrap();
        assert_eq!(prog.num_qubits, 2);
        // ctrl[0]=0, tgt[0]=1
        // instructions: MEASURE(0) / GotoIfNot{0,"__im_0"} / X(1) / Label("__im_0")
        assert!(matches!(prog.instructions[1], Instruction::GotoIfNot { qubit: 0, .. }));
        assert!(matches!(prog.instructions[2], Instruction::X(1)));
    }

    #[test]
    fn test_ifmeasured_nested_in_repeat() {
        // REPEAT + IFMEASURED together
        let prog = compile(
            "QREG 1\nMEASURE 0\nREPEAT 2\n  IFMEASURED 0 THEN\n    X 0\n  END\nEND"
        ).unwrap();
        // Each REPEAT iteration expands to: IFNOT 0 GOTO __im_N / X 0 / LABEL __im_N
        // 2 iterations → 1 MEASURE + 2×3 = 7 total
        assert_eq!(prog.instructions.len(), 7);
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
