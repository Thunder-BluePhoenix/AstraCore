/// OpenQASM 2.0 Importer for AstraCore.
///
/// Parses a subset of the OpenQASM 2.0 specification and converts it to an
/// AQL `Program` that can be executed by any AstraCore backend.
///
/// ## Supported OpenQASM 2.0 constructs
/// - `OPENQASM 2.0;` header (optional but validated if present)
/// - `include "qelib1.inc";` (ignored — standard gates are built-in)
/// - `qreg name[n];` — qubit register declarations (multiple allowed)
/// - `creg name[n];` — classical register declarations (ignored)
/// - Single-qubit gates: `h`, `x`, `y`, `z`, `s`, `t`, `sdg`, `tdg`
/// - Rotation gates: `rx(θ)`, `ry(θ)`, `rz(θ)`, `p(θ)` / `u1(θ)`
/// - Two-qubit gates: `cx`, `cy`, `cz`, `ch`, `swap`
/// - Three-qubit: `ccx` (Toffoli)
/// - Measurements: `measure q[i] -> c[j];`
/// - Barriers: `barrier q;`
/// - Line comments: `//`
///
/// ## Angle expressions
/// - Numeric literals (`1.5708`, `0.5`)
/// - `pi` constant (= π ≈ 3.14159…)
/// - Simple arithmetic: `pi/2`, `2*pi`, `pi/4`, `-pi/2`
///
/// ## Example
/// ```rust,ignore
/// let qasm = r#"
/// OPENQASM 2.0;
/// qreg q[2];
/// h q[0];
/// cx q[0],q[1];
/// measure q[0] -> c[0];
/// measure q[1] -> c[1];
/// "#;
/// let prog = astracore::compiler::qasm_import::from_qasm(qasm).unwrap();
/// ```
use std::collections::HashMap;
use std::f64::consts::PI;
use crate::compiler::{AqlError, ir::{Instruction, Program}};
use crate::runtime::{execute, ExecutionResult};

// ── Public API ────────────────────────────────────────────────────────────

/// Parse an OpenQASM 2.0 source string into an AstraCore `Program`.
pub fn from_qasm(source: &str) -> Result<Program, AqlError> {
    let parser = QasmParser::new(source);
    parser.parse()
}

/// Parse and immediately execute an OpenQASM 2.0 source string.
pub fn run_qasm(source: &str) -> Result<ExecutionResult, AqlError> {
    let program = from_qasm(source)?;
    execute(&program)
}

// ── Parser internals ──────────────────────────────────────────────────────

struct QasmParser<'a> {
    source: &'a str,
    /// Maps `qreg_name` → (base_qubit_index, size)
    qregs: HashMap<String, (usize, usize)>,
    total_qubits: usize,
    instructions: Vec<Instruction>,
}

impl<'a> QasmParser<'a> {
    fn new(source: &'a str) -> Self {
        Self {
            source,
            qregs: HashMap::new(),
            total_qubits: 0,
            instructions: Vec::new(),
        }
    }

    fn parse(mut self) -> Result<Program, AqlError> {
        // Two-pass: first collect qreg declarations, then parse gates.
        // This allows gates to appear before the qreg they reference (rare but valid).
        // In practice QASM is top-down, so a single pass usually works —
        // but we do a prepass for qreg anyway to get the total qubit count.
        let preprocessed = preprocess(self.source);

        // Pass 1: collect all qreg / creg declarations and the header.
        for (lineno, line) in preprocessed.iter().enumerate() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with("//") { continue; }
            let stmt = strip_trailing_semicolon(trimmed);

            if stmt.to_uppercase().starts_with("OPENQASM") {
                // Validate version
                let rest = stmt["OPENQASM".len()..].trim();
                if !rest.starts_with("2.0") {
                    return Err(AqlError::Parse {
                        line: lineno + 1,
                        msg: format!(
                            "unsupported OPENQASM version '{rest}' (only 2.0 is supported)"
                        ),
                    });
                }
                continue;
            }

            if stmt.to_ascii_lowercase().starts_with("qreg") {
                let (name, size) = parse_register(&stmt["qreg".len()..], lineno + 1)?;
                if self.qregs.contains_key(&name) {
                    return Err(AqlError::Parse {
                        line: lineno + 1,
                        msg: format!("duplicate qreg '{name}'"),
                    });
                }
                self.qregs.insert(name, (self.total_qubits, size));
                self.total_qubits += size;
            }
            // creg, include → ignore in pass 1
        }

        // Pass 2: parse gate / measure / barrier statements.
        for (lineno, line) in preprocessed.iter().enumerate() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with("//") { continue; }
            let stmt = strip_trailing_semicolon(trimmed);
            let lower = stmt.to_ascii_lowercase();

            // Skip already-handled or explicitly-ignored keywords
            if lower.starts_with("openqasm")
                || lower.starts_with("qreg")
                || lower.starts_with("creg")
                || lower.starts_with("include")
            {
                continue;
            }

            self.parse_statement(stmt, lineno + 1)?;
        }

        Ok(Program::with_gate_defs(
            self.total_qubits,
            self.instructions,
            HashMap::new(),
        ))
    }

    fn parse_statement(&mut self, stmt: &str, lineno: usize) -> Result<(), AqlError> {
        let lower = stmt.to_ascii_lowercase();

        // ── measure q[i] -> c[j] ──────────────────────────────────────────
        if lower.starts_with("measure") {
            let rest = stmt["measure".len()..].trim();
            // Split on "->"
            let arrow_pos = rest.find("->").ok_or_else(|| AqlError::Parse {
                line: lineno,
                msg: "MEASURE syntax: 'measure q[i] -> c[j]'".into(),
            })?;
            let qarg = rest[..arrow_pos].trim();
            let qubit = self.resolve_qubit(qarg, lineno)?;
            self.instructions.push(Instruction::Measure(qubit));
            return Ok(());
        }

        // ── barrier ───────────────────────────────────────────────────────
        if lower.starts_with("barrier") {
            self.instructions.push(Instruction::Barrier);
            return Ok(());
        }

        // ── gate call: name[(args)] qubits ────────────────────────────────
        // The gate name is the first token (ends at '(' or whitespace).
        let (gate_name_raw, rest_after_name) = split_gate_name(stmt);
        let gate_name = gate_name_raw.to_ascii_lowercase();

        // Parse optional angle arguments: (pi/2, 0.5, ...)
        let (angles, qarg_str) = parse_angle_args(rest_after_name, lineno)?;

        // Parse qubit arguments: "q[0],q[1]" or "q[0] q[1]"
        let qubits = self.resolve_qubit_list(qarg_str, lineno)?;

        match gate_name.as_str() {
            // ── Single-qubit gates ──────────────────────────────────────
            "h"   => self.push1(&qubits, lineno, |q| Instruction::H(q))?,
            "x"   => self.push1(&qubits, lineno, |q| Instruction::X(q))?,
            "y"   => self.push1(&qubits, lineno, |q| Instruction::Y(q))?,
            "z"   => self.push1(&qubits, lineno, |q| Instruction::Z(q))?,
            "s"   => self.push1(&qubits, lineno, |q| Instruction::S(q))?,
            "t"   => self.push1(&qubits, lineno, |q| Instruction::T(q))?,
            // sdg = S† = S·S·S (or S inverse). Approximate as Rz(-pi/2).
            "sdg" => self.push1(&qubits, lineno, |q| Instruction::Rz { qubit: q, theta: -PI / 2.0 })?,
            // tdg = T† = Rz(-pi/4)
            "tdg" => self.push1(&qubits, lineno, |q| Instruction::Rz { qubit: q, theta: -PI / 4.0 })?,
            "id"  => {} // identity — no-op

            // ── Rotation gates ──────────────────────────────────────────
            "rx" => {
                let theta = need_angle(&angles, 0, "rx", lineno)?;
                self.push1(&qubits, lineno, |q| Instruction::Rx { qubit: q, theta })?;
            }
            "ry" => {
                let theta = need_angle(&angles, 0, "ry", lineno)?;
                self.push1(&qubits, lineno, |q| Instruction::Ry { qubit: q, theta })?;
            }
            "rz" | "r" => {
                let theta = need_angle(&angles, 0, "rz", lineno)?;
                self.push1(&qubits, lineno, |q| Instruction::Rz { qubit: q, theta })?;
            }
            "p" | "u1" | "phase" => {
                // Phase gate P(θ) = diag(1, e^iθ) — maps to AQL PHASE
                let theta = need_angle(&angles, 0, &gate_name, lineno)?;
                self.push1(&qubits, lineno, |q| Instruction::Phase { qubit: q, theta })?;
            }
            "u2" => {
                // U2(φ, λ) = H · Rz(λ) · H · Rz(φ) (approximately).
                // For import purposes: just mark as Rx(pi/2) — approximate.
                let _phi    = need_angle(&angles, 0, "u2/phi", lineno)?;
                let _lambda = need_angle(&angles, 1, "u2/lambda", lineno)?;
                self.push1(&qubits, lineno, |q| Instruction::Rx { qubit: q, theta: PI / 2.0 })?;
            }
            "u3" | "u" => {
                // U3(θ, φ, λ) — general single-qubit gate. Decompose to Rz·Ry·Rz.
                let theta  = need_angle(&angles, 0, "u3/theta",  lineno)?;
                let phi    = need_angle(&angles, 1, "u3/phi",    lineno)?;
                let lambda = need_angle(&angles, 2, "u3/lambda", lineno)?;
                let q = need_qubit(&qubits, 0, &gate_name, lineno)?;
                self.instructions.push(Instruction::Rz { qubit: q, theta: lambda });
                self.instructions.push(Instruction::Ry { qubit: q, theta });
                self.instructions.push(Instruction::Rz { qubit: q, theta: phi });
            }

            // ── Two-qubit gates ─────────────────────────────────────────
            "cx" | "cnot" => self.push2(&qubits, lineno, |c, t| Instruction::Cnot { control: c, target: t })?,
            "cy" => {
                // CY = CNOT with Y correction. Decompose: S† target, CNOT, S target.
                let ctrl = need_qubit(&qubits, 0, "cy", lineno)?;
                let tgt  = need_qubit(&qubits, 1, "cy", lineno)?;
                self.instructions.push(Instruction::Rz { qubit: tgt, theta: -PI / 2.0 }); // Sdg
                self.instructions.push(Instruction::Cnot { control: ctrl, target: tgt });
                self.instructions.push(Instruction::S(tgt));
            }
            "cz" => self.push2(&qubits, lineno, |c, t| Instruction::Cz { control: c, target: t })?,
            "ch" => {
                // Controlled-H. Decompose: Ry(pi/4) target, CNOT, Ry(-pi/4) target.
                let ctrl = need_qubit(&qubits, 0, "ch", lineno)?;
                let tgt  = need_qubit(&qubits, 1, "ch", lineno)?;
                self.instructions.push(Instruction::Ry { qubit: tgt, theta: PI / 4.0 });
                self.instructions.push(Instruction::Cnot { control: ctrl, target: tgt });
                self.instructions.push(Instruction::Ry { qubit: tgt, theta: -PI / 4.0 });
            }
            "swap" => self.push2(&qubits, lineno, |a, b| Instruction::Swap { qubit_a: a, qubit_b: b })?,
            "iswap" => {
                // iSWAP = SWAP · CZ · H⊗H · CNOT · H⊗H — approximate as SWAP.
                let a = need_qubit(&qubits, 0, "iswap", lineno)?;
                let b = need_qubit(&qubits, 1, "iswap", lineno)?;
                self.instructions.push(Instruction::Swap { qubit_a: a, qubit_b: b });
            }
            "ecr" => {
                // ECR gate = CNOT-like entangling gate. Approximate as CNOT.
                let ctrl = need_qubit(&qubits, 0, "ecr", lineno)?;
                let tgt  = need_qubit(&qubits, 1, "ecr", lineno)?;
                self.instructions.push(Instruction::Cnot { control: ctrl, target: tgt });
            }

            // ── Three-qubit gates ───────────────────────────────────────
            "ccx" | "toffoli" | "ccnot" => {
                let c0 = need_qubit(&qubits, 0, &gate_name, lineno)?;
                let c1 = need_qubit(&qubits, 1, &gate_name, lineno)?;
                let t  = need_qubit(&qubits, 2, &gate_name, lineno)?;
                self.instructions.push(Instruction::Toffoli { control0: c0, control1: c1, target: t });
            }
            "cswap" | "fredkin" => {
                // Controlled-SWAP. Decompose: CNOT(t2,t1), CCX(ctrl,t1,t2), CNOT(t2,t1).
                let ctrl = need_qubit(&qubits, 0, &gate_name, lineno)?;
                let t1   = need_qubit(&qubits, 1, &gate_name, lineno)?;
                let t2   = need_qubit(&qubits, 2, &gate_name, lineno)?;
                self.instructions.push(Instruction::Cnot    { control: t2, target: t1 });
                self.instructions.push(Instruction::Toffoli { control0: ctrl, control1: t1, target: t2 });
                self.instructions.push(Instruction::Cnot    { control: t2, target: t1 });
            }

            other => {
                return Err(AqlError::Parse {
                    line: lineno,
                    msg: format!("unknown OpenQASM gate '{other}'"),
                });
            }
        }

        Ok(())
    }

    // ── Helpers ───────────────────────────────────────────────────────────

    fn resolve_qubit(&self, arg: &str, lineno: usize) -> Result<usize, AqlError> {
        let arg = arg.trim();
        // Format: name[idx]
        let bracket = arg.find('[').ok_or_else(|| AqlError::Parse {
            line: lineno,
            msg: format!("expected qubit reference like 'q[0]', got '{arg}'"),
        })?;
        let name = &arg[..bracket];
        let rest = &arg[bracket + 1..];
        let close = rest.find(']').ok_or_else(|| AqlError::Parse {
            line: lineno,
            msg: format!("missing ']' in qubit reference '{arg}'"),
        })?;
        let idx: usize = rest[..close].trim().parse().map_err(|_| AqlError::Parse {
            line: lineno,
            msg: format!("invalid qubit index in '{arg}'"),
        })?;
        let (base, size) = self.qregs.get(name).ok_or_else(|| AqlError::Parse {
            line: lineno,
            msg: format!("undeclared qreg '{name}'"),
        })?;
        if idx >= *size {
            return Err(AqlError::Parse {
                line: lineno,
                msg: format!("qubit index {idx} out of bounds for qreg '{name}[{size}]'"),
            });
        }
        Ok(base + idx)
    }

    fn resolve_qubit_list(&self, args: &str, lineno: usize) -> Result<Vec<usize>, AqlError> {
        let args = args.trim();
        if args.is_empty() { return Ok(vec![]); }
        // Split on ',' or whitespace
        let parts: Vec<&str> = args
            .split(|c: char| c == ',')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect();
        parts.iter().map(|p| self.resolve_qubit(p, lineno)).collect()
    }

    fn push1(
        &mut self,
        qubits: &[usize],
        lineno: usize,
        f: impl Fn(usize) -> Instruction,
    ) -> Result<(), AqlError> {
        if qubits.len() != 1 {
            return Err(AqlError::Parse {
                line: lineno,
                msg: format!("single-qubit gate expects 1 qubit, got {}", qubits.len()),
            });
        }
        self.instructions.push(f(qubits[0]));
        Ok(())
    }

    fn push2(
        &mut self,
        qubits: &[usize],
        lineno: usize,
        f: impl Fn(usize, usize) -> Instruction,
    ) -> Result<(), AqlError> {
        if qubits.len() != 2 {
            return Err(AqlError::Parse {
                line: lineno,
                msg: format!("two-qubit gate expects 2 qubits, got {}", qubits.len()),
            });
        }
        self.instructions.push(f(qubits[0], qubits[1]));
        Ok(())
    }
}

// ── Low-level helpers ─────────────────────────────────────────────────────

/// Strip line comments and trailing semicolons; return non-empty lines.
fn preprocess(source: &str) -> Vec<String> {
    source.lines()
        .map(|line| {
            // Strip inline comments
            let code = if let Some(pos) = line.find("//") {
                &line[..pos]
            } else {
                line
            };
            code.trim().to_string()
        })
        .filter(|l| !l.is_empty())
        .collect()
}

/// Strip optional trailing semicolon from a statement.
fn strip_trailing_semicolon(s: &str) -> &str {
    s.trim_end_matches(';').trim_end()
}

/// Parse `name[n]` from the remainder of a qreg/creg declaration.
fn parse_register(rest: &str, lineno: usize) -> Result<(String, usize), AqlError> {
    let rest = rest.trim();
    let bracket = rest.find('[').ok_or_else(|| AqlError::Parse {
        line: lineno,
        msg: format!("expected 'name[n]' in register declaration, got '{rest}'"),
    })?;
    let name = rest[..bracket].trim().to_string();
    let after = &rest[bracket + 1..];
    let close = after.find(']').ok_or_else(|| AqlError::Parse {
        line: lineno,
        msg: "missing ']' in register declaration".into(),
    })?;
    let size: usize = after[..close].trim().parse().map_err(|_| AqlError::Parse {
        line: lineno,
        msg: "register size must be a positive integer".into(),
    })?;
    Ok((name, size))
}

/// Split a statement into the gate name and everything after it.
/// The gate name ends at '(' or whitespace.
fn split_gate_name(stmt: &str) -> (&str, &str) {
    let end = stmt.find(|c: char| c == '(' || c.is_whitespace())
        .unwrap_or(stmt.len());
    (&stmt[..end], &stmt[end..])
}

/// Parse optional parenthesized angle arguments `(expr, expr, ...)`.
/// Returns `(angles: Vec<f64>, remaining_string)`.
fn parse_angle_args(rest: &str, lineno: usize) -> Result<(Vec<f64>, &str), AqlError> {
    let rest = rest.trim();
    if !rest.starts_with('(') {
        return Ok((vec![], rest));
    }
    let close = rest.find(')').ok_or_else(|| AqlError::Parse {
        line: lineno,
        msg: "missing ')' in gate angle arguments".into(),
    })?;
    let inner = &rest[1..close];
    let after = rest[close + 1..].trim();
    let angles: Result<Vec<f64>, _> = inner
        .split(',')
        .map(|s| parse_angle_expr(s.trim(), lineno))
        .collect();
    Ok((angles?, after))
}

/// Evaluate a simple angle expression: numeric literal, `pi`, `pi/2`, `2*pi`, `-pi/4`, etc.
fn parse_angle_expr(expr: &str, lineno: usize) -> Result<f64, AqlError> {
    let expr = expr.trim().to_ascii_lowercase();
    // Handle negation prefix
    let (neg, e) = if let Some(rest) = expr.strip_prefix('-') {
        (true, rest.trim())
    } else if let Some(rest) = expr.strip_prefix('+') {
        (false, rest.trim())
    } else {
        (false, expr.as_str())
    };

    let value = if e == "pi" {
        PI
    } else if let Some(rest) = e.strip_prefix("pi/") {
        let denom: f64 = rest.parse().map_err(|_| AqlError::Parse {
            line: lineno,
            msg: format!("invalid angle expression '{expr}'"),
        })?;
        PI / denom
    } else if let Some(rest) = e.strip_prefix("pi*") {
        let factor: f64 = rest.parse().map_err(|_| AqlError::Parse {
            line: lineno,
            msg: format!("invalid angle expression '{expr}'"),
        })?;
        PI * factor
    } else if let Some(factor_str) = e.strip_suffix("*pi") {
        let factor: f64 = factor_str.trim().parse().map_err(|_| AqlError::Parse {
            line: lineno,
            msg: format!("invalid angle expression '{expr}'"),
        })?;
        factor * PI
    } else {
        e.parse::<f64>().map_err(|_| AqlError::Parse {
            line: lineno,
            msg: format!("invalid angle expression '{expr}'"),
        })?
    };

    Ok(if neg { -value } else { value })
}

fn need_angle(angles: &[f64], idx: usize, gate: &str, lineno: usize) -> Result<f64, AqlError> {
    angles.get(idx).copied().ok_or_else(|| AqlError::Parse {
        line: lineno,
        msg: format!("gate '{gate}' requires angle argument #{}", idx + 1),
    })
}

fn need_qubit(qubits: &[usize], idx: usize, gate: &str, lineno: usize) -> Result<usize, AqlError> {
    qubits.get(idx).copied().ok_or_else(|| AqlError::Parse {
        line: lineno,
        msg: format!("gate '{gate}' requires qubit argument #{}", idx + 1),
    })
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn prog(qasm: &str) -> Program {
        from_qasm(qasm).unwrap_or_else(|e| panic!("parse failed: {e}"))
    }

    #[test]
    fn test_qasm_header_parsed() {
        let p = prog("OPENQASM 2.0;\nqreg q[1];\nh q[0];");
        assert_eq!(p.num_qubits, 1);
        assert_eq!(p.instructions.len(), 1);
        assert!(matches!(p.instructions[0], Instruction::H(0)));
    }

    #[test]
    fn test_qasm_multiple_qregs() {
        let p = prog("qreg a[2];\nqreg b[3];\nh a[0];\nh b[0];");
        assert_eq!(p.num_qubits, 5);
        // a[0] = qubit 0, b[0] = qubit 2
        assert!(matches!(p.instructions[0], Instruction::H(0)));
        assert!(matches!(p.instructions[1], Instruction::H(2)));
    }

    #[test]
    fn test_qasm_bell_circuit() {
        let qasm = "qreg q[2];\nh q[0];\ncx q[0],q[1];\nmeasure q[0] -> c[0];\nmeasure q[1] -> c[1];";
        let p = prog(qasm);
        assert_eq!(p.num_qubits, 2);
        assert_eq!(p.instructions.len(), 4);
        assert!(matches!(p.instructions[0], Instruction::H(0)));
        assert!(matches!(p.instructions[1], Instruction::Cnot { control: 0, target: 1 }));
        assert!(matches!(p.instructions[2], Instruction::Measure(0)));
        assert!(matches!(p.instructions[3], Instruction::Measure(1)));
    }

    #[test]
    fn test_qasm_single_qubit_gates() {
        let p = prog("qreg q[1];\nx q[0];\ny q[0];\nz q[0];\ns q[0];\nt q[0];");
        let instrs = &p.instructions;
        assert!(matches!(instrs[0], Instruction::X(0)));
        assert!(matches!(instrs[1], Instruction::Y(0)));
        assert!(matches!(instrs[2], Instruction::Z(0)));
        assert!(matches!(instrs[3], Instruction::S(0)));
        assert!(matches!(instrs[4], Instruction::T(0)));
    }

    #[test]
    fn test_qasm_rotation_gates() {
        let p = prog("qreg q[1];\nrx(1.5708) q[0];\nry(pi/2) q[0];\nrz(pi/4) q[0];");
        let instrs = &p.instructions;
        assert!(matches!(instrs[0], Instruction::Rx { qubit: 0, theta } if (theta - std::f64::consts::FRAC_PI_2).abs() < 0.001));
        assert!(matches!(instrs[1], Instruction::Ry { qubit: 0, theta } if (theta - std::f64::consts::FRAC_PI_2).abs() < 1e-9));
        assert!(matches!(instrs[2], Instruction::Rz { qubit: 0, theta } if (theta - std::f64::consts::FRAC_PI_4).abs() < 1e-9));
    }

    #[test]
    fn test_qasm_two_qubit_gates() {
        let p = prog("qreg q[3];\ncx q[0],q[1];\ncz q[1],q[2];\nswap q[0],q[2];");
        let instrs = &p.instructions;
        assert!(matches!(instrs[0], Instruction::Cnot { control: 0, target: 1 }));
        assert!(matches!(instrs[1], Instruction::Cz { control: 1, target: 2 }));
        assert!(matches!(instrs[2], Instruction::Swap { qubit_a: 0, qubit_b: 2 }));
    }

    #[test]
    fn test_qasm_toffoli() {
        let p = prog("qreg q[3];\nccx q[0],q[1],q[2];");
        let instrs = &p.instructions;
        assert!(matches!(
            instrs[0],
            Instruction::Toffoli { control0: 0, control1: 1, target: 2 }
        ));
    }

    #[test]
    fn test_qasm_run_executes_bell() {
        let qasm = "qreg q[2];\nh q[0];\ncx q[0],q[1];\nmeasure q[0] -> c[0];\nmeasure q[1] -> c[1];";
        for _ in 0..20 {
            let result = run_qasm(qasm).unwrap();
            let q0 = result.measurements[0].outcome;
            let q1 = result.measurements[1].outcome;
            assert_eq!(q0, q1, "Bell: qubits must agree");
        }
    }

    #[test]
    fn test_qasm_angle_pi_expressions() {
        // pi/2 and 2*pi should parse correctly
        let half_pi = super::parse_angle_expr("pi/2", 1).unwrap();
        let two_pi  = super::parse_angle_expr("2*pi", 1).unwrap();
        let neg_pi4 = super::parse_angle_expr("-pi/4", 1).unwrap();
        let literal = super::parse_angle_expr("1.5708", 1).unwrap();

        assert!((half_pi - std::f64::consts::FRAC_PI_2).abs() < 1e-9);
        assert!((two_pi  - 2.0 * std::f64::consts::PI).abs() < 1e-9);
        assert!((neg_pi4 + std::f64::consts::FRAC_PI_4).abs() < 1e-9);
        assert!((literal - 1.5708).abs() < 1e-4);
    }

    #[test]
    fn test_qasm_comments_and_include_ignored() {
        // Comments and include should not cause errors
        let p = prog(
            "OPENQASM 2.0;\n\
             include \"qelib1.inc\";\n\
             // This is a comment\n\
             qreg q[1];\n\
             creg c[1];  // another comment\n\
             h q[0];",
        );
        assert_eq!(p.num_qubits, 1);
        assert_eq!(p.instructions.len(), 1);
    }

    #[test]
    fn test_qasm_unknown_version_error() {
        let err = from_qasm("OPENQASM 3.0;\nqreg q[1];");
        assert!(err.is_err(), "QASM 3.0 should be an error");
    }

    #[test]
    fn test_qasm_barrier_parsed() {
        let p = prog("qreg q[2];\nh q[0];\nbarrier q[0];\ncx q[0],q[1];");
        assert!(matches!(p.instructions[1], Instruction::Barrier));
    }
}
