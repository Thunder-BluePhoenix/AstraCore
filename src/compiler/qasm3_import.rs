/// OpenQASM 3.0 Importer for AstraCore.
///
/// Parses a subset of the OpenQASM 3.0 specification and converts it to an
/// AQL `Program` that can be executed by any AstraCore backend.
///
/// ## Supported OpenQASM 3.0 constructs
/// - `OPENQASM 3;` / `OPENQASM 3.0;` header
/// - `include "stdgates.inc";` (ignored — built-in gates are available)
/// - `qubit[n] name;` / `qubit name;` — qubit register declarations
/// - `bit[n] name;` / `bit name;` — classical register declarations
/// - Single-qubit gates: `h`, `x`, `y`, `z`, `s`, `t`, `id`
/// - Rotation gates: `rx(θ)`, `ry(θ)`, `rz(θ)`, `p(θ)` / `u1(θ)`
/// - Two-qubit: `cx`, `cy`, `cz`, `ch`, `swap`
/// - Three-qubit: `ccx` (Toffoli)
/// - Measurements: `measure q[i] -> c[j];`
/// - Barriers: `barrier ...;`
/// - Conditionals: `if (creg == 1) gate ...;`
/// - For loops: `for i in [start:end] { body }`
/// - Gate definitions: `gate name params { body }`
use std::collections::HashMap;
use std::f64::consts::PI;
use crate::compiler::{AqlError, ir::{GateDef, Instruction, Program}};

// ── Public API ────────────────────────────────────────────────────────────

/// Parse an OpenQASM 3.0 source string into an AstraCore `Program`.
pub fn from_qasm3(source: &str) -> Result<Program, AqlError> {
    Qasm3Parser::new(source).parse()
}

// ── Parser internals ──────────────────────────────────────────────────────

struct Qasm3Parser<'a> {
    source: &'a str,
    /// Maps qreg_name → (base_qubit_index, size)
    qregs: HashMap<String, (usize, usize)>,
    /// Maps creg_name → (base_index, size) — classical registers
    cregs: HashMap<String, (usize, usize)>,
    total_qubits: usize,
    instructions: Vec<Instruction>,
    gate_defs: HashMap<String, GateDef>,
    creg_defs: HashMap<String, usize>,
    if_label_counter: usize,
}

impl<'a> Qasm3Parser<'a> {
    fn new(source: &'a str) -> Self {
        Self {
            source,
            qregs: HashMap::new(),
            cregs: HashMap::new(),
            total_qubits: 0,
            instructions: Vec::new(),
            gate_defs: HashMap::new(),
            creg_defs: HashMap::new(),
            if_label_counter: 0,
        }
    }

    fn parse(mut self) -> Result<Program, AqlError> {
        let lines = preprocess3(self.source);

        // Pass 1: collect qubit/bit/gate declarations
        let mut pass1_lines: Vec<(usize, String)> = Vec::new();
        let mut i = 0;
        while i < lines.len() {
            let (lineno, line) = &lines[i];
            let trimmed = line.trim();
            if trimmed.is_empty() { i += 1; continue; }

            let stmt = strip_semi(trimmed);
            let upper = stmt.to_ascii_uppercase();

            if upper.starts_with("OPENQASM") {
                i += 1; continue; // validated in pass 2
            }
            if upper.starts_with("INCLUDE") {
                i += 1; continue; // ignored
            }
            if upper.starts_with("QUBIT") {
                self.parse_qubit_decl(stmt, *lineno)?;
                i += 1; continue;
            }
            if upper.starts_with("BIT") && !upper.starts_with("BARRIER") {
                self.parse_bit_decl(stmt, *lineno)?;
                i += 1; continue;
            }
            if upper.starts_with("GATE ") || upper.starts_with("GATE\t") {
                // Collect multi-line gate body
                let mut body_lines = vec![(lineno, line.clone())];
                i += 1;
                while i < lines.len() {
                    let (blineno, bline) = &lines[i];
                    body_lines.push((blineno, bline.clone()));
                    i += 1;
                    if bline.trim() == "}" { break; }
                }
                let combined = body_lines.iter().map(|(_, l)| l.as_str()).collect::<Vec<_>>().join("\n");
                self.parse_gate_def(&combined, *lineno)?;
                continue;
            }
            pass1_lines.push((*lineno, line.clone()));
            i += 1;
        }

        if self.total_qubits == 0 {
            return Err(AqlError::Validation {
                msg: "no 'qubit[n] name' declaration found".into(),
            });
        }

        // Pass 2: parse gates/measurements/control flow
        let mut j = 0;
        while j < pass1_lines.len() {
            let (lineno, line) = &pass1_lines[j];
            let trimmed = line.trim();
            if trimmed.is_empty() { j += 1; continue; }

            let stmt = strip_semi(trimmed);
            let upper = stmt.to_ascii_uppercase();

            if upper.starts_with("OPENQASM") {
                let rest = stmt["OPENQASM".len()..].trim();
                if !rest.starts_with("3") {
                    return Err(AqlError::Parse {
                        line: *lineno,
                        msg: format!("unsupported OPENQASM version '{rest}' (expected 3 or 3.0)"),
                    });
                }
                j += 1; continue;
            }
            if upper.starts_with("BARRIER") {
                self.instructions.push(Instruction::Barrier);
                j += 1; continue;
            }
            if upper.starts_with("FOR ") || upper.starts_with("FOR\t") {
                // Collect for loop body
                let mut body_lines = vec![stmt.to_string()];
                if !stmt.contains('{') || !stmt.contains('}') {
                    j += 1;
                    while j < pass1_lines.len() {
                        let (_, bl) = &pass1_lines[j];
                        body_lines.push(bl.trim().to_string());
                        j += 1;
                        if bl.trim() == "}" { break; }
                    }
                }
                let combined = body_lines.join(" ");
                self.parse_for_loop(&combined, *lineno)?;
                j += 1;
                continue;
            }
            if upper.starts_with("IF") {
                self.parse_if_stmt(stmt, *lineno)?;
                j += 1; continue;
            }
            if upper.starts_with("MEASURE") {
                self.parse_measure(stmt, *lineno)?;
                j += 1; continue;
            }
            // Default: gate instruction
            self.parse_gate_stmt(stmt, *lineno)?;
            j += 1;
        }

        Ok(Program::with_cregs(
            self.total_qubits,
            self.instructions,
            self.gate_defs,
            self.creg_defs,
        ))
    }

    // ── Declaration parsers ───────────────────────────────────────────

    fn parse_qubit_decl(&mut self, stmt: &str, line: usize) -> Result<(), AqlError> {
        // qubit[n] name  OR  qubit name  (single qubit)
        let rest = stmt["qubit".len()..].trim();
        let (size, name) = if rest.starts_with('[') {
            let close = rest.find(']').ok_or_else(|| AqlError::Parse {
                line, msg: "expected ']' in qubit[n] declaration".into(),
            })?;
            let n: usize = rest[1..close].trim().parse().map_err(|_| AqlError::Parse {
                line, msg: format!("invalid qubit count in '{stmt}'"),
            })?;
            let name = rest[close + 1..].trim().to_ascii_lowercase();
            (n, name)
        } else {
            // qubit name — single qubit
            (1, rest.to_ascii_lowercase())
        };
        if size == 0 {
            return Err(AqlError::Parse { line, msg: "qubit count must be at least 1".into() });
        }
        let base = self.total_qubits;
        self.qregs.insert(name, (base, size));
        self.total_qubits += size;
        Ok(())
    }

    fn parse_bit_decl(&mut self, stmt: &str, line: usize) -> Result<(), AqlError> {
        // bit[n] name  OR  bit name  (single bit)
        let rest = stmt["bit".len()..].trim();
        let (size, name) = if rest.starts_with('[') {
            let close = rest.find(']').ok_or_else(|| AqlError::Parse {
                line, msg: "expected ']' in bit[n] declaration".into(),
            })?;
            let n: usize = rest[1..close].trim().parse().map_err(|_| AqlError::Parse {
                line, msg: format!("invalid bit count in '{stmt}'"),
            })?;
            let name = rest[close + 1..].trim().to_ascii_lowercase();
            (n, name)
        } else {
            (1, rest.to_ascii_lowercase())
        };
        let base = self.cregs.values().map(|(_, sz)| sz).sum::<usize>();
        self.cregs.insert(name.clone(), (base, size));
        self.creg_defs.insert(name, size);
        Ok(())
    }

    fn parse_gate_def(&mut self, text: &str, line: usize) -> Result<(), AqlError> {
        // gate name params { body }
        // e.g. gate mygate q { h q; }
        let after_gate = text["gate".len()..].trim();
        let brace_open = after_gate.find('{').ok_or_else(|| AqlError::Parse {
            line, msg: "gate definition missing '{'".into(),
        })?;
        let brace_close = after_gate.rfind('}').ok_or_else(|| AqlError::Parse {
            line, msg: "gate definition missing '}'".into(),
        })?;

        let header = after_gate[..brace_open].trim();
        let body_text = &after_gate[brace_open + 1..brace_close];

        // Parse header: name param1,param2,...
        let mut header_parts = header.split_whitespace();
        let gate_name = header_parts.next().unwrap_or("").to_ascii_lowercase();
        let params: Vec<String> = header_parts
            .flat_map(|p| p.split(','))
            .map(|p| p.trim().to_ascii_lowercase())
            .filter(|p| !p.is_empty())
            .collect();

        if gate_name.is_empty() {
            return Err(AqlError::Parse { line, msg: "gate definition has no name".into() });
        }
        let num_qubits = params.len().max(1);

        // Parse gate body — split on both newlines and semicolons to handle
        // single-line bodies like `h a; cx a, b;`
        let mut body: Vec<Instruction> = Vec::new();
        for raw in body_text.split(|c| c == '\n' || c == ';') {
            let g = raw.trim();
            if g.is_empty() || g.starts_with("//") { continue; }
            // Resolve local qubit parameters to indices 0..n-1
            let resolved = resolve_gate_params(g, &params);
            if let Some(instr) = parse_gate_line_local(&resolved, line)? {
                body.push(instr);
            }
        }

        self.gate_defs.insert(gate_name.clone(), GateDef::new(gate_name, num_qubits, body));
        Ok(())
    }

    // ── Statement parsers ─────────────────────────────────────────────

    fn parse_measure(&mut self, stmt: &str, line: usize) -> Result<(), AqlError> {
        // measure q[i] -> c[j]
        let rest = stmt["measure".len()..].trim();
        let arrow = rest.find("->").ok_or_else(|| AqlError::Parse {
            line, msg: "measure statement missing '->'".into(),
        })?;
        let qubit_part = rest[..arrow].trim();
        let creg_part  = rest[arrow + 2..].trim();

        let qubit = self.resolve_qubit(qubit_part, line)?;
        let (creg_name, creg_bit) = parse_reg_ref(creg_part, line)?;
        self.instructions.push(Instruction::MeasureInto { qubit, creg: creg_name, creg_bit });
        Ok(())
    }

    fn parse_if_stmt(&mut self, stmt: &str, line: usize) -> Result<(), AqlError> {
        // if (creg == 1) gate ...
        let rest = stmt[2..].trim(); // strip "if"
        let close_paren = rest.find(')').ok_or_else(|| AqlError::Parse {
            line, msg: "if statement missing ')'".into(),
        })?;
        let condition = rest[..close_paren].trim().trim_start_matches('(');
        let gate_stmt = rest[close_paren + 1..].trim();

        // Parse condition: creg[k] == 1 or creg == 1
        let (creg_name, creg_bit) = parse_condition(condition, line)?;

        // Synthetic skip label
        let skip_label = format!("__qasm3_if_{}", self.if_label_counter);
        self.if_label_counter += 1;

        // Emit: GotoIfNotCreg → gate → LABEL
        self.instructions.push(Instruction::GotoIfNotCreg {
            creg: creg_name,
            bit: creg_bit,
            label: skip_label.clone(),
        });
        self.parse_gate_stmt(strip_semi(gate_stmt), line)?;
        self.instructions.push(Instruction::Label(skip_label));
        Ok(())
    }

    fn parse_for_loop(&mut self, stmt: &str, line: usize) -> Result<(), AqlError> {
        // for i in [start:end] { h q[i]; }
        // Collect range and body
        let rest = if stmt.to_ascii_uppercase().starts_with("FOR") {
            stmt[3..].trim()
        } else {
            stmt
        };

        let in_pos = rest.to_ascii_uppercase().find(" IN ").ok_or_else(|| AqlError::Parse {
            line, msg: "for loop missing 'in'".into(),
        })?;
        let var = rest[..in_pos].trim().to_ascii_lowercase();
        let rest2 = rest[in_pos + 4..].trim();

        // [start:end]
        let bracket_open  = rest2.find('[').ok_or_else(|| AqlError::Parse {
            line, msg: "for loop range missing '['".into(),
        })?;
        let bracket_close = rest2.find(']').ok_or_else(|| AqlError::Parse {
            line, msg: "for loop range missing ']'".into(),
        })?;
        let range_str = &rest2[bracket_open + 1..bracket_close];
        let colon = range_str.find(':').ok_or_else(|| AqlError::Parse {
            line, msg: "for loop range missing ':'".into(),
        })?;
        let start: isize = range_str[..colon].trim().parse().map_err(|_| AqlError::Parse {
            line, msg: "invalid for loop range start".into(),
        })?;
        let end: isize = range_str[colon + 1..].trim().parse().map_err(|_| AqlError::Parse {
            line, msg: "invalid for loop range end".into(),
        })?;

        // Collect body between { }
        let brace_open = rest2.find('{').ok_or_else(|| AqlError::Parse {
            line, msg: "for loop body missing '{'".into(),
        })?;
        let brace_close = rest2.rfind('}').ok_or_else(|| AqlError::Parse {
            line, msg: "for loop body missing '}'".into(),
        })?;
        let body_text = &rest2[brace_open + 1..brace_close];

        // Unroll: for each i in start..=end, substitute and parse body
        for i in start..=end {
            for body_line in body_text.lines() {
                let g = strip_semi(body_line.trim());
                if g.is_empty() || g.starts_with("//") { continue; }
                let substituted = g.replace(&format!("[{var}]"), &format!("[{i}]"))
                                   .replace(&var, &i.to_string());
                self.parse_gate_stmt(&substituted, line)?;
            }
        }
        Ok(())
    }

    fn parse_gate_stmt(&mut self, stmt: &str, line: usize) -> Result<(), AqlError> {
        if stmt.is_empty() { return Ok(()); }
        let stmt_lc = stmt.to_ascii_lowercase();

        // Split: gate_name[angle] qubit_args
        let (gate_part, args_part) = split_gate_and_args(stmt);
        let gate_lc = gate_part.to_ascii_lowercase();
        let args: Vec<&str> = args_part.split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .collect();

        // Parse angle from gate_part if present
        let angle = extract_angle(&gate_part, line);

        match gate_lc.as_str() {
            "h" => {
                let q = self.resolve_qubit(args.first().copied().unwrap_or(""), line)?;
                self.instructions.push(Instruction::H(q));
            }
            "x" => {
                let q = self.resolve_qubit(args.first().copied().unwrap_or(""), line)?;
                self.instructions.push(Instruction::X(q));
            }
            "y" => {
                let q = self.resolve_qubit(args.first().copied().unwrap_or(""), line)?;
                self.instructions.push(Instruction::Y(q));
            }
            "z" => {
                let q = self.resolve_qubit(args.first().copied().unwrap_or(""), line)?;
                self.instructions.push(Instruction::Z(q));
            }
            "s" => {
                let q = self.resolve_qubit(args.first().copied().unwrap_or(""), line)?;
                self.instructions.push(Instruction::S(q));
            }
            "t" => {
                let q = self.resolve_qubit(args.first().copied().unwrap_or(""), line)?;
                self.instructions.push(Instruction::T(q));
            }
            "id" => { /* identity — no-op */ }
            "rx" => {
                let q = self.resolve_qubit(args.first().copied().unwrap_or(""), line)?;
                self.instructions.push(Instruction::Rx { qubit: q, theta: angle.unwrap_or(0.0) });
            }
            "ry" => {
                let q = self.resolve_qubit(args.first().copied().unwrap_or(""), line)?;
                self.instructions.push(Instruction::Ry { qubit: q, theta: angle.unwrap_or(0.0) });
            }
            "rz" | "p" | "u1" => {
                let q = self.resolve_qubit(args.first().copied().unwrap_or(""), line)?;
                self.instructions.push(Instruction::Rz { qubit: q, theta: angle.unwrap_or(0.0) });
            }
            "cx" | "cnot" => {
                if args.len() < 2 {
                    return Err(AqlError::Parse { line, msg: "cx requires 2 qubit arguments".into() });
                }
                let c = self.resolve_qubit(args[0], line)?;
                let t = self.resolve_qubit(args[1], line)?;
                self.instructions.push(Instruction::Cnot { control: c, target: t });
            }
            "cz" => {
                if args.len() < 2 {
                    return Err(AqlError::Parse { line, msg: "cz requires 2 qubit arguments".into() });
                }
                let c = self.resolve_qubit(args[0], line)?;
                let t = self.resolve_qubit(args[1], line)?;
                self.instructions.push(Instruction::Cz { control: c, target: t });
            }
            "swap" => {
                if args.len() < 2 {
                    return Err(AqlError::Parse { line, msg: "swap requires 2 qubit arguments".into() });
                }
                let a = self.resolve_qubit(args[0], line)?;
                let b = self.resolve_qubit(args[1], line)?;
                self.instructions.push(Instruction::Swap { qubit_a: a, qubit_b: b });
            }
            "ccx" | "toffoli" => {
                if args.len() < 3 {
                    return Err(AqlError::Parse { line, msg: "ccx requires 3 qubit arguments".into() });
                }
                let c0 = self.resolve_qubit(args[0], line)?;
                let c1 = self.resolve_qubit(args[1], line)?;
                let t  = self.resolve_qubit(args[2], line)?;
                self.instructions.push(Instruction::Toffoli { control0: c0, control1: c1, target: t });
            }
            "barrier" | "gphase" => { /* ignored */ }
            _ => {
                // Check user-defined gate
                if let Some(def) = self.gate_defs.get(&gate_lc).cloned() {
                    let qubits: Vec<usize> = args.iter()
                        .map(|a| self.resolve_qubit(a, line))
                        .collect::<Result<Vec<_>, _>>()?;
                    if qubits.len() == def.num_qubits {
                        self.instructions.push(Instruction::CallGate {
                            name: gate_lc,
                            qubits,
                        });
                    } else {
                        return Err(AqlError::Validation {
                            msg: format!(
                                "gate '{}' expects {} qubit(s), got {}",
                                gate_lc, def.num_qubits, qubits.len()
                            ),
                        });
                    }
                } else if !stmt_lc.is_empty() {
                    return Err(AqlError::Parse {
                        line,
                        msg: format!("unknown gate or statement '{stmt}'"),
                    });
                }
            }
        }
        Ok(())
    }

    // ── Qubit resolution ─────────────────────────────────────────────

    fn resolve_qubit(&self, s: &str, line: usize) -> Result<usize, AqlError> {
        let s = s.trim();
        // Name[idx] form
        if let Some((reg_name, idx)) = parse_reg_ref_opt(s) {
            let (base, size) = self.qregs.get(&reg_name).copied().ok_or_else(|| AqlError::Validation {
                msg: format!("undefined qubit register '{reg_name}'"),
            })?;
            if idx >= size {
                return Err(AqlError::Validation {
                    msg: format!("qubit '{reg_name}[{idx}]' is out of range (size {size})"),
                });
            }
            return Ok(base + idx);
        }
        // Bare name — single-qubit register
        let name = s.to_ascii_lowercase();
        if let Some(&(base, _)) = self.qregs.get(&name) {
            return Ok(base);
        }
        Err(AqlError::Validation {
            msg: format!("undefined qubit '{s}' at line {line}"),
        })
    }
}

// ── Free helper functions ─────────────────────────────────────────────────

/// Strip a trailing semicolon (and surrounding whitespace) from a statement.
fn strip_semi(s: &str) -> &str {
    s.trim().trim_end_matches(';').trim()
}

/// Preprocess QASM 3 source: strip // comments, join continuation lines,
/// return (1-based lineno, line) pairs.
fn preprocess3(source: &str) -> Vec<(usize, String)> {
    source.lines()
        .enumerate()
        .map(|(i, line)| {
            let without_comment = if let Some(p) = line.find("//") {
                &line[..p]
            } else {
                line
            };
            (i + 1, without_comment.to_string())
        })
        .filter(|(_, l)| !l.trim().is_empty())
        .collect()
}

/// Parse `name[index]` → (name, index). Returns None if no brackets.
fn parse_reg_ref_opt(s: &str) -> Option<(String, usize)> {
    let open = s.find('[')?;
    let close = s.find(']')?;
    let name = s[..open].trim().to_ascii_lowercase();
    let idx: usize = s[open + 1..close].trim().parse().ok()?;
    Some((name, idx))
}

/// Parse `name[index]` → (name, index). Returns error if not parseable.
fn parse_reg_ref(s: &str, line: usize) -> Result<(String, usize), AqlError> {
    parse_reg_ref_opt(s).ok_or_else(|| AqlError::Parse {
        line,
        msg: format!("expected 'name[index]', got '{s}'"),
    })
}

/// Parse condition like `c[0] == 1` or `c == 1` → (creg_name, bit_index).
fn parse_condition(cond: &str, _line: usize) -> Result<(String, usize), AqlError> {
    // Strip `== 1` or `==1` suffix
    let lhs = if let Some(pos) = cond.find("==") {
        cond[..pos].trim()
    } else {
        cond.trim()
    };
    if let Some((name, bit)) = parse_reg_ref_opt(lhs) {
        return Ok((name, bit));
    }
    // Bare register name → bit 0
    Ok((lhs.trim().to_ascii_lowercase(), 0))
}

/// Extract a parenthesised angle expression from a gate name like `rx(pi/2)`.
fn extract_angle(gate_part: &str, _line: usize) -> Option<f64> {
    let open  = gate_part.find('(')?;
    let close = gate_part.rfind(')')?;
    let expr = gate_part[open + 1..close].trim();
    parse_angle_expr(expr)
}

/// Parse angle expressions: `pi/2`, `2*pi`, `pi`, `-pi/2`, numeric literals.
pub(crate) fn parse_angle_expr(expr: &str) -> Option<f64> {
    let expr = expr.trim().to_ascii_lowercase();
    if expr == "pi" { return Some(PI); }
    if expr == "-pi" { return Some(-PI); }
    if expr == "tau" { return Some(2.0 * PI); }

    // Try `pi/n`
    if let Some(rest) = expr.strip_prefix("pi/") {
        if let Ok(d) = rest.trim().parse::<f64>() {
            return Some(PI / d);
        }
    }
    // Try `-pi/n`
    if let Some(rest) = expr.strip_prefix("-pi/") {
        if let Ok(d) = rest.trim().parse::<f64>() {
            return Some(-PI / d);
        }
    }
    // Try `n*pi`
    if let Some(rest) = expr.strip_suffix("*pi") {
        if let Ok(n) = rest.trim().parse::<f64>() {
            return Some(n * PI);
        }
    }
    // Try `pi*n`
    if expr.contains("*pi") {
        let parts: Vec<&str> = expr.splitn(2, "*pi").collect();
        if let Ok(n) = parts[0].trim().parse::<f64>() {
            return Some(n * PI);
        }
    }
    // Plain numeric
    expr.parse::<f64>().ok()
}

/// Split `rx(pi/2) q[0], q[1]` into (`rx(pi/2)`, `q[0], q[1]`).
fn split_gate_and_args(stmt: &str) -> (String, String) {
    let stmt = stmt.trim();
    // If parenthesised angle, find the closing ')' first
    if let Some(_paren_open) = stmt.find('(') {
        if let Some(paren_close) = stmt.find(')') {
            let gate = stmt[..=paren_close].to_string();
            let args = stmt[paren_close + 1..].trim().to_string();
            return (gate, args);
        }
    }
    // Otherwise split at first whitespace
    if let Some(sp) = stmt.find(|c: char| c.is_whitespace()) {
        (stmt[..sp].to_string(), stmt[sp..].trim().to_string())
    } else {
        (stmt.to_string(), String::new())
    }
}

/// Replace gate parameter names (like `q`, `a`, `b`) with numeric indices in gate body.
fn resolve_gate_params(stmt: &str, params: &[String]) -> String {
    let mut result = stmt.to_string();
    for (i, param) in params.iter().enumerate() {
        result = result.replace(param.as_str(), &i.to_string());
    }
    result
}

/// Parse one gate instruction using local numeric qubit indices (0..n-1).
/// Used when parsing gate definition bodies.
fn parse_gate_line_local(stmt: &str, line: usize) -> Result<Option<Instruction>, AqlError> {
    if stmt.is_empty() { return Ok(None); }
    let (gate_part, args_part) = split_gate_and_args(stmt);
    let gate_lc = gate_part.to_ascii_lowercase();
    let angle = extract_angle(&gate_part, line);
    let args: Vec<&str> = args_part.split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect();

    let parse_local = |s: &str| -> Result<usize, AqlError> {
        s.trim().parse::<usize>().map_err(|_| AqlError::Parse {
            line,
            msg: format!("expected integer qubit index in gate body, got '{s}'"),
        })
    };

    let instr = match gate_lc.as_str() {
        "h"    => Instruction::H(parse_local(args.first().copied().unwrap_or(""))?),
        "x"    => Instruction::X(parse_local(args.first().copied().unwrap_or(""))?),
        "y"    => Instruction::Y(parse_local(args.first().copied().unwrap_or(""))?),
        "z"    => Instruction::Z(parse_local(args.first().copied().unwrap_or(""))?),
        "s"    => Instruction::S(parse_local(args.first().copied().unwrap_or(""))?),
        "t"    => Instruction::T(parse_local(args.first().copied().unwrap_or(""))?),
        "id"   => return Ok(None),
        "rx"   => Instruction::Rx { qubit: parse_local(args.first().copied().unwrap_or(""))?, theta: angle.unwrap_or(0.0) },
        "ry"   => Instruction::Ry { qubit: parse_local(args.first().copied().unwrap_or(""))?, theta: angle.unwrap_or(0.0) },
        "rz" | "p" | "u1" => Instruction::Rz { qubit: parse_local(args.first().copied().unwrap_or(""))?, theta: angle.unwrap_or(0.0) },
        "cx" | "cnot" => {
            if args.len() < 2 { return Err(AqlError::Parse { line, msg: "cx needs 2 args".into() }); }
            Instruction::Cnot { control: parse_local(args[0])?, target: parse_local(args[1])? }
        }
        "cz" => {
            if args.len() < 2 { return Err(AqlError::Parse { line, msg: "cz needs 2 args".into() }); }
            Instruction::Cz { control: parse_local(args[0])?, target: parse_local(args[1])? }
        }
        "swap" => {
            if args.len() < 2 { return Err(AqlError::Parse { line, msg: "swap needs 2 args".into() }); }
            Instruction::Swap { qubit_a: parse_local(args[0])?, qubit_b: parse_local(args[1])? }
        }
        "ccx" | "toffoli" => {
            if args.len() < 3 { return Err(AqlError::Parse { line, msg: "ccx needs 3 args".into() }); }
            Instruction::Toffoli { control0: parse_local(args[0])?, control1: parse_local(args[1])?, target: parse_local(args[2])? }
        }
        "barrier" | "gphase" => return Ok(None),
        _ => return Err(AqlError::Parse {
            line,
            msg: format!("unknown gate '{gate_lc}' in gate body"),
        }),
    };
    Ok(Some(instr))
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::ir::Instruction;

    fn parse(src: &str) -> Program {
        from_qasm3(src).unwrap_or_else(|e| panic!("parse failed: {e}"))
    }

    #[test]
    fn test_qasm3_version_header() {
        let prog = parse("OPENQASM 3;\nqubit[1] q;\n");
        assert_eq!(prog.num_qubits, 1);
    }

    #[test]
    fn test_qasm3_qubit_declaration() {
        let prog = parse("OPENQASM 3;\nqubit[2] q;\n");
        assert_eq!(prog.num_qubits, 2);
    }

    #[test]
    fn test_qasm3_bit_declaration() {
        let prog = parse("OPENQASM 3;\nqubit[2] q;\nbit[2] c;\n");
        assert_eq!(prog.creg_defs.get("c"), Some(&2));
    }

    #[test]
    fn test_qasm3_h_gate() {
        let prog = parse("OPENQASM 3;\nqubit[1] q;\nh q[0];\n");
        assert_eq!(prog.instructions.len(), 1);
        assert!(matches!(prog.instructions[0], Instruction::H(0)));
    }

    #[test]
    fn test_qasm3_cx_gate() {
        let prog = parse("OPENQASM 3;\nqubit[2] q;\ncx q[0], q[1];\n");
        assert_eq!(prog.instructions.len(), 1);
        assert!(matches!(prog.instructions[0], Instruction::Cnot { control: 0, target: 1 }));
    }

    #[test]
    fn test_qasm3_measure_into() {
        let prog = parse("OPENQASM 3;\nqubit[1] q;\nbit[1] c;\nmeasure q[0] -> c[0];\n");
        assert_eq!(prog.instructions.len(), 1);
        assert!(matches!(
            &prog.instructions[0],
            Instruction::MeasureInto { qubit: 0, creg, creg_bit: 0 } if creg == "c"
        ));
    }

    #[test]
    fn test_qasm3_if_conditional() {
        // if (c[0] == 1) x q[1];
        // → GotoIfNotCreg + X + Label
        let prog = parse("OPENQASM 3;\nqubit[2] q;\nbit[1] c;\nif (c[0] == 1) x q[1];\n");
        assert!(prog.instructions.len() >= 3, "expected ≥3 instructions, got {}", prog.instructions.len());
        assert!(matches!(prog.instructions[0], Instruction::GotoIfNotCreg { .. }));
        assert!(matches!(prog.instructions[1], Instruction::X(1)));
        assert!(matches!(prog.instructions[2], Instruction::Label(_)));
    }

    #[test]
    fn test_qasm3_for_loop() {
        // for i in [0:2] { h q[i]; }  → H 0; H 1; H 2
        let prog = parse("OPENQASM 3;\nqubit[3] q;\nfor i in [0:2] { h q[i]; }\n");
        assert_eq!(prog.instructions.len(), 3);
        assert!(matches!(prog.instructions[0], Instruction::H(0)));
        assert!(matches!(prog.instructions[1], Instruction::H(1)));
        assert!(matches!(prog.instructions[2], Instruction::H(2)));
    }

    #[test]
    fn test_qasm3_gate_declaration() {
        let prog = parse(
            "OPENQASM 3;\nqubit[2] q;\ngate mybell a, b { h a; cx a, b; }\nmybell q[0], q[1];\n"
        );
        assert!(prog.gate_defs.contains_key("mybell"), "gate 'mybell' should be defined");
    }

    #[test]
    fn test_qasm3_bell_state() {
        // Bell state via QASM 3 — P(|00>) ≈ 0.5, P(|11>) ≈ 0.5
        let src = "OPENQASM 3;\nqubit[2] q;\nbit[2] c;\nh q[0];\ncx q[0], q[1];\nmeasure q[0] -> c[0];\nmeasure q[1] -> c[1];\n";
        let prog = from_qasm3(src).unwrap();
        let result = crate::runtime::execute(&prog).unwrap();
        let probs = result.pre_measurement_probs.as_deref().unwrap_or(&result.final_probabilities);
        assert!((probs[0] - 0.5).abs() < 0.01, "P(|00⟩) = {}", probs[0]);
        assert!((probs[3] - 0.5).abs() < 0.01, "P(|11⟩) = {}", probs[3]);
    }
}
