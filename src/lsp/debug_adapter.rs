/// AQL Debug Adapter Protocol (DAP) server.
///
/// Reads JSON-RPC messages from stdin and writes responses to stdout
/// using the DAP Content-Length framing. Wraps the existing `execute_steps()`
/// engine from `src/dashboard/server.rs` for step-by-step execution.
///
/// Supports a minimal subset of DAP sufficient for the VS Code circuit debugger:
/// - initialize, launch, setBreakpoints, threads, stackTrace, variables
/// - next (step), continue, disconnect
///
/// Usage: `astracore dap` (VS Code calls this via debuggers[].program)

use std::collections::HashMap;
use std::io::{self, BufRead, Read, Write};

use crate::compiler;
use crate::dashboard::server::{execute_steps, StepSnapshot};

// ── DAP message framing ───────────────────────────────────────────────────────

fn read_message(reader: &mut dyn BufRead) -> Option<serde_json::Value> {
    let mut content_length: usize = 0;
    // Read headers (terminated by \r\n\r\n)
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).ok()? == 0 { return None; }
        let line = line.trim_end_matches(['\r', '\n']);
        if line.is_empty() { break; }
        if let Some(rest) = line.to_ascii_lowercase().strip_prefix("content-length:") {
            content_length = rest.trim().parse().unwrap_or(0);
        }
    }
    if content_length == 0 { return None; }
    let mut body = vec![0u8; content_length];
    io::stdin().read_exact(&mut body).ok()?;
    serde_json::from_slice(&body).ok()
}

fn write_message(writer: &mut dyn Write, value: &serde_json::Value) {
    let body = serde_json::to_string(value).unwrap_or_default();
    let _ = write!(writer, "Content-Length: {}\r\n\r\n{}", body.len(), body);
    let _ = writer.flush();
}

// ── DAP server state ──────────────────────────────────────────────────────────

struct DapSession {
    seq:         usize,
    snapshots:   Vec<StepSnapshot>,
    step_idx:    usize,
    breakpoints: Vec<usize>,  // 0-indexed lines with breakpoints
    source:      String,
}

impl DapSession {
    fn new() -> Self {
        Self { seq: 1, snapshots: vec![], step_idx: 0, breakpoints: vec![], source: String::new() }
    }

    fn next_seq(&mut self) -> usize { let s = self.seq; self.seq += 1; s }

    fn response(&mut self, req_seq: u64, command: &str, body: serde_json::Value, success: bool) -> serde_json::Value {
        serde_json::json!({
            "seq": self.next_seq(),
            "type": "response",
            "request_seq": req_seq,
            "success": success,
            "command": command,
            "body": body,
        })
    }

    fn event(&mut self, event: &str, body: serde_json::Value) -> serde_json::Value {
        serde_json::json!({
            "seq": self.next_seq(),
            "type": "event",
            "event": event,
            "body": body,
        })
    }

    fn stopped_event(&mut self, reason: &str) -> serde_json::Value {
        self.event("stopped", serde_json::json!({
            "reason": reason,
            "threadId": 1,
            "allThreadsStopped": true,
        }))
    }

    fn current_frame_name(&self) -> String {
        self.snapshots.get(self.step_idx)
            .map(|s| s.label.clone())
            .unwrap_or_else(|| "Initial |0…0⟩".to_string())
    }

    fn current_probabilities(&self) -> Vec<(String, f64)> {
        let snap = match self.snapshots.get(self.step_idx) {
            Some(s) => s,
            None => return vec![],
        };
        let n = (snap.probabilities.len() as f64).log2() as usize;
        snap.probabilities.iter().enumerate()
            .map(|(i, &p)| (format!("|{:0>width$b}⟩", i, width = n), p))
            .collect()
    }
}

// ── Main serve loop ───────────────────────────────────────────────────────────

/// Run the DAP server on stdin/stdout until the client disconnects.
pub async fn serve() {
    let stdin  = io::stdin();
    let stdout = io::stdout();
    let mut reader = io::BufReader::new(stdin.lock());
    let mut writer = io::BufWriter::new(stdout.lock());
    let mut session = DapSession::new();
    let mut var_ref_counter: u64 = 0;
    let mut var_ref_map: HashMap<u64, Vec<(String, f64)>> = HashMap::new();

    loop {
        let msg = match read_message(&mut reader) {
            Some(m) => m,
            None    => break,
        };

        let seq     = msg["seq"].as_u64().unwrap_or(0);
        let command = msg["command"].as_str().unwrap_or("").to_string();
        let args    = &msg["arguments"];

        let responses: Vec<serde_json::Value> = match command.as_str() {
            "initialize" => {
                let resp = session.response(seq, "initialize", serde_json::json!({
                    "supportsStepBack": false,
                    "supportsConfigurationDoneRequest": true,
                    "supportsSetBreakpointsRequest": true,
                }), true);
                let init_evt = session.event("initialized", serde_json::Value::Null);
                vec![resp, init_evt]
            }

            "configurationDone" => {
                vec![session.response(seq, "configurationDone", serde_json::Value::Null, true)]
            }

            "launch" => {
                session.source = args["source"].as_str().unwrap_or("").to_string();
                let resp = match compiler::parse_source(&session.source) {
                    Ok(prog) => {
                        session.snapshots = execute_steps(&prog);
                        session.step_idx  = 0;
                        let r = session.response(seq, "launch", serde_json::Value::Null, true);
                        let e = session.stopped_event("entry");
                        vec![r, e]
                    }
                    Err(e) => {
                        let msg = format!("{e}");
                        vec![session.response(seq, "launch", serde_json::json!({"error": msg}), false)]
                    }
                };
                resp
            }

            "setBreakpoints" => {
                let bps = args["breakpoints"].as_array()
                    .map(|a| a.iter()
                        .filter_map(|bp| bp["line"].as_u64())
                        .map(|l| (l as usize).saturating_sub(1))
                        .collect::<Vec<_>>())
                    .unwrap_or_default();
                let bps_json: Vec<serde_json::Value> = bps.iter()
                    .map(|&l| serde_json::json!({ "verified": true, "line": l + 1 }))
                    .collect();
                session.breakpoints = bps;
                vec![session.response(seq, "setBreakpoints",
                    serde_json::json!({ "breakpoints": bps_json }), true)]
            }

            "threads" => {
                vec![session.response(seq, "threads", serde_json::json!({
                    "threads": [{ "id": 1, "name": "AQL Executor" }]
                }), true)]
            }

            "stackTrace" => {
                let name = session.current_frame_name();
                vec![session.response(seq, "stackTrace", serde_json::json!({
                    "stackFrames": [{
                        "id": 1,
                        "name": name,
                        "line": session.step_idx + 1,
                        "column": 0,
                    }],
                    "totalFrames": 1,
                }), true)]
            }

            "scopes" => {
                var_ref_counter += 1;
                let probs = session.current_probabilities();
                var_ref_map.insert(var_ref_counter, probs);
                vec![session.response(seq, "scopes", serde_json::json!({
                    "scopes": [{
                        "name": "Quantum State",
                        "variablesReference": var_ref_counter,
                        "expensive": false,
                    }]
                }), true)]
            }

            "variables" => {
                let var_ref = args["variablesReference"].as_u64().unwrap_or(0);
                let probs = var_ref_map.get(&var_ref).cloned().unwrap_or_default();
                let vars: Vec<serde_json::Value> = probs.iter()
                    .map(|(label, p)| serde_json::json!({
                        "name":  label,
                        "value": format!("{:.6}", p),
                        "variablesReference": 0,
                    }))
                    .collect();
                vec![session.response(seq, "variables", serde_json::json!({ "variables": vars }), true)]
            }

            "next" => {
                if session.step_idx + 1 < session.snapshots.len() {
                    session.step_idx += 1;
                    let r = session.response(seq, "next", serde_json::Value::Null, true);
                    let e = session.stopped_event("step");
                    vec![r, e]
                } else {
                    let r = session.response(seq, "next", serde_json::Value::Null, true);
                    let e = session.event("terminated", serde_json::Value::Null);
                    vec![r, e]
                }
            }

            "continue" => {
                // Advance to next breakpoint or end
                let start = session.step_idx + 1;
                let bp_hit = (start..session.snapshots.len())
                    .find(|&i| session.breakpoints.contains(&i));
                session.step_idx = bp_hit.unwrap_or(session.snapshots.len().saturating_sub(1));
                let r = session.response(seq, "continue",
                    serde_json::json!({ "allThreadsContinued": true }), true);
                let e = if bp_hit.is_some() { session.stopped_event("breakpoint") }
                        else { session.event("terminated", serde_json::Value::Null) };
                vec![r, e]
            }

            "evaluate" => {
                let expr = args["expression"].as_str().unwrap_or("");
                let probs = session.current_probabilities();
                let result = probs.iter()
                    .find(|(label, _)| label.contains(expr))
                    .map(|(l, p)| format!("{l} = {p:.6}"))
                    .unwrap_or_else(|| "not found".to_string());
                vec![session.response(seq, "evaluate",
                    serde_json::json!({ "result": result, "variablesReference": 0 }), true)]
            }

            "disconnect" => {
                let r = session.response(seq, "disconnect", serde_json::Value::Null, true);
                let _ = write_message(&mut writer, &r);
                break;
            }

            other => {
                eprintln!("[dap] unhandled command: {other}");
                vec![session.response(seq, other, serde_json::Value::Null, true)]
            }
        };

        for resp in &responses {
            write_message(&mut writer, resp);
        }
    }
}
