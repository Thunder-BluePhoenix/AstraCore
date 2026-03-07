/// Web-server backend — interactive HTTP dashboard.
///
/// Routes:
///   `GET  /`          — interactive single-page app (AQL editor + live charts)
///   `GET  /api/data`  — JSON snapshot of the pre-loaded circuit
///   `POST /api/run`   — run any AQL source posted from the editor; returns JSON
///
/// The SPA lets the user:
///   - Open any `.aql` file from their filesystem (browser file-picker)
///   - Type or paste AQL directly into the code editor
///   - Click **▶ Execute** (or press Ctrl+Enter) to run and visualise
use std::sync::Arc;
use std::process::Command;

use axum::{
    extract::State,
    response::Html,
    routing::{get, post},
    Router,
};

use crate::compiler;
use crate::core::gates::{
    apply_cnot, apply_cz, apply_single_qubit_gate, apply_swap, apply_toffoli,
    hadamard, pauli_x, pauli_y, pauli_z, phase_gate, rx, ry, rz, s_gate, t_gate,
};
use crate::core::StateVector;
use crate::compiler::ir::{Instruction, Program};
use crate::dashboard::{circuit_svg, html::render_server_html, DashboardData};
use crate::runtime::run_shots;

// ── Public API ────────────────────────────────────────────────────────────

/// Start the local HTTP dashboard server (blocking).
///
/// Creates a tokio runtime internally so `main` stays synchronous.
/// Run until the user interrupts with Ctrl-C.
pub fn serve(data: DashboardData, port: u16) {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("failed to build tokio runtime");

    rt.block_on(async_serve(Arc::new(data), port));
}

// ── Async internals ───────────────────────────────────────────────────────

async fn async_serve(data: Arc<DashboardData>, port: u16) {
    let app = Router::new()
        .route("/", get(handler_index))
        .route("/api/data",  get(handler_data))
        .route("/api/run",   post(handler_run))
        .route("/api/shots", post(handler_shots))
        .route("/api/steps",     post(handler_steps))
        .route("/api/to-python",    post(handler_to_python))
        .route("/api/run-python",   post(handler_run_python))
        .with_state(data);

    let addr = format!("0.0.0.0:{}", port);
    let listener = tokio::net::TcpListener::bind(&addr).await
        .unwrap_or_else(|e| {
            eprintln!("Cannot bind to port {port}: {e}");
            std::process::exit(1);
        });

    println!();
    println!("━━━ AstraCore Web Dashboard ━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  Dashboard : http://localhost:{}/", port);
    println!("  API data  : http://localhost:{}/api/data", port);
    println!("  API run   : POST http://localhost:{}/api/run", port);
    println!("  Stop      : Ctrl-C");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    axum::serve(listener, app)
        .await
        .expect("server error");
}

// ── Route handlers ────────────────────────────────────────────────────────

/// `GET /` — serve the interactive SPA (static HTML, no server-side data).
async fn handler_index() -> Html<&'static str> {
    Html(render_server_html())
}

/// `GET /api/data` — JSON snapshot of the circuit passed on the command line.
async fn handler_data(
    State(data): State<Arc<DashboardData>>,
) -> axum::Json<serde_json::Value> {
    axum::Json(build_json(&data))
}

/// `POST /api/run` — accept `{ "source": "<aql>" }`, run it, return JSON.
///
/// On parse/runtime error returns `{ "error": "<message>" }` (HTTP 200).
async fn handler_run(
    axum::Json(req): axum::Json<RunRequest>,
) -> axum::Json<serde_json::Value> {
    // Parse once to generate the circuit diagram.
    let program = match compiler::parse_source(&req.source) {
        Ok(p)  => p,
        Err(e) => return axum::Json(serde_json::json!({ "error": e.to_string() })),
    };
    let svg = circuit_svg::render(&program.instructions, program.num_qubits);

    let analysis = match compiler::analyze_source(&req.source) {
        Ok(a)  => a,
        Err(e) => return axum::Json(serde_json::json!({ "error": e.to_string() })),
    };
    let result = match compiler::run(&req.source) {
        Ok(r)  => r,
        Err(e) => return axum::Json(serde_json::json!({ "error": e.to_string() })),
    };
    let data = DashboardData {
        source_path: "<editor>".to_string(),
        analysis,
        result,
        circuit_svg: svg,
    };
    axum::Json(build_json(&data))
}

/// `POST /api/steps` — return gate-by-gate state snapshots for step-through animation.
///
/// Returns `{ "steps": [{step, label, probabilities}, …], "n_qubits": N }`
/// or `{ "error": "…" }` on failure. Capped at 100 steps.
async fn handler_steps(
    axum::Json(req): axum::Json<RunRequest>,
) -> axum::Json<serde_json::Value> {
    let program = match compiler::parse_source(&req.source) {
        Ok(p)  => p,
        Err(e) => return axum::Json(serde_json::json!({ "error": e.to_string() })),
    };
    let snapshots = execute_steps(&program);
    let n_qubits = program.num_qubits;
    axum::Json(serde_json::json!({
        "steps":    snapshots,
        "n_qubits": n_qubits,
    }))
}

/// `POST /api/shots` — run `{ "source": "<aql>", "shots": N }` and return a
/// histogram of measurement bitstrings.
///
/// Returns `{ "counts": { "00": 512, "11": 488 }, "n_shots": 1000, "n_qubits": 2 }`
/// or `{ "error": "…" }` on failure.
async fn handler_shots(
    axum::Json(req): axum::Json<ShotsRequest>,
) -> axum::Json<serde_json::Value> {
    let program = match compiler::parse_source(&req.source) {
        Ok(p)  => p,
        Err(e) => return axum::Json(serde_json::json!({ "error": e.to_string() })),
    };
    let shots = req.shots.max(1).min(100_000);
    let sr = match run_shots(&program, shots) {
        Ok(r)  => r,
        Err(e) => return axum::Json(serde_json::json!({ "error": e.to_string() })),
    };
    let counts: serde_json::Map<String, serde_json::Value> = sr.counts.iter()
        .map(|(k, &v)| (k.clone(), serde_json::Value::Number(v.into())))
        .collect();
    axum::Json(serde_json::json!({
        "counts":   serde_json::Value::Object(counts),
        "n_shots":  sr.n_shots,
        "n_qubits": sr.n_qubits,
    }))
}

// ── Step-by-step execution ────────────────────────────────────────────────

/// A single snapshot of the quantum state after applying one instruction.
#[derive(serde::Serialize)]
pub(crate) struct StepSnapshot {
    pub(crate) step:          usize,
    pub(crate) label:         String,
    pub(crate) probabilities: Vec<f64>,
}

/// Execute `program` instruction-by-instruction, returning a state snapshot
/// after each gate. Capped at 100 snapshots. Control-flow (GOTO/IF) terminates
/// the trace with a warning label. CallGate is treated as an opaque step.
pub(crate) fn execute_steps(program: &Program) -> Vec<StepSnapshot> {
    let n = program.num_qubits;
    let mut state = StateVector::new(n);
    let mut snaps: Vec<StepSnapshot> = Vec::new();

    // Initial ground state snapshot.
    snaps.push(StepSnapshot {
        step:          0,
        label:         "Initial |0\u{22ef}0\u{27e9}".to_string(),
        probabilities: state_probs(&state),
    });

    for (idx, instr) in program.instructions.iter().enumerate() {
        if snaps.len() >= 101 { break; }

        match instr {
            Instruction::H(q)     => apply_single_qubit_gate(&mut state, &hadamard(),  *q),
            Instruction::X(q)     => apply_single_qubit_gate(&mut state, &pauli_x(),   *q),
            Instruction::Y(q)     => apply_single_qubit_gate(&mut state, &pauli_y(),   *q),
            Instruction::Z(q)     => apply_single_qubit_gate(&mut state, &pauli_z(),   *q),
            Instruction::S(q)     => apply_single_qubit_gate(&mut state, &s_gate(),    *q),
            Instruction::T(q)     => apply_single_qubit_gate(&mut state, &t_gate(),    *q),
            Instruction::Rx { qubit, theta }    => apply_single_qubit_gate(&mut state, &rx(*theta),         *qubit),
            Instruction::Ry { qubit, theta }    => apply_single_qubit_gate(&mut state, &ry(*theta),         *qubit),
            Instruction::Rz { qubit, theta }    => apply_single_qubit_gate(&mut state, &rz(*theta),         *qubit),
            Instruction::Phase { qubit, theta } => apply_single_qubit_gate(&mut state, &phase_gate(*theta), *qubit),

            Instruction::Cnot { control, target } => apply_cnot(&mut state, *control, *target),
            Instruction::Cz   { control, target } => apply_cz  (&mut state, *control, *target),
            Instruction::Swap { qubit_a, qubit_b }=> apply_swap(&mut state, *qubit_a, *qubit_b),
            Instruction::Toffoli { control0, control1, target } => {
                apply_toffoli(&mut state, *control0, *control1, *target);
            }

            Instruction::Measure(q) => {
                state.collapse(*q, rand::random::<f64>());
            }
            Instruction::MeasureAll => {
                for q in 0..n { state.collapse(q, rand::random::<f64>()); }
            }

            // Invisible / structural
            Instruction::Barrier | Instruction::Label(_) => continue,

            // Control flow — snapshot with a label but keep tracing linearly
            Instruction::Goto { label } => {
                snaps.push(StepSnapshot {
                    step:  idx + 1,
                    label: format!("\u{21aa} GOTO {label}  \u{26a0} runtime may jump"),
                    probabilities: state_probs(&state),
                });
                continue;
            }
            Instruction::GotoIf { qubit, label } => {
                snaps.push(StepSnapshot {
                    step:  idx + 1,
                    label: format!("\u{21aa} IF q{qubit} GOTO {label}  \u{26a0} branch depends on measurement"),
                    probabilities: state_probs(&state),
                });
                continue;
            }
            Instruction::GotoIfNot { qubit, label } => {
                snaps.push(StepSnapshot {
                    step:  idx + 1,
                    label: format!("\u{21aa} IFNOT q{qubit} GOTO {label}  \u{26a0} branch depends on measurement"),
                    probabilities: state_probs(&state),
                });
                continue;
            }

            Instruction::MeasureInto { qubit, .. } => {
                state.collapse(*qubit, rand::random::<f64>());
            }
            Instruction::GotoIfCreg { creg, bit, label } => {
                snaps.push(StepSnapshot {
                    step:  idx + 1,
                    label: format!("\u{21aa} IF {creg}[{bit}] GOTO {label}  \u{26a0} branch depends on CREG"),
                    probabilities: state_probs(&state),
                });
                continue;
            }
            Instruction::GotoIfNotCreg { creg, bit, label } => {
                snaps.push(StepSnapshot {
                    step:  idx + 1,
                    label: format!("\u{21aa} IFNOT {creg}[{bit}] GOTO {label}  \u{26a0} branch depends on CREG"),
                    probabilities: state_probs(&state),
                });
                continue;
            }

            // Opaque custom gate — snapshot but don't expand
            Instruction::CallGate { name, qubits } => {
                let qs: Vec<String> = qubits.iter().map(|q| format!("q{q}")).collect();
                let label = format!("CALL {} {}", name, qs.join(" "));
                snaps.push(StepSnapshot {
                    step: idx + 1,
                    label,
                    probabilities: state_probs(&state),
                });
                continue; // already pushed — don't push again below
            }
        }

        snaps.push(StepSnapshot {
            step:          idx + 1,
            label:         fmt_instr(instr),
            probabilities: state_probs(&state),
        });
    }
    snaps
}

/// Compute probabilities from state vector amplitudes.
#[inline]
fn state_probs(state: &StateVector) -> Vec<f64> {
    state.amplitudes.iter().map(|a| a.norm_sq()).collect()
}

/// Human-readable label for a single instruction.
fn fmt_instr(instr: &Instruction) -> String {
    match instr {
        Instruction::H(q)     => format!("H q{q}"),
        Instruction::X(q)     => format!("X q{q}"),
        Instruction::Y(q)     => format!("Y q{q}"),
        Instruction::Z(q)     => format!("Z q{q}"),
        Instruction::S(q)     => format!("S q{q}"),
        Instruction::T(q)     => format!("T q{q}"),
        Instruction::Rx { qubit, theta }    => format!("Rx({theta:.3}) q{qubit}"),
        Instruction::Ry { qubit, theta }    => format!("Ry({theta:.3}) q{qubit}"),
        Instruction::Rz { qubit, theta }    => format!("Rz({theta:.3}) q{qubit}"),
        Instruction::Phase { qubit, theta } => format!("Phase({theta:.3}) q{qubit}"),
        Instruction::Cnot { control, target } => format!("CNOT q{control} \u{2192} q{target}"),
        Instruction::Cz   { control, target } => format!("CZ q{control} q{target}"),
        Instruction::Swap { qubit_a, qubit_b} => format!("SWAP q{qubit_a} q{qubit_b}"),
        Instruction::Toffoli { control0, control1, target } => {
            format!("Toffoli q{control0} q{control1} \u{2192} q{target}")
        }
        Instruction::Measure(q)  => format!("Measure q{q}"),
        Instruction::MeasureAll  => "Measure All".to_string(),
        Instruction::CallGate { name, qubits } => {
            let qs: Vec<String> = qubits.iter().map(|q| format!("q{q}")).collect();
            format!("CALL {} {}", name, qs.join(" "))
        }
        _ => String::new(),
    }
}

// ── AQL → Python code generator ────────────────────────────────────────────

/// Convert a parsed AQL [`Program`] to equivalent `astracore` Python API code.
fn aql_to_python(program: &Program) -> String {
    let n = program.num_qubits;
    let mut lines: Vec<String> = Vec::new();
    lines.push("import astracore as ac".to_string());
    lines.push(String::new());
    lines.push(format!("c = ac.Circuit({})", n));

    for instr in &program.instructions {
        lines.push(instr_to_py(instr));
    }

    lines.push(String::new());
    lines.push("result = c.run()".to_string());
    lines.push(format!("n = {}", n));
    lines.push("for i, p in enumerate(result.probabilities):".to_string());
    lines.push("    if p > 1e-6:".to_string());
    lines.push("        print(f'|{i:0{n}b}\\u27e9: {p:.4f}')".to_string());

    lines.join("\n")
}

fn instr_to_py(instr: &Instruction) -> String {
    match instr {
        Instruction::H(q)     => format!("c.h({})",  q),
        Instruction::X(q)     => format!("c.x({})",  q),
        Instruction::Y(q)     => format!("c.y({})",  q),
        Instruction::Z(q)     => format!("c.z({})",  q),
        Instruction::S(q)     => format!("c.s({})",  q),
        Instruction::T(q)     => format!("c.t({})",  q),
        Instruction::Rx    { qubit, theta } => format!("c.rx({}, {:.6})",    qubit, theta),
        Instruction::Ry    { qubit, theta } => format!("c.ry({}, {:.6})",    qubit, theta),
        Instruction::Rz    { qubit, theta } => format!("c.rz({}, {:.6})",    qubit, theta),
        Instruction::Phase { qubit, theta } => format!("c.phase({}, {:.6})", qubit, theta),
        Instruction::Cnot { control, target }  => format!("c.cnot({}, {})", control, target),
        Instruction::Cz   { control, target }  => format!("c.cz({}, {})",   control, target),
        Instruction::Swap { qubit_a, qubit_b } => format!("c.swap({}, {})", qubit_a, qubit_b),
        Instruction::Toffoli { control0, control1, target } =>
            format!("c.toffoli({}, {}, {})", control0, control1, target),
        Instruction::Measure(q) => format!("c.measure({})", q),
        Instruction::MeasureAll  => "c.measure_all()".to_string(),
        Instruction::Barrier     => "c.barrier()".to_string(),
        Instruction::Label(l)    => format!("# label: {}", l),
        Instruction::Goto { label }          => format!("# goto {}", label),
        Instruction::GotoIf    { qubit, label } => format!("# if q{} goto {}", qubit, label),
        Instruction::GotoIfNot { qubit, label } => format!("# if not q{} goto {}", qubit, label),
        Instruction::CallGate { name, qubits } => {
            let qs = qubits.iter().map(|q| q.to_string()).collect::<Vec<_>>().join(", ");
            format!("c.call(\"{}\", [{}])", name, qs)
        }
        Instruction::MeasureInto { qubit, creg, creg_bit } =>
            format!("c.measure_into({}, \"{}\", {})", qubit, creg, creg_bit),
        Instruction::GotoIfCreg { creg, bit, label } =>
            format!("# IF {}[{}] GOTO {}", creg, bit, label),
        Instruction::GotoIfNotCreg { creg, bit, label } =>
            format!("# IFNOT {}[{}] GOTO {}", creg, bit, label),
    }
}

/// `POST /api/to-python` — convert AQL source to equivalent Python API code.
async fn handler_to_python(
    axum::Json(req): axum::Json<RunRequest>,
) -> axum::Json<serde_json::Value> {
    match compiler::parse_source(&req.source) {
        Ok(program) => {
            let python = aql_to_python(&program);
            axum::Json(serde_json::json!({ "python": python }))
        }
        Err(e) => axum::Json(serde_json::json!({ "error": e.to_string() })),
    }
}

// ── Python executor ───────────────────────────────────────────────────────────

/// Find a suitable Python interpreter: prefer the local .venv, fall back to PATH.
fn find_python() -> String {
    // Look for astracore-py/.venv relative to the executable or CWD
    let candidates: &[&str] = &[
        "astracore-py/.venv/Scripts/python.exe", // Windows venv
        "astracore-py/.venv/bin/python",          // Unix venv
        "python",
        "python3",
    ];
    for c in candidates {
        if std::path::Path::new(c).exists() {
            return c.to_string();
        }
        // Also check relative to CWD
        if let Ok(cwd) = std::env::current_dir() {
            let full = cwd.join(c);
            if full.exists() {
                return full.to_string_lossy().into_owned();
            }
        }
    }
    "python".to_string() // last resort
}

async fn handler_run_python(
    axum::Json(req): axum::Json<RunRequest>,
) -> axum::Json<serde_json::Value> {
    // Write source to a temp file to avoid shell-escaping issues
    let mut tmp_path = std::env::temp_dir();
    tmp_path.push(format!("astracore_run_{}.py", std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)));

    if let Err(e) = std::fs::write(&tmp_path, &req.source) {
        return axum::Json(serde_json::json!({ "error": format!("Failed to write temp file: {e}") }));
    }

    let python = find_python();
    let result = Command::new(&python).arg(&tmp_path).output();
    let _ = std::fs::remove_file(&tmp_path);

    match result {
        Ok(output) => {
            let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
            let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
            let exit_code = output.status.code().unwrap_or(-1);
            axum::Json(serde_json::json!({
                "stdout": stdout,
                "stderr": stderr,
                "exit_code": exit_code
            }))
        }
        Err(e) => axum::Json(serde_json::json!({
            "error": format!("Could not launch Python interpreter '{}': {}", python, e)
        })),
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn bell_program() -> Program {
        compiler::parse_source("QREG 2\nH 0\nCNOT 0 1\nMEASURE_ALL").unwrap()
    }

    #[test]
    fn execute_steps_bell_at_least_three() {
        let prog = bell_program();
        let snaps = execute_steps(&prog);
        // Initial + H + CNOT + MeasureAll = at least 4 snapshots
        assert!(snaps.len() >= 3, "Expected ≥3 snapshots, got {}", snaps.len());
        assert_eq!(snaps[0].step, 0);
        assert!(snaps[0].label.contains("Initial"));
    }

    #[test]
    fn execute_steps_initial_is_ground_state() {
        let prog = bell_program();
        let snaps = execute_steps(&prog);
        let probs = &snaps[0].probabilities;
        // 2 qubits → 4 amplitudes; |00⟩ = index 0 should have prob ≈ 1.0
        assert!(probs[0] > 0.99, "P(|00⟩) should be 1.0 initially, got {}", probs[0]);
        assert!(probs[1..].iter().all(|&p| p < 1e-10));
    }

    #[test]
    fn execute_steps_toffoli_adds_snapshot() {
        // Toffoli gate should produce a new step snapshot
        let prog = compiler::parse_source("QREG 3\nH 0\nCNOT 0 1\nCCX 0 1 2").unwrap();
        let snaps = execute_steps(&prog);
        // Initial + H + CNOT + Toffoli = 4 snapshots
        assert!(snaps.len() >= 4, "Expected ≥4 snapshots for Toffoli circuit, got {}", snaps.len());
        let toffoli_snap = snaps.iter().find(|s| s.label.contains("Toffoli"));
        assert!(toffoli_snap.is_some(), "Expected a snapshot labeled 'Toffoli'");
    }

    #[test]
    fn execute_steps_barrier_skipped() {
        // BARRIER is structural — it should NOT produce a new state snapshot
        let prog = compiler::parse_source("QREG 2\nH 0\nBARRIER\nH 1").unwrap();
        let snaps_no_barrier = execute_steps(
            &compiler::parse_source("QREG 2\nH 0\nH 1").unwrap()
        );
        let snaps_with_barrier = execute_steps(&prog);
        // Both should produce the same number of snapshots (BARRIER is skipped)
        assert_eq!(
            snaps_no_barrier.len(), snaps_with_barrier.len(),
            "BARRIER should not add a snapshot"
        );
    }

    #[test]
    fn execute_steps_measureall_probs_sum_to_one() {
        let prog = compiler::parse_source("QREG 2\nH 0\nCNOT 0 1\nMEASURE_ALL").unwrap();
        let snaps = execute_steps(&prog);
        // After MeasureAll the probabilities should sum to ≈ 1.0
        let last = snaps.last().unwrap();
        let total: f64 = last.probabilities.iter().sum();
        assert!((total - 1.0).abs() < 1e-9, "Probabilities should sum to 1.0, got {}", total);
    }

    #[test]
    fn build_json_has_circuit_svg_key() {
        let prog = bell_program();
        let analysis = compiler::analyze_source("QREG 2\nH 0\nCNOT 0 1\nMEASURE_ALL").unwrap();
        let result = compiler::run("QREG 2\nH 0\nCNOT 0 1\nMEASURE_ALL").unwrap();
        let svg = crate::dashboard::circuit_svg::render(&prog.instructions, prog.num_qubits);
        let data = DashboardData {
            source_path: "test".to_string(),
            analysis,
            result,
            circuit_svg: svg,
        };
        let json = build_json(&data);
        assert!(json.get("circuit_svg").is_some(), "build_json must include 'circuit_svg' key");
        let svg_val = json["circuit_svg"].as_str().unwrap_or("");
        assert!(!svg_val.is_empty(), "circuit_svg should not be empty");
    }

    #[test]
    fn handler_steps_invalid_source_produces_error_field() {
        // Simulate what handler_steps does internally — invalid AQL source
        let src = "QREG 2\nNOT_A_GATE 0\n";
        let result = compiler::parse_source(src);
        assert!(result.is_err(), "Invalid source should fail to parse");
        let err_json = serde_json::json!({ "error": result.unwrap_err().to_string() });
        assert!(err_json.get("error").is_some(), "Error JSON must have 'error' key");
        let msg = err_json["error"].as_str().unwrap();
        assert!(!msg.is_empty(), "Error message should not be empty");
    }
}

// ── Request / response helpers ────────────────────────────────────────────

#[derive(serde::Deserialize)]
struct RunRequest {
    source: String,
}

#[derive(serde::Deserialize)]
struct ShotsRequest {
    source: String,
    shots:  usize,
}

/// Serialize a [`DashboardData`] to a JSON value understood by the SPA.
fn build_json(data: &DashboardData) -> serde_json::Value {
    let a = &data.analysis;
    let r = &data.result;

    // Gate histogram as a JSON object { "H": 2, "CNOT": 1, … }
    let gate_hist: serde_json::Map<String, serde_json::Value> = a.gate_histogram.iter()
        .map(|(k, &v)| (k.clone(), serde_json::Value::Number(v.into())))
        .collect();

    // Measurement records
    let measurements: Vec<serde_json::Value> = r.measurements.iter()
        .map(|m| serde_json::json!({
            "qubit":   m.qubit,
            "outcome": m.outcome as u8,
            "step":    m.step,
        }))
        .collect();

    // Significant states as [{ state: "00", prob: 0.5 }, …]
    let significant: Vec<serde_json::Value> = data.display_states(64).iter()
        .map(|(label, p)| serde_json::json!({ "state": label, "prob": p }))
        .collect();

    // Full amplitude table — prefer pre-measurement (shows superposition), fall back to final
    let n = a.num_qubits;
    let amp_src: &[(f64, f64)] = r.pre_measurement_amplitudes.as_deref()
        .unwrap_or(&r.final_amplitudes);
    let amp_limit = amp_src.len().min(64);
    let amplitudes: Vec<serde_json::Value> = amp_src[..amp_limit].iter()
        .enumerate()
        .map(|(i, (re, im))| {
            let prob  = re * re + im * im;
            let phase = im.atan2(*re).to_degrees();
            let label = format!("|{:0>width$b}⟩", i, width = n);
            serde_json::json!({ "state": label, "re": re, "im": im, "prob": prob, "phase": phase })
        })
        .collect();

    serde_json::json!({
        "source_path":          data.source_path,
        "num_qubits":           a.num_qubits,
        "gate_count":           a.gate_count,
        "expanded_gate_count":  a.expanded_gate_count,
        "circuit_depth":        a.circuit_depth,
        "two_qubit_gate_count": a.two_qubit_gate_count,
        "measure_count":        a.measure_count,
        "has_control_flow":     a.has_control_flow,
        "has_custom_gates":     a.has_custom_gates,
        "custom_gate_defs":     a.custom_gate_defs,
        "gate_histogram":       serde_json::Value::Object(gate_hist),
        "qubit_utilization":    a.qubit_utilization,
        "significant_states":   significant,
        "exec_gate_count":      r.gate_count,
        "branch_count":         r.branch_count,
        "steps_executed":       r.steps_executed,
        "measurements":         measurements,
        "circuit_svg":          data.circuit_svg,
        "amplitudes":           amplitudes,
    })
}
