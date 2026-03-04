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
        .route("/api/steps", post(handler_steps))
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
struct StepSnapshot {
    step:          usize,
    label:         String,
    probabilities: Vec<f64>,
}

/// Execute `program` instruction-by-instruction, returning a state snapshot
/// after each gate. Capped at 100 snapshots. Control-flow (GOTO/IF) terminates
/// the trace with a warning label. CallGate is treated as an opaque step.
fn execute_steps(program: &Program) -> Vec<StepSnapshot> {
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

            // Control flow — stop tracing
            Instruction::Goto { .. }
            | Instruction::GotoIf { .. }
            | Instruction::GotoIfNot { .. } => {
                snaps.push(StepSnapshot {
                    step:          idx + 1,
                    label:         "\u{26a0} Control flow \u{2014} remaining steps not shown".to_string(),
                    probabilities: state_probs(&state),
                });
                return snaps;
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
    })
}
