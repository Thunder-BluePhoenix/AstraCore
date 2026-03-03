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
use crate::dashboard::{html::render_server_html, DashboardData};
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
    };
    axum::Json(build_json(&data))
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
    })
}
