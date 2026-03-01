/// HTML Report backend — generates a standalone `report.html` file
/// with embedded Chart.js charts (no server required, no internet after load).
///
/// Also provides [`render_server_html`] — a fully interactive single-page app
/// served by the `astracore serve` command, with a live AQL editor and dynamic
/// chart updates via `POST /api/run`.
use crate::dashboard::DashboardData;

// ── Interactive server page ────────────────────────────────────────────────

/// Return the interactive SPA served by `astracore serve`.
///
/// The page has two panels:
/// - **Left** — AQL code editor (textarea) with "Open .aql" and "▶ Execute" buttons
/// - **Right** — charts and metrics updated dynamically via `POST /api/run`
///
/// Keyboard shortcuts: `Ctrl+Enter` = Execute, `Tab` = 2-space indent.
pub fn render_server_html() -> &'static str {
    INTERACTIVE_HTML
}

const INTERACTIVE_HTML: &str = r##"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AstraCore Interactive Dashboard</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: #0d1117; color: #e6edf3;
  font-family: 'Courier New', Courier, monospace;
  height: 100vh; overflow: hidden;
  display: flex; flex-direction: column;
}
/* ── Header ── */
.header {
  padding: 10px 18px; border-bottom: 1px solid #30363d;
  display: flex; align-items: center; gap: 14px; flex-shrink: 0;
}
.header-title { font-size: 1.25em; font-weight: bold; color: #58a6ff; }
.header-sub   { color: #8b949e; font-size: 0.82em; flex: 1; }
.header-api   { color: #3d4349; font-size: 0.75em; }
/* ── Two-panel layout ── */
.app { display: flex; flex: 1; overflow: hidden; }
/* ── Editor panel ── */
.editor-panel {
  width: 40%; min-width: 260px; max-width: 520px;
  border-right: 1px solid #30363d;
  display: flex; flex-direction: column; padding: 10px; gap: 8px; flex-shrink: 0;
}
.toolbar {
  display: flex; align-items: center; gap: 6px;
  background: #161b22; border: 1px solid #30363d; border-radius: 6px;
  padding: 6px 10px; flex-shrink: 0;
}
.filename { flex: 1; color: #8b949e; font-size: 0.8em; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.kbd-hint { color: #3d4349; font-size: 0.73em; white-space: nowrap; }
.btn {
  background: #21262d; color: #e6edf3;
  border: 1px solid #30363d; border-radius: 4px;
  padding: 3px 10px; cursor: pointer;
  font-family: 'Courier New', monospace; font-size: 0.8em; white-space: nowrap;
}
.btn:hover { background: #30363d; }
.btn-run  { background: #1a3a1a; border-color: #3fb950; color: #3fb950; font-weight: bold; }
.btn-run:hover   { background: #264d26; }
.btn-run:disabled { opacity: 0.45; cursor: not-allowed; }
#editor {
  flex: 1; background: #010409; color: #e6edf3;
  border: 1px solid #30363d; border-radius: 6px;
  padding: 12px; font-family: 'Courier New', monospace;
  font-size: 13px; line-height: 1.65; resize: none; outline: none; tab-size: 2;
}
#editor:focus { border-color: #58a6ff60; }
.error-msg { min-height: 20px; color: #f85149; font-size: 0.8em; padding: 0 2px; word-break: break-all; }
/* ── Results panel ── */
.results-panel { flex: 1; overflow-y: auto; padding: 12px; }
.placeholder {
  height: 100%; display: flex; align-items: center; justify-content: center;
  flex-direction: column; gap: 10px; color: #3d4349;
}
.placeholder-icon { font-size: 2.5em; }
.placeholder-text { font-size: 0.9em; }
/* ── Results grid ── */
.results-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
.card { background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 14px; }
.full-width { grid-column: 1 / -1; }
.card-title {
  color: #58a6ff; font-size: 0.73em; letter-spacing: 0.1em;
  border-bottom: 1px solid #21262d; padding-bottom: 6px; margin-bottom: 10px;
}
table { width: 100%; border-collapse: collapse; font-size: 0.83em; }
td { padding: 4px 5px; border-bottom: 1px solid #1c2128; }
td:first-child { color: #8b949e; }
.val { color: #3fb950; font-weight: bold; }
canvas { max-height: 185px; width: 100% !important; }
.badge-yes { color: #3fb950; } .badge-no { color: #4d5560; }
</style>
</head>
<body>
<div class="header">
  <div class="header-title">⚛ AstraCore</div>
  <div class="header-sub">Interactive Dashboard &nbsp;·&nbsp; write or open AQL, then <strong style="color:#3fb950">▶ Execute</strong></div>
  <div class="header-api">API: GET /api/data &nbsp; POST /api/run</div>
</div>

<div class="app">
  <!-- ══ LEFT: Editor ══════════════════════════════════════════════ -->
  <div class="editor-panel">
    <div class="toolbar">
      <button class="btn" id="openBtn">📂 Open .aql</button>
      <input type="file" id="fileInput" accept=".aql,.txt" hidden>
      <span class="filename" id="filename">bell.aql</span>
      <span class="kbd-hint">Ctrl+↵</span>
      <button class="btn btn-run" id="runBtn">▶ Execute</button>
    </div>

    <textarea id="editor" spellcheck="false" autocomplete="off">// Bell State  (|00⟩ + |11⟩) / √2
// Ctrl+Enter to run

QREG 2
H 0
CNOT 0 1
MEASURE_ALL</textarea>

    <div id="errorMsg" class="error-msg"></div>
  </div>

  <!-- ══ RIGHT: Results ════════════════════════════════════════════ -->
  <div class="results-panel">
    <div class="placeholder" id="placeholder">
      <div class="placeholder-icon">⚛</div>
      <div class="placeholder-text">Write AQL and click ▶ Execute to visualize</div>
    </div>

    <div id="results" style="display:none">
      <div class="results-grid">
        <!-- Metrics -->
        <div class="card">
          <div class="card-title">CIRCUIT METRICS</div>
          <div id="metricsContent"></div>
        </div>
        <!-- Probability -->
        <div class="card">
          <div class="card-title">PROBABILITY DISTRIBUTION</div>
          <canvas id="probChart"></canvas>
        </div>
        <!-- Gates -->
        <div class="card">
          <div class="card-title">GATE HISTOGRAM</div>
          <canvas id="gateChart"></canvas>
        </div>
        <!-- Qubit utilization -->
        <div class="card">
          <div class="card-title">QUBIT UTILIZATION</div>
          <canvas id="qubitChart"></canvas>
        </div>
        <!-- Measurements -->
        <div class="card full-width">
          <div class="card-title">MEASUREMENT OUTCOMES</div>
          <div id="measContent"></div>
        </div>
      </div>
    </div>
  </div>
</div>

<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script>
Chart.defaults.color = '#8b949e';
Chart.defaults.borderColor = '#21262d';

const editor    = document.getElementById('editor');
const runBtn    = document.getElementById('runBtn');
const errorMsg  = document.getElementById('errorMsg');
const filename  = document.getElementById('filename');
const fileInput = document.getElementById('fileInput');

// ── File open ──────────────────────────────────────────────────────
document.getElementById('openBtn').onclick = () => fileInput.click();

fileInput.onchange = (e) => {
  const file = e.target.files[0];
  if (!file) return;
  filename.textContent = file.name;
  const reader = new FileReader();
  reader.onload = ev => { editor.value = ev.target.result; };
  reader.readAsText(file);
  fileInput.value = '';  // allow reopening same file
};

// ── Keyboard shortcuts ─────────────────────────────────────────────
editor.addEventListener('keydown', e => {
  if (e.key === 'Tab') {
    e.preventDefault();
    const s = editor.selectionStart, end = editor.selectionEnd;
    editor.value = editor.value.substring(0, s) + '  ' + editor.value.substring(end);
    editor.selectionStart = editor.selectionEnd = s + 2;
  } else if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
    e.preventDefault();
    runCircuit();
  }
});

runBtn.onclick = runCircuit;

// ── Execute ────────────────────────────────────────────────────────
async function runCircuit() {
  const source = editor.value.trim();
  if (!source) return;
  runBtn.disabled = true;
  runBtn.textContent = '⏳ Running…';
  errorMsg.textContent = '';
  try {
    const res = await fetch('/api/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ source })
    });
    if (!res.ok) { errorMsg.textContent = `✗ HTTP ${res.status}`; return; }
    const data = await res.json();
    if (data.error) {
      errorMsg.textContent = '✗ ' + data.error;
    } else {
      renderResults(data);
    }
  } catch (err) {
    errorMsg.textContent = '✗ ' + err.message;
  } finally {
    runBtn.disabled = false;
    runBtn.textContent = '▶ Execute';
  }
}

// ── Chart management ───────────────────────────────────────────────
const chartRefs = new Map();

function updateChart(id, labels, values, color, yMax) {
  if (chartRefs.has(id)) { chartRefs.get(id).destroy(); }
  chartRefs.set(id, new Chart(document.getElementById(id), {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        data: values,
        backgroundColor: values.map(v => v > 0.001 ? color + 'cc' : color + '28'),
        borderColor: color, borderWidth: 1
      }]
    },
    options: {
      animation: { duration: 280 },
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { color: '#1c2128' }, ticks: { color: '#8b949e', font: { family: 'Courier New', size: 11 } } },
        y: { grid: { color: '#1c2128' }, ticks: { color: '#8b949e' }, min: 0,
             ...(yMax != null ? { max: yMax } : {}) }
      }
    }
  }));
}

// ── Render results ─────────────────────────────────────────────────
function renderResults(d) {
  document.getElementById('placeholder').style.display = 'none';
  document.getElementById('results').style.display = 'block';

  // ── Metrics table ──────────────────────────────────────────────
  const cf = d.has_control_flow
    ? '<span class="badge-yes">yes</span>'
    : '<span class="badge-no">no</span>';
  const cg = d.has_custom_gates
    ? `<span class="badge-yes">yes (${d.custom_gate_defs} defs)</span>`
    : '<span class="badge-no">no</span>';
  const entPct = d.gate_count > 0
    ? (d.two_qubit_gate_count / d.gate_count * 100).toFixed(1) + '%'
    : '0%';

  document.getElementById('metricsContent').innerHTML = `
    <table>
      <tr><td>Qubits</td>           <td class="val">${d.num_qubits}</td></tr>
      <tr><td>Gate count</td>       <td class="val">${d.gate_count}</td></tr>
      <tr><td>Expanded gates</td>   <td class="val">${d.expanded_gate_count}</td></tr>
      <tr><td>Circuit depth</td>    <td class="val">${d.circuit_depth}</td></tr>
      <tr><td>2-qubit gates</td>    <td class="val">${d.two_qubit_gate_count} (${entPct})</td></tr>
      <tr><td>Measurements</td>     <td class="val">${d.measure_count}</td></tr>
      <tr><td>Control flow</td>     <td>${cf}</td></tr>
      <tr><td>Custom gates</td>     <td>${cg}</td></tr>
      <tr><td colspan="2" style="padding:3px 0"></td></tr>
      <tr><td>Gates applied</td>    <td class="val">${d.exec_gate_count}</td></tr>
      <tr><td>Branches taken</td>   <td class="val">${d.branch_count}</td></tr>
      <tr><td>Steps executed</td>   <td class="val">${d.steps_executed}</td></tr>
    </table>`;

  // ── Probability chart ──────────────────────────────────────────
  updateChart('probChart',
    d.significant_states.map(s => '|' + s.state + '\u27e9'),
    d.significant_states.map(s => s.prob),
    '#58a6ff', 1);

  // ── Gate histogram ─────────────────────────────────────────────
  const gateEntries = Object.entries(d.gate_histogram).sort((a, b) => b[1] - a[1]);
  updateChart('gateChart',
    gateEntries.map(e => e[0]),
    gateEntries.map(e => e[1]),
    '#3fb950', null);

  // ── Qubit utilization ──────────────────────────────────────────
  updateChart('qubitChart',
    d.qubit_utilization.map((_, i) => 'q' + i),
    d.qubit_utilization,
    '#d2991c', null);

  // ── Measurements ───────────────────────────────────────────────
  if (d.measurements.length === 0) {
    document.getElementById('measContent').innerHTML =
      '<p style="color:#4d5560;text-align:center;padding:8px">No measurements in this circuit</p>';
  } else {
    const rows = d.measurements.map(m =>
      `<tr>
        <td>q${m.qubit}</td>
        <td style="color:${m.outcome ? '#3fb950' : '#f85149'};font-weight:bold">${m.outcome}</td>
        <td style="color:#8b949e">${m.step}</td>
      </tr>`
    ).join('');
    document.getElementById('measContent').innerHTML =
      `<table><tr style="color:#4d5560"><td>Qubit</td><td>Outcome</td><td>Step</td></tr>${rows}</table>`;
  }
}

// Auto-run the default Bell circuit on page load
window.addEventListener('DOMContentLoaded', () => setTimeout(runCircuit, 120));
</script>
</body>
</html>"##;

// ── Public API ────────────────────────────────────────────────────────────

/// Write a self-contained HTML dashboard to `output_path`.
///
/// The HTML file uses Chart.js (CDN) and is otherwise self-contained —
/// open it in any browser to view probability distribution, gate histogram,
/// qubit utilization, and circuit metrics.
pub fn generate_report(data: &DashboardData, output_path: &str) -> std::io::Result<()> {
    std::fs::write(output_path, render_html(data))
}

/// Render the full HTML string for embedding or serving.
pub fn render_html(data: &DashboardData) -> String {
    let analysis = &data.analysis;
    let result   = &data.result;

    // ── Probability distribution data ─────────────────────────────────────
    let states       = data.display_states(64);
    let state_labels = js_string_array(states.iter().map(|(l, _)| format!("|{}⟩", l)));
    let prob_vals    = js_number_array(states.iter().map(|(_, p)| format!("{:.6}", p)));

    // ── Gate histogram data ───────────────────────────────────────────────
    let gate_entries  = data.sorted_gate_histogram();
    let gate_labels   = js_string_array(gate_entries.iter().map(|(k, _)| k.clone()));
    let gate_counts   = js_number_array(gate_entries.iter().map(|(_, v)| v.to_string()));

    // ── Qubit utilization data ────────────────────────────────────────────
    let qubit_labels = js_string_array((0..analysis.num_qubits).map(|i| format!("q{}", i)));
    let qubit_utils  = js_number_array(analysis.qubit_utilization.iter().map(|u| u.to_string()));

    // ── Measurement rows ─────────────────────────────────────────────────
    let meas_rows = if result.measurements.is_empty() {
        r#"<tr><td colspan="3" style="text-align:center;color:#8b949e">No measurements</td></tr>"#
            .to_string()
    } else {
        result.measurements.iter()
            .map(|m| format!(
                "<tr><td>q{}</td><td style='color:{}'>{}</td><td style='color:#8b949e'>{}</td></tr>",
                m.qubit,
                if m.outcome { "#3fb950" } else { "#f85149" },
                m.outcome as u8,
                m.step
            ))
            .collect::<Vec<_>>()
            .join("\n")
    };

    // ── Badge helpers ─────────────────────────────────────────────────────
    let cf_badge = badge(analysis.has_control_flow);
    let cg_badge = if analysis.has_custom_gates {
        format!("<span class='badge badge-yes'>yes ({} def)</span>", analysis.custom_gate_defs)
    } else {
        badge(false)
    };

    // ── Build HTML ────────────────────────────────────────────────────────
    let mut html = String::with_capacity(24_000);

    // Head + static CSS
    html.push_str(HTML_HEAD);

    // Body header
    html.push_str(&format!(
        r#"<body>
<div class="header">
  <div class="header-title">⚛ AstraCore Dashboard</div>
  <div class="header-sub">Circuit: <span style="color:#e6edf3">{}</span> &nbsp;·&nbsp; AstraCore v0.1.0</div>
</div>
<div class="grid">
"#,
        escape_html(&data.source_path)
    ));

    // Card 1 — Circuit Metrics
    html.push_str(&format!(
        r#"  <div class="card">
    <div class="card-title">CIRCUIT METRICS</div>
    <table>
      <tr><td>Qubits</td>              <td class="val">{}</td></tr>
      <tr><td>Gate count</td>          <td class="val">{}</td></tr>
      <tr><td>Gate count (expanded)</td><td class="val">{}</td></tr>
      <tr><td>Circuit depth</td>       <td class="val">{}</td></tr>
      <tr><td>Two-qubit gates</td>     <td class="val">{}</td></tr>
      <tr><td>Entanglement ratio</td>  <td class="val">{:.1}%</td></tr>
      <tr><td>Measurements</td>        <td class="val">{}</td></tr>
      <tr><td>Control flow</td>        <td>{}</td></tr>
      <tr><td>Custom gates</td>        <td>{}</td></tr>
      <tr><td>Avg gates / qubit</td>   <td class="val">{:.2}</td></tr>
    </table>
    <div class="card-title" style="margin-top:18px">EXECUTION STATS</div>
    <table>
      <tr><td>Gates applied</td>       <td class="val">{}</td></tr>
      <tr><td>Branches taken</td>      <td class="val">{}</td></tr>
      <tr><td>Steps executed</td>      <td class="val">{}</td></tr>
    </table>
  </div>
"#,
        analysis.num_qubits,
        analysis.gate_count,
        analysis.expanded_gate_count,
        analysis.circuit_depth,
        analysis.two_qubit_gate_count,
        analysis.entanglement_ratio() * 100.0,
        analysis.measure_count,
        cf_badge,
        cg_badge,
        analysis.avg_gates_per_qubit(),
        result.gate_count,
        result.branch_count,
        result.steps_executed,
    ));

    // Card 2 — Probability Distribution
    html.push_str(r#"  <div class="card">
    <div class="card-title">PROBABILITY DISTRIBUTION</div>
    <canvas id="probChart"></canvas>
  </div>
"#);

    // Card 3 — Gate Histogram
    html.push_str(r#"  <div class="card">
    <div class="card-title">GATE HISTOGRAM</div>
    <canvas id="gateChart"></canvas>
  </div>
"#);

    // Card 4 — Qubit Utilization
    html.push_str(r#"  <div class="card">
    <div class="card-title">QUBIT UTILIZATION (gate touches)</div>
    <canvas id="qubitChart"></canvas>
  </div>
"#);

    // Card 5 — Measurements (full width)
    html.push_str(&format!(
        r#"  <div class="card full-width">
    <div class="card-title">MEASUREMENT OUTCOMES</div>
    <table>
      <tr style="color:#8b949e"><td>Qubit</td><td>Outcome</td><td>Step</td></tr>
      {}
    </table>
  </div>
</div>
"#,
        meas_rows
    ));

    // Chart.js scripts
    html.push_str(&format!(
        r#"<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script>
Chart.defaults.color = '#8b949e';
Chart.defaults.borderColor = '#30363d';

function makeChart(id, labels, data, color, yMax) {{
  new Chart(document.getElementById(id), {{
    type: 'bar',
    data: {{
      labels: labels,
      datasets: [{{ data: data,
        backgroundColor: data.map(v => v > 0.001 ? color + 'cc' : color + '33'),
        borderColor: color, borderWidth: 1 }}]
    }},
    options: {{
      plugins: {{ legend: {{ display: false }} }},
      scales: {{
        x: {{ grid: {{ color: '#21262d' }}, ticks: {{ font: {{ family: 'Courier New', size: 11 }} }} }},
        y: {{ grid: {{ color: '#21262d' }}, min: 0, ...(yMax ? {{max: yMax}} : {{}}) }}
      }}
    }}
  }});
}}

makeChart('probChart',  {state_labels}, {prob_vals},  '#58a6ff', 1);
makeChart('gateChart',  {gate_labels},  {gate_counts}, '#3fb950', null);
makeChart('qubitChart', {qubit_labels}, {qubit_utils}, '#d2991c', null);
</script>
</body></html>
"#,
        state_labels = state_labels,
        prob_vals    = prob_vals,
        gate_labels  = gate_labels,
        gate_counts  = gate_counts,
        qubit_labels = qubit_labels,
        qubit_utils  = qubit_utils,
    ));

    html
}

// ── Helpers ───────────────────────────────────────────────────────────────

fn js_string_array(iter: impl Iterator<Item = String>) -> String {
    let items: Vec<String> = iter.map(|s| format!("\"{}\"", s.replace('"', "\\\""))).collect();
    format!("[{}]", items.join(", "))
}

fn js_number_array(iter: impl Iterator<Item = String>) -> String {
    let items: Vec<String> = iter.collect();
    format!("[{}]", items.join(", "))
}

fn badge(value: bool) -> String {
    if value {
        "<span class='badge badge-yes'>yes</span>".to_string()
    } else {
        "<span class='badge badge-no'>no</span>".to_string()
    }
}

fn escape_html(s: &str) -> String {
    s.replace('&', "&amp;")
     .replace('<', "&lt;")
     .replace('>', "&gt;")
     .replace('"', "&quot;")
}

// ── Static HTML head + CSS ────────────────────────────────────────────────

const HTML_HEAD: &str = r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AstraCore Dashboard</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: #0d1117;
  color: #e6edf3;
  font-family: 'Courier New', Courier, monospace;
  padding: 24px;
  min-height: 100vh;
}
.header {
  border-bottom: 1px solid #30363d;
  padding-bottom: 16px;
  margin-bottom: 24px;
}
.header-title {
  font-size: 1.7em;
  font-weight: bold;
  color: #58a6ff;
  margin-bottom: 4px;
}
.header-sub { color: #8b949e; font-size: 0.88em; }

.grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
}
.card {
  background: #161b22;
  border: 1px solid #30363d;
  border-radius: 8px;
  padding: 20px;
}
.full-width { grid-column: 1 / -1; }

.card-title {
  color: #58a6ff;
  font-size: 0.78em;
  letter-spacing: 0.12em;
  margin-bottom: 14px;
  border-bottom: 1px solid #21262d;
  padding-bottom: 6px;
}
table { width: 100%; border-collapse: collapse; font-size: 0.88em; }
td { padding: 5px 6px; border-bottom: 1px solid #21262d; }
td:first-child { color: #8b949e; }
.val { color: #3fb950; font-weight: bold; }

canvas { max-height: 230px; width: 100% !important; }

.badge {
  display: inline-block;
  padding: 1px 8px;
  border-radius: 4px;
  font-size: 0.8em;
}
.badge-yes { background: #1a3a1a; color: #3fb950; }
.badge-no  { background: #2d1515; color: #f85149; }

@media (max-width: 800px) {
  .grid { grid-template-columns: 1fr; }
  .full-width { grid-column: 1; }
}
</style>
</head>
"#;
