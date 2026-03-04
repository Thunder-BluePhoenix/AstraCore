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
<title>AstraCore Hybrid Classical–Quantum Runtime Engine</title>
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
/* ── Left panel ── */
.editor-panel {
  width: 40%; min-width: 260px; max-width: 520px;
  border-right: 1px solid #30363d;
  display: flex; flex-direction: column; flex-shrink: 0;
}
/* ── Tabs ── */
.tabs { display: flex; border-bottom: 1px solid #30363d; flex-shrink: 0; }
.tab {
  padding: 8px 16px; cursor: pointer; font-size: 0.8em;
  border-bottom: 2px solid transparent; color: #8b949e;
  transition: color 0.15s;
}
.tab.active { color: #58a6ff; border-bottom-color: #58a6ff; }
.tab:hover:not(.active) { color: #e6edf3; }
/* ── Editor view ── */
#editorView {
  display: flex; flex-direction: column; flex: 1; gap: 8px;
  padding: 10px; overflow: hidden;
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
.btn-save { background: #1a2e4a; border-color: #58a6ff; color: #58a6ff; }
.btn-save:hover  { background: #243a5e; }
.shots-row { display:flex; align-items:center; gap:5px; flex-shrink:0; }
.shots-input {
  width: 72px; background: #010409; border: 1px solid #30363d; border-radius: 4px;
  color: #e6edf3; font-family: 'Courier New', monospace; font-size: 0.8em;
  padding: 3px 6px; outline: none;
}
.shots-input:focus { border-color: #58a6ff60; }
.btn-shots { background: #2d1f4e; border-color: #a371f7; color: #a371f7; }
.btn-shots:hover { background: #3d2a6a; }
.btn-shots:disabled { opacity: 0.45; cursor: not-allowed; }
#editor {
  flex: 1; background: #010409; color: #e6edf3;
  border: 1px solid #30363d; border-radius: 6px;
  padding: 12px; font-family: 'Courier New', monospace;
  font-size: 13px; line-height: 1.65; resize: none; outline: none; tab-size: 2;
}
#editor:focus { border-color: #58a6ff60; }
.error-msg { min-height: 20px; color: #f85149; font-size: 0.8em; padding: 0 2px; word-break: break-all; flex-shrink: 0; }
/* ── Examples view ── */
#examplesView {
  display: none; flex: 1; overflow-y: auto;
  flex-direction: column; padding: 10px; gap: 7px;
}
.ex-hint { color: #4d5560; font-size: 0.76em; text-align: center; padding: 2px 0 6px; flex-shrink: 0; }
.ex-card {
  background: #161b22; border: 1px solid #30363d; border-radius: 6px;
  padding: 10px 12px; cursor: pointer;
  transition: border-color 0.15s, background 0.15s; flex-shrink: 0;
}
.ex-card:hover { border-color: #58a6ff88; background: #1c2230; }
.ex-name { color: #58a6ff; font-size: 0.85em; font-weight: bold; margin-bottom: 4px; }
.ex-desc { color: #8b949e; font-size: 0.77em; line-height: 1.4; }
.ex-meta { color: #4d5560; font-size: 0.72em; margin-top: 5px; }
/* ── Syntax view ── */
#syntaxView {
  display: none; flex: 1; overflow-y: auto; padding: 10px;
}
.syn-section { margin-bottom: 14px; }
.syn-title {
  color: #58a6ff; font-size: 0.72em; letter-spacing: 0.1em; text-transform: uppercase;
  border-bottom: 1px solid #21262d; padding-bottom: 4px; margin-bottom: 6px;
}
.syn-row {
  display: flex; gap: 10px; padding: 4px 2px;
  border-bottom: 1px solid #1c2128; font-size: 0.79em; line-height: 1.35;
}
.syn-code { color: #3fb950; white-space: nowrap; min-width: 148px; flex-shrink: 0; }
.syn-desc { color: #8b949e; }
.syn-note { color: #4d5560; font-size: 0.76em; padding: 5px 2px 1px; }
.syn-example {
  background: #010409; border: 1px solid #21262d; border-radius: 4px;
  margin-top: 7px; padding: 7px 10px;
  display: flex; align-items: flex-start; gap: 8px;
}
.syn-example code {
  font-family: 'Courier New', monospace; font-size: 0.76em; color: #3fb950;
  white-space: pre; line-height: 1.5; flex: 1;
}
.syn-try-btn {
  background: none; border: 1px solid #30363d; border-radius: 3px;
  color: #4d5560; font-size: 0.71em; padding: 2px 7px;
  cursor: pointer; white-space: nowrap;
  font-family: 'Courier New', monospace; flex-shrink: 0; margin-top: 1px;
}
.syn-try-btn:hover { color: #3fb950; border-color: #3fb950; }
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
/* ── Share URL toast ── */
#shareToast {
  position: fixed; bottom: 18px; right: 18px;
  background: #21262d; border: 1px solid #3fb950; color: #3fb950;
  border-radius: 6px; padding: 8px 14px; font-size: 0.8em;
  opacity: 0; transition: opacity 0.3s; pointer-events: none; z-index: 100;
}
#shareToast.show { opacity: 1; }
/* ── Mobile ── */
@media (max-width: 780px) {
  body { overflow: auto; }
  .app { flex-direction: column; }
  .editor-panel { width: 100% !important; max-width: 100%; border-right: none; border-bottom: 1px solid #30363d; min-height: 300px; }
  .results-panel { overflow: visible; }
  .results-grid { grid-template-columns: 1fr; }
}
</style>
</head>
<body>
<div class="header">
  <div class="header-title">⚛ AstraCore</div>
  <div class="header-sub">Hybrid Classical–Quantum Runtime Engine &nbsp;·&nbsp; write AQL, pick an Example, or browse the Syntax guide</div>
  <div class="header-api">GET /api/data &nbsp; POST /api/run</div>
</div>

<div class="app">

  <!-- ══ LEFT panel ══════════════════════════════════════════════════════ -->
  <div class="editor-panel">

    <!-- Tab bar -->
    <div class="tabs">
      <div class="tab active" data-tab="editor">Editor</div>
      <div class="tab"        data-tab="examples">Examples</div>
      <div class="tab"        data-tab="syntax">Syntax</div>
    </div>

    <!-- ── Editor view ── -->
    <div id="editorView">
      <div class="toolbar">
        <button class="btn" id="openBtn">📂 Open</button>
        <input type="file" id="fileInput" accept=".aql,.txt" hidden>
        <button class="btn btn-save" id="saveBtn">💾 Save</button>
        <button class="btn btn-save" id="shareBtn" title="Copy shareable URL to clipboard">🔗</button>
        <span class="filename" id="filename">bell.aql</span>
        <span class="kbd-hint">Ctrl+↵</span>
        <button class="btn btn-run" id="runBtn">▶ Execute</button>
      </div>
      <div class="shots-row" style="padding: 0 0 4px 0;">
        <span style="color:#8b949e;font-size:0.78em">Shots:</span>
        <input type="number" id="shotsInput" class="shots-input" value="1000" min="1" max="100000" title="Number of shots for statistical sampling">
        <button class="btn btn-shots" id="shotsBtn" title="Run N shots and show measurement histogram">🎲 Sample</button>
        <button class="btn btn-shots" id="stepsBtn" onclick="loadSteps()" title="Walk through gate-by-gate state evolution (use ◀▶ arrow keys)" style="background:#312e81;border-color:#4f46e5;color:#c7d2fe">⏭ Steps</button>
      </div>

      <textarea id="editor" spellcheck="false" autocomplete="off">// Bell State  (|00⟩ + |11⟩) / √2
// Ctrl+Enter to run — or pick an example from the Examples tab

QREG 2
H 0
CNOT 0 1
MEASURE_ALL</textarea>

      <div id="errorMsg" class="error-msg"></div>
    </div>

    <!-- ── Examples view ── -->
    <div id="examplesView">
      <div class="ex-hint">Click any example to load it into the editor and run it</div>
      <div id="exampleList"></div>
    </div>

    <!-- ── Syntax view ── -->
    <div id="syntaxView">

      <!-- Register -->
      <div class="syn-section">
        <div class="syn-title">Register</div>
        <div class="syn-row"><span class="syn-code">QREG n</span><span class="syn-desc">declare n qubits — required, must be the first instruction</span></div>
        <div class="syn-example"><code>QREG 3   // allocate 3 qubits: q0, q1, q2
H 0
MEASURE_ALL</code><button class="syn-try-btn" onclick="trySyntaxExample(this)">&#9654; try</button></div>
      </div>

      <!-- Single-Qubit Gates -->
      <div class="syn-section">
        <div class="syn-title">Single-Qubit Gates</div>
        <div class="syn-row"><span class="syn-code">H q</span><span class="syn-desc">Hadamard — creates equal superposition of |0⟩ and |1⟩</span></div>
        <div class="syn-row"><span class="syn-code">X q</span><span class="syn-desc">Pauli-X (NOT / bit-flip) — |0⟩↔|1⟩</span></div>
        <div class="syn-row"><span class="syn-code">Y q</span><span class="syn-desc">Pauli-Y — bit &amp; phase flip</span></div>
        <div class="syn-row"><span class="syn-code">Z q</span><span class="syn-desc">Pauli-Z — phase-flip (|1⟩ → -|1⟩)</span></div>
        <div class="syn-row"><span class="syn-code">S q</span><span class="syn-desc">S gate — phase √Z  (π/2 rotation around Z)</span></div>
        <div class="syn-row"><span class="syn-code">T q</span><span class="syn-desc">T gate — phase ⁴√Z (π/4 rotation around Z)</span></div>
        <div class="syn-note">All gates are unitary (reversible). Apply twice to return to original state (X·X = I, H·H = I, Z·Z = I).</div>
        <div class="syn-example"><code>QREG 3
H 0        // q0: superposition (P(0)=P(1)=0.5)
X 1        // q1: |0⟩ → |1⟩
Z 2        // q2: phase flip (no visible effect without H)
MEASURE_ALL</code><button class="syn-try-btn" onclick="trySyntaxExample(this)">&#9654; try</button></div>
      </div>

      <!-- Rotation Gates -->
      <div class="syn-section">
        <div class="syn-title">Rotation Gates</div>
        <div class="syn-row"><span class="syn-code">RX q theta</span><span class="syn-desc">rotate around X axis by theta radians</span></div>
        <div class="syn-row"><span class="syn-code">RY q theta</span><span class="syn-desc">rotate around Y axis by theta radians</span></div>
        <div class="syn-row"><span class="syn-code">RZ q theta</span><span class="syn-desc">rotate around Z axis by theta radians</span></div>
        <div class="syn-row"><span class="syn-code">PHASE q theta</span><span class="syn-desc">diagonal phase shift: |1⟩ → e^(iθ)|1⟩</span></div>
        <div class="syn-note">Built-in angle constants: PI (π) &nbsp; PI_2 (π/2) &nbsp; PI_4 (π/4) &nbsp; PI_8 (π/8) &nbsp; or any float e.g. 1.047</div>
        <div class="syn-example"><code>QREG 3
RX 0 PI     // RX(π) = X gate → q0 becomes |1⟩
RY 1 PI_2   // RY(π/2) → 50/50 superposition like H
H 2
RZ 2 PI     // H·RZ(π)·H = X via phase kickback
MEASURE_ALL</code><button class="syn-try-btn" onclick="trySyntaxExample(this)">&#9654; try</button></div>
      </div>

      <!-- Multi-Qubit Gates -->
      <div class="syn-section">
        <div class="syn-title">Multi-Qubit Gates</div>
        <div class="syn-row"><span class="syn-code">CNOT ctrl tgt</span><span class="syn-desc">controlled-NOT — flips target if control is |1⟩; primary entangler</span></div>
        <div class="syn-row"><span class="syn-code">CZ ctrl tgt</span><span class="syn-desc">controlled-Z — phase-flips target if control is |1⟩</span></div>
        <div class="syn-row"><span class="syn-code">SWAP a b</span><span class="syn-desc">swap the quantum states of two qubits</span></div>
        <div class="syn-row"><span class="syn-code">CCX c0 c1 tgt</span><span class="syn-desc">Toffoli (doubly-controlled-NOT) — quantum AND gate</span></div>
        <div class="syn-note">CNOT with H creates a Bell pair: H q0 → CNOT q0 q1 → (|00⟩+|11⟩)/√2</div>
        <div class="syn-example"><code>QREG 3
H 0
CNOT 0 1    // Bell pair on q0, q1
X 0
X 1
CCX 0 1 2   // Toffoli: flip q2 only if q0=q1=1
MEASURE_ALL</code><button class="syn-try-btn" onclick="trySyntaxExample(this)">&#9654; try</button></div>
      </div>

      <!-- Measurement -->
      <div class="syn-section">
        <div class="syn-title">Measurement</div>
        <div class="syn-row"><span class="syn-code">MEASURE q</span><span class="syn-desc">collapse qubit q, store result in classical bit q</span></div>
        <div class="syn-row"><span class="syn-code">MEASURE_ALL</span><span class="syn-desc">collapse every qubit at once</span></div>
        <div class="syn-note">After MEASURE, classical bits are available for IF/GOTO control flow. Results appear in the Measurement Outcomes panel.</div>
        <div class="syn-example"><code>QREG 2
H 0
CNOT 0 1
MEASURE 0     // measure only q0 (collapses entanglement)
// q1 is now determined but not yet recorded
MEASURE_ALL   // measure everything (q0 again + q1)</code><button class="syn-try-btn" onclick="trySyntaxExample(this)">&#9654; try</button></div>
      </div>

      <!-- Control Flow -->
      <div class="syn-section">
        <div class="syn-title">Control Flow</div>
        <div class="syn-row"><span class="syn-code">BARRIER</span><span class="syn-desc">synchronisation point — optimizer will not move gates across it</span></div>
        <div class="syn-row"><span class="syn-code">LABEL name</span><span class="syn-desc">define a named jump target (any identifier)</span></div>
        <div class="syn-row"><span class="syn-code">GOTO name</span><span class="syn-desc">unconditional jump to label</span></div>
        <div class="syn-row"><span class="syn-code">IF q GOTO name</span><span class="syn-desc">jump to label if classical bit q == 1 (set by MEASURE)</span></div>
        <div class="syn-note">Classical bits persist after MEASURE. IF jumps when the bit is 1. Use GOTO to skip over a block.</div>
        <div class="syn-example"><code>// Coin flip: if q0=1 flip q1
QREG 2
H 0
MEASURE 0
BARRIER
IF 0 GOTO do_flip
GOTO done
LABEL do_flip
X 1            // only runs when q0 measured 1
LABEL done
MEASURE 1</code><button class="syn-try-btn" onclick="trySyntaxExample(this)">&#9654; try</button></div>
      </div>

      <!-- Custom Gates -->
      <div class="syn-section">
        <div class="syn-title">Custom Gates</div>
        <div class="syn-row"><span class="syn-code">GATE name n</span><span class="syn-desc">begin gate definition with n local qubits numbered 0..n-1</span></div>
        <div class="syn-row"><span class="syn-code">END</span><span class="syn-desc">close the GATE block</span></div>
        <div class="syn-row"><span class="syn-code">CALL name q0 q1 …</span><span class="syn-desc">invoke a defined gate, mapping locals to global qubits</span></div>
        <div class="syn-note">GATE blocks can appear anywhere — even after the code that calls them. They are extracted before execution.</div>
        <div class="syn-example"><code>GATE entangle 2  // 2 local qubits
  H 0
  CNOT 0 1
END

QREG 4
CALL entangle 0 1  // Bell pair on q0, q1
CALL entangle 2 3  // Bell pair on q2, q3
MEASURE_ALL</code><button class="syn-try-btn" onclick="trySyntaxExample(this)">&#9654; try</button></div>
      </div>

      <!-- Comments -->
      <div class="syn-section">
        <div class="syn-title">Comments</div>
        <div class="syn-row"><span class="syn-code">// text</span><span class="syn-desc">line comment — everything after // is ignored</span></div>
        <div class="syn-note">Comments can appear on their own line or after an instruction. There are no block comments.</div>
      </div>

    </div><!-- /syntaxView -->

  </div><!-- /editor-panel -->

  <!-- ══ RIGHT: Results ══════════════════════════════════════════════════ -->
  <div class="results-panel">
    <div class="placeholder" id="placeholder">
      <div class="placeholder-icon">⚛</div>
      <div class="placeholder-text">Write AQL and click ▶ Execute to visualize</div>
    </div>

    <div id="results" style="display:none">
      <div class="results-grid">
        <div class="card">
          <div class="card-title">CIRCUIT METRICS</div>
          <div id="metricsContent"></div>
        </div>
        <div class="card">
          <div class="card-title">PROBABILITY DISTRIBUTION</div>
          <canvas id="probChart"></canvas>
        </div>
        <div class="card">
          <div class="card-title">GATE HISTOGRAM</div>
          <canvas id="gateChart"></canvas>
        </div>
        <div class="card">
          <div class="card-title">QUBIT UTILIZATION</div>
          <canvas id="qubitChart"></canvas>
        </div>
        <div class="card full-width">
          <div class="card-title">MEASUREMENT OUTCOMES</div>
          <div id="measContent"></div>
        </div>
        <div class="card full-width" id="shotsCard" style="display:none">
          <div class="card-title" id="shotsCardTitle">SHOT HISTOGRAM</div>
          <canvas id="shotsChart" style="max-height:220px"></canvas>
        </div>
        <div class="card full-width" id="circuitDiagramCard" style="display:none">
          <div class="card-title">⬡ CIRCUIT DIAGRAM</div>
          <div id="circuitDiagram" style="overflow-x:auto;text-align:center;padding:4px 0"></div>
        </div>
        <div class="card full-width" id="stepPlayer" style="display:none">
          <div class="card-title">⏭ STEP-BY-STEP EXECUTION <span id="stepCounter" style="font-weight:normal;color:#64748b;font-size:0.78em;margin-left:8px"></span></div>
          <div id="stepLabel" style="color:#94a3b8;font-size:0.88em;margin-bottom:10px;min-height:1.2em">—</div>
          <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px">
            <button class="btn btn-shots" onclick="prevStep()" style="padding:3px 12px;font-size:0.85em">◀ Prev</button>
            <input type="range" id="stepSlider" min="0" max="0" value="0"
                   oninput="goToStep(this.value)"
                   style="flex:1;accent-color:#818cf8;cursor:pointer">
            <button class="btn btn-shots" onclick="nextStep()" style="padding:3px 12px;font-size:0.85em">Next ▶</button>
          </div>
          <canvas id="stepChart" style="max-height:200px"></canvas>
        </div>
      </div>
    </div>
  </div>

</div><!-- /app -->
<div id="shareToast">✓ URL copied to clipboard</div>

<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script>
Chart.defaults.color = '#8b949e';
Chart.defaults.borderColor = '#21262d';

const editor    = document.getElementById('editor');
const runBtn    = document.getElementById('runBtn');
const errorMsg  = document.getElementById('errorMsg');
const filename  = document.getElementById('filename');
const fileInput = document.getElementById('fileInput');
const shotsBtn  = document.getElementById('shotsBtn');
const shotsInput= document.getElementById('shotsInput');

// ── URL hash sharing ───────────────────────────────────────────────
function encodeCircuit(src) {
  try { return btoa(unescape(encodeURIComponent(src))); } catch(e) { return ''; }
}
function decodeCircuit(hash) {
  try { return decodeURIComponent(escape(atob(hash))); } catch(e) { return null; }
}
(function loadFromHash() {
  var h = location.hash.slice(1);
  if (!h) return;
  var src = decodeCircuit(h);
  if (src) { editor.value = src; filename.textContent = 'shared.aql'; }
})();

// ── Save .aql ─────────────────────────────────────────────────────
document.getElementById('saveBtn').onclick = function() {
  var blob = new Blob([editor.value], { type: 'text/plain' });
  var a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = filename.textContent || 'circuit.aql';
  a.click();
  URL.revokeObjectURL(a.href);
};

// ── Share URL ─────────────────────────────────────────────────────
document.getElementById('shareBtn').onclick = function() {
  var url = location.origin + location.pathname + '#' + encodeCircuit(editor.value);
  navigator.clipboard.writeText(url).then(function() {
    var toast = document.getElementById('shareToast');
    toast.classList.add('show');
    setTimeout(function() { toast.classList.remove('show'); }, 2000);
  });
};

// ── Tab switching ──────────────────────────────────────────────────
function switchTab(name) {
  document.querySelectorAll('.tab').forEach(t => {
    t.classList.toggle('active', t.dataset.tab === name);
  });
  document.getElementById('editorView').style.display   = name === 'editor'   ? 'flex'  : 'none';
  document.getElementById('examplesView').style.display = name === 'examples' ? 'flex'  : 'none';
  document.getElementById('syntaxView').style.display   = name === 'syntax'   ? 'block' : 'none';
}
document.querySelectorAll('.tab').forEach(tab => {
  tab.addEventListener('click', () => switchTab(tab.dataset.tab));
});

// ── Syntax example try-it ──────────────────────────────────────────
function trySyntaxExample(btn) {
  var code = btn.parentElement.querySelector('code').textContent.trim();
  editor.value = code;
  filename.textContent = 'scratch.aql';
  errorMsg.textContent = '';
  switchTab('editor');
  runCircuit();
}

// ── File open ──────────────────────────────────────────────────────
document.getElementById('openBtn').onclick = () => fileInput.click();

fileInput.onchange = (e) => {
  const file = e.target.files[0];
  if (!file) return;
  filename.textContent = file.name;
  const reader = new FileReader();
  reader.onload = ev => { editor.value = ev.target.result; };
  reader.readAsText(file);
  fileInput.value = '';
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
  } else if (e.key === 'ArrowLeft' && _stepData) {
    e.preventDefault();
    prevStep();
  } else if (e.key === 'ArrowRight' && _stepData) {
    e.preventDefault();
    nextStep();
  }
});

runBtn.onclick = runCircuit;

// ── Execute ────────────────────────────────────────────────────────
async function runCircuit() {
  const source = editor.value.trim();
  if (!source) return;
  runBtn.disabled = true;
  runBtn.textContent = '\u23f3 Running\u2026';
  errorMsg.textContent = '';
  // Update URL hash for sharing
  history.replaceState(null, '', '#' + encodeCircuit(source));
  try {
    const res = await fetch('/api/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ source })
    });
    if (!res.ok) { errorMsg.textContent = '\u2717 HTTP ' + res.status; return; }
    const data = await res.json();
    if (data.error) {
      errorMsg.textContent = '\u2717 ' + data.error;
    } else {
      renderResults(data);
    }
  } catch (err) {
    errorMsg.textContent = '\u2717 ' + err.message;
  } finally {
    runBtn.disabled = false;
    runBtn.textContent = '\u25b6 Execute';
  }
}

// ── Shot sampling ──────────────────────────────────────────────────
shotsBtn.onclick = runShots;
async function runShots() {
  const source = editor.value.trim();
  if (!source) return;
  const shots = parseInt(shotsInput.value, 10) || 1000;
  shotsBtn.disabled = true;
  shotsBtn.textContent = '\u23f3\u2026';
  errorMsg.textContent = '';
  try {
    const res = await fetch('/api/shots', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ source, shots })
    });
    if (!res.ok) { errorMsg.textContent = '\u2717 HTTP ' + res.status; return; }
    const data = await res.json();
    if (data.error) {
      errorMsg.textContent = '\u2717 ' + data.error;
    } else {
      renderShotsResults(data);
    }
  } catch (err) {
    errorMsg.textContent = '\u2717 ' + err.message;
  } finally {
    shotsBtn.disabled = false;
    shotsBtn.textContent = '\uD83C\uDFB2 Sample';
  }
}

function renderShotsResults(d) {
  var card = document.getElementById('shotsCard');
  card.style.display = 'block';
  document.getElementById('shotsCardTitle').textContent =
    'SHOT HISTOGRAM — ' + d.n_shots.toLocaleString() + ' shots, ' + d.n_qubits + ' qubits';
  var entries = Object.entries(d.counts).sort((a, b) => a[0].localeCompare(b[0]));
  var labels = entries.map(e => '|' + e[0] + '\u27e9');
  var vals   = entries.map(e => e[1]);
  if (chartRefs.has('shotsChart')) { chartRefs.get('shotsChart').destroy(); }
  chartRefs.set('shotsChart', new Chart(document.getElementById('shotsChart'), {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        data: vals,
        backgroundColor: '#a371f7cc',
        borderColor: '#a371f7', borderWidth: 1,
        label: 'count'
      }]
    },
    options: {
      animation: { duration: 280 },
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: {
          label: ctx => ' ' + ctx.parsed.y + '  (' + (ctx.parsed.y / d.n_shots * 100).toFixed(1) + '%)'
        }}
      },
      scales: {
        x: { grid: { color: '#1c2128' }, ticks: { color: '#8b949e', font: { family: 'Courier New', size: 11 } } },
        y: { grid: { color: '#1c2128' }, ticks: { color: '#8b949e' }, min: 0 }
      }
    }
  }));
}

// ── Example circuits ───────────────────────────────────────────────
var EXAMPLES = [
  {
    name: 'Bell State',
    file: 'bell.aql',
    desc: 'Simplest maximally entangled 2-qubit state (\u27e800\u27e9+\u27e811\u27e9)/\u221a2. Both qubits always agree on measurement.',
    meta: '2 qubits \u00b7 2 gates \u00b7 introductory',
    source: [
      '// Bell State \u2014 (|00\u27e9 + |11\u27e9) / \u221a2',
      '//',
      '// The simplest maximally entangled 2-qubit state.',
      '// Measuring either qubit gives a random result,',
      '// but both qubits always agree.',
      '',
      'QREG 2',
      '',
      'H 0      // put q0 into superposition',
      'CNOT 0 1 // entangle q1 with q0',
      '',
      'MEASURE_ALL'
    ].join('\n')
  },
  {
    name: 'GHZ State',
    file: 'ghz.aql',
    desc: 'Greenberger\u2013Horne\u2013Zeilinger: 3-qubit maximal entanglement. Every run gives all-0 or all-1, never mixed.',
    meta: '3 qubits \u00b7 3 gates \u00b7 introductory',
    source: [
      '// GHZ State \u2014 (|000\u27e9 + |111\u27e9) / \u221a2',
      '//',
      '// 3-qubit maximal entanglement.',
      '// All qubits always agree on measurement.',
      '',
      'QREG 3',
      '',
      'H 0      // superposition on q0',
      'CNOT 0 1 // entangle q1',
      'CNOT 0 2 // entangle q2',
      '',
      'BARRIER',
      '',
      'MEASURE_ALL'
    ].join('\n')
  },
  {
    name: "Grover's Search",
    file: 'grover.aql',
    desc: 'Searches 4 items for the marked state |11\u27e9 in ONE oracle query. Optimal result: P(|11\u27e9) = 1.0.',
    meta: '2 qubits \u00b7 10 gates \u00b7 algorithm',
    source: [
      "// Grover's Search \u2014 2 qubits, target |11\u27e9",
      '//',
      '// One Grover iteration gives P(|11\u27e9) = 1.0.',
      '// CZ is the phase oracle that marks |11\u27e9.',
      '',
      'QREG 2',
      '',
      'BARRIER  // Step 1: Uniform superposition',
      'H 0',
      'H 1',
      '',
      'BARRIER  // Step 2: Oracle \u2014 phase-flip |11\u27e9',
      'CZ 0 1',
      '',
      'BARRIER  // Step 3: Grover diffuser',
      'H 0',
      'H 1',
      'X 0',
      'X 1',
      'CZ 0 1',
      'X 0',
      'X 1',
      'H 0',
      'H 1',
      '',
      'BARRIER  // Measure',
      'MEASURE_ALL  // expect: both qubits = 1'
    ].join('\n')
  },
  {
    name: 'Quantum Teleportation',
    file: 'teleport.aql',
    desc: 'Teleports |+\u27e9 from q0 (Alice) to q2 (Bob) via a shared Bell pair and classical corrections. P(q2=1) \u2248 0.5.',
    meta: '3 qubits \u00b7 control flow \u00b7 algorithm',
    source: [
      '// Quantum Teleportation',
      '//',
      '// Teleports |+\u27e9 from q0 (Alice) to q2 (Bob).',
      '// q1 is the shared entanglement qubit.',
      '//',
      '// Expected: q2 in state |+\u27e9 \u2014 P(|1\u27e9) \u2248 0.5',
      '',
      'QREG 3',
      '',
      '// Step 1 \u2014 prepare message qubit in |+\u27e9',
      'H 0',
      '',
      '// Step 2 \u2014 create Bell pair (Alice q1, Bob q2)',
      'H 1',
      'CNOT 1 2',
      '',
      'BARRIER',
      '',
      '// Step 3 \u2014 Bell measurement by Alice',
      'CNOT 0 1',
      'H 0',
      'MEASURE 0',
      'MEASURE 1',
      '',
      'BARRIER',
      '',
      '// Step 4a \u2014 if q1=1, Bob applies X',
      'IF 1 GOTO apply_x',
      'GOTO skip_x',
      'LABEL apply_x',
      'X 2',
      'LABEL skip_x',
      '',
      '// Step 4b \u2014 if q0=1, Bob applies Z',
      'IF 0 GOTO apply_z',
      'GOTO done',
      'LABEL apply_z',
      'Z 2',
      'LABEL done',
      '',
      'BARRIER',
      '',
      '// Step 5 \u2014 verify: P(|1\u27e9) \u2248 0.5',
      'MEASURE 2'
    ].join('\n')
  },
  {
    name: 'Deutsch Algorithm',
    file: 'deutsch.aql',
    desc: 'Determines in ONE query if f(x) is constant or balanced. q0=0 \u2192 constant, q0=1 \u2192 balanced.',
    meta: '2 qubits \u00b7 4 gates \u00b7 algorithm',
    source: [
      '// Deutsch Algorithm \u2014 Balanced Oracle',
      '//',
      '// ONE query determines if f(x) is constant or balanced.',
      '// CNOT below = balanced oracle f(x)=x.',
      '// Remove the CNOT for constant oracle f(x)=0.',
      '//',
      '// Result: q0=0 \u2192 constant,  q0=1 \u2192 balanced',
      '',
      'QREG 2',
      '',
      'X 1    // ancilla to |1\u27e9',
      'H 0    // query qubit into superposition',
      'H 1    // ancilla into |-\u27e9',
      '',
      'BARRIER',
      '',
      'CNOT 0 1  // balanced oracle: f(x) = x',
      '',
      'BARRIER',
      '',
      'H 0    // interfere',
      '',
      'MEASURE 0  // 0 = constant,  1 = balanced'
    ].join('\n')
  },
  {
    name: 'Rotation Gates',
    file: 'rotations.aql',
    desc: 'Showcases RX, RY, RZ with built-in angle constants (PI, PI_2, PI_4). Demonstrates phase kickback via H\u00b7Rz\u00b7H.',
    meta: '4 qubits \u00b7 parametric gates \u00b7 showcase',
    source: [
      '// Rotation Gate Showcase',
      '//',
      '// RX(PI)     on q0 \u2192 same as X gate (expect 1)',
      '// RY(PI_2)   on q1 \u2192 50/50 superposition',
      '// H\u00b7RZ(PI_4)\u00b7H on q2 \u2192 T-gate effect',
      '// H\u00b7RZ(PI)\u00b7H  on q3 \u2192 X gate via phase kickback',
      '',
      'QREG 4',
      '',
      'BARRIER  // Rx(\u03c0) = X gate',
      'RX 0 PI',
      'MEASURE 0',
      '',
      'BARRIER  // Ry(\u03c0/2) = equal superposition',
      'RY 1 PI_2',
      'MEASURE 1',
      '',
      'BARRIER  // H\u00b7Rz(\u03c0/4)\u00b7H',
      'H 2',
      'RZ 2 PI_4',
      'H 2',
      'MEASURE 2',
      '',
      'BARRIER  // H\u00b7Z\u00b7H = X',
      'H 3',
      'RZ 3 PI',
      'H 3',
      'MEASURE 3'
    ].join('\n')
  },
  {
    name: 'Toffoli Gate',
    file: 'toffoli.aql',
    desc: 'CCX (doubly-controlled-NOT) flips the target only when BOTH controls are |1\u27e9. Acts as a quantum AND gate.',
    meta: '3 qubits \u00b7 3 gates \u00b7 showcase',
    source: [
      '// Toffoli Gate (CCX) \u2014 Quantum AND',
      '//',
      '// Flips target q2 only when BOTH c0=1 and c1=1.',
      '// Expected output: all three qubits = 1.',
      '',
      'QREG 3',
      '',
      'X 0  // c0 = |1\u27e9',
      'X 1  // c1 = |1\u27e9',
      '     // t  = |0\u27e9  (default)',
      '',
      'BARRIER',
      '',
      'CCX 0 1 2  // Toffoli: flip q2 because q0=1 and q1=1',
      '',
      'BARRIER',
      '',
      'MEASURE_ALL  // expect: q0=1, q1=1, q2=1'
    ].join('\n')
  },
  {
    name: 'Custom Gates',
    file: 'custom_gates.aql',
    desc: 'Defines reusable GATE blocks (bell, ghz, flip) and invokes them with CALL on a 6-qubit register.',
    meta: '6 qubits \u00b7 custom gates \u00b7 advanced',
    source: [
      '// Custom Gate Definitions',
      '//',
      '// GATE blocks can appear anywhere in the file.',
      '// CALL invokes them on global qubits.',
      '',
      'GATE bell 2',
      '  H 0',
      '  CNOT 0 1',
      'END',
      '',
      'GATE ghz 3',
      '  H 0',
      '  CNOT 0 1',
      '  CNOT 0 2',
      'END',
      '',
      'GATE flip 1',
      '  X 0',
      'END',
      '',
      'QREG 6',
      '',
      '// Bell pair on qubits 0,1  \u2192 P(|00\u27e9)=0.5, P(|11\u27e9)=0.5',
      'CALL bell 0 1',
      '',
      '// GHZ state on qubits 2,3,4',
      'CALL ghz 2 3 4',
      '',
      '// Flip qubit 5  \u2192 always measures 1',
      'CALL flip 5',
      '',
      'MEASURE_ALL'
    ].join('\n')
  }
];

// Build example cards
var exList = document.getElementById('exampleList');
EXAMPLES.forEach(function(ex) {
  var card = document.createElement('div');
  card.className = 'ex-card';
  card.innerHTML =
    '<div class="ex-name">' + ex.name + '</div>' +
    '<div class="ex-desc">' + ex.desc + '</div>' +
    '<div class="ex-meta">' + ex.meta + '</div>';
  card.addEventListener('click', function() {
    editor.value = ex.source;
    filename.textContent = ex.file;
    errorMsg.textContent = '';
    switchTab('editor');
    runCircuit();
  });
  exList.appendChild(card);
});

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

  var cf = d.has_control_flow
    ? '<span class="badge-yes">yes</span>'
    : '<span class="badge-no">no</span>';
  var cg = d.has_custom_gates
    ? '<span class="badge-yes">yes (' + d.custom_gate_defs + ' defs)</span>'
    : '<span class="badge-no">no</span>';
  var entPct = d.gate_count > 0
    ? (d.two_qubit_gate_count / d.gate_count * 100).toFixed(1) + '%'
    : '0%';

  document.getElementById('metricsContent').innerHTML =
    '<table>' +
    '<tr><td>Qubits</td>         <td class="val">' + d.num_qubits + '</td></tr>' +
    '<tr><td>Gate count</td>     <td class="val">' + d.gate_count + '</td></tr>' +
    '<tr><td>Expanded gates</td> <td class="val">' + d.expanded_gate_count + '</td></tr>' +
    '<tr><td>Circuit depth</td>  <td class="val">' + d.circuit_depth + '</td></tr>' +
    '<tr><td>2-qubit gates</td>  <td class="val">' + d.two_qubit_gate_count + ' (' + entPct + ')</td></tr>' +
    '<tr><td>Measurements</td>   <td class="val">' + d.measure_count + '</td></tr>' +
    '<tr><td>Control flow</td>   <td>' + cf + '</td></tr>' +
    '<tr><td>Custom gates</td>   <td>' + cg + '</td></tr>' +
    '<tr><td colspan="2" style="padding:3px 0"></td></tr>' +
    '<tr><td>Gates applied</td>  <td class="val">' + d.exec_gate_count + '</td></tr>' +
    '<tr><td>Branches taken</td> <td class="val">' + d.branch_count + '</td></tr>' +
    '<tr><td>Steps executed</td> <td class="val">' + d.steps_executed + '</td></tr>' +
    '</table>';

  updateChart('probChart',
    d.significant_states.map(s => '|' + s.state + '\u27e9'),
    d.significant_states.map(s => s.prob),
    '#58a6ff', 1);

  var gateEntries = Object.entries(d.gate_histogram).sort((a, b) => b[1] - a[1]);
  updateChart('gateChart',
    gateEntries.map(e => e[0]),
    gateEntries.map(e => e[1]),
    '#3fb950', null);

  updateChart('qubitChart',
    d.qubit_utilization.map((_, i) => 'q' + i),
    d.qubit_utilization,
    '#d2991c', null);

  if (d.measurements.length === 0) {
    document.getElementById('measContent').innerHTML =
      '<p style="color:#4d5560;text-align:center;padding:8px">No measurements in this circuit</p>';
  } else {
    var rows = d.measurements.map(m =>
      '<tr>' +
      '<td>q' + m.qubit + '</td>' +
      '<td style="color:' + (m.outcome ? '#3fb950' : '#f85149') + ';font-weight:bold">' + m.outcome + '</td>' +
      '<td style="color:#8b949e">' + m.step + '</td>' +
      '</tr>'
    ).join('');
    document.getElementById('measContent').innerHTML =
      '<table><tr style="color:#4d5560"><td>Qubit</td><td>Outcome</td><td>Step</td></tr>' + rows + '</table>';
  }

  // Circuit diagram
  if (d.circuit_svg) {
    document.getElementById('circuitDiagram').innerHTML = d.circuit_svg;
    document.getElementById('circuitDiagramCard').style.display = '';
  }
}

// ── Step-by-step player ─────────────────────────────────────────────
var _stepData = null, _stepIdx = 0;

async function loadSteps() {
  var src = document.getElementById('editor').value.trim();
  if (!src) return;
  var btn = document.getElementById('stepsBtn');
  btn.disabled = true; btn.textContent = '\u2026';
  var errorMsg = document.getElementById('errorMsg');
  errorMsg.textContent = '';
  try {
    var res = await fetch('/api/steps', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ source: src })
    });
    var d = await res.json();
    if (d.error) {
      errorMsg.textContent = '\u2717 ' + d.error;
      return;
    }
    _stepData = d;
    document.getElementById('stepSlider').max = d.steps.length - 1;
    document.getElementById('stepPlayer').style.display = '';
    goToStep(0);
  } catch (err) {
    errorMsg.textContent = '\u2717 ' + err.message;
  } finally {
    btn.disabled = false; btn.textContent = '\u23ed Steps';
  }
}

function goToStep(idx) {
  if (!_stepData) return;
  _stepIdx = Math.max(0, Math.min(parseInt(idx), _stepData.steps.length - 1));
  var s = _stepData.steps[_stepIdx];
  var total = _stepData.steps.length;
  document.getElementById('stepLabel').textContent = s.label;
  document.getElementById('stepCounter').textContent = '(' + (_stepIdx + 1) + ' / ' + total + ')';
  document.getElementById('stepSlider').value = _stepIdx;
  var n = _stepData.n_qubits;
  var labels = s.probabilities.map(function(_, i) {
    return '|' + i.toString(2).padStart(n, '0') + '\u27e9';
  });
  updateChart('stepChart', labels, s.probabilities, '#818cf8', 1.0);
}

function prevStep() { if (_stepIdx > 0) goToStep(_stepIdx - 1); }
function nextStep() { if (_stepData && _stepIdx < _stepData.steps.length - 1) goToStep(_stepIdx + 1); }

// Auto-run on page load (shared URL or default Bell circuit)
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
