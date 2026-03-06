//! Python bindings for AstraCore — high-performance hybrid Classical-Quantum
//! Runtime Engine.
//!
//! ## Installation
//! ```bash
//! cd astracore-py
//! pip install maturin
//! maturin develop   # installs into the current Python environment
//! ```
//!
//! ## Quick start
//! ```python
//! from astracore import Circuit, run_aql
//!
//! c = Circuit(2)
//! c.h(0)
//! c.cnot(0, 1)
//! c.measure_all()
//! result = c.run()
//! print(result.probabilities)   # [0.5, 0.0, 0.0, 0.5]  → |00⟩ and |11⟩
//! ```
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

// Use `::astracore::` to unambiguously refer to the crate (avoids E0659 with
// the local `fn astracore` PyModule init function of the same name).
use ::astracore::compiler::parse_source;
use ::astracore::compiler::qasm_import::run_qasm as _run_qasm;
use ::astracore::compiler::{analyze, AqlError};
use ::astracore::runtime::{execute, run_shots};
use ::astracore::simulator::{execute_mps, execute_clifford, execute_sparse};
use ::astracore::dashboard::{DashboardData, generate_report, serve as _serve, run_tui};
use ::astracore::core::gates::apply_single_qubit_gate as _apply_sqg;
use ::astracore::core::complex::Complex as _AqlComplex;
use ::astracore::plugins::{
    PluginRegistry, FnGatePlugin,
    run_with_plugins as _run_with_plugins,
    execute_with_plugins as _execute_with_plugins,
};

const PI:   f64 = std::f64::consts::PI;
const PI_2: f64 = std::f64::consts::FRAC_PI_2;

// ── Global gate registries ────────────────────────────────────────────────────

/// Matrix gate registry: name → (n_qubits, flat [re00, im00, re01, im01, …])
static MATRIX_GATES: std::sync::OnceLock<std::sync::Mutex<HashMap<String, (usize, Vec<f64>)>>>
    = std::sync::OnceLock::new();

/// AQL gate def registry: name → "GATE name n\n  body\nEND"
static AQL_GATES: std::sync::OnceLock<std::sync::Mutex<HashMap<String, String>>>
    = std::sync::OnceLock::new();

fn get_matrix_gates() -> &'static std::sync::Mutex<HashMap<String, (usize, Vec<f64>)>> {
    MATRIX_GATES.get_or_init(|| std::sync::Mutex::new(HashMap::new()))
}

fn get_aql_gates() -> &'static std::sync::Mutex<HashMap<String, String>> {
    AQL_GATES.get_or_init(|| std::sync::Mutex::new(HashMap::new()))
}

/// Prepend all `@gate`-registered AQL definitions to `source`.
fn prepend_aql_gate_defs(source: &str) -> String {
    let gates = get_aql_gates().lock().unwrap();
    if gates.is_empty() {
        return source.to_string();
    }
    let mut prefix = String::new();
    for def in gates.values() {
        prefix.push_str(def);
        prefix.push('\n');
    }
    format!("{}{}", prefix, source)
}

/// Build a `PluginRegistry` loaded with all `register_gate`-registered matrix gates.
fn build_plugin_registry() -> PluginRegistry {
    let mut reg = PluginRegistry::default();
    let gates = get_matrix_gates().lock().unwrap();
    for (name, (n_qubits, flat)) in gates.iter() {
        if *n_qubits == 1 && flat.len() == 8 {
            let mat: [[_AqlComplex; 2]; 2] = [
                [_AqlComplex::new(flat[0], flat[1]), _AqlComplex::new(flat[2], flat[3])],
                [_AqlComplex::new(flat[4], flat[5]), _AqlComplex::new(flat[6], flat[7])],
            ];
            let n = name.clone();
            reg.register_gate(Box::new(FnGatePlugin::new(&n, 1, move |state, qubits| {
                _apply_sqg(state, &mat, qubits[0]);
            })));
        }
    }
    reg
}

/// Execute a circuit N times with plugin support, collecting a shot histogram.
fn py_shot_result_with_plugins(
    py: Python<'_>,
    full_source: &str,
    registry: &PluginRegistry,
    n_shots: usize,
) -> PyResult<PyObject> {
    // Parse once (injecting plugin stubs); execute N times
    let program = ::astracore::plugins::integration::parse_source_with_plugins(
        full_source, registry,
    ).map_err(aql_err)?;
    let n_qubits = program.num_qubits;
    let mut counts: HashMap<String, usize> = HashMap::new();
    for _ in 0..n_shots {
        let r = _execute_with_plugins(&program, registry).map_err(aql_err)?;
        let measured: Vec<Option<bool>> = (0..n_qubits)
            .map(|q| r.measurements.iter().rev().find(|m| m.qubit == q).map(|m| m.outcome))
            .collect();
        if measured.iter().all(|o| o.is_some()) {
            let bs: String = measured.iter()
                .map(|o| if o.unwrap() { '1' } else { '0' })
                .collect();
            *counts.entry(bs).or_insert(0) += 1;
        }
    }
    Ok(Py::new(py, ShotSimResult { counts, n_shots, n_qubits })?.into_py(py))
}

// ── Error conversion ──────────────────────────────────────────────────────────

fn aql_err(e: AqlError) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(format!("{e}"))
}

// ── SimResult ─────────────────────────────────────────────────────────────────

/// Result of a single-shot quantum circuit execution.
///
/// Attributes
/// ----------
/// num_qubits : int
/// probabilities : list[float]
///     Final probability distribution over all 2^n basis states.
///     Index `i` = P(|i⟩) where i is the integer encoding of the bit-string.
/// measurements : list[tuple[int, bool]]
///     Measurement records in execution order: (qubit_index, outcome).
#[pyclass(name = "SimResult")]
pub struct SimResult {
    num_qubits:    usize,
    probabilities: Vec<f64>,
    measurements:  Vec<(usize, bool)>,
}

#[pymethods]
impl SimResult {
    #[getter]
    fn num_qubits(&self) -> usize { self.num_qubits }

    #[getter]
    fn probabilities(&self) -> Vec<f64> { self.probabilities.clone() }

    #[getter]
    fn measurements(&self) -> Vec<(usize, bool)> { self.measurements.clone() }

    /// Last measured outcome for `qubit`. Returns None if not yet measured.
    fn outcome(&self, qubit: usize) -> Option<bool> {
        self.measurements.iter().rev()
            .find(|(q, _)| *q == qubit)
            .map(|(_, o)| *o)
    }

    /// Returns the bitstring of all measurement outcomes (qubit 0 = leftmost),
    /// or None if any qubit was not measured.
    fn bitstring(&self) -> Option<String> {
        let measured: std::collections::HashSet<usize> =
            self.measurements.iter().map(|(q, _)| *q).collect();
        if (0..self.num_qubits).all(|q| measured.contains(&q)) {
            Some((0..self.num_qubits).map(|q| {
                if self.outcome(q).unwrap() { '1' } else { '0' }
            }).collect())
        } else {
            None
        }
    }

    /// Return probability for a basis state given as a bit-string, e.g. "01".
    fn prob_of(&self, bitstring: &str) -> PyResult<f64> {
        if bitstring.len() != self.num_qubits {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "bitstring length {} ≠ num_qubits {}", bitstring.len(), self.num_qubits
            )));
        }
        let idx: usize = bitstring.chars().rev().enumerate().try_fold(0, |acc, (i, c)| {
            match c {
                '0' => Ok(acc),
                '1' => Ok(acc | (1 << i)),
                _   => Err(pyo3::exceptions::PyValueError::new_err("bitstring must contain only '0' and '1'")),
            }
        })?;
        Ok(*self.probabilities.get(idx).unwrap_or(&0.0))
    }

    fn __repr__(&self) -> String {
        format!(
            "SimResult(num_qubits={}, measurements={} records)",
            self.num_qubits, self.measurements.len()
        )
    }
}

// ── ShotSimResult ─────────────────────────────────────────────────────────────

/// Result of running a circuit for multiple shots (statistical sampling).
///
/// Attributes
/// ----------
/// counts : dict[str, int]
///     Bitstring → number of times it was observed.
/// n_shots : int
/// n_qubits : int
#[pyclass(name = "ShotSimResult")]
pub struct ShotSimResult {
    counts:   HashMap<String, usize>,
    n_shots:  usize,
    n_qubits: usize,
}

#[pymethods]
impl ShotSimResult {
    #[getter]
    fn counts<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new_bound(py);
        for (k, v) in &self.counts {
            d.set_item(k, *v)?;
        }
        Ok(d)
    }

    #[getter]
    fn n_shots(&self) -> usize { self.n_shots }

    #[getter]
    fn n_qubits(&self) -> usize { self.n_qubits }

    /// Empirical probability estimate for `bitstring`.
    fn prob(&self, bitstring: &str) -> f64 {
        self.counts.get(bitstring).copied().unwrap_or(0) as f64 / self.n_shots as f64
    }

    /// All (bitstring, count) pairs sorted by count descending.
    fn most_common(&self) -> Vec<(String, usize)> {
        let mut v: Vec<(String, usize)> = self.counts.iter()
            .map(|(k, c)| (k.clone(), *c))
            .collect();
        v.sort_by(|a, b| b.1.cmp(&a.1));
        v
    }

    fn __repr__(&self) -> String {
        format!("ShotSimResult(n_shots={}, n_qubits={})", self.n_shots, self.n_qubits)
    }
}

// ── CircuitAnalysis ───────────────────────────────────────────────────────────

/// Static analysis result for an AQL program (no execution needed).
///
/// Attributes
/// ----------
/// num_qubits : int
/// gate_count : int
///     Top-level gate operations (CALL counts as 1).
/// expanded_gate_count : int
///     Gate count after expanding all CALL bodies.
/// measure_count : int
/// circuit_depth : int
///     Critical-path depth assuming unlimited parallelism.
/// two_qubit_gate_count : int
///     Multi-qubit gates — proxy for hardware entanglement cost.
/// is_clifford : bool
///     True if all gates belong to the Clifford gate set.
/// has_control_flow : bool
/// has_custom_gates : bool
/// custom_gate_defs : int
/// gate_histogram : dict[str, int]
///     Per-mnemonic gate counts.
/// qubit_utilization : list[int]
///     Number of gate references for each qubit.
#[pyclass(name = "CircuitAnalysis")]
pub struct PyCircuitAnalysis {
    num_qubits:           usize,
    gate_count:           usize,
    expanded_gate_count:  usize,
    measure_count:        usize,
    circuit_depth:        usize,
    two_qubit_gate_count: usize,
    is_clifford:          bool,
    has_control_flow:     bool,
    has_custom_gates:     bool,
    custom_gate_defs:     usize,
    gate_histogram:       std::collections::HashMap<String, usize>,
    qubit_utilization:    Vec<usize>,
}

impl PyCircuitAnalysis {
    fn from(a: ::astracore::compiler::CircuitAnalysis) -> Self {
        PyCircuitAnalysis {
            num_qubits:           a.num_qubits,
            gate_count:           a.gate_count,
            expanded_gate_count:  a.expanded_gate_count,
            measure_count:        a.measure_count,
            circuit_depth:        a.circuit_depth,
            two_qubit_gate_count: a.two_qubit_gate_count,
            is_clifford:          a.is_clifford,
            has_control_flow:     a.has_control_flow,
            has_custom_gates:     a.has_custom_gates,
            custom_gate_defs:     a.custom_gate_defs,
            gate_histogram:       a.gate_histogram,
            qubit_utilization:    a.qubit_utilization,
        }
    }
}

#[pymethods]
impl PyCircuitAnalysis {
    #[getter] fn num_qubits(&self)           -> usize { self.num_qubits }
    #[getter] fn gate_count(&self)           -> usize { self.gate_count }
    #[getter] fn expanded_gate_count(&self)  -> usize { self.expanded_gate_count }
    #[getter] fn measure_count(&self)        -> usize { self.measure_count }
    #[getter] fn circuit_depth(&self)        -> usize { self.circuit_depth }
    #[getter] fn two_qubit_gate_count(&self) -> usize { self.two_qubit_gate_count }
    #[getter] fn is_clifford(&self)          -> bool  { self.is_clifford }
    #[getter] fn has_control_flow(&self)     -> bool  { self.has_control_flow }
    #[getter] fn has_custom_gates(&self)     -> bool  { self.has_custom_gates }
    #[getter] fn custom_gate_defs(&self)     -> usize { self.custom_gate_defs }

    #[getter]
    fn gate_histogram<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new_bound(py);
        for (k, v) in &self.gate_histogram {
            d.set_item(k, *v)?;
        }
        Ok(d)
    }

    #[getter]
    fn qubit_utilization(&self) -> Vec<usize> { self.qubit_utilization.clone() }

    fn __repr__(&self) -> String {
        format!(
            "CircuitAnalysis(num_qubits={}, gate_count={}, depth={}, is_clifford={})",
            self.num_qubits, self.gate_count, self.circuit_depth, self.is_clifford
        )
    }
}

// ── Circuit ───────────────────────────────────────────────────────────────────

/// Quantum circuit builder.
///
/// Constructs a circuit gate-by-gate and runs it on a chosen backend.
///
/// Parameters
/// ----------
/// n_qubits : int
///     Number of qubits in the circuit.
///
/// Examples
/// --------
/// >>> from astracore import Circuit
/// >>> c = Circuit(3)
/// >>> c.h(0); c.cnot(0, 1); c.cnot(0, 2)
/// >>> c.measure_all()
/// >>> r = c.run()
/// >>> print(r.probabilities[0], r.probabilities[7])  # ~0.5 each
#[pyclass(name = "Circuit")]
pub struct Circuit {
    n_qubits:     usize,
    instructions: Vec<String>,
}

#[pymethods]
impl Circuit {
    #[new]
    pub fn new(n_qubits: usize) -> PyResult<Self> {
        if n_qubits == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "n_qubits must be ≥ 1"
            ));
        }
        Ok(Circuit { n_qubits, instructions: Vec::new() })
    }

    // ── Single-qubit gates ──────────────────────────────────────────────────

    fn h(&mut self, qubit: usize)  { self.instructions.push(format!("H {qubit}")); }
    fn x(&mut self, qubit: usize)  { self.instructions.push(format!("X {qubit}")); }
    fn y(&mut self, qubit: usize)  { self.instructions.push(format!("Y {qubit}")); }
    fn z(&mut self, qubit: usize)  { self.instructions.push(format!("Z {qubit}")); }
    fn s(&mut self, qubit: usize)  { self.instructions.push(format!("S {qubit}")); }
    fn t(&mut self, qubit: usize)  { self.instructions.push(format!("T {qubit}")); }

    fn rx(&mut self, qubit: usize, theta: f64) {
        self.instructions.push(format!("RX {qubit} {theta}"));
    }
    fn ry(&mut self, qubit: usize, theta: f64) {
        self.instructions.push(format!("RY {qubit} {theta}"));
    }
    fn rz(&mut self, qubit: usize, theta: f64) {
        self.instructions.push(format!("RZ {qubit} {theta}"));
    }
    fn phase(&mut self, qubit: usize, theta: f64) {
        self.instructions.push(format!("PHASE {qubit} {theta}"));
    }

    // ── Two-qubit gates ─────────────────────────────────────────────────────

    fn cnot(&mut self, control: usize, target: usize) {
        self.instructions.push(format!("CNOT {control} {target}"));
    }
    fn cz(&mut self, control: usize, target: usize) {
        self.instructions.push(format!("CZ {control} {target}"));
    }
    fn swap(&mut self, qubit_a: usize, qubit_b: usize) {
        self.instructions.push(format!("SWAP {qubit_a} {qubit_b}"));
    }

    // ── Three-qubit gates ───────────────────────────────────────────────────

    fn toffoli(&mut self, control0: usize, control1: usize, target: usize) {
        self.instructions.push(format!("TOFFOLI {control0} {control1} {target}"));
    }

    /// Toffoli (CCNOT) gate — alias for `toffoli()`.
    fn ccx(&mut self, control0: usize, control1: usize, target: usize) {
        self.instructions.push(format!("TOFFOLI {control0} {control1} {target}"));
    }

    // ── Custom gate call ────────────────────────────────────────────────────

    /// Call a named gate (user-defined AQL gate or plugin gate).
    ///
    /// Parameters
    /// ----------
    /// name : str
    ///     Gate name as declared with `GATE name n … END` in AQL.
    /// qubits : list[int]
    ///     Qubit indices to pass to the gate.
    fn call(&mut self, name: &str, qubits: Vec<usize>) {
        let args: Vec<String> = qubits.iter().map(|q| q.to_string()).collect();
        self.instructions.push(format!("CALL {} {}", name, args.join(" ")));
    }

    // ── Measurement & control ───────────────────────────────────────────────

    fn measure(&mut self, qubit: usize) {
        self.instructions.push(format!("MEASURE {qubit}"));
    }
    fn measure_all(&mut self) {
        self.instructions.push("MEASURE_ALL".to_string());
    }
    fn barrier(&mut self) {
        self.instructions.push("BARRIER".to_string());
    }

    // ── AQL source ──────────────────────────────────────────────────────────

    /// Return the AQL source code for this circuit.
    fn aql_source(&self) -> String {
        let mut lines = vec![format!("QREG {}", self.n_qubits)];
        lines.extend_from_slice(&self.instructions);
        lines.join("\n")
    }

    // ── Execution ───────────────────────────────────────────────────────────

    /// Run the circuit and return a result object.
    ///
    /// Parameters
    /// ----------
    /// backend : str, optional
    ///     One of "statevector" (default), "mps", "clifford", "sparse".
    /// bond_dim : int, optional
    ///     MPS bond dimension (default 128). Ignored for other backends.
    /// shots : int, optional
    ///     If set, run the circuit `shots` times and return a ShotSimResult.
    ///
    /// Returns
    /// -------
    /// SimResult or ShotSimResult
    #[pyo3(signature = (backend = "statevector", bond_dim = None, shots = None))]
    fn run(
        &self,
        py: Python<'_>,
        backend: &str,
        bond_dim: Option<usize>,
        shots: Option<usize>,
    ) -> PyResult<PyObject> {
        let source      = self.aql_source();
        let full_source = prepend_aql_gate_defs(&source);
        let has_matrix  = !get_matrix_gates().lock().unwrap().is_empty();

        if has_matrix && backend != "statevector" {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "register_gate() matrix plugins require the 'statevector' backend; \
                 use @gate decorator for other backends"
            ));
        }

        if let Some(n) = shots {
            if has_matrix {
                let reg = build_plugin_registry();
                return py_shot_result_with_plugins(py, &full_source, &reg, n);
            }
            let program = parse_source(&full_source).map_err(aql_err)?;
            let sr = run_shots(&program, n).map_err(aql_err)?;
            return Ok(Py::new(py, ShotSimResult {
                counts: sr.counts, n_shots: sr.n_shots, n_qubits: sr.n_qubits,
            })?.into_py(py));
        }

        if has_matrix {
            let reg  = build_plugin_registry();
            let exec = _run_with_plugins(&full_source, &reg).map_err(aql_err)?;
            return Ok(Py::new(py, SimResult {
                num_qubits:    exec.num_qubits,
                probabilities: exec.pre_measurement_probs.unwrap_or(exec.final_probabilities),
                measurements:  exec.measurements.iter().map(|m| (m.qubit, m.outcome)).collect(),
            })?.into_py(py));
        }

        let program = parse_source(&full_source).map_err(aql_err)?;
        let exec = match backend {
            "mps"      => execute_mps(&program, bond_dim.unwrap_or(128)).map_err(aql_err)?,
            "clifford" => execute_clifford(&program).map_err(aql_err)?,
            "sparse"   => execute_sparse(&program).map_err(aql_err)?,
            _          => execute(&program).map_err(aql_err)?,
        };
        Ok(Py::new(py, SimResult {
            num_qubits:    exec.num_qubits,
            probabilities: exec.pre_measurement_probs.unwrap_or(exec.final_probabilities),
            measurements:  exec.measurements.iter().map(|m| (m.qubit, m.outcome)).collect(),
        })?.into_py(py))
    }

    // ── Dashboard ───────────────────────────────────────────────────────────

    /// Generate a standalone HTML report and write it to `path`.
    ///
    /// Opens nicely in any browser — no server needed, Chart.js is embedded.
    ///
    /// Parameters
    /// ----------
    /// path : str
    ///     Output file path, e.g. "report.html".
    fn html_report(&self, path: &str) -> PyResult<()> {
        let data = build_dashboard_data(&self.aql_source(), path).map_err(aql_err)?;
        generate_report(&data, path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
    }

    /// Start the interactive browser dashboard (blocking).
    ///
    /// Opens an HTTP server at http://localhost:<port>.
    /// The interactive SPA lets you edit AQL and re-run circuits live.
    /// Press Ctrl-C to stop.
    ///
    /// Parameters
    /// ----------
    /// port : int, optional
    ///     TCP port to listen on (default 8080).
    #[pyo3(signature = (port = 8080))]
    fn serve(&self, port: u16) -> PyResult<()> {
        let data = build_dashboard_data(&self.aql_source(), "python").map_err(aql_err)?;
        _serve(data, port);
        Ok(())
    }

    /// Launch the terminal TUI dashboard (blocking).
    ///
    /// Requires a real terminal (TTY). Press q or Esc to quit.
    /// Shows probability distribution, gate histogram, and circuit metrics.
    fn dash(&self) -> PyResult<()> {
        let data = build_dashboard_data(&self.aql_source(), "python").map_err(aql_err)?;
        run_tui(&data)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    fn __repr__(&self) -> String {
        format!(
            "Circuit(n_qubits={}, {} instructions)",
            self.n_qubits, self.instructions.len()
        )
    }

    fn __len__(&self) -> usize { self.instructions.len() }
}

// ── Dashboard helper ──────────────────────────────────────────────────────────

fn build_dashboard_data(source: &str, label: &str) -> Result<DashboardData, AqlError> {
    let program = parse_source(source)?;
    let analysis = analyze(&program);
    let result   = execute(&program)?;
    Ok(DashboardData { source_path: label.to_string(), analysis, result, circuit_svg: String::new() })
}

// ── Module-level free functions ───────────────────────────────────────────────

/// Run an AQL source string and return a SimResult.
///
/// Parameters
/// ----------
/// source : str
///     Full AQL program text.
/// backend : str, optional
///     "statevector" (default), "mps", "clifford", "sparse".
/// bond_dim : int, optional
///     MPS bond dimension (default 128).
///
/// Returns
/// -------
/// SimResult
#[pyfunction]
#[pyo3(signature = (source, backend = "statevector", bond_dim = None))]
fn run_aql(
    py: Python<'_>,
    source: &str,
    backend: &str,
    bond_dim: Option<usize>,
) -> PyResult<PyObject> {
    let full_source = prepend_aql_gate_defs(source);
    let has_matrix  = !get_matrix_gates().lock().unwrap().is_empty();

    if has_matrix && backend != "statevector" {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "register_gate() matrix plugins require the 'statevector' backend; \
             use @gate decorator for other backends"
        ));
    }

    if has_matrix {
        let reg  = build_plugin_registry();
        let exec = _run_with_plugins(&full_source, &reg).map_err(aql_err)?;
        return Ok(Py::new(py, SimResult {
            num_qubits:    exec.num_qubits,
            probabilities: exec.pre_measurement_probs.unwrap_or(exec.final_probabilities),
            measurements:  exec.measurements.iter().map(|m| (m.qubit, m.outcome)).collect(),
        })?.into_py(py));
    }

    let program = parse_source(&full_source).map_err(aql_err)?;
    let exec = match backend {
        "mps"      => execute_mps(&program, bond_dim.unwrap_or(128)).map_err(aql_err)?,
        "clifford" => execute_clifford(&program).map_err(aql_err)?,
        "sparse"   => execute_sparse(&program).map_err(aql_err)?,
        _          => execute(&program).map_err(aql_err)?,
    };
    Ok(Py::new(py, SimResult {
        num_qubits:    exec.num_qubits,
        probabilities: exec.pre_measurement_probs.unwrap_or(exec.final_probabilities),
        measurements:  exec.measurements.iter().map(|m| (m.qubit, m.outcome)).collect(),
    })?.into_py(py))
}

/// Run shot-based sampling on an AQL program.
///
/// Parameters
/// ----------
/// source : str
///     Full AQL program text.
/// shots : int
///     Number of independent runs.
///
/// Returns
/// -------
/// ShotSimResult
#[pyfunction]
fn run_aql_shots(py: Python<'_>, source: &str, shots: usize) -> PyResult<PyObject> {
    let full_source = prepend_aql_gate_defs(source);
    let has_matrix  = !get_matrix_gates().lock().unwrap().is_empty();

    if has_matrix {
        let reg = build_plugin_registry();
        return py_shot_result_with_plugins(py, &full_source, &reg, shots);
    }

    let program = parse_source(&full_source).map_err(aql_err)?;
    let sr = run_shots(&program, shots).map_err(aql_err)?;
    Ok(Py::new(py, ShotSimResult {
        counts: sr.counts, n_shots: sr.n_shots, n_qubits: sr.n_qubits,
    })?.into_py(py))
}

/// Import and run an OpenQASM 2.0 program, returning a SimResult.
///
/// Parameters
/// ----------
/// source : str
///     OpenQASM 2.0 source code.
///
/// Returns
/// -------
/// SimResult
#[pyfunction]
fn run_qasm(py: Python<'_>, source: &str) -> PyResult<PyObject> {
    let exec = _run_qasm(source).map_err(aql_err)?;
    let result = SimResult {
        num_qubits:    exec.num_qubits,
        probabilities: exec.pre_measurement_probs.unwrap_or(exec.final_probabilities),
        measurements:  exec.measurements.iter()
                           .map(|m| (m.qubit, m.outcome))
                           .collect(),
    };
    Ok(Py::new(py, result)?.into_py(py))
}

/// Write an HTML dashboard report for any AQL source string to `path`.
///
/// Equivalent to `Circuit(...).html_report(path)` but works with raw AQL.
///
/// Parameters
/// ----------
/// source : str
///     Full AQL program text.
/// path : str
///     Output file path, e.g. "report.html".
#[pyfunction]
fn run_aql_report(source: &str, path: &str) -> PyResult<()> {
    let data = build_dashboard_data(source, path).map_err(aql_err)?;
    generate_report(&data, path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
}

/// Start the interactive browser dashboard for any AQL source string (blocking).
///
/// Opens an HTTP server at http://localhost:<port>.
/// The interactive SPA lets you edit AQL and re-run circuits live.
/// Press Ctrl-C to stop.
///
/// Parameters
/// ----------
/// source : str
///     Full AQL program text used as the initial circuit.
/// port : int, optional
///     TCP port (default 8080).
#[pyfunction]
#[pyo3(signature = (source, port = 8080))]
fn run_aql_serve(source: &str, port: u16) -> PyResult<()> {
    let data = build_dashboard_data(source, "python").map_err(aql_err)?;
    _serve(data, port);
    Ok(())
}

/// Launch the terminal TUI dashboard for any AQL source string (blocking).
///
/// Requires a real terminal (TTY). Press q or Esc to quit.
///
/// Parameters
/// ----------
/// source : str
///     Full AQL program text.
#[pyfunction]
fn run_aql_dash(source: &str) -> PyResult<()> {
    let data = build_dashboard_data(source, "python").map_err(aql_err)?;
    run_tui(&data)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

/// Load an AQL source file from disk and run it, returning a SimResult.
///
/// Parameters
/// ----------
/// path : str
///     Path to the `.aql` file.
/// backend : str, optional
///     "statevector" (default), "mps", "clifford", "sparse".
/// bond_dim : int, optional
///     MPS bond dimension (default 128).
///
/// Returns
/// -------
/// SimResult
#[pyfunction]
#[pyo3(signature = (path, backend = "statevector", bond_dim = None))]
fn run_file(
    py: Python<'_>,
    path: &str,
    backend: &str,
    bond_dim: Option<usize>,
) -> PyResult<PyObject> {
    let source = std::fs::read_to_string(path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(
            format!("cannot read '{}': {}", path, e)
        ))?;
    run_aql(py, &source, backend, bond_dim)
}

/// Load an AQL source file from disk and return static analysis (no execution).
///
/// Parameters
/// ----------
/// path : str
///     Path to the `.aql` file.
///
/// Returns
/// -------
/// CircuitAnalysis
#[pyfunction]
fn analyze_file(py: Python<'_>, path: &str) -> PyResult<PyObject> {
    let source = std::fs::read_to_string(path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(
            format!("cannot read '{}': {}", path, e)
        ))?;
    analyze_aql(py, &source)
}

/// Analyze an AQL source string (no execution), returning a CircuitAnalysis.
///
/// Parameters
/// ----------
/// source : str
///     Full AQL program text.
///
/// Returns
/// -------
/// CircuitAnalysis
#[pyfunction]
fn analyze_aql(py: Python<'_>, source: &str) -> PyResult<PyObject> {
    let program  = parse_source(source).map_err(aql_err)?;
    let analysis = analyze(&program);
    Ok(Py::new(py, PyCircuitAnalysis::from(analysis))?.into_py(py))
}

// ── Plugin gateway: register_gate + @gate decorator ──────────────────────────

/// Internal callable returned by `gate(n_qubits, name)`.
///
/// Using it as a decorator captures the gate body into the global AQL registry.
#[pyclass(name = "_GateDecorator")]
struct GateDecorator {
    n_qubits: usize,
    name:     Option<String>,
}

#[pymethods]
impl GateDecorator {
    fn __call__(&self, py: Python<'_>, func: Bound<'_, PyAny>) -> PyResult<PyObject> {
        let gate_name = match &self.name {
            Some(n) => n.clone(),
            None    => func.getattr("__name__")?.extract::<String>()?,
        };

        // Temporary Circuit to capture the gate body instructions.
        let temp = Py::new(py, Circuit { n_qubits: self.n_qubits, instructions: Vec::new() })?;

        // Call the user function: func(circuit, [0, 1, ..., n-1])
        let qubit_indices: Vec<usize> = (0..self.n_qubits).collect();
        func.call1((temp.clone_ref(py), qubit_indices))?;

        // Build "GATE name n\n  instr1\n  instr2\nEND"
        let body = temp
            .borrow(py)
            .instructions
            .iter()
            .map(|s| format!("  {}", s))
            .collect::<Vec<_>>()
            .join("\n");
        let aql_def = format!("GATE {} {}\n{}\nEND", gate_name, self.n_qubits, body);

        get_aql_gates()
            .lock()
            .unwrap()
            .insert(gate_name.to_lowercase(), aql_def);

        Ok(func.unbind())
    }
}

/// Register a custom single-qubit gate from a unitary matrix.
///
/// The gate can then be invoked via ``c.call(name, [q])`` or
/// ``CALL name q`` in AQL.  Only the **statevector** backend supports
/// matrix-registered gates; for MPS/Clifford/sparse use the ``@gate``
/// decorator instead.
///
/// Parameters
/// ----------
/// name : str
///     Gate name (case-insensitive).
/// matrix : array-like
///     2×2 complex unitary matrix (numpy array or list-of-lists of complex).
/// n_qubits : int, optional
///     Number of qubits. Currently only 1 is supported.  Default: ``1``.
///
/// Examples
/// --------
/// >>> import numpy as np
/// >>> import astracore as ac
/// >>> sqrt_x = np.array([[0.5+0.5j, 0.5-0.5j], [0.5-0.5j, 0.5+0.5j]])
/// >>> ac.register_gate("sqrt_x", sqrt_x)
/// >>> c = ac.Circuit(1)
/// >>> c.call("sqrt_x", [0])
/// >>> c.measure(0)
/// >>> result = c.run(shots=10000)
#[pyfunction]
#[pyo3(signature = (name, matrix, n_qubits = 1))]
fn register_gate(
    _py: Python<'_>,
    name: &str,
    matrix: &Bound<'_, PyAny>,
    n_qubits: usize,
) -> PyResult<()> {
    let flat = extract_complex_matrix(matrix)?;
    let expected_elems = 4usize.pow(n_qubits as u32); // 4 for 1-qubit, 16 for 2-qubit
    let expected_flat  = expected_elems * 2;
    if flat.len() != expected_flat {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "matrix for a {}-qubit gate must have {} complex elements (got {})",
            n_qubits, expected_elems, flat.len() / 2
        )));
    }
    get_matrix_gates()
        .lock()
        .unwrap()
        .insert(name.to_lowercase(), (n_qubits, flat));
    Ok(())
}

/// Decorator factory for defining custom named gates from Python functions.
///
/// The decorated function receives a temporary ``Circuit`` and a list of
/// local qubit indices ``[0, 1, ..., n_qubits-1]``.  Its gate calls are
/// captured and stored as an AQL ``GATE … END`` block that is prepended
/// to every subsequent execution.
///
/// Works with **all backends** and with shot-based sampling.
///
/// Parameters
/// ----------
/// n_qubits : int
///     Number of qubits the gate operates on.
/// name : str, optional
///     Gate name.  Defaults to the function's ``__name__``.
///
/// Examples
/// --------
/// >>> import astracore as ac
/// >>>
/// >>> @ac.gate(n_qubits=2, name="my_bell")
/// ... def bell_gate(c, qubits):
/// ...     c.h(qubits[0])
/// ...     c.cnot(qubits[0], qubits[1])
/// >>>
/// >>> circ = ac.Circuit(4)
/// >>> circ.call("my_bell", [0, 1])
/// >>> circ.call("my_bell", [2, 3])
/// >>> circ.measure_all()
/// >>> result = circ.run(shots=1000)
#[pyfunction]
#[pyo3(signature = (n_qubits, name = None))]
fn gate(n_qubits: usize, name: Option<String>) -> PyResult<GateDecorator> {
    Ok(GateDecorator { n_qubits, name })
}

fn extract_complex_matrix(matrix: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    let mut flat = Vec::new();
    for row in matrix.iter()? {
        let row = row?;
        for elem in row.iter()? {
            let elem = elem?;
            let (re, im) = extract_complex_elem(&elem)?;
            flat.push(re);
            flat.push(im);
        }
    }
    if flat.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "matrix must be a non-empty 2D array of complex numbers",
        ));
    }
    Ok(flat)
}

fn extract_complex_elem(elem: &Bound<'_, PyAny>) -> PyResult<(f64, f64)> {
    // Python complex or numpy complex128: has .real and .imag attributes
    if let (Ok(re), Ok(im)) = (
        elem.getattr("real").and_then(|v| v.extract::<f64>()),
        elem.getattr("imag").and_then(|v| v.extract::<f64>()),
    ) {
        return Ok((re, im));
    }
    // Plain float (real-only)
    if let Ok(re) = elem.extract::<f64>() {
        return Ok((re, 0.0));
    }
    // (re, im) tuple
    if let Ok((re, im)) = elem.extract::<(f64, f64)>() {
        return Ok((re, im));
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "matrix element must be a complex number, float, or (re, im) tuple",
    ))
}

// ── Module definition ─────────────────────────────────────────────────────────

/// AstraCore — high-performance hybrid Classical-Quantum Runtime Engine.
///
/// Classes
/// -------
/// Circuit          — quantum circuit builder (run, html_report, serve, dash, call, ccx)
/// SimResult        — single-shot execution result
/// ShotSimResult    — multi-shot sampling result
/// CircuitAnalysis  — static analysis result (no execution)
///
/// Constants
/// ---------
/// PI    — π  (3.14159…)
/// PI_2  — π/2 (1.5707…)
///
/// Simulation functions
/// --------------------
/// run_aql(source, backend="statevector", bond_dim=None)  → SimResult
/// run_aql_shots(source, shots)                           → ShotSimResult
/// run_qasm(source)                                       → SimResult
/// run_file(path, backend="statevector", bond_dim=None)   → SimResult
/// analyze_aql(source)                                    → CircuitAnalysis
/// analyze_file(path)                                     → CircuitAnalysis
///
/// Dashboard functions
/// -------------------
/// run_aql_report(source, path)                           → None  (HTML file)
/// run_aql_serve(source, port=8080)                       → None  (blocking HTTP)
/// run_aql_dash(source)                                   → None  (blocking TUI)
// The function name must match the .so library name (`astracore`).
// Since `astracore` is also the crate we depend on, we use `::astracore::...`
// imports above to disambiguate from this function.
#[pymodule]
fn astracore(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // classes
    m.add_class::<Circuit>()?;
    m.add_class::<SimResult>()?;
    m.add_class::<ShotSimResult>()?;
    m.add_class::<PyCircuitAnalysis>()?;
    m.add_class::<GateDecorator>()?;
    // simulation
    m.add_function(wrap_pyfunction!(run_aql, m)?)?;
    m.add_function(wrap_pyfunction!(run_aql_shots, m)?)?;
    m.add_function(wrap_pyfunction!(run_qasm, m)?)?;
    m.add_function(wrap_pyfunction!(run_file, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_aql, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_file, m)?)?;
    // plugin gateway
    m.add_function(wrap_pyfunction!(register_gate, m)?)?;
    m.add_function(wrap_pyfunction!(gate, m)?)?;
    // dashboard
    m.add_function(wrap_pyfunction!(run_aql_report, m)?)?;
    m.add_function(wrap_pyfunction!(run_aql_serve, m)?)?;
    m.add_function(wrap_pyfunction!(run_aql_dash, m)?)?;
    // constants
    m.add("PI",  PI)?;
    m.add("PI_2", PI_2)?;
    m.add("__version__", "0.1.0")?;
    Ok(())
}
