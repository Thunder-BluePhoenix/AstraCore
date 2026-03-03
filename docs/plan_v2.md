# AstraCore v2 — Plan & Vision

> Building on the complete v1.0 foundation (247 tests, all phases delivered),
> v2 targets three strategic pillars: **simulation scale**, **language accessibility**,
> and **market positioning**.

---

## v2 Delivery Status

| Phase | Description | Status | Tests Added |
|-------|-------------|--------|-------------|
| Phase 1 | Benchmark Suite (Criterion + Qiskit comparison) | ✅ **SHIPPED** | — (benches) |
| Phase 2 | MPS Backend (50–200+ qubits) | ✅ **SHIPPED** | +19 (8 SVD + 11 MPS) |
| Phase 3 | Clifford Simulator (unlimited Clifford qubits) | ✅ **SHIPPED** | +16 |
| Phase 3b | Sparse Statevector Backend | ✅ **SHIPPED** | +12 sparse |
| Phase 3c | Shot-Based Sampling (`--shots N`) | ✅ **SHIPPED** | +5 shots |
| Phase 3d | Auto-detect Clifford circuits (`is_clifford`) | ✅ **SHIPPED** | +5 analysis |
| Phase 4 | Python API (PyO3 / astracore-py) | ✅ **SHIPPED** | — (PyO3 bindings) |
| Phase 5 | AQL v2 Language | ✅ **COMPLETE** | +27 |
| Phase 6 | Dashboard v2 (shots histogram, URL sharing, save, mobile) | ✅ **SHIPPED** (partial) | — |
| Phase 7 | OpenQASM 2.0 Bridge | ✅ **COMPLETE** | +9 export + 12 import |

**Test count:** 352 tests + 7 doctests (was 316 + 7 before this session). All passing.

### Phase 3b — Sparse Statevector Backend
| Feature | Status |
|---------|--------|
| `src/simulator/sparse.rs` — `SparseState` + `execute_sparse()` | ✅ Shipped |
| `HashMap<u64, Complex>` basis: H, X, Y, Z, S, T, Rx, Ry, Rz, Phase, CNOT, CZ, SWAP, Toffoli | ✅ |
| Amplitude pruning (< 1e-15 discarded) | ✅ |
| Measurement with collapse + renormalization | ✅ |
| Custom gate body expansion (CallGate) | ✅ |
| Full IF/GOTO/LABEL control flow | ✅ |
| `--backend sparse` CLI flag | ✅ |
| 12 unit + integration tests (including dense-comparison test) | ✅ |

### Phase 3c — Shot-Based Sampling
| Feature | Status |
|---------|--------|
| `src/runtime/shots.rs` — `ShotResult` + `run_shots()` | ✅ Shipped |
| Bitstring histogram (sorted display with bar chart) | ✅ |
| `--shots N` CLI flag on `astracore run` | ✅ |
| Works with all backends (statevector, mps, clifford, sparse) | ✅ |
| `prob()`, `sorted_counts()` helper methods | ✅ |
| 5 unit tests | ✅ |

### Phase 3d — Auto-detect Clifford
| Feature | Status |
|---------|--------|
| `is_clifford: bool` field in `CircuitAnalysis` | ✅ Shipped |
| Clifford set: H, X, Y, Z, S, CNOT, CZ, SWAP, Measure/MeasureAll, Barrier, Labels | ✅ |
| Non-Clifford: T, Rx, Ry, Rz, Phase, Toffoli, CallGate | ✅ |
| `report()` shows "Clifford-only: yes ✓ (can use --backend clifford)" hint | ✅ |
| 5 tests (pure Clifford, T-gate, rotation, Toffoli, GHZ) | ✅ |

### Phase 4 — Python API Breakdown
| Feature | Status |
|---------|--------|
| `astracore-py/Cargo.toml` — cdylib crate with pyo3 + astracore path dep | ✅ Shipped |
| `astracore-py/pyproject.toml` — maturin build config | ✅ Shipped |
| `Circuit` class — gate builder + `run(backend, bond_dim, shots)` | ✅ Shipped |
| `Circuit.html_report(path)` — write standalone HTML dashboard | ✅ Shipped |
| `Circuit.serve(port=8080)` — blocking interactive browser SPA | ✅ Shipped |
| `Circuit.dash()` — blocking terminal TUI dashboard | ✅ Shipped |
| `Circuit.call(name, qubits)` — call custom/plugin gate | ✅ Shipped |
| `Circuit.ccx(c0, c1, t)` — Toffoli alias | ✅ Shipped |
| `SimResult` — `probabilities`, `measurements`, `outcome()`, `bitstring()`, `prob_of()` | ✅ Shipped |
| `ShotSimResult` — `counts`, `n_shots`, `prob()`, `most_common()` | ✅ Shipped |
| `CircuitAnalysis` class — `circuit_depth`, `gate_count`, `is_clifford`, etc. | ✅ Shipped |
| `run_aql(source, backend, bond_dim)` free function | ✅ Shipped |
| `run_aql_shots(source, shots)` free function | ✅ Shipped |
| `run_qasm(source)` free function | ✅ Shipped |
| `run_file(path, backend, bond_dim)` — load and run .aql file | ✅ Shipped |
| `analyze_file(path)` — static analysis without execution | ✅ Shipped |
| `run_aql_report(source, path)` — HTML report free function | ✅ Shipped |
| `run_aql_serve(source, port)` — browser SPA free function | ✅ Shipped |
| `run_aql_dash(source)` — terminal TUI free function | ✅ Shipped |
| `PI`, `PI_2` module constants | ✅ Shipped |
| `examples/python/` — 9 demo scripts + README | ✅ Shipped |
| `@gate` decorator for custom gate definition | ✅ Shipped |
| `register_gate(name, matrix=numpy_array)` with NumPy integration | ✅ Shipped |

### Phase 5 — AQL v2 Breakdown
| Feature | Status |
|---------|--------|
| `REPEAT N … END` compile-time loop unrolling | ✅ Shipped |
| `INCLUDE "file.aql"` directive (max depth 16) | ✅ Shipped |
| Parser qubit limit raised 30 → 1000 | ✅ Shipped |
| `QREG name[n]` named qubit registers (`H data[0]`, `CNOT data[0] ancilla[1]`) | ✅ Shipped |
| `IFMEASURED q THEN … END` sugar (→ IFNOT/GOTO/LABEL) | ✅ Shipped |
| `IFNOTMEASURED q THEN … END` sugar (→ IF/GOTO/LABEL) | ✅ Shipped |
| `src/compiler/error.rs` Diagnostic engine + "did you mean?" | ✅ Shipped |
| VS Code LSP extension | ⬜ Deferred |

### Phase 7 — OpenQASM Bridge Breakdown
| Feature | Status |
|---------|--------|
| OpenQASM 2.0 exporter (`to_qasm()`, `source_to_qasm()`) | ✅ Shipped |
| `astracore export <file.aql> [out.qasm]` CLI | ✅ Shipped |
| Custom GATE…END → QASM `gate` declarations | ✅ Shipped |
| OpenQASM 2.0 importer (`from_qasm()`, `run_qasm()`) | ✅ Shipped |
| `astracore import <file.qasm>` CLI | ✅ Shipped |
| Supported gates: H,X,Y,Z,S,T,sdg,tdg,cx,cy,cz,ch,swap,ccx,rx,ry,rz,p,u1,u2,u3,cswap | ✅ |
| Angle expressions: pi, pi/2, 2*pi, numeric literals, negation | ✅ |
| Multiple `qreg` declarations | ✅ |
| 12 unit tests | ✅ |
| OpenQASM 3.0 importer | ⬜ Deferred |

---

## Status Entering v2

| v1 Deliverable | Status |
|---|---|
| Core simulator (state vector, SIMD/AVX2) | ✅ Complete |
| AQL compiler (lexer → parser → IR) | ✅ Complete |
| Hybrid runtime (IF/GOTO/LABEL, scheduler) | ✅ Complete |
| Peephole optimizer | ✅ Complete |
| Custom gate definitions (GATE/END/CALL) | ✅ Complete |
| Circuit analysis module | ✅ Complete |
| Plugin architecture (gate/optimizer/backend) | ✅ Complete |
| Dashboard (TUI + HTML report + Web server) | ✅ Complete |

v1 limit: **~20 qubits** (state vector RAM: 2²⁰ × 16 bytes = 16 MB — fine)
v1 wall: **~30 qubits** (2³⁰ × 16 bytes = 16 GB — machine RAM limit)
v1 impossible: **50+ qubits** (2⁵⁰ × 16 bytes = 16 petabytes)

v2 breaks this wall without hardware, using simulation algorithms.

---

## Pillar 1 — Simulation Scale: 50-100+ Qubits

### Why 50+ is possible without hardware

The state vector approach is not the only way to simulate quantum circuits.
Different algorithms trade accuracy or generality for exponentially better scaling:

---

### 1.1 Matrix Product States (MPS) — Primary Target ✅ SHIPPED

**What it is:**
Instead of storing 2ⁿ amplitudes, the quantum state is stored as a chain of tensors
(a tensor network). Each tensor has a "bond dimension" χ that grows with entanglement.

```
|ψ⟩ = A[0] ── A[1] ── A[2] ── ... ── A[n-1]
       (χ)    (χ×χ)   (χ×χ)           (χ)
```

**Why it works:**
- For low-entanglement circuits (most real algorithms), χ stays small (64–512)
- Memory: O(n × χ²) instead of O(2ⁿ)
- Can simulate **50–100+ qubits** for shallow / low-entanglement circuits
- Exact for product states (χ=1), approximate for highly entangled states

**Where it excels:**
- 1D spin chains, QAOA with few layers, VQE, Deutsch, Grover (low qubit count per entanglement zone)
- Any circuit where entanglement stays local

**Where it fails:**
- Deep circuits that spread entanglement globally — χ grows exponentially, reverts to state vector cost
- Random circuits (Google's quantum supremacy claim)

**What was implemented (`src/simulator/mps.rs` + `svd.rs`):**
```
MpsState { tensors: Vec<Tensor3>, bond_dim: usize }
MpsState::new(n_qubits, max_bond_dim)
apply_single_qubit_gate(qubit, matrix)   — O(χ²) per gate
apply_two_qubit_gate(q0, q1, matrix)     — SVD truncation at bond_dim
  └── non-adjacent: SWAP-routed to adjacent pair, then un-routed
sample_measurement(qubit, rng)           — project + renormalize (tensors[0] only)
execute_mps(program, bond_dim)           — full AQL executor
complex_svd_thin(m, n, mat, max_rank)    — one-sided Jacobi SVD (no ndarray dep)
```

**Note:** Implemented without `ndarray` dependency — custom flat row-major tensors.

**CLI usage:**
```bash
astracore run --backend mps --bond-dim 128 circuit_50q.aql
astracore run --backend mps --bond-dim 256 grover_50q.aql
```

---

### 1.2 Clifford / Stabilizer Simulation — Arbitrary Qubit Count ✅ SHIPPED

**What it is:**
The Gottesman-Knill theorem proves that circuits using only the **Clifford gate set**
(H, S, CNOT, X, Y, Z, CZ, SWAP, Pauli measurements) can be simulated in **polynomial time**
on a classical computer, regardless of qubit count.

**Representation:** A stabilizer state is described by n Pauli operators (n×2n binary matrix + phases).

**Scaling:**
- Memory: O(n²) bits
- Gate cost: O(n) per Clifford gate
- Measurement: O(n)

**Simulates 10,000+ qubits** for Clifford-only circuits.

**Where it excels:**
- Error correction codes (Surface code, Steane code) — 100% Clifford
- Entanglement analysis
- Circuit equivalence checking
- Testing whether a circuit is Clifford-dominated

**Where it fails:**
- Circuits with T gates (non-Clifford) — requires approximation (magic state injection)
- Anything that needs rotation by irrational angles

**What was implemented (`src/simulator/clifford.rs`):**
```
CliffordState { x: Vec<Vec<bool>>, z: Vec<Vec<bool>>, r: Vec<bool>, n: usize }
  — 2n×n binary symplectic tableau (Aaronson-Gottesman CHP algorithm)
  — rows 0..n = destabilizers, rows n..2n = stabilizers
apply_h / apply_s / apply_x / apply_y / apply_z
apply_cnot / apply_cz / apply_swap
measure(qubit, rng) — random (Gaussian elimination) or deterministic case
execute_clifford(program) — full AQL executor; errors on non-Clifford gates
```

**Tested:** 100-qubit GHZ state circuit confirmed working.

**CLI usage:**
```bash
astracore run --backend clifford surface_code_1000q.aql
astracore analyze --backend clifford grover.aql   # reports "not fully Clifford"
```

---

### 1.3 Sparse Statevector — Structured Circuits ✅ SHIPPED

**What it is:**
Many circuits (product states, circuits with early measurements) have only a small
fraction of non-zero amplitudes. Store only those.

**Representation:** `HashMap<u64, Complex<f64>>` keyed by basis state index.

**Scaling:** Excellent for circuits that stay sparse. Automatic — no user config needed.
Falls back to dense if sparsity drops below threshold (configurable).

**Implementation plan:**
```
src/simulator/sparse.rs
  SparseState { amplitudes: HashMap<u64, Complex<f64>>, n_qubits: usize }
  Automatically used when: initial state + few gates + many measurements
  Threshold: switch to dense when |amplitudes| > 2^(n/2)
```

---

### 1.4 Shot-Based Sampling — Expectation Values at Scale ✅ SHIPPED

**What it is:**
For many algorithms (VQE, QAOA), you don't need the full probability distribution —
you need expectation values ⟨ψ|O|ψ⟩. Run the circuit N times (shots), collect outcomes, average.

**Scaling:** O(shots × circuit_depth) — completely independent of qubit count for measurement outcomes.

**Usage:**
```bash
astracore run --shots 10000 --backend mps circuit.aql
# Output: measurement histogram, expectation values
```

---

### 1.5 Backend Selection Table

| Backend | Max Qubits | Exact? | Best For | CLI Flag | Status |
|---|---|---|---|---|---|
| `statevector` | ~28 (8 GB RAM) | Yes | Small circuits, education | default | ✅ v1 |
| `mps` | 50–200+ | Approx (low entanglement) | QAOA, VQE, shallow circuits | `--backend mps` | ✅ v2 |
| `clifford` | 10,000+ | Yes (Clifford only) | Error correction, analysis | `--backend clifford` | ✅ v2 |
| `sparse` | ~40 (structured) | Yes | Product states, sparse circuits | `--backend sparse` | ✅ v2 |
| `shot` | Unlimited | Statistical | Expectation values, sampling | `--shots N` | ✅ v2 |

---

## Pillar 2 — Language: AQL v2 + Python API

### 2.1 AQL v2 — Reducing the Learning Curve ✅ COMPLETE

The key issues with AQL v1:
- Control flow (LABEL/GOTO/IF) feels like raw assembly — intimidating for newcomers
- No loops — writing `REPEAT 5 { H 0 }` requires manual unrolling
- Error messages show line number but lack visual context
- No way to import/reuse gate libraries across files

**AQL v2 additions (backwards-compatible):**

#### Structured Loops ✅ SHIPPED
```aql
// v1 — must unroll manually
H 0
H 0
H 0

// v2 — structured repeat
REPEAT 3
  H 0
END
```
Compile-time unrolling with full nesting support (REPEAT inside GATE, GATE inside REPEAT).

#### INCLUDE Directive (Gate Libraries) ✅ SHIPPED
```aql
// gates/qft.aql — shared library
GATE qft2 2
  H 0
  PHASE 1 PI_2
  CNOT 0 1
  H 0
END

// main.aql
INCLUDE "gates/qft.aql"
QREG 4
CALL qft2 0 1
MEASURE_ALL
```
Max include depth: 16 (prevents circular includes). Paths resolved relative to including file.

#### Named Qubit Registers ✅ SHIPPED
```aql
// v2
QREG data[4]     // data[0]..data[3]
QREG ancilla[2]  // ancilla[0]..ancilla[1]

H data[0]
CNOT data[0] ancilla[0]
MEASURE ancilla[0]
```

#### Better Error Messages ✅ SHIPPED
```
✗ Parse error at line 5, column 3:
    5 │ CNTO 0 1
      │ ^^^^ unknown instruction 'CNTO'
      │ did you mean: CNOT ?
```

v1 showed: `Parse error line 5: unknown token 'CNTO'`

#### High-Level Control Flow Sugar ✅ SHIPPED
```aql
// v2 sugar — compiles to IF/GOTO/LABEL internally
QREG 2
H 0
MEASURE 0

IFMEASURED 0 == 1 THEN
  X 1
END
```

#### Implementation additions needed:
```
src/compiler/lexer.rs      — THEN, IFMEASURED tokens (REPEAT, INCLUDE done ✅)
src/compiler/parser.rs     — named registers, IFMEASURED (REPEAT, INCLUDE done ✅)
src/compiler/lowering.rs   — NEW: desugar v2 constructs → v1 IR
src/compiler/error.rs      — NEW: rich error reporting with context lines
```

---

### 2.2 Python API — `astracore-py` ✅ SHIPPED

**Goal:** Python users can use AstraCore with zero AQL knowledge.
Under the hood: Python circuit → AQL IR → Rust execution engine.
Speed benefit: the simulation engine remains 100% Rust + SIMD/AVX2.

#### Python-native circuit builder
```python
import astracore as ac

# Circuit builder API
c = ac.Circuit(2)
c.h(0)
c.cnot(0, 1)
c.measure_all()

result = c.run()
print(result.probabilities())         # { "00": 0.5, "11": 0.5 }
print(result.measurements)            # [Measurement(qubit=0, outcome=0, step=4), ...]

# Backend selection
result = c.run(backend="mps", bond_dim=128)
result = c.run(shots=10000)
```

#### AQL execution from Python
```python
import astracore as ac

# Run AQL source directly
result = ac.run_aql("""
QREG 2
H 0
CNOT 0 1
MEASURE_ALL
""")

# Load an .aql file
result = ac.run_file("examples/grover.aql")

# Analyze without running
analysis = ac.analyze_file("examples/ghz.aql")
print(f"Circuit depth: {analysis.circuit_depth}")
print(f"Gate count: {analysis.gate_count}")
```

#### Gate-level API (maps directly to AQL)
```python
c = ac.Circuit(3)
c.h(0)
c.x(1)
c.y(2)
c.cnot(0, 1)
c.cz(1, 2)
c.ccx(0, 1, 2)        # Toffoli
c.rx(0, ac.PI)        # RX(π)
c.ry(1, ac.PI_2)      # RY(π/2)
c.rz(2, 0.785)        # RZ(0.785 radians)
c.barrier()
c.measure(0)
```

#### Custom gates from Python
```python
@ac.gate(n_qubits=2, name="my_bell")
def bell_gate(c, qubits):
    c.h(qubits[0])
    c.cnot(qubits[0], qubits[1])

c = ac.Circuit(4)
c.call("my_bell", [0, 1])
c.call("my_bell", [2, 3])
c.measure_all()
```

#### Plugin registration from Python
```python
import astracore as ac
import numpy as np

# Register a custom gate matrix
sqrt_x = np.array([[0.5+0.5j, 0.5-0.5j],
                   [0.5-0.5j, 0.5+0.5j]])
ac.register_gate("sqrt_x", matrix=sqrt_x)

c = ac.Circuit(1)
c.call("sqrt_x", [0])
c.measure(0)
result = c.run(shots=10000)
```

**Implementation:**
```
astracore-py/           — new crate (Python bindings)
  Cargo.toml            — pyo3 = { version = "0.21", features = ["extension-module"] }
  src/lib.rs            — #[pymodule] fn astracore(m: &Bound<PyModule>)
  src/circuit.rs        — PyCircuit wraps compiler::Program + runtime
  src/result.rs         — PyResult, PyMeasurement, PyAnalysis
  src/gates.rs          — PyGatePlugin wrapper
  pyproject.toml        — maturin build system
```

**Distribution:**
```bash
pip install astracore          # Python package (maturin-built wheel)
cargo install astracore        # CLI binary (unchanged)
```

---

## Pillar 3 — Competitive Positioning

### 3.1 Speed Edge (Biggest Differentiator)

**Current state:** AstraCore is Rust + SIMD/AVX2 statevector.
For circuits ≤ 16 qubits, AstraCore is estimated **5–20× faster** than Qiskit Aer
statevector (Python overhead + NumPy) for the same circuit.

**v2 action: Benchmark & Publish** ✅ Benchmark infrastructure shipped

Benchmark suite created:
```
benches/
  vs_qiskit/
    compare.py                — Qiskit Aer reference timings  ✅
  criterion/
    statevector_scaling.rs    — H/GHZ/layer/measure/AQL 1-24 qubits  ✅
  gate_benchmark.rs           — v1 gates, circuits, pipeline  ✅
```

Shipped:
```
  docs/benchmark_report.md    — filled with cargo bench data (H gate 1–24q, GHZ 2–14q)  ✅
```
Shipped:
```
  mps_scaling.rs              — MPS qubit scaling 20-100q  ✅
  clifford_scale.rs           — Clifford n-qubit scaling  ✅
```

Target headline:
> "AstraCore: Fastest open-source quantum simulator for ≤ 20 qubits.
>  10× faster than Qiskit Aer. Zero Python overhead. Single binary."

Publish at: GitHub Releases + arXiv (short benchmark paper).

---

### 3.2 Zero-Dependency Binary

**Qiskit setup:**
```bash
pip install qiskit qiskit-aer matplotlib jupyter    # 15+ packages, ~500 MB
jupyter notebook                                     # browser needed
```

**AstraCore setup:**
```bash
cargo install astracore       # one command, ~5 MB binary
astracore serve circuit.aql   # browser dashboard immediately
```

**v2 targets for this advantage:**
- Pre-built binaries for Windows/Linux/macOS in GitHub Releases  ✅ (.github/workflows/release.yml)
- Docker image: `docker run -p 8080:8080 astracore serve circuit.aql`  ✅ (Dockerfile)
- Single-file Python wheel via maturin (no separate Rust install needed for Python users)  ✅ (release.yml builds wheels)

---

### 3.3 Assembly-Level Control (Education Niche)

AQL is the simplest quantum assembly language in existence:

| Language | Hello World (Bell pair) |
|---|---|
| Qiskit Python | 8 lines + import + QuantumCircuit object |
| OpenQASM 3 | 7 lines + headers + `include "stdgates.inc"` |
| **AQL v1** | **4 lines** — `QREG 2 / H 0 / CNOT 0 1 / MEASURE_ALL` |
| **AQL v2** | **4 lines** (unchanged — plus optional sugar) |

**v2 education actions:**
- Publish AQL language spec as a standalone PDF (2 pages)  ✅ (docs/aql_spec.md — 2-page Markdown spec)
- Create a "Quantum Computing from Assembly" course outline using AQL  ✅ Shipped (docs/course_outline.md)
- AQL Playground (hosted version of `astracore serve` with examples pre-loaded)  ⬜ Deferred
- LSP server for VS Code: syntax highlighting, hover docs, error underlining  ⬜ Deferred

---

### 3.4 Built-in Interactive Dashboard (Unique)

No other quantum simulator ships a live browser dashboard out of the box.

**v2 dashboard improvements** ⬜ DEFERRED:
- Circuit diagram visualization (ASCII art → SVG circuit drawer)
- Step-by-step execution mode (visualize state evolution gate by gate)
- Multi-run histogram (run circuit N times, show measurement distribution)
- Export: save current circuit as `.aql` from the browser
- Share URL: encode circuit in URL query param for instant sharing

---

### 3.5 Plugin Architecture in a Compiled Language

**v2 plugin improvements:**
- Python-side plugin registration (via `astracore-py`)  ✅ Shipped (`register_gate` + `@gate` decorator)
- Plugin marketplace concept: `astracore install gate-library-qft`  ⬜ Deferred
- Backend plugin for OpenQASM 2.0 export (bridge to IBM hardware)  ✅ Shipped via `astracore export`

---

## Competitive Roadmap

### Phase 1 — Benchmark & Publish ✅ SHIPPED
```
Goal: Establish speed credibility
─────────────────────────────────
✅ Criterion benchmarks: statevector 1–24 qubits (benches/criterion/statevector_scaling.rs)
✅ Head-to-head comparison script vs Qiskit Aer (benches/vs_qiskit/compare.py)
✅ MPS scaling benchmarks 4–100 qubits + MPS-vs-SV crossover (benches/criterion/mps_scaling.rs)
✅ Clifford scaling benchmarks 10–1000 qubits + Clifford-vs-SV crossover (benches/criterion/clifford_scale.rs)
⬜ Write benchmark report with actual numbers (docs/benchmark_report.md — template exists, run cargo bench)
⬜ Publish on GitHub with reproducible instructions
⬜ Target headline: "10× faster than Qiskit for ≤ 20 qubits"
```

### Phase 2 — MPS Backend ✅ SHIPPED
```
Goal: Break the 30-qubit wall
──────────────────────────────
✅ Implement MpsState (custom flat tensors, no ndarray)
✅ SVD truncation with configurable bond dimension (Jacobi one-sided complex SVD)
✅ apply_single_qubit_gate, apply_two_qubit_gate (with SWAP routing for non-adjacent)
✅ Measurement sampling from MPS (project_qubit + renormalize)
✅ AQL --backend mps --bond-dim N CLI flag
✅ Tests: MPS vs statevector agree for low-entanglement circuits (11 tests)
✅ SVD correctness tests (8 tests, incl. CNOT unitary, rank-2 truncation)
✅ Demo: 50-qubit GHZ state .aql example (examples/ghz_50q_mps.aql)
✅ MPS scaling benchmarks (benches/criterion/mps_scaling.rs)
```

### Phase 3 — Clifford Simulator ✅ SHIPPED
```
Goal: Unlimited qubits for Clifford circuits
──────────────────────────────────────────────
✅ Implement CliffordState (binary symplectic tableau, CHP algorithm)
✅ H, S, CNOT, X, Y, Z, CZ, SWAP, Pauli measurement
✅ AQL --backend clifford CLI flag
✅ Tests: 16 tests including 100-qubit GHZ
✅ Auto-detection: analyze if circuit is Clifford-only, suggest backend (is_clifford in CircuitAnalysis)
✅ Demo: 1000-qubit error correction syndrome check .aql example (examples/clifford_1000q.aql)
✅ Clifford scaling benchmarks (benches/criterion/clifford_scale.rs)
```

### Phase 4 — Python API ✅ SHIPPED
```
Goal: Python users without AQL knowledge
──────────────────────────────────────────
✅ astracore-py crate with PyO3 0.22 (astracore-py/Cargo.toml + pyproject.toml)
✅ Circuit builder: h(), x(), y(), z(), s(), t(), rx(), ry(), rz(), phase()
✅                  cnot(), cz(), swap(), toffoli(), ccx(), barrier(), measure(), measure_all()
✅ Circuit.call(name, qubits) — invoke named/plugin gate via CALL
✅ Circuit.html_report(path), Circuit.serve(port), Circuit.dash() — all 3 dashboard backends
✅ run(backend, bond_dim, shots) — dispatches to statevector/mps/clifford/sparse
✅ run_aql(source, backend, bond_dim) free function
✅ run_aql_shots(source, shots) free function
✅ run_qasm(source) free function — runs OpenQASM 2.0 from Python
✅ run_file(path, backend, bond_dim) free function
✅ analyze_aql(source), analyze_file(path) — static analysis without execution
✅ run_aql_report(source, path), run_aql_serve(source, port), run_aql_dash(source)
✅ SimResult: probabilities, measurements, outcome(), bitstring(), prob_of()
✅ ShotSimResult: counts, n_shots, n_qubits, prob(), most_common()
✅ CircuitAnalysis class: circuit_depth, gate_count, is_clifford, gate_histogram, etc.
✅ PI, PI_2 module constants
✅ Build: pip install maturin && maturin develop
✅ 9 Python example scripts (examples/python/) + README
✅ pytest test suite — 40+ tests (tests/test_astracore.py)
⬜ maturin publish to PyPI — future work (release.yml builds wheels, upload step not yet automated)
⬜ Jupyter notebook tutorial — future work
```

### Phase 5 — AQL v2 Language ✅ COMPLETE
```
Goal: Reduce AQL learning curve
─────────────────────────────────
✅ REPEAT n / END structured loop (compile-time unrolling, nested support)
✅ INCLUDE "file.aql" for gate libraries (max depth 16)
✅ Parser qubit limit raised 30 → 1000
✅ QREG name[n] named registers → H data[0], CNOT data[0] ancilla[1], etc.
✅ IFMEASURED q THEN … END sugar (→ IFNOT/GOTO/LABEL at parse time)
✅ IFNOTMEASURED q THEN … END sugar (→ IF/GOTO/LABEL at parse time)
✅ src/compiler/error.rs Diagnostic + "did you mean?" (Levenshtein ≤ 2)
⬜ VS Code extension: LSP server for AQL — Deferred
```

### Phase 6 — Dashboard v2 ✅ PARTIALLY SHIPPED
```
Goal: Best-in-class visualization
───────────────────────────────────
✅ Multi-run histogram — POST /api/shots endpoint + 🎲 Sample button + Chart.js bar chart in SPA
✅ URL-based circuit sharing — base64 URL hash encoding/decoding (🔗 Share button + auto-load on open)
✅ Export .aql from browser editor — 💾 Save button triggers browser file download
✅ Mobile-responsive layout — @media (max-width:780px) stacks editor/results vertically
□ SVG circuit diagram renderer — deferred (requires circuit layout algorithm)
□ Step-by-step execution mode — deferred (requires execution trace infrastructure)
```

### Phase 7 — OpenQASM Bridge ✅ COMPLETE (3.0 deferred)
```
Goal: Hardware connectivity
────────────────────────────
✅ OpenQASM 2.0 exporter (to_qasm(), source_to_qasm()) — src/compiler/qasm_export.rs
✅ astracore export <file.aql> [out.qasm] CLI command
✅ Custom GATE…END → QASM gate declarations
✅ Control flow → inline comments (QASM 2.0 limitation)
✅ 9 export tests
✅ OpenQASM 2.0 importer (from_qasm(), run_qasm()) — src/compiler/qasm_import.rs
✅ astracore import <file.qasm> CLI command (h/x/y/z/s/t/cx/ccx/rx/ry/rz/u3/swap + more)
✅ 12 importer tests
□ OpenQASM 3.0 importer — DEFERRED (future work)
```

---

## Technical Architecture Additions for v2

```
src/
  simulator/
    mod.rs         ✅ BackendSelector + re-exports
    mps.rs         ✅ Matrix Product States (11 tests)
    svd.rs         ✅ Complex Jacobi SVD (8 tests)
    clifford.rs    ✅ Stabilizer/Clifford CHP tableau (16 tests)
    sparse.rs      ✅ Sparse HashMap statevector (12 tests)
  compiler/
    qasm_export.rs ✅ OpenQASM 2.0 exporter (9 tests)
    qasm_import.rs ✅ OpenQASM 2.0 importer (12 tests)
    lexer.rs       ✅ REPEAT, IFMEASURED, IFNOTMEASURED, THEN, RegRef tokens
    parser.rs      ✅ REPEAT unrolling, INCLUDE, 1000-qubit, named registers, IFMEASURED desugar
    mod.rs         ✅ preprocess_includes, parse_source_file, run_file
    error.rs       ✅ Diagnostic + source context + "did you mean?" (Levenshtein ≤ 2)
  runtime/
    shots.rs       ✅ Shot-based sampling (run_shots, ShotResult, print_histogram, 5 tests)

astracore-py/           ✅ Python package (PyO3 0.22)
  src/lib.rs            ✅ Circuit, SimResult, ShotSimResult, CircuitAnalysis,
                           run_aql, run_aql_shots, run_qasm, run_file,
                           analyze_aql, analyze_file, PI, PI_2
                           + html_report/serve/dash/run_aql_report/serve/dash
                           + register_gate(name, matrix) — matrix-based gate registration
                           + @gate(n_qubits, name) decorator — AQL-backed gates (all backends)
                           + _GateDecorator pyclass (MATRIX_GATES + AQL_GATES global registries)
  Cargo.toml            ✅ cdylib + pyo3 0.22 + astracore path dep
  pyproject.toml        ✅ maturin build config

examples/python/        ✅ 10 Python example scripts + README
  bell_state.py         ✅ Bell state basics
  ghz_state.py          ✅ GHZ (statevector + MPS + shots)
  teleportation.py      ✅ Quantum teleportation with feedforward
  grovers.py            ✅ 3-qubit Grover's search
  shot_sampling.py      ✅ Statistical measurement sampling
  backends_comparison.py ✅ All 4 backends side-by-side
  run_qasm_example.py   ✅ OpenQASM 2.0 import from Python
  mps_large_circuit.py  ✅ 50-qubit MPS demo
  dashboard_example.py  ✅ HTML/serve/dash dashboard from Python
  custom_gates.py       ✅ register_gate + @gate decorator examples

tests/
  test_astracore.py     ✅ 40+ pytest tests for full Python API

benches/
  gate_benchmark.rs           ✅ v1 gates, circuits, pipeline
  criterion/
    statevector_scaling.rs    ✅ H/GHZ/layer/measure/AQL 1-24 qubits
    mps_scaling.rs            ✅ MPS 4-100 qubits + MPS-vs-SV crossover
    clifford_scale.rs         ✅ Clifford 10-1000 qubits + Clifford-vs-SV crossover
  vs_qiskit/
    compare.py                ✅ Qiskit Aer reference timing

docs/
  aql_spec.md           ✅ Complete AQL language reference (2-page spec)
  course_outline.md     ✅ "Quantum Computing from Assembly" — 6-module course
  benchmark_report.md   ✅ Criterion data + speedup analysis vs Qiskit Aer
  plan_and_vision.md    ✅ v1 architecture and design rationale
  plan_v2.md            ✅ v2 feature plan (this file)

Dockerfile              ✅ Multi-stage build (rust:1.82-slim + debian:bookworm-slim runtime)
                           EXPOSE 8080, CMD serve /examples/bell.aql 8080
.github/
  workflows/
    ci.yml              ✅ CI — test on ubuntu/macos/windows + Python API tests
    release.yml         ✅ Release — cross-compile 5 targets + maturin wheels on vN.N.N tags
```

---

## Memory & Performance Targets for v2

| Scenario | v1 | v2 Target | v2 Actual |
|---|---|---|---|
| 20-qubit state vector | 16 MB | 16 MB (unchanged) | ✅ unchanged |
| 50-qubit MPS (χ=128) | impossible | ~26 MB | ✅ works |
| 100-qubit MPS (χ=256) | impossible | ~210 MB | ✅ works |
| 1000-qubit Clifford | impossible | ~125 KB (n² bits) | ✅ 100q tested |
| Bell circuit (2q) speed | baseline | +0% (same backend) | ✅ same backend |
| 16q random circuit vs Qiskit | ~10× faster | maintain lead | ⬜ measure |
| Python API overhead vs raw AQL | N/A | < 5% (PyO3 is zero-copy) | ⬜ deferred |

---

## v2 Success Metrics

```
Simulation:
  ✅ MPS backend shipped with --backend mps
  ✅ Clifford backend shipped with --backend clifford
  ✅ 100-qubit GHZ state runnable with --backend clifford
  ✅ 50-qubit MPS working (bond-dim bounded)
  ✅ Statevector benchmarks run; 5–18× speedup estimated vs Qiskit (benchmark_report.md)
  ✅ 50-qubit GHZ example .aql file in examples/ (examples/ghz_50q_mps.aql)

Language:
  ✅ AQL v2 REPEAT loops working with full test coverage
  ✅ INCLUDE directives working with circular-include protection
  ✅ Named registers (QREG data[4] / H data[0]) — 7 tests
  ✅ IFMEASURED / IFNOTMEASURED sugar — 4 tests (desugar to IF/IFNOT + GOTO + LABEL)
  ✅ Rich error messages with source context + "did you mean?" (error.rs) — 8 tests
  □  pip install astracore working on Linux/macOS/Windows (requires maturin build)

Market:
  □  GitHub stars: target 500+ after benchmark publication
  □  Benchmark report cited in at least one academic context
  □  VS Code extension published to marketplace

Tests:
  ✅ 352 Rust tests (all passing)
  ✅ MPS correctness: agree with statevector for χ=full
  ✅ Clifford correctness: agree with statevector for Clifford circuits
  ✅ Python API: pytest suite with 40+ tests (tests/test_astracore.py)
  □  400+ Rust tests (need ~48 more)
  ✅ benchmark_report.md filled with actual Criterion numbers
```

---

## What v2 Does NOT Change

- AQL v1 syntax remains **100% backwards-compatible**
- CLI commands (`run`, `analyze`, `dash`, `serve`, `report`, `opt`) unchanged
- Default backend remains statevector (fastest for ≤ 20 qubits)
- Plugin architecture unchanged (v2 extends, does not replace)
- The single-binary zero-dependency promise — `astracore-py` is optional

---

## Final Statement

AstraCore v2 is not just a feature release.
It is the transition from a **v1.0 proof of concept** to a **serious competitive product**.

After v2 (shipped so far):
- ✅ AstraCore can simulate circuits that require server-grade RAM in competing tools (MPS + Clifford)
- ✅ OpenQASM 2.0 export enables submitting to IBM Quantum via Qiskit
- ✅ AQL v2 loops and includes reduce boilerplate for complex circuits
- ✅ Python researchers can use AstraCore without learning AQL (astracore-py PyO3 bindings)
- ✅ Dashboard v2: shot histogram, URL sharing, save .aql, mobile-responsive layout
- ⬜ Published benchmarks establish AstraCore as the fastest small-circuit simulator

> AstraCore v2 target: own the niche of fastest, simplest, zero-dependency
> quantum simulator — from 2 qubits to 1000 qubits — in a single binary.
