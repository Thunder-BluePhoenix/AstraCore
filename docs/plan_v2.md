# AstraCore v2 — Plan & Vision

> Building on the complete v1.0 foundation (247 tests, all phases delivered),
> v2 targets three strategic pillars: **simulation scale**, **language accessibility**,
> and **market positioning**.

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

### 1.1 Matrix Product States (MPS) — Primary Target

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

**Implementation plan:**
```
src/simulator/mps.rs
  MpsState { tensors: Vec<Array3<Complex>>, bond_dim: usize }
  MpsState::new(n_qubits, max_bond_dim)
  apply_single_qubit_gate(qubit, matrix)   — O(χ²) per gate
  apply_two_qubit_gate(q0, q1, matrix)     — SVD truncation at bond_dim
  sample_measurement(qubit, rng)           — project + renormalize
  probabilities() -> Vec<f64>              — exact for small χ
```

**Cargo dependency:** `ndarray` (already common in Rust ML ecosystem)

**CLI usage:**
```bash
astracore run --backend mps --bond-dim 128 circuit_50q.aql
astracore run --backend mps --bond-dim 256 grover_50q.aql
```

---

### 1.2 Clifford / Stabilizer Simulation — Arbitrary Qubit Count

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

**Implementation plan:**
```
src/simulator/clifford.rs
  CliffordState { tableau: Vec<u64>, n: usize }  — binary symplectic form
  apply_h(qubit)
  apply_s(qubit)
  apply_cnot(ctrl, tgt)
  measure(qubit, rng) -> bool
  to_stabilizer_string() -> String             — human-readable output
```

**CLI usage:**
```bash
astracore run --backend clifford surface_code_1000q.aql
astracore analyze --backend clifford grover.aql   # reports "not fully Clifford"
```

---

### 1.3 Sparse Statevector — Structured Circuits

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

### 1.4 Shot-Based Sampling — Expectation Values at Scale

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

| Backend | Max Qubits | Exact? | Best For | CLI Flag |
|---|---|---|---|---|
| `statevector` | ~28 (8 GB RAM) | Yes | Small circuits, education | default |
| `mps` | 50–200+ | Approx (low entanglement) | QAOA, VQE, shallow circuits | `--backend mps` |
| `clifford` | 10,000+ | Yes (Clifford only) | Error correction, analysis | `--backend clifford` |
| `sparse` | ~40 (structured) | Yes | Product states, sparse circuits | `--backend sparse` |
| `shot` | Unlimited | Statistical | Expectation values, sampling | `--shots N` |

---

## Pillar 2 — Language: AQL v2 + Python API

### 2.1 AQL v2 — Reducing the Learning Curve

The key issues with AQL v1:
- Control flow (LABEL/GOTO/IF) feels like raw assembly — intimidating for newcomers
- No loops — writing `REPEAT 5 { H 0 }` requires manual unrolling
- Error messages show line number but lack visual context
- No way to import/reuse gate libraries across files

**AQL v2 additions (backwards-compatible):**

#### Structured Loops
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

#### Named Qubit Registers
```aql
// v2
QREG data[4]     // data[0]..data[3]
QREG ancilla[2]  // ancilla[0]..ancilla[1]

H data[0]
CNOT data[0] ancilla[0]
MEASURE ancilla[0]
```

#### INCLUDE Directive (Gate Libraries)
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

#### Better Error Messages
```
✗ Parse error at line 5, column 3:
    5 │ CNTO 0 1
      │ ^^^^ unknown instruction 'CNTO'
      │ did you mean: CNOT ?
```

v1 showed: `Parse error line 5: unknown token 'CNTO'`

#### High-Level Control Flow Sugar
```aql
// v2 sugar — compiles to IF/GOTO/LABEL internally
QREG 2
H 0
MEASURE 0

IFMEASURED 0 == 1 THEN
  X 1
END
```

#### Implementation additions:
```
src/compiler/lexer.rs      — add REPEAT, INCLUDE, THEN, IFMEASURED tokens
src/compiler/parser.rs     — parse new constructs → same IR (lowering pass)
src/compiler/lowering.rs   — NEW: desugar v2 constructs → v1 IR
src/compiler/error.rs      — NEW: rich error reporting with context lines
```

---

### 2.2 Python API — `astracore-py`

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

**v2 action: Benchmark & Publish**

Benchmark suite to create:
```
benches/
  vs_qiskit/
    bell_state_1q_to_20q.py      — Qiskit reference timings
    bell_state_1q_to_20q_aql     — AstraCore AQL timings
  criterion/
    statevector_1q_to_24q.rs     — Criterion benchmarks
    mps_20q_to_100q.rs
    clifford_100q_to_1000q.rs
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
- Pre-built binaries for Windows/Linux/macOS in GitHub Releases
- Docker image: `docker run -p 8080:8080 astracore serve circuit.aql`
- Single-file Python wheel via maturin (no separate Rust install needed for Python users)

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
- Publish AQL language spec as a standalone PDF (2 pages)
- Create a "Quantum Computing from Assembly" course outline using AQL
- AQL Playground (hosted version of `astracore serve` with examples pre-loaded)
- LSP server for VS Code: syntax highlighting, hover docs, error underlining

---

### 3.4 Built-in Interactive Dashboard (Unique)

No other quantum simulator ships a live browser dashboard out of the box.

**v2 dashboard improvements:**
- Circuit diagram visualization (ASCII art → SVG circuit drawer)
- Step-by-step execution mode (visualize state evolution gate by gate)
- Multi-run histogram (run circuit N times, show measurement distribution)
- Export: save current circuit as `.aql` from the browser
- Share URL: encode circuit in URL query param for instant sharing

---

### 3.5 Plugin Architecture in a Compiled Language

**v2 plugin improvements:**
- Python-side plugin registration (via `astracore-py`)
- Plugin marketplace concept: `astracore install gate-library-qft`
- Backend plugin for OpenQASM 2.0 export (bridge to IBM hardware)

---

## Competitive Roadmap

### Phase 1 — Benchmark & Publish (Month 1–2)
```
Goal: Establish speed credibility
─────────────────────────────────
□ Criterion benchmarks: statevector 1–24 qubits
□ Head-to-head comparison script vs Qiskit Aer
□ Write benchmark report (markdown + charts)
□ Publish on GitHub with reproducible instructions
□ Target headline: "10× faster than Qiskit for ≤ 20 qubits"
```

### Phase 2 — MPS Backend (Month 2–4)
```
Goal: Break the 30-qubit wall
──────────────────────────────
□ Implement MpsState (ndarray-based tensors)
□ SVD truncation with configurable bond dimension
□ apply_single_qubit_gate, apply_two_qubit_gate
□ Measurement sampling from MPS
□ AQL --backend mps CLI flag
□ Tests: MPS vs statevector agree for low-entanglement circuits
□ Demo: 50-qubit GHZ state, 100-qubit product state rotation
```

### Phase 3 — Clifford Simulator (Month 3–5)
```
Goal: Unlimited qubits for Clifford circuits
──────────────────────────────────────────────
□ Implement CliffordState (binary symplectic tableau)
□ H, S, CNOT, X, Y, Z, CZ, Pauli measurement
□ AQL --backend clifford CLI flag
□ Auto-detection: analyze if circuit is Clifford-only, suggest backend
□ Demo: 1000-qubit error correction syndrome check
```

### Phase 4 — Python API (Month 4–6)
```
Goal: Python users without AQL knowledge
──────────────────────────────────────────
□ astracore-py crate with PyO3
□ Circuit builder: h(), x(), cnot(), measure(), run()
□ run_aql(source: str) function
□ Backend selection: run(backend="mps", bond_dim=128)
□ maturin build + PyPI publish
□ pip install astracore working
□ Python docs + Jupyter notebook examples
```

### Phase 5 — AQL v2 Language (Month 5–7)
```
Goal: Reduce AQL learning curve
─────────────────────────────────
□ REPEAT n / END structured loop
□ QREG name[n] named registers
□ INCLUDE "file.aql" for gate libraries
□ Rich error messages with visual context + suggestions
□ IFMEASURED sugar (desugars to IF/GOTO/LABEL)
□ src/compiler/lowering.rs lowering pass
□ src/compiler/error.rs diagnostic engine
□ VS Code extension: LSP server for AQL
```

### Phase 6 — Dashboard v2 (Month 6–8)
```
Goal: Best-in-class visualization
───────────────────────────────────
□ SVG circuit diagram renderer
□ Step-by-step execution mode (gate-by-gate state evolution)
□ Multi-run histogram (N shots → measurement distribution chart)
□ URL-based circuit sharing (?circuit=base64encoded)
□ Export .aql from browser editor
□ Mobile-responsive layout
```

### Phase 7 — OpenQASM Bridge (Month 8–10)
```
Goal: Hardware connectivity
────────────────────────────
□ OpenQASM 2.0 exporter (AQL → .qasm file)
□ astracore export circuit.aql circuit.qasm
□ Enables: submit to IBM Quantum via Qiskit with AstraCore-designed circuits
□ AstraCore becomes: fast local simulator + hardware-ready transpiler frontend
□ OpenQASM 3.0 importer (run .qasm files in AstraCore)
```

---

## Technical Architecture Additions for v2

```
src/
  simulator/
    mod.rs         — BackendSelector: auto-choose based on circuit analysis
    statevector.rs — v1 statevector (unchanged, still default)
    mps.rs         — NEW: Matrix Product States
    clifford.rs    — NEW: Stabilizer/Clifford tableau
    sparse.rs      — NEW: Sparse HashMap statevector
  compiler/
    lowering.rs    — NEW: AQL v2 → v1 IR desugaring
    error.rs       — NEW: Rich diagnostic engine
    include.rs     — NEW: INCLUDE file resolution
  aql_v2/
    loop.rs        — REPEAT/END desugaring
    register.rs    — Named register resolution
    sugar.rs       — IFMEASURED desugaring

astracore-py/      — NEW crate
  src/lib.rs       — PyO3 module
  src/circuit.rs   — Python Circuit class
  src/result.rs    — Python result types
  pyproject.toml   — maturin build config

benches/
  vs_qiskit/       — NEW: competitive benchmarks
  mps_scaling.rs   — NEW: MPS qubit scaling
  clifford_scale.rs — NEW: Clifford n-qubit scaling
```

---

## Memory & Performance Targets for v2

| Scenario | v1 | v2 Target |
|---|---|---|
| 20-qubit state vector | 16 MB | 16 MB (unchanged) |
| 50-qubit MPS (χ=128) | impossible | ~26 MB |
| 100-qubit MPS (χ=256) | impossible | ~210 MB |
| 1000-qubit Clifford | impossible | ~125 KB (n² bits) |
| Bell circuit (2q) speed | baseline | +0% (same backend) |
| 16q random circuit vs Qiskit | ~10× faster | maintain lead |
| Python API overhead vs raw AQL | N/A | < 5% (PyO3 is zero-copy) |

---

## v2 Success Metrics

```
Simulation:
  □ 50-qubit GHZ state runnable with --backend mps
  □ 1000-qubit Clifford circuit runnable with --backend clifford
  □ Statevector benchmarks published showing ≥5× speedup vs Qiskit

Language:
  □ AQL v2 REPEAT loops working with full test coverage
  □ Named registers working
  □ Rich error messages with visual context
  □ pip install astracore working on Linux/macOS/Windows

Market:
  □ GitHub stars: target 500+ after benchmark publication
  □ Benchmark report cited in at least one academic context
  □ VS Code extension published to marketplace

Tests:
  □ 400+ total tests (v1: 247 + ~150 new)
  □ MPS correctness: agree with statevector for χ=full
  □ Clifford correctness: agree with statevector for Clifford circuits
  □ Python API: pytest suite with 30+ tests
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

After v2:
- AstraCore can simulate circuits that require server-grade RAM in competing tools
- Python researchers can use AstraCore without learning AQL
- AQL v2 is approachable enough for first-year quantum computing students
- Published benchmarks establish AstraCore as the fastest small-circuit simulator
- The interactive dashboard remains unmatched in the open-source ecosystem

> AstraCore v2 target: own the niche of fastest, simplest, zero-dependency
> quantum simulator — from 2 qubits to 1000 qubits — in a single binary.
