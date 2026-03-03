# How To Use AstraCore

AstraCore is a high-performance hybrid Classical-Quantum Runtime Engine.
Use it as a **CLI tool**, a **Rust library**, or a **Python package**.

---

## Table of Contents

1. [Setup](#1-setup)
   - [Rust + CLI](#11-rust--cli-setup)
   - [Python package](#12-python-package-setup)
2. [CLI Commands](#2-cli-commands)
3. [AQL Programs](#3-writing-aql-programs)
   - [Backends & shots](#31-backend-selection-and-shot-sampling)
   - [AQL v2 features](#32-aql-v2-language-features)
4. [Dashboard](#4-dashboard--visualization)
   - [CLI dashboard](#41-cli-dashboard)
   - [Python dashboard](#42-python-dashboard)
5. [Python API](#5-python-api)
   - [Circuit builder](#51-circuit-builder)
   - [Free functions](#52-free-functions)
   - [OpenQASM import](#53-openqasm-20-import)
6. [Rust Library API](#6-rust-library-api)
7. [Optimization & Analysis](#7-optimization--analysis)
8. [Benchmarking](#8-benchmarking)
9. [Testing](#9-testing)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Setup

### 1.1 Rust + CLI Setup

**Prerequisites:** Rust toolchain ≥ 1.70 (`rustup install stable`).

```bash
# Clone
git clone <repo-url>
cd AstraCore

# Build in release mode (AVX2 SIMD + LTO)
cargo build --release

# Add to PATH (optional)
export PATH="$PATH:$(pwd)/target/release"

# Or use cargo run for one-off commands
cargo run -- run examples/bell.aql
```

### 1.2 Python Package Setup

**Prerequisites:** Python ≥ 3.8, pip.

```bash
# Step 1 — install maturin (the PyO3 build tool)
pip install maturin

# Step 2 — build and install the Python extension into your environment
cd astracore-py
maturin develop --release    # installs `astracore` module into current Python env
cd ..

# Step 3 — verify
python -c "import astracore; print(astracore.__version__)"
```

> **Tip — virtual environment (recommended):**
> ```bash
> python -m venv .venv
> source .venv/bin/activate      # Linux/macOS
> .venv\Scripts\activate         # Windows
> pip install maturin
> cd astracore-py && maturin develop --release && cd ..
> ```

After setup, both CLI and Python work in parallel — they use the same simulation engine.

---

## 2. CLI Commands

```
astracore <command> [arguments]
```

| Command | Description |
|---------|-------------|
| `run <file.aql>` | Compile and execute an AQL source file |
| `opt <file.aql>` | Peephole-optimize then execute |
| `analyze <file.aql>` | Static circuit analysis (no execution) |
| `export <file.aql> [out.qasm]` | Export to OpenQASM 2.0 |
| `import <file.qasm>` | Import and run an OpenQASM 2.0 file |
| `dash <file.aql>` | Terminal TUI dashboard |
| `report <file.aql> [out.html]` | Standalone HTML dashboard report |
| `serve <file.aql> [port]` | Interactive browser dashboard (default :8080) |
| `demo` | Run all built-in demo circuits |
| `help` | Show help text |

### Backend and shot flags (all commands)

```bash
astracore run examples/ghz.aql                        # statevector (default)
astracore run --backend mps --bond-dim 128 file.aql   # Matrix Product States
astracore run --backend clifford file.aql             # Clifford/stabilizer
astracore run --backend sparse file.aql               # sparse statevector
astracore run --shots 1000 file.aql                   # shot-based sampling
```

### Examples

```bash
# Run circuits
astracore run examples/bell.aql
astracore run examples/ghz_50q_mps.aql --backend mps
astracore run examples/clifford_1000q.aql --backend clifford

# Analyze without running
astracore analyze examples/teleport.aql

# Dashboard
astracore dash examples/bell.aql
astracore report examples/bell.aql bell_report.html
astracore serve examples/bell.aql 8080   # open http://localhost:8080

# QASM round-trip
astracore export examples/bell.aql bell.qasm
astracore import bell.qasm

# Shot-based histogram
astracore run --shots 2000 examples/ghz.aql
```

---

## 3. Writing AQL Programs

Every AQL program starts with `QREG`:

```aql
QREG <n>          // allocate n qubits  (classic integer)
QREG name[n]      // named register     (AQL v2)
```

### Gate instructions

| Instruction | Gate |
|-------------|------|
| `H <q>` | Hadamard |
| `X/Y/Z <q>` | Pauli gates |
| `S/T <q>` | Phase / T gate |
| `RX/RY/RZ <q> <θ>` | Rotation gates (radians) |
| `PHASE <q> <θ>` | Arbitrary phase |
| `CNOT <ctrl> <tgt>` | Controlled-NOT |
| `CZ <ctrl> <tgt>` | Controlled-Z |
| `SWAP <a> <b>` | Swap |
| `TOFFOLI <c0> <c1> <tgt>` | Toffoli (CCNOT) |
| `MEASURE <q>` | Collapse qubit q |
| `MEASURE_ALL` | Collapse all qubits |
| `BARRIER` | Optimizer fence |

**Bell state:**
```aql
QREG 2
H 0
CNOT 0 1
MEASURE_ALL
```

**Custom gate definitions:**
```aql
GATE bell 2
  H 0
  CNOT 0 1
END

QREG 4
CALL bell 0 1   // Bell pair on q0, q1
CALL bell 2 3   // Bell pair on q2, q3
MEASURE_ALL
```

**Control flow:**
```aql
QREG 2
H 0
MEASURE 0
IF 0 GOTO apply_x
GOTO skip
LABEL apply_x
X 1
LABEL skip
MEASURE 1
```

### 3.1 Backend Selection and Shot Sampling

| Backend | Flag | Best for |
|---------|------|----------|
| Statevector | _(default)_ | ≤ 24 qubits, exact |
| MPS | `--backend mps` | 50–200+ qubits, low entanglement |
| Clifford | `--backend clifford` | Unlimited qubits, Clifford-only |
| Sparse | `--backend sparse` | Sparse states (few non-zero amplitudes) |
| Shots | `--shots N` | Statistical sampling, all backends |

`astracore analyze` reports `Clifford-only: yes ✓` when the circuit qualifies for the Clifford backend.

### 3.2 AQL v2 Language Features

**REPEAT — compile-time loop unrolling:**
```aql
QREG 10
REPEAT 10
  H 0
END
```

**INCLUDE — gate libraries:**
```aql
INCLUDE "lib/standard_gates.aql"
QREG 3
CALL qft 0 1 2
```

**Named registers:**
```aql
QREG data[3]       // data qubits: 0, 1, 2
QREG ancilla[2]    // ancilla qubits: 3, 4

H data[0]
CNOT data[0] ancilla[0]
MEASURE ancilla[0]
```

**IFMEASURED / IFNOTMEASURED (feedforward):**
```aql
QREG 2
H 0
MEASURE 0
IFMEASURED 0 THEN
  X 1
END
MEASURE 1
```

---

## 4. Dashboard & Visualization

Three backends — all show the same data: probability distribution, gate histogram, circuit metrics, qubit utilization.

### 4.1 CLI Dashboard

```bash
# Terminal TUI (press q or Esc to quit)
astracore dash examples/bell.aql

# Standalone HTML report (open in browser)
astracore report examples/ghz.aql report.html

# Interactive browser SPA (live AQL editor + charts)
astracore serve examples/teleport.aql 8080
# → open http://localhost:8080
# → edit AQL in the left panel, press ▶ Execute (or Ctrl+Enter) to re-run
```

### 4.2 Python Dashboard

```python
from astracore import Circuit, run_aql_report, run_aql_serve, run_aql_dash

c = Circuit(3)
c.h(0); c.cnot(0, 1); c.cnot(0, 2)
c.measure_all()

# HTML report — write file, open in browser
c.html_report("report.html")
# equivalent free function:
run_aql_report("QREG 3\nH 0\nCNOT 0 1\nCNOT 0 2\nMEASURE_ALL", "report.html")

# Interactive browser dashboard (blocking — Ctrl-C to stop)
c.serve(port=8080)
# equivalent:
run_aql_serve("QREG 2\nH 0\nCNOT 0 1\nMEASURE_ALL", port=8080)

# Terminal TUI (blocking — press q to quit)
c.dash()
# equivalent:
run_aql_dash("QREG 2\nH 0\nCNOT 0 1\nMEASURE_ALL")
```

See [`examples/python/dashboard_example.py`](../examples/python/dashboard_example.py) for full usage.

---

## 5. Python API

### 5.1 Circuit Builder

```python
from astracore import Circuit, PI, PI_2

c = Circuit(n_qubits)

# Single-qubit gates
c.h(q); c.x(q); c.y(q); c.z(q); c.s(q); c.t(q)
c.rx(q, PI); c.ry(q, PI_2); c.rz(q, theta); c.phase(q, theta)

# Two-qubit gates
c.cnot(control, target); c.cz(control, target); c.swap(a, b)

# Three-qubit gate
c.toffoli(c0, c1, target)   # or c.ccx(c0, c1, target) — same gate

# Custom / plugin gate call
c.call("gate_name", [q0, q1])   # CALL instruction in AQL

# Measurement & control
c.measure(q); c.measure_all(); c.barrier()

# Inspect AQL source
print(c.aql_source())
```

**Running the circuit:**
```python
# Single shot (exact statevector)
result = c.run()
result = c.run(backend="mps", bond_dim=128)  # MPS
result = c.run(backend="clifford")           # Clifford
result = c.run(backend="sparse")             # sparse statevector

# SimResult
result.num_qubits                  # int
result.probabilities               # list[float], len = 2^n
result.measurements                # list[(qubit, outcome)]
result.outcome(qubit)              # bool | None
result.bitstring()                 # "010..." | None
result.prob_of("01")               # float

# Shot-based sampling
shots = c.run(shots=1000)          # ShotSimResult
shots.counts                       # dict[str, int]
shots.n_shots                      # int
shots.prob("00")                   # float
shots.most_common()                # list[(str, int)] sorted by count desc

# Dashboard
c.html_report("report.html")       # write HTML file
c.serve(port=8080)                 # blocking browser SPA
c.dash()                           # blocking terminal TUI
```

### 5.2 Free Functions

```python
from astracore import run_aql, run_aql_shots, run_qasm
from astracore import run_file, analyze_aql, analyze_file
from astracore import run_aql_report, run_aql_serve, run_aql_dash

# Simulation
result = run_aql("QREG 2\nH 0\nCNOT 0 1\nMEASURE_ALL")
result = run_aql("...", backend="mps", bond_dim=64)
shots  = run_aql_shots("QREG 2\nH 0\nCNOT 0 1\nMEASURE_ALL", shots=1000)

# Load and run an .aql file from disk
result = run_file("examples/bell.aql")
result = run_file("examples/ghz.aql", backend="mps", bond_dim=128)

# Static analysis (no execution)
analysis = analyze_aql("QREG 2\nH 0\nCNOT 0 1\nMEASURE_ALL")
analysis = analyze_file("examples/bell.aql")
analysis.circuit_depth          # int
analysis.gate_count             # int
analysis.two_qubit_gate_count   # int
analysis.is_clifford            # bool — True if circuit qualifies for Clifford backend
analysis.gate_histogram         # dict[str, int]
analysis.qubit_utilization      # list[int]

# Dashboard
run_aql_report(source, "report.html")   # HTML file
run_aql_serve(source, port=8080)        # blocking browser SPA
run_aql_dash(source)                    # blocking terminal TUI

# OpenQASM 2.0
result = run_qasm("OPENQASM 2.0;\nqreg q[2];\nh q[0];\ncx q[0],q[1];")
```

### 5.3 OpenQASM 2.0 Import

```python
from astracore import run_qasm

qasm = """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
h q[0];
cx q[0], q[1];
cx q[0], q[2];
measure q[0] -> q[0];
measure q[1] -> q[1];
measure q[2] -> q[2];
"""

result = run_qasm(qasm)
print(result.prob_of("000"))  # ≈ 0.5
print(result.prob_of("111"))  # ≈ 0.5
```

Supported gates: `h x y z s t sdg tdg cx cy cz ch swap ccx cswap rx ry rz p u1 u2 u3`
Angle expressions: `pi`, `pi/2`, `3*pi/4`, `-pi/4`, numeric literals.

---

## 6. Rust Library API

```toml
[dependencies]
astracore = { path = "../AstraCore" }
```

```rust
use astracore::compiler::{parse_source, run, run_optimized, analyze_source};
use astracore::runtime::execute;
use astracore::simulator::{execute_mps, execute_clifford, execute_sparse};
use astracore::runtime::{run_shots, ShotResult};
use astracore::compiler::qasm_import::{from_qasm, run_qasm};
use astracore::dashboard::{DashboardData, generate_report, serve, run_tui};

// One-shot run
let result = run("QREG 2\nH 0\nCNOT 0 1\nMEASURE_ALL")?;

// With peephole optimizer
let result = run_optimized("QREG 1\nH 0\nH 0\nMEASURE 0")?;

// Pipeline
let program = parse_source("QREG 2\nH 0\nCNOT 0 1")?;
let result  = execute(&program)?;

// MPS backend (50-200+ qubits)
let result = execute_mps(&program, /*bond_dim=*/128)?;

// Clifford backend (unlimited qubits, Clifford-only)
let result = execute_clifford(&program)?;

// Sparse statevector
let result = execute_sparse(&program)?;

// Shot-based sampling
let shots = run_shots(&program, 1000)?;
println!("P(|00⟩) ≈ {:.3}", shots.prob("00"));
shots.print_histogram();

// OpenQASM import
let result = run_qasm("OPENQASM 2.0;\nqreg q[2];\nh q[0];\ncx q[0],q[1];")?;

// Dashboard
let analysis = analyze_source("QREG 2\nH 0\nCNOT 0 1\nMEASURE_ALL")?;
let data = DashboardData { source_path: "my.aql".into(), analysis, result };
generate_report(&data, "report.html")?;    // HTML file
run_tui(&data)?;                           // terminal TUI (blocking)
serve(data, 8080);                         // browser SPA (blocking)
```

---

## 7. Optimization & Analysis

```bash
# Peephole-optimize then run
astracore opt examples/bell.aql

# Circuit analysis report (no execution)
astracore analyze examples/ghz.aql
```

```rust
use astracore::compiler::{analyze_source, optimize};

let analysis = analyze_source(source)?;
println!("{}", analysis.report());
// Fields: circuit_depth, gate_count, two_qubit_gate_count,
//         gate_histogram, qubit_utilization, is_clifford, ...

let (opt_program, stats) = optimize(source)?;
println!("Removed {} gates", stats.gates_removed);
```

**Cancellation rules:** `H·H`, `X·X`, `Z·Z`, `S⁴`, `T⁸`, `CNOT·CNOT` → identity.
**BARRIER** and **CALL** act as optimizer fences.

---

## 8. Benchmarking

```bash
cargo bench --bench statevector_scaling   # statevector 1-24 qubits
cargo bench --bench mps_scaling           # MPS 4-100 qubits + crossover
cargo bench --bench clifford_scale        # Clifford 10-1000 qubits + crossover
cargo bench --bench gate_benchmark        # gate microbenchmarks
python benches/vs_qiskit/compare.py --runs 100   # Qiskit Aer comparison
```

HTML reports are generated in `target/criterion/`.

---

## 9. Testing

```bash
cargo test                   # all 352 tests + 7 doctests
cargo test --lib compiler    # only compiler module tests
cargo test -- --nocapture    # show println! output
cargo test --release         # faster for large state-vector tests
```

---

## 10. Troubleshooting

| Problem | Fix |
|---------|-----|
| `import astracore` fails | Run `maturin develop --release` inside `astracore-py/` |
| `maturin: command not found` | `pip install maturin` |
| TUI / `c.dash()` does nothing | Requires a real TTY — won't work inside Jupyter/IDE |
| `serve()` hangs | That's normal — press Ctrl-C to stop |
| `--backend clifford` returns error | Circuit uses non-Clifford gates (T, Rx, Ry, Rz, Toffoli, PHASE) |
| File fails to load | Check path and `.aql` extension |
| `astracore help` | Shows full CLI reference |
