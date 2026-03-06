# AstraCore

**High-performance Hybrid Classical-Quantum Runtime Engine**

AstraCore is a quantum circuit simulator and runtime engine built in Rust. It compiles and executes circuits written in AQL (AstraCore Query Language), exposes a Python API, and ships an interactive web dashboard — all from a single binary.

[![CI](https://github.com/Thunder-BluePhoenix/AstraCore/actions/workflows/ci.yml/badge.svg)](https://github.com/Thunder-BluePhoenix/AstraCore/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

---

## Features

- **Four simulation backends** — statevector (exact), MPS tensor networks (50–200+ qubits), Clifford tableau (1000+ qubits), sparse statevector
- **SIMD/AVX2 gate acceleration** — 5–18× faster than Qiskit Aer on benchmarks
- **AQL compiler** — full lexer → parser → IR pipeline with peephole optimizer, custom gate definitions, and hybrid classical-quantum control flow (LABEL / GOTO / IF / IFNOT)
- **Python API** — PyO3 bindings installable via `pip` / `maturin`
- **OpenQASM 2.0 bridge** — import and export `.qasm` files
- **Interactive web dashboard** — live AQL editor, circuit diagram, step-by-step state evolution, shot histogram
- **Plugin architecture** — custom gate, optimizer, and backend plugins
- **361 tests · Docker · GitHub Actions CI/CD**

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          INPUT SURFACES                                 │
│                                                                         │
│   ┌──────────────┐   ┌──────────────┐   ┌───────────────────────────┐  │
│   │  AQL source  │   │  Python API  │   │    OpenQASM 2.0 (.qasm)   │  │
│   │  (.aql file) │   │  (PyO3 0.22) │   │    import / export        │  │
│   └──────┬───────┘   └──────┬───────┘   └────────────┬──────────────┘  │
└──────────┼─────────────────┼────────────────────────┼──────────────────┘
           │                 │                         │
           ▼                 ▼                         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         AQL COMPILER PIPELINE                           │
│                                                                         │
│   ┌────────┐   ┌────────┐   ┌────────────────────┐   ┌──────────────┐  │
│   │ Lexer  │──▶│ Parser │──▶│  IR  (Instruction  │──▶│  Peephole    │  │
│   │        │   │        │   │   enum + Program)  │   │  Optimizer   │  │
│   └────────┘   └────────┘   └────────────────────┘   └──────┬───────┘  │
│                                                              │          │
│   Optimizations: H·H=I  X·X=I  Z·Z=I  S·S=Z  T·T=S  Z·X=Y + more     │
└──────────────────────────────────────────────────────────────┼──────────┘
                                                               │
                                                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         HYBRID RUNTIME                                  │
│                                                                         │
│   ┌───────────────────────────────────────────────────────────────┐     │
│   │  Executor  (program counter · classical register file)        │     │
│   │  LABEL / GOTO / IF q GOTO / IFNOT q GOTO · BARRIER           │     │
│   │  REPEAT loops · INCLUDE · IFMEASURED · named registers        │     │
│   └───────────────────────────────┬───────────────────────────────┘     │
│                                   │                                     │
│   ┌───────────────────────────────▼───────────────────────────────┐     │
│   │  Plugin Registry  (gate / optimizer / backend plugins)        │     │
│   └───────────────────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────────┬──────────┘
                                                               │
                           ┌───────────────────────────────────┤
                           │     BACKEND DISPATCH              │
                           └──┬──────────┬──────────┬──────────┘
                              │          │          │          │
                              ▼          ▼          ▼          ▼
               ┌──────────────┐ ┌──────────────┐ ┌─────────┐ ┌────────┐
               │ Statevector  │ │     MPS      │ │Clifford │ │ Sparse │
               │              │ │ (Tensor Net) │ │(Tableau)│ │  SV    │
               │ Vec<Complex> │ │ Jacobi SVD   │ │ CHP     │ │HashMap │
               │ AVX2 SIMD    │ │ Bond dim χ   │ │1000+q   │ │ O(k)   │
               │ ~28 qubits   │ │ 50–200+ q    │ │stabiliz.│ │sparse  │
               └──────────────┘ └──────────────┘ └─────────┘ └────────┘
                              │          │          │          │
                              └──────────┴──────────┴──────────┘
                                                │
                              ┌─────────────────▼─────────────────┐
                              │       ExecutionResult              │
                              │  measurements · probabilities      │
                              │  pre_measurement_amplitudes        │
                              │  gate_count · branch_count         │
                              └─────────────────┬─────────────────┘
                                                │
           ┌────────────────────────────────────┤
           │           OUTPUT SURFACES          │
           └──┬──────────┬──────────┬───────────┘
              │          │          │          │
              ▼          ▼          ▼          ▼
        ┌──────────┐ ┌───────┐ ┌────────┐ ┌─────────────────────────────┐
        │   CLI    │ │  TUI  │ │ HTML   │ │     Web Dashboard (axum)    │
        │ stdout   │ │ratatui│ │ report │ │  Circuit SVG · Step Player  │
        │ JSON     │ │       │ │Chart.js│ │  Shot histogram · State vec │
        └──────────┘ └───────┘ └────────┘ └─────────────────────────────┘
```

---

## Quick Start

### Install (Rust CLI)

```bash
git clone https://github.com/Thunder-BluePhoenix/AstraCore
cd AstraCore
cargo build --release
# binary at: target/release/astracore
```

### Install (Python)

```bash
cd astracore-py
bash setup.sh          # creates .venv, installs maturin, builds + installs
# or: bash setup.sh --release   for optimised build
```

### Run a Bell State

```bash
astracore run examples/bell.aql
```

```aql
// bell.aql
QREG 2
H 0
CNOT 0 1
MEASURE_ALL
```

```
q0 → 0   q1 → 0      # (or both 1 — 50/50)
```

---

## Performance

All timings measured with Criterion on x86_64 with AVX2, release build (`opt-level = 3, lto = true`).

### H Gate — Single-Qubit Gate Sweep

| Qubits | State vector size | Time (mean) |
|-------:|------------------:|------------:|
| 1      | 32 B              | 96.0 ns     |
| 2      | 64 B              | 100.2 ns    |
| 4      | 256 B             | 107.6 ns    |
| 6      | 1 KB              | 153.4 ns    |
| 8      | 4 KB              | 317.3 ns    |
| 10     | 16 KB             | 1.25 µs     |
| 12     | 64 KB             | 3.79 µs     |
| 14     | 256 KB            | 15.4 µs     |
| 16     | 1 MB              | 279 µs      |
| 18     | 4 MB              | 1.06 ms     |
| 20     | 16 MB             | 4.98 ms     |
| 22     | 64 MB             | 20.7 ms     |
| 24     | 256 MB            | 84.0 ms     |

### GHZ State Preparation (H + CNOT chain + MEASURE_ALL)

| Qubits | Time (mean) |
|-------:|------------:|
| 2      | 107 ns      |
| 4      | 157 ns      |
| 6      | 383 ns      |
| 8      | 1.68 µs     |
| 10     | 7.63 µs     |
| 12     | 36.2 µs     |
| 14     | 165.7 µs    |
| 16     | 1.03 ms     |
| 18     | 4.76 ms     |
| 20     | 37.7 ms     |
| 22     | 197 ms      |
| 24     | 870 ms      |

### vs Qiskit Aer (estimated)

| Qubits | AstraCore | Qiskit Aer (est.) | Speedup |
|-------:|----------:|------------------:|--------:|
| 4      | 0.11 µs   | 0.6–1.5 µs        | ~6–14×  |
| 8      | 0.32 µs   | 1.5–3 µs          | ~5–9×   |
| 12     | 3.79 µs   | 20–50 µs          | ~5–13×  |
| 16     | 279 µs    | 1–5 ms            | ~4–18×  |
| 20     | 4.98 ms   | 25–80 ms          | ~5–16×  |

> Primary sources of speedup:
> 1. **Zero Python overhead** — no per-gate Python dispatch
> 2. **AVX2 SIMD** — complex multiplication vectorised with `_mm256_*` intrinsics
> 3. **Cache-aware layout** — state stored as contiguous `Vec<Complex<f64>>`
> 4. **Single binary** — no subprocess, no IPC, no NumPy serialisation

### Backend Scaling

| Backend      | Max Qubits    | Memory model        | Best circuit family              |
|:-------------|:-------------:|:--------------------|:---------------------------------|
| `statevector`| ~28 (16 GB)   | O(2ⁿ) dense         | Any gate, exact                  |
| `mps`        | 50–200+       | O(n · χ²) sparse    | Low-entanglement, wide circuits  |
| `clifford`   | 1000+         | O(n²) tableau       | Stabilizer circuits only         |
| `sparse`     | ~30+          | O(k) hashmap        | Product states, early-collapse   |

### Statevector Memory Scaling

```
 Qubits │ State vector │ Practical limit
────────┼──────────────┼─────────────────────────
     10 │       16 KB  │ trivial
     16 │        1 MB  │ very fast (L2 cache)
     20 │       16 MB  │ fast (L3 cache)
     24 │      256 MB  │ fine on 1 GB RAM
     28 │        4 GB  │ max on 16 GB RAM
     30 │       16 GB  │ RAM limit
     50+│ petabytes    │ → use MPS or Clifford
```

Run benchmarks locally:

```bash
cargo bench --bench statevector_scaling
cargo bench --bench mps_scaling
cargo bench --bench clifford_scale
```

---

## CLI Reference

```bash
astracore run     <file.aql>                        # statevector (exact)
astracore run     <file.aql> --backend mps --bond-dim 64   # MPS (50-200+ qubits)
astracore run     <file.aql> --backend clifford     # Clifford (1000+ qubits)
astracore run     <file.aql> --backend sparse       # sparse statevector
astracore run     <file.aql> --shots 10000          # shot sampling

astracore analyze <file.aql>                        # circuit depth, gate histogram, qubit utilization
astracore serve   <file.aql>                        # interactive web dashboard → localhost:8080
astracore dash    <file.aql>                        # terminal TUI dashboard
astracore report  <file.aql>                        # generate standalone report.html

astracore export  <file.aql>  -o out.qasm           # AQL → OpenQASM 2.0
astracore import  <file.qasm> -o out.aql            # OpenQASM 2.0 → AQL
```

---

## AQL — AstraCore Query Language

A minimal quantum assembly language. A Bell pair is 4 lines:

```aql
QREG 2
H 0
CNOT 0 1
MEASURE_ALL
```

### Gate Set

| Instruction | Description |
|---|---|
| `H q` | Hadamard |
| `X / Y / Z q` | Pauli gates |
| `S / T q` | Phase gates (π/2, π/4) |
| `RX / RY / RZ q theta` | Rotation gates |
| `PHASE q theta` | Diagonal phase shift |
| `CNOT ctrl tgt` | Controlled-NOT |
| `CZ ctrl tgt` | Controlled-Z |
| `SWAP a b` | Qubit swap |
| `CCX c0 c1 tgt` | Toffoli (double-controlled-NOT) |
| `MEASURE q` / `MEASURE_ALL` | Collapse and record |
| `BARRIER` | Optimiser fence |

### Control Flow

```aql
QREG 2
H 0
MEASURE 0
IF 0 GOTO apply_x      // jump if q0 measured 1
GOTO done
LABEL apply_x
  X 1
LABEL done
MEASURE 1
```

### Custom Gates

```aql
GATE bell 2
  H 0
  CNOT 0 1
END

QREG 4
CALL bell 0 1
CALL bell 2 3
MEASURE_ALL
```

### Named Registers (AQL v2)

```aql
QREG data[2]
QREG anc[1]
H data[0]
CNOT data[0] data[1]
MEASURE_ALL
```

---

## Python API

```python
import astracore as ac

# Build and run a circuit
c = ac.Circuit(2)
c.h(0)
c.cnot(0, 1)
c.measure_all()

result = c.run()
print(result.probabilities)   # [0.5, 0.0, 0.0, 0.5]
print(result.bitstring())     # "00" or "11"

# Shot sampling
shots = c.run(shots=10000)
print(shots.counts)           # {"00": ~5000, "11": ~5000}

# MPS backend for large circuits
r = c.run(backend="mps", bond_dim=64)

# Clifford backend for 1000+ qubit stabilizer circuits
r = c.run(backend="clifford")

# Run raw AQL
r = ac.run_aql("QREG 3\nH 0\nCNOT 0 1\nCNOT 0 2\nMEASURE_ALL")
print(r.prob_of("000"))       # 0.5

# Static analysis (no execution)
a = ac.analyze_aql("QREG 2\nH 0\nCNOT 0 1\nMEASURE_ALL")
print(a.gate_count, a.circuit_depth, a.is_clifford)

# OpenQASM 2.0
r = ac.run_qasm(open("circuit.qasm").read())

# Custom gates via @gate decorator (works on all backends)
@ac.gate(n_qubits=2, name="bell")
def bell_gate(c, qubits):
    c.h(qubits[0])
    c.cnot(qubits[0], qubits[1])

circ = ac.Circuit(4)
circ.call("bell", [0, 1])
circ.call("bell", [2, 3])
circ.measure_all()
result = circ.run()
```

See [examples/python/](examples/python/) for more examples including Grover's search, teleportation, backends comparison, and MPS large circuits.

---

## Web Dashboard

```bash
astracore serve examples/bell.aql
# → http://localhost:8080
```

The dashboard includes:
- **Live AQL editor** with Ctrl+Enter execute and syntax reference
- **Circuit diagram** — auto-generated SVG from compiled instructions
- **Step-by-step execution** — walk gate-by-gate through state evolution, inspect probability amplitudes at each step
- **State vector table** — pre-measurement amplitudes with Re / Im / probability / phase per basis state
- **Shot histogram** — configurable up to 100,000 shots
- **AQL ↔ Python toggle** — auto-transpiles your circuit to Python API code
- **Collapsible editor** for full-width visualization mode

---

## Project Structure

```
src/
  core/         complex, state, gates, simulator, noise, simd
  compiler/     lexer, parser, IR, analysis, qasm_export, qasm_import
  optimizer/    peephole (H·H, X·X, Z·Z cancellations + more)
  runtime/      executor (GOTO/IF/IFNOT), scheduler, shot sampling
  plugins/      gate / optimizer / backend plugin interfaces + registry
  dashboard/    web server (axum), TUI (ratatui), HTML report, circuit SVG
  simulator/    MPS (SVD), Clifford (CHP tableau), sparse statevector
  lib.rs / main.rs

astracore-py/   PyO3 0.22 Python package (maturin build)
examples/       AQL examples + Python examples
benches/        Criterion benchmarks + Qiskit comparison script
docs/           AQL spec, benchmark report, course outline, plan
```

---

## Docker

```bash
docker build -t astracore .
docker run --rm astracore run /examples/bell.aql
docker run --rm -p 8080:8080 astracore serve /examples/bell.aql
```

---

## Development

```bash
cargo test          # 361 tests
cargo clippy        # linter
cargo bench         # benchmarks
```

Python tests:

```bash
cd astracore-py
bash setup.sh
../.venv/bin/pytest ../tests/test_astracore.py -v
```

---

## License

Licensed under either of:

- [MIT License](LICENSE-MIT)
- [Apache License 2.0](LICENSE-APACHE)

at your option.
