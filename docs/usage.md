# AstraCore — Usage Guide

This guide covers every way to use AstraCore: the CLI binary, the AQL language, and the
Rust library API.

---

## Table of Contents

1. [Installation](#installation)
2. [CLI Quick Reference](#cli-quick-reference)
3. [AQL Language Reference](#aql-language-reference)
   - [Register Declaration](#register-declaration)
   - [Single-Qubit Gates](#single-qubit-gates)
   - [Two-Qubit Gates](#two-qubit-gates)
   - [Three-Qubit Gates](#three-qubit-gates)
   - [Rotation Gates](#rotation-gates)
   - [Measurement](#measurement)
   - [Control Flow](#control-flow)
   - [Custom Gate Definitions](#custom-gate-definitions)
   - [Comments and Barriers](#comments-and-barriers)
4. [Example Programs](#example-programs)
   - [Bell State](#bell-state)
   - [GHZ State](#ghz-state)
   - [Quantum Teleportation](#quantum-teleportation)
   - [Custom Gates](#custom-gates)
5. [Rust API](#rust-api)
   - [Simulator (direct)](#simulator-direct)
   - [AQL Compiler Pipeline](#aql-compiler-pipeline)
   - [Circuit Analysis](#circuit-analysis)
   - [Noise Model](#noise-model)
6. [Optimization](#optimization)
7. [Benchmarking](#benchmarking)
8. [Running Tests](#running-tests)

---

## Installation

**Prerequisites:** Rust toolchain (stable 1.70+) and Cargo.

```bash
# Clone the repository
git clone <repo-url>
cd AstraCore

# Build in release mode (enables AVX2 SIMD and full LTO)
cargo build --release

# The binary is placed at:
#   target/release/astracore
```

To run without building a release binary use `cargo run --`:

```bash
cargo run -- run examples/bell.aql
```

---

## CLI Quick Reference

```
astracore <command> [arguments]
```

| Command | Description |
|---------|-------------|
| `run <file.aql>` | Compile and execute an AQL source file |
| `opt <file.aql>` | Run peephole optimizer then execute |
| `analyze <file.aql>` | Static circuit analysis (no execution) |
| `demo` | Run all built-in demonstration circuits |
| `help` | Show help text |

### Examples

```bash
# Run a circuit
astracore run examples/bell.aql

# Run with peephole optimization applied first
astracore opt examples/ghz.aql

# Print circuit metrics without running
astracore analyze examples/teleport.aql

# Run all built-in demos (Bell, GHZ, Grover, teleportation, noise, ...)
astracore demo
```

---

## AQL Language Reference

AQL (Astra Quantum Language) is a line-oriented quantum assembly language.
Each line holds one instruction. Whitespace between tokens is flexible.
Instructions are **case-insensitive** (`h 0` and `H 0` are identical).

### Register Declaration

Every AQL program must start with `QREG`:

```
QREG <n>
```

Allocates `n` qubits, all initialized to `|0⟩`. Qubit indices are `0` through `n-1`.

### Single-Qubit Gates

| Instruction | Gate | Effect |
|-------------|------|--------|
| `H <q>` | Hadamard | `|0⟩ → |+⟩`, `|1⟩ → |−⟩` |
| `X <q>` | Pauli-X (NOT) | `|0⟩ ↔ |1⟩` |
| `Y <q>` | Pauli-Y | `|0⟩ → i|1⟩`, `|1⟩ → -i|0⟩` |
| `Z <q>` | Pauli-Z | `|1⟩ → -|1⟩` |
| `S <q>` | Phase (S) | `|1⟩ → i|1⟩` |
| `T <q>` | T gate | `|1⟩ → e^{iπ/4}|1⟩` |

### Two-Qubit Gates

| Instruction | Gate |
|-------------|------|
| `CNOT <ctrl> <tgt>` | Controlled-NOT |
| `CZ <ctrl> <tgt>` | Controlled-Z |
| `SWAP <q0> <q1>` | Swap two qubits |

### Three-Qubit Gates

| Instruction | Gate |
|-------------|------|
| `TOFFOLI <c0> <c1> <tgt>` | Toffoli (CCNOT) |

### Rotation Gates

```
RX <q> <angle_radians>
RY <q> <angle_radians>
RZ <q> <angle_radians>
```

`angle_radians` is a floating-point literal (e.g. `1.5707963` for π/2).

```aql
QREG 1
RX 0 1.5707963    // rotate qubit 0 by π/2 around X axis
```

### Measurement

| Instruction | Behaviour |
|-------------|-----------|
| `MEASURE <q>` | Collapse qubit `q`; result stored in classical register `q` |
| `MEASURE_ALL` | Collapse all qubits; results stored in registers 0..n-1 |

After `MEASURE`, the classical result of qubit `q` is available to `IF`/`IFNOT`
control-flow instructions.

### Control Flow

AQL supports classical control flow based on measurement results.

```
LABEL <name>          # define a jump target
GOTO <name>           # unconditional jump
IF <q> GOTO <name>    # jump if classical result of qubit q is 1
IFNOT <q> GOTO <name> # jump if classical result of qubit q is 0
```

Label names are identifiers (`apply_x`, `done`, `loop_start`, etc.).
Keywords used as label names (like `end`) are also accepted.

**Example — conditional X correction:**
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

### Custom Gate Definitions

Define reusable composite gates with `GATE … END`:

```aql
GATE <name> <num_local_qubits>
  <body: instructions using local qubits 0..num_local_qubits-1>
END
```

Call a custom gate with:

```aql
CALL <name> <global_q0> <global_q1> ...
```

Local qubits 0, 1, 2 … inside the body are remapped to the supplied global qubit indices.

**Rules:**
- Definitions may appear anywhere in the file; they are extracted before execution.
- Bodies may contain any standard gate instruction and nested `CALL` instructions.
- The optimizer treats `CALL` as an opaque barrier — it will not merge gates across
  a custom-gate boundary.

**Example:**
```aql
GATE bell 2
  H 0
  CNOT 0 1
END

QREG 4
CALL bell 0 1   // creates Bell pair on q0, q1
CALL bell 2 3   // creates Bell pair on q2, q3
MEASURE_ALL
```

### Comments and Barriers

```aql
// This is a comment — everything after // is ignored

BARRIER   // logical separator — stops peephole optimizer from merging across it
```

---

## Example Programs

All examples are in the [examples/](../examples/) directory.

### Bell State

**File:** [examples/bell.aql](../examples/bell.aql)

```aql
QREG 2

H 0           // put q0 into superposition
CNOT 0 1      // entangle q1 with q0

MEASURE_ALL
```

**Expected output:** `[0, 0]` or `[1, 1]` with equal probability 0.5 each.

```bash
astracore run examples/bell.aql
```

---

### GHZ State

**File:** [examples/ghz.aql](../examples/ghz.aql)

Creates a 3-qubit GHZ state `(|000⟩ + |111⟩) / √2`.

```bash
astracore run examples/ghz.aql
```

**Expected output:** all-zeros or all-ones, never mixed.

---

### Quantum Teleportation

**File:** [examples/teleport.aql](../examples/teleport.aql)

Full teleportation protocol with mid-circuit measurement and classical corrections:

```aql
QREG 3

// Prepare message qubit |+⟩
H 0

// Create Bell pair between Alice (q1) and Bob (q2)
H 1
CNOT 1 2

BARRIER
CNOT 0 1
H 0
MEASURE 0
MEASURE 1

BARRIER
// Classical corrections
IF 1 GOTO apply_x
GOTO skip_x
LABEL apply_x
X 2
LABEL skip_x
IF 0 GOTO apply_z
GOTO done
LABEL apply_z
Z 2
LABEL done

BARRIER
MEASURE 2
```

**Expected output:** q2 measures `0` or `1` with equal probability — it has received the
`|+⟩` state from q0.

```bash
astracore run examples/teleport.aql
```

---

### Custom Gates

**File:** [examples/custom_gates.aql](../examples/custom_gates.aql)

```bash
astracore run examples/custom_gates.aql
```

Demonstrates `GATE bell 2`, `GATE ghz 3`, `GATE flip 1`, `GATE hzh 2` and their `CALL`
invocations across a 6-qubit register.

---

## Rust API

Add AstraCore as a library dependency in your `Cargo.toml`:

```toml
[dependencies]
astracore = { path = "../AstraCore" }
```

### Simulator (direct)

```rust
use astracore::core::Simulator;

// Allocate a 3-qubit simulator (all qubits start at |0⟩)
let mut sim = Simulator::new(3);

// Apply gates — methods return &mut Self, so they can be chained
sim.h(0).cnot(0, 1).cnot(0, 2);

// Inspect probabilities without collapsing the state
let probs = sim.probabilities();
// probs[i] = P(basis state i)
println!("P(|000⟩) = {:.4}", probs[0]); // ≈ 0.5
println!("P(|111⟩) = {:.4}", probs[7]); // ≈ 0.5

// Collapse all qubits and get classical bits
let results: Vec<u8> = sim.measure_all();
println!("{:?}", results); // e.g. [1, 1, 1]

// Measure a single qubit (collapses only that qubit)
let bit: u8 = sim.measure(0);
```

**Available gate methods on `Simulator`:**

| Method | Gate |
|--------|------|
| `sim.h(q)` | Hadamard |
| `sim.x(q)` | Pauli-X |
| `sim.y(q)` | Pauli-Y |
| `sim.z(q)` | Pauli-Z |
| `sim.s(q)` | S gate |
| `sim.t(q)` | T gate |
| `sim.rx(q, angle)` | Rx rotation |
| `sim.ry(q, angle)` | Ry rotation |
| `sim.rz(q, angle)` | Rz rotation |
| `sim.cnot(ctrl, tgt)` | CNOT |
| `sim.cz(ctrl, tgt)` | CZ |
| `sim.swap(q0, q1)` | SWAP |
| `sim.toffoli(c0, c1, tgt)` | Toffoli |
| `sim.measure(q)` | Measure single qubit |
| `sim.measure_all()` | Measure all qubits |
| `sim.probabilities()` | Probability vector (no collapse) |

### AQL Compiler Pipeline

```rust
use astracore::compiler;

// One-shot: parse → execute
let result = compiler::run("QREG 2\nH 0\nCNOT 0 1\nMEASURE_ALL")?;
println!("{:?}", result.measurements);

// One-shot: parse → optimize → execute
let result = compiler::run_optimized("QREG 1\nH 0\nH 0\nMEASURE 0")?;
// Peephole optimizer cancels H·H → I, so qubit 0 is always |0⟩

// Access individual pipeline stages:
use astracore::compiler::parse_source;

let program = parse_source("QREG 2\nH 0\nCNOT 0 1")?;
println!("num_qubits = {}", program.num_qubits);
println!("instructions = {:?}", program.instructions);

// Optimize the IR directly
let (opt_program, stats) = compiler::optimize("QREG 1\nH 0\nH 0")?;
println!("Cancelled {} gate pairs", stats.cancelled_pairs);

// Execute a Program value
use astracore::runtime::execute;
let result = execute(&program)?;
```

### Circuit Analysis

```rust
use astracore::compiler::{analyze_source, CircuitAnalysis};

let analysis = analyze_source("QREG 3\nH 0\nCNOT 0 1\nCNOT 0 2\nMEASURE_ALL")?;

println!("Qubits       : {}", analysis.num_qubits);
println!("Gate count   : {}", analysis.gate_count);
println!("Circuit depth: {}", analysis.circuit_depth);
println!("Two-qubit    : {}", analysis.two_qubit_gate_count);
println!("Utilization  : {:?}", analysis.qubit_utilization);
println!("Histogram    : {:?}", analysis.gate_histogram);
println!("Control flow : {}", analysis.has_control_flow);
println!("Custom gates : {}", analysis.has_custom_gates);

// Human-readable report with ASCII bar chart
println!("{}", analysis.report());

// Derived metrics
println!("Avg gates/qubit  : {:.2}", analysis.avg_gates_per_qubit());
println!("Entanglement ratio: {:.2}", analysis.entanglement_ratio());
```

Or from a `Program` value:

```rust
use astracore::compiler::{analyze, parse_source};

let program = parse_source(source)?;
let analysis = analyze(&program);
```

**`CircuitAnalysis` fields:**

| Field | Type | Description |
|-------|------|-------------|
| `num_qubits` | `usize` | Declared qubit count |
| `gate_count` | `usize` | Top-level gate operations |
| `expanded_gate_count` | `usize` | Gate count after CALL expansion |
| `measure_count` | `usize` | Measurement operations |
| `circuit_depth` | `usize` | Critical-path depth (CALL bodies expanded) |
| `two_qubit_gate_count` | `usize` | Multi-qubit gate count |
| `gate_histogram` | `HashMap<String, usize>` | Per-mnemonic counts |
| `qubit_utilization` | `Vec<usize>` | Gate references per qubit |
| `has_control_flow` | `bool` | Contains LABEL/GOTO/IF/IFNOT |
| `has_custom_gates` | `bool` | Contains GATE definitions |
| `custom_gate_defs` | `usize` | Number of distinct GATE definitions |

### Noise Model

```rust
use astracore::core::{Simulator, noise::NoiseChannel};

let mut sim = Simulator::new(2);
sim.h(0).cnot(0, 1);

// Apply depolarizing noise to qubit 0 with error probability 0.01
sim.apply_noise(0, NoiseChannel::Depolarizing(0.01));

// Other channels:
sim.apply_noise(1, NoiseChannel::BitFlip(0.05));
sim.apply_noise(1, NoiseChannel::PhaseFlip(0.02));

let results = sim.measure_all();
```

---

## Optimization

The peephole optimizer cancels known redundant gate sequences before execution.

**Automatically cancelled patterns:**

| Pattern | Result |
|---------|--------|
| `H · H` | Identity (removed) |
| `X · X` | Identity (removed) |
| `Z · Z` | Identity (removed) |
| `S · S · S · S` | Identity (removed) |
| `T · T · T · T · T · T · T · T` | Identity (removed) |
| `CNOT · CNOT` (same qubits) | Identity (removed) |

**Barriers and custom gates act as optimization fences** — no merging occurs across a
`BARRIER` instruction or a `CALL` instruction that touches the same qubit.

**CLI:**
```bash
astracore opt examples/bell.aql
```

**API:**
```rust
let (opt_program, stats) = compiler::optimize(source)?;
println!("Cancelled {} pairs, saved {} gates",
         stats.cancelled_pairs,
         stats.gates_removed);
```

---

## Benchmarking

AstraCore ships with a Criterion benchmark suite covering gate throughput, circuit
preparation, measurement, and the full AQL compiler pipeline.

```bash
# Run all benchmarks
cargo bench

# Run a specific benchmark group
cargo bench --bench gate_benchmark -- ghz_state

# Generate HTML reports (requires gnuplot)
# Reports are saved to target/criterion/
cargo bench
```

**Benchmark groups:**

| Group | What it measures |
|-------|-----------------|
| `gate_benches` | H, X throughput at 4/8/12/16 qubits; CNOT chain; Rx/Rz rotations |
| `circuit_benches` | Bell state, GHZ at various sizes, measure-all |
| `pipeline_benches` | Full AQL pipeline — Bell, GHZ-10, Grover-2q, teleportation |

---

## Running Tests

```bash
# Run all unit tests and doctests
cargo test

# Run only tests in a specific module
cargo test --lib compiler::analysis

# Run tests with output visible (useful for debugging)
cargo test -- --nocapture

# Run in release mode (faster for large state-vector tests)
cargo test --release
```

Current test count: **208 tests, 0 failures** (+ 2 doctests).
