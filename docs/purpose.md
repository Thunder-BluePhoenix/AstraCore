# AstraCore — Purpose & Rationale

## What Is AstraCore?

AstraCore is a high-performance **hybrid Classical–Quantum Runtime Engine** written in Rust.
It lets you design, compile, optimize, and simulate quantum circuits on classical hardware,
with a built-in domain-specific language (AQL — Astra Quantum Language) and a native Rust API.

The "hybrid" in its name is deliberate: AstraCore does not model a pure quantum computer in
isolation. It models the full system — quantum gates, classical control flow (branches, loops,
feedback), and the measurement–result pipeline that connects them. This reflects how real
quantum hardware actually works today.

---

## Why Does AstraCore Exist?

### The Problem Space

Quantum computing is real and growing, but the tooling landscape has significant gaps:

| Gap | What it means in practice |
|-----|--------------------------|
| **Simulator overhead** | Most Python-based simulators (Qiskit Aer, Cirq) carry the CPython interpreter overhead on every gate call. For tight benchmark loops or embedded pipelines this is prohibitive. |
| **No classical control** | Many simulators treat measurement as a final step. Real quantum algorithms (teleportation, error correction, variational methods) require mid-circuit measurement and classical branching. |
| **Opaque optimization** | Circuit optimizers are usually black boxes that the user cannot inspect or extend without diving into framework internals. |
| **No unified profiling** | Understanding circuit depth, qubit utilization, and gate composition requires separate tools or manual counting. |
| **High dependency weight** | Most quantum SDKs pull in large dependency graphs (numpy, scipy, sympy…). Deploying them in constrained or offline environments is painful. |

### What AstraCore Addresses

AstraCore was built to fill exactly these gaps:

1. **Native performance** — Written in Rust with SIMD/AVX2 acceleration on all qubit targets.
   Gate throughput is not bottlenecked by an interpreter or garbage collector.

2. **Hybrid control flow** — The AQL language and the runtime executor support `LABEL`, `GOTO`,
   `IF`, and `IFNOT` instructions. Mid-circuit measurement results can drive branching logic in
   the same program, without leaving the simulation loop.

3. **Transparent optimization** — The peephole optimizer reports what it cancels (H·H → I,
   X·X → I, etc.) and respects user-defined gate boundaries. You can inspect and understand
   every transformation.

4. **Built-in circuit analysis** — The `analyze` command and `CircuitAnalysis` API give circuit
   depth, gate histograms, qubit utilization, and entanglement ratios with zero external tools.

5. **Extensible gate library** — Users can define named gates (`GATE … END`) inline in AQL and
   call them like primitives. The optimizer treats these as opaque barriers, preserving
   user intent.

6. **Minimal dependencies** — AstraCore is a self-contained Rust crate. The only runtime
   dependency is `rand` for measurement sampling. No Python, no numpy, no heavy SDK required.

---

## Who Is It For?

- **Researchers** who need a fast simulation backend they can instrument and extend.
- **Educators** who want a simple, readable language (AQL) to teach quantum circuits without
  framework ceremony.
- **Systems engineers** embedding quantum simulation in a larger Rust application (robotics,
  HPC pipelines, cryptography research).
- **Algorithm developers** prototyping hybrid classical-quantum algorithms where mid-circuit
  measurement and conditional gates are essential.

---

## Pros

| Strength | Detail |
|----------|--------|
| **Rust safety** | Memory-safe, data-race-free by the type system. No segfaults in the simulator core. |
| **SIMD acceleration** | AVX2 paths for all qubit targets. 2–4× gate throughput over scalar on x86-64 with AVX2. |
| **Hybrid execution** | LABEL / GOTO / IF / IFNOT control flow with measurement feedback in a single pass. |
| **AQL compiler pipeline** | Lex → Parse → IR → Optimize → Execute, each stage independently accessible. |
| **Peephole optimizer** | Cancels redundant gate pairs (H·H, X·X, Z·Z, S·S·S·S, T·T·T·T·T·T·T·T, CNOT·CNOT) automatically. |
| **Custom gates** | `GATE name n … END` + `CALL` — user-defined composite gates with qubit remapping. |
| **Circuit analysis** | Depth, histogram, utilization, entanglement ratio — built in, zero extra tools. |
| **Noise model** | Depolarizing, bit-flip, and phase-flip channels available for realistic simulation. |
| **Single-crate deployment** | No Python runtime, no heavy SDK. Compiles to a single binary. |
| **Criterion benchmarks** | Reproducible, statistically sound benchmarks across gate types and circuit sizes. |

---

## Cons

| Limitation | Context |
|------------|---------|
| **State-vector only** | Uses a full 2ⁿ complex-amplitude state vector. Memory grows exponentially with qubit count. Practical limit is roughly 28–30 qubits on a 64 GB machine. |
| **No tensor-network backend** | Large sparse circuits cannot exploit tensor-network contraction to escape the exponential wall. |
| **Single-machine only** | Distributed simulation across nodes is not yet implemented. |
| **x86-64 AVX2 for SIMD** | The SIMD fast path requires an x86-64 processor with AVX2. ARM (Apple Silicon, AWS Graviton) falls back to scalar. |
| **No GPU acceleration** | CUDA/OpenCL compute paths are planned but not yet implemented. |
| **AQL is not OpenQASM** | AQL is AstraCore's own language. Importing OpenQASM 2/3 circuits requires manual translation. |
| **No gate error modeling** | Noise is channel-level (depolarizing etc.), not per-gate calibration data from real hardware. |
| **Early-stage project** | API surfaces may change between versions as the project matures. |

---

## How It Compares

| Feature | AstraCore | Qiskit Aer | Cirq (Python) | QuEST (C) |
|---------|:---------:|:----------:|:-------------:|:---------:|
| Language | Rust | Python/C++ | Python | C |
| SIMD acceleration | AVX2 (all qubits) | Partial | No | Yes |
| Hybrid control flow | Yes (AQL) | Limited | No | No |
| Custom gate definitions | Yes (AQL) | No (inline only) | Partial | No |
| Built-in circuit analysis | Yes | Via transpiler | Via cirq.Circuit | No |
| Noise model | Yes | Yes (extensive) | Yes | Yes |
| State vector limit | ~30 qubits | ~32 qubits | ~20 qubits | ~45 qubits |
| Deployment weight | Minimal | Heavy | Heavy | Light |

> Qiskit Aer and Cirq are more mature ecosystems with larger gate sets, hardware backends,
> and community support. AstraCore is not a replacement for them — it targets the niche of
> **fast, embeddable, inspectable simulation in pure Rust**.
