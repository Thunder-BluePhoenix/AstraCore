# AstraCore v3 — Plan & Vision

> Building on the complete v2 foundation (361 tests, all phases shipped),
> v3 targets three strategic pillars: **hardware acceleration**, **distributed scale**,
> and **developer experience**.
>
> Every new backend is **optional** — selected via `--backend` or `--device` CLI flags.
> A machine without a GPU or without a cluster simply uses the existing CPU backends unchanged.

---

## v3 Delivery Status

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | GPU Backend (CUDA/WebGPU statevector) | ✅ **SHIPPED** |
| Phase 2 | Distributed Simulation (multi-node TCP) | ✅ **SHIPPED** |
| Phase 3 | Visual Circuit Debugger (VS Code extension) | ✅ **SHIPPED** |

---

## Pillar 1 — GPU Backend

### Supported Hardware

Any GPU works — the two sub-backends cover the full hardware landscape:

| Sub-backend | Flag | Works on | Minimum requirement |
|-------------|------|----------|---------------------|
| CUDA | `--backend cuda` | **NVIDIA only** (GTX, RTX, Tesla, Quadro) | CUDA Toolkit 11+ installed; GTX 9xx (CC 5.x) and newer |
| WebGPU | `--backend wgpu` | NVIDIA + AMD + Intel + Apple Silicon | GPU driver with Vulkan (Windows/Linux) or Metal (macOS) support |

**NVIDIA GeForce GTX compatibility:**

| GTX Series | CUDA Compute Cap. | CUDA backend | WebGPU backend |
|------------|:-----------------:|:------------:|:--------------:|
| GTX 9xx (Maxwell) | 5.x | ✅ (needs CUDA 11) | ✅ |
| GTX 10xx (Pascal) | 6.1 | ✅ | ✅ |
| GTX 16xx (Turing) | 7.5 | ✅ | ✅ |
| RTX 20xx/30xx/40xx | 7.5–8.9 | ✅ | ✅ |

> **Recommended dev path for GTX machines:**
> Start with `--features wgpu` — no CUDA Toolkit install needed, just standard GPU drivers.
> Add `--features cuda` later for maximum performance (CUDA has ~20% lower kernel launch overhead vs WebGPU on NVIDIA).

**CPU-only machines (no GPU):**
```bash
cargo build --release          # default — no GPU features compiled in
astracore run circuit.aql      # uses CPU statevector unchanged, no warnings
```

---

### Why

The statevector gate application is a dense BLAS-like operation over `2^n` complex numbers.
A single H gate on a 24-qubit circuit iterates over 16 million amplitude pairs.
Modern GPUs have 10,000+ cores and 80–100 GB/s memory bandwidth — perfectly suited
for this workload. A 28-qubit circuit that takes **84 ms** on CPU can drop to **< 2 ms** on GPU.

### Design — Optional Compile Feature

GPU support is gated behind a Cargo feature flag so users without GPU hardware
still get a zero-overhead, zero-dependency build.

```toml
# Cargo.toml
[features]
default = []
cuda    = ["dep:cudarc"]       # NVIDIA CUDA via cudarc crate
wgpu    = ["dep:wgpu"]         # cross-platform (AMD / Intel / Apple Silicon) via wgpu
gpu     = ["cuda"]             # default gpu alias → CUDA
```

Build without GPU (default — any machine):
```bash
cargo build --release
astracore run circuit.aql                     # uses CPU backends unchanged
```

Build with CUDA (NVIDIA GPU required):
```bash
cargo build --release --features cuda
astracore run circuit.aql --backend gpu       # uses CUDA statevector
```

Build with WebGPU (cross-platform — AMD / Intel / Apple Silicon):
```bash
cargo build --release --features wgpu
astracore run circuit.aql --backend wgpu      # uses WebGPU compute shaders
```

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  GPU BACKEND DISPATCH                   │
│                                                         │
│  astracore run --backend gpu                            │
│         │                                               │
│         ▼                                               │
│  ┌──────────────────────────────────────────────┐       │
│  │  GpuBackend trait (src/simulator/gpu/mod.rs) │       │
│  │  upload_state(Vec<Complex>) → GpuBuffer      │       │
│  │  apply_gate(GpuBuffer, gate_matrix, qubit)   │       │
│  │  download_state(GpuBuffer) → Vec<Complex>    │       │
│  │  measure_all(GpuBuffer) → Vec<f64>           │       │
│  └──────────────────────────────────────────────┘       │
│         │                    │                          │
│         ▼                    ▼                          │
│  ┌─────────────┐   ┌──────────────────────┐            │
│  │  CudaBackend│   │  WgpuBackend          │            │
│  │  (cudarc)   │   │  (wgpu compute shader)│            │
│  │  NVIDIA GPU │   │  AMD/Intel/Apple      │            │
│  └─────────────┘   └──────────────────────┘            │
└─────────────────────────────────────────────────────────┘
```

### Key Implementation Details

**State transfer strategy:**
- Upload state vector to GPU once at circuit start
- Keep on GPU for entire gate sequence (avoid round-trips)
- Download only for measurement / final probability extraction
- For circuits with mid-circuit measurements: partial download + re-upload

**CUDA kernel (H gate on qubit k, n-qubit register):**
```
Thread i handles amplitude pair (i & ~(1<<k), i | (1<<k))
Grid: ceil(2^(n-1) / blockDim) blocks × 128 threads
Each thread: reads a0, a1 → writes new_a0 = (a0+a1)/√2, new_a1 = (a0-a1)/√2
```

**WebGPU shader (WGSL compute):**
```wgsl
@compute @workgroup_size(128)
fn apply_gate(@builtin(global_invocation_id) id: vec3u) {
    let pair_idx = id.x;
    // reconstruct (i0, i1) from pair_idx and qubit_mask
    // apply 2x2 matrix
}
```

**Memory layout:**
- `GpuStateVector`: interleaved `[re0, im0, re1, im1, …]` as `f32` or `f64` buffer
- For circuits ≤ 28 qubits: single contiguous buffer (fits in VRAM for any modern GPU)
- For 28–34 qubits: chunked streaming from system RAM → VRAM if VRAM < required

**Automatic fallback:**
```
--backend gpu
    GPU available? ──yes──▶ GPU execution
         │
         no
         ▼
    warn: "GPU not available, falling back to --backend statevector"
    CPU statevector execution
```

### CLI

```bash
# Run on GPU (auto-detects CUDA or WebGPU)
astracore run circuit.aql --backend gpu

# Explicitly choose sub-backend
astracore run circuit.aql --backend cuda
astracore run circuit.aql --backend wgpu

# List available devices
astracore devices
# → cuda:0  NVIDIA RTX 4090  (24 GB VRAM)
# → wgpu:0  AMD RX 7900      (16 GB VRAM)
# → cpu     (always available)

# Target a specific device
astracore run circuit.aql --backend gpu --device cuda:0
```

### New Source Files

```
src/simulator/gpu/
  mod.rs          — GpuBackend trait + backend selection logic
  cuda.rs         — CudaBackend (feature = "cuda")
  wgpu.rs         — WgpuBackend (feature = "wgpu")
  kernels/
    gate.cu       — CUDA gate kernel (H, X, Y, Z, Rx, CNOT, …)
    gate.wgsl     — WGSL compute shader equivalent
```

### Expected Performance

| Qubits | CPU (AVX2) | GPU (RTX 3080 est.) | Speedup |
|-------:|:----------:|:-------------------:|--------:|
| 20     | 4.98 ms    | ~0.1 ms             | ~50×    |
| 24     | 84 ms      | ~1.5 ms             | ~56×    |
| 28     | ~1.4 s     | ~25 ms              | ~56×    |
| 30     | OOM on CPU | ~100 ms             | ∞       |

### Tests to Add (~12)

```
gpu::tests::test_gpu_available_or_skip
gpu::tests::test_h_gate_gpu_matches_cpu_4q
gpu::tests::test_h_gate_gpu_matches_cpu_20q
gpu::tests::test_bell_state_gpu
gpu::tests::test_ghz_8q_gpu_matches_cpu
gpu::tests::test_gpu_fallback_to_cpu_when_unavailable
gpu::tests::test_cuda_backend_compile_feature
gpu::tests::test_wgpu_backend_compile_feature
gpu::tests::test_gpu_measure_all_probabilities
gpu::tests::test_gpu_cnot_entanglement
gpu::tests::test_gpu_30q_circuit_runs
gpu::tests::test_aql_pipeline_gpu_backend
```

---

## Pillar 2 — Distributed Simulation

### Why

For circuits approaching 30–40 qubits, the state vector exceeds available RAM on any single machine
(2³⁰ × 16 bytes = 16 GB). Distributed simulation splits the state vector across multiple nodes,
allowing each node to hold a slice. A 10-node cluster with 16 GB RAM each can simulate up to
34 qubits exactly.

### Design — Optional Feature Flag + CLI

```toml
[features]
dist = ["dep:tokio", "dep:bincode"]   # tokio already in deps; bincode for serialization
```

```bash
# Single machine — unchanged (no --dist flag)
astracore run circuit.aql

# Distributed — coordinator node (starts workers automatically via SSH or manually)
astracore run circuit.aql --dist --nodes "192.168.1.10:7700,192.168.1.11:7700"

# Worker node (started on each remote machine)
astracore worker --port 7700

# Auto-discover workers via config file
astracore run circuit.aql --dist --cluster cluster.toml
```

### Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    DISTRIBUTED EXECUTION MODEL                   │
│                                                                  │
│  Coordinator (your machine)                                      │
│  ┌───────────────────────────────────────────────┐               │
│  │  parse AQL → Program                          │               │
│  │  partition state vector: 2^n amplitudes       │               │
│  │    node 0: amplitudes [0,     2^n/k)          │               │
│  │    node 1: amplitudes [2^n/k, 2·2^n/k)        │               │
│  │    …                                          │               │
│  │  for each gate:                               │               │
│  │    local gate? → send to owning node only     │               │
│  │    cross-shard gate? → all-to-all exchange    │               │
│  └───────────────────────────────────────────────┘               │
│         │ TCP (tokio async) / bincode serialisation               │
│         ▼                                                        │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐         │
│  │  Worker 0    │   │  Worker 1    │   │  Worker k    │         │
│  │  shard[0]    │   │  shard[1]    │   │  shard[k]    │         │
│  │  (AVX2/GPU)  │   │  (AVX2/GPU)  │   │  (AVX2/GPU)  │         │
│  └──────────────┘   └──────────────┘   └──────────────┘         │
│                                                                  │
│  Cross-shard CNOT (qubit k spans shard boundary):               │
│    phase 1: each worker sends half its amplitudes to partner     │
│    phase 2: each worker applies local CNOT to its full pair      │
│    phase 3: workers exchange back                                │
└──────────────────────────────────────────────────────────────────┘
```

### Key Implementation Details

**State partitioning:**
```
n total qubits, k nodes (k must be power of 2)
partition_qubit = log2(k)   // top log2(k) qubits determine which node owns an amplitude
node i owns amplitudes where (amplitude_index >> (n - partition_qubit)) == i
```

**Gate classification:**
- **Local gate** on qubit q where q < n - partition_qubit → single worker, no communication
- **Cross-shard gate** on qubit q where q ≥ n - partition_qubit → requires all-to-all exchange

**Communication protocol (TCP / bincode):**
```
Coordinator → Worker:   GateMsg { instruction: Instruction, shard_info: ShardInfo }
Worker → Coordinator:   Ack | Probabilities(Vec<f64>) | Error(String)
Worker → Worker:        ShardExchange { amplitudes: Vec<Complex>, from_shard: usize }
```

**cluster.toml format:**
```toml
[cluster]
nodes = [
  { host = "192.168.1.10", port = 7700, role = "worker" },
  { host = "192.168.1.11", port = 7700, role = "worker" },
  { host = "192.168.1.12", port = 7700, role = "worker" },
]
ssh_key = "~/.ssh/id_rsa"     # optional: auto-start workers via SSH
```

### New Source Files

```
src/simulator/dist/
  mod.rs          — DistBackend + execute_distributed()
  coordinator.rs  — circuit partitioning, gate dispatch, result aggregation
  worker.rs       — shard storage, local gate application, exchange protocol
  protocol.rs     — message types (bincode-serialisable)
  partition.rs    — state vector partitioning math
src/bin/worker.rs — `astracore worker` binary entry point
```

### CLI

```bash
# Start workers on remote machines (manual)
ssh user@node1 "astracore worker --port 7700"
ssh user@node2 "astracore worker --port 7700"

# Run distributed circuit
astracore run ghz_40q.aql --dist --nodes "node1:7700,node2:7700"

# Auto-start workers via SSH (requires ssh_key in cluster.toml)
astracore run ghz_40q.aql --dist --cluster cluster.toml

# Simulate up to how many qubits per node
astracore run circuit.aql --dist --cluster cluster.toml --max-qubits-per-node 28
```

### Scalability

| Nodes | RAM per node | Max exact qubits |
|------:|:-----------:|:----------------:|
| 1     | 16 GB       | 30               |
| 2     | 16 GB       | 31               |
| 4     | 16 GB       | 32               |
| 8     | 16 GB       | 33               |
| 16    | 16 GB       | 34               |
| 64    | 16 GB       | 36               |
| 1024  | 16 GB       | 40               |

### Tests to Add (~10)

```
dist::tests::test_partition_math_power_of_2
dist::tests::test_local_gate_no_exchange
dist::tests::test_cross_shard_gate_exchanges
dist::tests::test_bell_state_2_nodes_loopback
dist::tests::test_ghz_32q_2_nodes
dist::tests::test_coordinator_worker_handshake
dist::tests::test_cluster_toml_parse
dist::tests::test_fallback_single_node
dist::tests::test_measurement_aggregation_across_shards
dist::tests::test_dist_matches_local_16q
```

---

## Pillar 3 — Visual Circuit Debugger (VS Code Extension)

### Why

Currently, the only way to debug a circuit is to run it and inspect the terminal output or
the web dashboard. A VS Code extension brings the debugger directly into the editor:
step through gates, watch amplitude panels update inline, set breakpoints on instructions,
and see the circuit diagram update live as you type.

### Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                    VS CODE EXTENSION                                 │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐     │
│  │  Language Server (src/lsp/server.rs)                        │     │
│  │  • Syntax highlighting  (TextMate grammar — aql.tmLanguage) │     │
│  │  • Hover documentation  (gate description on hover)         │     │
│  │  • Diagnostics          (lex/parse errors → red squiggles)  │     │
│  │  • Code completion      (gate mnemonics, register names)    │     │
│  │  • Go-to-definition     (LABEL / GATE definitions)          │     │
│  └────────────────────────────┬────────────────────────────────┘     │
│                               │  JSON-RPC / LSP over stdio           │
│  ┌────────────────────────────▼────────────────────────────────┐     │
│  │  Debug Adapter (src/lsp/debug_adapter.rs)                   │     │
│  │  Implements DAP (Debug Adapter Protocol)                    │     │
│  │                                                             │     │
│  │  Commands:                                                  │     │
│  │    launch    — compile + start step executor                │     │
│  │    next      — execute one instruction, return snapshot     │     │
│  │    continue  — run to next breakpoint                       │     │
│  │    pause     — halt execution                               │     │
│  │    evaluate  — query prob_of("00") at current step          │     │
│  │                                                             │     │
│  │  Breakpoints on:                                            │     │
│  │    • any instruction line                                   │     │
│  │    • LABEL (conditional — break when jumped to)             │     │
│  │    • MEASURE (break on measurement outcome == 0 or 1)       │     │
│  └────────────────────────────┬────────────────────────────────┘     │
│                               │                                      │
│  ┌────────────────────────────▼────────────────────────────────┐     │
│  │  Webview Panels (TypeScript + Chart.js)                     │     │
│  │                                                             │     │
│  │  ┌──────────────────┐  ┌────────────────────────────────┐  │     │
│  │  │ CIRCUIT DIAGRAM  │  │  QUANTUM STATE INSPECTOR        │  │     │
│  │  │ (SVG, updates    │  │  Amplitude table: |state⟩ re im │  │     │
│  │  │  current gate    │  │  Prob bar chart                 │  │     │
│  │  │  highlighted)    │  │  Phase wheel                    │  │     │
│  │  └──────────────────┘  └────────────────────────────────┘  │     │
│  │                                                             │     │
│  │  ┌──────────────────────────────────────────────────────┐  │     │
│  │  │  MEASUREMENT LOG  (scrolling — qubit, outcome, step) │  │     │
│  │  └──────────────────────────────────────────────────────┘  │     │
│  └─────────────────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────────────────┘
```

### VS Code Features

| Feature | Description |
|---------|-------------|
| Syntax highlighting | Gate mnemonics, register names, labels, comments, angles |
| Error squiggles | Lex/parse errors shown inline without running |
| Hover docs | Hover over `H` → shows "Hadamard gate — creates equal superposition" |
| Autocomplete | Type `CN` → suggests `CNOT ctrl tgt` with parameter hints |
| Go-to-definition | Ctrl+click a LABEL or GATE name to jump to its definition |
| Step debugger | F10 to step one gate, watch state update live |
| Breakpoints | Click gutter to set breakpoint; break on MEASURE outcome |
| Circuit diagram panel | Live SVG updated as you type (using existing `circuit_svg.rs`) |
| State inspector panel | Amplitude table + probability bar chart at current step |
| Inline hints | Show `P(|00⟩)=0.5` as ghost text after CNOT in Bell circuit |

### Extension Structure

```
vscode-astracore/
  package.json              — extension manifest (activationEvents, commands, languages)
  src/
    extension.ts            — entry point, starts LSP client + DAP client
    lspClient.ts            — LanguageClient wrapping astracore lsp stdio
    debugAdapter.ts         — DebugAdapterDescriptorFactory
    circuitPanel.ts         — WebviewPanel for circuit diagram (posts SVG from LSP)
    statePanel.ts           — WebviewPanel for amplitude/phase viewer
  syntaxes/
    aql.tmLanguage.json     — TextMate grammar for .aql files
  language-configuration.json — bracket pairs, comments config
  media/
    circuit.js              — Chart.js panel script
    state.js                — amplitude table + phase wheel script

src/lsp/
  mod.rs
  server.rs                 — LSP server (tower-lsp crate)
    hover.rs                — gate hover documentation
    completion.rs           — gate mnemonic completions
    diagnostics.rs          — lex/parse error → LSP Diagnostic
    goto.rs                 — go-to-definition for LABEL / GATE
  debug_adapter.rs          — DAP server (stepped execution via execute_steps())
```

### Rust LSP Server (tower-lsp)

New Cargo feature:
```toml
[features]
lsp = ["dep:tower-lsp", "dep:tokio-util"]
```

Start from CLI:
```bash
# VS Code calls this automatically via extension activation
astracore lsp          # starts LSP server on stdio
astracore dap          # starts DAP server on stdio (for debugger)
```

### DAP Debug Session Flow

```
User presses F5 in VS Code on bell.aql
    │
    ▼
extension.ts → launch request → DAP server (astracore dap)
    │
    ▼
astracore: parse bell.aql → Program
           allocate StepExecutor { state, pc: 0 }
           send: initialized + StoppedEvent (step=0, "Initial |0…0⟩")
    │
User presses F10 (step)
    │
    ▼
DAP server: execute one instruction (H 0)
            capture probabilities snapshot
            send: StoppedEvent(step=1, reason="step")
                  update circuitPanel SVG (current gate highlighted in green)
                  update statePanel amplitude table
    │
User sets breakpoint on MEASURE_ALL line
    │
    ▼
DAP server: continue until pc reaches MEASURE_ALL
            send: StoppedEvent(step=N, reason="breakpoint")
```

### TextMate Grammar (aql.tmLanguage.json)

Scopes for syntax highlighting:
```
keyword.control.aql       → QREG GATE END CALL IF GOTO LABEL BARRIER REPEAT INCLUDE
keyword.operator.aql      → MEASURE MEASURE_ALL
support.function.aql      → H X Y Z S T RX RY RZ PHASE CNOT CZ SWAP CCX
constant.numeric.aql      → PI PI_2 PI_4 float literals
variable.other.aql        → qubit register names (data, anc, q0)
comment.line.aql          → // …
string.quoted.double.aql  → "filename" in INCLUDE
```

### Tests to Add (~8)

```
lsp::tests::test_diagnostics_lex_error_squiggle
lsp::tests::test_diagnostics_parse_error_qubit_out_of_range
lsp::tests::test_hover_h_gate_returns_description
lsp::tests::test_completion_cn_suggests_cnot
lsp::tests::test_goto_definition_label
lsp::tests::test_goto_definition_gate
lsp::tests::test_dap_step_bell_circuit_3_snapshots
lsp::tests::test_dap_breakpoint_on_measure
```

---

## Implementation Order

```
Phase 1 — GPU Backend
  1a. Add Cargo feature flags (cuda / wgpu)
  1b. Define GpuBackend trait (src/simulator/gpu/mod.rs)
  1c. Implement WgpuBackend first (cross-platform, no NVIDIA required for dev)
  1d. Implement CudaBackend (NVIDIA CI runner needed)
  1e. Add --backend gpu / --backend wgpu CLI flags
  1f. Add astracore devices command
  1g. Add automatic CPU fallback
  1h. Add 12 tests (skip tests if GPU unavailable)
  1i. Update CI: add wgpu job (software-rendered WGPU works on any CI runner)

Phase 2 — Distributed Simulation
  2a. Add dist Cargo feature flag
  2b. Implement state partitioning math (partition.rs)
  2c. Implement worker binary (src/bin/worker.rs)
  2d. Implement coordinator (coordinator.rs) with TCP + bincode
  2e. Implement cross-shard gate exchange protocol
  2f. Add --dist / --nodes / --cluster CLI flags
  2g. Loopback tests (coordinator + worker on same machine via localhost)
  2h. Add 10 tests
  2i. Update CI: loopback distributed test job

Phase 3 — Visual Debugger
  3a. Add lsp Cargo feature flag + tower-lsp dependency
  3b. Implement diagnostics (lex/parse error → LSP Diagnostic)
  3c. Implement hover documentation (gate descriptions)
  3d. Implement code completion (gate mnemonics)
  3e. Implement go-to-definition (LABEL / GATE)
  3f. Implement DAP debug adapter (wrap existing execute_steps())
  3g. Add astracore lsp and astracore dap CLI entry points
  3h. Scaffold VS Code extension (vscode-astracore/) with TypeScript
  3i. Implement TextMate grammar (aql.tmLanguage.json)
  3j. Implement circuit diagram webview panel
  3k. Implement state inspector webview panel
  3l. Publish to VS Code Marketplace
  3m. Add 8 Rust LSP/DAP tests
```

---

## New Dependencies (if all features enabled)

| Crate | Version | Feature | Purpose |
|-------|---------|---------|---------|
| `cudarc` | 0.12 | `cuda` | NVIDIA CUDA device management + kernel launch |
| `wgpu` | 0.20 | `wgpu` | Cross-platform GPU compute (WebGPU standard) |
| `bytemuck` | 1.x | `cuda`/`wgpu` | Safe cast `Vec<Complex>` → `&[u8]` for GPU upload |
| `bincode` | 2.x | `dist` | Zero-copy binary serialisation for shard exchange |
| `tower-lsp` | 0.20 | `lsp` | Async LSP server framework |
| `tokio-util` | 0.7 | `lsp` | Codec for LSP framing |

Default build (no features) adds **zero** new dependencies.

---

## v3 Target Test Count

| Phase | New tests | Running total |
|-------|----------:|:--------------|
| v2 complete | — | 361 |
| Phase 1 (GPU) | +12 | 373 |
| Phase 2 (Dist) | +10 | 383 |
| Phase 3 (LSP/DAP) | +8 | 391 |

---

## Deferred (Post v3)

- **OpenQASM 3.0 importer** (complex type system — deferred from v2)
- **GPU-accelerated MPS** (tensor contractions on GPU using cuBLAS)
- **Noise model on GPU** (density matrix simulation on GPU)
- **AQL Playground** (hosted web service — cloud dependency)
- **Plugin marketplace** (package registry for community plugins)
- **VS Code Marketplace publish pipeline** (requires signing + CI publisher)
