# Quantum Computing from Assembly — Course Outline

**Using AstraCore and AQL as the teaching platform**

> No math prerequisites beyond high school algebra.
> No hardware needed. One command: `cargo install astracore`.

---

## Motivation

Most quantum computing courses start with linear algebra or quantum mechanics.
This course takes the opposite approach: **start with running circuits immediately**,
then build intuition by observing outputs, and finally connect the observations to theory.

AQL is the ideal teaching language because:
- 4 lines run a Bell pair (simplest entangled state)
- Error messages say "did you mean CNOT?" when you type CNTO
- The `astracore dash` command shows probabilities live as you type

---

## Module 1 — Your First Quantum Program (2 hours)

### 1.1 Setup (15 min)
```bash
cargo install astracore          # one command
echo "QREG 1\nH 0\nMEASURE 0" | astracore run /dev/stdin
```

### 1.2 The Superposition Experiment
- Write a 1-qubit circuit: `QREG 1 / H 0 / MEASURE 0`
- Run it 10 times. Sometimes 0, sometimes 1 — why?
- Concept: **superposition** — a qubit can be 0 and 1 simultaneously
- Run with `--shots 1000`: the law of large numbers → 50/50

### 1.3 Lab: Vary the Starting State
- Start in `|0⟩` (default), apply H, measure
- Start in `|1⟩` (apply X first), apply H, measure
- Predict: what does H·H do? Test it. Optimizer says "0 gates executed" — why?

### 1.4 Concept: Probability Amplitudes
- Connect to the dashboard: `astracore dash circuit.aql`
- The probability bars are |amplitude|²
- Complex amplitudes ≠ probabilities — but |z|² is always real and non-negative

**Exercises:**
1. What is P(0) after `RY 0 1.0`? Run and verify.
2. Find θ such that `RX 0 θ` gives P(1) = 0.75.

---

## Module 2 — Two-Qubit Gates and Entanglement (2 hours)

### 2.1 The CNOT Gate
- `QREG 2 / X 0 / CNOT 0 1 / MEASURE_ALL` — classical control
- Truth table: test all 4 input states (00, 01, 10, 11)
- Key insight: CNOT is **controlled-NOT** — only flips target if control is 1

### 2.2 Creating Entanglement
- `QREG 2 / H 0 / CNOT 0 1 / MEASURE_ALL`
- Run 1000 shots: always 00 or 11, never 01 or 10
- **Entanglement**: measuring one qubit instantly determines the other
- Not communication — both outcomes are random, but correlated

### 2.3 Bell States
The four maximally entangled 2-qubit states:

| State | Circuit | Outcomes |
|-------|---------|----------|
| Φ+ | `H 0 / CNOT 0 1` | 00 and 11 |
| Φ- | `H 0 / Z 0 / CNOT 0 1` | 00 and 11 (with phase) |
| Ψ+ | `H 0 / X 1 / CNOT 0 1` | 01 and 10 |
| Ψ- | `H 0 / X 1 / Z 0 / CNOT 0 1` | 01 and 10 |

### 2.4 Lab: Verify with the Dashboard
- Open `astracore serve circuit.aql` in browser
- The probability chart shows the quantum state visually
- Try the Share button to send your circuit to a classmate

**Exercises:**
1. Modify the Bell circuit so it produces only `|01⟩` and `|10⟩`.
2. How many CNOT gates does a 4-qubit GHZ state need? Write and test it.

---

## Module 3 — Quantum Algorithms (3 hours)

### 3.1 Deutsch's Algorithm — Quantum Speedup in 4 Lines
- Classical: 2 function evaluations to determine if f is constant or balanced
- Quantum: **1 evaluation** (superposition queries both inputs simultaneously)

```aql
# Deutsch: is f(x) = 0 constant or balanced?
QREG 2
X 1           # ancilla starts in |1⟩
H 0           # control in superposition
H 1
CNOT 0 1      # f_balanced oracle
H 0
MEASURE 0     # 0 → constant, 1 → balanced
```

### 3.2 Grover's Search — Finding a Needle in a Haystack
- Search N items with √N queries (vs N classical queries)
- Demo with 3 qubits (8 items): find item marked by oracle

```aql
# 3-qubit Grover (2 iterations) — finds |101⟩
QREG 3
H 0; H 1; H 2       # superposition
# Iteration 1
# ... (see examples/grover.aql)
MEASURE_ALL
```

Run `python examples/python/grovers.py` for the full circuit.

### 3.3 Quantum Teleportation
- Transfer a quantum state using 2 classical bits + entanglement
- Not "teleporting matter" — teleporting the description of a state

```aql
# Teleport qubit 0 → qubit 2  (see examples/teleportation.aql)
QREG 3
H 0
H 1; CNOT 1 2          # Bell pair
CNOT 0 1; H 0
MEASURE 0; MEASURE 1
IFMEASURED 1 THEN X 2 END
IFMEASURED 0 THEN Z 2 END
MEASURE 2
```

**Lab:** What happens if you skip the feedforward corrections?

### 3.4 Quantum Fourier Transform (Advanced)
- The quantum version of FFT: exponentially faster
- Building block of Shor's algorithm (factoring) and quantum phase estimation
- 2-qubit QFT circuit; scale to 4 qubits using `REPEAT` + `GATE…END`

---

## Module 4 — Noise and Error Correction (1 hour)

### 4.1 Why Quantum Computers Break
- Real qubits decohere: state randomly collapses to 0 or 1
- Gate errors: CNOT on real hardware has ~0.1–1% error rate
- Measurement errors: readout is noisy

### 4.2 The Bit-Flip Code (Classical Repetition)
- Protect 1 logical qubit using 3 physical qubits
- 3-qubit repetition code: can correct any single bit flip

### 4.3 Why Clifford Circuits Are Special
- H, S, CNOT, X, Y, Z form the **Clifford group**
- These can be simulated efficiently on a classical computer (Gottesman-Knill)
- AstraCore: `astracore run --backend clifford surface_code.aql`
- Check with `astracore analyze --backend clifford` — "Clifford-only: yes ✓"

---

## Module 5 — Scaling to 50+ Qubits (1 hour)

### 5.1 The Memory Wall
- Statevector: 2^n × 16 bytes. At n=30: 16 GB. At n=50: 16 petabytes.
- Solution: algorithms that don't need the full state vector

### 5.2 Matrix Product States (MPS)
- Store state as a chain of small tensors (bond dimension χ)
- Perfect for circuits with limited entanglement
- Demo: 50-qubit GHZ state

```bash
astracore run --backend mps --bond-dim 64 examples/ghz_50q_mps.aql
```

### 5.3 The Clifford Simulator
- 1000+ qubit circuits for surface codes and error correction
- O(n²) memory, O(n) per gate

```bash
astracore run --backend clifford examples/clifford_1000q.aql
```

---

## Module 6 — Python API for Researchers (1 hour)

```python
import astracore as ac
import numpy as np

# Familiar Python syntax, Rust speed
c = ac.Circuit(2)
c.h(0); c.cnot(0, 1); c.measure_all()
result = c.run(shots=10000)
print(result.counts)  # {'00': 5012, '11': 4988}

# Custom gates from matrices
sqrt_x = np.array([[0.5+0.5j, 0.5-0.5j], [0.5-0.5j, 0.5+0.5j]])
ac.register_gate("sx", sqrt_x)

# Custom gates from Python functions
@ac.gate(n_qubits=2, name="bell")
def bell(c, qubits):
    c.h(qubits[0])
    c.cnot(qubits[0], qubits[1])

# Works with all backends
result = c.run(backend="mps", bond_dim=128, shots=1000)
```

---

## Assessment Ideas

1. **Lab 1:** Implement the 3-qubit GHZ state. Verify with `--shots 1000`.
2. **Lab 2:** Implement Deutsch's algorithm for a constant oracle and a balanced oracle. Show outputs differ.
3. **Lab 3:** Write a 4-qubit Grover circuit that finds `|1010⟩`. Run and confirm P(1010) > 0.9.
4. **Lab 4:** Design a `GATE bell 2 … END` custom gate. Use it to create 3 Bell pairs in a 6-qubit circuit.
5. **Project:** Implement a 3-qubit quantum error correction code (bit-flip code). Simulate one bit-flip error. Show the syndrome circuit detects it.

---

## Appendix — AQL Cheat Sheet

See [docs/aql_spec.md](aql_spec.md) for the complete language reference.

```
QREG n          allocate n qubits
H q             Hadamard
X q / Y q / Z q Pauli gates
S q / T q       phase gates
RX q θ          rotation by θ radians
CNOT c t        controlled-NOT
TOFFOLI c0 c1 t Toffoli (CCNOT)
MEASURE q       measure qubit q
MEASURE_ALL     measure all
GATE name n     define custom gate
CALL name q...  invoke custom gate
REPEAT n...END  loop unrolling
GOTO label      control flow
```

**CLI:**
```bash
astracore run circuit.aql          # execute
astracore run --shots 1000 c.aql   # statistical
astracore dash circuit.aql         # TUI dashboard
astracore serve circuit.aql        # web dashboard
astracore analyze circuit.aql      # static analysis
```
