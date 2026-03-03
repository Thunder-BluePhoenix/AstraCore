# AQL — Assembler Quantum Language Reference

**Version:** 2.0  **AstraCore:** v2

> AQL is a minimal, human-readable quantum assembly language.
> A 2-qubit Bell pair needs exactly 4 lines. No imports, no headers, no boilerplate.

---

## Quick Reference

```
QREG 2          # allocate 2 qubits
H 0             # Hadamard on qubit 0
CNOT 0 1        # CNOT: control=0, target=1
MEASURE_ALL     # measure all qubits
```

---

## 1. Program Structure

Every AQL program has two sections:

1. **Gate/register declarations** (optional `GATE…END` blocks, `QREG` directive)
2. **Instructions** — executed top-to-bottom

Blank lines and `#` comments are ignored.

---

## 2. Qubit Register Declaration

### Anonymous register (v1)
```aql
QREG n          # allocate n qubits (indices 0 … n-1)
```

### Named registers (v2)
```aql
QREG data[4]      # allocates qubits 0–3, named data[0]..data[3]
QREG ancilla[2]   # allocates qubits 4–5, named ancilla[0]..ancilla[1]

H data[0]
CNOT data[0] ancilla[0]
```
Named and anonymous `QREG` cannot be mixed in the same program.
Maximum qubit index: **999**.

---

## 3. Gate Instructions

### Single-qubit gates

| Mnemonic | Effect | Matrix |
|----------|--------|--------|
| `H q` | Hadamard | `(1/√2)[[1,1],[1,-1]]` |
| `X q` | Pauli-X (NOT) | `[[0,1],[1,0]]` |
| `Y q` | Pauli-Y | `[[0,-i],[i,0]]` |
| `Z q` | Pauli-Z (phase flip) | `[[1,0],[0,-1]]` |
| `S q` | S gate (√Z) | `[[1,0],[0,i]]` |
| `T q` | T gate (⁸√Z) | `[[1,0],[0,e^(iπ/4)]]` |
| `RX q θ` | X rotation by θ radians | `cos(θ/2)I − i·sin(θ/2)X` |
| `RY q θ` | Y rotation by θ radians | `cos(θ/2)I − i·sin(θ/2)Y` |
| `RZ q θ` | Z rotation by θ radians | `cos(θ/2)I − i·sin(θ/2)Z` |
| `PHASE q θ` | Phase shift | `[[1,0],[0,e^(iθ)]]` |

### Multi-qubit gates

| Mnemonic | Qubits | Description |
|----------|--------|-------------|
| `CNOT c t` | 2 | Controlled-NOT: flips `t` if `c == 1` |
| `CZ c t` | 2 | Controlled-Z: applies Z to `t` if `c == 1` |
| `SWAP a b` | 2 | Exchange qubit states |
| `TOFFOLI c0 c1 t` | 3 | CCNOT: flips `t` if both controls are 1 |

### Measurement

| Mnemonic | Description |
|----------|-------------|
| `MEASURE q` | Collapse qubit `q`; store result in classical register `q` |
| `MEASURE_ALL` | Measure every qubit in order |

---

## 4. Control Flow (v1)

```aql
LABEL name          # define a jump target (name = any identifier)
GOTO name           # unconditional jump
IF q GOTO name      # jump if classical[q] == 1 (qubit must have been measured)
IFNOT q GOTO name   # jump if classical[q] == 0
```

### High-level sugar (v2)

```aql
IFMEASURED q THEN
  # body executed if classical[q] == 1
END

IFNOTMEASURED q THEN
  # body executed if classical[q] == 0
END
```

Both desugar to `IF/IFNOT + GOTO + LABEL` at parse time.

---

## 5. Structured Loops (v2)

```aql
REPEAT 3
  H 0
  X 1
END
```

`REPEAT N … END` is a **compile-time** unrolling — equivalent to writing the body N times.
Nesting is supported: `REPEAT` inside `GATE`, `GATE` inside `REPEAT`.

---

## 6. Custom Gate Definitions

```aql
GATE name num_qubits
  # body: gate instructions using local qubit indices 0, 1, …
  H 0
  CNOT 0 1
END

# Invoke with CALL
CALL name q0 q1     # maps local 0→q0, 1→q1
```

- Gate bodies can contain any instruction except `MEASURE`, `MEASURE_ALL`, and control flow.
- Recursion is **not** supported.
- Gate definitions can appear anywhere before they are called.

---

## 7. Include Directive (v2)

```aql
INCLUDE "gates/qft.aql"   # path relative to including file
```

- Maximum include depth: **16** (prevents circular includes).
- Included files may contain `GATE` definitions and instructions.
- `QREG` in an included file is ignored (qubit count comes from the top-level program).

---

## 8. Barrier

```aql
BARRIER    # marks a serialisation boundary (no gate is applied)
```

The optimizer will not reorder gates across a `BARRIER`.

---

## 9. Number Literals

| Form | Example | Notes |
|------|---------|-------|
| Integer | `0`, `4`, `999` | Qubit and repeat count arguments |
| Decimal float | `3.14159` | Rotation angle arguments |
| Scientific | `1.57e0` | Same as `1.57` |
| Negative | `-1.5708` | Prefix with `-` |

Common constants:

| Constant | Value | Usage |
|----------|-------|-------|
| `PI` | 3.14159265… | `RX 0 PI` |
| `PI_2` | 1.5707963… | `RY 0 PI_2` |

---

## 10. Execution Model

- **State:** `2^n`-dimensional complex state vector, initialized to `|0…0⟩`.
- **Measurement:** probabilistic collapse; recorded in a classical bit register.
- **Loops:** `GOTO`-based loops execute at runtime; step limit = 1,000,000.
- **Optimizer:** the peephole pass cancels pairs `H·H`, `X·X`, `Z·Z`, `S·S·S·S` automatically.

---

## 11. Complete Example — Quantum Teleportation

```aql
# Teleport qubit 0 (initially |+⟩) to qubit 2
QREG 3
H 0              # prepare qubit 0 in |+⟩

# Create Bell pair on qubits 1, 2
H 1
CNOT 1 2

# Bell measurement of qubits 0, 1
CNOT 0 1
H 0
MEASURE 0
MEASURE 1

# Feedforward corrections
IFMEASURED 1 THEN
  X 2
END
IFMEASURED 0 THEN
  Z 2
END

MEASURE 2        # qubit 2 now in |+⟩ state → P(0)=P(1)=0.5
```

---

## 12. CLI Reference

```bash
astracore run   <file.aql>                       # statevector
astracore run   --backend mps      <file.aql>    # Matrix Product States (50–200+ qubits)
astracore run   --backend clifford <file.aql>    # Clifford simulator (unlimited qubits)
astracore run   --backend sparse   <file.aql>    # sparse statevector
astracore run   --shots 1000       <file.aql>    # shot-based sampling

astracore analyze <file.aql>                     # circuit analysis (no execution)
astracore opt     <file.aql>                     # show optimized circuit

astracore dash    <file.aql>                     # terminal TUI dashboard
astracore report  <file.aql> [out.html]          # write HTML report
astracore serve   <file.aql> [port]              # web dashboard (default :8080)

astracore export  <file.aql> [out.qasm]          # export as OpenQASM 2.0
astracore import  <file.qasm>                    # run OpenQASM 2.0
```

---

## 13. Error Messages

AstraCore reports errors with source context and "did you mean?" suggestions:

```
✗ Parse error at line 5, column 3:
    5 │ CNTO 0 1
      │ ^^^^ unknown instruction 'CNTO'
      │ did you mean: CNOT ?
```

---

*AQL is designed to be the simplest quantum assembly language possible.*
*The entire language fits on two pages. Everything beyond this document is an extension.*
