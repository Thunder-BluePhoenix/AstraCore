"""custom_gates.py — register_gate and @gate decorator examples.

Demonstrates two ways to define custom named gates in astracore-py:

1. register_gate(name, matrix) — define a gate from a unitary matrix
   (statevector backend only; supports numpy arrays or lists-of-lists)

2. @gate(n_qubits, name) — define a gate from a Python circuit-builder function
   (works with all backends, supports shot-based sampling)
"""
import astracore as ac

# ── 1. register_gate: sqrt(X) gate via matrix ─────────────────────────────────
print("=== register_gate: sqrt(X) ===")

# SX = sqrt(X) = 0.5 * [[1+i, 1-i], [1-i, 1+i]]
sx_matrix = [
    [0.5 + 0.5j,  0.5 - 0.5j],
    [0.5 - 0.5j,  0.5 + 0.5j],
]
ac.register_gate("sx", sx_matrix)

# Applying SX twice = X (Pauli-X)
c = ac.Circuit(1)
c.call("sx", [0])   # first SX
c.call("sx", [0])   # second SX: SX^2 = X
c.measure(0)

result = c.run(shots=1000)
print(f"SX^2 on |0⟩: counts = {dict(sorted(result.counts.items()))}")
# Expected: mostly '1' (X gate flips |0⟩ → |1⟩)
ones = result.counts.get("1", 0)
print(f"  P(1) ≈ {ones / result.n_shots:.3f}  (expected ~1.0)\n")

# ── 2. @gate decorator: Bell-pair factory gate ─────────────────────────────────
print("=== @gate decorator: bell_pair ===")

@ac.gate(n_qubits=2, name="bell_pair")
def _define_bell_pair(c, qubits):
    """H on first qubit, then CNOT."""
    c.h(qubits[0])
    c.cnot(qubits[0], qubits[1])

# 4-qubit circuit: two independent Bell pairs
circ = ac.Circuit(4)
circ.call("bell_pair", [0, 1])   # pair 1
circ.call("bell_pair", [2, 3])   # pair 2
circ.measure_all()

result = circ.run(shots=4000)
counts = result.counts
print("4-qubit Bell circuit shot histogram (top 4):")
for bs, cnt in result.most_common()[:4]:
    print(f"  |{bs}⟩: {cnt} ({cnt/result.n_shots*100:.1f}%)")
# Expected: ~50% 0000, ~50% 1111

# ── 3. @gate with MPS backend ─────────────────────────────────────────────────
print("\n=== @gate with MPS backend (50-qubit GHZ step) ===")

@ac.gate(n_qubits=2, name="half_entangle")
def _half_entangle(c, qubits):
    c.cnot(qubits[0], qubits[1])

# 6-qubit GHZ with custom gate (statevector, for speed)
circ = ac.Circuit(6)
circ.h(0)
for i in range(5):
    circ.call("half_entangle", [i, i + 1])
circ.measure_all()

result = circ.run(shots=2000)
print("6-qubit GHZ via @gate (top outcomes):")
for bs, cnt in result.most_common()[:4]:
    print(f"  |{bs}⟩: {cnt} ({cnt/result.n_shots*100:.1f}%)")
# Expected: ~50% 000000, ~50% 111111

# ── 4. @gate for custom phase kick gate ───────────────────────────────────────
print("\n=== @gate: custom phase kick ===")

@ac.gate(n_qubits=1, name="phase_kick")
def _phase_kick(c, qubits):
    """T^2 = S (π/2 phase)."""
    c.t(qubits[0])
    c.t(qubits[0])

c = ac.Circuit(1)
c.h(0)
c.call("phase_kick", [0])   # applies S in superposition
c.h(0)
# H · S · H on |0⟩: results in a phase rotation
result = c.run()
print(f"H·S·H on |0⟩: P(0) = {result.prob_of('0'):.4f}, P(1) = {result.prob_of('1'):.4f}")

print("\nAll custom gate examples completed successfully.")
