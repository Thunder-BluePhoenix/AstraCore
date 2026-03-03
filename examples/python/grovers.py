"""
Grover's Search Algorithm — AstraCore Python API demo
======================================================
3-qubit Grover's algorithm searching for the marked state |101⟩.

One Grover iteration:
  1. Oracle: flips the phase of |101⟩ via a 3-controlled-Z equivalent
     (implemented using X gates + Toffoli + Phase π + Toffoli + X gates)
  2. Diffusion: amplifies the marked state

For n=3, one iteration suffices to get P(|101⟩) ≈ 0.78 from 1/8 = 0.125.

Run:
    python examples/python/grovers.py
"""
import math
from astracore import Circuit

n = 3
target = "101"  # search target in binary (q0=1, q1=0, q2=1)

c = Circuit(n)

# ── Initialisation: uniform superposition ────────────────────────────────────
for q in range(n):
    c.h(q)

# ── Grover oracle for |101⟩ ──────────────────────────────────────────────────
# Flip qubits where target bit is 0 (qubit 1 in "101")
c.x(1)         # qubit 1 target bit is 0 → flip it
# Now oracle marks |111⟩ — apply CCZ = Toffoli + Phase correction
# CCZ = H·Toffoli·H on the target qubit
c.h(2)
c.toffoli(0, 1, 2)
c.h(2)
# Unflip
c.x(1)

# ── Grover diffusion (inversion about average) ───────────────────────────────
for q in range(n):
    c.h(q)
for q in range(n):
    c.x(q)
c.h(2)
c.toffoli(0, 1, 2)
c.h(2)
for q in range(n):
    c.x(q)
for q in range(n):
    c.h(q)

# ── Measure all ──────────────────────────────────────────────────────────────
c.measure_all()

result = c.run()

print("=== Grover's Algorithm (3 qubits, target=|101⟩) ===")
print("Probability distribution after 1 Grover iteration:")
for i in range(8):
    bs = format(i, f"0{n}b")
    p  = result.probabilities[i]
    bar = "█" * int(p * 40)
    marker = " ← TARGET" if bs == target else ""
    print(f"  |{bs}⟩  {p:.4f}  {bar}{marker}")

print()
print(f"P(|{target}⟩) = {result.prob_of(target):.4f}  (uniform = {1/2**n:.4f})")
print(f"Speed-up factor ≈ {result.prob_of(target) / (1/2**n):.1f}×")
