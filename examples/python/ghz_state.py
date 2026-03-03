"""
GHZ State — AstraCore Python API demo
======================================
Creates the Greenberger–Horne–Zeilinger state:
    (|000…0⟩ + |111…1⟩) / √2

Demonstrates both the statevector and MPS backends.
The MPS backend can handle 50+ qubits where statevector would require 2^n memory.

Run:
    python examples/python/ghz_state.py
"""
import math
from astracore import Circuit

# ── 3-qubit GHZ (statevector backend) ────────────────────────────────────────

n = 3
c = Circuit(n)
c.h(0)
for i in range(n - 1):
    c.cnot(i, i + 1)
c.measure_all()

result = c.run(backend="statevector")
print(f"=== {n}-qubit GHZ (statevector) ===")
p_all_zero = result.prob_of("0" * n)
p_all_one  = result.prob_of("1" * n)
print(f"  P(|{'0'*n}⟩) = {p_all_zero:.4f}")   # expect ~0.5
print(f"  P(|{'1'*n}⟩) = {p_all_one:.4f}")    # expect ~0.5
print(f"  Sum of others = {1 - p_all_zero - p_all_one:.6f}")  # expect ~0.0
print()

# ── 10-qubit GHZ (MPS backend) ────────────────────────────────────────────────

n = 10
c = Circuit(n)
c.h(0)
for i in range(n - 1):
    c.cnot(i, i + 1)
c.measure_all()

result = c.run(backend="mps", bond_dim=16)
print(f"=== {n}-qubit GHZ (MPS, bond_dim=16) ===")
p_all_zero = result.prob_of("0" * n)
p_all_one  = result.prob_of("1" * n)
print(f"  P(|{'0'*n}⟩) = {p_all_zero:.4f}")   # expect ~0.5
print(f"  P(|{'1'*n}⟩) = {p_all_one:.4f}")    # expect ~0.5
print()

# ── Shot-based sampling ───────────────────────────────────────────────────────

n = 3
c = Circuit(n)
c.h(0)
for i in range(n - 1):
    c.cnot(i, i + 1)
c.measure_all()

shots = c.run(shots=2000)
print(f"=== {n}-qubit GHZ — 2000 shots ===")
for bitstring, count in shots.most_common():
    bar = "█" * (count * 40 // shots.n_shots)
    print(f"  |{bitstring}⟩  {count:5d}  {bar}")
print(f"  P(all zeros) estimate = {shots.prob('0'*n):.3f}")
print(f"  P(all ones)  estimate = {shots.prob('1'*n):.3f}")
