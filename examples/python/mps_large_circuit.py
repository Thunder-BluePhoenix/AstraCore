"""
MPS Backend — 50-Qubit Circuit Demo
====================================
Demonstrates the MPS backend on a 50-qubit circuit that is impossible
to simulate with the statevector backend (2^50 × 16 bytes ≈ 16 petabytes).

The GHZ state has only χ=2 entanglement, so MPS handles it in O(n × 4) memory.

Run:
    python examples/python/mps_large_circuit.py
"""
import time
from astracore import Circuit

# ── 50-qubit GHZ via MPS ─────────────────────────────────────────────────────

n = 50
c = Circuit(n)
c.h(0)
for i in range(n - 1):
    c.cnot(i, i + 1)
c.measure_all()

print(f"=== {n}-qubit GHZ (MPS backend, bond_dim=4) ===")
print(f"Statevector would need: 2^{n} × 16 bytes = {2**n * 16 / 1e15:.1f} petabytes")
print(f"MPS needs:              {n} × 4² × 16 bytes = {n * 16 * 16 / 1024:.1f} KB")
print()

t0 = time.perf_counter()
result = c.run(backend="mps", bond_dim=4)
elapsed = time.perf_counter() - t0

p_zero = result.prob_of("0" * n)
p_one  = result.prob_of("1" * n)

print(f"Results:")
print(f"  P(|{'0'*8}…⟩) = {p_zero:.4f}   (expected 0.5000)")
print(f"  P(|{'1'*8}…⟩) = {p_one:.4f}   (expected 0.5000)")
print(f"  Simulation time: {elapsed*1000:.1f} ms")
print()

# ── Bond dimension comparison ─────────────────────────────────────────────────

print("=== Bond dimension vs. accuracy (5-qubit GHZ) ===")
n = 5
c_small = Circuit(n)
c_small.h(0)
for i in range(n - 1):
    c_small.cnot(i, i + 1)
c_small.measure_all()

for bd in [1, 2, 4, 8, 16]:
    r = c_small.run(backend="mps", bond_dim=bd)
    p0 = r.prob_of("0" * n)
    p1 = r.prob_of("1" * n)
    err = abs(p0 - 0.5) + abs(p1 - 0.5)
    print(f"  bond_dim={bd:3d}:  P(all_zeros)={p0:.4f}  P(all_ones)={p1:.4f}  error={err:.6f}")
