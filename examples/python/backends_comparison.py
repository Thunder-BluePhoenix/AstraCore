"""
Backend Comparison — AstraCore Python API demo
==============================================
Runs the same circuit on all four backends and compares results.

Backends:
  statevector — exact simulation, exponential memory (default)
  mps         — Matrix Product States, polynomial for low entanglement
  clifford    — stabilizer tableau, O(n²) for Clifford-only circuits
  sparse      — sparse HashMap statevector, efficient for sparse states

Run:
    python examples/python/backends_comparison.py
"""
import time
from astracore import Circuit

# ── GHZ circuit (Clifford-only, works on all backends) ───────────────────────

n = 6
c = Circuit(n)
c.h(0)
for i in range(n - 1):
    c.cnot(i, i + 1)
c.measure_all()

print(f"=== {n}-qubit GHZ circuit on all backends ===")
print()

backends = [
    ("statevector", {}),
    ("mps",         {"bond_dim": 16}),
    ("clifford",    {}),
    ("sparse",      {}),
]

for backend, kwargs in backends:
    t0 = time.perf_counter()
    result = c.run(backend=backend, **kwargs)
    elapsed = time.perf_counter() - t0

    p_zero = result.prob_of("0" * n)
    p_one  = result.prob_of("1" * n)
    print(f"  [{backend:12s}]  P(|{'0'*n}⟩)={p_zero:.4f}  P(|{'1'*n}⟩)={p_one:.4f}  ({elapsed*1000:.2f} ms)")

print()

# ── Clifford-only: 20 qubits ─────────────────────────────────────────────────

n = 20
c20 = Circuit(n)
c20.h(0)
for i in range(n - 1):
    c20.cnot(i, i + 1)
c20.measure_all()

print(f"=== {n}-qubit GHZ — Clifford vs Statevector vs MPS ===")

for backend, kwargs in [("statevector", {}), ("mps", {"bond_dim": 32}), ("clifford", {})]:
    t0 = time.perf_counter()
    result = c20.run(backend=backend, **kwargs)
    elapsed = time.perf_counter() - t0
    p_zero = result.prob_of("0" * n)
    print(f"  [{backend:12s}]  P(all_zeros)={p_zero:.4f}  ({elapsed*1000:.2f} ms)")
