"""
Bell State — AstraCore Python API demo
======================================
Creates the maximally entangled Bell state (|00⟩ + |11⟩) / √2
using the Circuit builder API.

Run:
    pip install maturin
    cd astracore-py && maturin develop && cd ..
    python examples/python/bell_state.py
"""
from astracore import Circuit

# Build a 2-qubit Bell state circuit
c = Circuit(2)
c.h(0)         # H on qubit 0: |0⟩ → (|0⟩ + |1⟩) / √2
c.cnot(0, 1)   # CNOT 0→1: entangles the pair
c.measure_all()

# Run on the default statevector backend
result = c.run()

print("=== Bell State ===")
print(f"Probabilities: {[round(p, 4) for p in result.probabilities]}")
print(f"  P(|00⟩) = {result.prob_of('00'):.4f}")   # expect ~0.5
print(f"  P(|01⟩) = {result.prob_of('01'):.4f}")   # expect ~0.0
print(f"  P(|10⟩) = {result.prob_of('10'):.4f}")   # expect ~0.0
print(f"  P(|11⟩) = {result.prob_of('11'):.4f}")   # expect ~0.5
print(f"Measured bitstring: {result.bitstring()}")
print(f"Qubit 0 outcome: {result.outcome(0)}")
print(f"Qubit 1 outcome: {result.outcome(1)}")
