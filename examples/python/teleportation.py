"""
Quantum Teleportation — AstraCore Python API demo
==================================================
Teleports a single qubit state |ψ⟩ = cos(θ/2)|0⟩ + sin(θ/2)|1⟩
from Alice (qubit 0) to Bob (qubit 2) using an entangled ancilla pair
and two classical bits.

Protocol:
  1. Prepare |ψ⟩ on qubit 0 with Ry(θ)
  2. Create Bell pair on qubits 1 and 2
  3. Bell measurement on qubits 0 and 1 (Alice's side)
  4. Feedforward corrections on qubit 2 (Bob's side) using IFMEASURED

Run:
    python examples/python/teleportation.py
"""
import math
from astracore import run_aql

# Teleport |+y⟩ = (|0⟩ + i|1⟩)/√2  →  Ry(π/2)|0⟩
theta = math.pi / 2

aql_source = f"""
QREG 3

// Step 1: Prepare state |ψ⟩ on qubit 0
RY 0 {theta:.10f}

// Step 2: Bell pair on qubits 1, 2
H 1
CNOT 1 2

// Step 3: Alice's Bell measurement
CNOT 0 1
H 0
MEASURE 0
MEASURE 1

// Step 4: Bob's feedforward corrections
IFMEASURED 1 THEN
    X 2
END
IFMEASURED 0 THEN
    Z 2
END

// Step 5: Measure Bob's qubit
MEASURE 2
"""

result = run_aql(aql_source)

print("=== Quantum Teleportation ===")
print(f"Teleporting Ry({theta:.4f})|0⟩")
print(f"  Expected P(qubit2=1) = {math.sin(theta/2)**2:.4f}")
print(f"  Measured outcome of qubit 2: {int(result.outcome(2))}")
print()

# Run 1000 shots to estimate probability
from astracore import run_aql_shots
shots = run_aql_shots(aql_source, 1000)
outcomes = {"0": 0, "1": 0}
for bitstring, count in shots.counts.items():
    q2 = bitstring[2]  # qubit 2 is position 2 in "q0q1q2" bitstring
    outcomes[q2] = outcomes.get(q2, 0) + count

p1 = outcomes.get("1", 0) / shots.n_shots
print(f"Shot-based estimate of P(qubit2=1): {p1:.3f} (expected {math.sin(theta/2)**2:.3f})")
