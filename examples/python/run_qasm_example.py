"""
OpenQASM Import — AstraCore Python API demo
===========================================
Runs OpenQASM 2.0 programs directly from Python using run_qasm().

AstraCore supports: h, x, y, z, s, t, sdg, tdg, cx, cy, cz, ch,
                    swap, ccx, rx, ry, rz, p, u1, u2, u3, cswap
Angle arithmetic: pi, pi/2, 3*pi/4, -pi/4, numeric literals.

Run:
    python examples/python/run_qasm_example.py
"""
from astracore import run_qasm

# ── Bell state in QASM ────────────────────────────────────────────────────────

bell_qasm = """
OPENQASM 2.0;
include "qelib1.inc";

qreg q[2];
creg c[2];

h q[0];
cx q[0], q[1];

measure q[0] -> c[0];
measure q[1] -> c[1];
"""

result = run_qasm(bell_qasm)
print("=== Bell State (from QASM) ===")
print(f"  P(|00⟩) = {result.prob_of('00'):.4f}")  # expect ~0.5
print(f"  P(|11⟩) = {result.prob_of('11'):.4f}")  # expect ~0.5
print()

# ── Toffoli gate (CCX) in QASM ────────────────────────────────────────────────

toffoli_qasm = """
OPENQASM 2.0;
include "qelib1.inc";

qreg q[3];

// Set control qubits to |1⟩
x q[0];
x q[1];

// Toffoli: flips q[2] when q[0]=1 and q[1]=1
ccx q[0], q[1], q[2];

measure q[0] -> q[0];
measure q[1] -> q[1];
measure q[2] -> q[2];
"""

result = run_qasm(toffoli_qasm)
print("=== Toffoli Gate (CCX) ===")
print(f"  Input: |110⟩, Expected output: |111⟩")
print(f"  Measured qubit 2 = {int(result.outcome(2))}")  # expect 1
print()

# ── Parametric rotation in QASM ───────────────────────────────────────────────

rotation_qasm = """
OPENQASM 2.0;
include "qelib1.inc";

qreg q[1];

// Ry(pi/3): rotates by 60 degrees
ry(pi/3) q[0];

measure q[0] -> q[0];
"""

import math
result = run_qasm(rotation_qasm)
expected = math.sin(math.pi / 6) ** 2  # P(|1⟩) = sin²(θ/2) where θ = π/3
print("=== Rotation Ry(π/3) ===")
print(f"  P(|1⟩) = {result.probabilities[1]:.4f}  (expected {expected:.4f})")
