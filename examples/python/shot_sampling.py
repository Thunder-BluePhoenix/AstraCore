"""
Shot-Based Sampling — AstraCore Python API demo
================================================
Demonstrates statistical measurement sampling with run(shots=N).

Key idea: instead of computing the exact probability distribution (O(2^n) memory),
shot-based sampling runs the circuit N times independently and accumulates
a bitstring histogram. This is how real quantum hardware works.

Run:
    python examples/python/shot_sampling.py
"""
from astracore import Circuit

# ── Bell state — 2000 shots ───────────────────────────────────────────────────

c = Circuit(2)
c.h(0)
c.cnot(0, 1)
c.measure_all()

result = c.run(shots=2000)

print("=== Bell State — 2000 shots ===")
for bs, count in result.most_common():
    prob  = result.prob(bs)
    bar   = "█" * int(prob * 50)
    print(f"  |{bs}⟩  {count:5d} shots  ({prob:.3f})  {bar}")
print()

# ── Multi-qubit histogram ─────────────────────────────────────────────────────

n = 4
c4 = Circuit(n)
for q in range(n):
    c4.h(q)
c4.measure_all()

result4 = c4.run(shots=4000)

print(f"=== {n}-qubit uniform superposition — 4000 shots ===")
print("(Each of the 16 states should appear ~250 times)")
for bs, count in sorted(result4.counts.items()):
    bar = "█" * (count * 20 // result4.n_shots)
    print(f"  |{bs}⟩  {count:4d}  {bar}")
print()

# ── Convergence check ─────────────────────────────────────────────────────────

print("=== Bell P(|00⟩) convergence vs. shot count ===")
c_conv = Circuit(2)
c_conv.h(0)
c_conv.cnot(0, 1)
c_conv.measure_all()

for n_shots in [10, 100, 500, 2000, 10000]:
    r = c_conv.run(shots=n_shots)
    p = r.prob("00")
    err = abs(p - 0.5)
    print(f"  shots={n_shots:6d}  P(|00⟩) = {p:.4f}  error = {err:.4f}")
