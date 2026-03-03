# AstraCore Python Examples

These examples demonstrate the AstraCore Python API built with PyO3.

## Setup

```bash
# Install maturin (the PyO3 build tool)
pip install maturin

# Build and install the Python package in development mode
cd astracore-py
maturin develop --release

# Return to project root
cd ..
```

## Running Examples

```bash
python examples/python/bell_state.py          # Bell state basics
python examples/python/ghz_state.py           # GHZ state (statevector + MPS + shots)
python examples/python/teleportation.py       # Quantum teleportation with feedforward
python examples/python/grovers.py             # 3-qubit Grover's search
python examples/python/shot_sampling.py       # Statistical measurement sampling
python examples/python/backends_comparison.py # Same circuit on all 4 backends
python examples/python/run_qasm_example.py    # OpenQASM 2.0 import
python examples/python/mps_large_circuit.py   # 50-qubit MPS simulation
python examples/python/dashboard_example.py            # Generate HTML reports
python examples/python/dashboard_example.py serve      # Browser dashboard (Ctrl-C to stop)
python examples/python/dashboard_example.py dash       # Terminal TUI (q to quit)
python examples/python/custom_gates.py                 # register_gate + @gate decorator
```

## API Overview

```python
from astracore import Circuit, SimResult, ShotSimResult, CircuitAnalysis
from astracore import run_aql, run_aql_shots, run_qasm
from astracore import run_file, analyze_aql, analyze_file
from astracore import PI, PI_2

# Circuit builder
c = Circuit(n_qubits)
c.h(0); c.x(1); c.y(0); c.z(1); c.s(0); c.t(1)
c.rx(0, PI); c.ry(1, PI_2); c.rz(0, 0.78); c.phase(0, PI_2)
c.cnot(0, 1); c.cz(0, 1); c.swap(0, 1)
c.toffoli(0, 1, 2)       # or c.ccx(0, 1, 2)
c.call("my_gate", [0, 1])  # call a named AQL gate
c.measure(0); c.measure_all(); c.barrier()

# Single-shot execution
result = c.run()                          # statevector (default)
result = c.run(backend="mps", bond_dim=128)
result = c.run(backend="clifford")
result = c.run(backend="sparse")

# SimResult fields
result.num_qubits                         # int
result.probabilities                      # list[float], len=2^n
result.measurements                       # list[(qubit, outcome)]
result.outcome(qubit)                     # bool | None
result.bitstring()                        # "010..." | None
result.prob_of("010")                     # float

# Shot-based sampling
result = c.run(shots=1000)                # ShotSimResult
result.counts                             # dict[str, int]
result.n_shots                            # int
result.n_qubits                           # int
result.prob("01")                         # float
result.most_common()                      # list[(str, int)] sorted by count

# Free functions — run AQL source or file
run_aql("QREG 2\nH 0\nCNOT 0 1\nMEASURE_ALL")
run_aql_shots("QREG 2\nH 0\nCNOT 0 1\nMEASURE_ALL", shots=1000)
run_qasm("OPENQASM 2.0;\nqreg q[2];\nh q[0];\ncx q[0],q[1];")
run_file("examples/bell.aql")             # load and run .aql file from disk
run_file("examples/ghz.aql", backend="mps", bond_dim=128)

# Static analysis (no execution)
analysis = analyze_aql("QREG 2\nH 0\nCNOT 0 1\nMEASURE_ALL")
analysis = analyze_file("examples/bell.aql")
analysis.circuit_depth                    # int
analysis.gate_count                       # int
analysis.two_qubit_gate_count             # int
analysis.is_clifford                      # bool
analysis.gate_histogram                   # dict[str, int]
analysis.qubit_utilization                # list[int]

# Dashboard (also available as free functions: run_aql_report, run_aql_serve, run_aql_dash)
c.html_report("report.html")    # write standalone HTML — open in any browser
c.serve(port=8080)               # blocking: interactive SPA at http://localhost:8080
c.dash()                         # blocking: terminal TUI (press q to quit)

from astracore import run_aql_report, run_aql_serve, run_aql_dash
run_aql_report(source, "report.html")
run_aql_serve(source, port=8080)
run_aql_dash(source)

# Custom gate definition — two approaches
import astracore as ac

# 1. Matrix-based (statevector backend only)
sx = [[0.5+0.5j, 0.5-0.5j], [0.5-0.5j, 0.5+0.5j]]
ac.register_gate("sx", sx)           # register sqrt(X)

c = ac.Circuit(1)
c.call("sx", [0])                    # CALL sx 0 in AQL
result = c.run()

# 2. Decorator-based (all backends)
@ac.gate(n_qubits=2, name="bell")
def bell_gate(c, qubits):
    c.h(qubits[0])
    c.cnot(qubits[0], qubits[1])

c = ac.Circuit(4)
c.call("bell", [0, 1])
c.call("bell", [2, 3])
c.measure_all()
result = c.run(backend="mps", shots=1000)
```
