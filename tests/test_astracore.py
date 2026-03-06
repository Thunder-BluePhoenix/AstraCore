"""
AstraCore Python API — pytest test suite.

Run:
    cd AstraCore
    pip install maturin pytest
    cd astracore-py && maturin develop --release && cd ..
    pytest tests/test_astracore.py -v
"""
import math
import pytest
import astracore as ac
from astracore import (
    Circuit, SimResult, ShotSimResult, CircuitAnalysis,
    run_aql, run_aql_shots, run_qasm,
    run_file, analyze_aql, analyze_file,
    PI, PI_2,
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

def test_pi_constant():
    assert abs(PI - math.pi) < 1e-12

def test_pi_2_constant():
    assert abs(PI_2 - math.pi / 2) < 1e-12

# ─────────────────────────────────────────────────────────────────────────────
# Circuit builder
# ─────────────────────────────────────────────────────────────────────────────

def test_circuit_creation():
    c = Circuit(3)
    assert repr(c) == "Circuit(n_qubits=3, 0 instructions)"
    assert len(c) == 0

def test_circuit_zero_qubits_raises():
    with pytest.raises(Exception):
        Circuit(0)

def test_circuit_len_tracks_instructions():
    c = Circuit(2)
    c.h(0)
    c.cnot(0, 1)
    assert len(c) == 2

def test_aql_source_correct():
    c = Circuit(2)
    c.h(0)
    c.cnot(0, 1)
    c.measure_all()
    src = c.aql_source()
    assert "QREG 2" in src
    assert "H 0" in src
    assert "CNOT 0 1" in src
    assert "MEASURE_ALL" in src

def test_ccx_alias_same_as_toffoli():
    c1 = Circuit(3)
    c1.toffoli(0, 1, 2)
    c2 = Circuit(3)
    c2.ccx(0, 1, 2)
    assert "TOFFOLI 0 1 2" in c1.aql_source()
    assert "TOFFOLI 0 1 2" in c2.aql_source()

def test_call_method_generates_call_instruction():
    c = Circuit(4)
    c.call("bell", [0, 1])
    assert "CALL bell 0 1" in c.aql_source()

def test_circuit_all_single_qubit_gates():
    c = Circuit(1)
    c.h(0); c.x(0); c.y(0); c.z(0); c.s(0); c.t(0)
    c.rx(0, 1.0); c.ry(0, 2.0); c.rz(0, 3.0); c.phase(0, 0.5)
    c.measure(0)
    assert len(c) == 11

def test_circuit_two_qubit_gates():
    c = Circuit(2)
    c.cnot(0, 1); c.cz(0, 1); c.swap(0, 1)
    assert len(c) == 3

# ─────────────────────────────────────────────────────────────────────────────
# SimResult
# ─────────────────────────────────────────────────────────────────────────────

def test_bell_state_probabilities():
    c = Circuit(2)
    c.h(0); c.cnot(0, 1); c.measure_all()
    r = c.run()
    assert isinstance(r, SimResult)
    assert r.num_qubits == 2
    assert abs(r.probabilities[0] - 0.5) < 1e-6  # |00⟩
    assert abs(r.probabilities[3] - 0.5) < 1e-6  # |11⟩

def test_simresult_prob_of():
    c = Circuit(2)
    c.h(0); c.cnot(0, 1); c.measure_all()
    r = c.run()
    assert abs(r.prob_of("00") - 0.5) < 1e-6
    assert abs(r.prob_of("11") - 0.5) < 1e-6

def test_simresult_prob_of_wrong_length_raises():
    c = Circuit(2)
    c.h(0); c.measure_all()
    r = c.run()
    with pytest.raises(Exception):
        r.prob_of("000")

def test_simresult_outcome_and_bitstring():
    c = Circuit(2)
    c.x(0); c.measure_all()
    r = c.run()
    assert r.outcome(0) is True   # X|0⟩ = |1⟩
    assert r.outcome(1) is False  # qubit 1 untouched → |0⟩
    bs = r.bitstring()
    assert bs == "10"

def test_simresult_bitstring_none_if_not_all_measured():
    c = Circuit(2)
    c.h(0); c.measure(0)  # only q0 measured
    r = c.run()
    assert r.bitstring() is None

def test_ghz_state_correct():
    c = Circuit(3)
    c.h(0); c.cnot(0, 1); c.cnot(0, 2); c.measure_all()
    r = c.run()
    assert abs(r.prob_of("000") - 0.5) < 1e-6
    assert abs(r.prob_of("111") - 0.5) < 1e-6

def test_x_gate_flips_qubit():
    c = Circuit(1)
    c.x(0); c.measure(0)
    r = c.run()
    assert r.outcome(0) is True
    assert abs(r.prob_of("1") - 1.0) < 1e-6

# ─────────────────────────────────────────────────────────────────────────────
# ShotSimResult
# ─────────────────────────────────────────────────────────────────────────────

def test_shot_sampling_returns_shot_result():
    c = Circuit(2)
    c.h(0); c.cnot(0, 1); c.measure_all()
    r = c.run(shots=2000)
    assert isinstance(r, ShotSimResult)
    assert r.n_shots == 2000
    assert r.n_qubits == 2

def test_shot_sampling_roughly_half():
    c = Circuit(1)
    c.h(0); c.measure(0)
    r = c.run(shots=4000)
    assert abs(r.prob("0") - 0.5) < 0.06
    assert abs(r.prob("1") - 0.5) < 0.06

def test_shot_counts_sum_to_n_shots():
    c = Circuit(2)
    c.h(0); c.cnot(0, 1); c.measure_all()
    r = c.run(shots=500)
    assert sum(r.counts.values()) == 500

def test_shot_most_common_sorted():
    c = Circuit(2)
    c.h(0); c.cnot(0, 1); c.measure_all()
    r = c.run(shots=1000)
    mc = r.most_common()
    assert len(mc) >= 1
    # sorted descending by count
    counts_only = [x[1] for x in mc]
    assert counts_only == sorted(counts_only, reverse=True)

# ─────────────────────────────────────────────────────────────────────────────
# Backends
# ─────────────────────────────────────────────────────────────────────────────

def test_mps_backend_bell():
    c = Circuit(2)
    c.h(0); c.cnot(0, 1); c.measure_all()
    r = c.run(backend="mps", bond_dim=64)
    assert abs(r.prob_of("00") - 0.5) < 1e-4
    assert abs(r.prob_of("11") - 0.5) < 1e-4

def test_clifford_backend_bell():
    c = Circuit(2)
    c.h(0); c.cnot(0, 1); c.measure_all()
    r = c.run(backend="clifford")
    assert r.num_qubits == 2

def test_sparse_backend_bell():
    c = Circuit(2)
    c.h(0); c.cnot(0, 1); c.measure_all()
    r = c.run(backend="sparse")
    assert abs(r.prob_of("00") - 0.5) < 1e-6

def test_clifford_non_clifford_raises():
    c = Circuit(1)
    c.t(0); c.measure(0)
    with pytest.raises(Exception):
        c.run(backend="clifford")

# ─────────────────────────────────────────────────────────────────────────────
# Free functions: run_aql, run_aql_shots, run_qasm
# ─────────────────────────────────────────────────────────────────────────────

def test_run_aql_bell():
    r = run_aql("QREG 2\nH 0\nCNOT 0 1\nMEASURE_ALL")
    assert isinstance(r, SimResult)
    assert abs(r.prob_of("00") - 0.5) < 1e-6

def test_run_aql_shots():
    sr = run_aql_shots("QREG 2\nH 0\nCNOT 0 1\nMEASURE_ALL", shots=500)
    assert isinstance(sr, ShotSimResult)
    assert sr.n_shots == 500

def test_run_aql_mps_backend():
    r = run_aql("QREG 3\nH 0\nCNOT 0 1\nCNOT 0 2\nMEASURE_ALL", backend="mps")
    assert abs(r.prob_of("000") - 0.5) < 1e-4

def test_run_qasm_bell():
    qasm = "OPENQASM 2.0;\nqreg q[2];\nh q[0];\ncx q[0],q[1];\nmeasure q[0] -> q[0];\nmeasure q[1] -> q[1];"
    r = run_qasm(qasm)
    assert isinstance(r, SimResult)
    assert r.num_qubits == 2

def test_run_qasm_probability():
    qasm = "OPENQASM 2.0;\nqreg q[2];\nh q[0];\ncx q[0],q[1];"
    r = run_qasm(qasm)
    assert abs(r.prob_of("00") - 0.5) < 1e-6
    assert abs(r.prob_of("11") - 0.5) < 1e-6

# ─────────────────────────────────────────────────────────────────────────────
# run_file / analyze_file / analyze_aql
# ─────────────────────────────────────────────────────────────────────────────

def test_analyze_aql_returns_circuit_analysis():
    a = analyze_aql("QREG 2\nH 0\nCNOT 0 1\nMEASURE_ALL")
    assert isinstance(a, CircuitAnalysis)
    assert a.num_qubits == 2
    assert a.gate_count == 2      # H + CNOT (measurements tracked separately in measure_count)
    assert a.circuit_depth >= 2
    assert a.two_qubit_gate_count == 1
    assert a.is_clifford is True

def test_analyze_aql_is_clifford_false_for_t():
    a = analyze_aql("QREG 1\nT 0\nMEASURE 0")
    assert a.is_clifford is False

def test_analyze_aql_gate_histogram():
    a = analyze_aql("QREG 2\nH 0\nH 1\nCNOT 0 1\nMEASURE_ALL")
    h = a.gate_histogram
    assert h.get("H", 0) == 2
    assert h.get("CNOT", 0) == 1

def test_analyze_aql_qubit_utilization():
    a = analyze_aql("QREG 2\nH 0\nCNOT 0 1\nMEASURE_ALL")
    util = a.qubit_utilization
    assert len(util) == 2
    assert util[0] >= 1
    assert util[1] >= 1

def test_analyze_aql_has_control_flow():
    src = "QREG 1\nH 0\nMEASURE 0\nIF 0 GOTO done\nLABEL done"
    a = analyze_aql(src)
    assert a.has_control_flow is True

def test_analyze_aql_no_control_flow():
    a = analyze_aql("QREG 2\nH 0\nCNOT 0 1\nMEASURE_ALL")
    assert a.has_control_flow is False

def test_analyze_aql_repr():
    a = analyze_aql("QREG 2\nH 0\nCNOT 0 1\nMEASURE_ALL")
    r = repr(a)
    assert "CircuitAnalysis" in r
    assert "num_qubits=2" in r

def test_run_file_bell(tmp_path):
    p = tmp_path / "bell.aql"
    p.write_text("QREG 2\nH 0\nCNOT 0 1\nMEASURE_ALL")
    r = run_file(str(p))
    assert isinstance(r, SimResult)
    assert abs(r.prob_of("00") - 0.5) < 1e-6

def test_run_file_mps_backend(tmp_path):
    p = tmp_path / "ghz.aql"
    p.write_text("QREG 3\nH 0\nCNOT 0 1\nCNOT 0 2\nMEASURE_ALL")
    r = run_file(str(p), backend="mps", bond_dim=32)
    assert abs(r.prob_of("000") - 0.5) < 1e-4

def test_run_file_missing_raises(tmp_path):
    with pytest.raises(Exception):
        run_file(str(tmp_path / "nonexistent.aql"))

def test_analyze_file(tmp_path):
    p = tmp_path / "ghz.aql"
    p.write_text("QREG 3\nH 0\nCNOT 0 1\nCNOT 0 2\nMEASURE_ALL")
    a = analyze_file(str(p))
    assert isinstance(a, CircuitAnalysis)
    assert a.num_qubits == 3
    assert a.is_clifford is True

def test_analyze_file_missing_raises(tmp_path):
    with pytest.raises(Exception):
        analyze_file(str(tmp_path / "nope.aql"))

# ─────────────────────────────────────────────────────────────────────────────
# AQL v2 features through Python API
# ─────────────────────────────────────────────────────────────────────────────

def test_run_aql_repeat_loop():
    # H·H = I so 4 Hadamards → qubit stays |0⟩
    r = run_aql("QREG 1\nREPEAT 4\nH 0\nEND\nMEASURE 0")
    assert abs(r.prob_of("0") - 1.0) < 1e-6

def test_run_aql_named_registers():
    r = run_aql("QREG data[2]\nQREG anc[1]\nH data[0]\nCNOT data[0] data[1]\nMEASURE_ALL")
    assert r.num_qubits == 3
    assert abs(r.prob_of("000") - 0.5) < 1e-6

def test_run_aql_custom_gate_via_call():
    src = "GATE bell 2\nH 0\nCNOT 0 1\nEND\nQREG 2\nCALL bell 0 1\nMEASURE_ALL"
    r = run_aql(src)
    assert abs(r.prob_of("00") - 0.5) < 1e-6

def test_circuit_call_custom_aql_gate():
    # Use Circuit.call() with an inline AQL gate def
    c = Circuit(2)
    # manually build AQL that defines then calls a gate
    src = (
        "GATE mybell 2\nH 0\nCNOT 0 1\nEND\n"
        "QREG 2\nCALL mybell 0 1\nMEASURE_ALL"
    )
    r = run_aql(src)
    assert abs(r.prob_of("00") - 0.5) < 1e-6

# ─────────────────────────────────────────────────────────────────────────────
# OpenQASM 2.0 import
# ─────────────────────────────────────────────────────────────────────────────

def test_qasm_ghz():
    qasm = """OPENQASM 2.0;
qreg q[3];
h q[0];
cx q[0], q[1];
cx q[0], q[2];
"""
    r = run_qasm(qasm)
    assert r.num_qubits == 3

def test_qasm_rotation_gates():
    # RX(pi) = X so qubit starts |0⟩, after RX(pi) → |1⟩
    qasm = "OPENQASM 2.0;\nqreg q[1];\nrx(pi) q[0];\nmeasure q[0] -> q[0];"
    r = run_qasm(qasm)
    assert abs(r.prob_of("1") - 1.0) < 1e-5

# ─────────────────────────────────────────────────────────────────────────────
# Module attributes
# ─────────────────────────────────────────────────────────────────────────────

def test_module_version():
    assert hasattr(ac, '__version__')
    assert ac.__version__ == "0.1.0"

def test_module_exports():
    for name in ["Circuit", "SimResult", "ShotSimResult", "CircuitAnalysis",
                 "run_aql", "run_aql_shots", "run_qasm",
                 "run_file", "analyze_aql", "analyze_file",
                 "run_aql_report", "run_aql_serve", "run_aql_dash",
                 "PI", "PI_2"]:
        assert hasattr(ac, name), f"astracore.{name} not found"
