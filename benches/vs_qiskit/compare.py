#!/usr/bin/env python3
"""
AstraCore vs Qiskit Aer — Head-to-Head Timing Comparison
=========================================================

Measures wall-clock time for equivalent circuits on Qiskit Aer (statevector).
The output is a reference table to place next to AstraCore Criterion results.

Setup
-----
    pip install qiskit qiskit-aer

Run
---
    python benches/vs_qiskit/compare.py [--runs N]

Then run the AstraCore equivalent:
    cargo bench --bench statevector_scaling 2>&1 | tee benches/vs_qiskit/astracore_results.txt

See docs/benchmark_report.md for the full analysis template.
"""

import argparse
import statistics
import sys
import time

try:
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
except ImportError:
    print("ERROR: Qiskit not installed.")
    print("  pip install qiskit qiskit-aer")
    sys.exit(1)


# ── Configuration ─────────────────────────────────────────────────────────────

DEFAULT_RUNS = 50  # iterations per data point — increase for tighter std

H_SIZES    = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
GHZ_SIZES  = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
LAYER_SIZES = [4, 6, 8, 10, 12, 14, 16, 18, 20]


# ── Timing helper ─────────────────────────────────────────────────────────────

def time_fn(fn, n_runs: int):
    """Run fn() n_runs times; return (mean_µs, stddev_µs)."""
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)
    return statistics.mean(times), statistics.stdev(times) if n_runs > 1 else 0.0


# ── Benchmark: H gate on qubit 0 of n-qubit register ─────────────────────────

def bench_h_gate(n_runs: int) -> dict:
    """Allocate n-qubit statevector, apply H(0), retrieve result."""
    sim = AerSimulator(method="statevector")
    results = {}
    for n in H_SIZES:
        def run(n=n):
            qc = QuantumCircuit(n)
            qc.h(0)
            qc.save_statevector()
            sim.run(qc, shots=1).result()

        mean, std = time_fn(run, n_runs)
        results[n] = (mean, std)
        print(f"    H({n:>2}q): {mean:>8.1f} µs ± {std:>6.1f}", flush=True)
    return results


# ── Benchmark: GHZ / Bell state preparation ──────────────────────────────────

def bench_ghz(n_runs: int) -> dict:
    """H(0) + CNOT(0, q) for q in 1..n, then measure_all."""
    sim = AerSimulator(method="statevector")
    results = {}
    for n in GHZ_SIZES:
        def run(n=n):
            qc = QuantumCircuit(n, n)
            qc.h(0)
            for q in range(1, n):
                qc.cx(0, q)
            qc.measure_all()
            sim.run(qc, shots=1).result()

        mean, std = time_fn(run, n_runs)
        results[n] = (mean, std)
        print(f"    GHZ({n:>2}q): {mean:>8.1f} µs ± {std:>6.1f}", flush=True)
    return results


# ── Benchmark: Random circuit layer ─────────────────────────────────────────

def bench_random_layer(n_runs: int) -> dict:
    """H-all → CNOT-linear-chain → H-all (3 layers)."""
    sim = AerSimulator(method="statevector")
    results = {}
    for n in LAYER_SIZES:
        def run(n=n):
            qc = QuantumCircuit(n)
            for q in range(n):
                qc.h(q)
            for q in range(n - 1):
                qc.cx(q, q + 1)
            for q in range(n):
                qc.h(q)
            qc.save_statevector()
            sim.run(qc, shots=1).result()

        mean, std = time_fn(run, n_runs)
        results[n] = (mean, std)
        print(f"    Layer({n:>2}q): {mean:>8.1f} µs ± {std:>6.1f}", flush=True)
    return results


# ── Output helpers ────────────────────────────────────────────────────────────

HEADER = "  {qubits:>6}  {mean:>12}  {std:>12}"
ROW    = "  {qubits:>6}  {mean:>12.1f}  {std:>12.1f}"

def print_table(name: str, results: dict) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {name}")
    print(f"{'─' * 60}")
    print(HEADER.format(qubits="Qubits", mean="Mean (µs)", std="Std (µs)"))
    print(f"  {'─'*6}  {'─'*12}  {'─'*12}")
    for n in sorted(results):
        mean, std = results[n]
        print(ROW.format(qubits=n, mean=mean, std=std))


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="AstraCore vs Qiskit Aer — reference timing script"
    )
    parser.add_argument(
        "--runs", type=int, default=DEFAULT_RUNS,
        help=f"Iterations per data point (default: {DEFAULT_RUNS})"
    )
    args = parser.parse_args()
    n_runs = args.runs

    print("AstraCore vs Qiskit Aer — Reference Timings")
    print("=" * 60)
    print(f"Backend    : Qiskit AerSimulator (statevector)")
    print(f"Iterations : {n_runs} per data point")
    print()

    print("H gate on qubit 0 (n-qubit register):")
    h_results = bench_h_gate(n_runs)
    print_table("H gate — qubit 0 on n-qubit register", h_results)

    print("\nGHZ state preparation (H + CNOT chain + measure_all):")
    ghz_results = bench_ghz(n_runs)
    print_table("GHZ state (H + CNOT chain + measure_all)", ghz_results)

    print("\nRandom layer (H-all → CNOT-chain → H-all):")
    layer_results = bench_random_layer(n_runs)
    print_table("Random layer circuit (3 layers)", layer_results)

    print()
    print("=" * 60)
    print("Done. Compare with AstraCore Criterion results:")
    print("  cargo bench --bench statevector_scaling")
    print("  cargo bench --bench gate_benchmark")
    print()
    print("Note: Criterion reports nanoseconds; multiply by 1000 for µs.")
    print("=" * 60)


if __name__ == "__main__":
    main()
