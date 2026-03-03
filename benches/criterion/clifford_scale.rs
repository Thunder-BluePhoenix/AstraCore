/// AstraCore v2 — Clifford Simulator Scaling Benchmarks (Phase 3)
///
/// Measures CHP (Aaronson-Gottesman) stabilizer simulation scaling.
/// Memory: O(n²) bits.  Gate cost: O(n) per Clifford gate.
///
/// Benchmark groups:
///   clifford_h_nqubits      — H gate on n qubits (tableau update cost)
///   clifford_ghz_nqubits    — GHZ circuit (H + n CNOT): entanglement spread
///   clifford_cnot_chain     — full CNOT chain (worst-case tableau updates)
///   clifford_vs_sv_nqubits  — Clifford vs statevector for small circuits
use astracore::simulator::execute_clifford;
use astracore::compiler;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

/// Clifford can scale arbitrarily; benchmark up to 1000 qubits.
const CLIFFORD_H_SIZES: &[usize] = &[10, 50, 100, 200, 500, 1000];
const CLIFFORD_GHZ_SIZES: &[usize] = &[10, 50, 100, 200, 500];
const CLIFFORD_CHAIN_SIZES: &[usize] = &[10, 50, 100, 200, 500];

/// Overlap region: statevector is feasible; compare with Clifford.
const CROSSOVER_SIZES: &[usize] = &[4, 8, 12, 16, 20, 24];

// ── H sweep: n-qubit |0…0⟩, apply H on every qubit ──────────────────────

fn bench_clifford_h_sweep(c: &mut Criterion) {
    let mut group = c.benchmark_group("clifford_h_nqubits");
    group.sample_size(20);
    for &n in CLIFFORD_H_SIZES {
        let src = build_h_all_aql(n);
        group.bench_with_input(BenchmarkId::new("H_all", n), &n, |b, _| {
            b.iter(|| {
                let prog = compiler::parse_source(black_box(&src)).unwrap();
                execute_clifford(&prog).unwrap();
            });
        });
    }
    group.finish();
}

// ── GHZ sweep: H(0) + CNOT chain ─────────────────────────────────────────

fn bench_clifford_ghz_sweep(c: &mut Criterion) {
    let mut group = c.benchmark_group("clifford_ghz_nqubits");
    group.sample_size(20);
    for &n in CLIFFORD_GHZ_SIZES {
        let src = build_ghz_aql(n);
        group.bench_with_input(BenchmarkId::new("GHZ", n), &n, |b, _| {
            b.iter(|| {
                let prog = compiler::parse_source(black_box(&src)).unwrap();
                execute_clifford(&prog).unwrap();
            });
        });
    }
    group.finish();
}

// ── CNOT chain: worst-case O(n) per gate → O(n²) total ───────────────────

fn bench_clifford_cnot_chain(c: &mut Criterion) {
    let mut group = c.benchmark_group("clifford_cnot_chain");
    group.sample_size(20);
    for &n in CLIFFORD_CHAIN_SIZES {
        let src = build_cnot_chain_aql(n);
        group.bench_with_input(BenchmarkId::new("Chain", n), &n, |b, _| {
            b.iter(|| {
                let prog = compiler::parse_source(black_box(&src)).unwrap();
                execute_clifford(&prog).unwrap();
            });
        });
    }
    group.finish();
}

// ── Crossover: Clifford vs statevector for 4–24 qubits ───────────────────

fn bench_clifford_vs_sv(c: &mut Criterion) {
    use astracore::core::Simulator;

    let mut group = c.benchmark_group("clifford_vs_sv_nqubits");
    group.sample_size(30);
    for &n in CROSSOVER_SIZES {
        let src = build_ghz_aql(n);

        group.bench_with_input(BenchmarkId::new("Clifford", n), &n, |b, _| {
            b.iter(|| {
                let prog = compiler::parse_source(black_box(&src)).unwrap();
                execute_clifford(&prog).unwrap();
            });
        });

        group.bench_with_input(BenchmarkId::new("Statevector", n), &n, |b, &n| {
            b.iter(|| {
                let mut sim = Simulator::new(black_box(n));
                sim.h(0);
                for q in 1..n {
                    sim.cnot(0, q);
                }
            });
        });
    }
    group.finish();
}

// ── AQL source builders ───────────────────────────────────────────────────

fn build_h_all_aql(n: usize) -> String {
    let mut s = format!("QREG {n}\n");
    for q in 0..n { s.push_str(&format!("H {q}\n")); }
    s
}

fn build_ghz_aql(n: usize) -> String {
    let mut s = format!("QREG {n}\nH 0\n");
    for q in 1..n { s.push_str(&format!("CNOT 0 {q}\n")); }
    s
}

fn build_cnot_chain_aql(n: usize) -> String {
    let mut s = format!("QREG {n}\nH 0\n");
    // CNOT(0,1), CNOT(1,2), ..., CNOT(n-2, n-1)
    for q in 0..n - 1 { s.push_str(&format!("CNOT {q} {}\n", q + 1)); }
    s
}

criterion_group!(
    clifford_benches,
    bench_clifford_h_sweep,
    bench_clifford_ghz_sweep,
    bench_clifford_cnot_chain,
    bench_clifford_vs_sv,
);
criterion_main!(clifford_benches);
