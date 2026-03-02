/// AstraCore v2 — Statevector Scaling Benchmarks (Phase 1)
///
/// Measures how statevector simulation throughput scales with qubit count.
/// Results feed directly into the competitive benchmark report (docs/benchmark_report.md).
///
/// Benchmark groups:
///   h_gate_nqubits       — allocate + H(0) for n = 1..24
///   ghz_nqubits          — GHZ preparation H + CNOT chain for n = 2..24
///   random_layer_nqubits — H-all → CNOT-chain → H-all for n = 4..24
///   measure_all_nqubits  — GHZ state + measure_all for n = 4..24
///   aql_pipeline_nqubits — end-to-end lex→parse→execute (AQL) for n = 2..20
use astracore::compiler;
use astracore::core::Simulator;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

/// All qubit counts for single-qubit gate sweep.
/// 2^24 × 16 bytes = 256 MB per state — acceptable for a benchmark machine.
const H_SIZES: &[usize] = &[1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24];
/// Two-qubit (GHZ) sweep — requires at least 2 qubits.
const GHZ_SIZES: &[usize] = &[2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24];
/// Layer circuit and measurement sweep — at least 4 qubits for chain.
const LAYER_SIZES: &[usize] = &[4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24];
/// AQL pipeline capped at 20q (compiler overhead matters more than state size here).
const AQL_SIZES: &[usize] = &[2, 4, 6, 8, 10, 12, 14, 16, 18, 20];

// ── H gate sweep: allocate n-qubit register, apply H on qubit 0 ─────────────

fn bench_h_sweep(c: &mut Criterion) {
    let mut group = c.benchmark_group("h_gate_nqubits");
    // Reduce sample count for large states to keep benchmark time reasonable.
    group.sample_size(20);
    for &n in H_SIZES {
        group.bench_with_input(BenchmarkId::new("H0", n), &n, |b, &n| {
            b.iter(|| {
                let mut sim = Simulator::new(black_box(n));
                sim.h(0);
            });
        });
    }
    group.finish();
}

// ── GHZ sweep: H(0) + CNOT(0, q) for q in 1..n ─────────────────────────────

fn bench_ghz_sweep(c: &mut Criterion) {
    let mut group = c.benchmark_group("ghz_nqubits");
    group.sample_size(20);
    for &n in GHZ_SIZES {
        group.bench_with_input(BenchmarkId::new("GHZ", n), &n, |b, &n| {
            b.iter(|| {
                let mut sim = Simulator::new(black_box(n));
                sim.h(0);
                for q in 1..n {
                    sim.cnot(black_box(0), black_box(q));
                }
            });
        });
    }
    group.finish();
}

// ── Random layer: H-all → CNOT-linear-chain → H-all ────────────────────────
//
// This pattern approximates one layer of a random quantum circuit and stresses
// both single-qubit (H) and two-qubit (CNOT) paths equally.

fn bench_random_layer(c: &mut Criterion) {
    let mut group = c.benchmark_group("random_layer_nqubits");
    group.sample_size(20);
    for &n in LAYER_SIZES {
        group.bench_with_input(BenchmarkId::new("HL_CNOT", n), &n, |b, &n| {
            b.iter(|| {
                let mut sim = Simulator::new(black_box(n));
                // Layer 1: H on every qubit
                for q in 0..n {
                    sim.h(black_box(q));
                }
                // Layer 2: CNOT linear chain
                for q in 0..n - 1 {
                    sim.cnot(black_box(q), black_box(q + 1));
                }
                // Layer 3: H on every qubit
                for q in 0..n {
                    sim.h(black_box(q));
                }
            });
        });
    }
    group.finish();
}

// ── Measure_all after GHZ preparation ───────────────────────────────────────

fn bench_measure_sweep(c: &mut Criterion) {
    let mut group = c.benchmark_group("measure_all_nqubits");
    group.sample_size(20);
    for &n in LAYER_SIZES {
        group.bench_with_input(BenchmarkId::new("measure_all", n), &n, |b, &n| {
            b.iter(|| {
                let mut sim = Simulator::new(black_box(n));
                sim.h(0);
                for q in 1..n {
                    sim.cnot(0, q);
                }
                sim.measure_all()
            });
        });
    }
    group.finish();
}

// ── AQL end-to-end pipeline: lex → parse → IR → execute ────────────────────

fn ghz_aql_source(n: usize) -> String {
    let mut s = format!("QREG {}\nH 0\n", n);
    for q in 1..n {
        s.push_str(&format!("CNOT 0 {}\n", q));
    }
    s.push_str("MEASURE_ALL");
    s
}

fn bench_aql_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("aql_pipeline_nqubits");
    group.sample_size(20);
    // Pre-generate all sources so string allocation is not benchmarked.
    let sources: Vec<(usize, String)> = AQL_SIZES
        .iter()
        .map(|&n| (n, ghz_aql_source(n)))
        .collect();
    for (n, src) in &sources {
        group.bench_with_input(BenchmarkId::new("GHZ_AQL", n), src, |b, src| {
            b.iter(|| compiler::run(black_box(src.as_str())).unwrap());
        });
    }
    group.finish();
}

// ── Rotation gate sweep ─────────────────────────────────────────────────────
//
// Rx and Rz use the full complex 2×2 rotation matrix path — important for
// showing that parameterised gates don't regress vs Pauli gates.

fn bench_rotation_sweep(c: &mut Criterion) {
    use std::f64::consts::PI;
    let mut group = c.benchmark_group("rotation_gate_nqubits");
    group.sample_size(20);
    for &n in &[4usize, 8, 12, 16, 20] {
        group.bench_with_input(BenchmarkId::new("Rx", n), &n, |b, &n| {
            b.iter(|| {
                let mut sim = Simulator::new(black_box(n));
                for q in 0..n {
                    sim.rx(black_box(q), black_box(PI / 4.0));
                }
            });
        });
        group.bench_with_input(BenchmarkId::new("Rz", n), &n, |b, &n| {
            b.iter(|| {
                let mut sim = Simulator::new(black_box(n));
                for q in 0..n {
                    sim.rz(black_box(q), black_box(PI / 8.0));
                }
            });
        });
    }
    group.finish();
}

// ── Groups ───────────────────────────────────────────────────────────────────

criterion_group!(
    statevector_scaling,
    bench_h_sweep,
    bench_ghz_sweep,
    bench_random_layer,
    bench_measure_sweep,
    bench_aql_pipeline,
    bench_rotation_sweep,
);

criterion_main!(statevector_scaling);
