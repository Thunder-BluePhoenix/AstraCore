/// AstraCore v2 — MPS Backend Scaling Benchmarks (Phase 2)
///
/// Measures how MPS simulation throughput scales with qubit count.
/// Unlike statevector (exponential memory), MPS memory scales as O(n × χ²).
///
/// Benchmark groups:
///   mps_h_gate_nqubits    — single H gate on n-qubit product state (χ=1 stays χ=1)
///   mps_ghz_nqubits       — GHZ state preparation (entanglement grows χ to 2)
///   mps_layer_nqubits     — H-all + CNOT-chain: forces χ up to bond_dim
///   mps_vs_sv_nqubits     — MPS (χ=32) vs statevector crossover point
use astracore::simulator::{execute_mps, MpsState};
use astracore::compiler;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

/// Qubit counts for MPS single-qubit sweep — can go well beyond statevector limits.
const MPS_H_SIZES: &[usize] = &[4, 8, 16, 32, 50, 64, 100];

/// GHZ state — bond dimension grows to 2 (fully entangled chain).
const MPS_GHZ_SIZES: &[usize] = &[4, 8, 16, 32, 50, 64, 100];

/// Layer circuit — H-all + CNOT-chain; χ saturates at bond_dim quickly.
const MPS_LAYER_SIZES: &[usize] = &[4, 8, 16, 32, 50];

/// Crossover comparison: same qubit counts available to statevector AND MPS.
const CROSSOVER_SIZES: &[usize] = &[4, 8, 12, 16, 20, 24];

// ── H gate sweep: n-qubit |0…0⟩, apply H on qubit 0 ─────────────────────

fn bench_mps_h_sweep(c: &mut Criterion) {
    let mut group = c.benchmark_group("mps_h_gate_nqubits");
    group.sample_size(20);
    for &n in MPS_H_SIZES {
        group.bench_with_input(BenchmarkId::new("H0", n), &n, |b, &n| {
            b.iter(|| {
                let mut state = MpsState::new(black_box(n), 32);
                // Apply H via 2×2 matrix (Hadamard) — Matrix2x2 = [[Complex;2];2]
                let h = 1.0 / 2.0_f64.sqrt();
                let c = |re: f64| astracore::core::Complex::new(re, 0.0);
                let mat = [[c(h), c(h)], [c(h), c(-h)]];
                state.apply_single_qubit(0, &mat);
            });
        });
    }
    group.finish();
}

// ── GHZ sweep: H(0) + CNOT(k-1, k) chain — bond dim grows to 2 ──────────

fn bench_mps_ghz_sweep(c: &mut Criterion) {
    let mut group = c.benchmark_group("mps_ghz_nqubits");
    group.sample_size(20);
    for &n in MPS_GHZ_SIZES {
        let src = build_ghz_aql(n);
        group.bench_with_input(BenchmarkId::new("GHZ", n), &n, |b, _| {
            b.iter(|| {
                let prog = compiler::parse_source(black_box(&src)).unwrap();
                execute_mps(&prog, 64).unwrap();
            });
        });
    }
    group.finish();
}

// ── Layer circuit: H-all → CNOT-chain → H-all ────────────────────────────

fn bench_mps_layer_sweep(c: &mut Criterion) {
    let mut group = c.benchmark_group("mps_layer_nqubits");
    group.sample_size(20);
    for &n in MPS_LAYER_SIZES {
        let src = build_layer_aql(n);
        group.bench_with_input(BenchmarkId::new("Layer", n), &n, |b, _| {
            b.iter(|| {
                let prog = compiler::parse_source(black_box(&src)).unwrap();
                execute_mps(&prog, 32).unwrap();
            });
        });
    }
    group.finish();
}

// ── Crossover: MPS(χ=32) vs statevector for 4–24 qubits ─────────────────

fn bench_mps_vs_sv(c: &mut Criterion) {
    use astracore::core::Simulator;

    let mut group = c.benchmark_group("mps_vs_sv_nqubits");
    group.sample_size(30);
    for &n in CROSSOVER_SIZES {
        let src = build_ghz_aql(n);

        group.bench_with_input(BenchmarkId::new("MPS_chi32", n), &n, |b, _| {
            b.iter(|| {
                let prog = compiler::parse_source(black_box(&src)).unwrap();
                execute_mps(&prog, 32).unwrap();
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

fn build_ghz_aql(n: usize) -> String {
    let mut s = format!("QREG {n}\nH 0\n");
    for q in 1..n {
        s.push_str(&format!("CNOT 0 {q}\n"));
    }
    s
}

fn build_layer_aql(n: usize) -> String {
    let mut s = format!("QREG {n}\n");
    // Layer 1: H on all
    for q in 0..n { s.push_str(&format!("H {q}\n")); }
    // Layer 2: CNOT chain
    for q in 0..n - 1 { s.push_str(&format!("CNOT {q} {}\n", q + 1)); }
    // Layer 3: H on all again
    for q in 0..n { s.push_str(&format!("H {q}\n")); }
    s
}

criterion_group!(
    mps_benches,
    bench_mps_h_sweep,
    bench_mps_ghz_sweep,
    bench_mps_layer_sweep,
    bench_mps_vs_sv,
);
criterion_main!(mps_benches);
