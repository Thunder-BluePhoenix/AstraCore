/// AstraCore Criterion Benchmark Suite — Phase 3
///
/// Covers:
///   - Single-qubit gate throughput (H repeated on n-qubit register)
///   - Two-qubit gate throughput (CNOT)
///   - Entangled state preparation (Bell, GHZ at various sizes)
///   - Full measurement + collapse
///   - AQL compiler pipeline (lex → parse → execute)
///   - Hybrid runtime with control flow (GOTO, IF)
use astracore::compiler;
use astracore::core::Simulator;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

// ── Single-qubit gate throughput ──────────────────────────────────────────

fn bench_hadamard_10qubits(c: &mut Criterion) {
    c.bench_function("h_gate_10qubits_x10", |b| {
        b.iter(|| {
            let mut sim = Simulator::new(black_box(10));
            for _ in 0..10 {
                sim.h(black_box(0));
            }
        })
    });
}

fn bench_single_qubit_gates(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_qubit_gates");
    for n in [4usize, 8, 12, 16] {
        group.bench_with_input(BenchmarkId::new("H", n), &n, |b, &n| {
            b.iter(|| {
                let mut sim = Simulator::new(n);
                for q in 0..n {
                    sim.h(black_box(q));
                }
            });
        });
        group.bench_with_input(BenchmarkId::new("X", n), &n, |b, &n| {
            b.iter(|| {
                let mut sim = Simulator::new(n);
                for q in 0..n {
                    sim.x(black_box(q));
                }
            });
        });
    }
    group.finish();
}

// ── Two-qubit gate throughput ─────────────────────────────────────────────

fn bench_cnot_chain(c: &mut Criterion) {
    let mut group = c.benchmark_group("cnot_chain");
    for n in [4usize, 8, 12, 16] {
        group.bench_with_input(BenchmarkId::new("CNOT", n), &n, |b, &n| {
            b.iter(|| {
                let mut sim = Simulator::new(n);
                sim.h(0);
                for q in 1..n {
                    sim.cnot(black_box(0), black_box(q));
                }
            });
        });
    }
    group.finish();
}

// ── Entangled state preparation ───────────────────────────────────────────

fn bench_bell_state(c: &mut Criterion) {
    c.bench_function("bell_state_2qubits", |b| {
        b.iter(|| {
            let mut sim = Simulator::new(black_box(2));
            sim.h(0).cnot(0, 1);
        })
    });
}

fn bench_ghz_state(c: &mut Criterion) {
    let mut group = c.benchmark_group("ghz_state");
    for n in [4usize, 8, 12, 16, 20] {
        group.bench_with_input(BenchmarkId::new("GHZ", n), &n, |b, &n| {
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

// ── Measurement + collapse ────────────────────────────────────────────────

fn bench_measure_all(c: &mut Criterion) {
    let mut group = c.benchmark_group("measure_all");
    for n in [4usize, 8, 12, 16] {
        group.bench_with_input(BenchmarkId::new("n", n), &n, |b, &n| {
            b.iter(|| {
                let mut sim = Simulator::new(n);
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

// ── AQL compiler pipeline ─────────────────────────────────────────────────

fn bench_aql_bell(c: &mut Criterion) {
    let src = "QREG 2\nH 0\nCNOT 0 1\nMEASURE_ALL";
    c.bench_function("aql_bell_pipeline", |b| {
        b.iter(|| compiler::run(black_box(src)).unwrap())
    });
}

fn bench_aql_ghz(c: &mut Criterion) {
    let src = "QREG 10\nH 0\nCNOT 0 1\nCNOT 0 2\nCNOT 0 3\nCNOT 0 4\n\
               CNOT 0 5\nCNOT 0 6\nCNOT 0 7\nCNOT 0 8\nCNOT 0 9\nMEASURE_ALL";
    c.bench_function("aql_ghz_10qubits_pipeline", |b| {
        b.iter(|| compiler::run(black_box(src)).unwrap())
    });
}

fn bench_aql_grover_2qubits(c: &mut Criterion) {
    let src = "\
QREG 2
H 0
H 1
CZ 0 1
H 0
H 1
X 0
X 1
CZ 0 1
X 0
X 1
H 0
H 1
MEASURE_ALL";
    c.bench_function("aql_grover_2qubits_pipeline", |b| {
        b.iter(|| compiler::run(black_box(src)).unwrap())
    });
}

// ── Hybrid runtime (control flow) ─────────────────────────────────────────

fn bench_aql_teleportation(c: &mut Criterion) {
    let src = "\
QREG 3
H 0
H 1
CNOT 1 2
CNOT 0 1
H 0
MEASURE 0
MEASURE 1
IF 1 GOTO apply_x
GOTO skip_x
LABEL apply_x
X 2
LABEL skip_x
IF 0 GOTO apply_z
GOTO done
LABEL apply_z
Z 2
LABEL done
MEASURE 2";
    c.bench_function("aql_teleportation_hybrid", |b| {
        b.iter(|| compiler::run(black_box(src)).unwrap())
    });
}

// ── Rotation gate throughput ──────────────────────────────────────────────

fn bench_rotation_gates(c: &mut Criterion) {
    use std::f64::consts::PI;
    let mut group = c.benchmark_group("rotation_gates");
    for n in [4usize, 8, 12] {
        group.bench_with_input(BenchmarkId::new("Rx", n), &n, |b, &n| {
            b.iter(|| {
                let mut sim = Simulator::new(n);
                for q in 0..n {
                    sim.rx(black_box(q), black_box(PI / 4.0));
                }
            });
        });
        group.bench_with_input(BenchmarkId::new("Rz", n), &n, |b, &n| {
            b.iter(|| {
                let mut sim = Simulator::new(n);
                for q in 0..n {
                    sim.rz(black_box(q), black_box(PI / 8.0));
                }
            });
        });
    }
    group.finish();
}

// ── Groups ────────────────────────────────────────────────────────────────

criterion_group!(
    gate_benches,
    bench_hadamard_10qubits,
    bench_single_qubit_gates,
    bench_cnot_chain,
    bench_rotation_gates,
);
criterion_group!(
    circuit_benches,
    bench_bell_state,
    bench_ghz_state,
    bench_measure_all,
);
criterion_group!(
    pipeline_benches,
    bench_aql_bell,
    bench_aql_ghz,
    bench_aql_grover_2qubits,
    bench_aql_teleportation,
);

criterion_main!(gate_benches, circuit_benches, pipeline_benches);
