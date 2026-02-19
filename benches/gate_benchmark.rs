use astracore::core::Simulator;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_hadamard(c: &mut Criterion) {
    c.bench_function("hadamard_10qubits", |b| {
        b.iter(|| {
            let mut sim = Simulator::new(10);
            for _ in 0..10 {
                sim.h(black_box(0));
            }
        })
    });
}

fn bench_bell_state(c: &mut Criterion) {
    c.bench_function("bell_state_creation", |b| {
        b.iter(|| {
            let mut sim = Simulator::new(black_box(2));
            sim.h(0).cnot(0, 1);
        })
    });
}

fn bench_ghz_state_20qubits(c: &mut Criterion) {
    c.bench_function("ghz_state_20qubits", |b| {
        b.iter(|| {
            let mut sim = Simulator::new(black_box(20));
            sim.h(0);
            for q in 1..20 {
                sim.cnot(0, q);
            }
        })
    });
}

criterion_group!(benches, bench_hadamard, bench_bell_state, bench_ghz_state_20qubits);
criterion_main!(benches);
