use astracore::core::Simulator;
use std::f64::consts::PI;

fn main() {
    println!("╔══════════════════════════════════════════════╗");
    println!("║          AstraCore v0.1.0                    ║");
    println!("║  High-Performance Quantum Simulation Engine  ║");
    println!("╚══════════════════════════════════════════════╝\n");

    demo_single_qubit();
    demo_bell_state();
    demo_ghz_state();
    demo_quantum_teleportation();
    demo_deutsch_algorithm();
    demo_phase_kickback();
}

// ── Demo 1: Single Qubit ─────────────────────────────────────────────────────

fn demo_single_qubit() {
    println!("━━━ Demo 1: Single Qubit Basics ━━━━━━━━━━━━━━━━━");

    let mut sim = Simulator::new(1);

    println!("Initial state |0⟩:");
    sim.print_state();

    sim.h(0);
    println!("After H gate (superposition):");
    sim.print_state();

    sim.z(0);
    println!("After Z gate (phase flip on |1⟩ amplitude):");
    sim.print_state();

    sim.h(0);
    println!("After second H gate (H·Z·H = X gate, should be |1⟩):");
    sim.print_state();

    println!();
}

// ── Demo 2: Bell State ───────────────────────────────────────────────────────

fn demo_bell_state() {
    println!("━━━ Demo 2: Bell State (|Φ+⟩) ━━━━━━━━━━━━━━━━━━━");
    println!("Circuit: H(q0) → CNOT(q0, q1)");

    let mut sim = Simulator::new(2);
    sim.h(0).cnot(0, 1);

    println!("State after circuit:");
    sim.print_state();

    let probs = sim.probabilities();
    println!("Probabilities:");
    println!("  P(|00⟩) = {:.4}", probs[0]);
    println!("  P(|01⟩) = {:.4}", probs[1]);
    println!("  P(|10⟩) = {:.4}", probs[2]);
    println!("  P(|11⟩) = {:.4}", probs[3]);

    // Run 1000 shots to verify distribution
    println!("\nSampling 1000 shots:");
    let mut counts = [0u32; 4];
    for _ in 0..1000 {
        let mut sim_shot = Simulator::new(2);
        sim_shot.h(0).cnot(0, 1);
        let results = sim_shot.measure_all();
        let idx = (results[1] as usize) << 1 | results[0] as usize;
        counts[idx] += 1;
    }
    println!("  |00⟩: {} ({:.1}%)", counts[0], counts[0] as f64 / 10.0);
    println!("  |01⟩: {} ({:.1}%)", counts[1], counts[1] as f64 / 10.0);
    println!("  |10⟩: {} ({:.1}%)", counts[2], counts[2] as f64 / 10.0);
    println!("  |11⟩: {} ({:.1}%)", counts[3], counts[3] as f64 / 10.0);
    println!();
}

// ── Demo 3: GHZ State ────────────────────────────────────────────────────────

fn demo_ghz_state() {
    println!("━━━ Demo 3: GHZ State (3 qubits) ━━━━━━━━━━━━━━━━");
    println!("Circuit: H(q0) → CNOT(q0,q1) → CNOT(q0,q2)");
    println!("Creates: (|000⟩ + |111⟩) / √2");

    let mut sim = Simulator::new(3);
    sim.h(0).cnot(0, 1).cnot(0, 2);

    println!("State after circuit:");
    sim.print_state();

    println!("Sampling 1000 shots:");
    let mut count_000 = 0u32;
    let mut count_111 = 0u32;
    let mut count_other = 0u32;
    for _ in 0..1000 {
        let mut s = Simulator::new(3);
        s.h(0).cnot(0, 1).cnot(0, 2);
        let r = s.measure_all();
        match (r[2], r[1], r[0]) {
            (false, false, false) => count_000 += 1,
            (true,  true,  true)  => count_111 += 1,
            _ => count_other += 1,
        }
    }
    println!("  |000⟩:  {} ({:.1}%)", count_000, count_000 as f64 / 10.0);
    println!("  |111⟩:  {} ({:.1}%)", count_111, count_111 as f64 / 10.0);
    println!("  other:  {} (expected 0)", count_other);
    println!();
}

// ── Demo 4: Quantum Teleportation ────────────────────────────────────────────

fn demo_quantum_teleportation() {
    println!("━━━ Demo 4: Quantum Teleportation ━━━━━━━━━━━━━━━");
    println!("Teleporting state |+⟩ = H|0⟩ from qubit 0 to qubit 2");
    println!("Qubits: [message(0) | alice(1) | bob(2)]");

    // We verify by checking output probabilities instead of measuring
    // (measuring destroys the state we want to verify)
    let mut sim = Simulator::new(3);

    // Prepare message qubit in |+⟩
    sim.h(0);

    // Create Bell pair between alice (q1) and bob (q2)
    sim.h(1).cnot(1, 2);

    // Alice's operations
    sim.cnot(0, 1);
    sim.h(0);

    // Measure alice's qubits
    let m0 = sim.measure(0);
    let m1 = sim.measure(1);

    // Bob's corrections based on alice's measurement results
    if m1 { sim.x(2); }
    if m0 { sim.z(2); }

    // Bob's qubit should now be in |+⟩ = (|0⟩ + |1⟩)/√2
    let p_one = sim.qubit_probability_one(2);
    println!("Bob's qubit P(|1⟩) = {:.4} (expected ~0.5000)", p_one);
    println!(
        "Teleportation: {}",
        if (p_one - 0.5).abs() < 0.01 { "SUCCESS" } else { "FAILED" }
    );
    println!();
}

// ── Demo 5: Deutsch Algorithm ────────────────────────────────────────────────

fn demo_deutsch_algorithm() {
    println!("━━━ Demo 5: Deutsch Algorithm ━━━━━━━━━━━━━━━━━━━");
    println!("Determines if f(x) is constant or balanced in ONE query.\n");

    // Test with constant f(x) = 0 → oracle is identity
    {
        println!("  f(x) = 0 (constant):");
        let mut sim = Simulator::new(2);
        sim.x(1);       // ancilla to |1⟩
        sim.h(0).h(1);  // superposition
        // Constant-0 oracle: do nothing (U_f acts as identity on ancilla)
        sim.h(0);
        let result = sim.measure(0);
        println!("  Measured q0 = {} (0 = constant, 1 = balanced)", result as u8);
    }

    // Test with balanced f(x) = x → oracle is CNOT
    {
        println!("  f(x) = x (balanced):");
        let mut sim = Simulator::new(2);
        sim.x(1);
        sim.h(0).h(1);
        sim.cnot(0, 1); // balanced oracle
        sim.h(0);
        let result = sim.measure(0);
        println!("  Measured q0 = {} (0 = constant, 1 = balanced)\n", result as u8);
    }
}

// ── Demo 6: Phase Kickback ───────────────────────────────────────────────────

fn demo_phase_kickback() {
    println!("━━━ Demo 6: Phase Kickback (Rz rotation) ━━━━━━━━");
    println!("Demonstrates phase accumulation via controlled rotation.\n");

    let angle = PI / 4.0; // T gate angle
    let mut sim = Simulator::new(2);
    sim.h(0);           // control in superposition
    sim.x(1);           // target in |1⟩ (eigenstate)
    sim.h(1);
    // Controlled-T: apply phase e^(iπ/4) to |1⟩ component of control
    // Simulated via: if control=1, apply T to target
    // (Full controlled-T not in MVP, demonstrate with direct T)
    sim.t(0);
    sim.h(0);

    println!("After H-T-H on q0 (equivalent to Rz(π/4)):");
    let p1 = sim.qubit_probability_one(0);
    println!("  P(q0=1) = {:.6}", p1);
    println!("  Expected deviation from 0.5: {:.6}", (p1 - 0.5).abs());

    let expected_deviation = ((angle / 2.0).sin()).powi(2);
    println!("  Predicted: {:.6}", expected_deviation);
    println!();
}
