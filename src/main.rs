use astracore::compiler;
use astracore::core::{NoiseChannel, Simulator, SimdCapabilities};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    print_banner();

    match args.get(1).map(String::as_str) {
        None | Some("demo")           => run_all_demos(),
        Some("run")                   => cli_run(args.get(2).map(String::as_str)),
        Some("opt")                   => cli_optimize(args.get(2).map(String::as_str)),
        Some("analyze") | Some("stats") => cli_analyze(args.get(2).map(String::as_str)),
        Some("help") | Some("--help") => print_help(),
        Some(unknown) => {
            eprintln!("Unknown command '{}'. Run 'astracore help' for usage.", unknown);
            std::process::exit(1);
        }
    }
}

// ── CLI ───────────────────────────────────────────────────────────────────

fn cli_run(path: Option<&str>) {
    let path = match path {
        Some(p) => p,
        None => {
            eprintln!("Usage: astracore run <file.aql>");
            std::process::exit(1);
        }
    };

    let source = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => { eprintln!("Cannot read '{}': {}", path, e); std::process::exit(1); }
    };

    println!("━━━ AstraCore AQL Runner ━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("File: {path}\n");

    let program = match compiler::parse_source(&source) {
        Ok(p) => p,
        Err(e) => { eprintln!("{e}"); std::process::exit(1); }
    };

    println!("Program IR:");
    println!("  QREG {}", program.num_qubits);
    for instr in &program.instructions {
        println!("  {instr}");
    }
    println!();
    println!(
        "Circuit: {} gate(s) | {} measurement(s) | {} qubit(s)\n",
        program.gate_count, program.measure_count, program.num_qubits
    );

    let result = match compiler::execute(&program) {
        Ok(r) => r,
        Err(e) => { eprintln!("Runtime error: {e}"); std::process::exit(1); }
    };

    let display_probs = result.pre_measurement_probs.as_deref()
        .unwrap_or(&result.final_probabilities);
    let label = if result.pre_measurement_probs.is_some() {
        "Pre-measurement state"
    } else {
        "Final state (no measurements)"
    };

    println!("{label}:");
    for (lbl, prob) in result.significant_states(display_probs, 1e-6) {
        println!("  |{lbl}⟩  {prob:.6}");
    }
    println!();

    if !result.measurements.is_empty() {
        println!("Measurement results:");
        for m in &result.measurements {
            println!("  q{}  →  {}", m.qubit, m.outcome as u8);
        }
        if let Some(bs) = result.bitstring() {
            println!("  Bitstring (q0…qN): {bs}");
        }
    }
}

fn print_banner() {
    println!("╔══════════════════════════════════════════════╗");
    println!("║          AstraCore v0.1.0                    ║");
    println!("║  High-Performance Quantum Simulation Engine  ║");
    println!("╚══════════════════════════════════════════════╝");
    println!();
}

fn cli_optimize(path: Option<&str>) {
    let path = match path {
        Some(p) => p,
        None => { eprintln!("Usage: astracore opt <file.aql>"); std::process::exit(1); }
    };
    let source = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => { eprintln!("Cannot read '{}': {}", path, e); std::process::exit(1); }
    };
    println!("━━━ AstraCore Optimizer ━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("File: {path}\n");
    match compiler::optimize(&source) {
        Ok((prog, stats)) => {
            println!("Optimized IR:");
            println!("  QREG {}", prog.num_qubits);
            for instr in &prog.instructions { println!("  {instr}"); }
            println!();
            println!("Gates before : {}", stats.gates_before);
            println!("Gates after  : {}  (-{:.1}%)", stats.gates_after, stats.reduction_percent());
            println!("Passes       : {}", stats.passes);
        }
        Err(e) => { eprintln!("{e}"); std::process::exit(1); }
    }
}

fn cli_analyze(path: Option<&str>) {
    let path = match path {
        Some(p) => p,
        None => { eprintln!("Usage: astracore analyze <file.aql>"); std::process::exit(1); }
    };
    let source = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => { eprintln!("Cannot read '{}': {}", path, e); std::process::exit(1); }
    };
    println!("━━━ AstraCore Circuit Analyzer ━━━━━━━━━━━━━━━━━━━");
    println!("File: {path}\n");
    match compiler::analyze_source(&source) {
        Ok(a) => print!("{}", a.report()),
        Err(e) => { eprintln!("{e}"); std::process::exit(1); }
    }
}

fn print_help() {
    println!("Usage: astracore [COMMAND] [ARGS]\n");
    println!("Commands:");
    println!("  demo                Run built-in demonstration circuits");
    println!("  run <file.aql>      Parse and execute an AQL program");
    println!("  opt <file.aql>      Optimize and display the circuit");
    println!("  analyze <file.aql>  Static circuit analysis and profiling");
    println!("  help                Show this message\n");
    println!("AQL Instructions:");
    println!("  QREG <n>              Declare n qubits (must be first)");
    println!("  H|X|Y|Z|S|T <q>       Single-qubit gates");
    println!("  RX|RY|RZ <q> <θ>      Rotation gates (radians)");
    println!("  PHASE <q> <θ>         Phase gate");
    println!("  CNOT|CZ|SWAP <c> <t>  Two-qubit gates");
    println!("  CCX <c0> <c1> <t>     Toffoli (CCNOT) gate");
    println!("  MEASURE <q>           Measure qubit q");
    println!("  MEASURE_ALL           Measure all qubits");
    println!("  BARRIER               Visual separator (no-op)\n");
    println!("Constants: PI, TAU, PI_2, PI_4, PI_8, -PI, -PI_2, -PI_4");
    println!("Comments:  // or #");
}

// ── Demos (Rust API) ──────────────────────────────────────────────────────

fn run_all_demos() {
    demo_single_qubit();
    demo_bell_state();
    demo_ghz_state();
    demo_teleportation();
    demo_deutsch();
    demo_aql_pipeline();
    demo_optimizer();
    demo_noise_model();
    demo_simd_layer();
    demo_circuit_analysis();
}

fn demo_single_qubit() {
    println!("━━━ Demo 1: Single Qubit Basics ━━━━━━━━━━━━━━━━━");
    let mut sim = Simulator::new(1);
    println!("Initial |0⟩:");
    sim.print_state();
    sim.h(0);
    println!("After H (superposition):");
    sim.print_state();
    sim.z(0).h(0);
    println!("After H·Z·H = X (should be |1⟩):");
    sim.print_state();
    println!();
}

fn demo_bell_state() {
    println!("━━━ Demo 2: Bell State |Φ+⟩ ━━━━━━━━━━━━━━━━━━━━━");
    let mut sim = Simulator::new(2);
    sim.h(0).cnot(0, 1);
    sim.print_state();

    let mut counts = [0u32; 4];
    for _ in 0..1000 {
        let mut s = Simulator::new(2);
        s.h(0).cnot(0, 1);
        let r = s.measure_all();
        counts[(r[1] as usize) << 1 | r[0] as usize] += 1;
    }
    println!("Sampling 1000 shots:  |00⟩={} |11⟩={}", counts[0], counts[3]);
    println!();
}

fn demo_ghz_state() {
    println!("━━━ Demo 3: GHZ State (3 qubits) ━━━━━━━━━━━━━━━━");
    let mut sim = Simulator::new(3);
    sim.h(0).cnot(0, 1).cnot(0, 2);
    sim.print_state();

    let (mut c000, mut c111, mut other) = (0u32, 0u32, 0u32);
    for _ in 0..1000 {
        let mut s = Simulator::new(3);
        s.h(0).cnot(0, 1).cnot(0, 2);
        let r = s.measure_all();
        match (r[0], r[1], r[2]) {
            (false, false, false) => c000 += 1,
            (true,  true,  true)  => c111 += 1,
            _                     => other += 1,
        }
    }
    println!("Sampling 1000 shots:  |000⟩={c000}  |111⟩={c111}  other={other}");
    println!();
}

fn demo_teleportation() {
    println!("━━━ Demo 4: Quantum Teleportation ━━━━━━━━━━━━━━━");
    println!("Teleporting |+⟩ from q0 to q2.  [msg|alice|bob]");
    let mut sim = Simulator::new(3);
    sim.h(0);
    sim.h(1).cnot(1, 2);
    sim.cnot(0, 1).h(0);
    let m0 = sim.measure(0);
    let m1 = sim.measure(1);
    if m1 { sim.x(2); }
    if m0 { sim.z(2); }
    let p1 = sim.qubit_probability_one(2);
    println!("Bob P(|1⟩) = {p1:.4}  →  {}", if (p1 - 0.5).abs() < 0.01 { "SUCCESS" } else { "FAILED" });
    println!();
}

fn demo_deutsch() {
    println!("━━━ Demo 5: Deutsch Algorithm ━━━━━━━━━━━━━━━━━━━");
    println!("One oracle query reveals constant vs balanced.\n");

    let run = |oracle: fn(&mut Simulator)| {
        let mut sim = Simulator::new(2);
        sim.x(1).h(0).h(1);
        oracle(&mut sim);
        sim.h(0);
        sim.measure(0)
    };
    println!("  f(x)=0 (constant): q0={} (expect 0)", run(|_| {}) as u8);
    println!("  f(x)=x (balanced): q0={} (expect 1)", run(|s| { s.cnot(0, 1); }) as u8);
    println!();
}

fn demo_optimizer() {
    println!("━━━ Demo 7: Gate Optimizer (Phase 5) ━━━━━━━━━━━━━");

    let redundant = "\
// Intentionally redundant circuit
QREG 2
H 0          // ─┐ H·H cancels → identity
H 0          // ─┘
H 1          // kept (no pair)
RZ 0 0.7854  // Rz(π/8) ─┐ merge → Rz(π/4)
RZ 0 0.7854  //           ─┘
X 1          // ─┐ X·X cancels
X 1          // ─┘
CNOT 0 1
MEASURE_ALL
";

    println!("Source (redundant):");
    for line in redundant.lines() {
        let t = line.trim();
        if !t.is_empty() && !t.starts_with("//") {
            println!("  {t}");
        }
    }
    println!();

    match compiler::optimize(redundant) {
        Ok((opt_prog, stats)) => {
            println!("After peephole optimization:");
            for instr in &opt_prog.instructions {
                println!("  {instr}");
            }
            println!();
            println!(
                "  Gates before : {}",  stats.gates_before
            );
            println!(
                "  Gates after  : {}  (-{:.0}%)",
                stats.gates_after,
                stats.reduction_percent()
            );
            println!("  Passes       : {}", stats.passes);
        }
        Err(e) => eprintln!("Error: {e}"),
    }
    println!();
}

fn demo_noise_model() {
    println!("━━━ Demo 8: Noise Simulation (Phase 5) ━━━━━━━━━━━");

    // 1. Ideal Bell state — perfect correlation
    let ideal_agree = {
        let mut agree = 0u32;
        for _ in 0..500 {
            let mut sim = Simulator::new(2);
            sim.h(0).cnot(0, 1);
            let r = sim.measure_all();
            if r[0] == r[1] { agree += 1; }
        }
        agree
    };
    println!("Ideal Bell state (500 shots):");
    println!("  q0==q1  : {}/500  ({:.0}%)", ideal_agree, ideal_agree as f64 / 5.0);

    // 2. Noisy Bell state — depolarizing p=0.10
    let noisy_agree = {
        let mut agree = 0u32;
        for _ in 0..500 {
            let mut sim = Simulator::with_noise(2, NoiseChannel::Depolarizing { prob: 0.10 });
            sim.h(0).cnot(0, 1);
            let r = sim.measure_all();
            if r[0] == r[1] { agree += 1; }
        }
        agree
    };
    println!("Noisy Bell (depolarizing p=0.10, 500 shots):");
    println!("  q0==q1  : {}/500  ({:.0}%)", noisy_agree, noisy_agree as f64 / 5.0);

    // 3. Amplitude damping — |1⟩ decays toward |0⟩
    let decayed = {
        let mut ones = 0u32;
        for _ in 0..500 {
            let mut sim = Simulator::with_noise(1, NoiseChannel::AmplitudeDamping { gamma: 0.5 });
            sim.x(0); // prepare |1⟩, then noise immediately decays it
            let r = sim.measure(0);
            if r { ones += 1; }
        }
        ones
    };
    println!("Amplitude damping (γ=0.5) on X|0⟩=|1⟩ (500 shots):");
    println!("  P(|1⟩)  ≈ {:.2}  (expect ≈0.50)", decayed as f64 / 500.0);
    println!();
}

fn demo_simd_layer() {
    println!("━━━ Demo 9: Layer 2 — SIMD Optimization (Phase 3) ━━");

    let caps = SimdCapabilities::detect();
    println!("CPU SIMD capabilities:");
    println!("  SSE2    : {}", if caps.sse2    { "✓" } else { "✗" });
    println!("  AVX2    : {}", if caps.avx2    { "✓" } else { "✗" });
    println!("  AVX-512F: {}", if caps.avx512f { "✓" } else { "✗" });
    println!("  Active  : {}", caps.feature_string());
    println!("  Backend : {}", caps.best_backend());
    println!();

    // Demonstrate correctness: run an H·X·H = Z circuit on 4 qubits
    // with the SIMD path active on qubit 0.
    use astracore::core::gates::{apply_single_qubit_gate, hadamard, pauli_x};
    use astracore::core::StateVector;

    let n = 4;
    let mut sv = StateVector::new(n);
    // H on q0 — this triggers AVX2 path when available
    apply_single_qubit_gate(&mut sv, &hadamard(), 0);
    // X on q0 — AVX2 path
    apply_single_qubit_gate(&mut sv, &pauli_x(), 0);
    // H on q0 — AVX2 path (H·X·H = Z in terms of probabilities on |0⟩)
    apply_single_qubit_gate(&mut sv, &hadamard(), 0);

    let p0 = sv.amplitudes[0].norm_sq();
    let p1 = sv.amplitudes[1].norm_sq();
    println!("H·X·H |0...0⟩  →  |0...0⟩  (Z gate on qubit 0):");
    println!("  P(q0=0) = {p0:.6}  (expect ≈ 1.0)");
    println!("  P(q0=1) = {p1:.6}  (expect ≈ 0.0)");

    // Throughput comparison note
    if caps.avx2 {
        println!();
        println!("AVX2 active: qubit-0 gates process 2 complex amplitudes per cycle");
        println!("  (256-bit YMM register = 2 × Complex<f64> = one full amplitude pair)");
    } else {
        println!();
        println!("AVX2 not detected: using scalar fallback (results identical)");
    }
    println!();
}

fn demo_circuit_analysis() {
    println!("━━━ Demo 10: Circuit Analysis (Profiling) ━━━━━━━━━");

    // Three circuits of increasing complexity for comparison
    let circuits: &[(&str, &str)] = &[
        ("Bell pair", "QREG 2\nH 0\nCNOT 0 1\nMEASURE_ALL"),
        ("GHZ (5 qubits)",
         "QREG 5\nH 0\nCNOT 0 1\nCNOT 0 2\nCNOT 0 3\nCNOT 0 4\nMEASURE_ALL"),
        ("Custom-gate GHZ",
         "GATE ghz 3\n  H 0\n  CNOT 0 1\n  CNOT 0 2\nEND\n\
          QREG 3\nCALL ghz 0 1 2\nMEASURE_ALL"),
    ];

    for (name, src) in circuits {
        println!("  Circuit: {name}");
        match compiler::analyze_source(src) {
            Ok(a) => print!("{}", a.report()),
            Err(e) => eprintln!("  Error: {e}"),
        }
        println!();
    }
}

fn demo_aql_pipeline() {
    println!("━━━ Demo 6: AQL Compiler Pipeline ━━━━━━━━━━━━━━━");

    let aql = "\
// GHZ state via AQL
QREG 3
H 0
CNOT 0 1
CNOT 0 2
BARRIER       // visual separator
MEASURE_ALL
";

    println!("Source:");
    for line in aql.lines() {
        if !line.trim().is_empty() {
            println!("  {line}");
        }
    }
    println!();

    match compiler::run(aql) {
        Ok(result) => {
            let probs = result.pre_measurement_probs.as_ref().unwrap();
            println!("Pre-measurement state:");
            for (label, prob) in result.significant_states(probs, 1e-6) {
                println!("  |{label}⟩  {prob:.4}");
            }
            println!();
            println!("Measurements:");
            for m in &result.measurements {
                println!("  q{}  →  {}", m.qubit, m.outcome as u8);
            }
            if let Some(bs) = result.bitstring() {
                println!("  Bitstring: {bs}");
            }
        }
        Err(e) => eprintln!("AQL Error: {e}"),
    }
    println!();
}
