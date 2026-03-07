use astracore::compiler;
use astracore::core::{NoiseChannel, Simulator, SimdCapabilities};
use astracore::dashboard::{self, DashboardData};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    print_banner();

    match args.get(1).map(String::as_str) {
        None | Some("demo")             => run_all_demos(),
        Some("run")                     => cli_run(&args[2..]),
        Some("opt")                     => cli_optimize(args.get(2).map(String::as_str)),
        Some("analyze") | Some("stats") => cli_analyze(args.get(2).map(String::as_str)),
        Some("dash")                    => cli_dash(args.get(2).map(String::as_str)),
        Some("report")                  => cli_report(args.get(2).map(String::as_str), args.get(3).map(String::as_str)),
        Some("serve")                   => cli_serve(args.get(2).map(String::as_str), args.get(3).map(String::as_str)),
        Some("export")                  => cli_export(args.get(2).map(String::as_str), args.get(3).map(String::as_str)),
        Some("import")                  => cli_import(args.get(2).map(String::as_str)),
        Some("import3")                 => cli_import3(args.get(2).map(String::as_str)),
        Some("devices")                 => cli_devices(),
        Some("worker")                  => cli_worker(&args[2..]),
        Some("lsp")                     => cli_lsp(),
        Some("dap")                     => cli_dap(),
        Some("help") | Some("--help")   => print_help(),
        Some(unknown) => {
            eprintln!("Unknown command '{}'. Run 'astracore help' for usage.", unknown);
            std::process::exit(1);
        }
    }
}

// ── CLI ───────────────────────────────────────────────────────────────────

/// Simulation backend selection.
#[derive(Clone, Copy, PartialEq)]
enum Backend { Statevector, Mps, Clifford, Sparse, Gpu, Wgpu, Cuda, Dist }

/// Parse `run` arguments: `<file.aql> [--backend sv|mps|clifford|sparse] [--bond-dim N] [--shots N]`
fn cli_run(args: &[String]) {
    if args.is_empty() {
        eprintln!("Usage: astracore run <file.aql> [--backend statevector|mps|clifford|sparse] [--bond-dim N] [--shots N]");
        std::process::exit(1);
    }

    let path = &args[0];
    let mut backend  = Backend::Statevector;
    let mut bond_dim: usize = 64;
    let mut n_shots: Option<usize> = None;
    let mut dist_nodes: Option<String> = None;
    let mut dist_cluster: Option<String> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--backend" | "-b" => {
                i += 1;
                backend = match args.get(i).map(String::as_str) {
                    Some("mps")         => Backend::Mps,
                    Some("clifford") | Some("stabilizer") => Backend::Clifford,
                    Some("sparse")      => Backend::Sparse,
                    Some("gpu")         => Backend::Gpu,
                    Some("wgpu")        => Backend::Wgpu,
                    Some("cuda")        => Backend::Cuda,
                    Some("dist") | Some("distributed") => Backend::Dist,
                    Some("statevector") | Some("sv") | Some("default") => Backend::Statevector,
                    other => {
                        eprintln!("Unknown backend '{}'  (choices: statevector, mps, clifford, sparse, gpu, wgpu, cuda, dist)",
                                  other.unwrap_or(""));
                        std::process::exit(1);
                    }
                };
            }
            "--bond-dim" | "-d" => {
                i += 1;
                bond_dim = args.get(i)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or_else(|| {
                        eprintln!("--bond-dim requires a positive integer");
                        std::process::exit(1);
                    });
            }
            "--shots" | "-s" => {
                i += 1;
                n_shots = Some(args.get(i)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or_else(|| {
                        eprintln!("--shots requires a positive integer");
                        std::process::exit(1);
                    }));
            }
            "--dist" => { backend = Backend::Dist; }
            "--nodes" | "-n" => {
                i += 1;
                dist_nodes = args.get(i).map(String::clone);
            }
            "--cluster" => {
                i += 1;
                dist_cluster = args.get(i).map(String::clone);
            }
            flag => { eprintln!("Unknown flag '{flag}'"); std::process::exit(1); }
        }
        i += 1;
    }

    let source = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => { eprintln!("Cannot read '{}': {e}", path); std::process::exit(1); }
    };

    let backend_name = match backend {
        Backend::Statevector => "statevector".to_string(),
        Backend::Mps         => format!("mps (bond-dim={bond_dim})"),
        Backend::Clifford    => "clifford (stabilizer)".to_string(),
        Backend::Sparse      => "sparse statevector".to_string(),
        Backend::Gpu         => "gpu (auto: wgpu or cuda)".to_string(),
        Backend::Wgpu        => "wgpu (WebGPU)".to_string(),
        Backend::Cuda        => "cuda (NVIDIA CUDA)".to_string(),
        Backend::Dist        => {
            let nodes_str = dist_nodes.as_deref().or(dist_cluster.as_deref()).unwrap_or("(no nodes)");
            format!("distributed ({nodes_str})")
        }
    };
    println!("━━━ AstraCore AQL Runner ━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("File   : {path}");
    println!("Backend: {backend_name}");
    if let Some(shots) = n_shots {
        println!("Shots  : {shots}");
    }
    println!();

    let program = match compiler::parse_source(&source) {
        Ok(p) => p,
        Err(e) => { eprintln!("{e}"); std::process::exit(1); }
    };

    println!("Circuit: {} gate(s) | {} measurement(s) | {} qubit(s)\n",
             program.gate_count, program.measure_count, program.num_qubits);

    // ── Shot-based sampling mode ──────────────────────────────────────────
    if let Some(shots) = n_shots {
        let shot_result = run_shots_with_backend(&program, shots, backend, bond_dim);
        println!("Shot sampling — {shots} runs:");
        println!();
        shot_result.print_histogram();
        return;
    }

    // ── Single-run mode ───────────────────────────────────────────────────
    let result = match backend {
        Backend::Statevector => match compiler::execute(&program) {
            Ok(r) => r,
            Err(e) => { eprintln!("Runtime error: {e}"); std::process::exit(1); }
        },
        Backend::Mps => match astracore::simulator::execute_mps(&program, bond_dim) {
            Ok(r) => r,
            Err(e) => { eprintln!("MPS runtime error: {e}"); std::process::exit(1); }
        },
        Backend::Clifford => match astracore::simulator::execute_clifford(&program) {
            Ok(r) => r,
            Err(e) => { eprintln!("Clifford runtime error: {e}"); std::process::exit(1); }
        },
        Backend::Sparse => match astracore::simulator::execute_sparse(&program) {
            Ok(r) => r,
            Err(e) => { eprintln!("Sparse runtime error: {e}"); std::process::exit(1); }
        },
        Backend::Gpu => {
            #[cfg(any(feature = "wgpu", feature = "cuda"))]
            match astracore::simulator::execute_gpu(&program) {
                Ok(r) => r,
                Err(e) => { eprintln!("GPU runtime error: {e}"); std::process::exit(1); }
            }
            #[cfg(not(any(feature = "wgpu", feature = "cuda")))]
            {
                eprintln!("GPU backend not compiled in. Rebuild with: cargo build --release --features wgpu");
                std::process::exit(1);
            }
        },
        Backend::Wgpu => {
            #[cfg(feature = "wgpu")]
            match astracore::simulator::execute_wgpu(&program) {
                Ok(r) => r,
                Err(e) => { eprintln!("wgpu runtime error: {e}"); std::process::exit(1); }
            }
            #[cfg(not(feature = "wgpu"))]
            {
                eprintln!("wgpu backend not compiled in. Rebuild with: cargo build --release --features wgpu");
                std::process::exit(1);
            }
        },
        Backend::Cuda => {
            #[cfg(feature = "cuda")]
            match astracore::simulator::execute_cuda(&program) {
                Ok(r) => r,
                Err(e) => { eprintln!("CUDA runtime error: {e}"); std::process::exit(1); }
            }
            #[cfg(not(feature = "cuda"))]
            {
                eprintln!("CUDA backend not compiled in. Rebuild with: cargo build --release --features cuda");
                std::process::exit(1);
            }
        },
        Backend::Dist => {
            // Resolve node addresses from --nodes or --cluster.
            let addrs = if let Some(nodes_str) = dist_nodes.as_ref() {
                match astracore::simulator::parse_nodes(nodes_str) {
                    Ok(a)  => a,
                    Err(e) => { eprintln!("{e}"); std::process::exit(1); }
                }
            } else if let Some(cluster_path) = dist_cluster.as_ref() {
                let toml = match std::fs::read_to_string(cluster_path) {
                    Ok(s)  => s,
                    Err(e) => { eprintln!("Cannot read cluster file '{cluster_path}': {e}"); std::process::exit(1); }
                };
                match astracore::simulator::ClusterConfig::from_str(&toml) {
                    Ok(cfg) => cfg.addresses(),
                    Err(e)  => { eprintln!("{e}"); std::process::exit(1); }
                }
            } else {
                eprintln!("--dist requires --nodes 'host:port,...' or --cluster cluster.toml");
                std::process::exit(1);
            };

            match astracore::simulator::execute_distributed(&program, &addrs) {
                Ok(r)  => r,
                Err(e) => { eprintln!("Distributed runtime error: {e}"); std::process::exit(1); }
            }
        },
    };

    if !result.final_probabilities.is_empty() {
        let display_probs = result.pre_measurement_probs.as_deref()
            .unwrap_or(&result.final_probabilities);
        let label = if result.pre_measurement_probs.is_some() {
            "Pre-measurement state"
        } else {
            "Final state"
        };
        println!("{label}:");
        for (lbl, prob) in result.significant_states(display_probs, 1e-6) {
            println!("  |{lbl}⟩  {prob:.6}");
        }
        println!();
    }

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

/// Run a program `n_shots` times using the selected backend, collecting outcomes.
fn run_shots_with_backend(
    program: &astracore::compiler::ir::Program,
    n_shots: usize,
    backend: Backend,
    bond_dim: usize,
) -> astracore::runtime::ShotResult {
    use std::collections::HashMap;

    let mut counts: HashMap<String, usize> = HashMap::new();

    for _ in 0..n_shots {
        let result = match backend {
            Backend::Statevector => astracore::runtime::execute(program),
            Backend::Sparse      => astracore::simulator::execute_sparse(program),
            Backend::Mps         => astracore::simulator::execute_mps(program, bond_dim),
            Backend::Clifford    => astracore::simulator::execute_clifford(program),
            #[cfg(any(feature = "wgpu", feature = "cuda"))]
            Backend::Gpu  => astracore::simulator::execute_gpu(program).map_err(|e| e),
            #[cfg(feature = "wgpu")]
            Backend::Wgpu => astracore::simulator::execute_wgpu(program).map_err(|e| e),
            #[cfg(feature = "cuda")]
            Backend::Cuda => astracore::simulator::execute_cuda(program).map_err(|e| e),
            // Without GPU features compiled in, fall back to statevector
            #[cfg(not(any(feature = "wgpu", feature = "cuda")))]
            Backend::Gpu | Backend::Wgpu | Backend::Cuda => astracore::runtime::execute(program),
            // Distributed backend not supported in shot mode (single-node fallback)
            Backend::Dist => astracore::runtime::execute(program),
        };
        let result = match result {
            Ok(r) => r,
            Err(e) => { eprintln!("Shot error: {e}"); std::process::exit(1); }
        };
        let key = if result.measurements.is_empty() {
            "(no measurement)".to_string()
        } else if let Some(bs) = result.bitstring() {
            bs
        } else {
            (0..result.num_qubits)
                .map(|q| match result.outcome(q) {
                    Some(true)  => '1',
                    Some(false) => '0',
                    None        => '?',
                })
                .collect()
        };
        *counts.entry(key).or_insert(0) += 1;
    }

    astracore::runtime::ShotResult { counts, n_shots, n_qubits: program.num_qubits }
}

/// Import and run an OpenQASM 2.0 file.
fn cli_import(path: Option<&str>) {
    let path = path.unwrap_or_else(|| {
        eprintln!("Usage: astracore import <file.qasm>");
        std::process::exit(1);
    });
    let source = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => { eprintln!("Cannot read '{path}': {e}"); std::process::exit(1); }
    };
    println!("━━━ AstraCore OpenQASM 2.0 Importer ━━━━━━━━━━━━━━");
    println!("File: {path}\n");
    let program = match compiler::qasm_import::from_qasm(&source) {
        Ok(p) => p,
        Err(e) => { eprintln!("QASM parse error: {e}"); std::process::exit(1); }
    };
    println!("Circuit: {} gate(s) | {} qubit(s)\n",
             program.gate_count, program.num_qubits);
    let result = match compiler::execute(&program) {
        Ok(r) => r,
        Err(e) => { eprintln!("Runtime error: {e}"); std::process::exit(1); }
    };
    if !result.final_probabilities.is_empty() {
        let display_probs = result.pre_measurement_probs.as_deref()
            .unwrap_or(&result.final_probabilities);
        println!("Final state:");
        for (lbl, prob) in result.significant_states(display_probs, 1e-6) {
            println!("  |{lbl}⟩  {prob:.6}");
        }
        println!();
    }
    if !result.measurements.is_empty() {
        println!("Measurement results:");
        for m in &result.measurements {
            println!("  q{}  →  {}", m.qubit, m.outcome as u8);
        }
    }
}

/// Export AQL to OpenQASM 2.0.
fn cli_export(path: Option<&str>, out: Option<&str>) {
    let path = path.unwrap_or_else(|| {
        eprintln!("Usage: astracore export <file.aql> [output.qasm]");
        std::process::exit(1);
    });

    let output = out.map(String::from).unwrap_or_else(|| {
        let stem = std::path::Path::new(path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("circuit");
        format!("{stem}.qasm")
    });

    let source = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => { eprintln!("Cannot read '{path}': {e}"); std::process::exit(1); }
    };

    println!("━━━ AstraCore → OpenQASM 2.0 Exporter ━━━━━━━━━━━━");
    println!("Input : {path}");
    println!("Output: {output}\n");

    let qasm = match compiler::qasm_export::source_to_qasm(&source) {
        Ok(q) => q,
        Err(e) => { eprintln!("Export error: {e}"); std::process::exit(1); }
    };

    match std::fs::write(&output, &qasm) {
        Ok(()) => println!("OpenQASM 2.0 written to '{output}'."),
        Err(e) => { eprintln!("Write error: {e}"); std::process::exit(1); }
    }
}

/// `astracore lsp` — start AQL Language Server on stdin/stdout.
fn cli_lsp() {
    #[cfg(feature = "lsp")]
    {
        tokio::runtime::Runtime::new()
            .expect("tokio runtime")
            .block_on(astracore::lsp::run_lsp());
    }
    #[cfg(not(feature = "lsp"))]
    {
        eprintln!("AQL language server not compiled in. Rebuild with: cargo build --features lsp");
        std::process::exit(1);
    }
}

/// `astracore dap` — start AQL Debug Adapter on stdin/stdout.
fn cli_dap() {
    #[cfg(feature = "lsp")]
    {
        tokio::runtime::Runtime::new()
            .expect("tokio runtime")
            .block_on(astracore::lsp::run_dap());
    }
    #[cfg(not(feature = "lsp"))]
    {
        eprintln!("AQL debug adapter not compiled in. Rebuild with: cargo build --features lsp");
        std::process::exit(1);
    }
}

/// `astracore import3 <file.qasm>` — import and run an OpenQASM 3.0 file.
fn cli_import3(path: Option<&str>) {
    let path = path.unwrap_or_else(|| {
        eprintln!("Usage: astracore import3 <file.qasm>");
        std::process::exit(1);
    });
    let source = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => { eprintln!("Cannot read '{path}': {e}"); std::process::exit(1); }
    };
    println!("━━━ AstraCore OpenQASM 3.0 Importer ━━━━━━━━━━━━━━");
    println!("File: {path}\n");
    let program = match compiler::qasm3_import::from_qasm3(&source) {
        Ok(p) => p,
        Err(e) => { eprintln!("QASM 3.0 parse error: {e}"); std::process::exit(1); }
    };
    println!("Circuit: {} gate(s) | {} qubit(s)\n",
             program.gate_count, program.num_qubits);
    let result = match compiler::execute(&program) {
        Ok(r) => r,
        Err(e) => { eprintln!("Runtime error: {e}"); std::process::exit(1); }
    };
    if !result.final_probabilities.is_empty() {
        let display_probs = result.pre_measurement_probs.as_deref()
            .unwrap_or(&result.final_probabilities);
        println!("Final state:");
        for (lbl, prob) in result.significant_states(display_probs, 1e-6) {
            println!("  |{lbl}⟩  {prob:.6}");
        }
        println!();
    }
    if !result.measurements.is_empty() {
        println!("Measurement results:");
        for m in &result.measurements {
            println!("  q{}  →  {}", m.qubit, m.outcome as u8);
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

// ── Dashboard CLI ─────────────────────────────────────────────────────────

/// Build a [`DashboardData`] by parsing, analyzing, and executing an AQL file.
fn load_dashboard_data(path: &str) -> DashboardData {
    let source = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => { eprintln!("Cannot read '{}': {}", path, e); std::process::exit(1); }
    };
    let program = match compiler::parse_source(&source) {
        Ok(p) => p,
        Err(e) => { eprintln!("{e}"); std::process::exit(1); }
    };
    let circuit_svg = dashboard::circuit_svg::render(&program.instructions, program.num_qubits);
    let analysis = match compiler::analyze_source(&source) {
        Ok(a) => a,
        Err(e) => { eprintln!("{e}"); std::process::exit(1); }
    };
    let result = match compiler::run(&source) {
        Ok(r) => r,
        Err(e) => { eprintln!("{e}"); std::process::exit(1); }
    };
    DashboardData { source_path: path.to_string(), analysis, result, circuit_svg }
}

fn cli_dash(path: Option<&str>) {
    let path = path.unwrap_or_else(|| {
        eprintln!("Usage: astracore dash <file.aql>");
        std::process::exit(1);
    });
    println!("━━━ AstraCore TUI Dashboard ━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Loading {}…", path);
    let data = load_dashboard_data(path);
    if let Err(e) = dashboard::run_tui(&data) {
        eprintln!("TUI error: {e}");
        std::process::exit(1);
    }
}

fn cli_report(path: Option<&str>, out: Option<&str>) {
    let path = path.unwrap_or_else(|| {
        eprintln!("Usage: astracore report <file.aql> [output.html]");
        std::process::exit(1);
    });
    // Default output path: same directory as input, with .html extension
    let output = out.map(String::from).unwrap_or_else(|| {
        let stem = std::path::Path::new(path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("report");
        format!("{}.html", stem)
    });
    println!("━━━ AstraCore HTML Report ━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Source : {path}");
    println!("Output : {output}");
    println!();
    let data = load_dashboard_data(path);
    match dashboard::generate_report(&data, &output) {
        Ok(()) => println!("Report written to '{output}'. Open in a browser to view."),
        Err(e) => { eprintln!("Write error: {e}"); std::process::exit(1); }
    }
}

fn cli_serve(path: Option<&str>, port_arg: Option<&str>) {
    let path = path.unwrap_or_else(|| {
        eprintln!("Usage: astracore serve <file.aql> [port]");
        std::process::exit(1);
    });
    let port: u16 = port_arg
        .and_then(|s| s.parse().ok())
        .unwrap_or(8080);
    let data = load_dashboard_data(path);
    dashboard::serve(data, port);
}

/// `astracore worker --port <N>` — start a distributed simulation worker node.
fn cli_worker(args: &[String]) {
    let mut port: u16 = 7700;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--port" | "-p" => {
                i += 1;
                port = args.get(i)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or_else(|| {
                        eprintln!("--port requires a valid port number");
                        std::process::exit(1);
                    });
            }
            flag => { eprintln!("Unknown worker flag '{flag}'"); std::process::exit(1); }
        }
        i += 1;
    }
    if let Err(e) = astracore::simulator::dist::worker::run_worker_server(port) {
        eprintln!("Worker error: {e}");
        std::process::exit(1);
    }
}

/// `astracore devices` — list available simulation devices.
fn cli_devices() {
    println!("━━━ AstraCore — Available Simulation Devices ━━━━━━━");
    for dev in astracore::simulator::list_gpu_devices() {
        println!("  {dev}");
    }
    println!();
    println!("GPU features compiled in:");

    #[cfg(feature = "wgpu")]
    println!("  wgpu  ✓  (cross-platform WebGPU via Vulkan/Metal/DX12)");
    #[cfg(not(feature = "wgpu"))]
    println!("  wgpu  ✗  (rebuild with --features wgpu to enable)");

    #[cfg(feature = "cuda")]
    println!("  cuda  ✓  (NVIDIA CUDA via cudarc)");
    #[cfg(not(feature = "cuda"))]
    println!("  cuda  ✗  (rebuild with --features cuda to enable)");
}

fn print_help() {
    println!("Usage: astracore [COMMAND] [ARGS]\n");
    println!("Commands:");
    println!("  demo                              Run built-in demonstration circuits");
    println!("  run <file.aql> [--backend B]      Execute an AQL program");
    println!("       --backend statevector         Default: full state vector (≤30 qubits)");
    println!("       --backend mps [--bond-dim N]  Matrix Product State (50–200+ qubits)");
    println!("       --backend clifford            Stabilizer circuit (unlimited qubits)");
    println!("       --backend sparse              Sparse statevector (low-entanglement)");
    println!("       --backend gpu                 GPU statevector (needs --features wgpu or cuda)");
    println!("       --backend wgpu                WebGPU backend (needs --features wgpu)");
    println!("       --backend cuda                CUDA backend   (needs --features cuda)");
    println!("       --shots N                     Run N times, output measurement histogram");
    println!("  devices                           List available GPU devices");
    println!("  worker [--port N]                 Start a distributed worker node (default port 7700)");
    println!("       run circuit.aql --dist --nodes 'h1:7700,h2:7700'  distributed execution");
    println!("  opt <file.aql>                    Optimize and display the circuit");
    println!("  analyze <file.aql>                Static circuit analysis and profiling");
    println!("  export <file.aql> [out.qasm]      Export circuit to OpenQASM 2.0");
    println!("  import <file.qasm>                Import and run an OpenQASM 2.0 file");
    println!("  import3 <file.qasm>               Import and run an OpenQASM 3.0 file");
    println!("  lsp                               Start AQL Language Server (LSP, stdin/stdout)");
    println!("  dap                               Start AQL Debug Adapter (DAP, stdin/stdout)");
    println!("  dash <file.aql>                   Launch interactive TUI dashboard");
    println!("  report <file.aql> [out.html]      Generate standalone HTML report");
    println!("  serve <file.aql> [port]           Start local HTTP dashboard server");
    println!("  help                              Show this message\n");
    println!("AQL v2 Instructions:");
    println!("  QREG <n>              Declare n qubits (must be first; up to 1000 for MPS/Clifford)");
    println!("  H|X|Y|Z|S|T <q>       Single-qubit gates");
    println!("  RX|RY|RZ <q> <θ>      Rotation gates (radians) — statevector/MPS only");
    println!("  PHASE <q> <θ>         Phase gate");
    println!("  CNOT|CZ|SWAP <c> <t>  Two-qubit gates");
    println!("  CCX <c0> <c1> <t>     Toffoli (CCNOT) gate — statevector/MPS only");
    println!("  MEASURE <q>           Measure qubit q");
    println!("  MEASURE_ALL           Measure all qubits");
    println!("  BARRIER               Visual separator (no-op)");
    println!("  REPEAT N … END        Unroll body N times at compile time  [v2]");
    println!("  INCLUDE filename.aql  Inline another AQL file              [v2]");
    println!("  GATE name n … END     Define a custom n-qubit gate");
    println!("  CALL name q0 q1 …     Invoke a custom gate\n");
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
