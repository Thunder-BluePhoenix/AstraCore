/// Integration layer — connects the `PluginRegistry` to the AQL execution pipeline.
///
/// # Public API
/// - `execute_with_plugins(program, registry)` — run a compiled `Program` with plugins.
/// - `run_with_plugins(source, registry)` — one-shot: parse + optimize + execute.
///
/// Neither function modifies the existing `execute` or `run` functions.
use std::collections::HashMap;
use crate::compiler::{
    AqlError,
    ir::{Instruction, Program},
    parse_source,
};
use crate::core::StateVector;
use crate::runtime::{ExecutionResult, MeasurementRecord};
use super::backend::SimulationBackend;
use super::registry::PluginRegistry;

/// Hard step limit — same value as in the existing executor.
const MAX_STEPS: usize = 1_000_000;

// ── Public entry points ───────────────────────────────────────────────────

/// Execute an AQL `Program` with the full plugin stack.
///
/// Semantically identical to `crate::runtime::execute` with three extensions:
///
/// 1. **Backend dispatch** — every primitive gate is routed through
///    `registry.backend()` instead of being hardcoded to the SIMD path.
/// 2. **Gate plugin fallback** — when a `CallGate` name is absent from
///    `program.gate_defs`, the registry's gate plugins are checked.
/// 3. **Arity validation** — plugin arity is validated before dispatch,
///    returning `AqlError::Runtime` on mismatch rather than panicking.
///
/// The optimizer pipeline is **not** applied here; use `run_with_plugins` for
/// a parse-optimize-execute pipeline, or call `registry.run_optimizer_pipeline`
/// manually before calling this function.
///
/// # Errors
/// Returns `AqlError::Runtime` for undefined labels, arity mismatches,
/// undefined gate names, unmeasured qubit reads, and step-limit violations.
pub fn execute_with_plugins(
    program: &Program,
    registry: &PluginRegistry,
) -> Result<ExecutionResult, AqlError> {
    let label_table = build_label_table(&program.instructions);
    let backend     = registry.backend();

    let mut state       = StateVector::new(program.num_qubits);
    let mut classical   = vec![None::<bool>; program.num_qubits];
    let mut measurements: Vec<MeasurementRecord> = Vec::new();
    let mut pre_measurement_probs: Option<Vec<f64>> = None;
    let mut gate_count    = 0usize;
    let mut branch_count  = 0usize;
    let mut steps_executed = 0usize;
    let mut first_measure  = true;

    let mut pc: usize = 0;

    while pc < program.instructions.len() {
        if steps_executed >= MAX_STEPS {
            return Err(AqlError::Runtime {
                msg: format!(
                    "execution exceeded {MAX_STEPS} steps — possible infinite loop"
                ),
            });
        }
        steps_executed += 1;

        let instr = &program.instructions[pc];

        // Snapshot probabilities before the first measurement.
        if instr.is_measurement() && first_measure {
            pre_measurement_probs = Some(
                state.amplitudes.iter().map(|a| a.norm_sq()).collect()
            );
            first_measure = false;
        }

        match instr {
            // ── Single-qubit gates ────────────────────────────────────────
            Instruction::H(q) => {
                backend.apply_single_qubit_gate(
                    &mut state, &crate::core::gates::hadamard(), *q,
                )?;
                gate_count += 1;
            }
            Instruction::X(q) => {
                backend.apply_single_qubit_gate(
                    &mut state, &crate::core::gates::pauli_x(), *q,
                )?;
                gate_count += 1;
            }
            Instruction::Y(q) => {
                backend.apply_single_qubit_gate(
                    &mut state, &crate::core::gates::pauli_y(), *q,
                )?;
                gate_count += 1;
            }
            Instruction::Z(q) => {
                backend.apply_single_qubit_gate(
                    &mut state, &crate::core::gates::pauli_z(), *q,
                )?;
                gate_count += 1;
            }
            Instruction::S(q) => {
                backend.apply_single_qubit_gate(
                    &mut state, &crate::core::gates::s_gate(), *q,
                )?;
                gate_count += 1;
            }
            Instruction::T(q) => {
                backend.apply_single_qubit_gate(
                    &mut state, &crate::core::gates::t_gate(), *q,
                )?;
                gate_count += 1;
            }
            Instruction::Rx { qubit, theta } => {
                backend.apply_single_qubit_gate(
                    &mut state, &crate::core::gates::rx(*theta), *qubit,
                )?;
                gate_count += 1;
            }
            Instruction::Ry { qubit, theta } => {
                backend.apply_single_qubit_gate(
                    &mut state, &crate::core::gates::ry(*theta), *qubit,
                )?;
                gate_count += 1;
            }
            Instruction::Rz { qubit, theta } => {
                backend.apply_single_qubit_gate(
                    &mut state, &crate::core::gates::rz(*theta), *qubit,
                )?;
                gate_count += 1;
            }
            Instruction::Phase { qubit, theta } => {
                backend.apply_single_qubit_gate(
                    &mut state, &crate::core::gates::phase_gate(*theta), *qubit,
                )?;
                gate_count += 1;
            }

            // ── Multi-qubit gates ─────────────────────────────────────────
            Instruction::Cnot { control, target } => {
                backend.apply_cnot(&mut state, *control, *target)?;
                gate_count += 1;
            }
            Instruction::Cz { control, target } => {
                backend.apply_cz(&mut state, *control, *target)?;
                gate_count += 1;
            }
            Instruction::Swap { qubit_a, qubit_b } => {
                backend.apply_swap(&mut state, *qubit_a, *qubit_b)?;
                gate_count += 1;
            }
            Instruction::Toffoli { control0, control1, target } => {
                backend.apply_toffoli(&mut state, *control0, *control1, *target)?;
                gate_count += 1;
            }

            // ── Measurement ───────────────────────────────────────────────
            Instruction::Measure(q) => {
                let rng = rand::random::<f64>();
                let outcome = state.collapse(*q, rng);
                classical[*q] = Some(outcome);
                measurements.push(MeasurementRecord { qubit: *q, outcome, step: pc });
            }
            Instruction::MeasureAll => {
                for q in 0..program.num_qubits {
                    let rng = rand::random::<f64>();
                    let outcome = state.collapse(q, rng);
                    classical[q] = Some(outcome);
                    measurements.push(MeasurementRecord { qubit: q, outcome, step: pc });
                }
            }

            // ── Structural ────────────────────────────────────────────────
            Instruction::Barrier  => {}
            Instruction::Label(_) => {}

            // ── CallGate: gate_defs → plugin registry → error ─────────────
            Instruction::CallGate { name, qubits: arg_qubits } => {
                // Priority 1: AQL GATE…END definitions
                if let Some(def) = program.gate_defs.get(name.as_str()) {
                    for body_instr in def.body.clone() {
                        execute_remapped_with_backend(
                            &body_instr, arg_qubits, &mut state, backend,
                        )?;
                    }
                    gate_count += 1;
                }
                // Priority 2: registry gate plugins
                else if let Some(plugin) = registry.get_gate(name.as_str()) {
                    if plugin.num_qubits() != arg_qubits.len() {
                        return Err(AqlError::Runtime {
                            msg: format!(
                                "gate plugin '{}' expects {} qubits, got {}",
                                name, plugin.num_qubits(), arg_qubits.len()
                            ),
                        });
                    }
                    plugin.apply(&mut state, arg_qubits);
                    gate_count += 1;
                }
                // Priority 3: error
                else {
                    return Err(AqlError::Runtime {
                        msg: format!(
                            "undefined gate '{}' (not in gate_defs or plugin registry)", name
                        ),
                    });
                }
            }

            // ── Control flow ──────────────────────────────────────────────
            Instruction::Goto { label } => {
                let &target = label_table.get(label.as_str()).ok_or_else(|| {
                    AqlError::Runtime {
                        msg: format!("undefined label '{label}'"),
                    }
                })?;
                pc = target;
                continue;
            }
            Instruction::GotoIf { qubit, label } => {
                let reg = classical[*qubit].ok_or_else(|| AqlError::Runtime {
                    msg: format!("IF on qubit {qubit}: not yet measured"),
                })?;
                if reg {
                    let &target = label_table.get(label.as_str()).ok_or_else(|| {
                        AqlError::Runtime { msg: format!("undefined label '{label}'") }
                    })?;
                    pc = target;
                    branch_count += 1;
                    continue;
                }
            }
            Instruction::GotoIfNot { qubit, label } => {
                let reg = classical[*qubit].ok_or_else(|| AqlError::Runtime {
                    msg: format!("IFNOT on qubit {qubit}: not yet measured"),
                })?;
                if !reg {
                    let &target = label_table.get(label.as_str()).ok_or_else(|| {
                        AqlError::Runtime { msg: format!("undefined label '{label}'") }
                    })?;
                    pc = target;
                    branch_count += 1;
                    continue;
                }
            }
            Instruction::MeasureInto { .. } | Instruction::GotoIfCreg { .. }
            | Instruction::GotoIfNotCreg { .. } => {
                return Err(AqlError::Runtime {
                    msg: "CREG instructions not supported in plugin executor; use the default backend".to_string(),
                });
            }
        }

        pc += 1;
    }

    let final_amplitudes = state.amplitudes.iter().map(|a| (a.re, a.im)).collect();
    Ok(ExecutionResult {
        num_qubits: program.num_qubits,
        measurements,
        pre_measurement_probs,
        pre_measurement_amplitudes: None,
        final_probabilities: state.amplitudes.iter().map(|a| a.norm_sq()).collect(),
        final_amplitudes,
        gate_count,
        branch_count,
        steps_executed,
    })
}

/// One-shot: lex → parse → optimizer pipeline → execute with plugins.
///
/// Mirrors `compiler::run` but uses the plugin stack:
/// - Plugin-gate stubs are injected before parsing so CALL validation passes.
/// - Optimizer pipeline from `registry.run_optimizer_pipeline`.
/// - Execution through `execute_with_plugins`.
///
/// Gate definitions declared in the AQL source are preserved through optimization.
pub fn run_with_plugins(
    source: &str,
    registry: &PluginRegistry,
) -> Result<ExecutionResult, AqlError> {
    let program = parse_source_with_plugins(source, registry)?;
    let (opt_instrs, _stats) = registry.run_optimizer_pipeline(&program.instructions);
    let opt_program = Program::with_gate_defs(
        program.num_qubits, opt_instrs, program.gate_defs,
    );
    execute_with_plugins(&opt_program, registry)
}

/// Parse AQL source while allowing `CALL` references to plugin-registered gate names.
///
/// Plugin gates are Rust-level and unknown to the parser. This function:
/// 1. Pre-scans the source for `GATE` definitions it already contains.
/// 2. Generates a stub `GATE name n\nEND` for each plugin whose name is **not**
///    already defined in the source — so the parser's CALL arity validation passes.
/// 3. Removes the stubs from `program.gate_defs` after parsing, so
///    `execute_with_plugins` correctly resolves them via the registry at runtime.
///
/// AQL-defined gates always keep their source bodies and take execution priority
/// over any same-named plugin.
pub fn parse_source_with_plugins(
    source: &str,
    registry: &PluginRegistry,
) -> Result<Program, AqlError> {
    use std::collections::HashSet;

    // Step 1: names already defined by GATE blocks in the source
    let source_defined: HashSet<String> = source
        .lines()
        .filter_map(|line| {
            let mut parts = line.split_whitespace();
            match parts.next() {
                Some(kw) if kw.eq_ignore_ascii_case("GATE") => {
                    parts.next().map(|n| n.to_lowercase())
                }
                _ => None,
            }
        })
        .collect();

    // Step 2: stub definitions for plugin-only gates
    let mut preamble = String::new();
    for (name, plugin) in &registry.gates {
        if !source_defined.contains(name) {
            preamble.push_str(&format!("GATE {} {}\nEND\n", name, plugin.num_qubits()));
        }
    }

    // Step 3: parse combined source
    let full = format!("{}{}", preamble, source);
    let mut program = parse_source(&full)?;

    // Step 4: remove stubs so executor uses plugin at runtime
    for name in registry.gates.keys() {
        if !source_defined.contains(name) {
            program.gate_defs.remove(name.as_str());
        }
    }

    Ok(program)
}

// ── Private helpers ───────────────────────────────────────────────────────

fn build_label_table(instructions: &[Instruction]) -> HashMap<String, usize> {
    let mut table = HashMap::new();
    for (i, instr) in instructions.iter().enumerate() {
        if let Instruction::Label(name) = instr {
            table.insert(name.clone(), i + 1);
        }
    }
    table
}

/// Execute one instruction from a gate body with local→global qubit remapping,
/// dispatching through the provided backend.
fn execute_remapped_with_backend(
    instr: &Instruction,
    qubit_map: &[usize],
    state: &mut StateVector,
    backend: &dyn SimulationBackend,
) -> Result<(), AqlError> {
    let q = |local: usize| qubit_map[local];
    match instr {
        Instruction::H(i) =>
            backend.apply_single_qubit_gate(state, &crate::core::gates::hadamard(), q(*i))?,
        Instruction::X(i) =>
            backend.apply_single_qubit_gate(state, &crate::core::gates::pauli_x(), q(*i))?,
        Instruction::Y(i) =>
            backend.apply_single_qubit_gate(state, &crate::core::gates::pauli_y(), q(*i))?,
        Instruction::Z(i) =>
            backend.apply_single_qubit_gate(state, &crate::core::gates::pauli_z(), q(*i))?,
        Instruction::S(i) =>
            backend.apply_single_qubit_gate(state, &crate::core::gates::s_gate(), q(*i))?,
        Instruction::T(i) =>
            backend.apply_single_qubit_gate(state, &crate::core::gates::t_gate(), q(*i))?,
        Instruction::Rx { qubit, theta } =>
            backend.apply_single_qubit_gate(state, &crate::core::gates::rx(*theta), q(*qubit))?,
        Instruction::Ry { qubit, theta } =>
            backend.apply_single_qubit_gate(state, &crate::core::gates::ry(*theta), q(*qubit))?,
        Instruction::Rz { qubit, theta } =>
            backend.apply_single_qubit_gate(state, &crate::core::gates::rz(*theta), q(*qubit))?,
        Instruction::Phase { qubit, theta } =>
            backend.apply_single_qubit_gate(state, &crate::core::gates::phase_gate(*theta), q(*qubit))?,
        Instruction::Cnot { control, target } =>
            backend.apply_cnot(state, q(*control), q(*target))?,
        Instruction::Cz { control, target } =>
            backend.apply_cz(state, q(*control), q(*target))?,
        Instruction::Swap { qubit_a, qubit_b } =>
            backend.apply_swap(state, q(*qubit_a), q(*qubit_b))?,
        Instruction::Toffoli { control0, control1, target } =>
            backend.apply_toffoli(state, q(*control0), q(*control1), q(*target))?,
        Instruction::Barrier => {}
        _ => {} // Measure / Label / control flow not valid inside a gate body
    }
    Ok(())
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::parse_source;
    use crate::plugins::gate::FnGatePlugin;
    use crate::plugins::optimizer::{OptimizerPass, PeepholePass};
    use crate::plugins::backend::{CpuBackend, SimulationBackend};
    use crate::plugins::registry::PluginRegistry;
    use crate::core::gates::{apply_single_qubit_gate, pauli_x, hadamard, apply_cnot};
    use crate::core::{Matrix2x2, StateVector};
    use crate::compiler::AqlError;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    // ── Test 1: default registry matches existing execute ─────────────────

    #[test]
    fn default_registry_bell_matches_execute() {
        let src = "QREG 2\nH 0\nCNOT 0 1\nMEASURE_ALL";
        let program = parse_source(src).unwrap();
        let registry = PluginRegistry::default();
        let result = execute_with_plugins(&program, &registry).unwrap();
        assert_eq!(result.num_qubits, 2);
        assert_eq!(result.measurements.len(), 2);
        // Bell: qubits must agree
        let q0 = result.outcome(0).unwrap();
        let q1 = result.outcome(1).unwrap();
        assert_eq!(q0, q1, "Bell pair: both qubits must agree");
    }

    // ── Test 2: FnGatePlugin registration and lookup ──────────────────────

    #[test]
    fn gate_plugin_register_and_lookup() {
        let mut reg = PluginRegistry::new();
        let plugin = FnGatePlugin::new("rust_x", 1, |_, _| {});
        reg.register_gate(Box::new(plugin));
        let got = reg.get_gate("rust_x");
        assert!(got.is_some());
        assert_eq!(got.unwrap().num_qubits(), 1);
    }

    // ── Test 3: X gate plugin applied via CALL gives 1 ───────────────────

    #[test]
    fn call_x_plugin_measures_one() {
        let mut reg = PluginRegistry::new();
        reg.register_gate(Box::new(FnGatePlugin::new("rust_x", 1, |state, qubits| {
            apply_single_qubit_gate(state, &pauli_x(), qubits[0]);
        })));
        // Use run_with_plugins: it injects a stub so the parser accepts CALL rust_x
        let result = run_with_plugins("QREG 1\nCALL rust_x 0\nMEASURE 0", &reg).unwrap();
        assert_eq!(result.outcome(0), Some(true));
    }

    // ── Test 4: 2-qubit bell plugin via CALL ──────────────────────────────

    #[test]
    fn call_bell_plugin_entangles() {
        let mut reg = PluginRegistry::new();
        reg.register_gate(Box::new(FnGatePlugin::new("plugin_bell", 2, |state, qubits| {
            apply_single_qubit_gate(state, &hadamard(), qubits[0]);
            apply_cnot(state, qubits[0], qubits[1]);
        })));
        let src = "QREG 2\nCALL plugin_bell 0 1\nMEASURE_ALL";
        // Run many times; both outcomes (00, 11) must occur and qubits must agree
        let mut got_zero = false;
        let mut got_one  = false;
        for _ in 0..100 {
            let r = run_with_plugins(src, &reg).unwrap();
            let q0 = r.outcome(0).unwrap();
            let q1 = r.outcome(1).unwrap();
            assert_eq!(q0, q1, "Bell pair must agree");
            if !q0 { got_zero = true; } else { got_one = true; }
        }
        assert!(got_zero, "should sometimes measure 00");
        assert!(got_one,  "should sometimes measure 11");
    }

    // ── Test 5: gate_defs takes priority over same-named plugin ───────────

    #[test]
    fn gate_defs_priority_over_plugin() {
        // AQL GATE do_nothing acts as identity (empty body)
        // Plugin "do_nothing" applies X — if priority is wrong, outcome would be 1
        let mut reg = PluginRegistry::new();
        reg.register_gate(Box::new(FnGatePlugin::new("do_nothing", 1, |state, qubits| {
            apply_single_qubit_gate(state, &pauli_x(), qubits[0]);
        })));
        // run_with_plugins: source defines "do_nothing" so stub is not injected;
        // gate_defs entry (empty body) is preserved and wins over the plugin.
        let result = run_with_plugins(
            "GATE do_nothing 1\nEND\nQREG 1\nCALL do_nothing 0\nMEASURE 0",
            &reg,
        ).unwrap();
        // AQL gate is identity → qubit stays |0⟩ → measures 0
        assert_eq!(result.outcome(0), Some(false), "AQL gate_defs must win over plugin");
    }

    // ── Test 6: arity mismatch returns error ──────────────────────────────

    #[test]
    fn arity_mismatch_returns_error() {
        let mut reg = PluginRegistry::new();
        reg.register_gate(Box::new(FnGatePlugin::new("two_q", 2, |_, _| {})));
        // CALL two_q with 1 qubit arg — stub has 2, parser detects mismatch
        let result = run_with_plugins("QREG 2\nCALL two_q 0", &reg);
        assert!(result.is_err(), "arity mismatch must return an error");
    }

    // ── Test 7: undefined gate returns error ──────────────────────────────

    #[test]
    fn undefined_gate_returns_error() {
        let reg = PluginRegistry::new(); // empty — no stub for "ghost"
        // Parser rejects CALL ghost because no gate_defs and no plugin stub
        let result = run_with_plugins("QREG 1\nCALL ghost 0", &reg);
        assert!(result.is_err(), "undefined gate must return an error");
    }

    // ── Test 8: PeepholePass cancels H·H ─────────────────────────────────

    #[test]
    fn peephole_pass_cancels_hh() {
        let instrs = vec![Instruction::H(0), Instruction::H(0)];
        let (out, stats) = PeepholePass.run(&instrs);
        assert!(out.is_empty());
        assert_eq!(stats.gates_removed, 2);
    }

    // ── Test 9: optimizer pipeline aggregates stats ───────────────────────

    #[test]
    fn optimizer_pipeline_aggregates_stats() {
        let mut reg = PluginRegistry::new();
        reg.add_optimizer_pass(Box::new(PeepholePass));
        reg.add_optimizer_pass(Box::new(PeepholePass));
        // H·H and X·X — first pass cancels both in one sweep
        let instrs = vec![
            Instruction::H(0), Instruction::H(0),
            Instruction::X(1), Instruction::X(1),
        ];
        let (out, stats) = reg.run_optimizer_pipeline(&instrs);
        assert!(out.is_empty());
        assert_eq!(stats.gates_removed, 4);
    }

    // ── Test 10: empty optimizer pipeline leaves instructions unchanged ────

    #[test]
    fn empty_pipeline_no_change() {
        let reg = PluginRegistry::new();
        let instrs = vec![Instruction::H(0), Instruction::H(0)];
        let (out, stats) = reg.run_optimizer_pipeline(&instrs);
        assert_eq!(out.len(), 2);
        assert_eq!(stats.gates_removed, 0);
        assert_eq!(stats.passes, 0);
    }

    // ── Test 11: run_with_plugins cancels H·H via default peephole ────────

    #[test]
    fn run_with_plugins_hh_cancels() {
        let reg = PluginRegistry::default(); // has PeepholePass
        // H·H on q0 → identity → q0 stays |0⟩ → measures 0
        let result = run_with_plugins("QREG 1\nH 0\nH 0\nMEASURE 0", &reg).unwrap();
        assert_eq!(result.outcome(0), Some(false));
        assert_eq!(result.gate_count, 0, "H·H cancelled → no gates executed");
    }

    // ── Test 12: CpuBackend matches apply_gate_simd directly ─────────────

    #[test]
    fn cpu_backend_matches_simd_direct() {
        let gate = hadamard();
        let mut s1 = StateVector::new(2);
        let mut s2 = StateVector::new(2);
        CpuBackend.apply_single_qubit_gate(&mut s1, &gate, 0).unwrap();
        crate::core::simd::apply_gate_simd(&mut s2, &gate, 0);
        for (a, b) in s1.amplitudes.iter().zip(s2.amplitudes.iter()) {
            assert!((a.re - b.re).abs() < 1e-12);
            assert!((a.im - b.im).abs() < 1e-12);
        }
    }

    // ── Test 13: custom CountingBackend counts single-qubit gate calls ─────

    struct CountingBackend {
        count: Arc<AtomicUsize>,
    }
    impl SimulationBackend for CountingBackend {
        fn name(&self) -> &str { "counting" }
        fn apply_single_qubit_gate(
            &self,
            state: &mut StateVector,
            matrix: &Matrix2x2,
            target: usize,
        ) -> Result<(), AqlError> {
            self.count.fetch_add(1, Ordering::Relaxed);
            crate::core::simd::apply_gate_simd(state, matrix, target);
            Ok(())
        }
    }

    #[test]
    fn counting_backend_counts_gates() {
        let counter = Arc::new(AtomicUsize::new(0));
        let mut reg = PluginRegistry::new();
        reg.set_backend(Box::new(CountingBackend { count: counter.clone() }));
        let src = "QREG 2\nH 0\nX 1\nZ 0\nMEASURE_ALL";
        let program = parse_source(src).unwrap();
        execute_with_plugins(&program, &reg).unwrap();
        assert_eq!(counter.load(Ordering::Relaxed), 3, "3 single-qubit gates");
    }

    // ── Test 14: PluginRegistry::default fields ────────────────────────────

    #[test]
    fn default_registry_fields() {
        let reg = PluginRegistry::default();
        assert_eq!(reg.gate_count(), 0);
        assert_eq!(reg.backend_name(), "cpu");
        assert_eq!(reg.optimizer_pass_count(), 1);
    }

    // ── Test 15: teleportation circuit through execute_with_plugins ────────

    #[test]
    fn teleportation_via_plugins() {
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
        let reg = PluginRegistry::default();
        let program = parse_source(src).unwrap();
        // Run 200 times; q2 should measure both 0 and 1 (P ≈ 0.5)
        let mut zeros = 0usize;
        let mut ones  = 0usize;
        for _ in 0..200 {
            let r = execute_with_plugins(&program, &reg).unwrap();
            match r.outcome(2).unwrap() {
                false => zeros += 1,
                true  => ones  += 1,
            }
        }
        assert!(zeros > 50, "teleportation: q2 should sometimes measure 0");
        assert!(ones  > 50, "teleportation: q2 should sometimes measure 1");
    }
}
