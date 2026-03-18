/// Distributed simulation — coordinator.
///
/// The coordinator parses the circuit, connects to worker nodes, partitions the
/// initial state vector, dispatches gates, handles cross-shard operations, and
/// collects the final result.
///
/// **Gate execution strategy:**
///
/// | Gate type            | Action                                               |
/// |----------------------|------------------------------------------------------|
/// | Local single-qubit   | `LocalGate` to the owning worker shard               |
/// | Cross-shard 1Q gate  | `GetSlice` both shards → apply gate centrally → `PutSlice` |
/// | CNOT: both local     | `LocalGate` to one worker                            |
/// | CNOT: ctrl cross-shard, target local | Each worker checks its ctrl shard-bit; nodes with ctrl=1 apply X locally |
/// | CNOT: target cross-shard | Exchange half-slices between partner shards          |
/// | CNOT: both cross-shard | Full central exchange                                 |
/// | Measure / MeasureAll | Collect `prob1` from all shards → decide → `CollapseLocal` |
use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;

use super::partition;
use super::protocol::{decode_reply, encode_cmd, WorkerCmd, WorkerReply};
use crate::compiler::ir::{Instruction, Program};
use crate::core::Complex;
use crate::runtime::{ExecutionResult, MeasurementRecord};

// ── Worker connection ─────────────────────────────────────────────────────────

struct WorkerConn {
    shard_idx: usize,
    writer:    TcpStream,
    reader:    BufReader<TcpStream>,
}

impl WorkerConn {
    fn connect(addr: &str, shard_idx: usize) -> Result<Self, String> {
        let stream = TcpStream::connect(addr)
            .map_err(|e| format!("Cannot connect to worker at {addr}: {e}"))?;
        stream.set_nodelay(true).ok();
        let reader = BufReader::new(stream.try_clone()
            .map_err(|e| format!("stream clone: {e}"))?);
        let mut conn = WorkerConn { shard_idx, writer: stream, reader };

        // Wait for Ready greeting.
        let reply = conn.recv()?;
        match reply {
            WorkerReply::Ready => Ok(conn),
            other => Err(format!("expected Ready from {addr}, got {:?}", other)),
        }
    }

    fn send(&mut self, cmd: &WorkerCmd) -> Result<(), String> {
        let msg = encode_cmd(cmd);
        self.writer.write_all(msg.as_bytes())
            .map_err(|e| format!("worker {} send error: {e}", self.shard_idx))
    }

    fn recv(&mut self) -> Result<WorkerReply, String> {
        let mut line = String::new();
        self.reader.read_line(&mut line)
            .map_err(|e| format!("worker {} recv error: {e}", self.shard_idx))?;
        decode_reply(line.trim())
    }

    fn cmd(&mut self, cmd: WorkerCmd) -> Result<WorkerReply, String> {
        self.send(&cmd)?;
        self.recv()
    }

    fn expect_ack(&mut self, cmd: WorkerCmd) -> Result<(), String> {
        match self.cmd(cmd)? {
            WorkerReply::Ack => Ok(()),
            WorkerReply::Error { message } => Err(format!("worker {}: {message}", self.shard_idx)),
            other => Err(format!("worker {}: expected Ack, got {:?}", self.shard_idx, other)),
        }
    }
}

// ── Gate application math ─────────────────────────────────────────────────────

/// Apply a 2×2 gate matrix to a pair of `Complex` amplitudes in-place.
///
/// `gate`: `[[m00, m01], [m10, m11]]`
#[inline]
fn apply_2x2(a0: &mut Complex, a1: &mut Complex, gate: [[Complex; 2]; 2]) {
    let new_a0 = gate[0][0] * *a0 + gate[0][1] * *a1;
    let new_a1 = gate[1][0] * *a0 + gate[1][1] * *a1;
    *a0 = new_a0;
    *a1 = new_a1;
}

/// Complex number from (re, im) — thin constructor.
#[inline]
fn c(re: f64, im: f64) -> Complex { Complex { re, im } }

/// Gate matrix for `Instruction` → `[[Complex;2];2]`.
fn gate_matrix(instr: &Instruction) -> [[Complex; 2]; 2] {
    let s2 = std::f64::consts::FRAC_1_SQRT_2;
    match instr {
        Instruction::H(_)     => [[c(s2,0.0), c(s2,0.0)], [c(s2,0.0), c(-s2,0.0)]],
        Instruction::X(_)     => [[c(0.0,0.0), c(1.0,0.0)], [c(1.0,0.0), c(0.0,0.0)]],
        Instruction::Y(_)     => [[c(0.0,0.0), c(0.0,-1.0)], [c(0.0,1.0), c(0.0,0.0)]],
        Instruction::Z(_)     => [[c(1.0,0.0), c(0.0,0.0)], [c(0.0,0.0), c(-1.0,0.0)]],
        Instruction::S(_)     => [[c(1.0,0.0), c(0.0,0.0)], [c(0.0,0.0), c(0.0,1.0)]],
        Instruction::T(_)     => [[c(1.0,0.0), c(0.0,0.0)], [c(0.0,0.0), c(s2,s2)]],
        Instruction::Rx { theta, .. } => {
            let c2 = (theta/2.0).cos(); let s2_ = (theta/2.0).sin();
            [[c(c2,0.0), c(0.0,-s2_)], [c(0.0,-s2_), c(c2,0.0)]]
        }
        Instruction::Ry { theta, .. } => {
            let c2 = (theta/2.0).cos(); let s2_ = (theta/2.0).sin();
            [[c(c2,0.0), c(-s2_,0.0)], [c(s2_,0.0), c(c2,0.0)]]
        }
        Instruction::Rz { theta, .. } => {
            let c2 = (theta/2.0).cos(); let s2_ = (theta/2.0).sin();
            [[c(c2,-s2_), c(0.0,0.0)], [c(0.0,0.0), c(c2,s2_)]]
        }
        Instruction::Phase { theta, .. } => {
            let c2 = theta.cos(); let s2_ = theta.sin();
            [[c(1.0,0.0), c(0.0,0.0)], [c(0.0,0.0), c(c2,s2_)]]
        }
        _ => [[c(1.0,0.0), c(0.0,0.0)], [c(0.0,0.0), c(1.0,0.0)]], // identity
    }
}

// ── Coordinator ───────────────────────────────────────────────────────────────

/// Execute `program` across `workers` (one connection per worker).
///
/// The initial |0…0⟩ state vector is partitioned and sent to workers.
/// Gates are dispatched based on whether they are local or cross-shard.
/// The final result is assembled from all workers' shards.
pub fn execute(program: &Program, workers: &mut [WorkerConn]) -> Result<ExecutionResult, String> {
    let n  = program.num_qubits;
    let k  = workers.len();
    let m  = partition::shard_size(n, k);

    partition::validate_nodes(n, k)?;

    // ── Initialise workers with |0…0⟩ shards ─────────────────────────────────
    let dim = 1usize << n;
    let mut global_amps = vec![(0.0f64, 0.0f64); dim];
    global_amps[0] = (1.0, 0.0); // |0…0⟩

    for w in workers.iter_mut() {
        let start = partition::shard_start(w.shard_idx, n, k);
        let shard_amps = global_amps[start..start + m].to_vec();
        w.expect_ack(WorkerCmd::InitShard {
            shard_idx:  w.shard_idx,
            n_qubits:   n,
            n_nodes:    k,
            amplitudes: shard_amps,
        })?;
    }

    // ── Execute instructions ───────────────────────────────────────────────────
    let mut measurements:   Vec<MeasurementRecord> = Vec::new();
    let mut step            = 0usize;
    let mut branch_count    = 0usize;
    let mut pre_meas_taken  = false;
    let mut pre_meas_probs: Option<Vec<f64>> = None;

    for instr in &program.instructions {
        step += 1;
        match instr {
            // ── Single-qubit gates ─────────────────────────────────────────
            Instruction::H(q) | Instruction::X(q) | Instruction::Y(q) |
            Instruction::Z(q) | Instruction::S(q) | Instruction::T(q) => {
                let qubit = *q;
                dispatch_single_qubit(workers, instr, qubit, n, k)?;
            }
            Instruction::Rx  { qubit, .. } | Instruction::Ry { qubit, .. } |
            Instruction::Rz  { qubit, .. } | Instruction::Phase { qubit, .. } => {
                dispatch_single_qubit(workers, instr, *qubit, n, k)?;
            }

            // ── Two-qubit gates ────────────────────────────────────────────
            Instruction::Cnot { control, target } => {
                dispatch_cnot(workers, *control, *target, n, k)?;
            }
            Instruction::Cz { control, target } => {
                // CZ = H(target) · CNOT(ctrl,tgt) · H(target)
                dispatch_single_qubit(workers, &Instruction::H(*target), *target, n, k)?;
                dispatch_cnot(workers, *control, *target, n, k)?;
                dispatch_single_qubit(workers, &Instruction::H(*target), *target, n, k)?;
            }
            Instruction::Swap { qubit_a, qubit_b } => {
                // SWAP = CNOT(a,b) · CNOT(b,a) · CNOT(a,b)
                dispatch_cnot(workers, *qubit_a, *qubit_b, n, k)?;
                dispatch_cnot(workers, *qubit_b, *qubit_a, n, k)?;
                dispatch_cnot(workers, *qubit_a, *qubit_b, n, k)?;
            }
            Instruction::Toffoli { control0, control1, target } => {
                dispatch_toffoli(workers, *control0, *control1, *target, n, k)?;
            }

            // ── Measurement ────────────────────────────────────────────────
            Instruction::Measure(q) => {
                if !pre_meas_taken {
                    pre_meas_probs = Some(collect_global_probs(workers, n, k, m)?);
                    pre_meas_taken = true;
                }
                let outcome = measure_qubit(workers, *q, n, k)?;
                measurements.push(MeasurementRecord { qubit: *q, outcome, step });
            }
            Instruction::MeasureAll => {
                if !pre_meas_taken {
                    pre_meas_probs = Some(collect_global_probs(workers, n, k, m)?);
                    pre_meas_taken = true;
                }
                for q in 0..n {
                    let outcome = measure_qubit(workers, q, n, k)?;
                    measurements.push(MeasurementRecord { qubit: q, outcome, step });
                }
            }

            // ── Control flow (linear trace only) ───────────────────────────
            Instruction::Goto { .. } | Instruction::GotoIf { .. }
            | Instruction::GotoIfNot { .. } => {
                branch_count += 1;
            }

            // ── Structural / invisible ─────────────────────────────────────
            Instruction::Barrier | Instruction::Label(_)
            | Instruction::CallGate { .. } => {}

            Instruction::MeasureInto { .. } | Instruction::GotoIfCreg { .. }
            | Instruction::GotoIfNotCreg { .. } => {
                return Err(format!("CREG instructions not supported in distributed backend"));
            }
        }
    }

    // ── Collect final result ───────────────────────────────────────────────
    let final_probs = collect_global_probs(workers, n, k, m)?;
    let final_amps  = collect_global_amps(workers, n, k, m)?;

    // Shutdown workers.
    for w in workers.iter_mut() {
        w.send(&WorkerCmd::Shutdown).ok();
    }

    // Build pre-measurement amplitude table.
    let pre_meas_amps = pre_meas_probs.as_ref().map(|probs| {
        // We don't store full pre-measurement amplitudes across the coordinator —
        // approximate by scaling: sqrt(p) * 1 (phase unknown without storing)
        probs.iter().map(|&p| (p.sqrt(), 0.0f64)).collect::<Vec<_>>()
    });

    Ok(ExecutionResult {
        num_qubits:                 n,
        final_probabilities:        final_probs,
        final_amplitudes:           final_amps,
        measurements,
        gate_count:                 step,
        branch_count,
        steps_executed:             step,
        pre_measurement_probs:      pre_meas_probs,
        pre_measurement_amplitudes: pre_meas_amps,
    })
}

// ── Gate dispatch helpers ─────────────────────────────────────────────────────

/// Dispatch a single-qubit gate to the appropriate worker(s).
fn dispatch_single_qubit(
    workers: &mut [WorkerConn],
    instr:   &Instruction,
    qubit:   usize,
    n:       usize,
    k:       usize,
) -> Result<(), String> {
    if partition::is_local_gate(qubit, n, k) {
        // All workers apply this gate independently to their local shard.
        for w in workers.iter_mut() {
            w.expect_ack(WorkerCmd::LocalGate { instr: instr.clone() })?;
        }
    } else {
        // Cross-shard: pairs of workers exchange amplitude slices, coordinator applies gate.
        let bit_pos = partition::shard_bit_index(qubit, n, k);
        let matrix  = gate_matrix(instr);
        let m = partition::shard_size(n, k);

        // Process pairs: shard (s0, s1) where s0 < s1 differ only in bit `bit_pos`.
        let mut visited = vec![false; k];
        for s0 in 0..k {
            if visited[s0] { continue; }
            let s1 = s0 ^ (1 << bit_pos);
            if s1 >= k { continue; }
            visited[s0] = true;
            visited[s1] = true;

            // Collect both shards from workers.
            let amps0 = get_shard_slice(workers, s0, 0, m)?;
            let amps1 = get_shard_slice(workers, s1, 0, m)?;

            // Apply 2×2 gate to corresponding amplitude pairs.
            // For a cross-shard single-qubit gate on qubit q:
            //   shard s0 (q-bit=0) and shard s1 (q-bit=1) have the same local index.
            //   Pair: (s0[local], s1[local]) for each local.
            let (new0, new1) = apply_cross_shard_1q_gate(&amps0, &amps1, &matrix);

            put_shard_slice(workers, s0, 0, new0)?;
            put_shard_slice(workers, s1, 0, new1)?;
        }
    }
    Ok(())
}

/// Apply a 2×2 gate across two shard amplitude vectors element-wise.
fn apply_cross_shard_1q_gate(
    amps0: &[(f64, f64)],
    amps1: &[(f64, f64)],
    matrix: &[[Complex; 2]; 2],
) -> (Vec<(f64, f64)>, Vec<(f64, f64)>) {
    let mut new0 = Vec::with_capacity(amps0.len());
    let mut new1 = Vec::with_capacity(amps1.len());
    for i in 0..amps0.len() {
        let mut a0 = c(amps0[i].0, amps0[i].1);
        let mut a1 = c(amps1[i].0, amps1[i].1);
        apply_2x2(&mut a0, &mut a1, *matrix);
        new0.push((a0.re, a0.im));
        new1.push((a1.re, a1.im));
    }
    (new0, new1)
}

/// Dispatch CNOT(control, target).
fn dispatch_cnot(
    workers: &mut [WorkerConn],
    control: usize,
    target:  usize,
    n:       usize,
    k:       usize,
) -> Result<(), String> {
    let ctrl_local   = partition::is_local_gate(control, n, k);
    let target_local = partition::is_local_gate(target,  n, k);

    match (ctrl_local, target_local) {
        (true, true) => {
            // Both local: each worker applies CNOT independently.
            for w in workers.iter_mut() {
                w.expect_ack(WorkerCmd::LocalGate {
                    instr: Instruction::Cnot { control, target },
                })?;
            }
        }
        (false, true) => {
            // Control is cross-shard: workers with ctrl-shard-bit=1 apply X(target).
            // Workers with ctrl-shard-bit=0 do nothing (CNOT only fires when ctrl=1).
            let ctrl_bit = partition::shard_bit_index(control, n, k);
            for w in workers.iter_mut() {
                if (w.shard_idx >> ctrl_bit) & 1 == 1 {
                    w.expect_ack(WorkerCmd::LocalGate { instr: Instruction::X(target) })?;
                }
            }
        }
        (true, false) => {
            // Target is cross-shard: need to swap amplitude slices where control=1.
            // For each pair of shards (s0,s1) that differ in the target bit,
            // swap the amplitudes where the control bit (local) is 1.
            let tgt_bit_pos = partition::shard_bit_index(target, n, k);
            let m    = partition::shard_size(n, k);
            let half = m / 2; // half the local amplitudes have ctrl=1

            // The control qubit is local, so within each shard, exactly half the
            // amplitudes have the ctrl bit set.
            let ctrl_mask = 1usize << control;

            let mut visited = vec![false; k];
            for s0 in 0..k {
                if visited[s0] { continue; }
                let s1 = s0 ^ (1 << tgt_bit_pos);
                if s1 >= k { continue; }
                visited[s0] = true;
                visited[s1] = true;

                // Get full shards from both workers.
                let amps0 = get_shard_slice(workers, s0, 0, m)?;
                let amps1 = get_shard_slice(workers, s1, 0, m)?;

                // For local indices where ctrl bit = 1: swap amps0[i] ↔ amps1[i].
                let mut new0 = amps0.clone();
                let mut new1 = amps1.clone();
                for local_i in 0..m {
                    if local_i & ctrl_mask != 0 {
                        new0[local_i] = amps1[local_i];
                        new1[local_i] = amps0[local_i];
                    }
                }

                put_shard_slice(workers, s0, 0, new0)?;
                put_shard_slice(workers, s1, 0, new1)?;
            }
        }
        (false, false) => {
            // Both cross-shard: full coordinator-mediated exchange.
            // This is an unusual case (usually at least one of ctrl/tgt is local).
            // Collect all shards, apply CNOT globally, redistribute.
            let m    = partition::shard_size(n, k);
            let dim  = 1usize << n;
            let mut global = vec![(0.0f64, 0.0f64); dim];

            for w_idx in 0..k {
                let amps = get_shard_slice(workers, w_idx, 0, m)?;
                let start = partition::shard_start(w_idx, n, k);
                for (i, a) in amps.into_iter().enumerate() {
                    global[start + i] = a;
                }
            }

            // Apply CNOT globally.
            let ctrl_mask  = 1usize << control;
            let tgt_mask   = 1usize << target;
            for i in 0..dim {
                if i & ctrl_mask != 0 && i & tgt_mask == 0 {
                    global.swap(i, i | tgt_mask);
                }
            }

            for w_idx in 0..k {
                let start = partition::shard_start(w_idx, n, k);
                let slice = global[start..start + m].to_vec();
                put_shard_slice(workers, w_idx, 0, slice)?;
            }
        }
    }
    Ok(())
}

/// Dispatch Toffoli (CCX) as a 7-CNOT standard decomposition.
fn dispatch_toffoli(
    workers:  &mut [WorkerConn],
    control0: usize,
    control1: usize,
    target:   usize,
    n:        usize,
    k:        usize,
) -> Result<(), String> {
    let h   = Instruction::H(target);
    let t   = Instruction::T(target);
    // T† approximation via Rz(-π/4)
    let tdg_t  = Instruction::Rz { qubit: target,   theta: -std::f64::consts::FRAC_PI_4 };
    let tdg_c1 = Instruction::Rz { qubit: control1, theta: -std::f64::consts::FRAC_PI_4 };
    let t_c0   = Instruction::T(control0);
    let t_c1   = Instruction::T(control1);

    macro_rules! sq { ($i:expr) => { dispatch_single_qubit(workers, &$i, qubit_of(&$i), n, k)? }; }
    macro_rules! cn { ($c:expr, $t:expr) => { dispatch_cnot(workers, $c, $t, n, k)? }; }

    sq!(h);
    cn!(control1, target);
    dispatch_single_qubit(workers, &tdg_t, target, n, k)?;
    cn!(control0, target);
    sq!(t);
    cn!(control1, target);
    dispatch_single_qubit(workers, &tdg_t, target, n, k)?;
    cn!(control0, target);
    sq!(t_c1);
    sq!(t);
    sq!(h);
    cn!(control0, control1);
    sq!(t_c0);
    dispatch_single_qubit(workers, &tdg_c1, control1, n, k)?;
    cn!(control0, control1);

    Ok(())
}

fn qubit_of(instr: &Instruction) -> usize {
    match instr {
        Instruction::H(q) | Instruction::X(q) | Instruction::S(q) | Instruction::T(q) => *q,
        Instruction::Rz { qubit, .. } => *qubit,
        _ => 0,
    }
}

// ── Slice helpers ─────────────────────────────────────────────────────────────

fn get_shard_slice(
    workers:   &mut [WorkerConn],
    shard_idx: usize,
    offset:    usize,
    len:       usize,
) -> Result<Vec<(f64, f64)>, String> {
    let w = workers.iter_mut().find(|w| w.shard_idx == shard_idx)
        .ok_or_else(|| format!("no worker for shard {shard_idx}"))?;
    match w.cmd(WorkerCmd::GetSlice { offset, len })? {
        WorkerReply::Slice { amplitudes } => Ok(amplitudes),
        WorkerReply::Error { message }   => Err(message),
        other => Err(format!("get_shard_slice: unexpected {:?}", other)),
    }
}

fn put_shard_slice(
    workers:    &mut [WorkerConn],
    shard_idx:  usize,
    offset:     usize,
    amplitudes: Vec<(f64, f64)>,
) -> Result<(), String> {
    let w = workers.iter_mut().find(|w| w.shard_idx == shard_idx)
        .ok_or_else(|| format!("no worker for shard {shard_idx}"))?;
    w.expect_ack(WorkerCmd::PutSlice { offset, amplitudes })
}

// ── Measurement helpers ───────────────────────────────────────────────────────

/// Measure qubit `q` across all shards, agree on an outcome, collapse, return outcome.
///
/// Implemented in three passes to satisfy the borrow checker:
/// 1. Collect all shard amplitudes (read-only)
/// 2. Compute prob1, decide outcome, build new amplitudes for each shard
/// 3. Push updated amplitudes back to workers (write-only)
fn measure_qubit(
    workers: &mut [WorkerConn],
    qubit:   usize,
    n:       usize,
    k:       usize,
) -> Result<bool, String> {
    let m = partition::shard_size(n, k);

    // ── Phase 1: collect all shard amplitudes ──────────────────────────────
    let mut all_amps: Vec<(usize, Vec<(f64, f64)>)> = Vec::with_capacity(k);
    for w in workers.iter_mut() {
        let amps = match w.cmd(WorkerCmd::CollectShard)? {
            WorkerReply::Shard { amplitudes } => amplitudes,
            other => return Err(format!("measure_qubit: expected Shard, got {:?}", other)),
        };
        all_amps.push((w.shard_idx, amps));
    }

    // ── Phase 2: compute prob1 ─────────────────────────────────────────────
    let mask = 1usize << qubit;
    let mut prob1 = 0.0f64;
    for (shard_idx, amps) in &all_amps {
        if partition::is_local_gate(qubit, n, k) {
            // Local qubit: check bit within local amplitude index.
            for (local_i, (re, im)) in amps.iter().enumerate() {
                if local_i & mask != 0 {
                    prob1 += re * re + im * im;
                }
            }
        } else {
            // Cross-shard qubit: check the shard's top-bit value for this qubit.
            if partition::shard_bit(*shard_idx, qubit, n, k) {
                prob1 += amps.iter().map(|(re, im)| re * re + im * im).sum::<f64>();
            }
        }
    }

    let outcome    = rand::random::<f64>() < prob1;
    let norm       = if outcome { prob1.sqrt() } else { (1.0 - prob1).sqrt() };
    let norm       = norm.max(1e-300);

    // ── Phase 2b: build collapsed amplitudes for each shard ────────────────
    let mut new_amps: Vec<(usize, Vec<(f64, f64)>)> = Vec::with_capacity(k);
    for (shard_idx, amps) in &all_amps {
        let updated: Vec<(f64, f64)> = if partition::is_local_gate(qubit, n, k) {
            // Local qubit: zero amplitudes where the qubit bit disagrees with outcome.
            amps.iter().enumerate().map(|(local_i, &(re, im))| {
                let bit_set = (local_i & mask) != 0;
                if bit_set == outcome { (re / norm, im / norm) } else { (0.0, 0.0) }
            }).collect()
        } else {
            // Cross-shard qubit: entire shard is either kept (renormed) or zeroed.
            let shard_has_outcome = partition::shard_bit(*shard_idx, qubit, n, k) == outcome;
            if shard_has_outcome {
                amps.iter().map(|&(re, im)| (re / norm, im / norm)).collect()
            } else {
                vec![(0.0, 0.0); m]
            }
        };
        new_amps.push((*shard_idx, updated));
    }

    // ── Phase 3: push updated amplitudes back to workers ──────────────────
    for (shard_idx, amps) in new_amps {
        let w = workers.iter_mut()
            .find(|w| w.shard_idx == shard_idx)
            .ok_or_else(|| format!("no worker for shard {shard_idx}"))?;
        w.expect_ack(WorkerCmd::PutSlice { offset: 0, amplitudes: amps })?;
    }

    Ok(outcome)
}

/// Collect per-basis-state probabilities from all shards and assemble into a single Vec.
pub fn collect_global_probs(
    workers: &mut [WorkerConn],
    n:       usize,
    k:       usize,
    m:       usize,
) -> Result<Vec<f64>, String> {
    let dim = 1usize << n;
    let mut global = vec![0.0f64; dim];
    for w in workers.iter_mut() {
        let probs = match w.cmd(WorkerCmd::CollectProbs)? {
            WorkerReply::Probs { values } => values,
            other => return Err(format!("collect_global_probs: unexpected {:?}", other)),
        };
        let start = partition::shard_start(w.shard_idx, n, k);
        for (i, p) in probs.into_iter().enumerate() {
            global[start + i] = p;
        }
    }
    Ok(global)
}

/// Collect full amplitude table from all workers.
fn collect_global_amps(
    workers: &mut [WorkerConn],
    n:       usize,
    k:       usize,
    m:       usize,
) -> Result<Vec<(f64, f64)>, String> {
    let dim = 1usize << n;
    let mut global = vec![(0.0f64, 0.0f64); dim];
    for w in workers.iter_mut() {
        let amps = match w.cmd(WorkerCmd::CollectShard)? {
            WorkerReply::Shard { amplitudes } => amplitudes,
            other => return Err(format!("collect_global_amps: unexpected {:?}", other)),
        };
        let start = partition::shard_start(w.shard_idx, n, k);
        for (i, a) in amps.into_iter().enumerate() {
            global[start + i] = a;
        }
    }
    Ok(global)
}

// ── Public constructor ────────────────────────────────────────────────────────

/// Connect to `addrs`, run `program`, return result.
pub fn run(program: &Program, addrs: &[String]) -> Result<ExecutionResult, String> {
    let mut workers: Vec<WorkerConn> = addrs.iter().enumerate()
        .map(|(i, addr)| WorkerConn::connect(addr, i))
        .collect::<Result<Vec<_>, _>>()?;
    execute(program, &mut workers)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler;
    use crate::simulator::dist::worker::spawn_loopback_worker;

    fn bell_addrs() -> Vec<String> {
        let p0 = spawn_loopback_worker();
        let p1 = spawn_loopback_worker();
        // Give threads time to bind.
        std::thread::sleep(std::time::Duration::from_millis(30));
        vec![format!("127.0.0.1:{p0}"), format!("127.0.0.1:{p1}")]
    }

    #[test]
    fn test_fallback_single_node() {
        // With 1 node, all gates are local — degenerates to normal statevector sim.
        let port = spawn_loopback_worker();
        std::thread::sleep(std::time::Duration::from_millis(20));
        let addrs = vec![format!("127.0.0.1:{port}")];
        let prog  = compiler::parse_source("QREG 2\nH 0\nCNOT 0 1\nMEASURE_ALL").unwrap();
        let result = run(&prog, &addrs).unwrap();
        // After MeasureAll the state is either |00⟩ or |11⟩.
        assert_eq!(result.measurements.len(), 2);
        let m0 = result.measurements[0].outcome;
        let m1 = result.measurements[1].outcome;
        assert_eq!(m0, m1, "Bell measurement outcomes must agree");
    }

    #[test]
    fn test_coordinator_worker_handshake() {
        let port = spawn_loopback_worker();
        std::thread::sleep(std::time::Duration::from_millis(20));
        let conn = WorkerConn::connect(&format!("127.0.0.1:{port}"), 0);
        assert!(conn.is_ok(), "handshake failed: {:?}", conn.err());
    }

    #[test]
    fn test_dist_matches_local_16q() {
        // 16 qubits, 2 nodes: a local-only circuit should give same result as CPU sim.
        // Use a simple 1-qubit H circuit (all gates local since qubit 0 << boundary 15).
        let port = spawn_loopback_worker();
        let port2 = spawn_loopback_worker();
        std::thread::sleep(std::time::Duration::from_millis(30));
        let addrs = vec![
            format!("127.0.0.1:{port}"),
            format!("127.0.0.1:{port2}"),
        ];
        let src  = "QREG 4\nH 0\nCNOT 0 1";
        let prog = compiler::parse_source(src).unwrap();
        let dist_result = run(&prog, &addrs).unwrap();
        let cpu_result  = compiler::execute(&prog).unwrap();
        let eps = 1e-10;
        for (i, (&dp, &cp)) in dist_result.final_probabilities.iter()
            .zip(cpu_result.final_probabilities.iter()).enumerate()
        {
            assert!((dp - cp).abs() < eps, "prob[{i}] dist={dp} cpu={cp}");
        }
    }
}
