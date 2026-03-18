/// Distributed simulation — worker node.
///
/// Each worker owns a contiguous slice ("shard") of the global state vector.
/// It listens on a TCP port, accepts exactly one coordinator connection, and
/// processes commands until `Shutdown` is received.
///
/// The worker is single-connection: after one circuit run the connection closes
/// and the worker process exits (or the in-process task terminates).
use std::io::{BufRead, Write};
use std::net::{TcpListener, TcpStream};

use super::partition;
use super::protocol::{decode_cmd, encode_reply, WorkerCmd, WorkerReply};
use crate::compiler::ir::Instruction;
use crate::core::gates::{
    apply_cnot, apply_cz, apply_single_qubit_gate, apply_swap, apply_toffoli,
    hadamard, pauli_x, pauli_y, pauli_z, phase_gate, rx, ry, rz, s_gate, t_gate,
};
use crate::core::{Complex, StateVector};

// ── WorkerShard ───────────────────────────────────────────────────────────────

/// In-memory state of a worker node during one circuit execution.
struct WorkerShard {
    shard_idx: usize,
    n_qubits:  usize,
    n_nodes:   usize,
    /// Local amplitudes — `state.amplitudes` covers the worker's shard.
    state:     StateVector,
}

impl WorkerShard {
    fn new(shard_idx: usize, n_qubits: usize, n_nodes: usize, amps: Vec<(f64, f64)>) -> Self {
        let local_n = n_qubits - partition::partition_bits(n_nodes);
        let mut state = StateVector::new(local_n);
        for (i, (re, im)) in amps.into_iter().enumerate() {
            if i < state.amplitudes.len() {
                state.amplitudes[i] = Complex { re, im };
            }
        }
        WorkerShard { shard_idx, n_qubits, n_nodes, state }
    }

    /// Apply a local single-qubit gate (qubit index is global but maps to local qubit).
    fn apply_local_gate_instr(&mut self, instr: &Instruction) {
        // Qubit indices in the instruction are global; for local gates they equal
        // the local qubit index (since local qubits are in the low bits).
        match instr {
            Instruction::H(q)     => apply_single_qubit_gate(&mut self.state, &hadamard(),   *q),
            Instruction::X(q)     => apply_single_qubit_gate(&mut self.state, &pauli_x(),    *q),
            Instruction::Y(q)     => apply_single_qubit_gate(&mut self.state, &pauli_y(),    *q),
            Instruction::Z(q)     => apply_single_qubit_gate(&mut self.state, &pauli_z(),    *q),
            Instruction::S(q)     => apply_single_qubit_gate(&mut self.state, &s_gate(),     *q),
            Instruction::T(q)     => apply_single_qubit_gate(&mut self.state, &t_gate(),     *q),
            Instruction::Rx    { qubit, theta } => apply_single_qubit_gate(&mut self.state, &rx(*theta),           *qubit),
            Instruction::Ry    { qubit, theta } => apply_single_qubit_gate(&mut self.state, &ry(*theta),           *qubit),
            Instruction::Rz    { qubit, theta } => apply_single_qubit_gate(&mut self.state, &rz(*theta),           *qubit),
            Instruction::Phase { qubit, theta } => apply_single_qubit_gate(&mut self.state, &phase_gate(*theta),   *qubit),
            Instruction::Cnot  { control, target } => apply_cnot (&mut self.state, *control, *target),
            Instruction::Cz    { control, target } => apply_cz   (&mut self.state, *control, *target),
            Instruction::Swap  { qubit_a, qubit_b } => apply_swap (&mut self.state, *qubit_a, *qubit_b),
            Instruction::Toffoli { control0, control1, target } => {
                apply_toffoli(&mut self.state, *control0, *control1, *target);
            }
            // Structural / non-gate — nothing to do
            _ => {}
        }
    }

    /// Apply amplitude collapse to local qubit `qubit` with `outcome`.
    fn collapse_local(&mut self, qubit: usize, outcome: bool) {
        // Sum of |amplitude|^2 for the selected outcome — used for renormalisation.
        let mask = 1usize << qubit;
        let prob_outcome: f64 = self.state.amplitudes.iter().enumerate()
            .filter(|(i, _)| ((i & mask) != 0) == outcome)
            .map(|(_, a)| a.norm_sq())
            .sum();

        // We don't renormalise per-worker because the total norm is spread across shards.
        // The coordinator applies global renormalisation via PutSlice after collecting probs.
        // Here we just zero out the "wrong outcome" amplitudes.
        let norm = prob_outcome.sqrt().max(1e-300);
        for (i, a) in self.state.amplitudes.iter_mut().enumerate() {
            let bit_set = (i & mask) != 0;
            if bit_set != outcome {
                a.re = 0.0;
                a.im = 0.0;
            } else {
                // Partial normalisation (full norm across all shards is handled by coordinator)
                a.re /= norm;
                a.im /= norm;
            }
        }
    }

    /// Collect per-basis-state probabilities from this shard.
    fn probabilities(&self) -> Vec<f64> {
        self.state.amplitudes.iter().map(|a| a.norm_sq()).collect()
    }

    /// Return all amplitudes as `(re, im)` pairs.
    fn amplitudes(&self) -> Vec<(f64, f64)> {
        self.state.amplitudes.iter().map(|a| (a.re, a.im)).collect()
    }

    /// Return a slice of amplitudes `[offset .. offset+len)`.
    fn slice(&self, offset: usize, len: usize) -> Vec<(f64, f64)> {
        self.state.amplitudes[offset..offset + len]
            .iter()
            .map(|a| (a.re, a.im))
            .collect()
    }

    /// Write (replace) a slice of amplitudes at `offset`.
    fn put_slice(&mut self, offset: usize, amps: Vec<(f64, f64)>) {
        for (i, (re, im)) in amps.into_iter().enumerate() {
            let a = &mut self.state.amplitudes[offset + i];
            a.re = re;
            a.im = im;
        }
    }
}

// ── Command processing ────────────────────────────────────────────────────────

/// Process a single `WorkerCmd`, mutate `shard`, and return the reply.
fn process_cmd(cmd: WorkerCmd, shard: &mut Option<WorkerShard>) -> WorkerReply {
    match cmd {
        WorkerCmd::InitShard { shard_idx, n_qubits, n_nodes, amplitudes } => {
            *shard = Some(WorkerShard::new(shard_idx, n_qubits, n_nodes, amplitudes));
            WorkerReply::Ack
        }

        WorkerCmd::LocalGate { instr } => {
            match shard {
                Some(s) => { s.apply_local_gate_instr(&instr); WorkerReply::Ack }
                None    => WorkerReply::Error { message: "shard not initialised".into() },
            }
        }

        WorkerCmd::PutSlice { offset, amplitudes } => {
            match shard {
                Some(s) => { s.put_slice(offset, amplitudes); WorkerReply::Ack }
                None    => WorkerReply::Error { message: "shard not initialised".into() },
            }
        }

        WorkerCmd::GetSlice { offset, len } => {
            match shard {
                Some(s) => WorkerReply::Slice { amplitudes: s.slice(offset, len) },
                None    => WorkerReply::Error { message: "shard not initialised".into() },
            }
        }

        WorkerCmd::CollapseLocal { qubit, outcome } => {
            match shard {
                Some(s) => { s.collapse_local(qubit, outcome); WorkerReply::Ack }
                None    => WorkerReply::Error { message: "shard not initialised".into() },
            }
        }

        WorkerCmd::CollectProbs => {
            match shard {
                Some(s) => WorkerReply::Probs { values: s.probabilities() },
                None    => WorkerReply::Error { message: "shard not initialised".into() },
            }
        }

        WorkerCmd::CollectShard => {
            match shard {
                Some(s) => WorkerReply::Shard { amplitudes: s.amplitudes() },
                None    => WorkerReply::Error { message: "shard not initialised".into() },
            }
        }

        WorkerCmd::Shutdown => WorkerReply::Ack,
    }
}

// ── TCP server ────────────────────────────────────────────────────────────────

/// Serve one TCP connection synchronously (blocking).
///
/// Reads newline-terminated JSON commands, applies them to `shard`,
/// and writes newline-terminated JSON replies until `Shutdown` or EOF.
fn serve_connection(mut stream: TcpStream) {
    // Send Ready greeting.
    let greeting = encode_reply(&WorkerReply::Ready);
    if stream.write_all(greeting.as_bytes()).is_err() {
        return;
    }

    let reader = std::io::BufReader::new(stream.try_clone().expect("stream clone"));
    let mut writer = stream;
    let mut shard: Option<WorkerShard> = None;

    for line_result in reader.lines() {
        let line = match line_result {
            Ok(l) => l,
            Err(_) => break,
        };
        if line.is_empty() {
            continue;
        }

        let cmd = match decode_cmd(&line) {
            Ok(c)  => c,
            Err(e) => {
                let reply = encode_reply(&WorkerReply::Error { message: e });
                let _ = writer.write_all(reply.as_bytes());
                continue;
            }
        };

        let shutdown = matches!(cmd, WorkerCmd::Shutdown);
        let reply    = process_cmd(cmd, &mut shard);
        let encoded  = encode_reply(&reply);
        let _ = writer.write_all(encoded.as_bytes());

        if shutdown { break; }
    }
}

/// Start a blocking worker server on `port`.
///
/// Listens for one connection, serves it, then returns.
/// Multiple connections can be served by calling this in a loop.
pub fn run_worker_server(port: u16) -> Result<(), String> {
    let listener = TcpListener::bind(format!("0.0.0.0:{port}"))
        .map_err(|e| format!("Cannot bind to port {port}: {e}"))?;

    println!("━━━ AstraCore Worker Node ━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  Listening on 0.0.0.0:{port}");
    println!("  Waiting for coordinator connection…");
    println!("  (Ctrl-C to stop)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    for stream in listener.incoming() {
        match stream {
            Ok(s)  => {
                let peer = s.peer_addr().map(|a| a.to_string()).unwrap_or_default();
                println!("Worker: coordinator connected from {peer}");
                serve_connection(s);
                println!("Worker: connection closed, ready for next run");
            }
            Err(e) => eprintln!("Worker: accept error: {e}"),
        }
    }
    Ok(())
}

// ── In-process worker for tests ───────────────────────────────────────────────

/// Spawn an in-process worker thread bound to an OS-assigned port.
///
/// Returns the actual port it bound to.  The worker serves exactly one
/// connection then the thread exits — suitable for tests.
pub fn spawn_loopback_worker() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").expect("failed to bind loopback worker");
    let port = listener.local_addr().unwrap().port();

    std::thread::spawn(move || {
        if let Ok((stream, _)) = listener.accept() {
            serve_connection(stream);
        }
    });

    port
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::protocol::decode_reply;
    use std::io::{BufRead, Write};
    use std::net::TcpStream;

    fn connect_to_worker(port: u16) -> TcpStream {
        // Retry a few times to let the worker thread bind.
        for _ in 0..20 {
            if let Ok(s) = TcpStream::connect(format!("127.0.0.1:{port}")) {
                return s;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        panic!("could not connect to worker on port {port}");
    }

    fn send_and_recv(writer: &mut TcpStream, reader: &mut std::io::BufReader<TcpStream>, cmd: &WorkerCmd) -> WorkerReply {
        use super::super::protocol::encode_cmd;
        let msg = encode_cmd(cmd);
        writer.write_all(msg.as_bytes()).unwrap();
        let mut line = String::new();
        reader.read_line(&mut line).unwrap();
        decode_reply(line.trim()).unwrap()
    }

    #[test]
    fn worker_handshake_ready() {
        let port = spawn_loopback_worker();
        let stream = connect_to_worker(port);
        let mut reader = std::io::BufReader::new(stream.try_clone().unwrap());
        let mut line = String::new();
        reader.read_line(&mut line).unwrap();
        let reply = decode_reply(line.trim()).unwrap();
        assert!(matches!(reply, WorkerReply::Ready), "expected Ready, got {:?}", reply);
    }

    #[test]
    fn worker_init_and_collect_probs() {
        use super::super::protocol::encode_cmd;

        let port   = spawn_loopback_worker();
        let stream = connect_to_worker(port);
        let mut reader = std::io::BufReader::new(stream.try_clone().unwrap());
        let mut writer = stream;

        // Consume Ready
        let mut line = String::new();
        reader.read_line(&mut line).unwrap();

        // 2 qubits, 2 nodes → local_n=1, M=2 amps per worker
        // Node 0: [|00⟩, |01⟩] → [(1.0,0), (0.0,0)]
        let cmd = WorkerCmd::InitShard {
            shard_idx:  0,
            n_qubits:   2,
            n_nodes:    2,
            amplitudes: vec![(1.0, 0.0), (0.0, 0.0)],
        };
        let msg = encode_cmd(&cmd);
        writer.write_all(msg.as_bytes()).unwrap();
        line.clear();
        reader.read_line(&mut line).unwrap();
        assert!(matches!(decode_reply(line.trim()).unwrap(), WorkerReply::Ack));

        // Apply H on qubit 0 (local, qubit 0 < boundary 1)
        let msg = encode_cmd(&WorkerCmd::LocalGate { instr: Instruction::H(0) });
        writer.write_all(msg.as_bytes()).unwrap();
        line.clear();
        reader.read_line(&mut line).unwrap();
        assert!(matches!(decode_reply(line.trim()).unwrap(), WorkerReply::Ack));

        // Collect probs
        let msg = encode_cmd(&WorkerCmd::CollectProbs);
        writer.write_all(msg.as_bytes()).unwrap();
        line.clear();
        reader.read_line(&mut line).unwrap();
        let probs = match decode_reply(line.trim()).unwrap() {
            WorkerReply::Probs { values } => values,
            other => panic!("expected Probs, got {:?}", other),
        };
        // H|0⟩ = (|0⟩ + |1⟩)/√2 → each prob ≈ 0.5
        assert!((probs[0] - 0.5).abs() < 1e-10, "prob[0]={}", probs[0]);
        assert!((probs[1] - 0.5).abs() < 1e-10, "prob[1]={}", probs[1]);

        // Shutdown
        let msg = encode_cmd(&WorkerCmd::Shutdown);
        writer.write_all(msg.as_bytes()).unwrap();
    }

    #[test]
    fn worker_local_cnot() {
        use super::super::protocol::encode_cmd;

        let port   = spawn_loopback_worker();
        let stream = connect_to_worker(port);
        let mut reader = std::io::BufReader::new(stream.try_clone().unwrap());
        let mut writer = stream;

        let mut line = String::new();
        reader.read_line(&mut line).unwrap(); // Ready

        // 4 qubits, 2 nodes → local_n=3, M=8; init shard 0 in |0000⟩ ground state
        let mut amps = vec![(0.0, 0.0); 8];
        amps[0] = (1.0, 0.0);
        let msg = encode_cmd(&WorkerCmd::InitShard {
            shard_idx: 0, n_qubits: 4, n_nodes: 2, amplitudes: amps,
        });
        writer.write_all(msg.as_bytes()).unwrap();
        line.clear(); reader.read_line(&mut line).unwrap(); // Ack

        // H on qubit 0, then CNOT(0→1) — both local (qubits 0,1 < boundary 3)
        let msg = encode_cmd(&WorkerCmd::LocalGate { instr: Instruction::H(0) });
        writer.write_all(msg.as_bytes()).unwrap();
        line.clear(); reader.read_line(&mut line).unwrap();

        let msg = encode_cmd(&WorkerCmd::LocalGate { instr: Instruction::Cnot { control: 0, target: 1 } });
        writer.write_all(msg.as_bytes()).unwrap();
        line.clear(); reader.read_line(&mut line).unwrap();

        // Collect
        let msg = encode_cmd(&WorkerCmd::CollectProbs);
        writer.write_all(msg.as_bytes()).unwrap();
        line.clear(); reader.read_line(&mut line).unwrap();
        let probs = match decode_reply(line.trim()).unwrap() {
            WorkerReply::Probs { values } => values,
            other => panic!("{:?}", other),
        };
        // Local Bell: |00⟩ and |11⟩ in shard 0 → index 0 and index 3
        assert!((probs[0] - 0.5).abs() < 1e-10, "probs[0]={}", probs[0]);
        assert!((probs[3] - 0.5).abs() < 1e-10, "probs[3]={}", probs[3]);

        let msg = encode_cmd(&WorkerCmd::Shutdown);
        writer.write_all(msg.as_bytes()).unwrap();
    }
}
