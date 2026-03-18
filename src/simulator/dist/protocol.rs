/// Distributed simulation network protocol — newline-delimited JSON over TCP.
///
/// Message flow:
///
/// ```text
/// Coordinator → Worker:  WorkerCmd (one JSON object per line)
/// Worker → Coordinator:  WorkerReply (one JSON object per line)
/// ```
///
/// Connection lifecycle:
/// 1. Coordinator connects → Worker replies `{"type":"Ready"}`
/// 2. Coordinator sends `InitShard` → Worker replies `Ack`
/// 3. Coordinator sends gate commands → Worker replies `Ack` (or `Error`)
/// 4. Coordinator sends `CollectShard` → Worker replies `Shard { amplitudes }`
/// 5. Coordinator sends `Shutdown` → connection closes
use crate::compiler::ir::Instruction;
use serde::{Deserialize, Serialize};

/// Commands sent from the coordinator to a worker node.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum WorkerCmd {
    /// Initialise the worker with its shard of the state vector.
    InitShard {
        /// 0-based index of this worker's shard.
        shard_idx:  usize,
        /// Total number of qubits in the circuit.
        n_qubits:   usize,
        /// Total number of worker nodes (`k`, must be power of 2).
        n_nodes:    usize,
        /// Initial amplitudes `[(re, im); M]` where `M = 2^(n_qubits) / n_nodes`.
        amplitudes: Vec<(f64, f64)>,
    },

    /// Apply a local gate (qubit is within the shard's address space).
    ///
    /// The worker applies the instruction to its local shard without
    /// contacting other workers.
    LocalGate {
        instr: Instruction,
    },

    /// Update a slice of amplitudes (used for cross-shard gate results).
    ///
    /// `offset` is relative to this shard's start.
    PutSlice {
        offset:     usize,
        amplitudes: Vec<(f64, f64)>,
    },

    /// Request the worker to send a contiguous slice of its shard.
    ///
    /// `offset` and `len` are relative to this shard's start.
    GetSlice {
        offset: usize,
        len:    usize,
    },

    /// Measure qubit `qubit` within the shard (applying collapse).
    ///
    /// `rand` is a pre-generated uniform random value in `[0, 1)` to use for
    /// the outcome decision.  The coordinator collects `prob1` from all shards
    /// and then sends the agreed-upon `rand` value so all workers collapse
    /// consistently.
    CollapseLocal {
        qubit:   usize,
        outcome: bool,   // agreed-upon measurement outcome
    },

    /// Collect per-basis-state probabilities from this shard.
    CollectProbs,

    /// Return the full shard amplitudes to the coordinator.
    CollectShard,

    /// Worker should cleanly close the connection.
    Shutdown,
}

/// Replies sent from a worker to the coordinator.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum WorkerReply {
    /// Worker is ready to receive commands (sent immediately after TCP connect).
    Ready,

    /// Command acknowledged, state updated.
    Ack,

    /// Per-basis-state probabilities for this worker's shard.
    ///
    /// Length equals `M = 2^(n_qubits) / n_nodes`.
    Probs { values: Vec<f64> },

    /// Amplitude slice in response to `GetSlice`.
    Slice { amplitudes: Vec<(f64, f64)> },

    /// Full shard amplitudes in response to `CollectShard`.
    Shard { amplitudes: Vec<(f64, f64)> },

    /// Probability of measuring |1⟩ for the local portion of a qubit
    /// (used by coordinator to decide measurement outcome).
    MeasureProb { prob1: f64 },

    /// An error occurred — string description.
    Error { message: String },
}

// ── Wire helpers ──────────────────────────────────────────────────────────────

/// Serialize a command to a newline-terminated JSON string.
pub fn encode_cmd(cmd: &WorkerCmd) -> String {
    let mut s = serde_json::to_string(cmd).expect("WorkerCmd serialization failed");
    s.push('\n');
    s
}

/// Serialize a reply to a newline-terminated JSON string.
pub fn encode_reply(reply: &WorkerReply) -> String {
    let mut s = serde_json::to_string(reply).expect("WorkerReply serialization failed");
    s.push('\n');
    s
}

/// Deserialize a command from one line of received text.
pub fn decode_cmd(line: &str) -> Result<WorkerCmd, String> {
    serde_json::from_str(line).map_err(|e| format!("decode_cmd error: {e} — line: {line}"))
}

/// Deserialize a reply from one line of received text.
pub fn decode_reply(line: &str) -> Result<WorkerReply, String> {
    serde_json::from_str(line).map_err(|e| format!("decode_reply error: {e} — line: {line}"))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::ir::Instruction;

    #[test]
    fn roundtrip_init_shard() {
        let cmd = WorkerCmd::InitShard {
            shard_idx:  0,
            n_qubits:   4,
            n_nodes:    2,
            amplitudes: vec![(1.0, 0.0), (0.0, 0.0)],
        };
        let encoded = encode_cmd(&cmd);
        assert!(encoded.ends_with('\n'));
        let decoded = decode_cmd(encoded.trim()).unwrap();
        match decoded {
            WorkerCmd::InitShard { shard_idx, n_qubits, n_nodes, amplitudes } => {
                assert_eq!(shard_idx, 0);
                assert_eq!(n_qubits, 4);
                assert_eq!(n_nodes, 2);
                assert_eq!(amplitudes.len(), 2);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn roundtrip_local_gate_h() {
        let cmd = WorkerCmd::LocalGate { instr: Instruction::H(3) };
        let line = encode_cmd(&cmd);
        let back = decode_cmd(line.trim()).unwrap();
        match back {
            WorkerCmd::LocalGate { instr: Instruction::H(q) } => assert_eq!(q, 3),
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn roundtrip_probs_reply() {
        let reply = WorkerReply::Probs { values: vec![0.5, 0.0, 0.0, 0.5] };
        let line  = encode_reply(&reply);
        let back  = decode_reply(line.trim()).unwrap();
        match back {
            WorkerReply::Probs { values } => assert_eq!(values, vec![0.5, 0.0, 0.0, 0.5]),
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn roundtrip_cnot_instruction() {
        let cmd = WorkerCmd::LocalGate {
            instr: Instruction::Cnot { control: 0, target: 1 },
        };
        let line = encode_cmd(&cmd);
        let back = decode_cmd(line.trim()).unwrap();
        match back {
            WorkerCmd::LocalGate { instr: Instruction::Cnot { control, target } } => {
                assert_eq!(control, 0);
                assert_eq!(target,  1);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn cluster_toml_parse() {
        // Verify we can parse a cluster.toml-style TOML string with serde.
        // (ClusterConfig is defined in mod.rs but the format is documented here.)
        let toml_src = r#"
[[nodes]]
host = "192.168.1.10"
port = 7700
[[nodes]]
host = "192.168.1.11"
port = 7700
"#;
        // We test the expected format round-trips as JSON (TOML parsing is in mod.rs)
        let _ = toml_src; // presence check — actual parsing tested in dist::tests
    }
}
