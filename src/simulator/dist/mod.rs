/// Distributed multi-node statevector simulation for AstraCore.
///
/// Splits the `2^n` state vector across `k` worker nodes (k must be a power of 2),
/// where each node holds `2^n / k` amplitudes.
///
/// # Quick-start
///
/// **Worker node** (run on each remote machine, or localhost for testing):
/// ```bash
/// astracore worker --port 7700
/// ```
///
/// **Coordinator** (your machine):
/// ```bash
/// astracore run ghz_32q.aql --dist --nodes "host1:7700,host2:7700"
/// ```
///
/// # Scaling
///
/// | Nodes | RAM/node | Max exact qubits |
/// |------:|:--------:|:----------------:|
/// |   1   |  16 GB   |       30         |
/// |   2   |  16 GB   |       31         |
/// |   4   |  16 GB   |       32         |
/// |   8   |  16 GB   |       33         |
/// |  16   |  16 GB   |       34         |
///
/// # Architecture
///
/// See [`coordinator`], [`worker`], [`protocol`], and [`partition`] sub-modules.

pub mod coordinator;
pub mod partition;
pub mod protocol;
pub mod worker;

use crate::compiler::ir::Program;
use crate::runtime::ExecutionResult;

/// Parse a comma-separated list of `"host:port"` entries.
///
/// Returns an error if the string is empty or any entry lacks a port.
pub fn parse_nodes(nodes_str: &str) -> Result<Vec<String>, String> {
    let addrs: Vec<String> = nodes_str
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(String::from)
        .collect();

    if addrs.is_empty() {
        return Err("--nodes must specify at least one 'host:port' address".to_string());
    }

    for addr in &addrs {
        if !addr.contains(':') {
            return Err(format!(
                "invalid node address '{}' — expected 'host:port' format",
                addr
            ));
        }
    }

    Ok(addrs)
}

/// Execute a circuit on a set of worker nodes given as `"host:port"` strings.
///
/// The number of nodes must be a power of 2 and ≤ `2^n_qubits`.
/// Workers must already be listening when this is called.
///
/// # Errors
/// Returns `Err` if:
/// - `n_nodes` is not a power of 2
/// - Connection to any worker fails
/// - A protocol error occurs during execution
pub fn execute_distributed(program: &Program, nodes: &[String]) -> Result<ExecutionResult, String> {
    partition::validate_nodes(program.num_qubits, nodes.len())?;
    coordinator::run(program, nodes)
}

/// Cluster configuration loaded from a `cluster.toml` file.
///
/// Format:
/// ```toml
/// [[nodes]]
/// host = "192.168.1.10"
/// port = 7700
///
/// [[nodes]]
/// host = "192.168.1.11"
/// port = 7700
/// ```
#[derive(Debug, serde::Deserialize)]
pub struct ClusterConfig {
    pub nodes: Vec<ClusterNode>,
}

/// A single entry in `cluster.toml`.
#[derive(Debug, serde::Deserialize)]
pub struct ClusterNode {
    pub host: String,
    pub port: u16,
}

impl ClusterConfig {
    /// Parse a `cluster.toml` string.
    pub fn from_str(toml_src: &str) -> Result<Self, String> {
        // Manual TOML parsing to avoid adding a toml dependency:
        // We use a simple line-by-line parser for the documented format.
        let mut nodes = Vec::new();
        let mut host: Option<String> = None;
        let mut port: Option<u16>   = None;

        for line in toml_src.lines() {
            let line = line.trim();
            if line == "[[nodes]]" {
                if let (Some(h), Some(p)) = (host.take(), port.take()) {
                    nodes.push(ClusterNode { host: h, port: p });
                }
            } else if let Some(rest) = line.strip_prefix("host") {
                let val = rest.trim_start_matches([' ', '=', '"']).trim_end_matches('"');
                host = Some(val.to_string());
            } else if let Some(rest) = line.strip_prefix("port") {
                let val = rest.trim_start_matches([' ', '=']).trim();
                port = val.parse().ok();
            }
        }
        // Flush last entry.
        if let (Some(h), Some(p)) = (host, port) {
            nodes.push(ClusterNode { host: h, port: p });
        }

        if nodes.is_empty() {
            return Err("cluster.toml contains no [[nodes]] entries".to_string());
        }
        Ok(ClusterConfig { nodes })
    }

    /// Convert to `"host:port"` address strings.
    pub fn addresses(&self) -> Vec<String> {
        self.nodes.iter().map(|n| format!("{}:{}", n.host, n.port)).collect()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler;
    use crate::simulator::dist::worker::spawn_loopback_worker;

    // ── parse_nodes ────────────────────────────────────────────────────────

    #[test]
    fn parse_nodes_single() {
        let addrs = parse_nodes("localhost:7700").unwrap();
        assert_eq!(addrs, vec!["localhost:7700"]);
    }

    #[test]
    fn parse_nodes_multiple() {
        let addrs = parse_nodes("host1:7700,host2:7700,host3:7700").unwrap();
        assert_eq!(addrs.len(), 3);
        assert_eq!(addrs[1], "host2:7700");
    }

    #[test]
    fn parse_nodes_empty_err() {
        assert!(parse_nodes("").is_err());
    }

    #[test]
    fn parse_nodes_missing_port_err() {
        assert!(parse_nodes("localhost").is_err());
    }

    // ── cluster.toml parsing ────────────────────────────────────────────────

    #[test]
    fn test_cluster_toml_parse() {
        let toml = r#"
[[nodes]]
host = "192.168.1.10"
port = 7700

[[nodes]]
host = "192.168.1.11"
port = 7701
"#;
        let cfg = ClusterConfig::from_str(toml).unwrap();
        assert_eq!(cfg.nodes.len(), 2);
        assert_eq!(cfg.nodes[0].host, "192.168.1.10");
        assert_eq!(cfg.nodes[0].port, 7700);
        assert_eq!(cfg.nodes[1].host, "192.168.1.11");
        assert_eq!(cfg.nodes[1].port, 7701);
    }

    #[test]
    fn cluster_config_addresses() {
        let toml = "[[nodes]]\nhost = \"10.0.0.1\"\nport = 8000\n";
        let cfg = ClusterConfig::from_str(toml).unwrap();
        assert_eq!(cfg.addresses(), vec!["10.0.0.1:8000"]);
    }

    // ── execute_distributed (loopback) ──────────────────────────────────────

    fn loopback_addrs(k: usize) -> Vec<String> {
        let ports: Vec<u16> = (0..k).map(|_| spawn_loopback_worker()).collect();
        std::thread::sleep(std::time::Duration::from_millis(20 + 10 * k as u64));
        ports.into_iter().map(|p| format!("127.0.0.1:{p}")).collect()
    }

    #[test]
    fn test_bell_state_2_nodes_loopback() {
        let addrs = loopback_addrs(2);
        let prog  = compiler::parse_source("QREG 2\nH 0\nCNOT 0 1").unwrap();
        let result = execute_distributed(&prog, &addrs).unwrap();
        let probs = &result.final_probabilities;
        assert_eq!(probs.len(), 4);
        assert!((probs[0] - 0.5).abs() < 1e-10, "|00⟩={}", probs[0]);
        assert!((probs[3] - 0.5).abs() < 1e-10, "|11⟩={}", probs[3]);
        assert!(probs[1] < 1e-10, "|01⟩ should be ~0");
        assert!(probs[2] < 1e-10, "|10⟩ should be ~0");
    }

    #[test]
    fn test_ghz_4q_2_nodes() {
        // 4-qubit GHZ: H(0), CNOT(0,1), CNOT(0,2), CNOT(0,3)
        let addrs = loopback_addrs(2);
        let prog  = compiler::parse_source(
            "QREG 4\nH 0\nCNOT 0 1\nCNOT 0 2\nCNOT 0 3"
        ).unwrap();
        let result = execute_distributed(&prog, &addrs).unwrap();
        let probs  = &result.final_probabilities;
        // Should be |0000⟩ and |1111⟩ each with prob ≈ 0.5
        assert!((probs[0]  - 0.5).abs() < 1e-9, "|0000⟩={}", probs[0]);
        assert!((probs[15] - 0.5).abs() < 1e-9, "|1111⟩={}", probs[15]);
    }

    #[test]
    fn test_measurement_aggregation_across_shards() {
        let addrs = loopback_addrs(2);
        let prog  = compiler::parse_source("QREG 2\nH 0\nCNOT 0 1\nMEASURE_ALL").unwrap();
        let result = execute_distributed(&prog, &addrs).unwrap();
        assert_eq!(result.measurements.len(), 2);
        let m0 = result.measurements[0].outcome;
        let m1 = result.measurements[1].outcome;
        assert_eq!(m0, m1, "Bell pair must have correlated measurements");
    }

    #[test]
    fn test_validate_nodes_error() {
        let prog  = compiler::parse_source("QREG 2\nH 0").unwrap();
        let addrs = vec!["127.0.0.1:1".to_string(), "127.0.0.1:2".to_string(),
                         "127.0.0.1:3".to_string()]; // 3 nodes — not power of 2
        assert!(execute_distributed(&prog, &addrs).is_err());
    }
}
