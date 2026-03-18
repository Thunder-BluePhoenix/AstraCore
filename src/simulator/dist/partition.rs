/// State vector partitioning math for distributed simulation.
///
/// With `k = 2^p` nodes and `n` qubits:
/// - Each node holds `M = 2^(n-p)` amplitudes.
/// - Node `i` owns amplitudes whose global index has the top `p` bits equal to `i`.
/// - The *partition qubit* boundary is `n - p`: qubits `[0 .. n-p)` are local to each
///   node; qubits `[n-p .. n)` span node boundaries.
///
/// This is a "high-qubit" partitioning: the most significant qubit bits determine
/// node ownership, which means single-qubit gates on low-index qubits require no
/// inter-node communication.

/// Number of top-qubit bits used to address nodes (`log2(n_nodes)`).
///
/// Panics if `n_nodes` is not a power of two or is zero.
pub fn partition_bits(n_nodes: usize) -> usize {
    assert!(n_nodes.is_power_of_two() && n_nodes > 0, "n_nodes must be a positive power of 2");
    n_nodes.trailing_zeros() as usize
}

/// Return the number of amplitudes each node holds.
pub fn shard_size(n_qubits: usize, n_nodes: usize) -> usize {
    debug_assert!(n_nodes.is_power_of_two());
    (1usize << n_qubits) / n_nodes
}

/// Return the global amplitude index at which node `shard_idx`'s shard starts.
pub fn shard_start(shard_idx: usize, n_qubits: usize, n_nodes: usize) -> usize {
    shard_idx * shard_size(n_qubits, n_nodes)
}

/// Return which node owns global amplitude index `amp_idx`.
pub fn shard_for(amp_idx: usize, n_qubits: usize, n_nodes: usize) -> usize {
    amp_idx / shard_size(n_qubits, n_nodes)
}

/// Return `true` if a gate on `qubit` is local to each shard (no communication needed).
///
/// A gate is local when `qubit < n_qubits - partition_bits(n_nodes)`.
/// That means both amplitude indices in the gate's pair have the same top `p` bits
/// and therefore live on the same node.
pub fn is_local_gate(qubit: usize, n_qubits: usize, n_nodes: usize) -> bool {
    let p = partition_bits(n_nodes);
    qubit < n_qubits.saturating_sub(p)
}

/// For a cross-shard gate on `qubit` (which is in the upper / node-address bits),
/// return the *shard-level bit index* — i.e., which bit of the `shard_idx` corresponds
/// to `qubit`.
///
/// `qubit` must be ≥ `n_qubits - p` (i.e., a cross-shard qubit).
/// Returns `qubit - (n_qubits - p)`.
pub fn shard_bit_index(qubit: usize, n_qubits: usize, n_nodes: usize) -> usize {
    let p = partition_bits(n_nodes);
    let boundary = n_qubits - p;
    debug_assert!(qubit >= boundary, "qubit {} is local (boundary {})", qubit, boundary);
    qubit - boundary
}

/// Return the value of the shard-address bit for `qubit` on node `shard_idx`.
pub fn shard_bit(shard_idx: usize, qubit: usize, n_qubits: usize, n_nodes: usize) -> bool {
    let bit_pos = shard_bit_index(qubit, n_qubits, n_nodes);
    (shard_idx >> bit_pos) & 1 == 1
}

/// For a single-qubit gate on cross-shard `qubit`, return the partner shard index
/// for `shard_idx` (the shard with the opposite value of that qubit bit).
pub fn partner_shard(shard_idx: usize, qubit: usize, n_qubits: usize, n_nodes: usize) -> usize {
    let bit_pos = shard_bit_index(qubit, n_qubits, n_nodes);
    shard_idx ^ (1 << bit_pos)
}

/// Validate that `n_nodes` is a valid node count (positive power of two ≤ 2^n_qubits).
pub fn validate_nodes(n_qubits: usize, n_nodes: usize) -> Result<(), String> {
    if n_nodes == 0 {
        return Err("n_nodes must be at least 1".to_string());
    }
    if !n_nodes.is_power_of_two() {
        return Err(format!("n_nodes ({n_nodes}) must be a power of 2"));
    }
    if n_nodes > (1usize << n_qubits) {
        return Err(format!(
            "n_nodes ({n_nodes}) exceeds number of amplitudes ({})",
            1usize << n_qubits
        ));
    }
    Ok(())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_partition_bits() {
        assert_eq!(partition_bits(1),  0);
        assert_eq!(partition_bits(2),  1);
        assert_eq!(partition_bits(4),  2);
        assert_eq!(partition_bits(8),  3);
        assert_eq!(partition_bits(16), 4);
    }

    #[test]
    fn test_shard_size() {
        assert_eq!(shard_size(4, 2), 8);   // 16 / 2 = 8
        assert_eq!(shard_size(4, 4), 4);   // 16 / 4 = 4
        assert_eq!(shard_size(8, 4), 64);  // 256 / 4 = 64
    }

    #[test]
    fn test_shard_for() {
        // 4 qubits, 2 nodes: M=8 → node 0: [0..8), node 1: [8..16)
        assert_eq!(shard_for(0,  4, 2), 0);
        assert_eq!(shard_for(7,  4, 2), 0);
        assert_eq!(shard_for(8,  4, 2), 1);
        assert_eq!(shard_for(15, 4, 2), 1);
    }

    #[test]
    fn test_is_local_gate() {
        // 4 qubits, 2 nodes → p=1, boundary=3
        // Qubits 0,1,2 are local; qubit 3 is cross-shard
        assert!( is_local_gate(0, 4, 2));
        assert!( is_local_gate(1, 4, 2));
        assert!( is_local_gate(2, 4, 2));
        assert!(!is_local_gate(3, 4, 2));
        // 4 qubits, 4 nodes → p=2, boundary=2
        // Qubits 0,1 local; 2,3 cross-shard
        assert!( is_local_gate(0, 4, 4));
        assert!( is_local_gate(1, 4, 4));
        assert!(!is_local_gate(2, 4, 4));
        assert!(!is_local_gate(3, 4, 4));
    }

    #[test]
    fn test_partner_shard() {
        // 4 qubits, 2 nodes → p=1, boundary=3 → shard_bit_index(3)=0
        // partner of node 0 for qubit 3 is node 1, vice versa
        assert_eq!(partner_shard(0, 3, 4, 2), 1);
        assert_eq!(partner_shard(1, 3, 4, 2), 0);
        // 4 qubits, 4 nodes → p=2
        // qubit 3 → shard_bit_index=1 → partner flips bit 1
        assert_eq!(partner_shard(0, 3, 4, 4), 2);
        assert_eq!(partner_shard(1, 3, 4, 4), 3);
        assert_eq!(partner_shard(2, 3, 4, 4), 0);
        assert_eq!(partner_shard(3, 3, 4, 4), 1);
    }

    #[test]
    fn test_partition_math_power_of_2() {
        // All shards collectively cover all 2^n amplitudes exactly once
        let n = 4;
        let k = 4;
        let m = shard_size(n, k);
        let mut covered = vec![false; 1 << n];
        for s in 0..k {
            let start = shard_start(s, n, k);
            for local in 0..m {
                covered[start + local] = true;
            }
        }
        assert!(covered.iter().all(|&v| v), "shards don't cover all amplitudes");
    }

    #[test]
    fn test_validate_nodes_errors() {
        assert!(validate_nodes(4, 0).is_err());
        assert!(validate_nodes(4, 3).is_err());   // not power of 2
        assert!(validate_nodes(2, 8).is_err());   // more nodes than amplitudes
    }

    #[test]
    fn test_validate_nodes_ok() {
        assert!(validate_nodes(4, 1).is_ok());
        assert!(validate_nodes(4, 2).is_ok());
        assert!(validate_nodes(4, 4).is_ok());
        assert!(validate_nodes(10, 8).is_ok());
    }
}
