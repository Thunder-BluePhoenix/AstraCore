/// Simulation backends for AstraCore.
///
/// - `svd`      — Complex Jacobi SVD utility (used internally by MPS)
/// - `mps`      — Matrix Product State simulator (50–200+ qubits)
/// - `clifford` — Stabilizer / Clifford-circuit simulator (unlimited qubits)
/// - `sparse`   — Sparse statevector (HashMap-backed, efficient for low-entanglement circuits)
/// - `gpu`      — GPU statevector (wgpu / CUDA, behind `--features wgpu/cuda`)
/// - `dist`     — Distributed multi-node statevector (behind `--features dist`)

pub mod clifford;
pub mod dist;
pub mod gpu;
pub mod mps;
pub mod sparse;
pub mod svd;

pub use clifford::execute_clifford;
pub use dist::{execute_distributed, parse_nodes, ClusterConfig};
pub use gpu::list_devices as list_gpu_devices;
pub use mps::{execute_mps, MpsState};
pub use sparse::{execute_sparse, SparseState};

#[cfg(any(feature = "wgpu", feature = "cuda"))]
pub use gpu::execute_gpu;

#[cfg(feature = "wgpu")]
pub use gpu::execute_wgpu;

#[cfg(feature = "cuda")]
pub use gpu::execute_cuda;
