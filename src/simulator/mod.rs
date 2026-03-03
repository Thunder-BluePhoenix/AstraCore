/// Simulation backends for AstraCore.
///
/// - `svd`      — Complex Jacobi SVD utility (used internally by MPS)
/// - `mps`      — Matrix Product State simulator (50–200+ qubits)
/// - `clifford` — Stabilizer / Clifford-circuit simulator (unlimited qubits)
/// - `sparse`   — Sparse statevector (HashMap-backed, efficient for low-entanglement circuits)

pub mod clifford;
pub mod mps;
pub mod sparse;
pub mod svd;

pub use clifford::execute_clifford;
pub use mps::{execute_mps, MpsState};
pub use sparse::{execute_sparse, SparseState};
