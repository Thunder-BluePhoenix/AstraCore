/// Simulation backends for AstraCore.
///
/// - `svd`      — Complex Jacobi SVD utility (used internally by MPS)
/// - `mps`      — Matrix Product State simulator (50–200+ qubits)
/// - `clifford` — Stabilizer / Clifford-circuit simulator (unlimited qubits)

pub mod clifford;
pub mod mps;
pub mod svd;

pub use mps::{execute_mps, MpsState};
pub use clifford::execute_clifford;
