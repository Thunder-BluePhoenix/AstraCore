/// AstraCore Gate Optimizer — Phase 5
///
/// Transforms an AQL instruction sequence to an equivalent but shorter or
/// cheaper sequence using algebraic identities.
///
/// Current passes (all in one streaming scan):
///   1. Zero-angle rotation removal  — Rx/Ry/Rz/Phase with angle ≈ 0 (mod 2π)
///   2. Self-inverse cancellation    — H·H, X·X, Y·Y, Z·Z → identity
///   3. Pauli product merging        — S·S → Z,  T·T → S
///   4. Rotation gate merging        — Rx(a)·Rx(b) → Rx(a+b),  and Ry, Rz, Phase
///
/// The optimizer works only on adjacent gates on the same qubit.
/// BARRIER instructions act as full optimization fences.
/// Programs containing control flow (LABEL/GOTO/IF/IFNOT) are returned
/// unchanged — branching invalidates qubit-adjacency analysis.
///
/// Multi-pass: the scan repeats until no further reductions are found.
pub mod peephole;

pub use peephole::{optimize, OptimizationStats};
