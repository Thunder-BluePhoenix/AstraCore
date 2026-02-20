//! # AstraCore
//!
//! High-performance hybrid Classical–Quantum Runtime Engine.
//!
//! ## Quick Start
//!
//! ```rust
//! use astracore::core::Simulator;
//!
//! // Create a 2-qubit simulator
//! let mut sim = Simulator::new(2);
//!
//! // Build a Bell state: (|00⟩ + |11⟩) / √2
//! sim.h(0).cnot(0, 1);
//!
//! // Inspect probabilities (no collapse)
//! let probs = sim.probabilities();
//! println!("P(|00⟩) = {:.4}", probs[0]);
//! println!("P(|11⟩) = {:.4}", probs[3]);
//!
//! // Measure — collapses the state
//! let results = sim.measure_all();
//! ```

pub mod compiler;
pub mod core;
pub mod optimizer;
pub mod runtime;
