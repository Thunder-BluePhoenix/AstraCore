/// AstraCore Visualization Dashboard
///
/// Three output backends — pick the one that fits your workflow:
///
/// | Backend  | Command                          | Description                              |
/// |----------|----------------------------------|------------------------------------------|
/// | TUI      | `astracore dash <file.aql>`      | Live terminal UI (ratatui + crossterm)   |
/// | HTML     | `astracore report <file.aql>`    | Standalone HTML file (Chart.js)          |
/// | Server   | `astracore serve <file.aql>`     | Local HTTP server at localhost:8080      |
///
/// All three backends consume the same [`DashboardData`] struct, which bundles
/// the circuit analysis and execution result for a single AQL program.
pub mod html;
pub mod server;
pub mod tui;

pub use html::generate_report;
pub use server::serve;
pub use tui::run_tui;

use crate::compiler::CircuitAnalysis;
use crate::runtime::ExecutionResult;

// ── Shared data model ─────────────────────────────────────────────────────

/// All data needed to render any dashboard backend.
pub struct DashboardData {
    /// Source file path (display only).
    pub source_path: String,
    /// Static circuit analysis (gate counts, depth, utilization…).
    pub analysis: CircuitAnalysis,
    /// Execution result (probabilities, measurements, timing…).
    pub result: ExecutionResult,
}

impl DashboardData {
    /// Returns `(binary_label, probability)` pairs for states above `threshold`.
    ///
    /// Uses pre-measurement probabilities when available so the chart shows
    /// the quantum state before collapse.
    pub fn significant_states(&self, threshold: f64) -> Vec<(String, f64)> {
        let n = self.result.num_qubits;
        let probs = self.result.pre_measurement_probs.as_deref()
            .unwrap_or(&self.result.final_probabilities);
        (0..(1usize << n))
            .filter_map(|i| {
                let p = probs[i];
                if p > threshold {
                    Some((format!("{:0>width$b}", i, width = n), p))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Returns ALL `(binary_label, probability)` pairs (no threshold).
    pub fn all_states(&self) -> Vec<(String, f64)> {
        let n = self.result.num_qubits;
        let probs = self.result.pre_measurement_probs.as_deref()
            .unwrap_or(&self.result.final_probabilities);
        (0..(1usize << n))
            .map(|i| (format!("{:0>width$b}", i, width = n), probs[i]))
            .collect()
    }

    /// Gate histogram sorted by count descending, then gate name ascending.
    pub fn sorted_gate_histogram(&self) -> Vec<(String, usize)> {
        let mut entries: Vec<(String, usize)> = self.analysis.gate_histogram.iter()
            .map(|(k, &v)| (k.clone(), v))
            .collect();
        entries.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        entries
    }

    /// State labels and probabilities capped at `max_states` entries.
    ///
    /// For small circuits (≤ 5 qubits) returns all states.
    /// For larger circuits returns only states with probability > 1e-4,
    /// further capped at `max_states`.
    pub fn display_states(&self, max_states: usize) -> Vec<(String, f64)> {
        if self.result.num_qubits <= 5 {
            let mut s = self.all_states();
            s.truncate(max_states);
            s
        } else {
            let mut s = self.significant_states(1e-4);
            s.truncate(max_states);
            s
        }
    }
}
