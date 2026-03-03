/// Shot-based sampling for AstraCore.
///
/// Runs a circuit N times independently, accumulating measurement bitstring counts
/// into a histogram.  This provides expectation-value estimation without computing
/// the full probability distribution.
///
/// # Scaling
/// O(shots × circuit_cost) — completely independent of 2ⁿ state-space size.
///
/// # Usage (library)
/// ```rust,ignore
/// use astracore::runtime::{run_shots, ShotResult};
///
/// let prog = astracore::compiler::parse_source("QREG 2\nH 0\nCNOT 0 1\nMEASURE_ALL").unwrap();
/// let shots = run_shots(&prog, 1000).unwrap();
/// shots.print_histogram();
/// ```
use std::collections::HashMap;
use crate::compiler::{AqlError, ir::Program};
use super::execute;

// ── ShotResult ────────────────────────────────────────────────────────────

/// Result of running a circuit for `n_shots` independent shots.
#[derive(Debug)]
pub struct ShotResult {
    /// Measurement bitstring → count.
    /// Key: e.g. `"010"` (qubit 0 first, MSB-right convention like bitstring()).
    pub counts: HashMap<String, usize>,
    /// Total number of shots executed.
    pub n_shots: usize,
    /// Number of qubits in the circuit.
    pub n_qubits: usize,
}

impl ShotResult {
    /// Probability estimate for a given bitstring (count / n_shots).
    pub fn prob(&self, bitstring: &str) -> f64 {
        self.counts.get(bitstring).copied().unwrap_or(0) as f64 / self.n_shots as f64
    }

    /// Return counts sorted by bitstring (lexicographic order).
    pub fn sorted_counts(&self) -> Vec<(&str, usize)> {
        let mut v: Vec<(&str, usize)> = self.counts.iter()
            .map(|(k, &v)| (k.as_str(), v))
            .collect();
        v.sort_by(|a, b| a.0.cmp(b.0));
        v
    }

    /// Print a text histogram to stdout.
    pub fn print_histogram(&self) {
        let max_count = self.counts.values().copied().max().unwrap_or(1).max(1);
        let bar_width = 30usize;

        println!("  Bitstring   Count    Freq      Bar");
        println!("  {}", "─".repeat(60));
        for (bits, count) in self.sorted_counts() {
            let freq = count as f64 / self.n_shots as f64;
            let bar_len = (count * bar_width / max_count).min(bar_width);
            let bar = "█".repeat(bar_len);
            println!("  |{bits}⟩  {:>6}  {:.4}  {bar}", count, freq);
        }
        println!();
        println!("  Total shots: {}", self.n_shots);
        println!("  Distinct outcomes: {}", self.counts.len());
    }
}

// ── run_shots ─────────────────────────────────────────────────────────────

/// Run a circuit `n_shots` times using the **statevector** backend, collecting
/// measurement bitstrings into a histogram.
///
/// Only counts shots where **all** qubits are measured (i.e., `bitstring()` is
/// `Some`).  Shots with no measurements are still run but contribute `"(no measurement)"`
/// to the counts.
///
/// For other backends use [`run_shots_with`].
pub fn run_shots(program: &Program, n_shots: usize) -> Result<ShotResult, AqlError> {
    run_shots_statevector(program, n_shots)
}

/// Run shots with the statevector backend.
pub fn run_shots_statevector(program: &Program, n_shots: usize) -> Result<ShotResult, AqlError> {
    let mut counts: HashMap<String, usize> = HashMap::new();

    for _ in 0..n_shots {
        let result = execute(program)?;
        let key = if result.measurements.is_empty() {
            "(no measurement)".to_string()
        } else if let Some(bs) = result.bitstring() {
            bs
        } else {
            // Partial measurement — join available outcomes
            (0..result.num_qubits)
                .map(|q| match result.outcome(q) {
                    Some(true)  => '1',
                    Some(false) => '0',
                    None        => '?',
                })
                .collect()
        };
        *counts.entry(key).or_insert(0) += 1;
    }

    Ok(ShotResult {
        counts,
        n_shots,
        n_qubits: program.num_qubits,
    })
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::parse_source;

    #[test]
    fn test_shots_x_measure_always_one() {
        let prog = parse_source("QREG 1\nX 0\nMEASURE 0").unwrap();
        let result = run_shots(&prog, 50).unwrap();
        assert_eq!(result.n_shots, 50);
        assert_eq!(result.counts.get("1").copied().unwrap_or(0), 50);
        assert!(result.counts.get("0").is_none());
    }

    #[test]
    fn test_shots_bell_state_only_correlated() {
        let prog = parse_source("QREG 2\nH 0\nCNOT 0 1\nMEASURE_ALL").unwrap();
        let result = run_shots(&prog, 200).unwrap();
        assert_eq!(result.n_shots, 200);
        // Only "00" and "11" should appear
        for key in result.counts.keys() {
            assert!(
                key == "00" || key == "11",
                "Unexpected bitstring '{key}' in Bell state shots"
            );
        }
        // Both outcomes should appear with non-zero probability (statistical)
        let c00 = result.counts.get("00").copied().unwrap_or(0);
        let c11 = result.counts.get("11").copied().unwrap_or(0);
        assert!(c00 > 0, "Expected |00⟩ shots");
        assert!(c11 > 0, "Expected |11⟩ shots");
        assert_eq!(c00 + c11, 200);
    }

    #[test]
    fn test_shots_ghz_correlations() {
        let prog = parse_source(
            "QREG 3\nH 0\nCNOT 0 1\nCNOT 0 2\nMEASURE_ALL"
        ).unwrap();
        let result = run_shots(&prog, 100).unwrap();
        for key in result.counts.keys() {
            assert!(
                key == "000" || key == "111",
                "GHZ: unexpected bitstring '{key}'"
            );
        }
    }

    #[test]
    fn test_shots_prob_estimate() {
        // X|0⟩ = |1⟩ → P("1") = 1.0
        let prog = parse_source("QREG 1\nX 0\nMEASURE 0").unwrap();
        let result = run_shots(&prog, 100).unwrap();
        assert!((result.prob("1") - 1.0).abs() < 1e-9);
        assert!((result.prob("0") - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_shots_sorted_counts() {
        let prog = parse_source("QREG 2\nH 0\nCNOT 0 1\nMEASURE_ALL").unwrap();
        let result = run_shots(&prog, 100).unwrap();
        let sorted = result.sorted_counts();
        // Keys should be in lex order
        for i in 1..sorted.len() {
            assert!(sorted[i - 1].0 <= sorted[i].0, "counts not sorted");
        }
    }

    #[test]
    fn test_shots_h_superposition_roughly_half() {
        let prog = parse_source("QREG 1\nH 0\nMEASURE 0").unwrap();
        let result = run_shots(&prog, 1000).unwrap();
        let p0 = result.prob("0");
        let p1 = result.prob("1");
        // Each should be close to 0.5 (within 5% margin)
        assert!(
            (p0 - 0.5).abs() < 0.07,
            "H gate: P(0)={p0:.3} expected ≈0.5"
        );
        assert!(
            (p1 - 0.5).abs() < 0.07,
            "H gate: P(1)={p1:.3} expected ≈0.5"
        );
    }
}
