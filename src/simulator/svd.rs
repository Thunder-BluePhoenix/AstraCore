/// Complex Jacobi SVD — internal utility for the MPS simulator.
///
/// Computes the thin SVD of a complex matrix M (m×n) using the one-sided
/// complex Jacobi algorithm.  No external dependencies required.
///
/// Algorithm:
///   For each off-diagonal pair (p, q) of M†M:
///     1. Phase-rotate column q so the off-diagonal becomes real positive.
///     2. Apply a real Givens rotation to zero it out.
///   After convergence: columns of A = U·Σ, V accumulates right singular vectors.
///
/// Returns (U, sigma, V):
///   - U: m×r, column-major: U[col*m + row]
///   - sigma: r singular values in descending order
///   - V: n×r, column-major: V[col*n + row]
///
/// Guarantee: M ≈ U · diag(sigma) · V† (conjugate transpose of V).
use crate::core::complex::Complex;

/// Thin complex SVD.  Keeps at most `max_rank` singular triplets.
/// For MPS bond dimension χ, the matrices are (2χ × 2χ) at most.
pub fn complex_svd_thin(
    m:        usize,
    n:        usize,
    mat:      &[Complex],  // m×n row-major input
    max_rank: usize,
) -> (Vec<Complex>, Vec<f64>, Vec<Complex>) {
    debug_assert_eq!(mat.len(), m * n);
    let r = max_rank.min(m).min(n);

    // ── Copy to column-major A ─────────────────────────────────────────
    // A[col*m + row]
    let mut a: Vec<Complex> = vec![Complex::zero(); m * n];
    for row in 0..m {
        for col in 0..n {
            a[col * m + row] = mat[row * n + col];
        }
    }

    // ── V accumulates right singular vectors (n×n identity, column-major) ──
    let mut v: Vec<Complex> = vec![Complex::zero(); n * n];
    for i in 0..n { v[i * n + i] = Complex::one(); }

    // ── One-sided Jacobi sweeps ────────────────────────────────────────
    const MAX_SWEEPS: usize = 50;

    'outer: for _sweep in 0..MAX_SWEEPS {
        let mut any_rotation = false;

        for p in 0..n {
            for q in (p + 1)..n {
                // γ = A[:,p]† · A[:,q]
                let gamma: Complex = (0..m)
                    .map(|i| a[p * m + i].conj() * a[q * m + i])
                    .fold(Complex::zero(), |s, x| s + x);

                let gamma_norm = gamma.norm();
                if gamma_norm < 1e-15 { continue; }

                any_rotation = true;

                // α = ‖A[:,p]‖², β = ‖A[:,q]‖²
                let alpha: f64 = (0..m).map(|i| a[p * m + i].norm_sq()).sum();
                let beta:  f64 = (0..m).map(|i| a[q * m + i].norm_sq()).sum();

                // Step 1: phase-rotate column q so γ → real positive.
                // Multiply A[:,q] and V[:,q] by e^{-i·arg(γ)}.
                let phase = Complex::from_polar(1.0, -(gamma.im.atan2(gamma.re)));
                for i in 0..m { let x = a[q * m + i]; a[q * m + i] = phase * x; }
                for i in 0..n { let x = v[q * n + i]; v[q * n + i] = phase * x; }

                // Step 2: real Givens rotation to zero out the (now real) |γ|.
                let xi = (beta - alpha) / (2.0 * gamma_norm);
                let t = if xi >= 0.0 {
                    1.0 / (xi + (1.0 + xi * xi).sqrt())
                } else {
                    1.0 / (xi - (1.0 + xi * xi).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;

                // A[:,p] ←  c·A[:,p] + s·A[:,q]
                // A[:,q] ← -s·A[:,p] + c·A[:,q]  (using saved originals)
                for i in 0..m {
                    let ap = a[p * m + i];
                    let aq = a[q * m + i];
                    a[p * m + i] = ap.scale(c) + aq.scale(s);
                    a[q * m + i] = aq.scale(c) - ap.scale(s);
                }
                for i in 0..n {
                    let vp = v[p * n + i];
                    let vq = v[q * n + i];
                    v[p * n + i] = vp.scale(c) + vq.scale(s);
                    v[q * n + i] = vq.scale(c) - vp.scale(s);
                }
            }
        }

        if !any_rotation { break 'outer; }
    }

    // ── Singular values = column norms of A ──────────────────────────────
    let sigmas: Vec<f64> = (0..n)
        .map(|j| (0..m).map(|i| a[j * m + i].norm_sq()).sum::<f64>().sqrt())
        .collect();

    // ── Sort columns by descending singular value ─────────────────────
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| sigmas[b].partial_cmp(&sigmas[a])
        .unwrap_or(std::cmp::Ordering::Equal));

    // ── Build column-major output ─────────────────────────────────────
    let mut u_out = vec![Complex::zero(); m * r];
    let mut s_out = vec![0.0_f64;       r];
    let mut v_out = vec![Complex::zero(); n * r];

    for (k, &j) in order.iter().take(r).enumerate() {
        s_out[k] = sigmas[j];
        let inv_s = if sigmas[j] > 1e-14 { 1.0 / sigmas[j] } else { 0.0 };
        for i in 0..m { u_out[k * m + i] = a[j * m + i].scale(inv_s); }
        for i in 0..n { v_out[k * n + i] = v[j * n + i]; }
    }

    (u_out, s_out, v_out)
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn nearly_eq(a: f64, b: f64) -> bool { (a - b).abs() < 1e-9 }
    fn c(re: f64, im: f64) -> Complex { Complex::new(re, im) }

    /// Reconstruct M from SVD factors and check M ≈ U Σ V†.
    fn check_reconstruction(m: usize, n: usize, mat: &[Complex],
                            u: &[Complex], sigma: &[f64], v: &[Complex], r: usize)
    {
        for i in 0..m {
            for j in 0..n {
                let mut amp = Complex::zero();
                for k in 0..r {
                    // U[k*m + i] * sigma[k] * conj(V[k*n + j])
                    amp = amp + u[k * m + i] * v[k * n + j].conj().scale(sigma[k]);
                }
                let orig = mat[i * n + j];
                assert!(
                    (amp.re - orig.re).abs() < 1e-8 && (amp.im - orig.im).abs() < 1e-8,
                    "SVD reconstruction mismatch at ({i},{j}): got {amp} expected {orig}"
                );
            }
        }
    }

    #[test]
    fn test_svd_real_2x2_identity() {
        let m = c(1.0, 0.0);
        let z = c(0.0, 0.0);
        let mat = [m, z, z, m]; // identity
        let (u, s, v) = complex_svd_thin(2, 2, &mat, 2);
        assert!(nearly_eq(s[0], 1.0));
        assert!(nearly_eq(s[1], 1.0));
        check_reconstruction(2, 2, &mat, &u, &s, &v, 2);
    }

    #[test]
    fn test_svd_diagonal_matrix() {
        // Diagonal [3, 1] — singular values should be [3, 1]
        let mat = [c(3.0, 0.0), c(0.0, 0.0),
                   c(0.0, 0.0), c(1.0, 0.0)];
        let (u, s, v) = complex_svd_thin(2, 2, &mat, 2);
        assert!(nearly_eq(s[0], 3.0), "s[0]={}", s[0]);
        assert!(nearly_eq(s[1], 1.0), "s[1]={}", s[1]);
        check_reconstruction(2, 2, &mat, &u, &s, &v, 2);
    }

    #[test]
    fn test_svd_complex_rank1() {
        // Rank-1: outer product u·v† where u=[1,i]/√2, v=[1,0]
        // σ = 1, only one non-zero singular value
        let s2 = 1.0 / 2.0_f64.sqrt();
        let mat = [c(s2, 0.0), c(0.0, 0.0),
                   c(0.0, s2), c(0.0, 0.0)];
        let (u, s, v) = complex_svd_thin(2, 2, &mat, 2);
        assert!(nearly_eq(s[0], 1.0), "s[0]={}", s[0]);
        assert!(s[1] < 1e-10, "s[1]={} (should be ~0)", s[1]);
        check_reconstruction(2, 2, &mat, &u, &s, &v, 2);
    }

    #[test]
    fn test_svd_hadamard() {
        // Hadamard: singular values both = 1
        let s2 = 1.0 / 2.0_f64.sqrt();
        let mat = [c(s2, 0.0), c( s2, 0.0),
                   c(s2, 0.0), c(-s2, 0.0)];
        let (u, s, v) = complex_svd_thin(2, 2, &mat, 2);
        assert!(nearly_eq(s[0], 1.0), "s[0]={}", s[0]);
        assert!(nearly_eq(s[1], 1.0), "s[1]={}", s[1]);
        check_reconstruction(2, 2, &mat, &u, &s, &v, 2);
    }

    #[test]
    fn test_svd_complex_off_diagonal() {
        // Matrix M = [[2, i], [-i, 2]].
        // M†M = [[5, 4i], [-4i, 5]] has eigenvalues 9 and 1,
        // so singular values are σ₀=3 and σ₁=1 (sorted descending).
        let mat = [c(2.0, 0.0), c(0.0, 1.0),
                   c(0.0, -1.0), c(2.0, 0.0)];
        let (u, s, v) = complex_svd_thin(2, 2, &mat, 2);
        assert!((s[0] - 3.0).abs() < 1e-8, "s[0]={}", s[0]);
        assert!((s[1] - 1.0).abs() < 1e-8, "s[1]={}", s[1]);
        check_reconstruction(2, 2, &mat, &u, &s, &v, 2);
    }

    #[test]
    fn test_svd_4x4_cnot_gate() {
        // CNOT unitary — all singular values should be 1
        let mat = [
            c(1.0,0.0), c(0.0,0.0), c(0.0,0.0), c(0.0,0.0),
            c(0.0,0.0), c(1.0,0.0), c(0.0,0.0), c(0.0,0.0),
            c(0.0,0.0), c(0.0,0.0), c(0.0,0.0), c(1.0,0.0),
            c(0.0,0.0), c(0.0,0.0), c(1.0,0.0), c(0.0,0.0),
        ];
        let (u, s, v) = complex_svd_thin(4, 4, &mat, 4);
        for k in 0..4 {
            assert!((s[k] - 1.0).abs() < 1e-8, "s[{k}]={}", s[k]);
        }
        check_reconstruction(4, 4, &mat, &u, &s, &v, 4);
    }

    #[test]
    fn test_svd_truncation_rank2() {
        // 4×4 rank-2 matrix: only 2 non-zero singular values
        let mat = [
            c(1.0,0.0), c(0.0,0.0), c(0.0,0.0), c(0.0,0.0),
            c(0.0,0.0), c(2.0,0.0), c(0.0,0.0), c(0.0,0.0),
            c(0.0,0.0), c(0.0,0.0), c(0.0,0.0), c(0.0,0.0),
            c(0.0,0.0), c(0.0,0.0), c(0.0,0.0), c(0.0,0.0),
        ];
        let (_, s, _) = complex_svd_thin(4, 4, &mat, 4);
        assert!((s[0] - 2.0).abs() < 1e-8);
        assert!((s[1] - 1.0).abs() < 1e-8);
        assert!(s[2] < 1e-10);
        assert!(s[3] < 1e-10);
    }
}
