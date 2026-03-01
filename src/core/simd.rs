/// Layer 2 — Assembly Optimization Layer.
///
/// SIMD-accelerated quantum state vector operations using AVX2 (256-bit).
///
/// # Architecture
///
/// - [`SimdCapabilities`]: Runtime CPU feature detection (SSE2, AVX2, AVX-512F).
/// - [`Backend`]: Enum indicating which compute path was used.
/// - [`apply_gate_simd`]: Public dispatch — picks AVX2 or scalar at runtime.
/// - Internal `apply_gate_avx2_target0`: AVX2 fast path for qubit 0 (adjacent pairs).
/// - Internal `apply_gate_avx2_generic`: AVX2 generic path for all other qubits.
/// - Internal `apply_gate_scalar`: Portable fallback identical to the original loop.
///
/// # Shared ADDSUB Kernel
///
/// Both AVX2 paths share the same 2×2 complex matrix-vector product kernel.
/// For any pair `(c0, c1)` packed into a 256-bit YMM register as
/// `[c0.re, c0.im, c1.re, c1.im]`, the gate product
///
/// ```text
/// new_c0 = g00·c0 + g01·c1
/// new_c1 = g10·c0 + g11·c1
/// ```
///
/// is computed via the AVX2 ADDSUB trick (complex multiply without division):
///
/// ```text
/// For s·c where s = g_re + i·g_im, c = [c.re, c.im] duplicated to both halves:
///   t1 = [g_re, g_re, g_re, g_re] × [c.re, c.im, c.re, c.im]
///   t2 = [g_im, g_im, g_im, g_im] × [c.im, c.re, c.im, c.re]   ← re/im swapped
///   ADDSUB(t1, t2) = [t1[0]−t2[0], t1[1]+t2[1], …]
///                  = [s.re·c.re − s.im·c.im, s.re·c.im + s.im·c.re]
///                  = [Re(s·c), Im(s·c)]  ✓
/// ```
///
/// # AVX2 Path — Qubit 0 (Adjacent Pairs, Fastest)
///
/// When `target == 0`, pairs are always adjacent: `(2k, 2k+1)`.
/// Two Complex values fit in one 256-bit YMM load, so two gates compute per register.
///
/// ```text
/// YMM = [c0.re, c0.im, c1.re, c1.im]  (single _mm256_loadu_pd)
/// ```
///
/// Instructions per pair: 1 load + 4 permutes + 4 muls + 2 addsubs + 1 add + 1 store.
/// Throughput: ~3–4× scalar on wide pipelines.
///
/// # AVX2 Path — Qubit k > 0 (Gather-Pack-Scatter)
///
/// When `target == k > 0`, pairs `(i0, i0 | mask)` are `mask = 2^k` elements apart.
/// We load each Complex in a separate 128-bit half, pack into one 256-bit register,
/// apply the shared kernel, then unpack the two 128-bit results back to their locations.
///
/// ```text
/// load c0 → XMM_lo   load c1 → XMM_hi
/// YMM = _mm256_set_m128d(XMM_hi, XMM_lo)  →  [c0.re, c0.im, c1.re, c1.im]
/// … shared ADDSUB kernel …
/// store YMM[127:0]   → amplitudes[i0]
/// store YMM[255:128] → amplitudes[i1]
/// ```
///
/// Instructions per pair: 2 × 128-bit load + 1 pack + kernel + 2 × 128-bit store.
/// Throughput: ~2–3× scalar (extra scatter penalty vs target-0).

use super::gates::Matrix2x2;
use super::state::StateVector;

// ── Public API ─────────────────────────────────────────────────────────────

/// Detected SIMD instruction sets available on the executing CPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SimdCapabilities {
    /// SSE2 — baseline 128-bit double-precision SIMD (x86_64 guaranteed).
    pub sse2: bool,
    /// AVX2 — 256-bit integer + FMA. Primary acceleration target.
    pub avx2: bool,
    /// AVX-512F Foundation — 512-bit. Reported but not yet used.
    pub avx512f: bool,
}

impl SimdCapabilities {
    /// Probe the CPU at runtime and return the capability report.
    ///
    /// Uses `is_x86_feature_detected!` on x86/x86_64; returns all-false elsewhere.
    pub fn detect() -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            Self {
                sse2:    is_x86_feature_detected!("sse2"),
                avx2:    is_x86_feature_detected!("avx2"),
                avx512f: is_x86_feature_detected!("avx512f"),
            }
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        {
            Self { sse2: false, avx2: false, avx512f: false }
        }
    }

    /// Human-readable list of active instruction sets, e.g. `"SSE2, AVX2"`.
    pub fn feature_string(&self) -> String {
        let mut parts = Vec::new();
        if self.sse2    { parts.push("SSE2"); }
        if self.avx2    { parts.push("AVX2"); }
        if self.avx512f { parts.push("AVX-512F"); }
        if parts.is_empty() { "scalar only".to_owned() } else { parts.join(", ") }
    }

    /// Return the widest backend this CPU can use.
    pub fn best_backend(&self) -> Backend {
        if self.avx2 { Backend::Avx2 } else { Backend::Scalar }
    }
}

/// Compute backend selected by the runtime dispatcher.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// Portable scalar loop (always available).
    Scalar,
    /// AVX2 256-bit path (requires `avx2` CPU feature).
    Avx2,
}

impl std::fmt::Display for Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Scalar => write!(f, "Scalar"),
            Self::Avx2   => write!(f, "AVX2"),
        }
    }
}

/// Apply a single-qubit gate using the fastest backend available at runtime.
///
/// - **AVX2 path**: selected for **all target qubits** when the CPU supports AVX2.
///   - `target == 0`: adjacent-pair fast path (`apply_gate_avx2_target0`).
///   - `target > 0`: gather-pack-scatter path (`apply_gate_avx2_generic`).
/// - **Scalar fallback**: used only when AVX2 is absent.
///
/// Returns the backend that was used.
pub fn apply_gate_simd(
    state: &mut StateVector,
    gate: &Matrix2x2,
    target: usize,
) -> Backend {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx2") {
        // SAFETY:
        //   1. AVX2 confirmed at runtime by is_x86_feature_detected!.
        //   2. state.amplitudes.len() == 2^n >= 2 (enforced by StateVector::new).
        //   3. Complex is #[repr(C)] with layout {re: f64 @ +0, im: f64 @ +8},
        //      so casting *mut Complex → *mut f64 is sound.
        if target == 0 {
            unsafe { apply_gate_avx2_target0(state, gate); }
        } else {
            unsafe { apply_gate_avx2_generic(state, gate, target); }
        }
        return Backend::Avx2;
    }

    apply_gate_scalar(state, gate, target);
    Backend::Scalar
}

// ── Scalar fallback ────────────────────────────────────────────────────────

/// Standard O(2ⁿ) scalar loop — identical to the original gate application.
#[inline]
fn apply_gate_scalar(state: &mut StateVector, gate: &Matrix2x2, target: usize) {
    let dim         = state.dim();
    let target_mask = 1 << target;
    let mut i0 = 0;
    while i0 < dim {
        if i0 & target_mask != 0 { i0 += 1; continue; }
        let i1 = i0 | target_mask;
        let a0 = state.amplitudes[i0];
        let a1 = state.amplitudes[i1];
        state.amplitudes[i0] = gate[0][0] * a0 + gate[0][1] * a1;
        state.amplitudes[i1] = gate[1][0] * a0 + gate[1][1] * a1;
        i0 += 1;
    }
}

// ── AVX2 path ─────────────────────────────────────────────────────────────

/// AVX2-accelerated gate application for qubit 0 (adjacent-pair layout).
///
/// # Safety
/// Caller must have confirmed `is_x86_feature_detected!("avx2")` is true.
/// The `Complex` type must be `#[repr(C)]` with `{re: f64, im: f64}` layout.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn apply_gate_avx2_target0(state: &mut StateVector, gate: &Matrix2x2) {
    use std::arch::x86_64::*;

    // ── Precompute gate coefficient vectors ────────────────────────────────
    //
    // Layout goal (after _mm256_set_pd which fills high→low):
    //   ymm_g00g10_re  =  [g00.re, g00.re, g10.re, g10.re]
    //   ymm_g00g10_im  =  [g00.im, g00.im, g10.im, g10.im]
    //   ymm_g01g11_re  =  [g01.re, g01.re, g11.re, g11.re]
    //   ymm_g01g11_im  =  [g01.im, g01.im, g11.im, g11.im]
    //
    // Lower 128-bit lane = row-0 coefficient, upper lane = row-1 coefficient.
    // (Note: _mm256_set_pd(e3,e2,e1,e0) → indices [0]=e0, [1]=e1, [2]=e2, [3]=e3)

    let g00 = gate[0][0];   // row 0, col 0
    let g01 = gate[0][1];   // row 0, col 1
    let g10 = gate[1][0];   // row 1, col 0
    let g11 = gate[1][1];   // row 1, col 1

    let vg00g10_re = _mm256_set_pd(g10.re, g10.re, g00.re, g00.re);
    let vg00g10_im = _mm256_set_pd(g10.im, g10.im, g00.im, g00.im);
    let vg01g11_re = _mm256_set_pd(g11.re, g11.re, g01.re, g01.re);
    let vg01g11_im = _mm256_set_pd(g11.im, g11.im, g01.im, g01.im);

    // Cast the amplitude slice to a *mut f64 for direct SIMD load/store.
    // Each Complex = 2 × f64, so amplitudes[k] is at f64 offset 2k.
    let ptr = state.amplitudes.as_mut_ptr() as *mut f64;
    let len = state.amplitudes.len(); // 2^n, always ≥ 2 and even

    // Process one pair (two Complex = 4 f64 = 256 bits) per iteration.
    // For target=0 the pairs are (0,1), (2,3), (4,5), …
    let mut k = 0usize; // index of the first Complex in the current pair
    while k < len {
        // ── Step 1: Load pair ─────────────────────────────────────────────
        // data = [c0.re, c0.im, c1.re, c1.im]
        let data = _mm256_loadu_pd(ptr.add(k * 2));

        // ── Step 2: Duplicate each element across both 128-bit lanes ──────
        // c0_dup = [c0.re, c0.im, c0.re, c0.im]  (permute4x64 imm 0x44 = {0,1,0,1})
        // c1_dup = [c1.re, c1.im, c1.re, c1.im]  (permute4x64 imm 0xEE = {2,3,2,3})
        let c0_dup = _mm256_permute4x64_pd(data, 0x44);
        let c1_dup = _mm256_permute4x64_pd(data, 0xEE);

        // ── Step 3: Swap re↔im within each complex (for ADDSUB trick) ────
        // permute_pd imm=0b0101 swaps within each 128-bit lane:
        //   [c.re, c.im, c.re, c.im] → [c.im, c.re, c.im, c.re]
        let c0_swap = _mm256_permute_pd(c0_dup, 0b0101);
        let c1_swap = _mm256_permute_pd(c1_dup, 0b0101);

        // ── Step 4: Compute g·c for both rows using the ADDSUB trick ─────
        //
        // For row 0 (lower lane) and row 1 (upper lane) simultaneously:
        //   t1 = [g00.re·c0.re, g00.re·c0.im, g10.re·c0.re, g10.re·c0.im]
        //   t2 = [g00.im·c0.im, g00.im·c0.re, g10.im·c0.im, g10.im·c0.re]
        //   ADDSUB(t1,t2) = [g00.re·c0.re − g00.im·c0.im,     ← Re(g00·c0)
        //                    g00.re·c0.im + g00.im·c0.re,     ← Im(g00·c0)
        //                    g10.re·c0.re − g10.im·c0.im,     ← Re(g10·c0)
        //                    g10.re·c0.im + g10.im·c0.re]     ← Im(g10·c0)
        //                 = [g00·c0, g10·c0]  (packed, lower/upper lanes)
        let t1 = _mm256_mul_pd(vg00g10_re, c0_dup);
        let t2 = _mm256_mul_pd(vg00g10_im, c0_swap);
        let from_c0 = _mm256_addsub_pd(t1, t2); // [g00·c0, g10·c0]

        let t3 = _mm256_mul_pd(vg01g11_re, c1_dup);
        let t4 = _mm256_mul_pd(vg01g11_im, c1_swap);
        let from_c1 = _mm256_addsub_pd(t3, t4); // [g01·c1, g11·c1]

        // ── Step 5: Accumulate and store ──────────────────────────────────
        // result = [g00·c0 + g01·c1, g10·c0 + g11·c1]
        //        = [new_c0,           new_c1          ]
        let result = _mm256_add_pd(from_c0, from_c1);
        _mm256_storeu_pd(ptr.add(k * 2), result);

        k += 2; // advance by 2 Complex elements (one pair)
    }
}

// ── AVX2 generic path (target k > 0) ──────────────────────────────────────

/// AVX2-accelerated gate application for any qubit `k > 0` (gather-pack-scatter).
///
/// For target qubit `k`, pairs are `(i0, i0 + mask)` where `mask = 1 << k` and
/// `(i0 & mask) == 0`.  The two Complex values are not adjacent in memory, so we:
///
/// 1. Load each into a 128-bit XMM half.
/// 2. Pack them into one 256-bit YMM register (`_mm256_set_m128d`).
/// 3. Apply the shared ADDSUB kernel (identical to `apply_gate_avx2_target0`).
/// 4. Scatter the two 128-bit result halves back to their original addresses.
///
/// # Safety
/// Caller must have confirmed `is_x86_feature_detected!("avx2")` is true.
/// The `Complex` type must be `#[repr(C)]` with `{re: f64, im: f64}` layout.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn apply_gate_avx2_generic(state: &mut StateVector, gate: &Matrix2x2, target: usize) {
    use std::arch::x86_64::*;

    let g00 = gate[0][0];
    let g01 = gate[0][1];
    let g10 = gate[1][0];
    let g11 = gate[1][1];

    // Same gate coefficient vectors as target-0
    let vg00g10_re = _mm256_set_pd(g10.re, g10.re, g00.re, g00.re);
    let vg00g10_im = _mm256_set_pd(g10.im, g10.im, g00.im, g00.im);
    let vg01g11_re = _mm256_set_pd(g11.re, g11.re, g01.re, g01.re);
    let vg01g11_im = _mm256_set_pd(g11.im, g11.im, g01.im, g01.im);

    let ptr  = state.amplitudes.as_mut_ptr() as *mut f64;
    let dim  = state.amplitudes.len();
    let mask = 1usize << target;

    // Iterate over only the "bit-k = 0" half of the index space.
    // step = 2*mask; within [chunk, chunk+mask) all indices have bit k = 0.
    let step = mask * 2;
    let mut chunk = 0usize;
    while chunk < dim {
        let mut i0 = chunk;
        let chunk_end = chunk + mask;
        while i0 < chunk_end {
            let i1 = i0 + mask; // == i0 | mask (bit k of i0 is guaranteed 0 here)

            // ── Load c0 and c1 from non-adjacent memory ───────────────────
            let c0_xmm = _mm_loadu_pd(ptr.add(i0 * 2)); // [c0.re, c0.im]
            let c1_xmm = _mm_loadu_pd(ptr.add(i1 * 2)); // [c1.re, c1.im]

            // ── Pack into 256-bit: lower lane = c0, upper lane = c1 ───────
            // _mm256_set_m128d(hi, lo) → YMM[127:0]=lo, YMM[255:128]=hi
            let data = _mm256_set_m128d(c1_xmm, c0_xmm);
            // data = [c0.re, c0.im, c1.re, c1.im]

            // ── Shared ADDSUB kernel (identical to target-0) ──────────────
            let c0_dup  = _mm256_permute4x64_pd(data, 0x44);  // [c0, c0]
            let c1_dup  = _mm256_permute4x64_pd(data, 0xEE);  // [c1, c1]
            let c0_swap = _mm256_permute_pd(c0_dup, 0b0101);  // re↔im swap
            let c1_swap = _mm256_permute_pd(c1_dup, 0b0101);

            let from_c0 = _mm256_addsub_pd(
                _mm256_mul_pd(vg00g10_re, c0_dup),
                _mm256_mul_pd(vg00g10_im, c0_swap),
            );
            let from_c1 = _mm256_addsub_pd(
                _mm256_mul_pd(vg01g11_re, c1_dup),
                _mm256_mul_pd(vg01g11_im, c1_swap),
            );
            let result = _mm256_add_pd(from_c0, from_c1);
            // result = [new_c0.re, new_c0.im, new_c1.re, new_c1.im]

            // ── Scatter: store 128-bit halves to original addresses ────────
            _mm_storeu_pd(ptr.add(i0 * 2), _mm256_castpd256_pd128(result));
            _mm_storeu_pd(ptr.add(i1 * 2), _mm256_extractf128_pd(result, 1));

            i0 += 1;
        }
        chunk += step;
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::complex::Complex;
    use crate::core::gates::{hadamard, pauli_x, pauli_y, pauli_z, rx, rz};
    use std::f64::consts::PI;

    // Tolerance for floating-point comparisons
    const EPS: f64 = 1e-10;

    fn nearly(a: f64, b: f64) -> bool { (a - b).abs() < EPS }

    // Apply gate via SIMD dispatch and return the backend chosen.
    fn apply(state: &mut StateVector, gate: &Matrix2x2, target: usize) -> Backend {
        apply_gate_simd(state, gate, target)
    }

    // ── SimdCapabilities ──────────────────────────────────────────────────

    #[test]
    fn test_detect_runs_without_panic() {
        let caps = SimdCapabilities::detect();
        // On x86_64 SSE2 is always present; on other arches all are false.
        #[cfg(target_arch = "x86_64")]
        assert!(caps.sse2, "SSE2 must be available on x86_64");
        let _ = caps; // suppress unused warning on non-x86
    }

    #[test]
    fn test_feature_string_nonempty_on_x86_64() {
        let caps = SimdCapabilities::detect();
        let s = caps.feature_string();
        #[cfg(target_arch = "x86_64")]
        assert!(!s.is_empty());
        let _ = s;
    }

    #[test]
    fn test_best_backend_consistency() {
        let caps = SimdCapabilities::detect();
        let b = caps.best_backend();
        // If AVX2 is available, best must be Avx2; otherwise Scalar.
        if caps.avx2 {
            assert_eq!(b, Backend::Avx2);
        } else {
            assert_eq!(b, Backend::Scalar);
        }
    }

    // ── Correctness: SIMD result must match scalar ─────────────────────────

    fn apply_scalar_ref(state: &mut StateVector, gate: &Matrix2x2, target: usize) {
        apply_gate_scalar(state, gate, target);
    }

    fn state_with_amplitudes(n: usize, amps: &[(usize, Complex)]) -> StateVector {
        let mut sv = StateVector::new(n);
        // Reset all amplitudes
        for a in sv.amplitudes.iter_mut() { *a = Complex::zero(); }
        for &(idx, amp) in amps { sv.amplitudes[idx] = amp; }
        sv
    }

    #[test]
    fn test_x_gate_qubit0_matches_scalar() {
        let gate = pauli_x();
        let mut s_simd   = StateVector::new(3);
        let mut s_scalar = StateVector::new(3);
        apply(&mut s_simd, &gate, 0);
        apply_scalar_ref(&mut s_scalar, &gate, 0);
        for i in 0..s_simd.amplitudes.len() {
            assert!(nearly(s_simd.amplitudes[i].re, s_scalar.amplitudes[i].re),
                "re mismatch at {i}");
            assert!(nearly(s_simd.amplitudes[i].im, s_scalar.amplitudes[i].im),
                "im mismatch at {i}");
        }
    }

    #[test]
    fn test_h_gate_qubit0_matches_scalar() {
        let gate = hadamard();
        let mut s_simd   = StateVector::new(4);
        let mut s_scalar = StateVector::new(4);
        apply(&mut s_simd, &gate, 0);
        apply_scalar_ref(&mut s_scalar, &gate, 0);
        for i in 0..s_simd.amplitudes.len() {
            assert!(nearly(s_simd.amplitudes[i].re, s_scalar.amplitudes[i].re));
            assert!(nearly(s_simd.amplitudes[i].im, s_scalar.amplitudes[i].im));
        }
    }

    #[test]
    fn test_h_gate_higher_qubit_matches_scalar() {
        // Target > 0 now uses AVX2 (if available) — result must always match scalar.
        let gate = hadamard();
        let mut s_simd   = StateVector::new(3);
        let mut s_scalar = StateVector::new(3);
        let _backend = apply(&mut s_simd, &gate, 2);
        apply_scalar_ref(&mut s_scalar, &gate, 2);
        for i in 0..s_simd.amplitudes.len() {
            assert!(nearly(s_simd.amplitudes[i].re, s_scalar.amplitudes[i].re),
                "re[{i}] simd={} scalar={}", s_simd.amplitudes[i].re, s_scalar.amplitudes[i].re);
            assert!(nearly(s_simd.amplitudes[i].im, s_scalar.amplitudes[i].im),
                "im[{i}] simd={} scalar={}", s_simd.amplitudes[i].im, s_scalar.amplitudes[i].im);
        }
    }

    #[test]
    fn test_avx2_all_qubit_targets_match_scalar() {
        // Verify that every qubit target in a 4-qubit system gives the same result
        // as scalar, regardless of which backend is chosen.
        let n = 4;
        let gate = hadamard();
        for target in 0..n {
            let mut s_simd   = StateVector::new(n);
            let mut s_scalar = StateVector::new(n);
            // Put the state in a non-trivial superposition first
            apply_scalar_ref(&mut s_simd,   &gate, 0);
            apply_scalar_ref(&mut s_scalar, &gate, 0);
            // Now apply the gate to each target via SIMD dispatch vs scalar
            apply(&mut s_simd,   &gate, target);
            apply_scalar_ref(&mut s_scalar, &gate, target);
            for i in 0..(1 << n) {
                assert!(nearly(s_simd.amplitudes[i].re, s_scalar.amplitudes[i].re),
                    "target={target} re[{i}]: simd={} scalar={}",
                    s_simd.amplitudes[i].re, s_scalar.amplitudes[i].re);
                assert!(nearly(s_simd.amplitudes[i].im, s_scalar.amplitudes[i].im),
                    "target={target} im[{i}]: simd={} scalar={}",
                    s_simd.amplitudes[i].im, s_scalar.amplitudes[i].im);
            }
        }
    }

    #[test]
    fn test_x_gate_qubit1_flips_correctly() {
        // X on qubit 1 of |00⟩ → |10⟩ (in LSB convention qubit 1 is bit 1)
        let mut sv = StateVector::new(2); // |00⟩
        apply(&mut sv, &pauli_x(), 1);
        // |10⟩ = index 2 (binary: bit0=0, bit1=1 → index = 0b10 = 2)
        assert!(nearly(sv.amplitudes[0].norm_sq(), 0.0), "|00⟩ should have prob 0");
        assert!(nearly(sv.amplitudes[1].norm_sq(), 0.0), "|01⟩ should have prob 0");
        assert!(nearly(sv.amplitudes[2].norm_sq(), 1.0), "|10⟩ should have prob 1");
        assert!(nearly(sv.amplitudes[3].norm_sq(), 0.0), "|11⟩ should have prob 0");
    }

    #[test]
    fn test_generic_backend_used_for_target_gt0() {
        // When AVX2 is present, ALL targets must report Avx2.
        // When AVX2 is absent, ALL targets report Scalar.
        let caps = SimdCapabilities::detect();
        let expected = caps.best_backend();
        let gate = pauli_x();
        for target in 0..4usize {
            let mut sv = StateVector::new(4);
            let got = apply(&mut sv, &gate, target);
            assert_eq!(got, expected,
                "target={target}: expected {expected}, got {got}");
        }
    }

    #[test]
    fn test_x_gate_flips_qubit0() {
        let mut sv = StateVector::new(1);   // |0⟩
        apply(&mut sv, &pauli_x(), 0);      // → |1⟩
        assert!(nearly(sv.amplitudes[0].norm_sq(), 0.0));
        assert!(nearly(sv.amplitudes[1].norm_sq(), 1.0));
    }

    #[test]
    fn test_x_x_is_identity_qubit0() {
        let gate = pauli_x();
        let mut sv = StateVector::new(2);
        apply(&mut sv, &gate, 0);
        apply(&mut sv, &gate, 0);
        // Should be back to |00⟩
        assert!(nearly(sv.amplitudes[0].norm_sq(), 1.0));
        for i in 1..4 { assert!(nearly(sv.amplitudes[i].norm_sq(), 0.0)); }
    }

    #[test]
    fn test_h_creates_superposition_qubit0() {
        let mut sv = StateVector::new(1);
        apply(&mut sv, &hadamard(), 0);
        assert!(nearly(sv.amplitudes[0].norm_sq(), 0.5));
        assert!(nearly(sv.amplitudes[1].norm_sq(), 0.5));
    }

    #[test]
    fn test_h_h_is_identity_qubit0() {
        let gate = hadamard();
        let mut sv = StateVector::new(3);
        apply(&mut sv, &gate, 0);
        apply(&mut sv, &gate, 0);
        assert!(nearly(sv.amplitudes[0].norm_sq(), 1.0));
        for i in 1..8 { assert!(nearly(sv.amplitudes[i].norm_sq(), 0.0)); }
    }

    #[test]
    fn test_y_gate_qubit0_matches_scalar() {
        let gate = pauli_y();
        let mut sv_simd   = StateVector::new(2);
        let mut sv_scalar = StateVector::new(2);
        apply(&mut sv_simd, &gate, 0);
        apply_scalar_ref(&mut sv_scalar, &gate, 0);
        for i in 0..4 {
            assert!(nearly(sv_simd.amplitudes[i].re, sv_scalar.amplitudes[i].re));
            assert!(nearly(sv_simd.amplitudes[i].im, sv_scalar.amplitudes[i].im));
        }
    }

    #[test]
    fn test_z_gate_qubit0_matches_scalar() {
        let gate = pauli_z();
        // Start in superposition for interesting state
        let mut sv_simd   = StateVector::new(1);
        let mut sv_scalar = StateVector::new(1);
        apply(&mut sv_simd, &hadamard(), 0);
        apply_scalar_ref(&mut sv_scalar, &hadamard(), 0);
        apply(&mut sv_simd, &gate, 0);
        apply_scalar_ref(&mut sv_scalar, &gate, 0);
        for i in 0..2 {
            assert!(nearly(sv_simd.amplitudes[i].re, sv_scalar.amplitudes[i].re));
            assert!(nearly(sv_simd.amplitudes[i].im, sv_scalar.amplitudes[i].im));
        }
    }

    #[test]
    fn test_rx_pi_qubit0_matches_scalar() {
        let gate = rx(PI);
        let mut sv_simd   = StateVector::new(2);
        let mut sv_scalar = StateVector::new(2);
        apply(&mut sv_simd, &gate, 0);
        apply_scalar_ref(&mut sv_scalar, &gate, 0);
        for i in 0..4 {
            assert!(nearly(sv_simd.amplitudes[i].re, sv_scalar.amplitudes[i].re));
            assert!(nearly(sv_simd.amplitudes[i].im, sv_scalar.amplitudes[i].im));
        }
    }

    #[test]
    fn test_rz_gate_complex_state_qubit0() {
        // Verify on a non-trivial complex state
        let gate = rz(PI / 3.0);
        let mut sv_simd   = state_with_amplitudes(2, &[
            (0, Complex::new(0.5,  0.5)),
            (1, Complex::new(0.5, -0.5)),
            (2, Complex::new(0.0,  0.0)),
            (3, Complex::new(0.0,  0.0)),
        ]);
        let mut sv_scalar = sv_simd.clone();
        apply(&mut sv_simd, &gate, 0);
        apply_scalar_ref(&mut sv_scalar, &gate, 0);
        for i in 0..4 {
            assert!(nearly(sv_simd.amplitudes[i].re, sv_scalar.amplitudes[i].re),
                "re[{i}]: {} ≠ {}", sv_simd.amplitudes[i].re, sv_scalar.amplitudes[i].re);
            assert!(nearly(sv_simd.amplitudes[i].im, sv_scalar.amplitudes[i].im),
                "im[{i}]: {} ≠ {}", sv_simd.amplitudes[i].im, sv_scalar.amplitudes[i].im);
        }
    }

    #[test]
    fn test_normalization_preserved_after_simd_h() {
        let mut sv = StateVector::new(4);   // 16 amplitudes
        apply(&mut sv, &hadamard(), 0);
        let total = sv.total_probability();
        assert!(nearly(total, 1.0),
            "total probability {total} drifted from 1.0 after SIMD gate");
    }

    #[test]
    fn test_normalization_preserved_after_simd_y() {
        let mut sv = StateVector::new(5);   // 32 amplitudes
        apply(&mut sv, &pauli_y(), 0);
        let total = sv.total_probability();
        assert!(nearly(total, 1.0));
    }

    // ── Backend display ───────────────────────────────────────────────────

    #[test]
    fn test_backend_display() {
        assert_eq!(format!("{}", Backend::Scalar), "Scalar");
        assert_eq!(format!("{}", Backend::Avx2),   "AVX2");
    }

    // ── Large-state correctness ───────────────────────────────────────────

    #[test]
    fn test_simd_correct_on_16_qubit_system() {
        // 2^16 = 65536 amplitudes — stress-tests the main loop.
        let gate = hadamard();
        let mut sv_simd   = StateVector::new(16);
        let mut sv_scalar = StateVector::new(16);
        apply(&mut sv_simd, &gate, 0);
        apply_scalar_ref(&mut sv_scalar, &gate, 0);
        // Spot-check: first two amplitudes
        assert!(nearly(sv_simd.amplitudes[0].re, sv_scalar.amplitudes[0].re));
        assert!(nearly(sv_simd.amplitudes[1].re, sv_scalar.amplitudes[1].re));
        // Check normalization
        assert!(nearly(sv_simd.total_probability(), 1.0));
    }
}
