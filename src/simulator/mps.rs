/// Matrix Product State (MPS) quantum simulator.
///
/// Enables simulation of 50–200+ qubit circuits for
/// low-to-moderate entanglement.  State memory is O(n × χ²)
/// instead of O(2^n) for statevector.
///
/// # Representation
/// Each qubit k has a site tensor A[k] of shape (chi_L, 2, chi_R):
///   - chi_L  = left bond dimension  (1 for first site)
///   - 2      = physical dimension   (|0⟩ / |1⟩)
///   - chi_R  = right bond dimension (1 for last site)
///
/// The quantum state is:
///   |ψ⟩ = Σ_{b₀…bₙ₋₁} A[0][b₀] · A[1][b₁] · … · A[n-1][bₙ₋₁] |b₀…bₙ₋₁⟩
///
/// # Limitations
/// - Two-qubit gates on non-adjacent qubits are implemented via SWAP routing.
/// - Circuits with high entanglement will cause χ to grow; once χ > max_bond_dim,
///   amplitudes are truncated (accuracy trade-off).
use std::collections::HashMap;
use crate::compiler::{
    AqlError,
    ir::{Instruction, Program},
};
use crate::core::{
    complex::Complex,
    gates::{self, Matrix2x2},
};
use crate::runtime::{ExecutionResult, MeasurementRecord};
use super::svd::complex_svd_thin;

const MAX_QUBITS_FULL_CONTRACTION: usize = 24;

// ── Site tensor ────────────────────────────────────────────────────────────

#[derive(Clone)]
struct SiteTensor {
    chi_left:  usize,
    chi_right: usize,
    /// Row-major: data[l * 2 * chi_right + p * chi_right + r]
    data: Vec<Complex>,
}

impl SiteTensor {
    fn new(chi_left: usize, chi_right: usize) -> Self {
        Self {
            chi_left,
            chi_right,
            data: vec![Complex::zero(); chi_left * 2 * chi_right],
        }
    }

    #[inline(always)]
    fn get(&self, l: usize, p: usize, r: usize) -> Complex {
        self.data[l * 2 * self.chi_right + p * self.chi_right + r]
    }

    #[inline(always)]
    fn set(&mut self, l: usize, p: usize, r: usize, val: Complex) {
        self.data[l * 2 * self.chi_right + p * self.chi_right + r] = val;
    }
}

// ── MPS state ──────────────────────────────────────────────────────────────

pub struct MpsState {
    pub num_qubits:   usize,
    pub max_bond_dim: usize,
    tensors:          Vec<SiteTensor>,
}

impl MpsState {
    /// Initialise to |0…0⟩ (all tensors are trivial chi=1 product state).
    pub fn new(num_qubits: usize, max_bond_dim: usize) -> Self {
        assert!(num_qubits >= 1, "need at least one qubit");
        let tensors: Vec<SiteTensor> = (0..num_qubits).map(|_| {
            let mut t = SiteTensor::new(1, 1);
            t.set(0, 0, 0, Complex::one()); // |0⟩ amplitude = 1
            t
        }).collect();
        Self { num_qubits, max_bond_dim, tensors }
    }

    // ── Single-qubit gate ────────────────────────────────────────────────

    /// Apply 2×2 gate `gate` to qubit `target`.
    pub fn apply_single_qubit(&mut self, target: usize, gate: &Matrix2x2) {
        let t = &mut self.tensors[target];
        let chi_l = t.chi_left;
        let chi_r = t.chi_right;
        let mut new_data = vec![Complex::zero(); chi_l * 2 * chi_r];
        for l in 0..chi_l {
            for p_out in 0..2 {
                for r in 0..chi_r {
                    let mut amp = Complex::zero();
                    for p_in in 0..2 {
                        amp = amp + gate[p_out][p_in] * t.get(l, p_in, r);
                    }
                    new_data[l * 2 * chi_r + p_out * chi_r + r] = amp;
                }
            }
        }
        t.data = new_data;
    }

    // ── Two-qubit gate (adjacent sites only) ─────────────────────────────

    /// Apply a 4×4 two-qubit gate `gate` on adjacent sites `k` and `k+1`.
    ///
    /// `gate` is row-major indexed as `gate[(p1_out*2+p2_out)*4 + p1_in*2+p2_in]`.
    fn apply_two_qubit_adjacent(&mut self, k: usize, gate: &[Complex; 16]) {
        debug_assert_eq!(k + 1 < self.num_qubits, true);

        let chi_l    = self.tensors[k].chi_left;
        let chi_bond = self.tensors[k].chi_right;   // = tensors[k+1].chi_left
        let chi_r    = self.tensors[k + 1].chi_right;

        let m_rows = chi_l * 2;
        let m_cols = 2 * chi_r;

        // ── Step 1: Contract sites k and k+1 into theta ──────────────
        // theta[(l, p1), (p2, r)] = Σ_bond A[k][l,p1,bond] · A[k+1][bond,p2,r]
        let mut theta = vec![Complex::zero(); m_rows * m_cols];
        for l in 0..chi_l {
            for p1 in 0..2 {
                for p2 in 0..2 {
                    for r in 0..chi_r {
                        let row = l * 2 + p1;
                        let col = p2 * chi_r + r;
                        let mut amp = Complex::zero();
                        for bond in 0..chi_bond {
                            amp = amp + self.tensors[k].get(l, p1, bond)
                                      * self.tensors[k + 1].get(bond, p2, r);
                        }
                        theta[row * m_cols + col] = amp;
                    }
                }
            }
        }

        // ── Step 2: Apply gate ──────────────────────────────────────
        let mut theta_prime = vec![Complex::zero(); m_rows * m_cols];
        for l in 0..chi_l {
            for p1_out in 0..2 {
                for p2_out in 0..2 {
                    for r in 0..chi_r {
                        let row_out = l * 2 + p1_out;
                        let col_out = p2_out * chi_r + r;
                        let mut amp = Complex::zero();
                        for p1_in in 0..2 {
                            for p2_in in 0..2 {
                                let row_in = l * 2 + p1_in;
                                let col_in = p2_in * chi_r + r;
                                let g = gate[(p1_out * 2 + p2_out) * 4 + p1_in * 2 + p2_in];
                                amp = amp + g * theta[row_in * m_cols + col_in];
                            }
                        }
                        theta_prime[row_out * m_cols + col_out] = amp;
                    }
                }
            }
        }

        // ── Step 3: SVD + truncate → update tensors ──────────────────
        let max_rank = self.max_bond_dim.min(m_rows).min(m_cols);
        let (u, sigma, v) = complex_svd_thin(m_rows, m_cols, &theta_prime, max_rank);

        // Number of kept singular values
        let chi_new = sigma.iter().filter(|&&s| s > 1e-14).count().max(1).min(max_rank);

        // New left tensor: A[k][l, p1, r_new] = U[r_new * m_rows + (l*2+p1)]
        let mut new_left = SiteTensor::new(chi_l, chi_new);
        for l in 0..chi_l {
            for p1 in 0..2 {
                let row = l * 2 + p1;
                for r_new in 0..chi_new {
                    new_left.set(l, p1, r_new, u[r_new * m_rows + row]);
                }
            }
        }

        // New right tensor: A[k+1][r_new, p2, r] = σ[r_new] · V[r_new*m_cols + (p2*chi_r+r)]*
        // SVD: theta = U σ V† so theta[i,j] = Σ_k U[i,k]·σ[k]·conj(V[j,k])
        let mut new_right = SiteTensor::new(chi_new, chi_r);
        for r_new in 0..chi_new {
            for p2 in 0..2 {
                for r in 0..chi_r {
                    let col = p2 * chi_r + r;
                    new_right.set(r_new, p2, r,
                        v[r_new * m_cols + col].conj().scale(sigma[r_new]));
                }
            }
        }

        self.tensors[k]     = new_left;
        self.tensors[k + 1] = new_right;
    }

    // ── Two-qubit gate (arbitrary order — SWAP routing) ──────────────────

    fn apply_two_qubit(&mut self, site_a: usize, site_b: usize, gate: &[Complex; 16]) {
        if site_a == site_b { return; }
        let (lo, hi, swapped) = if site_a < site_b {
            (site_a, site_b, false)
        } else {
            (site_b, site_a, true)
        };

        // Bring hi down to lo+1 by SWAP-routing (each swap moves hi leftward by 1)
        // SWAP(pos, pos+1) = apply_two_qubit_adjacent(pos, SWAP)
        for pos in (lo + 1..hi).rev() {
            self.apply_two_qubit_adjacent(pos, &SWAP_GATE);
        }

        // Apply gate (possibly with swapped qubit ordering)
        let effective_gate = if swapped { swap_qubit_order(gate) } else { *gate };
        self.apply_two_qubit_adjacent(lo, &effective_gate);

        // Undo SWAP routing (restore hi back from lo+1 to its original position)
        for pos in lo + 1..hi {
            self.apply_two_qubit_adjacent(pos, &SWAP_GATE);
        }
    }

    // ── Full state contraction ────────────────────────────────────────────

    /// Contract the MPS to obtain the full 2ⁿ amplitude vector.
    /// Only practical for n ≤ MAX_QUBITS_FULL_CONTRACTION.
    pub fn to_statevector(&self) -> Vec<Complex> {
        // c[basis] = Vec<Complex>[chi_right] — running contraction
        let mut c: Vec<Vec<Complex>> = vec![vec![Complex::one()]];

        for k in 0..self.num_qubits {
            let t       = &self.tensors[k];
            let chi_r   = t.chi_right;
            let num_old = c.len();
            let mut c_new = vec![vec![Complex::zero(); chi_r]; num_old * 2];

            for (old_basis, left_vec) in c.iter().enumerate() {
                for phys in 0..2 {
                    let new_basis = old_basis * 2 + phys;
                    for r in 0..chi_r {
                        let mut amp = Complex::zero();
                        for l in 0..left_vec.len() {
                            amp = amp + left_vec[l] * t.get(l, phys, r);
                        }
                        c_new[new_basis][r] = c_new[new_basis][r] + amp;
                    }
                }
            }
            c = c_new;
        }

        c.iter().map(|v| v[0]).collect()
    }

    // ── Probabilities ─────────────────────────────────────────────────────

    /// Full probability distribution.  Only feasible for n ≤ MAX_QUBITS_FULL_CONTRACTION.
    pub fn probabilities(&self) -> Vec<f64> {
        self.to_statevector().iter().map(|a| a.norm_sq()).collect()
    }

    /// Marginal probability P(qubit `q` = |1⟩) via left/right environment contraction.
    pub fn marginal_prob_one(&self, q: usize) -> f64 {
        // Left environments: L[k] is (chi_k, chi_k) density matrix
        // L[k][m, m'] = Σ_{l,l'} L[k-1][l,l'] Σ_p A[k][l,p,m] A*[k][l',p,m']
        let n = self.num_qubits;

        // Build left env up to site q-1
        let mut left = vec![Complex::one()]; // 1×1: L[-1][0,0] = 1
        let mut left_chi = 1usize;
        for k in 0..q {
            let t = &self.tensors[k];
            let chi_r = t.chi_right;
            let mut new_left = vec![Complex::zero(); chi_r * chi_r];
            for m in 0..chi_r {
                for mp in 0..chi_r {
                    let mut val = Complex::zero();
                    for l in 0..left_chi {
                        for lp in 0..left_chi {
                            let env = left[l * left_chi + lp];
                            for p in 0..2 {
                                val = val + env * t.get(l, p, m) * t.get(lp, p, mp).conj();
                            }
                        }
                    }
                    new_left[m * chi_r + mp] = val;
                }
            }
            left = new_left;
            left_chi = chi_r;
        }

        // Build right env from n-1 down to q+1
        let mut right = vec![Complex::one()];
        let mut right_chi = 1usize;
        for k in (q + 1..n).rev() {
            let t = &self.tensors[k];
            let chi_l = t.chi_left;
            let mut new_right = vec![Complex::zero(); chi_l * chi_l];
            for l in 0..chi_l {
                for lp in 0..chi_l {
                    let mut val = Complex::zero();
                    for m in 0..right_chi {
                        for mp in 0..right_chi {
                            let env = right[m * right_chi + mp];
                            for p in 0..2 {
                                val = val + env * t.get(l, p, m) * t.get(lp, p, mp).conj();
                            }
                        }
                    }
                    new_right[l * chi_l + lp] = val;
                }
            }
            right = new_right;
            right_chi = chi_l;
        }

        // Contract at site q for p=1
        let t = &self.tensors[q];
        let chi_r = t.chi_right;
        let mut prob = 0.0_f64;
        for l in 0..left_chi {
            for lp in 0..left_chi {
                let lenv = left[l * left_chi + lp];
                for m in 0..chi_r {
                    for mp in 0..chi_r {
                        let renv = right[m * right_chi + mp];
                        let amp = t.get(l, 1, m) * t.get(lp, 1, mp).conj();
                        prob += (lenv * amp * renv).re;
                    }
                }
            }
        }
        prob.clamp(0.0, 1.0)
    }

    // ── Measurement / projection ──────────────────────────────────────────

    /// Project qubit `q` onto |`outcome`⟩ and renormalise.
    fn project_qubit(&mut self, q: usize, outcome: bool) {
        let p = outcome as usize;
        let t = &mut self.tensors[q];
        // Zero out the opposite physical index
        let chi_l = t.chi_left;
        let chi_r = t.chi_right;
        for l in 0..chi_l {
            for r in 0..chi_r {
                t.set(l, 1 - p, r, Complex::zero());
            }
        }

        // Renormalise MPS via full norm for small systems, or local norm otherwise
        let norm_sq: f64 = if self.num_qubits <= MAX_QUBITS_FULL_CONTRACTION {
            self.to_statevector().iter().map(|a| a.norm_sq()).sum()
        } else {
            // Approximate: use local tensor norm (good for right-orthogonal MPS)
            self.tensors[q].data.iter().map(|a| a.norm_sq()).sum()
        };
        if norm_sq > 1e-14 {
            // Scale only the FIRST tensor — in an n-tensor MPS the amplitude is a
            // product A[0]·A[1]·…·A[n-1]; scaling one tensor scales the product once.
            let inv = 1.0 / norm_sq.sqrt();
            for a in &mut self.tensors[0].data { *a = a.scale(inv); }
        }
    }

    /// Measure qubit `q` (collapses state).  `rng` ∈ [0, 1).
    pub fn measure(&mut self, q: usize, rng: f64) -> bool {
        let prob_one = if self.num_qubits <= MAX_QUBITS_FULL_CONTRACTION {
            let sv = self.to_statevector();
            (0..(1 << self.num_qubits))
                .filter(|&i| (i >> q) & 1 == 1)
                .map(|i| sv[i].norm_sq())
                .sum::<f64>()
        } else {
            self.marginal_prob_one(q)
        };
        let outcome = rng < prob_one;
        self.project_qubit(q, outcome);
        outcome
    }
}

// ── Gate constants ────────────────────────────────────────────────────────

const fn c(re: f64, im: f64) -> Complex { Complex::new(re, im) }

const SWAP_GATE: [Complex; 16] = [
    // |00⟩→|00⟩, |01⟩→|10⟩, |10⟩→|01⟩, |11⟩→|11⟩
    // Row-major, indices (p1_out*2+p2_out)*4 + p1_in*2+p2_in
    c(1.0,0.0), c(0.0,0.0), c(0.0,0.0), c(0.0,0.0),
    c(0.0,0.0), c(0.0,0.0), c(1.0,0.0), c(0.0,0.0),
    c(0.0,0.0), c(1.0,0.0), c(0.0,0.0), c(0.0,0.0),
    c(0.0,0.0), c(0.0,0.0), c(0.0,0.0), c(1.0,0.0),
];

/// Re-index a 4×4 gate to swap control/target qubit ordering.
fn swap_qubit_order(gate: &[Complex; 16]) -> [Complex; 16] {
    let mut g = [Complex::zero(); 16];
    for p1_out in 0..2 { for p2_out in 0..2 { for p1_in in 0..2 { for p2_in in 0..2 {
        // Original acts as: gate[(p1_out,p2_out),(p1_in,p2_in)]
        // Swapped acts as: gate[(p2_out,p1_out),(p2_in,p1_in)]
        let src = (p2_out * 2 + p1_out) * 4 + p2_in * 2 + p1_in;
        let dst = (p1_out * 2 + p2_out) * 4 + p1_in * 2 + p2_in;
        g[dst] = gate[src];
    }}}}
    g
}

/// Convert a Matrix2x2 unitary to a 4×4 gate (tensor product U ⊗ I or by index).
/// Build CNOT in the (control, target) basis.
fn cnot_gate() -> [Complex; 16] {
    // CNOT: index = (p1_out*2+p2_out)*4 + p1_in*2+p2_in
    // |00⟩→|00⟩: out=(0,0), in=(0,0): 0*4+0=0
    // |01⟩→|01⟩: out=(0,1), in=(0,1): 1*4+1=5
    // |10⟩→|11⟩: out=(1,1), in=(1,0): 3*4+2=14
    // |11⟩→|10⟩: out=(1,0), in=(1,1): 2*4+3=11
    let mut g = [Complex::zero(); 16];
    g[0]  = c(1.0, 0.0);
    g[5]  = c(1.0, 0.0);
    g[14] = c(1.0, 0.0);
    g[11] = c(1.0, 0.0);
    g
}

fn cz_gate() -> [Complex; 16] {
    let mut g = [Complex::zero(); 16];
    // CZ: |00⟩→|00⟩, |01⟩→|01⟩, |10⟩→|10⟩, |11⟩→-|11⟩
    g[0]  = c(1.0,0.0);
    g[5]  = c(1.0,0.0);
    g[10] = c(1.0,0.0);
    g[15] = c(-1.0,0.0);
    g
}

/// Build a two-qubit gate as U_a ⊗ U_b acting on site a then site b.
/// Only used for decomposable gates; for entangling gates use specific constructions.
#[allow(dead_code)]
fn tensor_product_gate(ua: &Matrix2x2, ub: &Matrix2x2) -> [Complex; 16] {
    let mut g = [Complex::zero(); 16];
    for p1_out in 0..2 { for p2_out in 0..2 { for p1_in in 0..2 { for p2_in in 0..2 {
        let idx = (p1_out * 2 + p2_out) * 4 + p1_in * 2 + p2_in;
        g[idx] = ua[p1_out][p1_in] * ub[p2_out][p2_in];
    }}}}
    g
}

// ── MPS Executor ──────────────────────────────────────────────────────────

/// Execute an AQL Program using the MPS backend.
///
/// Returns an `ExecutionResult` compatible with the statevector executor,
/// with full probabilities for n ≤ 24 and measurement-only for larger circuits.
pub fn execute_mps(program: &Program, bond_dim: usize) -> Result<ExecutionResult, AqlError> {
    let n = program.num_qubits;
    let mut state = MpsState::new(n, bond_dim);
    let mut measurements: Vec<MeasurementRecord> = Vec::new();
    let mut classical: HashMap<usize, bool> = HashMap::new();
    let mut gate_count = 0usize;
    let mut branch_count = 0usize;
    let mut steps = 0usize;
    let mut pre_probs: Option<Vec<f64>> = None;
    let mut first_measure = true;

    // Build label → PC table
    let mut labels: HashMap<String, usize> = HashMap::new();
    for (pc, instr) in program.instructions.iter().enumerate() {
        if let Instruction::Label(name) = instr {
            labels.insert(name.clone(), pc);
        }
    }

    let instrs = &program.instructions;
    let mut pc = 0usize;
    const MAX_STEPS: usize = 1_000_000;

    while pc < instrs.len() {
        if steps > MAX_STEPS {
            return Err(AqlError::Runtime {
                msg: format!("exceeded MAX_STEPS ({MAX_STEPS}) — possible infinite loop"),
            });
        }
        steps += 1;

        match &instrs[pc] {
            Instruction::H(q) => {
                state.apply_single_qubit(*q, &gates::hadamard());
                gate_count += 1;
            }
            Instruction::X(q) => {
                state.apply_single_qubit(*q, &gates::pauli_x());
                gate_count += 1;
            }
            Instruction::Y(q) => {
                state.apply_single_qubit(*q, &gates::pauli_y());
                gate_count += 1;
            }
            Instruction::Z(q) => {
                state.apply_single_qubit(*q, &gates::pauli_z());
                gate_count += 1;
            }
            Instruction::S(q) => {
                state.apply_single_qubit(*q, &gates::s_gate());
                gate_count += 1;
            }
            Instruction::T(q) => {
                state.apply_single_qubit(*q, &gates::t_gate());
                gate_count += 1;
            }
            Instruction::Rx { qubit, theta } => {
                state.apply_single_qubit(*qubit, &gates::rx(*theta));
                gate_count += 1;
            }
            Instruction::Ry { qubit, theta } => {
                state.apply_single_qubit(*qubit, &gates::ry(*theta));
                gate_count += 1;
            }
            Instruction::Rz { qubit, theta } => {
                state.apply_single_qubit(*qubit, &gates::rz(*theta));
                gate_count += 1;
            }
            Instruction::Phase { qubit, theta } => {
                state.apply_single_qubit(*qubit, &gates::phase_gate(*theta));
                gate_count += 1;
            }
            Instruction::Cnot { control, target } => {
                state.apply_two_qubit(*control, *target, &cnot_gate());
                gate_count += 1;
            }
            Instruction::Cz { control, target } => {
                state.apply_two_qubit(*control, *target, &cz_gate());
                gate_count += 1;
            }
            Instruction::Swap { qubit_a, qubit_b } => {
                state.apply_two_qubit(*qubit_a, *qubit_b, &SWAP_GATE);
                gate_count += 1;
            }
            Instruction::Toffoli { control0, control1, target } => {
                // Decompose Toffoli into single/two-qubit gates
                apply_toffoli_mps(&mut state, *control0, *control1, *target);
                gate_count += 1;
            }
            Instruction::Measure(q) => {
                if first_measure && n <= MAX_QUBITS_FULL_CONTRACTION {
                    pre_probs = Some(state.probabilities());
                    first_measure = false;
                }
                let outcome = state.measure(*q, rand::random::<f64>());
                classical.insert(*q, outcome);
                measurements.push(MeasurementRecord { qubit: *q, outcome, step: pc });
            }
            Instruction::MeasureAll => {
                if first_measure && n <= MAX_QUBITS_FULL_CONTRACTION {
                    pre_probs = Some(state.probabilities());
                    first_measure = false;
                }
                for q in 0..n {
                    let outcome = state.measure(q, rand::random::<f64>());
                    classical.insert(q, outcome);
                    measurements.push(MeasurementRecord { qubit: q, outcome, step: pc });
                }
            }
            Instruction::Label(_) | Instruction::Barrier => {}
            Instruction::Goto { label } => {
                if let Some(&tgt) = labels.get(label) { pc = tgt; continue; }
                return Err(AqlError::Runtime { msg: format!("undefined label '{label}'") });
            }
            Instruction::GotoIf { qubit, label } => {
                let val = classical.get(qubit).copied().ok_or_else(|| AqlError::Runtime {
                    msg: format!("IF: qubit {qubit} has not been measured"),
                })?;
                if val {
                    branch_count += 1;
                    if let Some(&tgt) = labels.get(label) { pc = tgt; continue; }
                    return Err(AqlError::Runtime { msg: format!("undefined label '{label}'") });
                }
            }
            Instruction::GotoIfNot { qubit, label } => {
                let val = classical.get(qubit).copied().ok_or_else(|| AqlError::Runtime {
                    msg: format!("IFNOT: qubit {qubit} has not been measured"),
                })?;
                if !val {
                    branch_count += 1;
                    if let Some(&tgt) = labels.get(label) { pc = tgt; continue; }
                    return Err(AqlError::Runtime { msg: format!("undefined label '{label}'") });
                }
            }
            Instruction::CallGate { name, qubits } => {
                if let Some(gate_def) = program.gate_defs.get(name) {
                    apply_gate_def_mps(&mut state, gate_def, qubits, &program.gate_defs)?;
                    gate_count += 1;
                } else {
                    return Err(AqlError::Runtime {
                        msg: format!("undefined gate '{name}'"),
                    });
                }
            }
            Instruction::MeasureInto { .. } | Instruction::GotoIfCreg { .. }
            | Instruction::GotoIfNotCreg { .. } => {
                return Err(AqlError::Runtime {
                    msg: "CREG instructions not supported in MPS backend; use --backend statevector".to_string(),
                });
            }
        }
        pc += 1;
    }

    let final_probabilities = if n <= MAX_QUBITS_FULL_CONTRACTION {
        state.probabilities()
    } else {
        // For large n: return per-qubit marginal probabilities in a flat format
        // (not a full 2^n distribution — use measurements for outcomes)
        vec![]
    };

    Ok(ExecutionResult {
        num_qubits: n,
        measurements,
        pre_measurement_probs: pre_probs,
        pre_measurement_amplitudes: None,
        final_probabilities,
        final_amplitudes: vec![], // MPS: amplitudes not fully contracted
        gate_count,
        branch_count,
        steps_executed: steps,
    })
}

/// Toffoli decomposition for MPS.
/// Standard decomposition into 1-qubit + CNOT gates (7 gates).
fn apply_toffoli_mps(state: &mut MpsState, c0: usize, c1: usize, tgt: usize) {
    state.apply_single_qubit(tgt, &gates::hadamard());
    state.apply_two_qubit(c1, tgt, &cnot_gate());
    state.apply_single_qubit(tgt, &gates::t_dagger_gate());
    state.apply_two_qubit(c0, tgt, &cnot_gate());
    state.apply_single_qubit(tgt, &gates::t_gate());
    state.apply_two_qubit(c1, tgt, &cnot_gate());
    state.apply_single_qubit(tgt, &gates::t_dagger_gate());
    state.apply_two_qubit(c0, tgt, &cnot_gate());
    state.apply_single_qubit(c1, &gates::t_dagger_gate());
    state.apply_single_qubit(tgt, &gates::t_gate());
    state.apply_single_qubit(tgt, &gates::hadamard());
    state.apply_two_qubit(c0, c1, &cnot_gate());
    state.apply_single_qubit(c0, &gates::t_gate());
    state.apply_single_qubit(c1, &gates::t_dagger_gate());
    state.apply_two_qubit(c0, c1, &cnot_gate());
}

fn apply_gate_def_mps(
    state:     &mut MpsState,
    gate_def:  &crate::compiler::ir::GateDef,
    qubits:    &[usize],
    gate_defs: &std::collections::HashMap<String, crate::compiler::ir::GateDef>,
) -> Result<(), AqlError> {
    for instr in &gate_def.body {
        // Remap local qubit indices to global
        let remap = |local: usize| -> usize { qubits[local] };
        match instr {
            Instruction::H(q)     => state.apply_single_qubit(remap(*q), &gates::hadamard()),
            Instruction::X(q)     => state.apply_single_qubit(remap(*q), &gates::pauli_x()),
            Instruction::Y(q)     => state.apply_single_qubit(remap(*q), &gates::pauli_y()),
            Instruction::Z(q)     => state.apply_single_qubit(remap(*q), &gates::pauli_z()),
            Instruction::S(q)     => state.apply_single_qubit(remap(*q), &gates::s_gate()),
            Instruction::T(q)     => state.apply_single_qubit(remap(*q), &gates::t_gate()),
            Instruction::Rx { qubit, theta } => state.apply_single_qubit(remap(*qubit), &gates::rx(*theta)),
            Instruction::Ry { qubit, theta } => state.apply_single_qubit(remap(*qubit), &gates::ry(*theta)),
            Instruction::Rz { qubit, theta } => state.apply_single_qubit(remap(*qubit), &gates::rz(*theta)),
            Instruction::Phase { qubit, theta } => state.apply_single_qubit(remap(*qubit), &gates::phase_gate(*theta)),
            Instruction::Cnot { control, target } => state.apply_two_qubit(remap(*control), remap(*target), &cnot_gate()),
            Instruction::Cz   { control, target } => state.apply_two_qubit(remap(*control), remap(*target), &cz_gate()),
            Instruction::Swap { qubit_a, qubit_b } => state.apply_two_qubit(remap(*qubit_a), remap(*qubit_b), &SWAP_GATE),
            Instruction::Toffoli { control0, control1, target } =>
                apply_toffoli_mps(state, remap(*control0), remap(*control1), remap(*target)),
            Instruction::CallGate { name, qubits: sub_q } => {
                let sub_qubits: Vec<usize> = sub_q.iter().map(|&l| remap(l)).collect();
                if let Some(gd) = gate_defs.get(name) {
                    apply_gate_def_mps(state, gd, &sub_qubits, gate_defs)?;
                } else {
                    return Err(AqlError::Runtime { msg: format!("undefined gate '{name}'") });
                }
            }
            _ => {} // BARRIER, LABEL, etc. — no-op
        }
    }
    Ok(())
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn nearly(a: f64, b: f64) -> bool { (a - b).abs() < 1e-6 }

    #[test]
    fn test_mps_initial_state_is_zero() {
        let mps = MpsState::new(2, 16);
        let sv = mps.to_statevector();
        assert_eq!(sv.len(), 4);
        assert!(nearly(sv[0].re, 1.0), "|00⟩ amplitude should be 1");
        assert!(nearly(sv[1].re, 0.0));
        assert!(nearly(sv[2].re, 0.0));
        assert!(nearly(sv[3].re, 0.0));
    }

    #[test]
    fn test_mps_h_gate_superposition() {
        let mut mps = MpsState::new(1, 16);
        mps.apply_single_qubit(0, &gates::hadamard());
        let sv = mps.to_statevector();
        let s2 = 1.0 / 2.0_f64.sqrt();
        assert!(nearly(sv[0].re, s2));
        assert!(nearly(sv[1].re, s2));
    }

    #[test]
    fn test_mps_x_gate() {
        let mut mps = MpsState::new(1, 16);
        mps.apply_single_qubit(0, &gates::pauli_x());
        let sv = mps.to_statevector();
        assert!(nearly(sv[0].re, 0.0));
        assert!(nearly(sv[1].re, 1.0));
    }

    #[test]
    fn test_mps_bell_state() {
        let mut mps = MpsState::new(2, 16);
        mps.apply_single_qubit(0, &gates::hadamard());
        mps.apply_two_qubit(0, 1, &cnot_gate());
        let probs = mps.probabilities();
        assert!(nearly(probs[0], 0.5), "P(|00⟩) = {}", probs[0]);
        assert!(nearly(probs[1], 0.0), "P(|01⟩) = {}", probs[1]);
        assert!(nearly(probs[2], 0.0), "P(|10⟩) = {}", probs[2]);
        assert!(nearly(probs[3], 0.5), "P(|11⟩) = {}", probs[3]);
    }

    #[test]
    fn test_mps_ghz_3qubits() {
        let mut mps = MpsState::new(3, 16);
        mps.apply_single_qubit(0, &gates::hadamard());
        mps.apply_two_qubit(0, 1, &cnot_gate());
        mps.apply_two_qubit(0, 2, &cnot_gate());
        let probs = mps.probabilities();
        assert!(nearly(probs[0], 0.5), "P(|000⟩) = {}", probs[0]);
        assert!(nearly(probs[7], 0.5), "P(|111⟩) = {}", probs[7]);
    }

    #[test]
    fn test_mps_execute_bell_aql() {
        let src = "QREG 2\nH 0\nCNOT 0 1\nMEASURE_ALL";
        let prog = crate::compiler::parse_source(src).unwrap();
        let result = execute_mps(&prog, 16).unwrap();
        assert_eq!(result.measurements.len(), 2);
        let q0 = result.measurements[0].outcome;
        let q1 = result.measurements[1].outcome;
        assert_eq!(q0, q1, "Bell state: q0 and q1 should be equal");
    }

    #[test]
    fn test_mps_measure_collapses() {
        let mut mps = MpsState::new(2, 16);
        mps.apply_single_qubit(0, &gates::hadamard());
        mps.apply_two_qubit(0, 1, &cnot_gate());
        // After Bell state, measure qubit 0
        let _outcome = mps.measure(0, 0.3); // rng < 0.5 → outcome = false
        // Total probability must still sum to 1
        let total: f64 = mps.probabilities().iter().sum();
        assert!((total - 1.0).abs() < 1e-6, "total prob = {total}");
    }

    #[test]
    fn test_mps_non_adjacent_cnot() {
        // CNOT(0, 2) in a 3-qubit system — non-adjacent
        let mut mps = MpsState::new(3, 16);
        mps.apply_single_qubit(0, &gates::pauli_x()); // |100⟩
        mps.apply_two_qubit(0, 2, &cnot_gate());      // should flip q2 → |101⟩
        let probs = mps.probabilities();
        // |101⟩ = index 5 (binary 101)
        assert!(nearly(probs[5], 1.0), "P(|101⟩) = {}", probs[5]);
    }

    #[test]
    fn test_mps_marginal_prob_bell() {
        let mut mps = MpsState::new(4, 16);
        mps.apply_single_qubit(0, &gates::hadamard());
        mps.apply_two_qubit(0, 1, &cnot_gate());
        // Qubits 0 and 1 in Bell state; qubits 2, 3 in |0⟩
        let p1 = mps.marginal_prob_one(0);
        let p2 = mps.marginal_prob_one(1);
        let p3 = mps.marginal_prob_one(2);
        assert!((p1 - 0.5).abs() < 1e-6, "P(q0=1) = {p1}");
        assert!((p2 - 0.5).abs() < 1e-6, "P(q1=1) = {p2}");
        assert!(p3 < 1e-10, "P(q2=1) = {p3}");
    }
}
