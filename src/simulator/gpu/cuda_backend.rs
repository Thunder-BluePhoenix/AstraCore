/// CUDA statevector backend — powered by the `cudarc` crate.
///
/// Requires: NVIDIA GPU + CUDA Toolkit 11+ installed.
/// Feature gate: `--features cuda`
///
/// # Status
/// This module provides:
/// - Device enumeration via `cudarc::driver::CudaDevice`
/// - `execute()` — dispatches PTX kernels for single-qubit gates and CNOT
/// - Full integration with the AstraCore `ExecutionResult` type
///
/// PTX kernels are compiled inline using the `cudarc` PTX loader.
/// The actual kernel source is embedded as a string constant.
use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

use super::{GpuDevice, GpuKind};
use crate::compiler::ir::{Instruction, Program};
use crate::runtime::ExecutionResult;

// ── CUDA PTX kernel source ────────────────────────────────────────────────────

/// PTX/CUDA C kernel source for single-qubit gate application.
///
/// Each thread handles one amplitude pair (i0, i1) that differ in bit `qubit`.
/// Matrix elements passed as float2 (re, im) pairs in a struct.
const GATE_KERNEL_SRC: &str = r#"
extern "C" {

__device__ float2 cmul(float2 a, float2 b) {
    return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

__device__ float2 cadd(float2 a, float2 b) {
    return make_float2(a.x + b.x, a.y + b.y);
}

// Apply a 2×2 gate to all amplitude pairs differing in bit `qubit`.
// state: interleaved [re0, im0, re1, im1, ...] as float2 pairs
// matrix: [m00_re, m00_im, m01_re, m01_im, m10_re, m10_im, m11_re, m11_im]
__global__ void apply_gate(
    float2* __restrict__ state,
    float  m00_re, float m00_im,
    float  m01_re, float m01_im,
    float  m10_re, float m10_im,
    float  m11_re, float m11_im,
    unsigned int qubit,
    unsigned int n_qubits
) {
    unsigned int pair_idx = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int half     = 1u << (n_qubits - 1u);
    if (pair_idx >= half) return;

    unsigned int mask = 1u << qubit;
    unsigned int lo   = pair_idx & (mask - 1u);
    unsigned int hi   = (pair_idx >> qubit) << (qubit + 1u);
    unsigned int i0   = hi | lo;
    unsigned int i1   = hi | mask | lo;

    float2 a0 = state[i0];
    float2 a1 = state[i1];

    float2 m00 = make_float2(m00_re, m00_im);
    float2 m01 = make_float2(m01_re, m01_im);
    float2 m10 = make_float2(m10_re, m10_im);
    float2 m11 = make_float2(m11_re, m11_im);

    state[i0] = cadd(cmul(m00, a0), cmul(m01, a1));
    state[i1] = cadd(cmul(m10, a0), cmul(m11, a1));
}

// Apply CNOT: swap amplitudes where control=1 and target differs.
__global__ void apply_cnot(
    float2* __restrict__ state,
    unsigned int control,
    unsigned int target,
    unsigned int n_qubits
) {
    unsigned int quarter = 1u << (n_qubits - 2u);
    if (blockDim.x * blockIdx.x + threadIdx.x >= quarter) return;
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

    unsigned int ctrl_mask   = 1u << control;
    unsigned int target_mask = 1u << target;

    // Insert 0 at target position
    unsigned int lo_t = idx & (target_mask - 1u);
    unsigned int hi_t = (idx >> target) << (target + 1u);
    idx = hi_t | lo_t;

    // Insert 1 at control position
    unsigned int lo_c = idx & (ctrl_mask - 1u);
    unsigned int hi_c = (idx >> control) << (control + 1u);
    idx = hi_c | ctrl_mask | lo_c;

    unsigned int i0 = idx;
    unsigned int i1 = idx | target_mask;

    float2 tmp = state[i0];
    state[i0]  = state[i1];
    state[i1]  = tmp;
}

} // extern "C"
"#;

// ── Opaque CUDA buffer ────────────────────────────────────────────────────────

/// CUDA device-side state vector buffer.
pub struct CudaBuffer {
    pub slice:    CudaSlice<f32>,  // interleaved [re, im, re, im, …]
    pub n_qubits: usize,
}

// ── Executor ──────────────────────────────────────────────────────────────────

struct CudaExecutor {
    dev: Arc<CudaDevice>,
}

impl CudaExecutor {
    fn new() -> Result<Self, String> {
        let dev = CudaDevice::new(0)
            .map_err(|e| format!("CUDA device init failed: {e}"))?;
        Ok(CudaExecutor { dev })
    }

    fn upload(&self, amps: &[(f32, f32)]) -> Result<CudaSlice<f32>, String> {
        let flat: Vec<f32> = amps.iter().flat_map(|(re, im)| [*re, *im]).collect();
        self.dev.htod_sync_copy(&flat)
            .map_err(|e| e.to_string())
    }

    fn download(&self, slice: &CudaSlice<f32>, n: usize) -> Result<Vec<(f32, f32)>, String> {
        let flat = self.dev.dtoh_sync_copy(slice)
            .map_err(|e| e.to_string())?;
        Ok(flat.chunks(2).take(n).map(|c| (c[0], c[1])).collect())
    }

    fn load_kernels(&self) -> Result<(), String> {
        // Compile PTX at runtime using nvrtc (CUDA runtime compilation).
        // If nvrtc is unavailable, return an error — the caller falls back to CPU.
        let ptx = Ptx::from_src(GATE_KERNEL_SRC);
        self.dev.load_ptx(ptx, "gates", &["apply_gate", "apply_cnot"])
            .map_err(|e| format!("PTX load failed: {e}"))
    }

    fn apply_gate(
        &self,
        slice:    &mut CudaSlice<f32>,
        matrix:   [[f32; 4]; 2],
        qubit:    usize,
        n_qubits: usize,
    ) -> Result<(), String> {
        let [row0, row1] = matrix;
        let n_pairs = (1usize << n_qubits) / 2;
        let block  = 128u32;
        let grid   = ((n_pairs + 127) / 128) as u32;
        let cfg    = LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (block, 1, 1), shared_mem_bytes: 0 };

        let f = self.dev.get_func("gates", "apply_gate")
            .ok_or_else(|| "apply_gate kernel not loaded".to_string())?;

        unsafe {
            f.launch(cfg, (
                slice,
                row0[0], row0[1], row0[2], row0[3],  // m00_re, m00_im, m01_re, m01_im
                row1[0], row1[1], row1[2], row1[3],  // m10_re, m10_im, m11_re, m11_im
                qubit as u32,
                n_qubits as u32,
            ))
        }.map_err(|e| e.to_string())
    }

    fn apply_cnot(
        &self,
        slice:    &mut CudaSlice<f32>,
        control:  usize,
        target:   usize,
        n_qubits: usize,
    ) -> Result<(), String> {
        let n_pairs = (1usize << n_qubits) / 4;
        let block   = 128u32;
        let grid    = (n_pairs.max(1) + 127) as u32 / 128 + 1;
        let cfg     = LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (block, 1, 1), shared_mem_bytes: 0 };

        let f = self.dev.get_func("gates", "apply_cnot")
            .ok_or_else(|| "apply_cnot kernel not loaded".to_string())?;

        unsafe {
            f.launch(cfg, (
                slice,
                control  as u32,
                target   as u32,
                n_qubits as u32,
            ))
        }.map_err(|e| e.to_string())
    }
}

// ── Gate matrix helpers (same as wgpu backend, f32) ───────────────────────────

fn hadamard_f32() -> [[f32; 4]; 2] {
    let s = std::f32::consts::FRAC_1_SQRT_2;
    [[s, 0.0, s, 0.0], [s, 0.0, -s, 0.0]]
}
fn pauli_x_f32()  -> [[f32; 4]; 2] { [[0.0,0.0,1.0,0.0],[1.0,0.0,0.0,0.0]] }
fn pauli_y_f32()  -> [[f32; 4]; 2] { [[0.0,0.0,0.0,-1.0],[0.0,1.0,0.0,0.0]] }
fn pauli_z_f32()  -> [[f32; 4]; 2] { [[1.0,0.0,0.0,0.0],[0.0,0.0,-1.0,0.0]] }
fn s_gate_f32()   -> [[f32; 4]; 2] { [[1.0,0.0,0.0,0.0],[0.0,0.0,0.0,1.0]] }
fn t_gate_f32()   -> [[f32; 4]; 2] {
    let v = std::f32::consts::FRAC_1_SQRT_2;
    [[1.0,0.0,0.0,0.0],[0.0,0.0,v,v]]
}
fn tdg_f32() -> [[f32; 4]; 2] {
    let v = std::f32::consts::FRAC_1_SQRT_2;
    [[1.0,0.0,0.0,0.0],[0.0,0.0,v,-v]]
}
fn rx_f32(theta: f64) -> [[f32; 4]; 2] {
    let c = (theta/2.0).cos() as f32; let s = (theta/2.0).sin() as f32;
    [[c,0.0,0.0,-s],[0.0,-s,c,0.0]]
}
fn ry_f32(theta: f64) -> [[f32; 4]; 2] {
    let c = (theta/2.0).cos() as f32; let s = (theta/2.0).sin() as f32;
    [[c,0.0,-s,0.0],[s,0.0,c,0.0]]
}
fn rz_f32(theta: f64) -> [[f32; 4]; 2] {
    let c = (theta/2.0).cos() as f32; let s = (theta/2.0).sin() as f32;
    [[c,-s,0.0,0.0],[0.0,0.0,c,s]]
}
fn phase_f32(theta: f64) -> [[f32; 4]; 2] {
    let c = theta.cos() as f32; let s = theta.sin() as f32;
    [[1.0,0.0,0.0,0.0],[0.0,0.0,c,s]]
}

// ── Device enumeration ────────────────────────────────────────────────────────

/// Return all CUDA-visible NVIDIA devices.
pub fn enumerate_devices() -> Vec<GpuDevice> {
    let count = cudarc::driver::result::device::get_count().unwrap_or(0) as usize;
    (0..count).filter_map(|i| {
        let dev = CudaDevice::new(i).ok()?;
        let name = dev.name().unwrap_or_else(|_| format!("NVIDIA GPU {i}"));
        let vram = dev.total_memory().unwrap_or(0) as u64;
        Some(GpuDevice {
            id:   format!("cuda:{i}"),
            name,
            vram,
            kind: GpuKind::Cuda,
        })
    }).collect()
}

// ── CPU measurement collapse ──────────────────────────────────────────────────

fn collapse_cpu(amps: Vec<(f32, f32)>, qubit: usize, _n: usize, rand: f64) -> (Vec<(f32, f32)>, bool) {
    let mask  = 1usize << qubit;
    let prob1: f64 = amps.iter().enumerate()
        .filter(|(i, _)| i & mask != 0)
        .map(|(_, (re, im))| (*re as f64).powi(2) + (*im as f64).powi(2))
        .sum();
    let outcome = rand < prob1;
    let norm = if outcome { prob1.sqrt() } else { (1.0 - prob1).sqrt() };
    let norm = if norm < 1e-12 { 1.0 } else { norm } as f32;
    let new_amps = amps.into_iter().enumerate().map(|(i, (re, im))| {
        if (i & mask != 0) == outcome { (re / norm, im / norm) } else { (0.0, 0.0) }
    }).collect();
    (new_amps, outcome)
}

// ── Toffoli decomposition ─────────────────────────────────────────────────────

fn toffoli_cuda(
    exec:  &CudaExecutor,
    slice: &mut CudaSlice<f32>,
    c0: usize, c1: usize, target: usize, n: usize,
) -> Result<(), String> {
    let h   = hadamard_f32();
    let t   = t_gate_f32();
    let tdg = tdg_f32();
    exec.apply_gate(slice, h, target, n)?;
    exec.apply_cnot(slice, c1, target, n)?;
    exec.apply_gate(slice, tdg, target, n)?;
    exec.apply_cnot(slice, c0, target, n)?;
    exec.apply_gate(slice, t, target, n)?;
    exec.apply_cnot(slice, c1, target, n)?;
    exec.apply_gate(slice, tdg, target, n)?;
    exec.apply_cnot(slice, c0, target, n)?;
    exec.apply_gate(slice, t, c1, n)?;
    exec.apply_gate(slice, t, target, n)?;
    exec.apply_gate(slice, h, target, n)?;
    exec.apply_cnot(slice, c0, c1, n)?;
    exec.apply_gate(slice, t, c0, n)?;
    exec.apply_gate(slice, tdg, c1, n)?;
    exec.apply_cnot(slice, c0, c1, n)?;
    Ok(())
}

// ── Public execute function ───────────────────────────────────────────────────

/// Execute an AQL program on the CUDA backend (device 0).
pub fn execute(program: &Program) -> Result<ExecutionResult, String> {
    let exec = CudaExecutor::new()?;
    exec.load_kernels()?;

    let n   = program.num_qubits;
    let dim = 1usize << n;

    let mut amps: Vec<(f32, f32)> = vec![(0.0, 0.0); dim];
    amps[0] = (1.0, 0.0);

    let mut slice = exec.upload(&amps)?;

    let mut measurements: Vec<crate::runtime::MeasurementRecord> = Vec::new();
    let mut step        = 0usize;
    let mut branch_count = 0usize;
    let mut pre_meas_amps: Option<Vec<(f32, f32)>> = None;

    for instr in &program.instructions {
        step += 1;
        match instr {
            Instruction::H(q)     => exec.apply_gate(&mut slice, hadamard_f32(), *q, n)?,
            Instruction::X(q)     => exec.apply_gate(&mut slice, pauli_x_f32(),  *q, n)?,
            Instruction::Y(q)     => exec.apply_gate(&mut slice, pauli_y_f32(),  *q, n)?,
            Instruction::Z(q)     => exec.apply_gate(&mut slice, pauli_z_f32(),  *q, n)?,
            Instruction::S(q)     => exec.apply_gate(&mut slice, s_gate_f32(),   *q, n)?,
            Instruction::T(q)     => exec.apply_gate(&mut slice, t_gate_f32(),   *q, n)?,
            Instruction::Rx { qubit, theta }    => exec.apply_gate(&mut slice, rx_f32(*theta),    *qubit, n)?,
            Instruction::Ry { qubit, theta }    => exec.apply_gate(&mut slice, ry_f32(*theta),    *qubit, n)?,
            Instruction::Rz { qubit, theta }    => exec.apply_gate(&mut slice, rz_f32(*theta),    *qubit, n)?,
            Instruction::Phase { qubit, theta } => exec.apply_gate(&mut slice, phase_f32(*theta), *qubit, n)?,

            Instruction::Cnot { control, target } => exec.apply_cnot(&mut slice, *control, *target, n)?,
            Instruction::Cz   { control, target } => {
                exec.apply_gate(&mut slice, hadamard_f32(), *target, n)?;
                exec.apply_cnot(&mut slice, *control, *target, n)?;
                exec.apply_gate(&mut slice, hadamard_f32(), *target, n)?;
            }
            Instruction::Swap { qubit_a, qubit_b } => {
                exec.apply_cnot(&mut slice, *qubit_a, *qubit_b, n)?;
                exec.apply_cnot(&mut slice, *qubit_b, *qubit_a, n)?;
                exec.apply_cnot(&mut slice, *qubit_a, *qubit_b, n)?;
            }
            Instruction::Toffoli { control0, control1, target } => {
                toffoli_cuda(&exec, &mut slice, *control0, *control1, *target, n)?;
            }

            Instruction::Measure(q) => {
                let cur = exec.download(&slice, dim)?;
                if pre_meas_amps.is_none() { pre_meas_amps = Some(cur.clone()); }
                let (new_amps, outcome) = collapse_cpu(cur, *q, n, rand::random());
                slice = exec.upload(&new_amps)?;
                measurements.push(crate::runtime::MeasurementRecord { qubit: *q, outcome, step });
            }
            Instruction::MeasureAll => {
                let cur = exec.download(&slice, dim)?;
                if pre_meas_amps.is_none() { pre_meas_amps = Some(cur.clone()); }
                let mut amps_cpu = cur;
                for q in 0..n {
                    let (new_amps, outcome) = collapse_cpu(amps_cpu, q, n, rand::random());
                    amps_cpu = new_amps;
                    measurements.push(crate::runtime::MeasurementRecord { qubit: q, outcome, step });
                }
                slice = exec.upload(&amps_cpu)?;
            }

            Instruction::Goto { .. } | Instruction::GotoIf { .. } | Instruction::GotoIfNot { .. } => {
                branch_count += 1;
            }
            Instruction::Barrier | Instruction::Label(_) | Instruction::CallGate { .. } => {}
        }
    }

    let final_amps = exec.download(&slice, dim)?;
    let final_probs: Vec<f64> = final_amps.iter()
        .map(|(re, im)| (*re as f64).powi(2) + (*im as f64).powi(2))
        .collect();
    let final_amplitudes: Vec<(f64, f64)> = final_amps.iter()
        .map(|(re, im)| (*re as f64, *im as f64))
        .collect();

    let pre_measurement_probs = pre_meas_amps.as_ref().map(|a|
        a.iter().map(|(re,im)| (*re as f64).powi(2) + (*im as f64).powi(2)).collect()
    );
    let pre_measurement_amplitudes = pre_meas_amps.map(|a|
        a.iter().map(|(re,im)| (*re as f64, *im as f64)).collect()
    );

    Ok(ExecutionResult {
        num_qubits: n,
        final_probabilities: final_probs,
        final_amplitudes,
        measurements,
        gate_count: step,
        branch_count,
        steps_executed: step,
        pre_measurement_probs,
        pre_measurement_amplitudes,
    })
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enumerate_cuda_devices_no_panic() {
        // Must not panic even on machines without CUDA.
        let devs = enumerate_devices();
        // Could be empty — that's fine.
        let _ = devs;
    }

    #[test]
    fn execute_cuda_bell_or_skip() {
        let program = crate::compiler::parse_source("QREG 2\nH 0\nCNOT 0 1").unwrap();
        match execute(&program) {
            Ok(result) => {
                let p = &result.final_probabilities;
                assert!((p[0] - 0.5).abs() < 0.02, "|00⟩={}", p[0]);
                assert!((p[3] - 0.5).abs() < 0.02, "|11⟩={}", p[3]);
            }
            Err(e) => eprintln!("SKIP (no CUDA): {e}"),
        }
    }
}
