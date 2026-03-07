/// GPU statevector simulation backend for AstraCore.
///
/// Two sub-backends, both behind optional Cargo features:
///
/// | Feature | Flag            | Works on                        |
/// |---------|-----------------|----------------------------------|
/// | `wgpu`  | `--backend wgpu`| NVIDIA / AMD / Intel / Apple (Vulkan/Metal) |
/// | `cuda`  | `--backend cuda`| NVIDIA GPU with CUDA Toolkit 11+ |
///
/// Usage:
/// ```bash
/// cargo build --release --features wgpu
/// astracore run circuit.aql --backend wgpu
///
/// cargo build --release --features cuda
/// astracore run circuit.aql --backend cuda
/// ```
///
/// Without any GPU feature flag, `list_devices()` returns only `"cpu"` and
/// `execute_gpu` / `execute_wgpu` / `execute_cuda` are not available.

#[cfg(feature = "wgpu")]
pub mod wgpu_backend;

#[cfg(feature = "cuda")]
pub mod cuda_backend;

#[cfg(any(feature = "wgpu", feature = "cuda"))]
use crate::compiler::ir::Program;
#[cfg(any(feature = "wgpu", feature = "cuda"))]
use crate::runtime::ExecutionResult;

// ── Device listing ────────────────────────────────────────────────────────────

/// A GPU device visible to AstraCore.
#[derive(Debug, Clone)]
pub struct GpuDevice {
    /// Unique identifier string, e.g. `"wgpu:0"` or `"cuda:0"`.
    pub id:    String,
    /// Human-readable device name, e.g. `"NVIDIA RTX 4090"`.
    pub name:  String,
    /// Approximate VRAM in bytes (0 if unknown).
    pub vram:  u64,
    /// Which sub-backend drives this device.
    pub kind:  GpuKind,
}

/// Which GPU runtime backs a [`GpuDevice`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuKind {
    Wgpu,
    Cuda,
}

impl std::fmt::Display for GpuDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let vram_str = if self.vram > 0 {
            format!("  ({} MB VRAM)", self.vram / 1_048_576)
        } else {
            String::new()
        };
        write!(f, "{}  {}{}", self.id, self.name, vram_str)
    }
}

/// List all available GPU devices plus the fallback CPU entry.
///
/// Always includes `"cpu"` as the last entry so callers can print a unified
/// device table.  Returns GPU devices only when the corresponding feature is
/// compiled in and at least one device is found at runtime.
pub fn list_devices() -> Vec<String> {
    let mut devices: Vec<String> = Vec::new();

    #[cfg(feature = "wgpu")]
    {
        for d in wgpu_backend::enumerate_devices() {
            devices.push(d.to_string());
        }
    }

    #[cfg(feature = "cuda")]
    {
        for d in cuda_backend::enumerate_devices() {
            devices.push(d.to_string());
        }
    }

    devices.push("cpu     (always available)".to_string());
    devices
}

// ── GpuBackend trait ──────────────────────────────────────────────────────────

/// Trait implemented by each GPU sub-backend.
///
/// The state vector is represented as interleaved `[re0, im0, re1, im1, …]` f32
/// values in device memory.  A gate is described by its 2×2 complex matrix as
/// four `(f32, f32)` tuples: `[[m00, m01], [m10, m11]]`.
pub trait GpuBackend: Send + Sync {
    /// Upload a CPU state vector onto the device, returning an opaque handle.
    fn upload(&self, amplitudes: &[(f32, f32)]) -> GpuBuffer;

    /// Apply a 2×2 single-qubit gate (column-major, row-major order: m00, m01, m10, m11)
    /// in-place on the device buffer.
    fn apply_gate(&self, buf: &mut GpuBuffer, matrix: [[f32; 4]; 2], qubit: usize);

    /// Apply a CNOT gate on `(control, target)` qubits in-place.
    fn apply_cnot(&self, buf: &mut GpuBuffer, control: usize, target: usize);

    /// Download the state vector back to CPU memory.
    fn download(&self, buf: &GpuBuffer) -> Vec<(f32, f32)>;

    /// Compute per-basis-state probabilities from the device buffer without
    /// full download.  Returns `2^n` values.
    fn probabilities(&self, buf: &GpuBuffer) -> Vec<f64>;

    /// Human-readable device name for diagnostics.
    fn device_name(&self) -> &str;
}

/// Opaque device-side state vector handle.
///
/// Internally holds whatever the sub-backend needs (e.g. a wgpu `Buffer`
/// or a CUDA device pointer).  Dropped when execution finishes.
pub struct GpuBuffer {
    /// Number of qubits (determines buffer size = 2^n complex numbers).
    pub n_qubits: usize,
    /// Sub-backend opaque payload.
    pub inner:    GpuBufferInner,
}

/// Sub-backend-specific storage.
pub enum GpuBufferInner {
    /// CPU-side fallback buffer (used in tests and when GPU is absent).
    Cpu(Vec<(f32, f32)>),
    /// wgpu device buffer (only present when `feature = "wgpu"`).
    #[cfg(feature = "wgpu")]
    Wgpu(wgpu_backend::WgpuBuffer),
    /// CUDA device buffer (only present when `feature = "cuda"`).
    #[cfg(feature = "cuda")]
    Cuda(cuda_backend::CudaBuffer),
}

// ── High-level execute functions ──────────────────────────────────────────────

/// Execute a circuit on the best available GPU (wgpu > cuda > cpu fallback).
///
/// Requires at least one GPU feature (`wgpu` or `cuda`) to be compiled in.
/// Falls back silently to the CPU statevector if no GPU is available at runtime.
#[cfg(any(feature = "wgpu", feature = "cuda"))]
pub fn execute_gpu(program: &Program) -> Result<ExecutionResult, String> {
    #[cfg(feature = "wgpu")]
    if let Ok(r) = wgpu_backend::execute(program) {
        return Ok(r);
    }

    #[cfg(feature = "cuda")]
    if let Ok(r) = cuda_backend::execute(program) {
        return Ok(r);
    }

    // Both GPU backends unavailable at runtime — fall back to CPU.
    eprintln!("warning: no GPU device available, falling back to statevector CPU backend");
    crate::runtime::execute(program).map_err(|e| e.to_string())
}

/// Execute a circuit using the wgpu (WebGPU) backend.
#[cfg(feature = "wgpu")]
pub fn execute_wgpu(program: &Program) -> Result<ExecutionResult, String> {
    wgpu_backend::execute(program)
}

/// Execute a circuit using the CUDA backend.
#[cfg(feature = "cuda")]
pub fn execute_cuda(program: &Program) -> Result<ExecutionResult, String> {
    cuda_backend::execute(program)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn list_devices_always_has_cpu() {
        let devs = list_devices();
        assert!(!devs.is_empty(), "list_devices must return at least one entry");
        assert!(
            devs.last().unwrap().contains("cpu"),
            "last entry should be cpu, got: {:?}",
            devs.last()
        );
    }

    #[test]
    fn gpu_device_display_format() {
        let d = GpuDevice {
            id:   "wgpu:0".to_string(),
            name: "Test GPU".to_string(),
            vram: 8 * 1_073_741_824,
            kind: GpuKind::Wgpu,
        };
        let s = d.to_string();
        assert!(s.contains("wgpu:0"));
        assert!(s.contains("Test GPU"));
        assert!(s.contains("8192 MB VRAM"));
    }

    #[test]
    fn gpu_device_zero_vram_no_vram_string() {
        let d = GpuDevice {
            id:   "cuda:0".to_string(),
            name: "Unknown GPU".to_string(),
            vram: 0,
            kind: GpuKind::Cuda,
        };
        let s = d.to_string();
        assert!(!s.contains("VRAM"), "zero VRAM should not print VRAM suffix");
    }
}
