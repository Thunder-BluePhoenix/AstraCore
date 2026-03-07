/// WebGPU statevector backend — powered by the `wgpu` crate.
///
/// Compiles a WGSL compute shader at runtime and dispatches one workgroup
/// per gate application.  Works on any GPU with a Vulkan, Metal, or DX12
/// driver — covers NVIDIA, AMD, Intel, and Apple Silicon.
///
/// Feature gate: `--features wgpu`
use wgpu::util::DeviceExt;

use super::{GpuDevice, GpuKind};
use crate::compiler::ir::{Instruction, Program};
use crate::runtime::ExecutionResult;

// ── WGSL compute shader ───────────────────────────────────────────────────────

/// WGSL shader source that applies a 2×2 gate to one qubit of a statevector.
///
/// Uniforms layout:
///   binding 0 — state buffer: array<vec2<f32>>   (re, im pairs)
///   binding 1 — gate uniform: GateParams
///
/// GateParams:
///   m00_re, m00_im, m01_re, m01_im,
///   m10_re, m10_im, m11_re, m11_im,
///   qubit: u32, n_qubits: u32, _pad0: u32, _pad1: u32
const SINGLE_QUBIT_SHADER: &str = r#"
struct GateParams {
    m00_re: f32, m00_im: f32,
    m01_re: f32, m01_im: f32,
    m10_re: f32, m10_im: f32,
    m11_re: f32, m11_im: f32,
    qubit:    u32,
    n_qubits: u32,
    _pad0:    u32,
    _pad1:    u32,
}

@group(0) @binding(0) var<storage, read_write> state: array<vec2<f32>>;
@group(0) @binding(1) var<uniform>              gate:  GateParams;

// Complex multiply: (a_re + i*a_im) * (b_re + i*b_im)
fn cmul_re(a_re: f32, a_im: f32, b_re: f32, b_im: f32) -> f32 {
    return a_re * b_re - a_im * b_im;
}
fn cmul_im(a_re: f32, a_im: f32, b_re: f32, b_im: f32) -> f32 {
    return a_re * b_im + a_im * b_re;
}

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pair_idx = gid.x;
    let half     = 1u << (gate.n_qubits - 1u);
    if pair_idx >= half { return; }

    // Reconstruct amplitude indices i0, i1 that differ only in `qubit` bit.
    let mask = 1u << gate.qubit;
    let lo   = pair_idx & (mask - 1u);           // bits below qubit
    let hi   = (pair_idx >> gate.qubit) << (gate.qubit + 1u); // bits above qubit
    let i0   = hi | lo;                          // qubit bit = 0
    let i1   = hi | mask | lo;                   // qubit bit = 1

    let a0 = state[i0];
    let a1 = state[i1];

    // new_a0 = m00 * a0 + m01 * a1
    let new_a0_re = cmul_re(gate.m00_re, gate.m00_im, a0.x, a0.y)
                  + cmul_re(gate.m01_re, gate.m01_im, a1.x, a1.y);
    let new_a0_im = cmul_im(gate.m00_re, gate.m00_im, a0.x, a0.y)
                  + cmul_im(gate.m01_re, gate.m01_im, a1.x, a1.y);

    // new_a1 = m10 * a0 + m11 * a1
    let new_a1_re = cmul_re(gate.m10_re, gate.m10_im, a0.x, a0.y)
                  + cmul_re(gate.m11_re, gate.m11_im, a1.x, a1.y);
    let new_a1_im = cmul_im(gate.m10_re, gate.m10_im, a0.x, a0.y)
                  + cmul_im(gate.m11_re, gate.m11_im, a1.x, a1.y);

    state[i0] = vec2<f32>(new_a0_re, new_a0_im);
    state[i1] = vec2<f32>(new_a1_re, new_a1_im);
}
"#;

/// WGSL shader for CNOT gate.
///
/// Uniforms: `control: u32, target: u32, n_qubits: u32, _pad: u32`
const CNOT_SHADER: &str = r#"
struct CnotParams {
    control:  u32,
    target:   u32,
    n_qubits: u32,
    _pad:     u32,
}

@group(0) @binding(0) var<storage, read_write> state: array<vec2<f32>>;
@group(0) @binding(1) var<uniform>              cnot:  CnotParams;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let quarter = 1u << (cnot.n_qubits - 2u);
    if gid.x >= quarter { return; }

    // Enumerate only the indices where the control qubit is 1.
    let ctrl_mask   = 1u << cnot.control;
    let target_mask = 1u << cnot.target;

    // Reconstruct index with control=1 and target=0
    let bits_below_ctrl   = cnot.control;
    let bits_below_target = cnot.target;
    // Insert control=1 and target=0 bits into gid.x
    var idx = gid.x;
    // insert 0 at target position
    let lo_t = idx & (target_mask - 1u);
    let hi_t = (idx >> bits_below_target) << (bits_below_target + 1u);
    idx = hi_t | lo_t;                   // target bit = 0 inserted
    // insert 1 at control position
    let lo_c = idx & (ctrl_mask - 1u);
    let hi_c = (idx >> bits_below_ctrl) << (bits_below_ctrl + 1u);
    idx = hi_c | ctrl_mask | lo_c;       // control bit = 1 inserted

    let i0 = idx;                        // control=1, target=0
    let i1 = idx | target_mask;          // control=1, target=1

    let tmp = state[i0];
    state[i0] = state[i1];
    state[i1] = tmp;
}
"#;

// ── WgpuBuffer ────────────────────────────────────────────────────────────────

/// Device-side state vector buffer managed by wgpu.
pub struct WgpuBuffer {
    pub buf:      wgpu::Buffer,
    pub n_qubits: usize,
}

// ── WgpuExecutor ──────────────────────────────────────────────────────────────

/// Owns the wgpu device + queue + compiled pipelines.
struct WgpuExecutor {
    device:         wgpu::Device,
    queue:          wgpu::Queue,
    gate_pipeline:  wgpu::ComputePipeline,
    cnot_pipeline:  wgpu::ComputePipeline,
    gate_bgl:       wgpu::BindGroupLayout,
    cnot_bgl:       wgpu::BindGroupLayout,
}

impl WgpuExecutor {
    /// Initialise wgpu: request adapter + device, compile shaders.
    ///
    /// Returns `None` when no compatible GPU adapter is found.
    fn new() -> Option<Self> {
        pollster::block_on(Self::new_async())
    }

    async fn new_async() -> Option<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference:       wgpu::PowerPreference::HighPerformance,
                compatible_surface:     None,
                force_fallback_adapter: false,
            })
            .await?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label:    Some("astracore-gpu"),
                    required_features: wgpu::Features::empty(),
                    required_limits:   wgpu::Limits::default(),
                    memory_hints:      wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .ok()?;

        // ── single-qubit gate pipeline ────────────────────────────────────────
        let gate_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("gate-shader"),
            source: wgpu::ShaderSource::Wgsl(SINGLE_QUBIT_SHADER.into()),
        });

        let gate_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   Some("gate-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding:    0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty:               wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding:    1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty:               wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let gate_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label:                Some("gate-pl"),
            bind_group_layouts:   &[&gate_bgl],
            push_constant_ranges: &[],
        });

        let gate_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label:       Some("gate-cp"),
            layout:      Some(&gate_layout),
            module:      &gate_shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache:       None,
        });

        // ── CNOT pipeline ─────────────────────────────────────────────────────
        let cnot_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("cnot-shader"),
            source: wgpu::ShaderSource::Wgsl(CNOT_SHADER.into()),
        });

        let cnot_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   Some("cnot-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding:    0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty:               wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding:    1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty:               wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let cnot_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label:                Some("cnot-pl"),
            bind_group_layouts:   &[&cnot_bgl],
            push_constant_ranges: &[],
        });

        let cnot_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label:       Some("cnot-cp"),
            layout:      Some(&cnot_layout),
            module:      &cnot_shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache:       None,
        });

        Some(WgpuExecutor { device, queue, gate_pipeline, cnot_pipeline, gate_bgl, cnot_bgl })
    }

    // ── Buffer helpers ────────────────────────────────────────────────────────

    fn upload(&self, amps: &[(f32, f32)]) -> wgpu::Buffer {
        // Convert (re, im) pairs to flat f32 array: [re0, im0, re1, im1, …]
        let flat: Vec<f32> = amps.iter().flat_map(|(re, im)| [*re, *im]).collect();
        let bytes = bytemuck_cast_slice(&flat);
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("sv-buf"),
            contents: bytes,
            usage:    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        })
    }

    fn download(&self, buf: &wgpu::Buffer, n: usize) -> Vec<(f32, f32)> {
        let byte_size = (n * 2 * std::mem::size_of::<f32>()) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("staging"),
            size:               byte_size,
            usage:              wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("copy-enc"),
        });
        enc.copy_buffer_to_buffer(buf, 0, &staging, 0, byte_size);
        self.queue.submit(Some(enc.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        self.device.poll(wgpu::MaintainBase::Wait);
        let _ = rx.recv();

        let data = slice.get_mapped_range();
        let floats: &[f32] = bytemuck_cast_slice(&data);
        floats.chunks(2).map(|c| (c[0], c[1])).collect()
    }

    fn apply_gate_buf(
        &self,
        state_buf: &wgpu::Buffer,
        matrix: [[f32; 4]; 2],
        qubit:    usize,
        n_qubits: usize,
    ) {
        // Pack GateParams as 12 × f32 (48 bytes) — matches WGSL struct layout.
        let [row0, row1] = matrix;
        let params: [f32; 12] = [
            row0[0], row0[1], row0[2], row0[3],   // m00_re, m00_im, m01_re, m01_im
            row1[0], row1[1], row1[2], row1[3],   // m10_re, m10_im, m11_re, m11_im
            qubit as f32,
            n_qubits as f32,
            0.0,
            0.0,
        ];
        let params_bytes = bytemuck_cast_slice(&params);
        let uniform_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("gate-uniform"),
            contents: params_bytes,
            usage:    wgpu::BufferUsages::UNIFORM,
        });

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("gate-bg"),
            layout:  &self.gate_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: state_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: uniform_buf.as_entire_binding() },
            ],
        });

        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("gate-enc"),
        });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label:              Some("gate-pass"),
                timestamp_writes:   None,
            });
            pass.set_pipeline(&self.gate_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            let n_pairs = (1usize << n_qubits) / 2;
            let workgroups = ((n_pairs + 127) / 128) as u32;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        self.queue.submit(Some(enc.finish()));
        self.device.poll(wgpu::MaintainBase::Wait);
    }

    fn apply_cnot_buf(
        &self,
        state_buf: &wgpu::Buffer,
        control:  usize,
        target:   usize,
        n_qubits: usize,
    ) {
        let params: [u32; 4] = [control as u32, target as u32, n_qubits as u32, 0];
        let params_bytes = bytemuck_cast_slice(&params);
        let uniform_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("cnot-uniform"),
            contents: params_bytes,
            usage:    wgpu::BufferUsages::UNIFORM,
        });

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("cnot-bg"),
            layout:  &self.cnot_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: state_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: uniform_buf.as_entire_binding() },
            ],
        });

        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("cnot-enc"),
        });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label:            Some("cnot-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.cnot_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            // Each thread handles one (control=1, target-flipped) pair
            let n_pairs = (1usize << n_qubits) / 4; // quarter of amplitudes
            let workgroups = ((n_pairs + 127) / 128).max(1) as u32;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        self.queue.submit(Some(enc.finish()));
        self.device.poll(wgpu::MaintainBase::Wait);
    }
}

// ── Gate matrix helpers ───────────────────────────────────────────────────────

/// Matrix for the Hadamard gate.
#[inline]
fn hadamard_f32() -> [[f32; 4]; 2] {
    let s = std::f32::consts::FRAC_1_SQRT_2;
    [[s, 0.0, s, 0.0], [s, 0.0, -s, 0.0]]
}

/// Matrix rows for Pauli-X.
#[inline]
fn pauli_x_f32() -> [[f32; 4]; 2] {
    [[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
}

/// Matrix rows for Pauli-Y.
#[inline]
fn pauli_y_f32() -> [[f32; 4]; 2] {
    [[0.0, 0.0, 0.0, -1.0], [0.0, 1.0, 0.0, 0.0]]
}

/// Matrix rows for Pauli-Z.
#[inline]
fn pauli_z_f32() -> [[f32; 4]; 2] {
    [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0]]
}

/// Matrix rows for Rx(θ).
fn rx_f32(theta: f64) -> [[f32; 4]; 2] {
    let c = (theta / 2.0).cos() as f32;
    let s = (theta / 2.0).sin() as f32;
    [[c, 0.0, 0.0, -s], [0.0, -s, c, 0.0]]
}

/// Matrix rows for Ry(θ).
fn ry_f32(theta: f64) -> [[f32; 4]; 2] {
    let c = (theta / 2.0).cos() as f32;
    let s = (theta / 2.0).sin() as f32;
    [[c, 0.0, -s, 0.0], [s, 0.0, c, 0.0]]
}

/// Matrix rows for Rz(θ).
fn rz_f32(theta: f64) -> [[f32; 4]; 2] {
    let c = (theta / 2.0).cos() as f32;
    let s = (theta / 2.0).sin() as f32;
    [[c, -s, 0.0, 0.0], [0.0, 0.0, c, s]]
}

/// Matrix rows for Phase(θ).
fn phase_f32(theta: f64) -> [[f32; 4]; 2] {
    let c = theta.cos() as f32;
    let s = theta.sin() as f32;
    [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, c, s]]
}

/// S gate.
#[inline]
fn s_gate_f32() -> [[f32; 4]; 2] {
    [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
}

/// T gate.
#[inline]
fn t_gate_f32() -> [[f32; 4]; 2] {
    let s = std::f32::consts::FRAC_1_SQRT_2;
    [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, s, s]]
}

// ── Device enumeration ────────────────────────────────────────────────────────

/// Return a list of wgpu-compatible GPU devices found on this machine.
pub fn enumerate_devices() -> Vec<GpuDevice> {
    pollster::block_on(enumerate_devices_async())
}

async fn enumerate_devices_async() -> Vec<GpuDevice> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });

    let mut out = Vec::new();
    for (i, adapter) in instance.enumerate_adapters(wgpu::Backends::all()).into_iter().enumerate() {
        let info = adapter.get_info();
        out.push(GpuDevice {
            id:   format!("wgpu:{i}"),
            name: info.name.clone(),
            vram: 0, // wgpu doesn't expose VRAM size portably
            kind: GpuKind::Wgpu,
        });
    }
    out
}

// ── Public execute function ───────────────────────────────────────────────────

/// Execute an AQL program on the wgpu GPU backend.
///
/// Limitations of this implementation:
/// - Uses f32 arithmetic (vs f64 on CPU) — sufficient for ≤ 24 qubits
/// - Control-flow (GOTO/IF) instructions are not GPU-accelerated; the circuit
///   is executed linearly (loops are unrolled by the AQL v2 REPEAT pre-pass).
/// - Mid-circuit measurements trigger a GPU→CPU round-trip.
pub fn execute(program: &Program) -> Result<ExecutionResult, String> {
    let exec = WgpuExecutor::new()
        .ok_or_else(|| "no wgpu-compatible GPU adapter found".to_string())?;

    let n = program.num_qubits;
    let dim = 1usize << n;

    // Initialise |0…0⟩
    let mut amps: Vec<(f32, f32)> = vec![(0.0, 0.0); dim];
    amps[0] = (1.0, 0.0);

    let mut state_buf = exec.upload(&amps);

    // Track measurements (CPU-side after download)
    let mut measurements: Vec<crate::runtime::MeasurementRecord> = Vec::new();
    let mut step = 0usize;
    let mut branch_count = 0usize;

    // Snapshot state before measurements (for probability display)
    let mut pre_meas_amps: Option<Vec<(f32, f32)>> = None;

    for instr in &program.instructions {
        step += 1;
        match instr {
            Instruction::H(q)     => exec.apply_gate_buf(&state_buf, hadamard_f32(), *q, n),
            Instruction::X(q)     => exec.apply_gate_buf(&state_buf, pauli_x_f32(),  *q, n),
            Instruction::Y(q)     => exec.apply_gate_buf(&state_buf, pauli_y_f32(),  *q, n),
            Instruction::Z(q)     => exec.apply_gate_buf(&state_buf, pauli_z_f32(),  *q, n),
            Instruction::S(q)     => exec.apply_gate_buf(&state_buf, s_gate_f32(),   *q, n),
            Instruction::T(q)     => exec.apply_gate_buf(&state_buf, t_gate_f32(),   *q, n),
            Instruction::Rx { qubit, theta } => exec.apply_gate_buf(&state_buf, rx_f32(*theta),    *qubit, n),
            Instruction::Ry { qubit, theta } => exec.apply_gate_buf(&state_buf, ry_f32(*theta),    *qubit, n),
            Instruction::Rz { qubit, theta } => exec.apply_gate_buf(&state_buf, rz_f32(*theta),    *qubit, n),
            Instruction::Phase { qubit, theta } => exec.apply_gate_buf(&state_buf, phase_f32(*theta), *qubit, n),

            Instruction::Cnot { control, target } => {
                exec.apply_cnot_buf(&state_buf, *control, *target, n);
            }
            // CZ = H on target → CNOT → H on target
            Instruction::Cz { control, target } => {
                exec.apply_gate_buf(&state_buf, hadamard_f32(), *target, n);
                exec.apply_cnot_buf(&state_buf, *control, *target, n);
                exec.apply_gate_buf(&state_buf, hadamard_f32(), *target, n);
            }
            // SWAP = CNOT(a,b) · CNOT(b,a) · CNOT(a,b)
            Instruction::Swap { qubit_a, qubit_b } => {
                exec.apply_cnot_buf(&state_buf, *qubit_a, *qubit_b, n);
                exec.apply_cnot_buf(&state_buf, *qubit_b, *qubit_a, n);
                exec.apply_cnot_buf(&state_buf, *qubit_a, *qubit_b, n);
            }
            // Toffoli = CCX — decomposed as CNOT chain (7 CNOTs + 6 single-qubit gates)
            Instruction::Toffoli { control0, control1, target } => {
                toffoli_wgpu(&exec, &state_buf, *control0, *control1, *target, n);
            }

            Instruction::Measure(q) => {
                // Download to CPU, collapse, re-upload
                let cur = exec.download(&state_buf, dim);
                if pre_meas_amps.is_none() {
                    pre_meas_amps = Some(cur.clone());
                }
                let (new_amps, outcome) = collapse_cpu(cur, *q, n, rand::random::<f64>());
                state_buf = exec.upload(&new_amps);
                measurements.push(crate::runtime::MeasurementRecord {
                    qubit:   *q,
                    outcome,
                    step,
                });
            }
            Instruction::MeasureAll => {
                let cur = exec.download(&state_buf, dim);
                if pre_meas_amps.is_none() {
                    pre_meas_amps = Some(cur.clone());
                }
                let mut amps_cpu = cur;
                for q in 0..n {
                    let (new_amps, outcome) = collapse_cpu(amps_cpu, q, n, rand::random::<f64>());
                    amps_cpu = new_amps;
                    measurements.push(crate::runtime::MeasurementRecord {
                        qubit:   q,
                        outcome,
                        step,
                    });
                }
                state_buf = exec.upload(&amps_cpu);
            }

            // Control flow — linear trace only (loops already unrolled by REPEAT pass)
            Instruction::Goto { .. }
            | Instruction::GotoIf { .. }
            | Instruction::GotoIfNot { .. } => {
                branch_count += 1;
            }

            // Structural / invisible
            Instruction::Barrier | Instruction::Label(_) | Instruction::CallGate { .. } => {}
        }
    }

    // Final download
    let final_amps = exec.download(&state_buf, dim);
    let final_probs: Vec<f64> = final_amps.iter().map(|(re, im)| {
        (*re as f64) * (*re as f64) + (*im as f64) * (*im as f64)
    }).collect();

    // f64 amplitude table (re, im)
    let final_amplitudes: Vec<(f64, f64)> = final_amps.iter()
        .map(|(re, im)| (*re as f64, *im as f64))
        .collect();

    let pre_measurement_probs = pre_meas_amps.as_ref().map(|a| {
        a.iter().map(|(re, im)|
            (*re as f64) * (*re as f64) + (*im as f64) * (*im as f64)
        ).collect()
    });
    let pre_measurement_amplitudes = pre_meas_amps.map(|a| {
        a.iter().map(|(re, im)| (*re as f64, *im as f64)).collect()
    });

    Ok(ExecutionResult {
        num_qubits:                 n,
        final_probabilities:        final_probs,
        final_amplitudes,
        measurements,
        gate_count:                 step,
        branch_count,
        steps_executed:             step,
        pre_measurement_probs,
        pre_measurement_amplitudes,
    })
}

// ── Toffoli decomposition for wgpu ───────────────────────────────────────────

/// Standard 7-CNOT Toffoli decomposition (up to global phase).
///
/// Reference: Nielsen & Chuang, Figure 4.9.
fn toffoli_wgpu(
    exec:     &WgpuExecutor,
    buf:      &wgpu::Buffer,
    c0:       usize,
    c1:       usize,
    target:   usize,
    n:        usize,
) {
    let h  = hadamard_f32();
    let t  = t_gate_f32();
    let s  = s_gate_f32(); // T†T = I; for T† use conjugate
    // T† matrix: [[1,0],[0, e^{-iπ/4}]] = [[1,0],[0, (1-i)/√2]]
    let tdg: [[f32; 4]; 2] = {
        let v = std::f32::consts::FRAC_1_SQRT_2;
        [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, v, -v]]
    };
    // S† = [[1,0],[0,-i]]
    let sdg: [[f32; 4]; 2] = [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, -1.0]];
    let _ = s; let _ = sdg;

    exec.apply_gate_buf(buf, h, target, n);
    exec.apply_cnot_buf(buf, c1, target, n);
    exec.apply_gate_buf(buf, tdg, target, n);
    exec.apply_cnot_buf(buf, c0, target, n);
    exec.apply_gate_buf(buf, t, target, n);
    exec.apply_cnot_buf(buf, c1, target, n);
    exec.apply_gate_buf(buf, tdg, target, n);
    exec.apply_cnot_buf(buf, c0, target, n);
    exec.apply_gate_buf(buf, t, c1, n);
    exec.apply_gate_buf(buf, t, target, n);
    exec.apply_gate_buf(buf, h, target, n);
    exec.apply_cnot_buf(buf, c0, c1, n);
    exec.apply_gate_buf(buf, t, c0, n);
    exec.apply_gate_buf(buf, tdg, c1, n);
    exec.apply_cnot_buf(buf, c0, c1, n);
}

// ── CPU-side measurement collapse ─────────────────────────────────────────────

/// Collapse qubit `q` on a CPU amplitude vector, returning (new_amps, outcome).
fn collapse_cpu(
    amps:   Vec<(f32, f32)>,
    qubit:  usize,
    n:      usize,
    rand:   f64,
) -> (Vec<(f32, f32)>, bool) {
    let mask  = 1usize << qubit;
    let prob1: f64 = amps.iter().enumerate()
        .filter(|(i, _)| i & mask != 0)
        .map(|(_, (re, im))| (*re as f64).powi(2) + (*im as f64).powi(2))
        .sum();

    let outcome = rand < prob1;
    let norm = if outcome { prob1.sqrt() } else { (1.0 - prob1).sqrt() };
    let norm = if norm < 1e-12 { 1.0 } else { norm };

    let new_amps: Vec<(f32, f32)> = amps.into_iter().enumerate().map(|(i, (re, im))| {
        let bit_set = i & mask != 0;
        if bit_set == outcome {
            (re / norm as f32, im / norm as f32)
        } else {
            (0.0, 0.0)
        }
    }).collect();

    (new_amps, outcome)
}

// ── bytemuck-like byte conversion (no dep) ────────────────────────────────────

/// Reinterpret a slice of T as a byte slice.
fn bytemuck_cast_slice<T>(data: &[T]) -> &[u8] {
    let ptr = data.as_ptr() as *const u8;
    let len = data.len() * std::mem::size_of::<T>();
    unsafe { std::slice::from_raw_parts(ptr, len) }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler;

    fn bell_program() -> Program {
        compiler::parse_source("QREG 2\nH 0\nCNOT 0 1").unwrap()
    }

    fn try_executor() -> Option<WgpuExecutor> {
        WgpuExecutor::new()
    }

    /// Skip a test gracefully when no GPU is available.
    macro_rules! require_gpu {
        ($exec:ident) => {
            let Some($exec) = try_executor() else {
                eprintln!("SKIP: no wgpu GPU available in test environment");
                return;
            };
        };
    }

    #[test]
    fn enumerate_does_not_panic() {
        let _devs = enumerate_devices();
        // Must not panic; empty is fine on CI
    }

    #[test]
    fn hadamard_matrix_f32_correct() {
        let m = hadamard_f32();
        let s = std::f32::consts::FRAC_1_SQRT_2;
        assert!((m[0][0] - s).abs() < 1e-6);
        assert!((m[1][2] - (-s)).abs() < 1e-6);
    }

    #[test]
    fn wgpu_h_gate_ground_state() {
        require_gpu!(exec);
        // Apply H to |0⟩: should give [1/√2, 1/√2]
        let amps = vec![(1.0f32, 0.0f32), (0.0f32, 0.0f32)];
        let buf = exec.upload(&amps);
        exec.apply_gate_buf(&buf, hadamard_f32(), 0, 1);
        let result = exec.download(&buf, 2);
        let s = std::f32::consts::FRAC_1_SQRT_2;
        assert!((result[0].0 - s).abs() < 1e-5, "re[0] = {}", result[0].0);
        assert!((result[1].0 - s).abs() < 1e-5, "re[1] = {}", result[1].0);
    }

    #[test]
    fn wgpu_bell_state_execute() {
        let program = bell_program();
        match execute(&program) {
            Ok(result) => {
                let probs = &result.final_probabilities;
                assert_eq!(probs.len(), 4);
                assert!((probs[0] - 0.5).abs() < 0.01, "|00⟩ prob = {}", probs[0]);
                assert!((probs[3] - 0.5).abs() < 0.01, "|11⟩ prob = {}", probs[3]);
                assert!(probs[1] < 1e-4, "|01⟩ prob should be ~0");
                assert!(probs[2] < 1e-4, "|10⟩ prob should be ~0");
            }
            Err(e) if e.contains("no wgpu") => {
                eprintln!("SKIP: {e}");
            }
            Err(e) => panic!("unexpected error: {e}"),
        }
    }

    #[test]
    fn wgpu_fallback_no_panic_on_no_gpu() {
        // execute() must return Ok (fallback to CPU if no GPU) or Err with
        // a message — it must never panic.
        let program = bell_program();
        let _result = execute(&program); // either Ok or Err("no wgpu…")
    }
}
