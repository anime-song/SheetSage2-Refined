//! The encoder on a GPU through wgpu (Vulkan on Linux; built with `--features gpu`): the same computation as
//! `encoder.rs` in WGSL compute shaders, float16 weights (the CPU engine's 16-column panels) and float32 arithmetic.
//! The log-mel spectrogram is computed on the CPU and the layer mixes are read back for the heads, which also run
//! on the CPU.  Made for integrated GPUs, which share the system's memory: all buffers are
//! allocated once for the longest window and reused, about 280 MB of activations next to 1.2 GB of weights.

use std::borrow::Cow;

use wgpu::util::DeviceExt;

use crate::Result;
use crate::encoder::{Encoder, HEAD_DIM, HEADS, Mix, N_MELS, STATES, WIDTH};
use crate::lora::Adapter;
use crate::weights::{Linear, PANEL, Panels, Weights};

const COMMON: &str = include_str!("common.wgsl");

/// A WGSL module without the iteration limit that wgpu adds to every loop: that limit is a branch per iteration
/// which keeps the shader compiler from unrolling the kernels' fixed loops (on RADV it made the attention kernel
/// one LDS read, a wait and four multiply-adds per step).  Every loop in these shaders has a bound.  Bounds checks
/// stay on.
fn shader(device: &wgpu::Device, name: &str, code: String) -> wgpu::ShaderModule {
    let checks = wgpu::ShaderRuntimeChecks { force_loop_bounding: false, ..wgpu::ShaderRuntimeChecks::checked() };
    // SAFETY: the loops terminate (each has a fixed or a frame-count bound), which is what the limit guards.
    unsafe {
        device.create_shader_module_trusted(
            wgpu::ShaderModuleDescriptor { label: Some(name), source: wgpu::ShaderSource::Wgsl(Cow::Owned(code)) },
            checks,
        )
    }
}

/// Queries per workgroup of attention.wgsl.
const ATTENTION_QUERIES: u32 = 256;

struct Kernels {
    gemm: wgpu::ComputePipeline,
    layer_norm: wgpu::ComputePipeline,
    conv_norm: wgpu::ComputePipeline,
    glu: wgpu::ComputePipeline,
    rope: wgpu::ComputePipeline,
    mix: wgpu::ComputePipeline,
    col_squares: wgpu::ComputePipeline,
    grn_gain: wgpu::ComputePipeline,
    attention: wgpu::ComputePipeline,
}

struct GLinear {
    n: u32,
    k: u32,
    w: wgpu::Buffer,
    bias: wgpu::Buffer,
    lora: Option<Box<GLora>>,
}

/// A LoRA adapter (`set_adapter`): down `[R, K]`, and per part its first output, first column of `x A^T` and up.
struct GLora {
    rank: u32,
    down: GLinear,
    parts: Vec<(u32, u32, GLinear)>,
}

struct GConvNeXt {
    taps: wgpu::Buffer, // [7][C]
    bias: wgpu::Buffer,
    fc1: GLinear,
    gamma: wgpu::Buffer, // GRN
    fc2: GLinear,
}

struct GStage {
    down: Option<GLinear>,
    channels: u32,
    layers: Vec<GConvNeXt>,
}

struct GBlock {
    ffn1: (GLinear, GLinear),
    qkv: GLinear,
    out: GLinear,
    pw_in: GLinear,
    taps: wgpu::Buffer, // [31][WIDTH]
    dw_norm: (wgpu::Buffer, wgpu::Buffer),
    pw_out: GLinear,
    ffn2: (GLinear, GLinear),
    final_norm: (wgpu::Buffer, wgpu::Buffer),
}

pub struct GpuEncoder {
    device: wgpu::Device,
    queue: wgpu::Queue,
    k: Kernels,
    pub adapter: String,
    /// What the encoder holds in GPU memory (on an integrated GPU, system memory), in bytes: weights, activations.
    pub memory: (u64, u64),
    /// What the current LoRA adapter holds in GPU memory, in bytes.
    pub adapter_bytes: u64,
    cpu: Encoder, // the log-mel spectrogram
    stages: Vec<GStage>,
    blocks: Vec<GBlock>,
    rope: wgpu::Buffer, // cos [max frames][32], then sin
    max_frames: u32,
    dummy: wgpu::Buffer,
    // activations, allocated for the longest window: three [frames][WIDTH] rows, one wide buffer (q k v, the
    // convolution module's input, the feed-forward and ConvNeXt hidden rows), GRN's sums and gains, the mixes
    rows: [wgpu::Buffer; 3],
    wide: wgpu::Buffer,
    grn: wgpu::Buffer,
    mixes: Vec<wgpu::Buffer>,
    low: wgpu::Buffer, // an adapter's x A^T, [rows][R]
}

fn f16_bits(v: f32) -> u16 {
    half::f16::from_f32(v).to_bits()
}

impl GpuEncoder {
    /// The encoder on the first GPU wgpu finds (`SHEETSAGE_GPU` = part of an adapter's name picks another), for
    /// `mixes` layer mixes.
    pub fn load(weights: &Weights, mixes: usize) -> Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let wanted = std::env::var("SHEETSAGE_GPU").ok().map(|s| s.to_lowercase());
        let adapters = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()));
        // a real GPU before a software one (llvmpipe), unless one is named
        let rank = |t: wgpu::DeviceType| match t {
            wgpu::DeviceType::IntegratedGpu | wgpu::DeviceType::DiscreteGpu => 0,
            wgpu::DeviceType::VirtualGpu => 1,
            _ => 2,
        };
        let adapter = adapters
            .into_iter()
            .filter(|a| wanted.as_ref().is_none_or(|w| a.get_info().name.to_lowercase().contains(w)))
            .min_by_key(|a| rank(a.get_info().device_type))
            .ok_or("no GPU adapter found (SHEETSAGE_GPU selects one by name)")?;
        let info = adapter.get_info();
        let limits = adapter.limits();
        if limits.max_compute_workgroup_storage_size < 32768 {
            return Err(format!("{}: 32 KiB of workgroup memory needed", info.name).into());
        }
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("sheetsage2"),
            required_features: wgpu::Features::empty(),
            required_limits: limits.clone(),
            ..Default::default()
        }))?;
        let module = |name: &str, src: &str, common: bool| {
            let code = if common { format!("{COMMON}\n{src}") } else { src.to_string() };
            shader(&device, name, code)
        };
        let gemm = module("gemm", include_str!("gemm.wgsl"), true);
        let rows = module("rows", include_str!("rows.wgsl"), true);
        let misc = module("misc", include_str!("misc.wgsl"), false);
        let attention = module("attention", include_str!("attention.wgsl"), false);
        let pipeline = |m: &wgpu::ShaderModule, entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: None,
                module: m,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let k = Kernels {
            gemm: pipeline(&gemm, "main"),
            layer_norm: pipeline(&rows, "layer_norm"),
            conv_norm: pipeline(&rows, "conv_norm"),
            glu: pipeline(&misc, "glu"),
            rope: pipeline(&misc, "rope"),
            mix: pipeline(&misc, "mix"),
            col_squares: pipeline(&misc, "col_squares"),
            grn_gain: pipeline(&misc, "grn_gain"),
            attention: pipeline(&attention, "main"),
        };
        let cpu = Encoder::load(weights)?;
        let vector = |name: &str| -> Result<wgpu::Buffer> {
            let v = weights.f32(name)?;
            Ok(Self::upload(&device, name, bytes(v)))
        };
        let taps = |name: &str, channels: usize| -> Result<wgpu::Buffer> {
            let raw = weights.f32(name)?; // [C, taps]
            let n = raw.len() / channels;
            let mut t = vec![0.0f32; raw.len()];
            for c in 0..channels {
                for j in 0..n {
                    t[j * channels + c] = raw[c * n + j];
                }
            }
            Ok(Self::upload(&device, name, bytes(&t)))
        };
        let linear = |name: &str| -> Result<GLinear> { Self::linear(&device, &weights.linear(name)?, name) };
        let channels = [N_MELS, 128, 512, 1024];
        let mut stages = Vec::new();
        for (i, &depth) in [3usize, 4, 5].iter().enumerate() {
            let p = format!("stages.{i}");
            let down = if weights.has(&format!("{p}.down.w")) { Some(linear(&format!("{p}.down"))?) } else { None };
            let c = channels[i + 1];
            let layers = (0..depth)
                .map(|j| {
                    let q = format!("{p}.layers.{j}");
                    Ok(GConvNeXt {
                        taps: taps(&format!("{q}.dw.w"), c)?,
                        bias: vector(&format!("{q}.dw.b"))?,
                        fc1: linear(&format!("{q}.fc1"))?,
                        gamma: vector(&format!("{q}.grn_gamma"))?,
                        fc2: linear(&format!("{q}.fc2"))?,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            stages.push(GStage { down, channels: c as u32, layers });
        }
        let blocks = (0..STATES - 1)
            .map(|i| {
                let p = format!("layers.{i}");
                let l = |n: &str| linear(&format!("{p}.{n}"));
                let v = |n: &str| vector(&format!("{p}.{n}"));
                Ok(GBlock {
                    ffn1: (l("ffn1.fc1")?, l("ffn1.fc2")?),
                    qkv: l("attn.qkv")?,
                    out: l("attn.out")?,
                    pw_in: l("conv.pw_in")?,
                    taps: taps(&format!("{p}.conv.dw.w"), WIDTH)?,
                    dw_norm: (v("conv.dw_norm.w")?, v("conv.dw_norm.b")?),
                    pw_out: l("conv.pw_out")?,
                    ffn2: (l("ffn2.fc1")?, l("ffn2.fc2")?),
                    final_norm: (v("final_norm.w")?, v("final_norm.b")?),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let cos = weights.f32("rope.cos")?;
        let sin = weights.f32("rope.sin")?;
        let max_frames = (cos.len() / (HEAD_DIM / 2)) as u32;
        let rope = Self::upload(&device, "rope", &[bytes(cos), bytes(sin)].concat());
        let storage = |name: &str, floats: usize| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(name),
                size: (floats.max(1) * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        // row buffers hold frames x 1024 floats (= 50 Hz x 512 > 100 Hz x 128), the wide one frames x 4096 (ConvNeXt's
        // hidden rows at every rate, the feed-forward's, q k v)
        let frames = max_frames as usize;
        let row_floats = (frames + 2) * WIDTH;
        let wide_floats = (frames + 2) * 4 * WIDTH;
        let weight_bytes = stages
            .iter()
            .flat_map(|s| s.down.iter().chain(s.layers.iter().flat_map(|l| [&l.fc1, &l.fc2])))
            .chain(
                blocks
                    .iter()
                    .flat_map(|b| [&b.ffn1.0, &b.ffn1.1, &b.qkv, &b.out, &b.pw_in, &b.pw_out, &b.ffn2.0, &b.ffn2.1]),
            )
            .map(|l| l.w.size() + l.bias.size())
            .sum::<u64>();
        let activation_bytes = (3 * row_floats + wide_floats + 2 * 4096 + mixes * frames * WIDTH) as u64 * 4;
        Ok(Self {
            adapter: format!("{} ({:?})", info.name, info.backend),
            memory: (weight_bytes, activation_bytes),
            adapter_bytes: 0,
            rows: [storage("rows0", row_floats), storage("rows1", row_floats), storage("rows2", row_floats)],
            wide: storage("wide", wide_floats),
            grn: storage("grn", 2 * 4096),
            mixes: (0..mixes).map(|i| storage(&format!("mix{i}"), frames * WIDTH)).collect(),
            dummy: storage("dummy", 4),
            low: storage("low", 1),
            device,
            queue,
            k,
            cpu,
            stages,
            blocks,
            rope,
            max_frames,
        })
    }

    fn upload(device: &wgpu::Device, name: &str, data: &[u8]) -> wgpu::Buffer {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(name),
            contents: if data.is_empty() { &[0u8; 4] } else { data },
            usage: wgpu::BufferUsages::STORAGE,
        })
    }

    /// A linear layer's panels as float16 (int8 weights times their scales, float32 rounded).
    fn linear(device: &wgpu::Device, lin: &Linear, name: &str) -> Result<GLinear> {
        let half: Vec<u16> = match &lin.w {
            Panels::F16(w) => w.to_vec(),
            Panels::F32(w) => w.iter().map(|&v| f16_bits(v)).collect(),
            Panels::I8(w) => {
                let scale = lin.scale.ok_or("int8 weights without scales")?;
                w.iter()
                    .enumerate()
                    .map(|(i, &v)| {
                        let col = (i / (lin.k * PANEL)) * PANEL + i % PANEL;
                        f16_bits(if col < lin.n { v as f32 * scale[col] } else { 0.0 })
                    })
                    .collect()
            }
        };
        let raw: Vec<u8> = half.iter().flat_map(|v| v.to_le_bytes()).collect();
        Ok(GLinear {
            n: lin.n as u32,
            k: lin.k as u32,
            w: Self::upload(device, name, &raw),
            bias: Self::upload(device, &format!("{name}.b"), bytes(&lin.bias)),
            lora: None,
        })
    }

    /// Every linear layer with its name in the exported file.
    fn linears_mut(&mut self) -> Vec<(String, &mut GLinear)> {
        let mut out = Vec::new();
        for (i, stage) in self.stages.iter_mut().enumerate() {
            if let Some(down) = &mut stage.down {
                out.push((format!("stages.{i}.down"), down));
            }
            for (j, l) in stage.layers.iter_mut().enumerate() {
                out.push((format!("stages.{i}.layers.{j}.fc1"), &mut l.fc1));
                out.push((format!("stages.{i}.layers.{j}.fc2"), &mut l.fc2));
            }
        }
        for (i, b) in self.blocks.iter_mut().enumerate() {
            let p = format!("layers.{i}");
            out.push((format!("{p}.ffn1.fc1"), &mut b.ffn1.0));
            out.push((format!("{p}.ffn1.fc2"), &mut b.ffn1.1));
            out.push((format!("{p}.attn.qkv"), &mut b.qkv));
            out.push((format!("{p}.attn.out"), &mut b.out));
            out.push((format!("{p}.conv.pw_in"), &mut b.pw_in));
            out.push((format!("{p}.conv.pw_out"), &mut b.pw_out));
            out.push((format!("{p}.ffn2.fc1"), &mut b.ffn2.0));
            out.push((format!("{p}.ffn2.fc2"), &mut b.ffn2.1));
        }
        out
    }

    /// Apply a LoRA adapter from now on (`None`: the base weights alone): its layers go to GPU memory as float16,
    /// the previous adapter's are freed.
    pub fn set_adapter(&mut self, adapter: Option<&Adapter>) -> Result<()> {
        if let Some(a) = adapter {
            let layers: Vec<(String, u32, u32)> = self.linears_mut().into_iter().map(|(n, l)| (n, l.n, l.k)).collect();
            for (name, lora) in &a.layers {
                let (_, n, k) = layers
                    .iter()
                    .find(|(n, ..)| n == name)
                    .ok_or_else(|| format!("adapter layer {name}: not in the encoder"))?;
                lora.check(name, *n as usize, *k as usize)?;
            }
        }
        let device = self.device.clone();
        let mut rank = 0;
        let mut bytes = 0;
        for (name, l) in self.linears_mut() {
            l.lora = None;
            let Some(lora) = adapter.and_then(|a| a.layers.get(&name)) else { continue };
            lora.check(&name, l.n as usize, l.k as usize)?;
            let down = Self::linear(&device, &lora.down, &format!("{name}.lora.down"))?;
            bytes += down.w.size() + down.bias.size();
            let mut parts = Vec::new();
            for part in &lora.parts {
                let up = Self::linear(&device, &part.up, &format!("{name}.lora.up.{}", part.row))?;
                bytes += up.w.size() + up.bias.size();
                parts.push((part.row as u32, part.low as u32, up));
            }
            rank = rank.max(lora.rank);
            l.lora = Some(Box::new(GLora { rank: lora.rank as u32, down, parts }));
        }
        // the most rows a linear layer sees: the first stage's, 4 per 25 Hz frame
        let rows = 4 * (self.max_frames as usize + 1);
        self.low = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("low"),
            size: (rows * rank).max(1) as u64 * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        self.adapter_bytes = bytes + self.low.size();
        Ok(())
    }

    pub fn max_frames(&self) -> usize {
        self.max_frames as usize
    }

    /// As `Encoder::forward`: encode a 24 kHz waveform, the layer mixes (`mixes`, at most as many as `load` was
    /// given) read back into `Mix::sum`; returns the number of 25 Hz frames.
    pub fn forward(&self, wave: &[f32], mixes: &mut [&mut Mix]) -> Result<usize> {
        if mixes.len() > self.mixes.len() {
            return Err("more layer mixes than the GPU encoder was loaded for".into());
        }
        let mel = crate::profile::time("mel", || self.cpu.mel(wave));
        let mut frames = (mel.len() / N_MELS) as u32;
        if frames / 4 > self.max_frames {
            return Err(
                format!("{} frames, more than the {} the rotary tables cover", frames / 4, self.max_frames).into()
            );
        }
        self.queue.write_buffer(&self.rows[0], 0, bytes(&mel));
        let mut x = 0usize; // which of `rows` holds the activations
        let mut channels = N_MELS as u32;
        let mut enc = self.encoder();
        for stage in &self.stages {
            if let Some(down) = &stage.down {
                let (norm, out) = ((x + 1) % 3, (x + 2) % 3);
                self.layer_norm(&mut enc, &self.rows[x], &self.rows[norm], channels, frames, None, false, 1e-6);
                frames /= 2;
                self.gemm(
                    &mut enc,
                    &self.rows[norm],
                    2 * channels,
                    frames,
                    down,
                    &self.rows[out],
                    down.n,
                    0,
                    false,
                    false,
                    None,
                );
                x = out;
            }
            channels = stage.channels;
            for l in &stage.layers {
                let y = (x + 1) % 3;
                self.conv_norm(
                    &mut enc,
                    &self.rows[x],
                    &self.rows[y],
                    channels,
                    frames,
                    (&l.taps, 7),
                    Some(&l.bias),
                    None,
                    false,
                    1e-6,
                );
                let hidden = l.fc1.n;
                self.gemm(&mut enc, &self.rows[y], channels, frames, &l.fc1, &self.wide, hidden, 0, true, false, None);
                let shape = (hidden, frames, hidden);
                self.misc(
                    &mut enc,
                    &self.k.col_squares,
                    Some(&self.wide),
                    None,
                    &self.grn,
                    shape,
                    0,
                    0.0,
                    (hidden.div_ceil(256), 1),
                );
                self.misc(&mut enc, &self.k.grn_gain, None, Some(&l.gamma), &self.grn, shape, 0, 0.0, (1, 1));
                self.gemm(
                    &mut enc,
                    &self.wide,
                    hidden,
                    frames,
                    &l.fc2,
                    &self.rows[x],
                    channels,
                    0,
                    false,
                    true,
                    Some(hidden),
                );
            }
            let mut since = std::time::Instant::now();
            self.checkpoint(&mut enc, "gpu: convnext", &mut since);
            self.submit(enc);
            enc = self.encoder();
        }
        let n = frames * WIDTH as u32;
        for (m, buf) in mixes.iter().zip(&self.mixes) {
            self.axpy(&mut enc, &self.rows[x], buf, n, m.weights[0], true);
        }
        self.submit(enc);
        for (i, b) in self.blocks.iter().enumerate() {
            let mut enc = self.encoder();
            x = self.block(&mut enc, b, x, frames);
            for (m, buf) in mixes.iter().zip(&self.mixes) {
                self.axpy(&mut enc, &self.rows[x], buf, n, m.weights[i + 1], false);
            }
            self.submit(enc);
        }
        for (m, buf) in mixes.iter_mut().zip(&self.mixes) {
            m.sum = self.read(buf, n as usize)?;
        }
        Ok(frames as usize)
    }

    /// With `SHEETSAGE_PROFILE=1`: run what `enc` holds, wait for it and add the time since `since` to `name`.
    fn checkpoint(&self, enc: &mut wgpu::CommandEncoder, name: &'static str, since: &mut std::time::Instant) {
        if !crate::profile::enabled() {
            return;
        }
        self.submit(std::mem::replace(enc, self.encoder()));
        let _ = self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        crate::profile::add(name, since.elapsed().as_secs_f64());
        *since = std::time::Instant::now();
    }

    /// One Conformer block on `rows[x]`; returns the index of the buffer holding its output.
    fn block(&self, enc: &mut wgpu::CommandEncoder, b: &GBlock, x: usize, frames: u32) -> usize {
        let (a, c) = ((x + 1) % 3, (x + 2) % 3);
        let w = WIDTH as u32;
        let mut since = std::time::Instant::now();
        self.checkpoint(enc, "gpu: waiting", &mut since);
        self.feed_forward(enc, &b.ffn1, x, a, frames);
        self.checkpoint(enc, "gpu: ffn", &mut since);
        // attention
        self.layer_norm(enc, &self.rows[x], &self.rows[a], w, frames, None, false, 1e-5);
        self.gemm(enc, &self.rows[a], w, frames, &b.qkv, &self.wide, 3 * w, 0, false, false, None);
        self.checkpoint(enc, "gpu: qkv", &mut since);
        self.misc(
            enc,
            &self.k.rope,
            None,
            Some(&self.rope),
            &self.wide,
            (frames * 1024, self.max_frames, 0),
            0,
            0.0,
            groups(frames * 1024),
        );
        {
            let params = self.params(&[frames, 0, 0, 0]);
            self.dispatch(
                enc,
                &self.k.attention,
                &[(0, &params), (1, &self.wide), (2, &self.rows[a])],
                (frames.div_ceil(ATTENTION_QUERIES), HEADS as u32),
            );
        }
        self.checkpoint(enc, "gpu: attention core", &mut since);
        self.gemm(enc, &self.rows[a], w, frames, &b.out, &self.rows[x], w, 0, false, true, None);
        self.checkpoint(enc, "gpu: attention out", &mut since);
        // convolution module
        self.layer_norm(enc, &self.rows[x], &self.rows[a], w, frames, None, false, 1e-5);
        self.gemm(enc, &self.rows[a], w, frames, &b.pw_in, &self.wide, 2 * w, 0, false, false, None);
        self.misc(
            enc,
            &self.k.glu,
            Some(&self.wide),
            None,
            &self.rows[a],
            (frames * w, frames, w),
            0,
            0.0,
            groups(frames * w),
        );
        self.conv_norm(enc, &self.rows[a], &self.rows[c], w, frames, (&b.taps, 31), None, Some(&b.dw_norm), true, 1e-5);
        self.gemm(enc, &self.rows[c], w, frames, &b.pw_out, &self.rows[x], w, 0, false, true, None);
        self.checkpoint(enc, "gpu: conv module", &mut since);
        self.feed_forward(enc, &b.ffn2, x, a, frames);
        self.checkpoint(enc, "gpu: ffn", &mut since);
        // the final norm writes the next buffer, which then holds the activations
        self.layer_norm(enc, &self.rows[x], &self.rows[a], w, frames, Some(&b.final_norm), false, 1e-5);
        a
    }

    /// `rows[x] += fc2(gelu(fc1(norm(rows[x]))))`, normalising into `rows[norm]`.
    fn feed_forward(
        &self,
        enc: &mut wgpu::CommandEncoder,
        (fc1, fc2): &(GLinear, GLinear),
        x: usize,
        norm: usize,
        frames: u32,
    ) {
        let w = WIDTH as u32;
        self.layer_norm(enc, &self.rows[x], &self.rows[norm], w, frames, None, false, 1e-5);
        self.gemm(enc, &self.rows[norm], w, frames, fc1, &self.wide, fc1.n, 0, true, false, None);
        self.gemm(enc, &self.wide, fc1.n, frames, fc2, &self.rows[x], w, 0, false, true, None);
    }

    fn wait(&self) {
        let _ = self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
    }

    /// For checking and timing the kernels (`examples/gpu_bench.rs`): block 0's feed-forward fc1 (`which` 0, 1024 ->
    /// 4096), fc2 (1, 4096 -> 1024) or q k v (2, 1024 -> 3072) on `m` rows of `a`; the output and the seconds the
    /// GPU took (the best of `reps`).
    pub fn bench_linear(&self, which: usize, a: &[f32], m: u32, reps: usize) -> Result<(Vec<f32>, f64)> {
        let b = &self.blocks[0];
        let l = [&b.ffn1.0, &b.ffn1.1, &b.qkv][which];
        let (input, output) = if which == 1 { (&self.wide, &self.rows[0]) } else { (&self.rows[0], &self.wide) };
        self.queue.write_buffer(input, 0, bytes(a));
        let mut best = f64::INFINITY;
        for _ in 0..reps {
            self.wait();
            let start = std::time::Instant::now();
            let mut enc = self.encoder();
            self.gemm(&mut enc, input, l.k, m, l, output, l.n, 0, false, false, None);
            self.submit(enc);
            self.wait();
            best = best.min(start.elapsed().as_secs_f64());
        }
        Ok((self.read(output, (m * l.n) as usize)?, best))
    }

    /// The attention core on q k v rows `[frames][3072]` (rotary positions already applied): the output
    /// `[frames][1024]` and the best of `reps` GPU times.  `kernel` replaces the engine's with WGSL source of the
    /// same interface that takes `queries` per workgroup (to compare kernels).
    pub fn bench_attention(
        &self,
        qkv: &[f32],
        frames: u32,
        reps: usize,
        kernel: Option<(&str, u32)>,
    ) -> Result<(Vec<f32>, f64)> {
        let other = kernel.map(|(src, _)| {
            let module = shader(&self.device, "attention (bench)", src.to_string());
            self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("attention (bench)"),
                layout: None,
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            })
        });
        let (pipeline, queries) = match (&other, kernel) {
            (Some(p), Some((_, q))) => (p, q),
            _ => (&self.k.attention, ATTENTION_QUERIES),
        };
        self.queue.write_buffer(&self.wide, 0, bytes(qkv));
        let mut best = f64::INFINITY;
        for _ in 0..reps {
            self.wait();
            let start = std::time::Instant::now();
            let mut enc = self.encoder();
            let params = self.params(&[frames, 0, 0, 0]);
            self.dispatch(
                &mut enc,
                pipeline,
                &[(0, &params), (1, &self.wide), (2, &self.rows[0])],
                (frames.div_ceil(queries), HEADS as u32),
            );
            self.submit(enc);
            self.wait();
            best = best.min(start.elapsed().as_secs_f64());
        }
        Ok((self.read(&self.rows[0], frames as usize * WIDTH)?, best))
    }

    fn encoder(&self) -> wgpu::CommandEncoder {
        self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None })
    }

    /// Submissions stay short (a block or a stage each): GPU drivers reset a queue whose job runs for seconds.
    fn submit(&self, enc: wgpu::CommandEncoder) {
        self.queue.submit([enc.finish()]);
    }

    fn params(&self, words: &[u32]) -> wgpu::Buffer {
        let mut w = words.to_vec();
        w.resize(w.len().div_ceil(4) * 4, 0);
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: &w.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>(),
            usage: wgpu::BufferUsages::UNIFORM,
        })
    }

    fn dispatch(
        &self,
        enc: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::ComputePipeline,
        entries: &[(u32, &wgpu::Buffer)],
        (gx, gy): (u32, u32),
    ) {
        let layout = pipeline.get_bind_group_layout(0);
        let entries: Vec<wgpu::BindGroupEntry> = entries
            .iter()
            .map(|&(binding, b)| wgpu::BindGroupEntry { binding, resource: b.as_entire_binding() })
            .collect();
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &entries,
        });
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(gx, gy, 1);
    }

    /// `c (+)= act(a W^T + b)` over `m` rows of `a` (row stride `lda`); `scale` = GRN's gains (`grn`, after its sums).
    #[allow(clippy::too_many_arguments)]
    fn gemm(
        &self,
        enc: &mut wgpu::CommandEncoder,
        a: &wgpu::Buffer,
        lda: u32,
        m: u32,
        l: &GLinear,
        c: &wgpu::Buffer,
        ldc: u32,
        c_off: u32,
        gelu: bool,
        add: bool,
        scale: Option<u32>,
    ) {
        let Some(lora) = &l.lora else {
            return self.product(enc, a, lda, m, 0, l, c, ldc, c_off, gelu, add, scale);
        };
        // the adapter's (a A^T) B^T + db joins the product before the activation (applied with the last part, so
        // only on a layer that one part covers whole: the feed-forward's fc1)
        assert!(!gelu || (lora.parts.len() == 1 && lora.parts[0].2.n == l.n));
        self.product(enc, a, lda, m, 0, l, c, ldc, c_off, false, add, scale);
        self.product(enc, a, lda, m, 0, &lora.down, &self.low, lora.rank, 0, false, false, scale);
        for (row, low, up) in &lora.parts {
            self.product(enc, &self.low, lora.rank, m, *low, up, c, ldc, c_off + row, gelu, true, None);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn product(
        &self,
        enc: &mut wgpu::CommandEncoder,
        a: &wgpu::Buffer,
        lda: u32,
        m: u32,
        a_off: u32,
        l: &GLinear,
        c: &wgpu::Buffer,
        ldc: u32,
        c_off: u32,
        gelu: bool,
        add: bool,
        scale: Option<u32>,
    ) {
        let flags = gelu as u32 | (add as u32) << 1 | (scale.is_some() as u32) << 2;
        let n_pad = l.n.div_ceil(PANEL as u32) * PANEL as u32;
        let params = self.params(&[m, l.n, n_pad, l.k, lda, ldc, a_off, c_off, flags, 0, 0, 0]);
        // GRN's gains follow its sums in `grn` (`grn_gain`): bound from there
        let scale_buf = if scale.is_some() { &self.grn } else { &self.dummy };
        let layout = self.k.gemm.get_bind_group_layout(0);
        let scale_entry = match scale {
            Some(width) => wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: scale_buf,
                offset: width as u64 * 4,
                size: wgpu::BufferSize::new(width as u64 * 4),
            }),
            None => scale_buf.as_entire_binding(),
        };
        let entries = [
            wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: a.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: l.w.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: l.bias.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: scale_entry },
            wgpu::BindGroupEntry { binding: 5, resource: c.as_entire_binding() },
        ];
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &entries,
        });
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        pass.set_pipeline(&self.k.gemm);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(l.n.div_ceil(128), m.div_ceil(128), 1);
    }

    #[allow(clippy::too_many_arguments)]
    fn layer_norm(
        &self,
        enc: &mut wgpu::CommandEncoder,
        x: &wgpu::Buffer,
        out: &wgpu::Buffer,
        width: u32,
        rows: u32,
        affine: Option<&(wgpu::Buffer, wgpu::Buffer)>,
        gelu: bool,
        eps: f32,
    ) {
        let flags = affine.is_some() as u32 | (gelu as u32) << 1;
        let params = self.params(&[width, rows, width, width, 0, 0, flags, 0, eps.to_bits()]);
        let (g, b) = affine.map_or((&self.dummy, &self.dummy), |(g, b)| (g, b));
        self.dispatch(enc, &self.k.layer_norm, &[(0, &params), (1, x), (2, g), (3, b), (6, out)], row_groups(rows));
    }

    #[allow(clippy::too_many_arguments)]
    fn conv_norm(
        &self,
        enc: &mut wgpu::CommandEncoder,
        x: &wgpu::Buffer,
        out: &wgpu::Buffer,
        width: u32,
        rows: u32,
        (taps, n_taps): (&wgpu::Buffer, u32),
        bias: Option<&wgpu::Buffer>,
        affine: Option<&(wgpu::Buffer, wgpu::Buffer)>,
        gelu: bool,
        eps: f32,
    ) {
        let flags = affine.is_some() as u32 | (gelu as u32) << 1 | (bias.is_some() as u32) << 2;
        let params = self.params(&[width, rows, width, width, 0, 0, flags, n_taps, eps.to_bits()]);
        let (g, b) = affine.map_or((&self.dummy, &self.dummy), |(g, b)| (g, b));
        let entries = [(0, &params), (1, x), (2, g), (3, b), (4, taps), (5, bias.unwrap_or(&self.dummy)), (6, out)];
        self.dispatch(enc, &self.k.conv_norm, &entries, row_groups(rows));
    }

    /// One of `misc.wgsl`'s passes: `x`, `aux` (either may be absent: the pass does not read it) and `out`.
    #[allow(clippy::too_many_arguments)]
    fn misc(
        &self,
        enc: &mut wgpu::CommandEncoder,
        kernel: &wgpu::ComputePipeline,
        x: Option<&wgpu::Buffer>,
        aux: Option<&wgpu::Buffer>,
        out: &wgpu::Buffer,
        (n, rows, width): (u32, u32, u32),
        flags: u32,
        weight: f32,
        groups: (u32, u32),
    ) {
        let params = self.params(&[n, rows, width, flags, weight.to_bits()]);
        let mut entries = vec![(0, &params), (3, out)];
        entries.extend(x.map(|x| (1, x)));
        entries.extend(aux.map(|a| (2, a)));
        self.dispatch(enc, kernel, &entries, groups);
    }

    fn axpy(
        &self,
        enc: &mut wgpu::CommandEncoder,
        x: &wgpu::Buffer,
        out: &wgpu::Buffer,
        n: u32,
        weight: f32,
        first: bool,
    ) {
        self.misc(enc, &self.k.mix, Some(x), None, out, (n, 0, 0), first as u32, weight, groups(n));
    }

    fn read(&self, buf: &wgpu::Buffer, floats: usize) -> Result<Vec<f32>> {
        let size = (floats * 4) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("read back"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self.encoder();
        enc.copy_buffer_to_buffer(buf, 0, &staging, 0, size);
        self.submit(enc);
        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |r| r.expect("map"));
        self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None })?;
        let data = slice.get_mapped_range().map_err(|e| format!("reading back: {e:?}"))?;
        let out = data.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
        drop(data);
        staging.unmap();
        Ok(out)
    }
}

fn bytes(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

/// Workgroups for one row each: x + 32768 y.
fn row_groups(rows: u32) -> (u32, u32) {
    (rows.min(32768), rows.div_ceil(32768))
}

/// Workgroups of 256 threads for `n` elements: x + 65535 y.
fn groups(n: u32) -> (u32, u32) {
    let g = n.div_ceil(256);
    (g.min(65535), g.div_ceil(65535))
}
