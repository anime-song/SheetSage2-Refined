//! MERT-v2-FullSong encoder (`sheetsage2_refined/encoder.py`) on the CPU.
//!
//! Activations are row-major `[frames, channels]`.  Instead of returning all 25 hidden states the encoder adds each
//! state, weighted, into the layer mixes of the heads that read it (`Mix`), so only the current state is kept.
//! Feed-forward blocks run per block of rows (their 4096-wide hidden rows never exist for the whole window), and
//! attention is computed block by block with an online softmax (no frames x frames matrix).

use rayon::prelude::*;
use realfft::RealFftPlanner;

use crate::Result;
use crate::gemm::{self, Act, MR, NR};
use crate::lora::Adapter;
use crate::math;
use crate::weights::{Linear, Weights};

pub const SAMPLE_RATE: usize = 24_000;
pub const HOP: usize = 240;
pub const N_FFT: usize = 2048;
pub const N_MELS: usize = 128;
pub const SAMPLES_PER_FRAME: usize = HOP * 4;
pub const WIDTH: usize = 1024;
pub const HEADS: usize = 16;
pub const HEAD_DIM: usize = WIDTH / HEADS;
pub const STATES: usize = 25;
const STAGE_EPS: f32 = 1e-6;
const EPS: f32 = 1e-5;

/// A head's layer mix: `sum_l weights[l] * state_l`, accumulated as the encoder runs.
pub struct Mix {
    pub weights: Vec<f32>,
    pub sum: Vec<f32>,
}

impl Mix {
    pub fn new(weights: &[f32]) -> Self {
        assert_eq!(weights.len(), STATES);
        Self { weights: weights.to_vec(), sum: Vec::new() }
    }

    fn add(&mut self, layer: usize, state: &[f32]) {
        let w = self.weights[layer];
        if layer == 0 {
            self.sum.clear();
            self.sum.extend(state.iter().map(|v| w * v));
        } else {
            self.sum.par_chunks_mut(1 << 14).zip(state.par_chunks(1 << 14)).for_each(|(s, x)| {
                for (s, x) in s.iter_mut().zip(x) {
                    *s += w * x;
                }
            });
        }
    }
}

struct ConvNeXtLayer {
    dw_w: Vec<f32>, // [7][C]
    dw_b: &'static [f32],
    fc1: Linear,
    grn_gamma: &'static [f32],
    fc2: Linear,
}

struct Stage {
    down: Option<Linear>,
    channels: usize,
    layers: Vec<ConvNeXtLayer>,
}

struct Block {
    ffn1: (Linear, Linear),
    qkv: Linear,
    out: Linear,
    pw_in: Linear,
    dw_w: Vec<f32>, // [31][C]
    dw_norm: (&'static [f32], &'static [f32]),
    pw_out: Linear,
    ffn2: (Linear, Linear),
    final_norm: (&'static [f32], &'static [f32]),
}

pub struct Encoder {
    mel_fb: Vec<(usize, Vec<f32>)>, // per mel band: first FFT bin and its non-zero weights
    window: &'static [f32],
    mel_mean: &'static [f32],
    mel_std: &'static [f32],
    stages: Vec<Stage>,
    blocks: Vec<Block>,
    rope_cos: &'static [f32], // [max frames][HEAD_DIM / 2]
    rope_sin: &'static [f32],
}

/// `[C, k]` -> `[k][C]`, so a depthwise convolution vectorises over channels.
fn transpose_taps(w: &[f32], channels: usize) -> Vec<f32> {
    let k = w.len() / channels;
    let mut out = vec![0.0; w.len()];
    for c in 0..channels {
        for j in 0..k {
            out[j * channels + c] = w[c * k + j];
        }
    }
    out
}

impl Encoder {
    pub fn load(w: &Weights) -> Result<Self> {
        let fb = w.f32("frontend.mel_fb")?; // [N_FFT / 2 + 1, N_MELS]
        let bins = N_FFT / 2 + 1;
        let mel_fb = (0..N_MELS)
            .map(|m| {
                let col: Vec<f32> = (0..bins).map(|f| fb[f * N_MELS + m]).collect();
                let first = col.iter().position(|&v| v != 0.0).unwrap_or(0);
                let last = col.iter().rposition(|&v| v != 0.0).map_or(0, |i| i + 1);
                (first, col[first..last.max(first)].to_vec())
            })
            .collect();
        let mut stages = Vec::new();
        let channels = [N_MELS, 128, 512, 1024];
        let depths = [3, 4, 5];
        for (i, &depth) in depths.iter().enumerate() {
            let p = format!("stages.{i}");
            let down = if w.has(&format!("{p}.down.w")) { Some(w.linear(&format!("{p}.down"))?) } else { None };
            let c = channels[i + 1];
            let layers = (0..depth)
                .map(|j| {
                    let q = format!("{p}.layers.{j}");
                    Ok(ConvNeXtLayer {
                        dw_w: transpose_taps(w.f32(&format!("{q}.dw.w"))?, c),
                        dw_b: w.f32(&format!("{q}.dw.b"))?,
                        fc1: w.linear(&format!("{q}.fc1"))?,
                        grn_gamma: w.f32(&format!("{q}.grn_gamma"))?,
                        fc2: w.linear(&format!("{q}.fc2"))?,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            stages.push(Stage { down, channels: c, layers });
        }
        let mut blocks = Vec::new();
        for i in 0.. {
            let p = format!("layers.{i}");
            if !w.has(&format!("{p}.attn.qkv.w")) {
                break;
            }
            let l = |name: &str| w.linear(&format!("{p}.{name}"));
            let v = |name: &str| w.f32(&format!("{p}.{name}"));
            blocks.push(Block {
                ffn1: (l("ffn1.fc1")?, l("ffn1.fc2")?),
                qkv: l("attn.qkv")?,
                out: l("attn.out")?,
                pw_in: l("conv.pw_in")?,
                dw_w: transpose_taps(v("conv.dw.w")?, WIDTH),
                dw_norm: (v("conv.dw_norm.w")?, v("conv.dw_norm.b")?),
                pw_out: l("conv.pw_out")?,
                ffn2: (l("ffn2.fc1")?, l("ffn2.fc2")?),
                final_norm: (v("final_norm.w")?, v("final_norm.b")?),
            });
        }
        if blocks.len() + 1 != STATES {
            return Err(format!("{} Conformer blocks, expected {}", blocks.len(), STATES - 1).into());
        }
        Ok(Self {
            mel_fb,
            window: w.f32("frontend.window")?,
            mel_mean: w.f32("frontend.mel_mean")?,
            mel_std: w.f32("frontend.mel_std")?,
            stages,
            blocks,
            rope_cos: w.f32("rope.cos")?,
            rope_sin: w.f32("rope.sin")?,
        })
    }

    /// Every linear layer with its name in the exported file.
    pub fn linears_mut(&mut self) -> Vec<(String, &mut Linear)> {
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

    /// Apply a LoRA adapter from now on (`None`: the base weights alone).
    pub fn set_adapter(&mut self, adapter: Option<&Adapter>) -> Result<()> {
        Adapter::attach(adapter, self.linears_mut())
    }

    pub fn max_frames(&self) -> usize {
        self.rope_cos.len() / (HEAD_DIM / 2)
    }

    /// Encode a 24 kHz waveform, adding every hidden state into `mixes`; returns the number of 25 Hz frames.
    pub fn forward(&self, wave: &[f32], mixes: &mut [&mut Mix]) -> Result<usize> {
        let mel = crate::profile::time("mel", || self.mel(wave));
        let mut x = mel;
        let mut frames = x.len() / N_MELS;
        let mut channels = N_MELS;
        for stage in &self.stages {
            if let Some(down) = &stage.down {
                (x, frames) = downsample(&x, frames, channels, down);
            }
            channels = stage.channels;
            for layer in &stage.layers {
                crate::profile::time("convnext", || convnext_layer(&mut x, frames, channels, layer));
            }
        }
        if frames > self.max_frames() {
            return Err(format!("{frames} frames, more than the {} the rotary tables cover", self.max_frames()).into());
        }
        for mix in mixes.iter_mut() {
            mix.add(0, &x);
        }
        let mut scratch = Scratch::new(frames);
        for (i, block) in self.blocks.iter().enumerate() {
            self.block(block, &mut x, frames, &mut scratch);
            for mix in mixes.iter_mut() {
                mix.add(i + 1, &x);
            }
        }
        Ok(frames)
    }

    /// Normalised power log-mel spectrogram, `[waveform.len() / 240, 128]` (`MelFrontend`).
    pub fn mel(&self, wave: &[f32]) -> Vec<f32> {
        let n = wave.len();
        let pad = N_FFT / 2;
        let reflect = |i: isize| -> f32 {
            let n = n as isize;
            let j = if i < 0 {
                -i
            } else if i >= n {
                2 * (n - 1) - i
            } else {
                i
            };
            wave[j.clamp(0, n - 1) as usize]
        };
        let frames = n / HOP; // torch.stft's 1 + n / hop frames, less the trailing one
        let mut out = vec![0.0f32; frames * N_MELS];
        let fft = RealFftPlanner::<f32>::new().plan_fft_forward(N_FFT);
        out.par_chunks_mut(N_MELS * 64).enumerate().for_each(|(chunk, out)| {
            let mut input = fft.make_input_vec();
            let mut spectrum = fft.make_output_vec();
            let mut scratch = fft.make_scratch_vec();
            let mut power = vec![0.0f32; N_FFT / 2 + 1];
            for (r, row) in out.chunks_mut(N_MELS).enumerate() {
                let t = chunk * 64 + r;
                let start = (t * HOP) as isize - pad as isize;
                for (i, (v, w)) in input.iter_mut().zip(self.window).enumerate() {
                    *v = reflect(start + i as isize) * w;
                }
                fft.process_with_scratch(&mut input, &mut spectrum, &mut scratch).expect("fft");
                for (p, c) in power.iter_mut().zip(&spectrum) {
                    *p = c.re * c.re + c.im * c.im;
                }
                for (m, (first, weights)) in self.mel_fb.iter().enumerate() {
                    let e: f32 = weights.iter().zip(&power[*first..]).map(|(w, p)| w * p).sum();
                    let db = 10.0 * e.max(1e-10).log10();
                    row[m] = (db - self.mel_mean[m]) / self.mel_std[m].max(1e-5);
                }
            }
        });
        out
    }

    fn block(&self, b: &Block, x: &mut [f32], frames: usize, s: &mut Scratch) {
        crate::profile::time("ffn", || feed_forward(x, frames, &b.ffn1));
        crate::profile::time("attention", || self.attention(b, x, frames, s));
        crate::profile::time("conv module", || conv_module(b, x, frames, s));
        crate::profile::time("ffn", || feed_forward(x, frames, &b.ffn2));
        let (g, beta) = b.final_norm;
        x.par_chunks_mut(WIDTH).for_each(|row| {
            let copy = row.to_vec();
            math::layer_norm(&copy, row, g, beta, EPS);
        });
    }

    /// `x += out(attention(norm(x)))`.  q, k and v are made per block of rows: q goes to `s.rows` (where attention
    /// leaves its output), the rotated keys and the values straight into their per-head layouts.
    fn attention(&self, b: &Block, x: &mut [f32], frames: usize, s: &mut Scratch) {
        const QKV: usize = 3 * WIDTH;
        let half = HEAD_DIM / 2;
        let (cos, sin) = (self.rope_cos, self.rope_sin);
        let padded = s.padded;
        let (q_out, keys_out, values_out) =
            (SendPtr(s.rows.as_mut_ptr()), SendPtr(s.keys.as_mut_ptr()), SendPtr(s.values.as_mut_ptr()));
        let block = gemm::rows().div_ceil(NR) * NR; // whole key panels per block
        x[..frames * WIDTH].par_chunks(block * WIDTH).enumerate().for_each(|(i, x)| {
            let (t0, rows) = (i * block, x.len() / WIDTH);
            let mut norm = vec![0.0f32; rows * WIDTH];
            for (o, row) in norm.chunks_mut(WIDTH).zip(x.chunks(WIDTH)) {
                math::normalize(row, o, EPS);
            }
            let mut qkv = vec![0.0f32; rows * QKV];
            gemm::linear_block(&norm, WIDTH, rows, &b.qkv, &mut qkv, QKV, Act::None, false, None);
            for (r, row) in qkv.chunks_mut(QKV).enumerate() {
                let t = t0 + r;
                // rotary positions on q and k (`RotarySelfAttention`)
                let (c, si) = (&cos[t * half..(t + 1) * half], &sin[t * half..(t + 1) * half]);
                for head in 0..2 * HEADS {
                    let v = &mut row[head * HEAD_DIM..(head + 1) * HEAD_DIM];
                    for i in 0..half {
                        let (a, b) = (v[i], v[i + half]);
                        v[i] = a * c[i] - b * si[i];
                        v[i + half] = b * c[i] + a * si[i];
                    }
                }
                // Each block writes its own rows of q, and of every head's keys (whole panels) and values.
                unsafe {
                    std::slice::from_raw_parts_mut(q_out.get().add(t * WIDTH), WIDTH).copy_from_slice(&row[..WIDTH]);
                    for head in 0..HEADS {
                        let k = &row[WIDTH + head * HEAD_DIM..][..HEAD_DIM];
                        let panel = keys_out.get().add(head * HEAD_DIM * padded + (t / NR) * HEAD_DIM * NR + t % NR);
                        for (d, &v) in k.iter().enumerate() {
                            *panel.add(d * NR) = v;
                        }
                        let v = &row[2 * WIDTH + head * HEAD_DIM..][..HEAD_DIM];
                        std::slice::from_raw_parts_mut(values_out.get().add((head * padded + t) * HEAD_DIM), HEAD_DIM)
                            .copy_from_slice(v);
                    }
                }
            }
        });
        crate::profile::time("attention core", || attend(&mut s.rows, &s.keys, &s.values, padded, frames));
        gemm::linear(&s.rows, WIDTH, frames, &b.out, x, WIDTH, Act::None, true, None);
    }
}

/// Buffers reused by every block of one window: `[frames, WIDTH]` rows (q, then attention's output, then the
/// convolution module's GLU output) and the keys and values (the latter also the convolution module's output).
struct Scratch {
    rows: Vec<f32>,   // [frames + MR, WIDTH]
    keys: Vec<f32>,   // [HEADS][padded / 16][HEAD_DIM][16]: rotated keys in panels of 16
    values: Vec<f32>, // [HEADS][padded][HEAD_DIM]
    padded: usize,    // frames rounded up to a whole panel
}

impl Scratch {
    fn new(frames: usize) -> Self {
        let padded = frames.div_ceil(NR) * NR;
        Self {
            rows: vec![0.0; (frames + MR) * WIDTH],
            keys: vec![0.0; HEADS * HEAD_DIM * padded],
            values: vec![0.0; HEADS * HEAD_DIM * padded],
            padded,
        }
    }
}

fn normalize_rows(x: &[f32], out: &mut [f32], width: usize, eps: f32) {
    out[..x.len()].par_chunks_mut(width).zip(x.par_chunks(width)).for_each(|(o, x)| math::normalize(x, o, eps));
}

/// `x += 0.5 * fc2(gelu(fc1(norm(x))))` (the 0.5 and the norm's scale and shift are folded into the weights).
fn feed_forward(x: &mut [f32], frames: usize, (fc1, fc2): &(Linear, Linear)) {
    let _ = frames;
    x.par_chunks_mut(gemm::rows() * WIDTH).for_each(|x| {
        let rows = x.len() / WIDTH;
        let mut norm = vec![0.0f32; rows * WIDTH];
        let mut hidden = vec![0.0f32; rows * fc1.n];
        for (o, row) in norm.chunks_mut(WIDTH).zip(x.chunks(WIDTH)) {
            math::normalize(row, o, EPS);
        }
        gemm::linear_block(&norm, WIDTH, rows, fc1, &mut hidden, fc1.n, Act::Gelu, false, None);
        gemm::linear_block(&hidden, fc1.n, rows, fc2, x, WIDTH, Act::None, true, None);
    });
}

/// Scaled dot-product attention of all heads (scale folded into q), flash-style: per head and block of `QB` query
/// rows the keys are visited in blocks of `KB` with a running max and sum.  `out` is `[frames, WIDTH]`.
/// `qo` holds q `[frames + MR, WIDTH]` (the last MR rows padding) and gets the output in its place: each task reads
/// only its own rows of its own head and writes them when it is done with them.
pub fn attend(qo: &mut [f32], keys: &[f32], values: &[f32], padded: usize, frames: usize) {
    attend_heads(qo, WIDTH, HEADS, keys, values, padded, frames);
}

/// `attend` on `heads` heads whose q rows (and outputs) are `ldq` floats apart, head h in columns h * HEAD_DIM ..
/// (h + 1) * HEAD_DIM; `keys` and `values` hold those heads only.
pub fn attend_heads(
    qo: &mut [f32],
    ldq: usize,
    heads: usize,
    keys: &[f32],
    values: &[f32],
    padded: usize,
    frames: usize,
) {
    const QB: usize = 16 * MR;
    const KB: usize = 256;
    assert!(qo.len() >= (frames + MR) * ldq);
    let blocks = frames.div_ceil(QB);
    let out_ptr = SendPtr(qo.as_mut_ptr());
    (0..heads * blocks).into_par_iter().for_each(|task| {
        let (head, qb) = (task / blocks, task % blocks);
        let q0 = qb * QB;
        let rows = QB.min(frames - q0);
        let row_panels = rows.div_ceil(MR);
        let keys = &keys[head * HEAD_DIM * padded..(head + 1) * HEAD_DIM * padded];
        let values = &values[head * HEAD_DIM * padded..(head + 1) * HEAD_DIM * padded];
        let mut scores = vec![0.0f32; QB * KB];
        let mut acc = vec![0.0f32; QB * HEAD_DIM];
        let mut max = vec![f32::NEG_INFINITY; QB];
        let mut sum = vec![0.0f32; QB];
        let q_len = (row_panels * MR - 1) * ldq + HEAD_DIM;
        for k0 in (0..frames).step_by(KB) {
            let kb = KB.min(frames - k0);
            // The task's q rows, read only while no output is written.
            let q = unsafe { std::slice::from_raw_parts(out_ptr.get().add(q0 * ldq + head * HEAD_DIM), q_len) };
            // scores[r, j] = q_r . k_(k0 + j)
            for rp in 0..row_panels {
                for cp in 0..kb.div_ceil(NR) {
                    let panel = &keys[(k0 / NR + cp) * HEAD_DIM * NR..][..HEAD_DIM * NR];
                    let c = &mut scores[rp * MR * KB + cp * NR..];
                    gemm::kernel_into(HEAD_DIM, &q[rp * MR * ldq..], ldq, 1, panel, NR, c, KB, false);
                }
            }
            for r in 0..rows {
                let row = &mut scores[r * KB..r * KB + kb];
                let new_max = max[r].max(row_max(row));
                let correction = (max[r] - new_max).exp();
                max[r] = new_max;
                sum[r] = sum[r] * correction + exp_shifted(row, new_max);
                if correction != 1.0 {
                    for v in &mut acc[r * HEAD_DIM..(r + 1) * HEAD_DIM] {
                        *v *= correction;
                    }
                }
            }
            for r in rows..row_panels * MR {
                scores[r * KB..r * KB + kb].fill(0.0);
            }
            // acc += p v
            for rp in 0..row_panels {
                for cp in 0..HEAD_DIM / NR {
                    let c = &mut acc[rp * MR * HEAD_DIM + cp * NR..];
                    gemm::kernel_into(
                        kb,
                        &scores[rp * MR * KB..],
                        KB,
                        1,
                        &values[k0 * HEAD_DIM + cp * NR..],
                        HEAD_DIM,
                        c,
                        HEAD_DIM,
                        true,
                    );
                }
            }
        }
        for r in 0..rows {
            let inv = 1.0 / sum[r];
            // Each task writes its own rows of its own head's columns.
            let o = unsafe {
                std::slice::from_raw_parts_mut(out_ptr.get().add((q0 + r) * ldq + head * HEAD_DIM), HEAD_DIM)
            };
            for (o, a) in o.iter_mut().zip(&acc[r * HEAD_DIM..(r + 1) * HEAD_DIM]) {
                *o = a * inv;
            }
        }
    });
}

fn row_max(row: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if crate::simd::avx2() {
        return unsafe { row_max_avx2(row) };
    }
    row.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn row_max_avx2(row: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let mut m = [_mm256_set1_ps(f32::NEG_INFINITY); 4];
    let mut chunks = row.chunks_exact(32);
    for c in &mut chunks {
        for (i, m) in m.iter_mut().enumerate() {
            *m = _mm256_max_ps(*m, _mm256_loadu_ps(c.as_ptr().add(8 * i)));
        }
    }
    let v = _mm256_max_ps(_mm256_max_ps(m[0], m[1]), _mm256_max_ps(m[2], m[3]));
    let mut lanes = [0.0f32; 8];
    _mm256_storeu_ps(lanes.as_mut_ptr(), v);
    lanes.iter().chain(chunks.remainder()).fold(f32::NEG_INFINITY, |m, &v| m.max(v))
}

/// Probabilities below `exp(FLUSH)` of the row's best are set to zero: they change nothing at float32 precision
/// (the best key alone weighs 1), and multiplied by a value they would make denormal numbers, which x86 cores
/// handle 10-50x slower (`examples/attention.rs --sharp`).
const FLUSH: f32 = -40.0;

/// `row[j] = exp(row[j] - max)` (zero below `FLUSH`), returning the sum.
fn exp_shifted(row: &mut [f32], max: f32) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if crate::simd::avx2() {
        return unsafe { exp_shifted_avx2(row, max) };
    }
    let mut sum = 0.0;
    for v in row.iter_mut() {
        *v = if *v - max < FLUSH { 0.0 } else { (*v - max).exp() };
        sum += *v;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn exp_shifted_avx2(row: &mut [f32], max: f32) -> f32 {
    use std::arch::x86_64::*;
    let (m, flush) = (_mm256_set1_ps(max), _mm256_set1_ps(FLUSH));
    let mut total = _mm256_setzero_ps();
    let mut chunks = row.chunks_exact_mut(8);
    for c in &mut chunks {
        let x = _mm256_sub_ps(_mm256_loadu_ps(c.as_ptr()), m);
        let e = _mm256_and_ps(math::exp8(x), _mm256_cmp_ps::<_CMP_GE_OQ>(x, flush));
        _mm256_storeu_ps(c.as_mut_ptr(), e);
        total = _mm256_add_ps(total, e);
    }
    let mut lanes = [0.0f32; 8];
    _mm256_storeu_ps(lanes.as_mut_ptr(), total);
    let mut sum: f32 = lanes.iter().sum();
    for v in chunks.into_remainder() {
        *v = if *v - max < FLUSH { 0.0 } else { (*v - max).exp() };
        sum += *v;
    }
    sum
}

#[derive(Clone, Copy)]
struct SendPtr(*mut f32);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}
impl SendPtr {
    fn get(self) -> *mut f32 {
        self.0
    }
}

/// `x += pw_out(gelu(norm(depthwise(glu(pw_in(norm(x)))))))` (`ConvolutionModule`): the GLU output per block of rows
/// into `s.rows`, the depthwise convolution's into `s.values` (free again after attention).
fn conv_module(b: &Block, x: &mut [f32], frames: usize, s: &mut Scratch) {
    const IN: usize = 2 * WIDTH;
    let block = gemm::rows();
    s.rows[..frames * WIDTH].par_chunks_mut(block * WIDTH).zip(x[..frames * WIDTH].par_chunks(block * WIDTH)).for_each(
        |(glu, x)| {
            let rows = x.len() / WIDTH;
            let mut norm = vec![0.0f32; rows * WIDTH];
            for (o, row) in norm.chunks_mut(WIDTH).zip(x.chunks(WIDTH)) {
                math::normalize(row, o, EPS);
            }
            let mut wide = vec![0.0f32; rows * IN];
            gemm::linear_block(&norm, WIDTH, rows, &b.pw_in, &mut wide, IN, Act::None, false, None);
            for (o, row) in glu.chunks_mut(WIDTH).zip(wide.chunks(IN)) {
                let (a, gate) = row.split_at(WIDTH);
                for ((o, a), g) in o.iter_mut().zip(a).zip(gate) {
                    *o = a * math::sigmoid(*g);
                }
            }
        },
    );
    let taps = b.dw_w.len() / WIDTH;
    let half = (taps / 2) as isize;
    let glu = &s.rows;
    let (g, beta) = b.dw_norm;
    s.values[..frames * WIDTH].par_chunks_mut(WIDTH).enumerate().for_each(|(t, o)| {
        o.fill(0.0);
        for j in 0..taps {
            let src = t as isize + j as isize - half;
            if src < 0 || src >= frames as isize {
                continue;
            }
            let row = &glu[src as usize * WIDTH..(src as usize + 1) * WIDTH];
            for ((o, w), v) in o.iter_mut().zip(&b.dw_w[j * WIDTH..(j + 1) * WIDTH]).zip(row) {
                *o += w * v;
            }
        }
        let copy = o.to_vec();
        math::layer_norm(&copy, o, g, beta, EPS);
        math::gelu_slice(o);
    });
    gemm::linear(&s.values, WIDTH, frames, &b.pw_out, x, WIDTH, Act::None, true, None);
}

/// LayerNorm each frame of `x` `[frames, cin]` (scale and shift folded) and apply the stride-2 convolution as one
/// linear layer over pairs of frames.
fn downsample(x: &[f32], frames: usize, cin: usize, down: &Linear) -> (Vec<f32>, usize) {
    let mut norm = vec![0.0f32; frames * cin];
    normalize_rows(x, &mut norm, cin, STAGE_EPS);
    let out_frames = frames / 2;
    let mut out = vec![0.0f32; out_frames * down.n];
    gemm::linear(&norm, 2 * cin, out_frames, down, &mut out, down.n, Act::None, false, None);
    (out, out_frames)
}

/// `x += fc2(grn(gelu(fc1(norm(depthwise(x))))))` (`ConvNeXtLayer`).  GRN's gain per channel needs the whole
/// window, so the hidden rows are kept; GRN's shift is folded into fc2's bias and its gain scales fc2's input.
fn convnext_layer(x: &mut [f32], frames: usize, c: usize, l: &ConvNeXtLayer) {
    // depthwise convolution, norm, fc1 and GELU per block of rows; GRN needs all of them before fc2
    let hidden_width = l.fc1.n;
    let mut hidden = vec![0.0f32; frames * hidden_width];
    let block = gemm::rows();
    let input = &*x;
    hidden.par_chunks_mut(block * hidden_width).enumerate().for_each(|(i, hidden)| {
        let (t0, rows) = (i * block, hidden.len() / hidden_width);
        let mut y = vec![0.0f32; rows * c];
        for (r, o) in y.chunks_mut(c).enumerate() {
            let t = t0 + r;
            o.copy_from_slice(l.dw_b);
            for j in 0..7 {
                let src = t as isize + j as isize - 3;
                if src < 0 || src >= frames as isize {
                    continue;
                }
                let row = &input[src as usize * c..(src as usize + 1) * c];
                for ((o, w), v) in o.iter_mut().zip(&l.dw_w[j * c..(j + 1) * c]).zip(row) {
                    *o += w * v;
                }
            }
            let copy = o.to_vec();
            math::normalize(&copy, o, STAGE_EPS);
        }
        gemm::linear_block(&y, c, rows, &l.fc1, hidden, hidden_width, Act::Gelu, false, None);
    });
    let squares = hidden
        .par_chunks(hidden_width * 64)
        .map(|rows| {
            let mut s = vec![0.0f64; hidden_width];
            for row in rows.chunks(hidden_width) {
                for (s, v) in s.iter_mut().zip(row) {
                    *s += (*v as f64) * (*v as f64);
                }
            }
            s
        })
        .reduce(
            || vec![0.0f64; hidden_width],
            |mut a, b| {
                a.iter_mut().zip(&b).for_each(|(a, b)| *a += b);
                a
            },
        );
    let magnitude: Vec<f64> = squares.iter().map(|s| s.sqrt()).collect();
    let mean = magnitude.iter().sum::<f64>() / hidden_width as f64;
    let gain: Vec<f32> = magnitude.iter().zip(l.grn_gamma).map(|(m, g)| 1.0 + g * (m / (mean + 1e-6)) as f32).collect();
    gemm::linear(&hidden, hidden_width, frames, &l.fc2, x, c, Act::None, true, Some(&gain));
}
