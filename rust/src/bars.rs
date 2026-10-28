//! Sections and keys on the tracked bars (`sheetsage2_refined/sections.py`, `keys.py`): the per-beat inputs pooled
//! from the encoder states (averaged over three groups of blocks), the melody head's line and the chord head, the
//! section / key model (`SectionModel`, `KeyModel` with `LocalKeys`: convolutions over the beats, the mean per bar,
//! a Transformer over the bars) and the decoders (`segments`, `decode`).

use rayon::prelude::*;
use serde_json::Value;

use crate::Result;
use crate::encoder::WIDTH;
use crate::gemm::{self, Act};
use crate::math;
use crate::weights::{Linear, Weights};
use crate::{Outputs, post};

/// Encoder states `[lo, hi)` averaged into the per-beat states: low, middle, high.
pub const BLOCKS: [(usize, usize); 3] = [(0, 9), (9, 17), (17, 25)];
pub const GROUPS: usize = BLOCKS.len();
const SIXTEENTHS: usize = 4;
const CLASSES: usize = 13; // 12 pitch classes and the rest
const MELODY: usize = SIXTEENTHS * CLASSES;
const CHORD: usize = 38; // the chord head's root (13), bass (13) and tones (12)
const MAX_BAR_BEATS: usize = 16;
const LONGEST: usize = 64; // the longest phrase in bars that `segments` considers
const FRAME_RATE: f64 = 25.0;
const MELODY_LOWEST: usize = 21;
const PITCHES: usize = 89; // the line's rest and MIDI 21..108

pub const LABELS: [&str; 8] = ["silence", "intro", "verse", "pre-chorus", "chorus", "bridge", "interlude", "outro"];
pub const MODES: [&str; 9] =
    ["major", "dorian", "phrygian", "lydian", "mixolydian", "minor", "locrian", "harmonic_minor", "phrygian_dominant"];
const SHARP_NAMES: [&str; 12] = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];

fn f16_round(x: f32) -> f32 {
    half::f16::from_f32(x).to_f32()
}

fn lin(x: &[f32], m: usize, l: &Linear, act: Act) -> Vec<f32> {
    let mut out = vec![0.0f32; m * l.n];
    gemm::linear(x, l.k, m, l, &mut out, l.n, act, false, None);
    out
}

fn normalized(x: &[f32], width: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    out.par_chunks_mut(width).zip(x.par_chunks(width)).for_each(|(o, x)| math::normalize(x, o, 1e-5));
    out
}

fn log_softmax(x: &[f32]) -> Vec<f32> {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = x.iter().map(|v| (v - max).exp()).sum();
    let log = sum.ln();
    x.iter().map(|v| v - max - log).collect()
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn softmax(x: &[f32]) -> Vec<f32> {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = x.iter().map(|v| (v - max).exp()).collect();
    let sum: f32 = e.iter().sum();
    e.iter().map(|v| v / sum).collect()
}

// ------------------------------------------------------------------ per-frame and per-beat inputs

/// What the bar models read per 25 Hz frame (`HeadSections.frames`).
pub struct FrameInputs<'a> {
    pub frames: usize,
    pub states: &'a [u16],       // [frames, GROUPS * WIDTH] float16 bits
    pub pitch: Vec<f32>,         // [frames, 89] the line's softmax
    pub sung: Vec<f32>,          // [frames]
    pub chord: Option<Vec<f32>>, // [frames, 38]: root and bass softmaxes, tones' sigmoids
}

fn layer<'a>(outputs: &'a Outputs, name: &str) -> Result<(usize, &'a [f32])> {
    outputs
        .iter()
        .find(|(n, _, _)| n == name)
        .map(|(_, w, v)| (*w, v.as_slice()))
        .ok_or_else(|| format!("no {name} outputs").into())
}

impl<'a> FrameInputs<'a> {
    /// From the melody head's (and the chord head's) logits and the block means.
    pub fn new(melody: &Outputs, chord: Option<&Outputs>, states: &'a [u16]) -> Result<Self> {
        let (w, line) = layer(melody, "line").map_err(|_| "the bar models need a line melody head (melody_pitch)")?;
        let frames = line.len() / w;
        if states.len() != frames * GROUPS * WIDTH {
            return Err(
                format!("{} frames of block states for {frames} frames", states.len() / (GROUPS * WIDTH)).into()
            );
        }
        let mut pitch = Vec::with_capacity(frames * PITCHES);
        let mut sung = Vec::with_capacity(frames);
        for row in line.chunks(w) {
            let p = softmax(&row[..PITCHES]);
            sung.push((1.0 - p[0]) * sigmoid(row[w - 1]));
            pitch.extend(p);
        }
        let chord = match chord {
            Some(outs) => {
                let (root, bass, tones) = (layer(outs, "root")?, layer(outs, "bass")?, layer(outs, "tones")?);
                let mut out = Vec::with_capacity(frames * CHORD);
                for t in 0..frames {
                    out.extend(softmax(&root.1[t * 13..(t + 1) * 13]));
                    out.extend(softmax(&bass.1[t * 13..(t + 1) * 13]));
                    out.extend(tones.1[t * 12..(t + 1) * 12].iter().map(|&v| sigmoid(v)));
                }
                Some(out)
            }
            None => None,
        };
        Ok(Self { frames, states, pitch, sung, chord })
    }
}

fn median(x: &[f64]) -> f64 {
    let mut v = x.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 { v[n / 2] } else { (v[n / 2 - 1] + v[n / 2]) / 2.0 }
}

/// `beats` with one more edge a median beat after the last.
fn beat_edges(beats: &[f64]) -> Vec<f64> {
    let diffs: Vec<f64> = beats.windows(2).map(|w| w[1] - w[0]).collect();
    let step = if beats.len() > 1 { median(&diffs) } else { 0.5 };
    let mut edges = beats.to_vec();
    edges.push(beats[beats.len() - 1] + step);
    edges
}

/// A span's first and last-plus-one frame from its edges in frames (`sections.beat_spans`).
fn span(lo: f64, hi: f64, frames: usize) -> (usize, usize) {
    let lo = lo.ceil().clamp(0.0, frames as f64 - 1.0) as usize;
    let hi = hi.ceil().clamp(1.0, frames as f64) as usize;
    (lo, hi.max(lo + 1))
}

/// Means of the rows of `values` `[frames, width]` over each span.
fn pool(values: &[f32], width: usize, spans: &[(usize, usize)]) -> Vec<f32> {
    let mut out = vec![0.0f32; spans.len() * width];
    out.par_chunks_mut(width).zip(spans).for_each(|(o, &(lo, hi))| {
        let mut sum = vec![0.0f64; width];
        for row in values[lo * width..hi * width].chunks(width) {
            for (s, v) in sum.iter_mut().zip(row) {
                *s += *v as f64;
            }
        }
        for (o, s) in o.iter_mut().zip(sum) {
            *o = (s / (hi - lo) as f64) as f32;
        }
    });
    out
}

/// Per-beat inputs (`sections.beat_features`, the chord as `HeadKeys.keys`), float16-rounded as the training
/// features.
pub struct BeatFeatures {
    pub states: Vec<f32>,        // [B, GROUPS * WIDTH]
    pub melody: Vec<f32>,        // [B, 4 * 13]
    pub vocal: Vec<f32>,         // [B]
    pub chord: Option<Vec<f32>>, // [B, 38]
}

pub fn beat_features(f: &FrameInputs, beats: &[f64]) -> BeatFeatures {
    let frames = f.frames;
    let edges = beat_edges(beats);
    let spans: Vec<(usize, usize)> =
        edges.windows(2).map(|e| span(e[0] * FRAME_RATE - 0.5, e[1] * FRAME_RATE - 0.5, frames)).collect();
    let width = GROUPS * WIDTH;
    let states: Vec<f32> = {
        let mut out = vec![0.0f32; spans.len() * width];
        out.par_chunks_mut(width).zip(&spans).for_each(|(o, &(lo, hi))| {
            let mut sum = vec![0.0f64; width];
            for row in f.states[lo * width..hi * width].chunks(width) {
                for (s, v) in sum.iter_mut().zip(row) {
                    *s += half::f16::from_bits(*v).to_f32() as f64;
                }
            }
            for (o, s) in o.iter_mut().zip(sum) {
                *o = f16_round((s / (hi - lo) as f64) as f32);
            }
        });
        out
    };
    // the line folded onto the pitch classes and the rest, averaged over each sixteenth
    let mut classes = vec![0.0f32; frames * CLASSES];
    for (o, p) in classes.chunks_mut(CLASSES).zip(f.pitch.chunks(PITCHES)) {
        for (i, v) in p[1..].iter().enumerate() {
            o[(i + MELODY_LOWEST) % 12] += v;
        }
        o[12] = p[0];
    }
    let slots: Vec<(usize, usize)> = edges
        .windows(2)
        .flat_map(|e| {
            let d = e[1] - e[0];
            (0..SIXTEENTHS).map(move |k| {
                let at = |k: usize| e[0] + d * k as f64 / SIXTEENTHS as f64;
                span(at(k) * FRAME_RATE - 0.5, at(k + 1) * FRAME_RATE - 0.5, frames)
            })
        })
        .collect();
    let melody = pool(&classes, CLASSES, &slots).into_iter().map(f16_round).collect();
    let vocal = pool(&f.sung, 1, &spans).into_iter().map(f16_round).collect();
    let chord = f.chord.as_ref().map(|c| pool(c, CHORD, &spans).into_iter().map(f16_round).collect());
    BeatFeatures { states, melody, vocal, chord }
}

/// Each beat's bar (beats before the first downbeat make a bar of their own) and each bar's first beat.
pub fn bar_layout(beats: &[f64], downbeats: &[f64]) -> (Vec<usize>, Vec<usize>) {
    let index: Vec<i64> = beats.iter().map(|&b| downbeats.partition_point(|&d| d <= b + 1e-3) as i64 - 1).collect();
    let min = index.iter().copied().min().unwrap_or(0);
    let index: Vec<usize> = index.iter().map(|&i| (i - min) as usize).collect();
    let starts = (0..index.len()).filter(|&i| i == 0 || index[i] != index[i - 1]).collect();
    (index, starts)
}

// ------------------------------------------------------------------ the models

struct ConvBlock {
    norm: (&'static [f32], &'static [f32]),
    depth: (&'static [f32], &'static [f32]), // [C, kernel], [C]
    point: Linear,
    out: Linear,
}

impl ConvBlock {
    fn load(w: &Weights, p: &str) -> Result<Self> {
        Ok(Self {
            norm: (w.f32(&format!("{p}.norm.w"))?, w.f32(&format!("{p}.norm.b"))?),
            depth: (w.f32(&format!("{p}.depth.w"))?, w.f32(&format!("{p}.depth.b"))?),
            point: w.linear(&format!("{p}.point"))?,
            out: w.linear(&format!("{p}.out"))?,
        })
    }

    /// `x` `[seqs * len, C]`: `seqs` sequences of `len` rows, each convolved on its own (zero padding).
    fn forward(&self, x: &mut [f32], seqs: usize, len: usize) {
        let c = self.norm.0.len();
        let kernel = self.depth.0.len() / c;
        let half = kernel / 2;
        let mut y = vec![0.0f32; x.len()];
        y.par_chunks_mut(c)
            .zip(x.par_chunks(c))
            .for_each(|(o, x)| math::layer_norm(x, o, self.norm.0, self.norm.1, 1e-5));
        let mut z = vec![0.0f32; x.len()];
        z.par_chunks_mut(c).enumerate().for_each(|(row, o)| {
            let (s, t) = (row / len, row % len);
            for (ch, o) in o.iter_mut().enumerate() {
                let mut acc = self.depth.1[ch];
                for j in 0..kernel {
                    let at = t as isize + j as isize - half as isize;
                    if at >= 0 && (at as usize) < len {
                        acc += self.depth.0[ch * kernel + j] * y[(s * len + at as usize) * c + ch];
                    }
                }
                *o = acc;
            }
        });
        let rows = seqs * len;
        let h = lin(&z, rows, &self.point, Act::Gelu);
        let out = lin(&h, rows, &self.out, Act::None);
        x.iter_mut().zip(&out).for_each(|(x, o)| *x += o);
    }
}

struct AttentionBlock {
    qkv: Linear, // the pre-LayerNorms folded in
    proj: Linear,
    ff1: Linear,
    ff2: Linear,
}

impl AttentionBlock {
    /// `x` `[n, width]`, `bias` `[heads, n, n]`.
    fn forward(&self, x: &mut [f32], n: usize, heads: usize, bias: &[f32]) {
        let width = self.proj.n;
        let dim = width / heads;
        let scale = 1.0 / (dim as f32).sqrt();
        let qkv = lin(&normalized(x, width), n, &self.qkv, Act::None);
        let mut y = vec![0.0f32; n * width];
        y.par_chunks_mut(width).enumerate().for_each(|(i, out)| {
            let mut scores = vec![0.0f32; n];
            for h in 0..heads {
                let q = &qkv[i * 3 * width + h * dim..][..dim];
                for (j, s) in scores.iter_mut().enumerate() {
                    let k = &qkv[j * 3 * width + width + h * dim..][..dim];
                    *s = q.iter().zip(k).map(|(a, b)| a * b).sum::<f32>() * scale + bias[(h * n + i) * n + j];
                }
                let p = softmax(&scores);
                let o = &mut out[h * dim..(h + 1) * dim];
                for (j, p) in p.iter().enumerate() {
                    let v = &qkv[j * 3 * width + 2 * width + h * dim..][..dim];
                    for (o, v) in o.iter_mut().zip(v) {
                        *o += p * v;
                    }
                }
            }
        });
        let out = lin(&y, n, &self.proj, Act::None);
        x.iter_mut().zip(&out).for_each(|(x, o)| *x += o);
        let h = lin(&normalized(x, width), n, &self.ff1, Act::Gelu);
        let out = lin(&h, n, &self.ff2, Act::None);
        x.iter_mut().zip(&out).for_each(|(x, o)| *x += o);
    }
}

/// `keys.LocalKeys`: per bar and candidate tonic, the modes' logits from the chords and melody around the bar.
struct LocalKeys {
    input: Linear,
    beat_position: &'static [f32],
    beat_layers: Vec<ConvBlock>,
    bar_layers: Vec<ConvBlock>,
    head: Linear, // its LayerNorm folded in
}

/// Section priors (`sections.Priors`): log P(first label), log P(label | previous), log P(length | label), log P(last).
struct Priors {
    first: Vec<f64>,
    transition: Vec<f64>, // [labels, labels]
    length: Vec<f64>,     // [labels, LONGEST + 1]
    last: Vec<f64>,
}

pub struct BarModel {
    width: usize,
    heads: usize,
    relative: usize,
    beat_in: Linear,
    beat_position: &'static [f32],
    bar_beats: &'static [f32],
    song_position: Linear,
    beat_layers: Vec<ConvBlock>,
    bar_layers: Vec<AttentionBlock>,
    distance: &'static [f32],
    similar: Option<&'static [f32]>,
    outputs: Vec<(String, Linear)>,
    local: Option<LocalKeys>,
    local_weight: f32,
    priors: Option<Priors>,
    pub decoding: Value,
}

/// One song's model inputs (`sections.model_inputs`, `keys.key_inputs`).
struct Inputs {
    beats: usize,
    bars: usize,
    beat_x: Vec<f32>, // [B, beat_in.k]: the normalised states, the melody, the sung share and the chord
    beat_bar: Vec<usize>,
    beat_position: Vec<usize>,
    bar_beats: Vec<usize>,
    bar_melody: Vec<f32>, // [bars, 16 * 12]
    melody: Vec<f32>,
    vocal: Vec<f32>,
    chord: Vec<f32>,
}

impl BarModel {
    /// `name`: `sections` or `keys`, as `export_cpu` writes them (`bar_models` in the metadata).
    pub fn load(w: &Weights, name: &str) -> Result<Self> {
        let all: Value = serde_json::from_str(w.metadata.get("bar_models").map_or("{}", |s| s))?;
        let spec = all.get(name).ok_or_else(|| {
            format!("the weights have no {name} model (export them again with this version's export_cpu)")
        })?;
        let config = &spec["config"];
        let get = |k: &str| config[k].as_u64().map(|v| v as usize).ok_or_else(|| format!("{name}: no {k} in config"));
        let mut decoding = spec["decoding"].clone();
        if name == "keys" {
            let counts: Vec<f64> = serde_json::from_str(w.metadata.get("mode_counts").map_or("[]", |s| s))?;
            decoding["mode_counts"] = counts.into();
        }
        let p = name;
        let blocks = |kind: &str, w: &Weights| -> Result<Vec<ConvBlock>> {
            let mut out = Vec::new();
            while w.has(&format!("{p}.{kind}.{}.point.w", out.len())) {
                out.push(ConvBlock::load(w, &format!("{p}.{kind}.{}", out.len()))?);
            }
            Ok(out)
        };
        let bar_layers = (0..get("bar_layers")?)
            .map(|i| {
                let q = format!("{p}.bar_layers.{i}");
                Ok(AttentionBlock {
                    qkv: w.linear(&format!("{q}.qkv"))?,
                    proj: w.linear(&format!("{q}.proj"))?,
                    ff1: w.linear(&format!("{q}.ff1"))?,
                    ff2: w.linear(&format!("{q}.ff2"))?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let mut outputs = Vec::new();
        for out in ["label", "boundary", "phrase", "key"] {
            if w.has(&format!("{p}.out.{out}.w")) {
                outputs.push((out.to_string(), w.linear(&format!("{p}.out.{out}"))?));
            }
        }
        let local = if w.has(&format!("{p}.local.input.w")) {
            Some(LocalKeys {
                input: w.linear(&format!("{p}.local.input"))?,
                beat_position: w.f32(&format!("{p}.local.beat_position"))?,
                beat_layers: blocks("local.beat_layers", w)?,
                bar_layers: blocks("local.bar_layers", w)?,
                head: w.linear(&format!("{p}.local.head"))?,
            })
        } else {
            None
        };
        let priors = if w.has(&format!("{p}.priors.first")) {
            let v = |k: &str| -> Result<Vec<f64>> {
                Ok(w.f32(&format!("{p}.priors.{k}"))?.iter().map(|&v| v as f64).collect())
            };
            Some(Priors { first: v("first")?, transition: v("transition")?, length: v("length")?, last: v("last")? })
        } else {
            None
        };
        let local_weight = decoding
            .get("local_weight")
            .and_then(Value::as_f64)
            .or_else(|| config.get("local_weight").and_then(Value::as_f64))
            .unwrap_or(1.0) as f32;
        let model = Self {
            width: get("width")?,
            heads: get("heads")?,
            relative: get("relative")?,
            beat_in: w.linear(&format!("{p}.beat_in"))?,
            beat_position: w.f32(&format!("{p}.beat_position"))?,
            bar_beats: w.f32(&format!("{p}.bar_beats"))?,
            song_position: w.linear(&format!("{p}.song_position"))?,
            beat_layers: blocks("beat_layers", w)?,
            bar_layers,
            distance: w.f32(&format!("{p}.distance"))?,
            similar: if w.has(&format!("{p}.similar")) { Some(w.f32(&format!("{p}.similar"))?) } else { None },
            outputs,
            local,
            local_weight,
            priors,
            decoding,
        };
        Ok(model)
    }

    /// Whether the model reads the chord head.
    pub fn needs_chord(&self) -> bool {
        self.beat_in.k > GROUPS * WIDTH + MELODY + 1
    }

    fn inputs(&self, beats: &[f64], downbeats: &[f64], f: &BeatFeatures) -> Inputs {
        let (beat_bar, starts) = bar_layout(beats, downbeats);
        let (n, bars) = (beats.len(), starts.len());
        let beat_position = (0..n).map(|i| i - starts[beat_bar[i]]).collect();
        let mut bar_beats = vec![0usize; bars];
        for &b in &beat_bar {
            bar_beats[b] += 1;
        }
        let mut bar_melody = vec![0.0f32; bars * 4 * MELODY / CLASSES * 12];
        for (bar, &first) in starts.iter().enumerate() {
            for k in 0..4 {
                let beat = first + k;
                if beat < n && beat_bar[beat] == bar {
                    for s in 0..SIXTEENTHS {
                        let to = &mut bar_melody[((bar * 4 + k) * SIXTEENTHS + s) * 12..][..12];
                        to.copy_from_slice(&f.melody[(beat * SIXTEENTHS + s) * CLASSES..][..12]);
                    }
                }
            }
        }
        let k = self.beat_in.k;
        let chord = f.chord.clone().unwrap_or_default();
        let mut beat_x = vec![0.0f32; n * k];
        beat_x.par_chunks_mut(k).enumerate().for_each(|(i, row)| {
            for g in 0..GROUPS {
                let at = (i * GROUPS + g) * WIDTH;
                math::normalize(&f.states[at..at + WIDTH], &mut row[g * WIDTH..(g + 1) * WIDTH], 1e-5);
            }
            let rest = &mut row[GROUPS * WIDTH..];
            rest[..MELODY].copy_from_slice(&f.melody[i * MELODY..(i + 1) * MELODY]);
            rest[MELODY] = f.vocal[i];
            if rest.len() > MELODY + 1 {
                rest[MELODY + 1..].copy_from_slice(&chord[i * CHORD..(i + 1) * CHORD]);
            }
        });
        Inputs {
            beats: n,
            bars,
            beat_x,
            beat_bar,
            beat_position,
            bar_beats,
            bar_melody,
            melody: f.melody.clone(),
            vocal: f.vocal.clone(),
            chord,
        }
    }

    /// Per bar `[bars, width]` before the output layers (normalised: the final LayerNorm is in them), and the bars.
    fn encode(&self, x: &Inputs) -> Vec<f32> {
        let (n, bars, w) = (x.beats, x.bars, self.width);
        let mut h = lin(&x.beat_x, n, &self.beat_in, Act::None);
        for (row, &p) in h.chunks_mut(w).zip(&x.beat_position) {
            let e = &self.beat_position[p.min(MAX_BAR_BEATS - 1) * w..][..w];
            row.iter_mut().zip(e).for_each(|(r, e)| *r += e);
        }
        for layer in &self.beat_layers {
            layer.forward(&mut h, 1, n);
        }
        let pooled = bar_means(&h, &x.beat_bar, bars, w, 1, n);
        let fourier: Vec<f32> = (0..bars)
            .flat_map(|b| {
                let at = b as f32 / bars as f32;
                [1.0f64, 2.0, 4.0, 8.0].into_iter().flat_map(move |k| {
                    let v = (std::f64::consts::PI * k) as f32 * at;
                    [v.sin(), v.cos()]
                })
            })
            .collect();
        let mut y = lin(&fourier, bars, &self.song_position, Act::None);
        for (b, row) in y.chunks_mut(w).enumerate() {
            let e = &self.bar_beats[x.bar_beats[b].min(MAX_BAR_BEATS) * w..][..w];
            for ((r, e), p) in row.iter_mut().zip(e).zip(&pooled[b * w..]) {
                *r = p + *r + e;
            }
        }
        let similar = self.similar.map(|_| [cosine(&x.bar_melody, bars), cosine(&pooled, bars)]);
        let r = self.relative as isize;
        let span = 2 * self.relative + 1;
        for (l, layer) in self.bar_layers.iter().enumerate() {
            let mut bias = vec![0.0f32; self.heads * bars * bars];
            for h in 0..self.heads {
                let distance = &self.distance[(l * self.heads + h) * span..][..span];
                for i in 0..bars {
                    for j in 0..bars {
                        let mut v = distance[((j as isize - i as isize).clamp(-r, r) + r) as usize];
                        if let (Some(weights), Some(sim)) = (self.similar, &similar) {
                            let s = &weights[(l * self.heads + h) * 2..][..2];
                            v += s[0] * sim[0][i * bars + j] + s[1] * sim[1][i * bars + j];
                        }
                        bias[(h * bars + i) * bars + j] = v;
                    }
                }
            }
            layer.forward(&mut y, bars, self.heads, &bias);
        }
        normalized(&y, w)
    }

    fn output(&self, y: &[f32], bars: usize, name: &str) -> Vec<f32> {
        let (_, l) = self.outputs.iter().find(|(n, _)| n == name).expect("output layer");
        lin(y, bars, l, Act::None)
    }

    /// `keys.LocalKeys.forward`: `[bars, 9 * 12]` logits, index `12 * mode + tonic`.
    fn local_keys(&self, local: &LocalKeys, x: &Inputs) -> Vec<f32> {
        let (n, bars) = (x.beats, x.bars);
        let context = chord_context(&x.chord, n);
        let features = LocalKeys::FEATURES;
        let mut input = vec![0.0f32; 12 * n * features];
        input.par_chunks_mut(features).enumerate().for_each(|(row, out)| {
            let (t, i) = (row / n, row % n);
            let roll = |v: &[f32], j: usize| v[(j + t) % 12];
            let parts = |c: &[f32]| -> [f32; CHORD] {
                let mut o = [0.0f32; CHORD];
                for j in 0..12 {
                    o[j] = roll(&c[0..12], j);
                    o[13 + j] = roll(&c[13..25], j);
                    o[26 + j] = roll(&c[26..38], j);
                }
                o[12] = c[12];
                o[25] = c[25];
                o
            };
            let cur = parts(&x.chord[i * CHORD..(i + 1) * CHORD]);
            let prev = parts(&context[i * 2 * CHORD..i * 2 * CHORD + CHORD]);
            let next = parts(&context[i * 2 * CHORD + CHORD..(i + 1) * 2 * CHORD]);
            let mut at = 0;
            for part in [&cur, &prev, &next] {
                out[at..at + CHORD].copy_from_slice(part);
                at += CHORD;
            }
            for (a, b) in [(&prev[0..12], &cur[0..12]), (&cur[0..12], &next[0..12]), (&prev[13..25], &cur[13..25])] {
                for p in a {
                    for q in b {
                        out[at] = p * q;
                        at += 1;
                    }
                }
            }
            let melody = &x.melody[i * MELODY..(i + 1) * MELODY];
            for s in 0..SIXTEENTHS {
                for j in 0..12 {
                    out[at] = roll(&melody[s * CLASSES..s * CLASSES + 12], j);
                    at += 1;
                }
            }
            for s in 0..SIXTEENTHS {
                out[at] = melody[s * CLASSES + 12];
                at += 1;
            }
            out[at] = x.vocal[i];
        });
        let w = local.input.n;
        let mut h = lin(&input, 12 * n, &local.input, Act::None);
        for (row, chunk) in h.chunks_mut(w).enumerate() {
            let p = x.beat_position[row % n].min(MAX_BAR_BEATS - 1);
            chunk.iter_mut().zip(&local.beat_position[p * w..(p + 1) * w]).for_each(|(c, e)| *c += e);
        }
        for layer in &local.beat_layers {
            layer.forward(&mut h, 12, n);
        }
        let mut y = bar_means(&h, &x.beat_bar, bars, w, 12, n);
        for layer in &local.bar_layers {
            layer.forward(&mut y, 12, bars);
        }
        let logits = lin(&normalized(&y, w), 12 * bars, &local.head, Act::None); // [tonic, bar, mode]
        let modes = local.head.n;
        let mut out = vec![0.0f32; bars * modes * 12];
        for t in 0..12 {
            for b in 0..bars {
                for m in 0..modes {
                    out[(b * modes + m) * 12 + t] = logits[(t * bars + b) * modes + m];
                }
            }
        }
        out
    }

    /// The sections (`HeadSections.sections`): `(first bar, end bar, label)` and the bars' first beats.
    pub fn sections(&self, beats: &[f64], downbeats: &[f64], f: &BeatFeatures) -> Vec<(usize, usize, usize)> {
        let x = self.inputs(beats, downbeats, f);
        let y = self.encode(&x);
        let labels = self.output(&y, x.bars, "label");
        let label_logp: Vec<f32> = labels.chunks(LABELS.len()).flat_map(log_softmax).collect();
        let boundary: Vec<f32> = self.output(&y, x.bars, "boundary").into_iter().map(sigmoid).collect();
        let get = |k: &str, d: f64| self.decoding.get(k).and_then(Value::as_f64).unwrap_or(d);
        segments(
            &label_logp,
            &boundary,
            self.priors.as_ref().expect("section priors"),
            get("boundary_weight", 1.0),
            get("length_weight", 1.0),
            get("transition_weight", 1.0),
        )
    }

    /// The key of each bar (`HeadKeys.keys`), indices `12 * mode + tonic`.
    pub fn keys(&self, beats: &[f64], downbeats: &[f64], f: &BeatFeatures) -> Vec<usize> {
        let x = self.inputs(beats, downbeats, f);
        let y = self.encode(&x);
        let whole = self.output(&y, x.bars, "key");
        let keys = whole.len() / x.bars;
        let mut key: Vec<f32> = whole.chunks(keys).flat_map(log_softmax).collect();
        if let Some(local) = &self.local {
            let l = self.local_keys(local, &x);
            for (k, l) in key.chunks_mut(keys).zip(l.chunks(keys)) {
                k.iter_mut().zip(log_softmax(l)).for_each(|(k, l)| *k += self.local_weight * l);
            }
        }
        let tau = self.decoding.get("mode_tau").and_then(Value::as_f64).unwrap_or(0.0);
        let counts: Vec<f64> = self.decoding["mode_counts"]
            .as_array()
            .map(|a| a.iter().filter_map(Value::as_f64).collect())
            .unwrap_or_default();
        let total: f64 = counts.iter().sum();
        let key_logp: Vec<f64> = key
            .chunks(keys)
            .flat_map(log_softmax)
            .enumerate()
            .map(|(i, v)| {
                let mode = (i % keys) / 12;
                if tau != 0.0 && !counts.is_empty() { v as f64 + -tau * (counts[mode] / total).ln() } else { v as f64 }
            })
            .collect();
        let boundary: Vec<f32> = self.output(&y, x.bars, "boundary").into_iter().map(sigmoid).collect();
        let weight = self.decoding.get("boundary_weight").and_then(Value::as_f64).unwrap_or(1.0);
        decode(&key_logp, keys, &boundary, weight)
    }
}

impl LocalKeys {
    const FEATURES: usize = 3 * CHORD + 3 * 144 + 4 * 13 + 1;
}

/// Means of the rows of `x` `[seqs * len, width]` over each bar's beats: `[seqs * bars, width]`.
fn bar_means(x: &[f32], beat_bar: &[usize], bars: usize, width: usize, seqs: usize, len: usize) -> Vec<f32> {
    let mut count = vec![0usize; bars];
    for &b in beat_bar {
        count[b] += 1;
    }
    let mut out = vec![0.0f32; seqs * bars * width];
    out.par_chunks_mut(bars * width).enumerate().for_each(|(s, out)| {
        for (i, &b) in beat_bar.iter().enumerate() {
            let row = &x[(s * len + i) * width..][..width];
            out[b * width..(b + 1) * width].iter_mut().zip(row).for_each(|(o, v)| *o += v);
        }
        for (b, row) in out.chunks_mut(width).enumerate() {
            let c = count[b].max(1) as f32;
            row.iter_mut().for_each(|v| *v /= c);
        }
    });
    out
}

/// Cosine similarities of the rows of `x` `[n, d]`: `[n, n]` (zero rows give zeros).
fn cosine(x: &[f32], n: usize) -> Vec<f32> {
    let d = x.len() / n;
    let unit: Vec<f32> = x
        .chunks(d)
        .flat_map(|r| {
            let norm = r.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
            r.iter().map(move |v| v / norm)
        })
        .collect();
    let mut out = vec![0.0f32; n * n];
    out.par_chunks_mut(n).enumerate().for_each(|(i, o)| {
        let a = &unit[i * d..(i + 1) * d];
        for (j, o) in o.iter_mut().enumerate() {
            *o = a.iter().zip(&unit[j * d..(j + 1) * d]).map(|(a, b)| a * b).sum();
        }
    });
    out
}

/// `keys.chord_context`: per beat the chord before its chord and the one after it (`[B, 76]`).
fn chord_context(chord: &[f32], n: usize) -> Vec<f32> {
    let signature = |i: usize| -> (usize, [bool; 12]) {
        let c = &chord[i * CHORD..(i + 1) * CHORD];
        let mut root = 0;
        for j in 1..13 {
            if c[j] > c[root] {
                root = j;
            }
        }
        let mut tones = [false; 12];
        for j in 0..12 {
            tones[j] = c[26 + j] > 0.5;
        }
        (root, tones)
    };
    let signatures: Vec<_> = (0..n).map(signature).collect();
    let first: Vec<usize> = (0..n).filter(|&i| i == 0 || signatures[i] != signatures[i - 1]).collect();
    let last: Vec<usize> = first.iter().skip(1).map(|f| f - 1).chain([n - 1]).collect();
    let mut out = vec![0.0f32; n * 2 * CHORD];
    let mut run = 0;
    for i in 0..n {
        if run + 1 < first.len() && first[run + 1] == i {
            run += 1;
        }
        let row = &mut out[i * 2 * CHORD..(i + 1) * 2 * CHORD];
        if run > 0 {
            let b = last[run - 1];
            row[..CHORD].copy_from_slice(&chord[b * CHORD..(b + 1) * CHORD]);
        }
        if run + 1 < first.len() {
            let b = first[run + 1];
            row[CHORD..].copy_from_slice(&chord[b * CHORD..(b + 1) * CHORD]);
        }
    }
    out
}

fn argmax(x: impl IntoIterator<Item = f64>) -> usize {
    let (mut best, mut at) = (f64::NEG_INFINITY, 0);
    for (i, v) in x.into_iter().enumerate() {
        if v > best || i == 0 {
            (best, at) = (v, i);
        }
    }
    at
}

/// `sections.segments`: the likeliest phrases `(first bar, end bar, label)` (semi-Markov Viterbi) from per-bar label
/// log-probabilities `[bars, labels]` and boundary probabilities `[bars]`.
fn segments(
    label_logp: &[f32],
    boundary_p: &[f32],
    priors: &Priors,
    boundary_weight: f64,
    length_weight: f64,
    transition_weight: f64,
) -> Vec<(usize, usize, usize)> {
    let bars = boundary_p.len();
    let n = label_logp.len() / bars;
    // as numpy: the boundary terms and the cumulative sums in float32
    let p: Vec<f32> = boundary_p.iter().map(|p| p.clamp(1e-6, 1.0 - 1e-6)).collect();
    let mut start_score: Vec<f32> = p.iter().map(|p| boundary_weight as f32 * p.ln()).collect();
    let inside: Vec<f32> = p.iter().map(|p| boundary_weight as f32 * (-p).ln_1p()).collect();
    start_score[0] = 0.0;
    let mut cum_label = vec![0.0f64; (bars + 1) * n];
    let mut running = vec![0.0f32; n];
    for b in 0..bars {
        for l in 0..n {
            running[l] += label_logp[b * n + l];
            cum_label[(b + 1) * n + l] = running[l] as f64;
        }
    }
    let mut cum_inside = vec![0.0f64; bars + 1];
    let mut acc = 0.0f32;
    for b in 0..bars {
        acc += inside[b];
        cum_inside[b + 1] = acc as f64;
    }
    let mut best = vec![f64::NEG_INFINITY; (bars + 1) * n];
    let mut back = vec![(0usize, -1i64); (bars + 1) * n];
    let lengths = LONGEST + 1;
    for end in 1..=bars {
        for length in 1..=LONGEST.min(end) {
            let start = end - length;
            for l in 0..n {
                let score =
                    cum_label[end * n + l] - cum_label[start * n + l] + start_score[start] as f64 + cum_inside[end]
                        - cum_inside[start + 1]
                        + length_weight * priors.length[l * lengths + length];
                let (total, previous) = if start == 0 {
                    (score + transition_weight * priors.first[l], -1)
                } else {
                    let previous =
                        argmax((0..n).map(|q| best[start * n + q] + transition_weight * priors.transition[q * n + l]));
                    let option = best[start * n + previous] + transition_weight * priors.transition[previous * n + l];
                    (score + option, previous as i64)
                };
                if total > best[end * n + l] {
                    best[end * n + l] = total;
                    back[end * n + l] = (length, previous);
                }
            }
        }
    }
    let mut label = argmax((0..n).map(|l| best[bars * n + l] + transition_weight * priors.last[l]));
    let (mut out, mut end) = (Vec::new(), bars);
    while end > 0 {
        let (length, previous) = back[end * n + label];
        out.push((end - length, end, label));
        end -= length;
        label = previous.max(0) as usize;
    }
    out.reverse();
    out
}

/// `keys.decode`: the likeliest key per bar from per-bar key log-probabilities `[bars, keys]` and change
/// probabilities `[bars]`.
fn decode(key_logp: &[f64], n: usize, boundary_p: &[f32], boundary_weight: f64) -> Vec<usize> {
    let bars = boundary_p.len();
    let p: Vec<f32> = boundary_p.iter().map(|p| p.clamp(1e-6, 1.0 - 1e-6)).collect();
    let change: Vec<f32> = p.iter().map(|p| boundary_weight as f32 * p.ln()).collect();
    let stay: Vec<f32> = p.iter().map(|p| boundary_weight as f32 * (-p).ln_1p()).collect();
    let mut best = key_logp[..n].to_vec();
    let mut back = vec![0usize; bars * n];
    for i in 1..bars {
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| best[b].partial_cmp(&best[a]).unwrap());
        let (first, second) = (order[0], order[1]);
        let mut next = vec![0.0f64; n];
        for k in 0..n {
            let other = if k == first { second } else { first };
            let moved = best[other] + change[i] as f64;
            let kept = best[k] + stay[i] as f64;
            back[i * n + k] = if kept >= moved { k } else { other };
            next[k] = kept.max(moved) + key_logp[i * n + k];
        }
        best = next;
    }
    let mut out = vec![0usize; bars];
    out[bars - 1] = argmax(best.iter().copied());
    for i in (1..bars).rev() {
        out[i - 1] = back[i * n + out[i]];
    }
    out
}

pub fn key_name(index: usize) -> String {
    format!("{}:{}", SHARP_NAMES[index % 12], MODES[index / 12])
}

// ------------------------------------------------------------------ JSON

/// The bars' first beats and the song's end: the edges of the sections and keys.
fn bar_edges(beats: &[f64], downbeats: &[f64], duration: f64) -> Vec<f64> {
    let (_, starts) = bar_layout(beats, downbeats);
    let mut edges: Vec<f64> = starts.iter().map(|&s| beats[s]).collect();
    edges.push(duration.max(beats[beats.len() - 1]));
    edges
}

/// `sections.json`'s `sections` for the tracked (rounded) `beats` and `downbeats`.
pub fn sections_json(model: &BarModel, f: &FrameInputs, beats: &[f64], downbeats: &[f64], duration: f64) -> Value {
    if beats.len() < 2 {
        return Value::Array(Vec::new());
    }
    let features = beat_features(f, beats);
    let edges = bar_edges(beats, downbeats, duration);
    let found = model.sections(beats, downbeats, &features);
    found
        .iter()
        .map(|&(s, e, label)| {
            let (start, end) = (post::r4(edges[s]), post::r4(edges[e]));
            serde_json::json!({"start": start, "end": end, "label": LABELS[label], "bars": e - s})
        })
        .collect()
}

/// `keys.json`'s `keys` for the tracked (rounded) `beats` and `downbeats`.
pub fn keys_json(model: &BarModel, f: &FrameInputs, beats: &[f64], downbeats: &[f64], duration: f64) -> Value {
    if beats.len() < 2 {
        return Value::Array(Vec::new());
    }
    let features = beat_features(f, beats);
    let mut edges = bar_edges(beats, downbeats, duration);
    edges[0] = 0.0;
    let mut found: Vec<Value> = Vec::new();
    for (i, key) in model.keys(beats, downbeats, &features).into_iter().enumerate() {
        let name = key_name(key);
        match found.last_mut() {
            Some(last) if last["key"] == name.as_str() => {
                last["end"] = post::r4(edges[i + 1]).into();
                last["bars"] = (last["bars"].as_u64().unwrap_or(0) + 1).into();
            }
            _ => found.push(
                serde_json::json!({"start": post::r4(edges[i]), "end": post::r4(edges[i + 1]), "key": name, "bars": 1}),
            ),
        }
    }
    Value::Array(found)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bar_layout_counts_beats_before_the_first_downbeat_as_a_bar() {
        let beats = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
        let (index, starts) = bar_layout(&beats, &[1.0, 2.5]);
        assert_eq!(index, vec![0, 1, 1, 1, 2, 2]);
        assert_eq!(starts, vec![0, 1, 4]);
    }

    #[test]
    fn chord_context_reads_the_neighbouring_chords() {
        let mut chord = vec![0.0f32; 4 * CHORD];
        for (i, root) in [0usize, 0, 5, 7].iter().enumerate() {
            chord[i * CHORD + root] = 1.0;
        }
        let out = chord_context(&chord, 4);
        let row = |i: usize| &out[i * 2 * CHORD..(i + 1) * 2 * CHORD];
        assert!(row(0)[..CHORD].iter().all(|&v| v == 0.0) && row(0)[CHORD + 5] == 1.0);
        assert!(row(2)[0] == 1.0 && row(2)[CHORD + 7] == 1.0);
        assert!(row(3)[5] == 1.0 && row(3)[CHORD..].iter().all(|&v| v == 0.0));
    }

    #[test]
    fn decode_changes_key_on_a_likely_boundary() {
        let n = 3;
        let logp = [-0.1, -3.0, -3.0, -0.1, -3.0, -3.0, -3.0, -0.1, -3.0, -3.0, -0.1, -3.0];
        assert_eq!(decode(&logp, n, &[0.5, 0.01, 0.9, 0.01], 1.0), vec![0, 0, 1, 1]);
    }
}
