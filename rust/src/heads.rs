//! Frame heads (`sheetsage2_refined/heads.py`'s `FrameHead`) on a layer mix: norm, linear + GELU, two grouped
//! convolutions, a bidirectional GRU (on 60 s pieces with 15 s of context either side, as `CHUNK`) and the output
//! layers the trackers read.

use rayon::prelude::*;

use crate::Result;
use crate::encoder::WIDTH;
use crate::gemm::{self, Act, MR, NR};
use crate::math;
use crate::weights::{Linear, Weights};

const CHUNK: (usize, usize) = (1500, 375);
const GROUP: usize = 32; // channels per group of the heads' convolutions
const TAPS: usize = 9;

struct Gru {
    ih: Linear,
    hh_w: &'static [f32], // [3 H, H], gates r, z, n
    hh_b: &'static [f32],
}

pub struct Head {
    pub mix: &'static [f32],
    inp: Linear,
    convs: Vec<(Vec<f32>, &'static [f32])>, // weights [groups][TAPS][GROUP in][GROUP out], bias
    rnn: Vec<[Gru; 2]>,
    pub outputs: Vec<(String, Linear)>,
}

impl Head {
    pub fn load(w: &Weights, name: &str) -> Result<Self> {
        let p = format!("heads.{name}");
        let inp = w.linear(&format!("{p}.inp"))?;
        let hidden = inp.n;
        let convs = (0..2)
            .map(|i| {
                let raw = w.f32(&format!("{p}.convs.{i}.w"))?; // [C, GROUP, TAPS]
                let mut out = vec![0.0; raw.len()];
                for o in 0..hidden {
                    let (g, ol) = (o / GROUP, o % GROUP);
                    for i in 0..GROUP {
                        for j in 0..TAPS {
                            out[((g * TAPS + j) * GROUP + i) * GROUP + ol] = raw[(o * GROUP + i) * TAPS + j];
                        }
                    }
                }
                Ok((out, w.f32(&format!("{p}.convs.{i}.b"))?))
            })
            .collect::<Result<Vec<_>>>()?;
        let mut rnn = Vec::new();
        while w.has(&format!("{p}.rnn.{}.fwd.ih.w", rnn.len())) {
            let q = format!("{p}.rnn.{}", rnn.len());
            let gru = |d: &str| -> Result<Gru> {
                Ok(Gru {
                    ih: w.linear(&format!("{q}.{d}.ih"))?,
                    hh_w: w.f32(&format!("{q}.{d}.hh.w"))?,
                    hh_b: w.f32(&format!("{q}.{d}.hh.b"))?,
                })
            };
            rnn.push([gru("fwd")?, gru("bwd")?]);
        }
        let mut outputs = Vec::new();
        for layer in ["binary", "bar", "offset", "unit", "melody", "line"] {
            let key = format!("{p}.out.{layer}");
            if w.has(&format!("{key}.w")) {
                outputs.push((layer.to_string(), w.linear(&key)?));
            }
        }
        Ok(Self { mix: w.f32(&format!("{p}.mix"))?, inp, convs, rnn, outputs })
    }

    /// The output layers' logits for a layer mix `[frames, WIDTH]`: `(name, width, [frames, width])`.
    pub fn forward(&self, mixed: &[f32], frames: usize) -> Vec<(String, usize, Vec<f32>)> {
        let hidden = self.inp.n;
        let mut norm = vec![0.0f32; frames * WIDTH];
        norm.par_chunks_mut(WIDTH).zip(mixed.par_chunks(WIDTH)).for_each(|(o, x)| math::normalize(x, o, 1e-5));
        let mut x = vec![0.0f32; frames * hidden];
        gemm::linear(&norm, WIDTH, frames, &self.inp, &mut x, hidden, Act::Gelu, false, None);
        drop(norm);
        for (w, b) in &self.convs {
            let y = grouped_conv(&x, frames, hidden, w, b);
            x.par_iter_mut().zip(&y).for_each(|(x, y)| *x += math::gelu(*y));
        }
        let x = self.rnn_chunked(&x, frames);
        self.outputs
            .iter()
            .map(|(name, lin)| {
                let mut out = vec![0.0f32; frames * lin.n];
                gemm::linear(&x, hidden, frames, lin, &mut out, lin.n, Act::None, false, None);
                (name.clone(), lin.n, out)
            })
            .collect()
    }

    fn rnn_chunked(&self, x: &[f32], frames: usize) -> Vec<f32> {
        let hidden = self.inp.n;
        let (core, context) = CHUNK;
        if frames <= core + 2 * context {
            return self.rnn(x, frames);
        }
        let mut out = vec![0.0f32; frames * hidden];
        let pieces: Vec<(usize, usize, usize)> = (0..frames)
            .step_by(core)
            .map(|s| (s, s.saturating_sub(context), (s + core + context).min(frames)))
            .collect();
        let results: Vec<Vec<f32>> =
            pieces.par_iter().map(|&(_, lo, hi)| self.rnn(&x[lo * hidden..hi * hidden], hi - lo)).collect();
        for (&(s, lo, _), y) in pieces.iter().zip(&results) {
            let end = (s + core).min(frames);
            out[s * hidden..end * hidden].copy_from_slice(&y[(s - lo) * hidden..(end - lo) * hidden]);
        }
        out
    }

    /// The bidirectional GRU over one piece, `[frames, hidden]` (forward and backward halves concatenated).
    fn rnn(&self, x: &[f32], frames: usize) -> Vec<f32> {
        let mut input = x.to_vec();
        for layer in &self.rnn {
            let h = layer[0].hh_b.len() / 3;
            let width = layer[0].ih.k;
            let mut out = vec![0.0f32; frames * 2 * h];
            let halves: Vec<Vec<f32>> = layer
                .par_iter()
                .enumerate()
                .map(|(d, gru)| {
                    let mut gates = vec![0.0f32; frames * 3 * h];
                    gemm::linear_block(&input, width, frames, &gru.ih, &mut gates, 3 * h, Act::None, false, None);
                    gru_pass(gru, &gates, frames, h, d == 1)
                })
                .collect();
            for t in 0..frames {
                out[t * 2 * h..t * 2 * h + h].copy_from_slice(&halves[0][t * h..(t + 1) * h]);
                out[t * 2 * h + h..(t + 1) * 2 * h].copy_from_slice(&halves[1][t * h..(t + 1) * h]);
            }
            input = out;
        }
        input
    }
}

/// One direction of a GRU layer (torch's equations) given its input gates `[frames, 3 H]`.
fn gru_pass(gru: &Gru, gates: &[f32], frames: usize, h: usize, reverse: bool) -> Vec<f32> {
    let mut out = vec![0.0f32; frames * h];
    let mut state = vec![0.0f32; h];
    let mut rec = vec![0.0f32; 3 * h];
    for step in 0..frames {
        let t = if reverse { frames - 1 - step } else { step };
        math::matvec(gru.hh_w, &state, gru.hh_b, &mut rec);
        let gi = &gates[t * 3 * h..(t + 1) * 3 * h];
        for i in 0..h {
            let r = math::sigmoid(gi[i] + rec[i]);
            let z = math::sigmoid(gi[h + i] + rec[h + i]);
            let n = (gi[2 * h + i] + r * rec[2 * h + i]).tanh();
            state[i] = (1.0 - z) * n + z * state[i];
        }
        out[t * h..(t + 1) * h].copy_from_slice(&state);
    }
    out
}

/// A grouped 1-D convolution, kernel 9, padding 4, groups of 32 channels, over `x` `[frames, c]`.
fn grouped_conv(x: &[f32], frames: usize, c: usize, w: &[f32], b: &[f32]) -> Vec<f32> {
    // Per group and tap a [rows, 32] x [32, 32] product on the GEMM kernel, 6 frames x 16 channels at a time, over
    // `x` with TAPS / 2 zero rows either side (and rows to a multiple of 6 after it).
    let half = TAPS / 2;
    let rows = frames.div_ceil(MR) * MR;
    let mut padded = vec![0.0f32; (rows + TAPS) * c];
    padded[half * c..(half + frames) * c].copy_from_slice(&x[..frames * c]);
    let mut y = vec![0.0f32; rows * c];
    y.par_chunks_mut(MR * c).enumerate().for_each(|(panel, out)| {
        for o in out.chunks_exact_mut(c) {
            o.copy_from_slice(b);
        }
        for g in 0..c / GROUP {
            for j in 0..TAPS {
                let a = &padded[(panel * MR + j) * c + g * GROUP..];
                let wj = &w[(g * TAPS + j) * GROUP * GROUP..][..GROUP * GROUP];
                for col in (0..GROUP).step_by(NR) {
                    gemm::kernel_into(GROUP, a, c, 1, &wj[col..], GROUP, &mut out[g * GROUP + col..], c, true);
                }
            }
        }
    });
    y.truncate(frames * c);
    y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grouped_conv_matches_direct_sums() {
        // 2 groups of 32 channels, 13 frames (not a multiple of 6, and shorter than twice the taps)
        let (frames, c) = (13, 2 * GROUP);
        let x: Vec<f32> = (0..frames * c).map(|i| ((i * 31) % 17) as f32 / 8.0 - 1.0).collect();
        let w: Vec<f32> = (0..c / GROUP * TAPS * GROUP * GROUP).map(|i| ((i * 13) % 23) as f32 / 50.0 - 0.2).collect();
        let b: Vec<f32> = (0..c).map(|o| o as f32 * 0.01).collect();
        let y = grouped_conv(&x, frames, c, &w, &b);
        for t in 0..frames {
            for o in 0..c {
                let (g, ol) = (o / GROUP, o % GROUP);
                let mut want = b[o] as f64;
                for j in 0..TAPS {
                    let Some(src) = (t + j).checked_sub(TAPS / 2).filter(|&s| s < frames) else { continue };
                    for i in 0..GROUP {
                        want += x[src * c + g * GROUP + i] as f64 * w[((g * TAPS + j) * GROUP + i) * GROUP + ol] as f64;
                    }
                }
                assert!((y[t * c + o] as f64 - want).abs() < 1e-4, "frame {t}, channel {o}");
            }
        }
    }
}
