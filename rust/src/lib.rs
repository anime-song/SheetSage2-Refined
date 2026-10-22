//! SheetSage2-Refined's beat and melody heads on the CPU: the MERT-v2 encoder and the frame heads, from weights
//! exported by `python -m sheetsage2_refined.export_cpu`.

#![allow(unsafe_op_in_unsafe_fn)]

pub mod audio;
pub mod encoder;
pub mod gemm;
pub mod heads;
pub mod math;
pub mod melody;
pub mod post;
pub mod profile;
pub mod simd;
pub mod tracker;
pub mod weights;

use encoder::{Encoder, Mix, SAMPLE_RATE};
use heads::Head;
use weights::Weights;

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

/// Encoder windows of 300 s, each encoded at its own length (`beats.frame_outputs`, `model.WINDOW_SECONDS`).
pub const WINDOW_SAMPLES: usize = 300 * SAMPLE_RATE;

pub struct Model {
    pub encoder: Encoder,
    pub heads: Vec<(String, Head)>,
    pub weights: Weights,
}

/// One head's outputs over a whole song: `(output name, width, [frames, width] logits)`.
pub type Outputs = Vec<(String, usize, Vec<f32>)>;

impl Model {
    pub fn load(path: &std::path::Path, heads: &[&str]) -> Result<Self> {
        let weights = Weights::open(path)?;
        let encoder = Encoder::load(&weights)?;
        let heads = heads.iter().map(|&h| Ok((h.to_string(), Head::load(&weights, h)?))).collect::<Result<Vec<_>>>()?;
        Ok(Self { encoder, heads, weights })
    }

    /// The heads' logits for a 24 kHz waveform (`beats.frame_outputs`): per head, its outputs over all windows.
    pub fn frame_outputs(&self, wave: &[f32]) -> Result<Vec<(String, Outputs)>> {
        let mut all: Vec<(String, Outputs)> = self.heads.iter().map(|(n, _)| (n.clone(), Vec::new())).collect();
        for start in (0..wave.len()).step_by(WINDOW_SAMPLES) {
            let window = &wave[start..(start + WINDOW_SAMPLES).min(wave.len())];
            let mut mixes: Vec<Mix> = self.heads.iter().map(|(_, h)| Mix::new(h.mix)).collect();
            let frames = self.encoder.forward(window, &mut mixes.iter_mut().collect::<Vec<_>>())?;
            for ((_, head), (mix, (_, outputs))) in self.heads.iter().zip(mixes.iter().zip(all.iter_mut())) {
                let outs = profile::time("heads", || head.forward(&mix.sum, frames));
                for (i, (name, width, logits)) in outs.into_iter().enumerate() {
                    if outputs.len() <= i {
                        outputs.push((name, width, logits));
                    } else {
                        outputs[i].2.extend(logits);
                    }
                }
            }
        }
        Ok(all)
    }
}

/// Peak resident memory of this process in MiB (Linux), if known.
pub fn peak_rss_mib() -> Option<f64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|l| l.starts_with("VmHWM:"))?;
    let kib: f64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kib / 1024.0)
}
