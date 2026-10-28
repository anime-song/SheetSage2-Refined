//! SheetSage2-Refined's beat and melody heads on the CPU: the MERT-v2 encoder and the frame heads, from weights
//! exported by `python -m sheetsage2_refined.export_cpu`.

#![allow(unsafe_op_in_unsafe_fn)]

pub mod audio;
pub mod bars;
pub mod encoder;
pub mod gemm;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod heads;
pub mod lora;
pub mod math;
pub mod melody;
pub mod post;
pub mod profile;
pub mod simd;
pub mod tracker;
pub mod weights;

use bars::BarModel;
use encoder::{Encoder, Mix, N_FFT, SAMPLE_RATE, SAMPLES_PER_FRAME, WIDTH};
use heads::Head;
use weights::Weights;

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

/// Encoder windows of 300 s, each encoded at its own length (`beats.frame_outputs`, `model.WINDOW_SECONDS`).
pub const WINDOW_SAMPLES: usize = 300 * SAMPLE_RATE;

pub struct Model {
    pub encoder: Encoder,
    #[cfg(feature = "gpu")]
    pub gpu: Option<gpu::GpuEncoder>,
    pub heads: Vec<(String, Head)>,
    /// The section and key models (`--heads section,key`), which read the encoder states' block means.
    pub sections: Option<BarModel>,
    pub keys: Option<BarModel>,
    pub weights: Weights,
}

pub type Outputs = Vec<(String, usize, Vec<f32>)>;

/// A song's 25 Hz outputs: per frame head its logits, and the encoder states averaged over `bars::BLOCKS`
/// (`[frames, 3 * WIDTH]` float16 bits; empty without a bar model).
pub struct Frames {
    pub heads: Vec<(String, Outputs)>,
    pub states: Vec<u16>,
}

impl Model {
    /// `outputs`: frame heads (`beat`, `melody`, `chord`) and bar models (`section`, `key`); a bar model loads the
    /// frame heads it reads.
    pub fn load(path: &std::path::Path, outputs: &[&str]) -> Result<Self> {
        let weights = Weights::open(path)?;
        let encoder = Encoder::load(&weights)?;
        let sections = if outputs.contains(&"section") { Some(BarModel::load(&weights, "sections")?) } else { None };
        let keys = if outputs.contains(&"key") { Some(BarModel::load(&weights, "keys")?) } else { None };
        let mut names: Vec<&str> = Vec::new();
        for &o in outputs {
            let needs = match o {
                "section" => vec!["beat", "melody"],
                "key" if keys.as_ref().is_some_and(BarModel::needs_chord) => vec!["beat", "melody", "chord"],
                "key" => vec!["beat", "melody"],
                head => vec![head],
            };
            for n in needs {
                if !names.contains(&n) {
                    names.push(n);
                }
            }
        }
        let heads = names.iter().map(|&h| Ok((h.to_string(), Head::load(&weights, h)?))).collect::<Result<Vec<_>>>()?;
        Ok(Self {
            encoder,
            heads,
            sections,
            keys,
            weights,
            #[cfg(feature = "gpu")]
            gpu: None,
        })
    }

    /// Whether the bar models need the block means of the encoder states.
    fn blocks(&self) -> bool {
        self.sections.is_some() || self.keys.is_some()
    }

    #[cfg(feature = "gpu")]
    pub fn load_gpu(path: &std::path::Path, outputs: &[&str]) -> Result<Self> {
        let mut model = Self::load(path, outputs)?;
        let mixes = model.heads.iter().map(|(_, h)| h.mixes.len()).sum::<usize>()
            + if model.blocks() { bars::GROUPS } else { 0 };
        model.gpu = Some(gpu::GpuEncoder::load(&model.weights, mixes)?);
        model.weights.release_pages();
        Ok(model)
    }

    pub fn set_adapter(&mut self, adapter: Option<&lora::Adapter>) -> Result<()> {
        #[cfg(feature = "gpu")]
        if let Some(gpu) = &mut self.gpu {
            return gpu.set_adapter(adapter);
        }
        self.encoder.set_adapter(adapter)
    }

    fn encode(&self, window: &[f32], mixes: &mut [&mut Mix]) -> Result<usize> {
        #[cfg(feature = "gpu")]
        if let Some(gpu) = &self.gpu {
            return gpu.forward(window, mixes);
        }
        self.encoder.forward(window, mixes)
    }

    /// The heads' logits for a 24 kHz waveform (`beats.frame_outputs`): per head, its outputs over all windows.
    pub fn frame_outputs(&self, wave: &[f32]) -> Result<Vec<(String, Outputs)>> {
        Ok(self.frames(wave)?.heads)
    }

    /// The heads' logits for a 24 kHz waveform (`beats.frame_outputs`, `sections.HeadSections.frames`): per head,
    /// its outputs over all windows, and the block means for the bar models.  A last window too short for the STFT
    /// (at most `N_FFT / 2` samples) is padded with silence to 1 s and keeps the frames its own samples cover
    /// (`beats.encode_windows`).
    pub fn frames(&self, wave: &[f32]) -> Result<Frames> {
        let mut all: Vec<(String, Outputs)> = self.heads.iter().map(|(n, _)| (n.clone(), Vec::new())).collect();
        let mut states = Vec::new();
        for start in (0..wave.len()).step_by(WINDOW_SAMPLES) {
            let mut window = &wave[start..(start + WINDOW_SAMPLES).min(wave.len())];
            let mut keep = None;
            let padded;
            if window.len() <= N_FFT / 2 {
                keep = Some(window.len() / SAMPLES_PER_FRAME);
                padded = [window, &vec![0.0; SAMPLE_RATE - window.len()]].concat();
                window = &padded;
            }
            let mut mixes: Vec<Mix> =
                self.heads.iter().flat_map(|(_, h)| h.mixes.iter().map(|m| Mix::new(m))).collect();
            if self.blocks() {
                for (lo, hi) in bars::BLOCKS {
                    let w: Vec<f32> = (0..encoder::STATES)
                        .map(|l| if (lo..hi).contains(&l) { 1.0 / (hi - lo) as f32 } else { 0.0 })
                        .collect();
                    mixes.push(Mix::new(&w));
                }
            }
            let frames = self.encode(window, &mut mixes.iter_mut().collect::<Vec<_>>())?;
            let keep = keep.unwrap_or(frames).min(frames);
            let mut at = 0;
            for ((_, head), (_, outputs)) in self.heads.iter().zip(all.iter_mut()) {
                let mixed: Vec<&[f32]> = mixes[at..at + head.mixes.len()].iter().map(|m| m.sum.as_slice()).collect();
                at += head.mixes.len();
                let outs = profile::time("heads", || head.forward(&mixed, frames));
                for (i, (name, width, mut logits)) in outs.into_iter().enumerate() {
                    logits.truncate(keep * width);
                    if outputs.len() <= i {
                        outputs.push((name, width, logits));
                    } else {
                        outputs[i].2.extend(logits);
                    }
                }
            }
            if self.blocks() {
                let groups = &mixes[at..];
                for t in 0..keep {
                    for g in groups {
                        states.extend(
                            g.sum[t * WIDTH..(t + 1) * WIDTH].iter().map(|&v| half::f16::from_f32(v).to_bits()),
                        );
                    }
                }
            }
        }
        Ok(Frames { heads: all, states })
    }
}

pub fn release_memory() {
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    unsafe {
        libc::malloc_trim(0);
    }
}

fn status_mib(status: &str, key: &str) -> Option<f64> {
    let line = status.lines().find(|l| l.starts_with(key))?;
    let kib: f64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kib / 1024.0)
}

/// Peak resident memory of this process in MiB (Linux), if known.
pub fn peak_rss_mib() -> Option<f64> {
    status_mib(&std::fs::read_to_string("/proc/self/status").ok()?, "VmHWM:")
}

/// Samples this process's resident memory (Linux) every few milliseconds until `stop`: the peak of its own
/// allocations (`RssAnon`, what a request costs) and of the mapped weights (`RssFile`, page cache that other
/// processes mapping the same file share and the kernel can reclaim).
pub struct MemoryWatch {
    done: std::sync::Arc<std::sync::atomic::AtomicBool>,
    thread: std::thread::JoinHandle<(f64, f64)>,
}

impl MemoryWatch {
    pub fn start() -> Self {
        let done = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let flag = done.clone();
        let thread = std::thread::spawn(move || {
            let (mut anon, mut file) = (0.0f64, 0.0f64);
            let mut phases: Vec<(&'static str, f64)> = Vec::new();
            loop {
                if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
                    let a = status_mib(&status, "RssAnon:").unwrap_or(0.0);
                    anon = anon.max(a);
                    file = file.max(status_mib(&status, "RssFile:").unwrap_or(0.0));
                    let phase = *profile::PHASE.lock().unwrap();
                    match phases.iter_mut().find(|(p, _)| *p == phase) {
                        Some((_, m)) => *m = m.max(a),
                        None => phases.push((phase, a)),
                    }
                }
                if flag.load(std::sync::atomic::Ordering::Relaxed) {
                    if std::env::var_os("SHEETSAGE_PROFILE").is_some() {
                        for (p, m) in &phases {
                            eprintln!("  peak memory in {p:16} {m:6.0} MiB");
                        }
                    }
                    return (anon, file);
                }
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
        });
        Self { done, thread }
    }

    /// `(peak RssAnon, peak RssFile)` in MiB.
    pub fn stop(self) -> (f64, f64) {
        self.done.store(true, std::sync::atomic::Ordering::Relaxed);
        self.thread.join().unwrap_or((f64::NAN, f64::NAN))
    }
}
