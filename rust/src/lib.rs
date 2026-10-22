//! SheetSage2-Refined's beat and melody heads on the CPU: the MERT-v2 encoder and the frame heads, from weights
//! exported by `python -m sheetsage2_refined.export_cpu`.

#![allow(unsafe_op_in_unsafe_fn)]

pub mod audio;
pub mod encoder;
pub mod gemm;
#[cfg(feature = "gpu")]
pub mod gpu;
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
    /// The encoder on a GPU instead (`load_gpu`).
    #[cfg(feature = "gpu")]
    pub gpu: Option<gpu::GpuEncoder>,
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
        Ok(Self {
            encoder,
            heads,
            weights,
            #[cfg(feature = "gpu")]
            gpu: None,
        })
    }

    /// As `load`, with the encoder on the GPU.
    #[cfg(feature = "gpu")]
    pub fn load_gpu(path: &std::path::Path, heads: &[&str]) -> Result<Self> {
        let mut model = Self::load(path, heads)?;
        model.gpu = Some(gpu::GpuEncoder::load(&model.weights, heads.len())?);
        model.weights.release_pages();
        Ok(model)
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
        let mut all: Vec<(String, Outputs)> = self.heads.iter().map(|(n, _)| (n.clone(), Vec::new())).collect();
        for start in (0..wave.len()).step_by(WINDOW_SAMPLES) {
            let window = &wave[start..(start + WINDOW_SAMPLES).min(wave.len())];
            let mut mixes: Vec<Mix> = self.heads.iter().map(|(_, h)| Mix::new(h.mix)).collect();
            let frames = self.encode(window, &mut mixes.iter_mut().collect::<Vec<_>>())?;
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

/// Give the memory a request freed back to the system (glibc keeps freed blocks in its per-thread arenas, so a
/// long-running process would otherwise sit at its peak between requests).
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
