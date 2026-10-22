//! sheetsage2-cpu --weights cpu.safetensors [--threads N] [--out-dir DIR] [--heads beat,melody] AUDIO...
//!
//! Writes `<out-dir>/<stem>.heads.safetensors` with each head's logits (`<head>.<output>`, `[frames, width]`) for
//! `python -m sheetsage2_refined.cpu` to turn into beats and notes.  AUDIO may be a WAV / FLAC / MP3 / Ogg file, or
//! a `.f32` file of raw little-endian float32 samples, mono at 24 kHz.

use std::path::{Path, PathBuf};
use std::time::Instant;

use safetensors::tensor::{Dtype, TensorView};
use sheetsage2_cpu::{Model, Result, audio, encoder::SAMPLE_RATE, peak_rss_mib};

fn usage() -> ! {
    eprintln!("usage: sheetsage2-cpu --weights FILE [--threads N] [--out-dir DIR] [--heads beat,melody] AUDIO...");
    std::process::exit(2)
}

fn load_wave(path: &Path) -> Result<Vec<f32>> {
    if path.extension().is_some_and(|e| e == "f32") {
        let bytes = std::fs::read(path)?;
        return Ok(bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect());
    }
    audio::load(path, SAMPLE_RATE)
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let (mut weights, mut threads, mut out_dir, mut heads, mut inputs) =
        (None, None, PathBuf::from("outputs/cpu"), "beat,melody".to_string(), Vec::new());
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--weights" => weights = Some(PathBuf::from(args.next().unwrap_or_else(|| usage()))),
            "--threads" => threads = Some(args.next().and_then(|v| v.parse::<usize>().ok()).unwrap_or_else(|| usage())),
            "--out-dir" => out_dir = PathBuf::from(args.next().unwrap_or_else(|| usage())),
            "--heads" => heads = args.next().unwrap_or_else(|| usage()),
            "-h" | "--help" => usage(),
            _ => inputs.push(PathBuf::from(arg)),
        }
    }
    let Some(weights) = weights else { usage() };
    if inputs.is_empty() {
        usage();
    }
    if let Some(n) = threads {
        rayon::ThreadPoolBuilder::new().num_threads(n).build_global()?;
    }
    let started = Instant::now();
    let heads: Vec<&str> = heads.split(',').filter(|h| !h.is_empty()).collect();
    let model = Model::load(&weights, &heads)?;
    eprintln!(
        "loaded {} ({}) in {:.2} s, AVX2: {}",
        weights.display(),
        model.weights.metadata.get("dtype").map_or("?", |s| s),
        started.elapsed().as_secs_f64(),
        sheetsage2_cpu::simd::avx2()
    );
    std::fs::create_dir_all(&out_dir)?;
    for input in &inputs {
        let t0 = Instant::now();
        let wave = load_wave(input)?;
        let t1 = Instant::now();
        let outputs = model.frame_outputs(&wave)?;
        let t2 = Instant::now();
        sheetsage2_cpu::profile::report();
        let mut tensors = Vec::new();
        for (head, outs) in &outputs {
            for (name, width, logits) in outs {
                let bytes: Vec<u8> = logits.iter().flat_map(|v| v.to_le_bytes()).collect();
                tensors.push((format!("{head}.{name}"), vec![logits.len() / width, *width], bytes));
            }
        }
        let views: Vec<(String, TensorView)> = tensors
            .iter()
            .map(|(n, s, b)| (n.clone(), TensorView::new(Dtype::F32, s.clone(), b).expect("tensor")))
            .collect();
        let stem = input.file_stem().map_or("audio".into(), |s| s.to_string_lossy().to_string());
        let path = out_dir.join(format!("{stem}.heads.safetensors"));
        let duration = wave.len() as f64 / SAMPLE_RATE as f64;
        let meta = std::collections::HashMap::from([("duration".to_string(), duration.to_string())]);
        safetensors::serialize_to_file(views, Some(meta), &path)?;
        let encode = (t2 - t1).as_secs_f64();
        eprintln!(
            "{}: {duration:.1} s of audio, decode {:.2} s, encoder + heads {encode:.2} s ({:.1}x real time), peak RSS {:.0} MiB -> {}",
            input.display(),
            (t1 - t0).as_secs_f64(),
            duration / encode,
            peak_rss_mib().unwrap_or(f64::NAN),
            path.display()
        );
    }
    Ok(())
}
