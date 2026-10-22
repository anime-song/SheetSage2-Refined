//! sheetsage2-cpu --weights cpu.safetensors [--threads N] [--out-dir DIR] [--heads beat,melody] [--logits]
//!                [--tempo-weight W] [--hold P] AUDIO...
//!
//! Writes `<out-dir>/<stem>.beats.json` and `<stem>.melody.json` as `python -m sheetsage2_refined.beats` / `.melody`
//! do, and with `--logits` also `<stem>.heads.safetensors` with each head's logits (`<head>.<output>`,
//! `[frames, width]`).  AUDIO may be a WAV / FLAC / MP3 / Ogg file, or a `.f32` file of raw little-endian float32
//! samples, mono at 24 kHz.

use std::path::{Path, PathBuf};
use std::time::Instant;

use safetensors::tensor::{Dtype, TensorView};
use sheetsage2_cpu::{MemoryWatch, Model, Outputs, Result, audio, encoder::SAMPLE_RATE, post};

fn usage() -> ! {
    eprintln!(
        "usage: sheetsage2-cpu --weights FILE [--threads N] [--out-dir DIR] [--heads beat,melody] [--logits] \
         [--tempo-weight W] [--hold P] AUDIO..."
    );
    std::process::exit(2)
}

fn load_wave(path: &Path) -> Result<Vec<f32>> {
    if path.extension().is_some_and(|e| e == "f32") {
        let bytes = std::fs::read(path)?;
        return Ok(bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect());
    }
    audio::load(path, SAMPLE_RATE)
}

/// A `--logits` file back as the heads' outputs, and its duration.
fn read_logits(path: &Path) -> Result<(Vec<(String, Outputs)>, f64)> {
    let bytes = std::fs::read(path)?;
    let (_, meta) = safetensors::SafeTensors::read_metadata(&bytes)?;
    let duration: f64 = meta.metadata().as_ref().and_then(|m| m.get("duration")).ok_or("no duration")?.parse()?;
    let tensors = safetensors::SafeTensors::deserialize(&bytes)?;
    let mut names: Vec<String> = tensors.names().into_iter().map(str::to_string).collect();
    names.sort();
    let mut heads: Vec<(String, Outputs)> = Vec::new();
    for key in names {
        let view = tensors.tensor(&key)?;
        let (head, layer) = key.split_once('.').ok_or("bad tensor name")?;
        let data: Vec<f32> =
            view.data().chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
        let width = view.shape()[1];
        match heads.iter_mut().find(|(h, _)| h == head) {
            Some((_, outs)) => outs.push((layer.to_string(), width, data)),
            None => heads.push((head.to_string(), vec![(layer.to_string(), width, data)])),
        }
    }
    Ok((heads, duration))
}

fn write_logits(path: &Path, outputs: &[(String, Outputs)], duration: f64) -> Result<()> {
    let mut tensors = Vec::new();
    for (head, outs) in outputs {
        for (name, width, logits) in outs {
            let bytes: Vec<u8> = logits.iter().flat_map(|v| v.to_le_bytes()).collect();
            tensors.push((format!("{head}.{name}"), vec![logits.len() / width, *width], bytes));
        }
    }
    let views: Vec<(String, TensorView)> = tensors
        .iter()
        .map(|(n, s, b)| (n.clone(), TensorView::new(Dtype::F32, s.clone(), b).expect("tensor")))
        .collect();
    let meta = std::collections::HashMap::from([("duration".to_string(), duration.to_string())]);
    safetensors::serialize_to_file(views, Some(meta), path)?;
    Ok(())
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let (mut weights, mut threads, mut out_dir) = (None, None, PathBuf::from("outputs/cpu"));
    let (mut heads, mut logits, mut tempo_weight, mut inputs) = ("beat,melody".to_string(), false, 4.0, Vec::new());
    let mut hold = 0.0f64; // the beat tracker's fermata holds (`beats --hold`)
    let mut from_logits = false; // inputs are `--logits` files: run the trackers only (for testing them)
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--weights" => weights = Some(PathBuf::from(args.next().unwrap_or_else(|| usage()))),
            "--threads" => threads = Some(args.next().and_then(|v| v.parse::<usize>().ok()).unwrap_or_else(|| usage())),
            "--out-dir" => out_dir = PathBuf::from(args.next().unwrap_or_else(|| usage())),
            "--heads" => heads = args.next().unwrap_or_else(|| usage()),
            "--logits" => logits = true,
            "--from-logits" => from_logits = true,
            "--hold" => hold = args.next().and_then(|v| v.parse().ok()).unwrap_or_else(|| usage()),
            "--tempo-weight" => tempo_weight = args.next().and_then(|v| v.parse().ok()).unwrap_or_else(|| usage()),
            "-h" | "--help" => usage(),
            _ => inputs.push(PathBuf::from(arg)),
        }
    }
    let Some(weights) = weights else { usage() };
    if inputs.is_empty() {
        usage();
    }
    sheetsage2_cpu::simd::init_threads(threads)?;
    let started = Instant::now();
    let heads: Vec<&str> = heads.split(',').filter(|h| !h.is_empty()).collect();
    let model = Model::load(&weights, &heads)?;
    let decoding: serde_json::Value =
        serde_json::from_str(model.weights.metadata.get("melody_decoding").map_or("{}", |s| s))?;
    eprintln!(
        "loaded {} ({}) in {:.2} s, AVX2: {}",
        weights.display(),
        model.weights.metadata.get("dtype").map_or("?", |s| s),
        started.elapsed().as_secs_f64(),
        sheetsage2_cpu::simd::avx2()
    );
    let tracker_options = sheetsage2_cpu::tracker::Options { p_hold: hold, ..Default::default() };
    std::fs::create_dir_all(&out_dir)?;
    for input in &inputs {
        let watch = MemoryWatch::start();
        let t0 = Instant::now();
        let (outputs, duration, t1) = if from_logits {
            let (outputs, duration) = read_logits(input)?;
            (outputs, duration, Instant::now())
        } else {
            let wave = load_wave(input)?;
            let t1 = Instant::now();
            (model.frame_outputs(&wave)?, wave.len() as f64 / SAMPLE_RATE as f64, t1)
        };
        let t2 = Instant::now();
        sheetsage2_cpu::profile::report();
        let stem = input.file_stem().map_or("audio".into(), |s| s.to_string_lossy().to_string());
        let stem = stem.strip_suffix(".heads").map_or(stem.clone(), str::to_string);
        let name = input.to_string_lossy().to_string();
        let mut written = Vec::new();
        for (head, outs) in &outputs {
            let (suffix, result) = match head.as_str() {
                "beat" => ("beats", post::beats_json(outs, &name, duration, &tracker_options, tempo_weight)?),
                "melody" => ("melody", post::melody_json(outs, &name, duration, &decoding)?),
                _ => continue,
            };
            let path = out_dir.join(format!("{stem}.{suffix}.json"));
            std::fs::write(&path, serde_json::to_string(&result)?)?;
            written.push(path);
        }
        if logits && !from_logits {
            let path = out_dir.join(format!("{stem}.heads.safetensors"));
            write_logits(&path, &outputs, duration)?;
            written.push(path);
        }
        sheetsage2_cpu::profile::report();
        let t3 = Instant::now();
        let (anon, file) = watch.stop();
        drop(outputs);
        sheetsage2_cpu::release_memory();
        if std::env::var_os("SHEETSAGE_PROFILE").is_some() {
            let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
            eprintln!("  after the request: {}", status.lines().find(|l| l.starts_with("RssAnon")).unwrap_or(""));
        }
        let encode = (t2 - t1).as_secs_f64();
        eprintln!(
            "{}: {duration:.1} s of audio, decode {:.2} s, encoder + heads {encode:.2} s ({:.1}x real time), trackers {:.2} s, \
             memory {anon:.0} MiB + weights {file:.0} MiB mapped -> {}",
            input.display(),
            (t1 - t0).as_secs_f64(),
            duration / encode,
            (t3 - t2).as_secs_f64(),
            written.iter().map(|p| p.display().to_string()).collect::<Vec<_>>().join(", ")
        );
    }
    Ok(())
}
