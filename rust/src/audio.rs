//! Audio files as a mono 24 kHz waveform: decoded by Symphonia (WAV, FLAC, MP3, Ogg Vorbis), channels averaged and
//! resampled as `torchaudio.functional.resample` does (windowed sinc, Hann window, 6 zero crossings, rolloff 0.99), so
//! the waveform matches `transcribe.load_waveform`'s.

use std::fs::File;
use std::path::Path;

use symphonia::core::codecs::audio::AudioDecoderOptions;
use symphonia::core::errors::Error;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::{FormatOptions, TrackType};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;

use rayon::prelude::*;

use crate::Result;

/// Decode `path` to mono float32 at its own rate: `(samples, rate)`.
pub fn decode(path: &Path) -> Result<(Vec<f32>, u32)> {
    let file = Box::new(File::open(path).map_err(|e| format!("{}: {e}", path.display()))?);
    let stream = MediaSourceStream::new(file, Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }
    let mut format =
        symphonia::default::get_probe().probe(&hint, stream, FormatOptions::default(), MetadataOptions::default())?;
    let track = format.default_track(TrackType::Audio).ok_or("no audio track")?;
    let params = track.codec_params.as_ref().and_then(|p| p.audio()).ok_or("no audio codec parameters")?;
    let mut decoder = symphonia::default::get_codecs().make_audio_decoder(params, &AudioDecoderOptions::default())?;
    let track_id = track.id;
    let (mut mono, mut rate, mut interleaved) = (Vec::new(), 0u32, Vec::<f32>::new());
    while let Some(packet) = format.next_packet()? {
        if packet.track_id != track_id {
            continue;
        }
        match decoder.decode(&packet) {
            Ok(buffer) => {
                rate = buffer.spec().rate();
                let channels = buffer.spec().channels().count().max(1);
                interleaved.resize(buffer.samples_interleaved(), 0.0);
                buffer.copy_to_slice_interleaved(&mut interleaved);
                mono.extend(interleaved.chunks(channels).map(|f| f.iter().sum::<f32>() / channels as f32));
            }
            Err(Error::DecodeError(_)) => continue,
            Err(e) => return Err(e.into()),
        }
    }
    if rate == 0 {
        return Err(format!("{}: no audio decoded", path.display()).into());
    }
    Ok((mono, rate))
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// `torchaudio.functional.resample(wave, from, to)` with its defaults.
pub fn resample(wave: &[f32], from: usize, to: usize) -> Vec<f32> {
    if from == to {
        return wave.to_vec();
    }
    let g = gcd(from, to);
    let (orig, new) = (from / g, to / g);
    let zeros = 6.0f64;
    let base = orig.min(new) as f64 * 0.99;
    let width = (zeros * orig as f64 / base).ceil() as usize;
    let taps = 2 * width + orig;
    // kernels[j][k]: output phase j, input offset k - width
    let mut kernels = vec![0.0f32; new * taps];
    for j in 0..new {
        for k in 0..taps {
            let idx = (k as f64 - width as f64) / orig as f64;
            let mut t = (-(j as f64) / new as f64 + idx) * base;
            t = t.clamp(-zeros, zeros);
            let window = (t * std::f64::consts::PI / zeros / 2.0).cos().powi(2);
            let t = t * std::f64::consts::PI;
            let sinc = if t == 0.0 { 1.0 } else { t.sin() / t };
            kernels[j * taps + k] = (sinc * window * base / orig as f64) as f32;
        }
    }
    let length = wave.len();
    let mut padded = vec![0.0f32; length + 2 * width + orig];
    padded[width..width + length].copy_from_slice(wave);
    let blocks = (padded.len() - taps) / orig + 1;
    let target = (new as u128 * length as u128).div_ceil(orig as u128) as usize;
    let mut out = vec![0.0f32; blocks * new];
    out.par_chunks_mut(new).enumerate().for_each(|(b, out)| {
        let input = &padded[b * orig..b * orig + taps];
        for (o, kernel) in out.iter_mut().zip(kernels.chunks_exact(taps)) {
            *o = kernel.iter().zip(input).map(|(w, x)| w * x).sum();
        }
    });
    out.truncate(target);
    out
}

/// A file as the encoder's input: mono at 24 kHz.
pub fn load(path: &Path, rate: usize) -> Result<Vec<f32>> {
    let (wave, from) = decode(path)?;
    Ok(resample(&wave, from as usize, rate))
}
