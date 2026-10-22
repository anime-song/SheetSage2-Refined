//! From the heads' logits to the beat and melody JSON that `python -m sheetsage2_refined.beats` / `.melody` write:
//! probabilities as `HeadBeats.probabilities` / `HeadMelody.probabilities`, then `tracker` and `melody`.

use serde_json::{Value, json};

use crate::Outputs;
use crate::melody::{self, KeyProbs, LineOptions, LineProbs};
use crate::tracker::{self, BeatProbs, Options};

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn softmax(x: &[f32]) -> Vec<f32> {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = x.iter().map(|v| (v - max).exp()).collect();
    let sum: f32 = e.iter().sum();
    e.iter().map(|v| v / sum).collect()
}

/// `round(x, 4)` as Python writes it.
fn r4(x: f64) -> f64 {
    format!("{x:.4}").parse().unwrap()
}

/// A head output layer's logits as rows.
fn layer<'a>(outputs: &'a Outputs, name: &str) -> Option<(usize, &'a [f32])> {
    outputs.iter().find(|(n, _, _)| n == name).map(|(_, w, v)| (*w, v.as_slice()))
}

fn rows(width: usize, data: &[f32], from: usize, to: usize) -> impl Iterator<Item = &[f32]> {
    data.chunks(width).map(move |r| &r[from..to])
}

pub fn beat_probs(outputs: &Outputs) -> Result<BeatProbs, String> {
    let (bw, binary) = layer(outputs, "binary").ok_or("the beat head has no binary outputs")?;
    let (_, bar) = layer(outputs, "bar").ok_or("beat tracking needs a bar head")?;
    let soft16 = |from: usize| -> Vec<[f32; 16]> {
        rows(40, bar, from, from + 16).map(|r| softmax(r).try_into().unwrap()).collect()
    };
    Ok(BeatProbs {
        beat: rows(bw, binary, 2, 3).map(|r| sigmoid(r[0])).collect(),
        downbeat: rows(bw, binary, 3, 4).map(|r| sigmoid(r[0])).collect(),
        bar_beats: soft16(0),
        beats_left: soft16(16),
        bar_unit: layer(outputs, "unit")
            .map(|(w, u)| rows(w, u, 0, w).map(|r| softmax(r).try_into().unwrap()).collect()),
        beat_offset: layer(outputs, "offset").map(|(w, o)| rows(w, o, 0, 1).map(|r| r[0]).collect()),
    })
}

pub fn beats_json(outputs: &Outputs, audio: &str, duration: f64, tempo_weight: f64) -> Result<Value, String> {
    let probs = beat_probs(outputs)?;
    let t = tracker::track_beats(&probs, &Options::default(), tempo_weight);
    let bars: Vec<Value> = t
        .bars
        .iter()
        .map(|b| match b.unit {
            Some(u) => json!({"start": r4(b.start), "beats": b.beats, "unit": u, "meter": format!("{}/{u}", b.beats)}),
            None => json!({"start": r4(b.start), "beats": b.beats}),
        })
        .collect();
    Ok(json!({
        "audio": audio,
        "duration": duration,
        "beats": t.beats.iter().map(|&b| r4(b)).collect::<Vec<_>>(),
        "downbeats": t.downbeats.iter().map(|&d| r4(d)).collect::<Vec<_>>(),
        "bars": bars,
        "meters": t.meters,
    }))
}

/// The melody decoding settings stored by `export_cpu` (`melody_tracker.decoding`).
pub fn line_options(decoding: &Value) -> LineOptions {
    let mut o = LineOptions::default();
    let get = |k: &str| decoding.get(k).and_then(Value::as_f64);
    o.onset_weight = get("onset_weight").unwrap_or(o.onset_weight);
    o.start_penalty = get("start_penalty").unwrap_or(o.start_penalty);
    o.interval_weight = get("interval_weight").unwrap_or(o.interval_weight);
    o.octave_penalty = get("octave_penalty").unwrap_or(o.octave_penalty);
    o.end_weight = get("end_weight").unwrap_or(o.end_weight);
    o
}

pub fn melody_json(outputs: &Outputs, audio: &str, duration: f64, decoding: &Value) -> Result<Value, String> {
    let notes = if let Some((w, line)) = layer(outputs, "line") {
        let soft = |from: usize, to: usize| -> Vec<Vec<f32>> { rows(w, line, from, to).map(softmax).collect() };
        let probs = LineProbs {
            pitch: soft(0, 89),
            note_steps: Some(soft(89, 105)),
            steps_left: Some(soft(105, 121)),
            onset_pos: Some(soft(121, 125)),
            onset: rows(w, line, w - 3, w - 2).map(|r| sigmoid(r[0])).collect(),
            offset: rows(w, line, w - 2, w - 1).map(|r| r[0]).collect(),
            vocal: rows(w, line, w - 1, w).map(|r| sigmoid(r[0])).collect(),
        };
        crate::profile::time("melody viterbi", || melody::line_notes(&probs, &line_options(decoding)))
    } else if let Some((w, m)) = layer(outputs, "melody") {
        let keys = melody::KEYS;
        let sig = |from: usize, to: usize| -> Vec<Vec<f32>> {
            rows(w, m, from, to).map(|r| r.iter().map(|&v| sigmoid(v)).collect()).collect()
        };
        let probs = KeyProbs {
            sounding: sig(0, keys),
            onset: sig(keys, 2 * keys),
            offset: rows(w, m, w - 2, w - 1).map(|r| r[0]).collect(),
            vocal: rows(w, m, w - 1, w).map(|r| sigmoid(r[0])).collect(),
        };
        let get = |k: &str, d: f64| decoding.get(k).and_then(Value::as_f64).unwrap_or(d) as f32;
        melody::key_notes(&probs, get("onset", 0.5), get("frame", 0.5))
    } else {
        return Err("the melody head has neither line nor melody outputs".into());
    };
    let notes: Vec<Value> = notes
        .iter()
        .map(|n| {
            let mut v = json!({"start": r4(n.start), "end": r4(n.end), "pitch": n.pitch, "track": n.track});
            if let (Some(s), Some(p)) = (n.steps, n.pos) {
                v["steps"] = json!(s);
                v["pos"] = json!(p);
            }
            v
        })
        .collect();
    Ok(json!({"audio": audio, "duration": duration, "notes": notes}))
}
