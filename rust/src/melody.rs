//! Notes from a melody head's probabilities: `sheetsage2_refined/melody.py`'s `line_notes` (a `line` head: Viterbi
//! over the rest and the 88 keys) and `notes` (an 88-key head: onset peaks) in Rust.

use crate::tracker::HEAD_RATE;

pub const LOWEST: usize = 21; // MIDI of the first key
pub const KEYS: usize = 88;

/// A `line` head's 25 Hz probabilities (`HeadMelody.probabilities`): `pitch` `[frames][89]` (rest, then keys),
/// the onset bit, its sub-frame offset (frames), the vocal call and the note facts.
pub struct LineProbs {
    pub pitch: Vec<Vec<f32>>,
    pub onset: Vec<f32>,
    pub offset: Vec<f32>,
    pub vocal: Vec<f32>,
    pub note_steps: Option<Vec<Vec<f32>>>,
    pub steps_left: Option<Vec<Vec<f32>>>,
    pub onset_pos: Option<Vec<Vec<f32>>>,
}

/// An 88-key head's probabilities: per key sounding and starting, the onset's offset and the vocal call.
pub struct KeyProbs {
    pub sounding: Vec<Vec<f32>>,
    pub onset: Vec<Vec<f32>>,
    pub offset: Vec<f32>,
    pub vocal: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct LineOptions {
    pub onset_weight: f64,
    pub start_penalty: f64,
    pub interval_weight: f64,
    pub octave_penalty: f64,
    pub end_weight: f64,
}

impl Default for LineOptions {
    fn default() -> Self {
        Self { onset_weight: 1.0, start_penalty: 0.0, interval_weight: 0.0, octave_penalty: 0.0, end_weight: 0.0 }
    }
}

pub struct Note {
    pub start: f64,
    pub end: f64,
    pub pitch: usize,
    pub track: u8,
    pub steps: Option<usize>,
    pub pos: Option<usize>,
}

fn argmax(v: impl IntoIterator<Item = f64>) -> usize {
    let (mut best, mut arg) = (f64::NEG_INFINITY, 0);
    for (i, x) in v.into_iter().enumerate() {
        if x > best || (i == 0 && x.is_nan()) {
            (best, arg) = (x, i);
        }
    }
    arg
}

fn mean_rows(rows: &[Vec<f32>]) -> Vec<f64> {
    let mut m = vec![0.0f64; rows[0].len()];
    for r in rows {
        m.iter_mut().zip(r).for_each(|(m, &v)| *m += v as f64);
    }
    m.iter_mut().for_each(|v| *v /= rows.len() as f64);
    m
}

/// `line_notes`: the likeliest path through the rest and the keys, then its notes.
pub fn line_notes(p: &LineProbs, o: &LineOptions) -> Vec<Note> {
    let count = p.pitch.len();
    if count == 0 {
        return Vec::new();
    }
    let states = p.pitch[0].len();
    let keys = states - 1;
    // per-frame scores, computed in float32 as numpy does on the head's float32 probabilities
    let (w, pen) = (o.onset_weight as f32, o.start_penalty as f32);
    let pitch: Vec<Vec<f32>> = p.pitch.iter().map(|r| r.iter().map(|&v| v.clamp(1e-9, 1.0).ln()).collect()).collect();
    let on: Vec<f32> = p.onset.iter().map(|&v| v.clamp(1e-6, (1.0f64 - 1e-6) as f32)).collect();
    let begin: Vec<f32> = on.iter().map(|&v| w * v.ln() - pen).collect();
    let keep: Vec<f32> = on.iter().map(|&v| w * (-v).ln_1p()).collect();
    let ending: Vec<f64> = match (&p.steps_left, o.end_weight != 0.0) {
        (Some(left), true) => left.iter().map(|r| o.end_weight * (r[0].clamp(1e-4, 1.0).ln() as f64)).collect(),
        _ => vec![0.0; count],
    };
    let mv = |from: usize, to: usize| -> f64 {
        let leap = from.abs_diff(to) as f64;
        let octave = leap > 0.0 && (leap as usize).is_multiple_of(12);
        -o.interval_weight * leap / 12.0 - o.octave_penalty * if octave { 1.0 } else { 0.0 }
    };
    let width = 1 + 2 * keys;
    let mut came = vec![0i16; count * width];
    let mut rest = (pitch[0][0] + keep[0]) as f64;
    let mut first: Vec<f64> = (0..keys).map(|k| (pitch[0][k + 1] + begin[0]) as f64).collect();
    let mut held = vec![f64::NEG_INFINITY; keys];
    let mut next_first = vec![0.0f64; keys];
    let mut next_held = vec![0.0f64; keys];
    for t in 1..count {
        let end = ending[t - 1];
        let first_arg = argmax(first.iter().copied());
        let held_arg = argmax(held.iter().copied());
        let options = [rest, first[first_arg] + end, held[held_arg] + end];
        let pick = argmax(options);
        let row = &mut came[t * width..(t + 1) * width];
        row[0] = [0, 1 + first_arg, 1 + keys + held_arg][pick] as i16;
        for dst in 0..keys {
            // the best held key to leap from (argmax over sources, first on ties)
            let (mut best, mut source) = (f64::NEG_INFINITY, 0);
            for (src, &h) in held.iter().enumerate() {
                let v = h + end + mv(src, dst);
                if v > best || (src == 0 && v.is_nan()) {
                    (best, source) = (v, src);
                }
            }
            row[1 + dst] = if best > rest { (1 + keys + source) as i16 } else { 0 };
            row[1 + keys + dst] = if first[dst] >= held[dst] { (1 + dst) as i16 } else { (1 + keys + dst) as i16 };
            let pt = pitch[t][dst + 1] as f64;
            next_first[dst] = best.max(rest) + begin[t] as f64 + pt;
            next_held[dst] = first[dst].max(held[dst]) + keep[t] as f64 + pt;
        }
        rest = options[pick] + keep[t] as f64 + pitch[t][0] as f64;
        std::mem::swap(&mut first, &mut next_first);
        std::mem::swap(&mut held, &mut next_held);
    }
    let finals = [rest, first[argmax(first.iter().copied())], held[argmax(held.iter().copied())]];
    let mut state = [0, 1 + argmax(first.iter().copied()), 1 + keys + argmax(held.iter().copied())][argmax(finals)];
    let mut path = vec![0usize; count];
    let mut starts = vec![false; count];
    for t in (0..count).rev() {
        path[t] = if state == 0 { 0 } else { 1 + (state - 1) % keys };
        starts[t] = (1..=keys).contains(&state);
        state = came[t * width + state] as usize;
    }
    let prev = |t: usize| if t == 0 { path[count - 1] } else { path[t - 1] }; // numpy's path[-1] at t = 0
    let onsets: Vec<usize> = (0..count).filter(|&t| path[t] > 0 && (starts[t] || path[t] != prev(t))).collect();
    let times: Vec<f64> = onsets.iter().map(|&t| (t as f64 + 0.5 + p.offset[t] as f64) / HEAD_RATE).collect();
    let mut out = Vec::new();
    for (i, &t) in onsets.iter().enumerate() {
        let mut end = t + 1;
        while end < count && path[end] == path[t] && !starts[end] {
            end += 1;
        }
        let finish = if i + 1 < onsets.len() && onsets[i + 1] == end { times[i + 1] } else { end as f64 / HEAD_RATE };
        let vocal = p.vocal[t..end].iter().map(|&v| v as f64).sum::<f64>() / (end - t) as f64;
        let (steps, pos) = match (&p.note_steps, &p.onset_pos) {
            (Some(steps), Some(pos)) => {
                (Some(argmax(mean_rows(&steps[t..end])) + 1), Some(argmax(mean_rows(&pos[t..end]))))
            }
            _ => (None, None),
        };
        out.push(Note {
            start: times[i],
            end: finish.max(times[i] + 0.01),
            pitch: LOWEST + path[t] - 1,
            track: if vocal >= 0.5 { 0 } else { 1 },
            steps,
            pos,
        });
    }
    out
}

/// `notes`: an 88-key head's onset peaks, each held while its key sounds.
pub fn key_notes(p: &KeyProbs, onset: f32, frame: f32) -> Vec<Note> {
    let count = p.onset.len();
    let strength: Vec<f32> = p.onset.iter().map(|r| r.iter().cloned().fold(f32::NEG_INFINITY, f32::max)).collect();
    let peaks: Vec<usize> = (0..count)
        .filter(|&t| {
            strength[t] >= onset
                && (t == 0 || strength[t] > strength[t - 1])
                && (t + 1 == count || strength[t] >= strength[t + 1])
        })
        .collect();
    let starts: Vec<f64> = peaks.iter().map(|&t| (t as f64 + 0.5 + p.offset[t] as f64) / HEAD_RATE).collect();
    let mut out = Vec::new();
    for (i, &t) in peaks.iter().enumerate() {
        let stop = peaks.get(i + 1).copied().unwrap_or(count);
        let rows = &p.sounding[t..(t + 2).min(count)];
        let key = argmax(
            (0..KEYS).map(|k| p.onset[t][k] as f64 + rows.iter().map(|r| r[k] as f64).sum::<f64>() / rows.len() as f64),
        );
        let mut end = t + 1;
        while end < stop && p.sounding[end][key] >= frame {
            end += 1;
        }
        let finish = if end == stop && i + 1 < peaks.len() {
            (end as f64 / HEAD_RATE).min(starts[i + 1])
        } else {
            end as f64 / HEAD_RATE
        };
        let vocal = p.vocal[t..end].iter().map(|&v| v as f64).sum::<f64>() / (end - t) as f64;
        out.push(Note {
            start: starts[i],
            end: finish.max(starts[i] + 0.01),
            pitch: LOWEST + key,
            track: if vocal >= 0.5 { 0 } else { 1 },
            steps: None,
            pos: None,
        });
    }
    out
}
