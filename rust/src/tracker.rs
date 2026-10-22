//! Beats, bar heads and meters from a bar head's probabilities: `sheetsage2_refined/bar_tracker.py` in Rust (the
//! same HMM over (beats in the bar, beat of the bar, beat interval, position in the beat) at 100 fps, the same
//! sub-frame offsets and the same choice of denominators), so the engine needs no Python.
#![allow(clippy::needless_range_loop)] // the Viterbis index several parallel arrays, as the Python they mirror

pub const FPS: f64 = 100.0;
pub const HEAD_RATE: f64 = 25.0;
const HALF: usize = 4;
const BAND: f64 = 200.0;
pub const UNITS: [u32; 4] = [2, 4, 8, 16];

/// `track`'s settings (`bar_tracker.track`'s defaults).
#[derive(Clone, Debug)]
pub struct Options {
    pub lam: f64,
    pub switch: f64,
    pub meter_change: f64,
    pub w_bar: f64,
    pub w_down: f64,
    pub min_bpm: f64,
    pub max_bpm: f64,
    pub n_tau: usize,
    pub metre_min: f64,
    pub local_min: f64,
    pub local_seconds: f64,
    pub p_hold: f64,
    pub q_hold: f64,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            lam: 100.0,
            switch: 0.02,
            meter_change: 0.02,
            w_bar: 1.0,
            w_down: 1.0,
            min_bpm: 50.0,
            max_bpm: 440.0,
            n_tau: 60,
            metre_min: 0.04,
            local_min: 0.5,
            local_seconds: 2.0,
            p_hold: 0.0,
            q_hold: 0.98,
        }
    }
}

/// A bar head's 25 Hz probabilities (`beats.HeadBeats.probabilities`): `bar_beats`, `beats_left` `[frames][16]`,
/// `bar_unit` `[frames][4]`.
pub struct BeatProbs {
    pub beat: Vec<f32>,
    pub downbeat: Vec<f32>,
    pub bar_beats: Vec<[f32; 16]>,
    pub beats_left: Vec<[f32; 16]>,
    pub bar_unit: Option<Vec<[f32; 4]>>,
    pub beat_offset: Option<Vec<f32>>,
}

pub struct Bar {
    pub start: f64,
    pub beats: usize,
    pub unit: Option<u32>,
}

pub struct Tracked {
    pub beats: Vec<f64>,
    pub downbeats: Vec<f64>,
    pub bars: Vec<Bar>,
    pub meters: Vec<usize>,
}

/// numpy's `logaddexp`.
pub fn logaddexp(x: f64, y: f64) -> f64 {
    if x == y {
        x + std::f64::consts::LN_2
    } else {
        let d = x - y;
        if d > 0.0 {
            x + (-d).exp().ln_1p()
        } else if d <= 0.0 {
            y + d.exp().ln_1p()
        } else {
            d
        }
    }
}

/// `np.interp(x, xp, fp)` for increasing `xp`.
fn interp(x: &[f64], xp: &[f64], fp: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(x.len());
    let mut j = 0;
    for &v in x {
        if v <= xp[0] {
            out.push(fp[0]);
            continue;
        }
        if v >= xp[xp.len() - 1] {
            out.push(fp[fp.len() - 1]);
            continue;
        }
        while xp[j + 1] < v {
            j += 1;
        }
        if v == xp[j + 1] {
            out.push(fp[j + 1]);
            continue;
        }
        let slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]);
        out.push(slope * (v - xp[j]) + fp[j]);
    }
    out
}

/// Beats-per-bar the tracker considers (`candidate_metres`), from `bar_beats` at 100 fps.
fn candidate_metres(bar_beats: &[[f32; 16]], metre_min: f64, local_min: f64, local_frames: usize) -> Vec<usize> {
    let n = bar_beats.len();
    let mut mean = [0.0f64; 16];
    for row in bar_beats {
        for (m, &v) in mean.iter_mut().zip(row) {
            *m += v as f64;
        }
    }
    mean.iter_mut().for_each(|m| *m /= n as f64);
    let mut metres: Vec<usize> = (1..16).filter(|&b| mean[b] >= metre_min).map(|b| b + 1).collect();
    let favourite = (1..16).fold(1, |best, b| if mean[b] > mean[best] { b } else { best });
    metres.push(favourite + 1);
    if local_min > 0.0 && n > local_frames {
        let mut c = vec![[0.0f64; 16]; n + 1];
        for (i, row) in bar_beats.iter().enumerate() {
            for b in 0..16 {
                c[i + 1][b] = c[i][b] + row[b] as f64;
            }
        }
        let mut peak = [f64::NEG_INFINITY; 16];
        for i in 0..=n - local_frames {
            for b in 0..16 {
                peak[b] = peak[b].max((c[i + local_frames][b] - c[i][b]) / local_frames as f64);
            }
        }
        metres.extend((1..16).filter(|&b| peak[b] >= local_min).map(|b| b + 1));
    }
    metres.sort_unstable();
    metres.dedup();
    metres
}

/// numpy's `geomspace(start, stop, num)`.
fn geomspace(start: f64, stop: f64, num: usize) -> Vec<f64> {
    let (a, b) = (start.log10(), stop.log10());
    let step = (b - a) / (num - 1) as f64;
    let mut out: Vec<f64> = (0..num).map(|i| 10f64.powf(i as f64 * step + a)).collect();
    out[0] = start;
    out[num - 1] = stop;
    out
}

/// numpy's `round` (half to even) to an integer.
fn round_even(x: f64) -> i64 {
    let r = x.round();
    if (x - x.trunc()).abs() == 0.5 { (2.0 * (x / 2.0).round()) as i64 } else { r as i64 }
}

fn norm_rows(m: &mut [Vec<f64>]) {
    for row in m {
        let total = row.iter().skip(1).fold(row[0], |acc, &v| logaddexp(acc, v));
        row.iter_mut().for_each(|v| *v -= total);
    }
}

/// Interval transitions as a dense `[from][to]` matrix (rows padded to `stride`, a multiple of 4), `-inf` outside
/// `BAND` below each target's best source, and each target's first source within the band (the choice when every
/// source scores `-inf`).
struct Band {
    val: Vec<f64>,
    first: Vec<f64>,
    stride: usize,
}

fn band(tr: &[Vec<f64>]) -> Band {
    let nt = tr.len();
    let stride = nt.div_ceil(4) * 4;
    let mut val = vec![f64::NEG_INFINITY; nt * stride];
    let mut first = vec![0.0; stride];
    for j in 0..nt {
        let best = (0..nt).map(|i| tr[i][j]).fold(f64::NEG_INFINITY, f64::max);
        first[j] = (0..nt).find(|&i| tr[i][j] >= best - BAND).unwrap_or(0) as f64;
        for i in 0..nt {
            if tr[i][j] >= best - BAND {
                val[i * stride + j] = tr[i][j];
            }
        }
    }
    Band { val, first, stride }
}

/// `best[j] = max_i end[i] + val[i][j]` and its argmax (the first on ties; as f64) for every target `j` at once.
/// `best` and `arg` are `stride` long.
fn max_plus(band: &Band, end: &[f64], best: &mut [f64], arg: &mut [f64]) {
    #[cfg(target_arch = "x86_64")]
    if crate::simd::avx2() {
        unsafe { max_plus_avx2(band, end, best, arg) };
        return;
    }
    let stride = band.stride;
    best.fill(f64::NEG_INFINITY);
    arg.copy_from_slice(&band.first);
    for (i, &e) in end.iter().enumerate() {
        for ((b, a), &v) in best.iter_mut().zip(arg.iter_mut()).zip(&band.val[i * stride..(i + 1) * stride]) {
            if e + v > *b {
                (*b, *a) = (e + v, i as f64);
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn max_plus_avx2(band: &Band, end: &[f64], best: &mut [f64], arg: &mut [f64]) {
    use std::arch::x86_64::*;
    // four groups of four targets per pass, so four compare-and-blend chains run side by side
    const GROUP: usize = 16;
    let stride = band.stride;
    for j0 in (0..stride).step_by(GROUP) {
        let lanes = (stride - j0).min(GROUP) / 4;
        let mut b = [_mm256_set1_pd(f64::NEG_INFINITY); 4];
        let mut a = [_mm256_setzero_pd(); 4];
        for g in 0..lanes {
            a[g] = _mm256_loadu_pd(band.first.as_ptr().add(j0 + 4 * g));
        }
        for (i, &e) in end.iter().enumerate() {
            let (ev, iv) = (_mm256_set1_pd(e), _mm256_set1_pd(i as f64));
            let row = band.val.as_ptr().add(i * stride + j0);
            for g in 0..lanes {
                let v = _mm256_add_pd(ev, _mm256_loadu_pd(row.add(4 * g)));
                let gt = _mm256_cmp_pd::<_CMP_GT_OQ>(v, b[g]);
                b[g] = _mm256_blendv_pd(b[g], v, gt);
                a[g] = _mm256_blendv_pd(a[g], iv, gt);
            }
        }
        for g in 0..lanes {
            _mm256_storeu_pd(best.as_mut_ptr().add(j0 + 4 * g), b[g]);
            _mm256_storeu_pd(arg.as_mut_ptr().add(j0 + 4 * g), a[g]);
        }
    }
}

/// Beat and bar-head times (seconds) and the candidate meters from 100 fps observations (`bar_tracker.track`).
pub fn track(
    pb: &[f64],
    pd: &[f64],
    bar_beats: &[[f32; 16]],
    beats_left: &[[f32; 16]],
    o: &Options,
) -> (Vec<f64>, Vec<f64>, Vec<usize>) {
    let mut taus: Vec<usize> = geomspace(60.0 * FPS / o.max_bpm, 60.0 * FPS / o.min_bpm, o.n_tau)
        .iter()
        .map(|&v| round_even(v) as usize)
        .collect();
    taus.sort_unstable();
    taus.dedup();
    let nt = taus.len();
    let mut starts = vec![0usize; nt];
    for j in 1..nt {
        starts[j] = starts[j - 1] + taus[j - 1];
    }
    let size: usize = taus.iter().sum();
    let metres = candidate_metres(bar_beats, o.metre_min, o.local_min, (o.local_seconds * FPS) as usize);
    let rows: Vec<(usize, usize)> = metres.iter().flat_map(|&b| (0..b).map(move |k| (b, k))).collect();
    let nbk = rows.len();
    let k0: Vec<bool> = rows.iter().map(|&(_, k)| k == 0).collect();
    let to_down: Vec<bool> = rows.iter().map(|&(b, k)| k == b - 1).collect();
    let last: Vec<usize> = (0..nbk).filter(|&i| rows[i].1 == rows[i].0 - 1).collect();
    let first: Vec<usize> = (0..nbk).filter(|&i| rows[i].1 == 0).collect();
    let inner: Vec<usize> = (0..nbk).filter(|&i| rows[i].1 > 0).collect();
    let nb = metres.len();
    let ratio = |from: usize, to: usize| taus[to] as f64 / taus[from] as f64;
    let mut beat_tr: Vec<Vec<f64>> =
        (0..nt).map(|i| (0..nt).map(|j| -o.lam * (ratio(i, j) - 1.0).abs()).collect()).collect();
    norm_rows(&mut beat_tr);
    let ls = o.switch.ln();
    let mut bar_tr: Vec<Vec<f64>> = (0..nt)
        .map(|i| {
            (0..nt)
                .map(|j| {
                    let r = ratio(i, j);
                    let a = logaddexp(-o.lam * (r - 1.0).abs(), ls - o.lam * (2.0 * r - 1.0).abs());
                    logaddexp(a, ls - o.lam * (r / 2.0 - 1.0).abs())
                })
                .collect()
        })
        .collect();
    norm_rows(&mut bar_tr);
    let mut metre_tr = vec![vec![(o.meter_change / (nb.max(2) - 1) as f64).ln(); nb]; nb];
    for (m, row) in metre_tr.iter_mut().enumerate() {
        row[m] = if nb > 1 { (1.0 - o.meter_change).ln() } else { 0.0 };
    }
    let eps = 1e-6;
    let frames = pb.len();
    let mut gain_down = vec![0.0; frames];
    let mut gain_beat = vec![0.0; frames];
    for t in 0..frames {
        let (b, d) = (pb[t].clamp(eps, 1.0 - eps), pd[t].clamp(eps, 1.0 - eps));
        let other = (1.0 - b).ln();
        gain_down[t] = b.ln() + o.w_down * d.ln() - other;
        gain_beat[t] = b.ln() + o.w_down * (1.0 - d).ln() - other;
    }
    // evidence added when a beat starts; computed in float32 as numpy does on the heads' float32 probabilities
    let (w_bar, eps32) = (o.w_bar as f32, eps as f32);
    let mut bar_ev = vec![0.0f64; frames * nbk];
    for t in 0..frames {
        for (i, &(b, k)) in rows.iter().enumerate() {
            let left = w_bar * beats_left[t][(b - k).min(16) - 1].clamp(eps32, 1.0).ln();
            let v = if k == 0 { left + w_bar * bar_beats[t][b.min(16) - 1].clamp(eps32, 1.0).ln() } else { left };
            bar_ev[t * nbk + i] = v as f64;
        }
    }
    let beat_src = band(&beat_tr);
    let bar_src = band(&bar_tr);
    let mut is_first = vec![false; nbk];
    first.iter().for_each(|&i| is_first[i] = true);
    let net = Net {
        taus: &taus,
        starts: &starts,
        size,
        inner: &inner,
        first: &first,
        last: &last,
        k0: &k0,
        to_down: &to_down,
        is_first: &is_first,
    };
    let (mut beats, mut downs) = crate::profile::time("beat viterbi", || {
        viterbi(&net, &beat_src, &bar_src, &metre_tr, &gain_down, &gain_beat, &bar_ev, o.p_hold, o.q_hold)
    });
    beats.sort_unstable();
    downs.sort_unstable();
    (beats.iter().map(|&b| b as f64 / FPS).collect(), downs.iter().map(|&d| d as f64 / FPS).collect(), metres)
}

struct Net<'a> {
    taus: &'a [usize],
    starts: &'a [usize],
    size: usize,
    inner: &'a [usize],
    first: &'a [usize],
    last: &'a [usize],
    k0: &'a [bool],
    to_down: &'a [bool],
    is_first: &'a [bool],
}

/// `bar_tracker.viterbi`: the best path as (beat frames, bar-head frames), last first.
#[allow(clippy::too_many_arguments)]
fn viterbi(
    n: &Net,
    beat_src: &Band,
    bar_src: &Band,
    metre_tr: &[Vec<f64>],
    gain_down: &[f64],
    gain_beat: &[f64],
    bar_ev: &[f64],
    p_hold: f64,
    q_hold: f64,
) -> (Vec<usize>, Vec<usize>) {
    let frames = gain_down.len();
    let (nbk, nt, nb) = (n.k0.len(), n.taus.len(), n.last.len());
    let size = n.size;
    let mut buf = vec![-((nbk * size) as f64).ln(); nbk * size];
    // Back pointers, one byte each: the interval a state's beat came from (top bit: through a hold); the row is
    // the one before for a bar's inner beats, and for a bar's first beat the last of the metre in `metre_back`.
    assert!(nt < 128 && nb < 256, "too many intervals or meters for one-byte back pointers");
    let mut back = vec![0u8; frames * nbk * nt];
    let mut metre_back = vec![0u8; frames * nb * nt];
    let mut metre_of = vec![0usize; nbk];
    for (m, &f) in n.first.iter().enumerate() {
        metre_of[f] = m;
    }
    let mut end = vec![0.0f64; nbk * nt];
    let mut start = vec![0.0f64; nbk * nt];
    let mut tbest = vec![0.0f64; nb * nt];
    let mut targ = vec![0usize; nb * nt];
    let use_hold = p_hold > 0.0;
    let mut hold = vec![f64::NEG_INFINITY; nbk * nt];
    let mut stay = if use_hold { vec![false; frames * nbk * nt] } else { Vec::new() };
    let mut via = vec![false; nbk * nt];
    let (mut best, mut args) = (vec![0.0f64; beat_src.stride], vec![0.0f64; beat_src.stride]);
    const SLOTS: usize = 2 + 2 * HALF; // the new beat's slot, then HALF + 1 after it and HALF before the next
    let mut slots = vec![0usize; nt * SLOTS];
    const HELD: u8 = 0x80;
    let l_go = (1.0 - p_hold).ln();
    let l_enter = if use_hold { p_hold.ln() } else { f64::NEG_INFINITY };
    let (l_stay, l_exit) =
        if use_hold { (q_hold.ln(), (1.0 - q_hold).ln()) } else { (f64::NEG_INFINITY, f64::NEG_INFINITY) };
    let modulo = |a: isize, m: usize| a.rem_euclid(m as isize) as usize;
    for t in 0..frames {
        for r in 0..nbk {
            for j in 0..nt {
                let e = buf[r * size + n.starts[j] + t % n.taus[j]];
                let s = r * nt + j;
                if use_hold {
                    let (d, h) = (e + l_go, hold[s] + l_exit);
                    via[s] = h > d;
                    end[s] = if h > d { h } else { d };
                    let (a, b) = (hold[s] + l_stay, e + l_enter);
                    stay[(t * nbk + r) * nt + j] = a >= b;
                    hold[s] = if a >= b { a } else { b };
                } else {
                    end[s] = e;
                }
            }
        }
        let back_t = &mut back[t * nbk * nt..(t + 1) * nbk * nt];
        for &r in n.inner {
            max_plus(beat_src, &end[(r - 1) * nt..r * nt], &mut best, &mut args);
            start[r * nt..(r + 1) * nt].copy_from_slice(&best[..nt]);
            for j in 0..nt {
                let arg = args[j] as usize;
                back_t[r * nt + j] = arg as u8 | if via[(r - 1) * nt + arg] { HELD } else { 0 };
            }
        }
        for m in 0..nb {
            let l = n.last[m];
            max_plus(bar_src, &end[l * nt..(l + 1) * nt], &mut best, &mut args);
            tbest[m * nt..(m + 1) * nt].copy_from_slice(&best[..nt]);
            for j in 0..nt {
                targ[m * nt + j] = args[j] as usize;
            }
        }
        for m2 in 0..nb {
            for j in 0..nt {
                let (mut best, mut arg) = (f64::NEG_INFINITY, 0);
                for m in 0..nb {
                    let v = tbest[m * nt + j] + metre_tr[m][m2];
                    if v > best {
                        (best, arg) = (v, m);
                    }
                }
                let f = n.first[m2];
                start[f * nt + j] = best;
                let (lr, tj) = (n.last[arg], targ[arg * nt + j]);
                back_t[f * nt + j] = tj as u8 | if via[lr * nt + tj] { HELD } else { 0 };
                metre_back[(t * nb + m2) * nt + j] = arg as u8;
            }
        }
        // the slots of positions 0..=HALF (just after the beat) and tau - HALF..tau (just before the next), per
        // interval: the same for every row, and computed without a division per slot
        for j in 0..nt {
            let (tau, base) = (n.taus[j], n.starts[j]);
            let now = t % tau;
            let wrap = |p: usize| base + if now >= p { now - p } else { now + tau - p };
            slots[j * SLOTS] = base + now;
            for p in 0..=HALF {
                slots[j * SLOTS + 1 + p] = wrap(p);
            }
            for (i, p) in (tau - HALF..tau).enumerate() {
                slots[j * SLOTS + 2 + HALF + i] = wrap(p);
            }
        }
        for r in 0..nbk {
            let row = &mut buf[r * size..(r + 1) * size];
            let after = if n.k0[r] { gain_down[t] } else { gain_beat[t] };
            let before = if n.to_down[r] { gain_down[t] } else { gain_beat[t] };
            for j in 0..nt {
                let s = &slots[j * SLOTS..(j + 1) * SLOTS];
                row[s[0]] = start[r * nt + j] + bar_ev[t * nbk + r];
                for &slot in &s[1..2 + HALF] {
                    row[slot] += after;
                }
                for &slot in &s[2 + HALF..] {
                    row[slot] += before;
                }
            }
        }
        if t % 256 == 255 {
            let top = buf.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            buf.iter_mut().for_each(|v| *v -= top);
            hold.iter_mut().for_each(|v| *v -= top);
        }
    }
    let (mut best, mut row, mut j, mut p) = (f64::NEG_INFINITY, 0usize, 0usize, 0usize);
    for r in 0..nbk {
        for jj in 0..nt {
            for slot in 0..n.taus[jj] {
                let v = buf[r * size + n.starts[jj] + slot];
                if v > best {
                    (best, row, j, p) = (v, r, jj, modulo(frames as isize - 1 - slot as isize, n.taus[jj]));
                }
            }
        }
    }
    let (mut beats, mut downs) = (Vec::new(), Vec::new());
    let mut t = frames as isize - 1;
    loop {
        let t0 = t - p as isize;
        if t0 < 0 {
            break;
        }
        beats.push(t0 as usize);
        if n.is_first[row] {
            downs.push(t0 as usize);
        }
        let t0u = t0 as usize;
        let prev = back[(t0u * nbk + row) * nt + j];
        let held = prev & HELD != 0;
        let prev_row =
            if n.is_first[row] { n.last[metre_back[(t0u * nb + metre_of[row]) * nt + j] as usize] } else { row - 1 };
        (row, j) = (prev_row, (prev & !HELD) as usize);
        t = t0 - 1;
        if held {
            while t > 0 && stay[(t as usize * nbk + row) * nt + j] {
                t -= 1;
            }
            t -= 1;
        }
        p = n.taus[j] - 1;
    }
    (beats, downs)
}

/// `track` on a head's 25 Hz probabilities (`track_head`): beat and bar-head probabilities interpolated to 100
/// fps, the bar facts held per head frame.
pub fn track_head(probs: &BeatProbs, o: &Options) -> (Vec<f64>, Vec<f64>, Vec<usize>) {
    let n = probs.beat.len();
    let head_t: Vec<f64> = (0..n).map(|i| (i as f64 + 0.5) / HEAD_RATE).collect();
    let count = (head_t[n - 1] * FPS) as usize + 1;
    let t100: Vec<f64> = (0..count).map(|i| i as f64 / FPS).collect();
    let near: Vec<usize> = t100.iter().map(|&t| ((t * HEAD_RATE) as usize).min(n - 1)).collect();
    let beat: Vec<f64> = probs.beat.iter().map(|&v| v as f64).collect();
    let down: Vec<f64> = probs.downbeat.iter().map(|&v| v as f64).collect();
    let bar_beats: Vec<[f32; 16]> = near.iter().map(|&i| probs.bar_beats[i]).collect();
    let beats_left: Vec<[f32; 16]> = near.iter().map(|&i| probs.beats_left[i]).collect();
    track(&interp(&t100, &head_t, &beat), &interp(&t100, &head_t, &down), &bar_beats, &beats_left, o)
}

/// Beats moved by the head's sub-frame offsets (`apply_offsets`); bar heads move with their beat.
pub fn apply_offsets(
    beats: &[f64],
    downbeats: &[f64],
    beat_prob: &[f32],
    beat_offset: &[f32],
    limit: f64,
) -> (Vec<f64>, Vec<f64>) {
    let mut moved_beats = beats.to_vec();
    for (i, &t) in beats.iter().enumerate() {
        let j0 = (t * HEAD_RATE) as isize;
        let mut best: Option<usize> = None;
        for j in [j0 - 1, j0, j0 + 1] {
            if j >= 0 && (j as usize) < beat_prob.len() && best.is_none_or(|b| beat_prob[j as usize] > beat_prob[b]) {
                best = Some(j as usize);
            }
        }
        let Some(j) = best else { continue };
        let moved = (j as f64 + 0.5 + beat_offset[j].clamp(-1.5, 1.5) as f64) / HEAD_RATE;
        if (moved - t).abs() <= limit {
            moved_beats[i] = moved;
        }
    }
    let downs = if beats.is_empty() || downbeats.is_empty() {
        downbeats.to_vec()
    } else {
        downbeats
            .iter()
            .map(|&d| {
                let k =
                    (0..beats.len()).fold(0, |b, k| if (beats[k] - d).abs() < (beats[b] - d).abs() { k } else { b });
                moved_beats[k]
            })
            .collect()
    };
    (moved_beats, downs)
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 { v[n / 2] } else { (v[n / 2 - 1] + v[n / 2]) / 2.0 }
}

/// Each bar's denominator from the head's `bar_unit`, keeping the written tempo (`denominators`).
pub fn denominators(
    beats: &[f64],
    downbeats: &[f64],
    bar_unit: &[[f32; 4]],
    duration: f64,
    tempo_weight: f64,
) -> Vec<u32> {
    if downbeats.is_empty() {
        return Vec::new();
    }
    let nu = UNITS.len();
    let mut emit: Vec<[f64; 4]> = Vec::new();
    let mut ibi = Vec::new();
    for (i, &start) in downbeats.iter().enumerate() {
        let end = if i + 1 < downbeats.len() { downbeats[i + 1] } else { duration };
        let f0 = (start * HEAD_RATE) as usize;
        let mut e = [0.0f64; 4];
        if f0 < bar_unit.len() {
            let f1 = ((end * HEAD_RATE) as usize).max(f0 + 1).min(bar_unit.len());
            for (u, e) in e.iter_mut().enumerate() {
                let mean = bar_unit[f0..f1].iter().map(|r| r[u] as f64).sum::<f64>() / (f1 - f0) as f64;
                *e = ((mean as f32 + 1e-6f32).ln()) as f64;
            }
        } else {
            e.iter_mut().for_each(|e| *e = (1.0 / nu as f64 + 1e-6).ln());
        }
        emit.push(e);
        let inside: Vec<f64> = beats.iter().cloned().filter(|&b| b >= start - 0.01 && b < end + 0.01).collect();
        ibi.push(if inside.len() > 1 { median(inside.windows(2).map(|w| w[1] - w[0]).collect()) } else { f64::NAN });
    }
    let q: Vec<f64> = UNITS.iter().map(|&u| (u as f64 / 4.0).ln()).collect();
    let mut score = emit[0];
    let mut back = vec![[0usize; 4]; emit.len()];
    for i in 1..emit.len() {
        let mut next = [f64::NEG_INFINITY; 4];
        for to in 0..nu {
            for from in 0..nu {
                let cost = if ibi[i - 1].is_nan() || ibi[i].is_nan() {
                    0.0
                } else {
                    tempo_weight * ((ibi[i].ln() + q[to]) - (ibi[i - 1].ln() + q[from])).abs()
                };
                let v = score[from] - cost;
                if v > next[to] {
                    next[to] = v;
                    back[i][to] = from;
                }
            }
        }
        for u in 0..nu {
            score[u] = next[u] + emit[i][u];
        }
    }
    let mut path = vec![(0..nu).fold(0, |b, u| if score[u] > score[b] { u } else { b })];
    for i in (1..emit.len()).rev() {
        path.push(back[i][*path.last().unwrap()]);
    }
    path.iter().rev().map(|&u| UNITS[u]).collect()
}

/// `HeadBeats.track_probabilities`: beats, bar heads and bars with their meters.
pub fn track_beats(probs: &BeatProbs, o: &Options, tempo_weight: f64) -> Tracked {
    let (mut beats, mut downbeats, meters) = crate::profile::time("beat tracker", || track_head(probs, o));
    if let Some(offset) = &probs.beat_offset {
        (beats, downbeats) = apply_offsets(&beats, &downbeats, &probs.beat, offset, 0.06);
    }
    let mut bars: Vec<Bar> = downbeats
        .iter()
        .enumerate()
        .map(|(i, &d)| {
            let end = downbeats.get(i + 1).copied().unwrap_or(f64::INFINITY);
            let count = beats.iter().filter(|&&b| b >= d - 1e-6 && b < end - 1e-6).count();
            Bar { start: d, beats: count, unit: None }
        })
        .collect();
    if let Some(unit) = &probs.bar_unit {
        let duration = probs.beat.len() as f64 / HEAD_RATE;
        for (bar, u) in bars.iter_mut().zip(denominators(&beats, &downbeats, unit, duration, tempo_weight)) {
            bar.unit = Some(u);
        }
    }
    Tracked { beats, downbeats, bars, meters }
}
