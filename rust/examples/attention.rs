//! Times the attention core (`encoder::attend`) on random q, k, v: `cargo run --release --example attention -- FRAMES [THREADS]`

use std::time::Instant;

use sheetsage2_cpu::encoder::{HEAD_DIM, HEADS, WIDTH, attend};
use sheetsage2_cpu::gemm::{MR, NR};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let frames: usize = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(2500);
    let threads: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(1);
    sheetsage2_cpu::simd::init_threads(Some(threads)).unwrap();
    let padded = frames.div_ceil(NR) * NR;
    let mut seed = 1u32;
    let mut rand = || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        (seed >> 8) as f32 / (1 << 24) as f32 - 0.5
    };
    // `--sharp S` scales q so the scores spread over +-S/2 or so: real attention has many keys far below the best
    let sharp: f32 =
        args.iter().position(|a| a == "--sharp").and_then(|i| args.get(i + 1)?.parse().ok()).unwrap_or(1.0);
    let qkv: Vec<f32> =
        (0..(frames + MR) * 3 * WIDTH).map(|i| rand() * if i % (3 * WIDTH) < WIDTH { sharp } else { 1.0 }).collect();
    let keys: Vec<f32> = (0..HEADS * HEAD_DIM * padded).map(|_| rand() * 4.0).collect();
    let values: Vec<f32> = (0..HEADS * HEAD_DIM * padded).map(|_| rand()).collect();
    let mut q: Vec<f32> = qkv.chunks(3 * WIDTH).flat_map(|r| r[..WIDTH].to_vec()).collect();
    attend(&mut q.clone(), &keys, &values, padded, frames);
    let mut best = f64::INFINITY;
    for _ in 0..3 {
        q.iter_mut().zip(qkv.chunks(3 * WIDTH).flat_map(|r| r[..WIDTH].to_vec())).for_each(|(q, v)| *q = v);
        let start = Instant::now();
        attend(&mut q, &keys, &values, padded, frames);
        best = best.min(start.elapsed().as_secs_f64());
    }
    let secs = best;
    let flops = 4.0 * (frames * frames * WIDTH) as f64;
    println!("{frames} frames, {threads} threads: {secs:.3} s, {:.1} GFLOP/s", flops / secs / 1e9);
}
