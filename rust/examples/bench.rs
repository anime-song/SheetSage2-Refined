//! Micro-benchmarks: the 6x16 kernel with its operands in cache (the machine's FMA peak) and `linear` on the
//! encoder's shapes; `--tune` searches the blocking (`SHEETSAGE_ROWS`, `SHEETSAGE_KC`) for this machine.
//!
//!     cargo run --release --example bench [-- --threads N] [--tune]

use std::time::Instant;

use rayon::prelude::*;
use sheetsage2_cpu::gemm::{self, Act, MR, NR};
use sheetsage2_cpu::weights::{Linear, Panels};

fn leak<T>(v: Vec<T>) -> &'static [T] {
    Box::leak(v.into_boxed_slice())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let threads = args.iter().position(|a| a == "--threads").and_then(|i| args.get(i + 1)?.parse().ok());
    let threads: usize = threads.unwrap_or(std::thread::available_parallelism().map_or(1, |n| n.get()));
    sheetsage2_cpu::simd::init_threads(Some(threads)).unwrap();
    let (a, b) = (vec![1.0f32; MR * 256], vec![0.5f32; 256 * NR]);
    let reps = 200_000;
    for t in [1, threads] {
        let start = Instant::now();
        (0..t).into_par_iter().for_each(|_| {
            let mut tile = [0.0f32; MR * NR];
            for _ in 0..reps {
                gemm::kernel(256, &a, 1, MR, &b, NR, &mut tile);
                std::hint::black_box(&tile);
            }
        });
        let flops = 2.0 * (MR * NR * 256 * reps * t) as f64;
        println!("kernel in cache, {t} threads: {:.1} GFLOP/s", flops / start.elapsed().as_secs_f64() / 1e9);
    }
    let m = 2500;
    let shapes = [(4096, 1024), (1024, 4096), (3072, 1024)];
    let layers: Vec<Linear> = shapes
        .iter()
        .map(|&(n, k)| Linear {
            n,
            k,
            w: Panels::F16(leak(vec![half::f16::from_f32(0.01).to_bits(); n * k])),
            scale: None,
            bias: leak(vec![0.0; n]),
        })
        .collect();
    let x = vec![1.0f32; m * 4096];
    let mut out = vec![0.0f32; m * 4096];
    let mut run = |lin: &Linear| {
        gemm::linear(&x, lin.k, m, lin, &mut out, lin.n, Act::None, false, None);
        // the best of several runs: the cloud machines this was tuned on are noisy
        let mut best = f64::INFINITY;
        for _ in 0..7 {
            let start = Instant::now();
            gemm::linear(&x, lin.k, m, lin, &mut out, lin.n, Act::Gelu, false, None);
            best = best.min(start.elapsed().as_secs_f64());
        }
        2.0 * (m * lin.n * lin.k) as f64 / best / 1e9
    };
    if args.iter().any(|a| a == "--tune") {
        let mut best = (0.0, 0, 0);
        for rows in [48, 96, 144, 192] {
            for kc in [128, 192, 256, 384, 512] {
                gemm::set_blocking(rows, kc);
                let gflops = layers.iter().map(&mut run).sum::<f64>() / layers.len() as f64;
                println!("SHEETSAGE_ROWS={rows} SHEETSAGE_KC={kc}: {gflops:.1} GFLOP/s");
                if gflops > best.0 {
                    best = (gflops, rows, kc);
                }
            }
        }
        println!("best: SHEETSAGE_ROWS={} SHEETSAGE_KC={} ({:.1} GFLOP/s)", best.1, best.2, best.0);
    } else {
        for lin in &layers {
            println!("linear {m}x{} -> {} (f16): {:.1} GFLOP/s", lin.k, lin.n, run(lin));
        }
    }
}
