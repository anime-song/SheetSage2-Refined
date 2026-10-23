//! The GPU encoder's kernels against the CPU engine's, and their speed: block 0's linear layers and the attention
//! core on random inputs.  `cargo run --release --features gpu --example gpu_bench -- cpu.safetensors [FRAMES]`
//! (FRAMES defaults to 7500, a whole 300 s window).

use sheetsage2_cpu::encoder::{HEAD_DIM, HEADS, WIDTH, attend};
use sheetsage2_cpu::gemm::{self, Act, MR, NR};
use sheetsage2_cpu::gpu::GpuEncoder;
use sheetsage2_cpu::weights::Weights;

fn max_diff(a: &[f32], b: &[f32]) -> (f32, f32) {
    let d = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max);
    (d, a.iter().map(|v| v.abs()).fold(0.0, f32::max))
}

fn main() -> sheetsage2_cpu::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let weights = Weights::open(std::path::Path::new(args.get(1).ok_or("usage: gpu_bench cpu.safetensors [FRAMES]")?))?;
    let frames: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(7500);
    sheetsage2_cpu::simd::init_threads(None)?;
    let gpu = GpuEncoder::load(&weights, 1)?;
    println!("{}", gpu.adapter);
    let mut seed = 7u32;
    let mut rand = || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        (seed >> 8) as f32 / (1 << 24) as f32 - 0.5
    };
    for (which, name) in [(0, "ffn1.fc1"), (1, "ffn1.fc2"), (2, "attn.qkv")] {
        let lin = weights.linear(&format!("layers.0.{name}"))?;
        let a: Vec<f32> = (0..frames * lin.k).map(|_| rand()).collect();
        let (out, secs) = gpu.bench_linear(which, &a, frames as u32, 3)?;
        let mut want = vec![0.0f32; frames * lin.n];
        gemm::linear(&a, lin.k, frames, &lin, &mut want, lin.n, Act::None, false, None);
        let (d, range) = max_diff(&out, &want);
        let flops = 2.0 * (frames * lin.n * lin.k) as f64;
        println!(
            "linear {name:9} {frames} x {} -> {}: {:.1} GFLOP/s, max |gpu - cpu| {d:.2e} (values up to {range:.1})",
            lin.k,
            lin.n,
            flops / secs / 1e9
        );
    }
    // attention: q k v with the scale of real ones, then the CPU's layouts
    let qkv: Vec<f32> =
        (0..frames * 3 * WIDTH).map(|i| rand() * if i % (3 * WIDTH) < WIDTH { 8.0 } else { 2.0 }).collect();
    let (out, secs) = gpu.bench_attention(&qkv, frames as u32, 2, None)?;
    let others = [
        (
            "examples/attention_pair.wgsl: two threads per query, half the registers",
            include_str!("attention_pair.wgsl"),
            128,
        ),
        (
            "examples/attention_scalar.wgsl: keys and values read directly, no workgroup memory",
            include_str!("attention_scalar.wgsl"),
            256,
        ),
        ("examples/attention_tiled.wgsl: the previous kernel, 32 x 32 tiles", include_str!("attention_tiled.wgsl"), 32),
    ];
    let mut timed = Vec::new();
    for (name, src, queries) in others {
        timed.push((name, gpu.bench_attention(&qkv, frames as u32, 2, Some((src, queries)))?));
    }
    let padded = frames.div_ceil(NR) * NR;
    let mut q = vec![0.0f32; (frames + MR) * WIDTH];
    let mut keys = vec![0.0f32; HEADS * HEAD_DIM * padded];
    let mut values = vec![0.0f32; HEADS * HEAD_DIM * padded];
    for t in 0..frames {
        let row = &qkv[t * 3 * WIDTH..(t + 1) * 3 * WIDTH];
        q[t * WIDTH..(t + 1) * WIDTH].copy_from_slice(&row[..WIDTH]);
        for h in 0..HEADS {
            for d in 0..HEAD_DIM {
                keys[h * HEAD_DIM * padded + (t / NR) * HEAD_DIM * NR + d * NR + t % NR] =
                    row[WIDTH + h * HEAD_DIM + d];
                values[(h * padded + t) * HEAD_DIM + d] = row[2 * WIDTH + h * HEAD_DIM + d];
            }
        }
    }
    attend(&mut q, &keys, &values, padded, frames);
    let (d, range) = max_diff(&out, &q[..frames * WIDTH]);
    let flops = 4.0 * (frames * frames * WIDTH) as f64;
    println!(
        "attention {frames} frames: {:.1} GFLOP/s, {secs:.3} s per layer, max |gpu - cpu| {d:.2e} (values up to {range:.1})",
        flops / secs / 1e9
    );
    for (name, (other, other_secs)) in timed {
        let (d, _) = max_diff(&other, &q[..frames * WIDTH]);
        println!(
            "  vs {name}: {:.1} GFLOP/s, {other_secs:.3} s per layer, max |gpu - cpu| {d:.2e}",
            flops / other_secs / 1e9
        );
    }
    Ok(())
}
