# sheetsage2-cpu

SheetSage2's MERT-v2 encoder, SheetSage2-Refined's beat and melody heads and their trackers on the CPU, in Rust, for
machines without a GPU. One pass of the encoder feeds both heads; no Python or PyTorch at run time.

## Use

```bash
python -m sheetsage2_refined.export_cpu cpu.safetensors [--dtype f16|f32|i8] [--lora ADAPTER --lora-scale S]
cargo build --release
./target/release/sheetsage2-cpu --weights cpu.safetensors [--adapter LORA] [--threads N] [--out-dir DIR] \
    [--heads beat,melody] [--tempo-weight W] [--logits] song.mp3 ...
```

writes `<out-dir>/<song>.beats.json` and `<song>.melody.json` (default `outputs/cpu/`) as
`python -m sheetsage2_refined.beats` / `.melody` do; `--logits` also writes the heads' logits
(`<song>.heads.safetensors`) and `--from-logits` runs only the trackers on such files.

`--threads` defaults to one per logical CPU; on a CPU with SMT try the number of physical cores as well.
WAV, FLAC, MP3 and Ogg Vorbis are decoded by Symphonia and resampled to 24 kHz as `torchaudio.functional.resample`
does; a `.f32` file is taken as raw float32 samples, mono at 24 kHz.

## LoRA adapters at run time

```bash
python -m sheetsage2_refined.export_cpu my.lora.safetensors --adapter LORA.safetensors [--lora-scale S]
./target/release/sheetsage2-cpu --weights cpu.safetensors --adapter my.lora.safetensors song.mp3
```

`--adapter` (in `export_cpu`) writes only the adapter, folded the way the weights it changes are: `A` with the
norm's scale, `B` with the scale, the feed-forward's 0.5 and q's `1/sqrt(64)`, and the shift the norm's `beta` (or
GRN's) sends through the adapter as an extra bias, all as float16 (a rank-16 adapter on q, k, v, out and both
feed-forwards of all 24 blocks: 22 MiB). The engine keeps the 1.2 GB base export as it is and adds
`(x A^T) B^T + db` to each adapted layer's output, before its GELU; q, k and v keep their own `B`s inside the
shared `qkv` layer. Any of the encoder's linear layers can be adapted (the Conformer's, the pointwise convolutions,
the ConvNeXt stages' and their downsampling); a LoRA on a depthwise convolution has to be merged with
`export_cpu --lora` instead. `Model::set_adapter` swaps or removes an adapter between songs, on the CPU or the GPU.

On a 60 s song the rank-16 adapter above costs no measurable time (10.9-11.3 s with it, 10.9-11.2 s without) and
33 MiB of memory. Against the merged export the logits agree within 1.5e-5 with the adapter in float32, and within
5e-3 in float16 for a test adapter that moves them by up to 1.6 (float16 base weights alone differ from float32 by
up to 1.6e-2).

## On a GPU

```bash
cargo build --release --features gpu
./target/release/sheetsage2-cpu --device gpu --weights cpu.safetensors song.mp3
cargo run --release --features gpu --example gpu_bench -- cpu.safetensors   # kernels vs the CPU's, GFLOP/s
```

`--device gpu` runs the encoder through wgpu (Vulkan on Linux: Mesa's RADV for an AMD GPU, `vulkaninfo --summary`
lists what is there; `SHEETSAGE_GPU=<part of the name>` picks an adapter). The log-mel spectrogram, the heads and
the trackers stay on the CPU. The shaders (`src/gpu/*.wgsl`) compute what the CPU kernels do, in float32 with the
float16 weights: a 128 x 128-tile GEMM with 8 x 8 outputs per thread, the row norms and convolutions one workgroup
per row, flash-style attention with one query per thread (its q and output rows in registers, 16 keys and values
at a time shared through workgroup memory; `gpu_bench` also times the previous kernel, 32 x 32 tiles in workgroup
memory, `examples/attention_tiled.wgsl`). GPU memory (on an integrated GPU, system memory):
1.2 GB of weights and 264 MB of activations, allocated once for the longest window; the CPU's mapping of the
weights is dropped after the upload. The work is submitted a block at a time, so no submission runs long enough
for the driver to reset the GPU.

Checked on Mesa's software Vulkan (llvmpipe): the logits of a 60 s synthetic song are within 5e-4 of the CPU
engine's (beats, bars and notes identical) and `gpu_bench`'s linear layers and attention on a whole 300 s window
within 2e-5. Its speed has to be measured on the GPU it runs on:
`SHEETSAGE_PROFILE=1` prints the GPU time per part (it waits for the GPU after each, so the total grows a little).

## How it works

* `export_cpu` folds every constant it can: LayerNorm scales and shifts into the linear layer that follows, the
  feed-forward `0.5`, attention's `1/sqrt(64)`, the downsampling LayerNorm into the stride-2 convolution, GRN's shift
  into `fc2`'s bias. q, k and v are one matrix. Linear weights are stored in 16-column panels, the order the kernel
  reads them, as float16, float32 or int8 (scale per output channel), and the file is memory-mapped.
* Matrix products (`src/gemm.rs`) are BLIS-style: rows in parallel blocks, packed into 6-row panels per slice of `k`,
  each weight panel converted to float32 once per block and swept by a 6x16 AVX2 micro-kernel (12 accumulators)
  that adds straight into the output: `llvm-mca` puts its inner loop at 12 FMAs in 6 cycles, the FMA peak of a
  core with two 256-bit FMA units.
* Attention (`src/encoder.rs`) is flash-style: per head and block of 96 queries, keys in blocks of 256 with a running
  max and sum, so no frames x frames matrix exists. Probabilities below e^-40 of a row's best are flushed to zero and
  every thread runs with FTZ / DAZ: products of tiny probabilities and values are denormal numbers, which made
  attention on real (peaked) scores 10-45x slower (`examples/attention.rs --sharp 20`). The bfloat16-rounded rotary
  tables SheetSage2 was trained with come from the export.
* The heads read learned mixes of all 25 encoder states. The encoder adds each state into the mixes as it goes, so
  only the current state is kept, and the feed-forward blocks run block by block (no 4096-wide activations for the
  whole window).
* The heads (`src/heads.rs`) run their GRU on 60 s pieces with 15 s of context, as `heads.CHUNK`, in parallel.
* The trackers (`src/tracker.rs`, `src/melody.rs`) are `bar_tracker.py` and `melody.py`'s `line_notes` / `notes`
  line by line, the beat tracker's interval transitions as AVX2 max-plus products.

## Accuracy

`scripts/check.py` runs PyTorch (float32, CPU) and the engine on the same waveform and compares the heads' logits and
what the trackers make of them; `scripts/check_post.py` runs the Rust and the Python trackers on the same logits
(with `--noise`, also on noisy copies); `scripts/synth_song.py` makes a test song (drums, bass, chords, a melody,
meter changes) when no audio is at hand. On a 330 s synthetic song (two encoder windows):

| weights | file | largest logit difference (beat / melody) | beats, downbeats, meters, notes vs PyTorch |
|---|---|---|---|
| f32 | 2.4 GB | 0.006 / 0.03 (100 s song) | identical |
| f16 (default) | 1.2 GB | 0.03 / 0.08 | identical (523 beats, 459 notes) |
| i8 | 0.6 GB | 0.37 / 0.82 (100 s song) | identical on the synthetic songs; check on real music first |

Differences of a few hundredths are the model's own sensitivity: a waveform that differs by 2.5e-6 (resampling in
float64 or float32) moves the logits as much. The Rust trackers give the same beats, bars, meters and notes as the
Python ones on the logits of both songs and on 20 noisy copies (noise up to 3 on every logit).

## Speed and memory

With the AVX2 code path and float16 weights, a 300 s song (times depend on the CPU; the comparison is on one
machine):

| | time | memory |
|---|---|---|
| this engine: encoder + both heads + both trackers | 69 s + 1.4 s | 217 MB own + 1.2 GB mapped weights |
| PyTorch, AVX2 only: encoder + beat head | 70 s, again for the melody head | 5.5 GB |
| PyTorch with AVX-512: encoder + beat head | 49 s, again for the melody head | 5.5 GB |

Per encoder pass the engine is about as fast as PyTorch's MKL / oneDNN kernels on the same instruction set; it is
faster when both heads are needed. Of a 300 s run: feed-forward 26 s, attention 32 s (25 s of it the scores,
softmax and weighted values, at about 80% of the FMA peak), convolution module 6 s, ConvNeXt stages 5 s, heads 3 s
(`SHEETSAGE_PROFILE=1` prints this, and the peak memory per part).

Memory, for a server: the weights are a memory-mapped file, so they live in the page cache, shared by every process
that maps the same file and reclaimable by the kernel (`--dtype i8` halves them to 0.6 GB, but moved a bar line of
the 330 s synthetic song: check it on real music first). A request's own memory is about 220 MB for 5 minutes of
audio (75 MB for one minute): the 25 encoder states are never kept (each is added into the heads' layer mixes),
q / k / v, the convolution module and the ConvNeXt layers run per block of rows, attention's output overwrites q,
and the beat tracker's back pointers are one byte each. The engine gives freed memory back to the system after each
song (`malloc_trim`), 16 MB between songs.

To measure and tune it on a machine (N threads):

```bash
cargo run --release --example bench -- --threads N --tune     # kernel peak, linear layers, best SHEETSAGE_ROWS / SHEETSAGE_KC
cargo run --release --example attention -- 7500 N --sharp 20  # the attention core on a 300 s window
SHEETSAGE_PROFILE=1 ./target/release/sheetsage2-cpu --threads N --weights cpu.safetensors song.wav
```
