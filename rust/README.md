# sheetsage2-cpu

SheetSage2's MERT-v2 encoder and SheetSage2-Refined's beat and melody heads on the CPU, in Rust, for machines
without a GPU. One pass of the encoder feeds both heads.

## Use

```bash
python -m sheetsage2_refined.export_cpu cpu.safetensors [--dtype f16|f32|i8] [--lora ADAPTER --lora-scale S]
cargo build --release
./target/release/sheetsage2-cpu --weights cpu.safetensors [--threads N] [--out-dir DIR] song.mp3 ...
python -m sheetsage2_refined.cpu song.mp3 --weights cpu.safetensors      # beats, bars and notes as JSON
```

`--threads` defaults to one per logical CPU; on a Ryzen try the number of physical cores (8 on a 5900HX) as well.
WAV, FLAC, MP3 and Ogg Vorbis are decoded by Symphonia and resampled to 24 kHz as `torchaudio.functional.resample`
does; a `.f32` file is taken as raw float32 samples, mono at 24 kHz.

## How it works

* `export_cpu` folds every constant it can: LayerNorm scales and shifts into the linear layer that follows, the
  feed-forward `0.5`, attention's `1/sqrt(64)`, the downsampling LayerNorm into the stride-2 convolution, GRN's shift
  into `fc2`'s bias. q, k and v are one matrix. Linear weights are stored in 16-column panels, the order the kernel
  reads them, as float16, float32 or int8 (scale per output channel), and the file is memory-mapped.
* Matrix products (`src/gemm.rs`) are BLIS-style: rows in parallel blocks, packed into 6-row panels per slice of `k`,
  each weight panel converted to float32 once per block and swept by a 6x16 AVX2 micro-kernel (12 accumulators).
  `llvm-mca -mcpu=znver3` puts its inner loop at 6 cycles for 12 FMAs: Zen 3's FMA peak.
* Attention (`src/encoder.rs`) is flash-style: per head and block of 96 queries, keys in blocks of 256 with a running
  max and sum, so no frames x frames matrix exists. The bfloat16-rounded rotary tables SheetSage2 was trained with
  come from the export.
* The heads read learned mixes of all 25 encoder states. The encoder adds each state into the mixes as it goes, so
  only the current state is kept, and the feed-forward blocks run block by block (no 4096-wide activations for the
  whole window).
* The heads (`src/heads.rs`) run their GRU on 60 s pieces with 15 s of context, as `heads.CHUNK`, in parallel.

## Accuracy

`scripts/check.py` runs PyTorch (float32, CPU) and the engine on the same waveform and compares the heads' logits and
what the trackers make of them; `scripts/synth_song.py` makes a test song (drums, bass, chords, a melody, meter
changes) when no audio is at hand. On a 330 s synthetic song (two encoder windows):

| weights | file | largest logit difference (beat / melody) | beats, downbeats, meters, notes vs PyTorch |
|---|---|---|---|
| f32 | 2.4 GB | 0.006 / 0.03 (100 s song) | identical |
| f16 (default) | 1.2 GB | 0.03 / 0.08 | identical (523 beats, 459 notes) |
| i8 | 0.6 GB | 0.37 / 0.82 (100 s song) | identical on the synthetic songs; check on real music first |

Differences of a few hundredths are the model's own sensitivity: a waveform that differs by 2.5e-6 (resampling in
float64 or float32) moves the logits as much.

## Speed and memory

On the 4-vCPU cloud machine it was developed on (Intel Xeon, AVX2 code path only), float16 weights:

| audio | encoder + both heads | peak memory |
|---|---|---|
| 100 s | 18 s | 1.3 GB |
| 300 s (one full window) | 71 s | 1.5 GB |

PyTorch on the same machine (with AVX-512 and oneDNN) takes 48 s per encoder pass for 330 s, 5.8 GB of memory for
100 s, and runs the encoder once per head. Of the 300 s run: feed-forward 24 s, attention 35 s (29 s of it the
scores and softmax), convolution module 5 s, ConvNeXt stages 4 s, heads 2.4 s (`SHEETSAGE_PROFILE=1` prints this).
The micro-kernel reaches 283 GFLOP/s on these 4 vCPUs and the linear layers 240-260 GFLOP/s.

A Ryzen 9 5900HX (8 Zen 3 cores, 2 x 256-bit FMA each) has about three times the FMA throughput of this machine, so
a 5-minute song should take about 25 s there; measure it with:

```bash
cargo run --release --example bench -- --threads 8 --tune   # kernel peak, linear layers, best SHEETSAGE_ROWS / SHEETSAGE_KC
SHEETSAGE_PROFILE=1 ./target/release/sheetsage2-cpu --threads 8 --weights cpu.safetensors song.wav
```
