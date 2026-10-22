# sheetsage2-cpu

SheetSage2's MERT-v2 encoder, SheetSage2-Refined's beat and melody heads and their trackers on the CPU, in Rust, for
machines without a GPU. One pass of the encoder feeds both heads; no Python or PyTorch at run time.

## Use

```bash
python -m sheetsage2_refined.export_cpu cpu.safetensors [--dtype f16|f32|i8] [--lora ADAPTER --lora-scale S]
cargo build --release
./target/release/sheetsage2-cpu --weights cpu.safetensors [--threads N] [--out-dir DIR] [--heads beat,melody] \
    [--tempo-weight W] [--logits] song.mp3 ...
```

writes `<out-dir>/<song>.beats.json` and `<song>.melody.json` (default `outputs/cpu/`) as
`python -m sheetsage2_refined.beats` / `.melody` do; `--logits` also writes the heads' logits
(`<song>.heads.safetensors`) and `--from-logits` runs only the trackers on such files.

`--threads` defaults to one per logical CPU; on a Ryzen try the number of physical cores (8 on a 5900HX) as well.
WAV, FLAC, MP3 and Ogg Vorbis are decoded by Symphonia and resampled to 24 kHz as `torchaudio.functional.resample`
does; a `.f32` file is taken as raw float32 samples, mono at 24 kHz.

## How it works

* `export_cpu` folds every constant it can: LayerNorm scales and shifts into the linear layer that follows, the
  feed-forward `0.5`, attention's `1/sqrt(64)`, the downsampling LayerNorm into the stride-2 convolution, GRN's shift
  into `fc2`'s bias. q, k and v are one matrix. Linear weights are stored in 16-column panels, the order the kernel
  reads them, as float16, float32 or int8 (scale per output channel), and the file is memory-mapped.
* Matrix products (`src/gemm.rs`) are BLIS-style: rows in parallel blocks, packed into 6-row panels per slice of `k`,
  each weight panel converted to float32 once per block and swept by a 6x16 AVX2 micro-kernel (12 accumulators)
  that adds straight into the output. `llvm-mca -mcpu=znver3` puts its inner loop at 6 cycles for 12 FMAs: Zen 3's
  FMA peak.
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

On the 4-vCPU cloud machine it was developed on (Intel Xeon, AVX2 code path only, timings vary by 5-10% from run to
run), float16 weights, a 300 s song:

| | time | peak memory |
|---|---|---|
| this engine: encoder + both heads + both trackers | 69 s + 1.4 s | 1.5 GB |
| PyTorch, AVX2 only (as on a Zen 3): encoder + beat head | 70 s, again for the melody head | 5.5 GB |
| PyTorch with AVX-512: encoder + beat head | 49 s, again for the melody head | 5.5 GB |

Per encoder pass the engine is about as fast as PyTorch's MKL / oneDNN kernels on the same instruction set; it is
faster when both heads are needed and uses a quarter of the memory. Of a 300 s run: feed-forward 26 s, attention
32 s (25 s of it the scores, softmax and weighted values, at about 80% of the FMA peak), convolution module 6 s,
ConvNeXt stages 5 s, heads 3 s (`SHEETSAGE_PROFILE=1` prints this).

A Ryzen 9 5900HX (8 Zen 3 cores, 2 x 256-bit FMA each) has about three times the FMA throughput of this machine, so
a 5-minute song should take 20-30 s there; measure it with:

```bash
cargo run --release --example bench -- --threads 8 --tune     # kernel peak, linear layers, best SHEETSAGE_ROWS / SHEETSAGE_KC
cargo run --release --example attention -- 7500 8 --sharp 20  # the attention core on a 300 s window
SHEETSAGE_PROFILE=1 ./target/release/sheetsage2-cpu --threads 8 --weights cpu.safetensors song.wav
```
