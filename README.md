# SheetSage2-Refined

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Model-SheetSage2--Refined-yellow)](https://huggingface.co/anime-song/SheetSage2-Refined)
[![Downloads](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fmodels%2Fanime-song%2FSheetSage2-Refined&query=%24.downloads&label=downloads&color=blue)](https://huggingface.co/anime-song/SheetSage2-Refined)
[![Base model](https://img.shields.io/badge/base-m--a--p%2FSheetSage2-lightgrey)](https://huggingface.co/m-a-p/SheetSage2)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Code: MIT](https://img.shields.io/badge/code-MIT-green)](LICENSE)
[![Weights: CC BY-NC 4.0](https://img.shields.io/badge/weights-CC%20BY--NC%204.0-orange)](https://creativecommons.org/licenses/by-nc/4.0/)

Transcribe whole songs into beats, downbeats and time signatures, melody, chords and keys.
A fine-tuned [SheetSage2](https://huggingface.co/m-a-p/SheetSage2); details and results are on the
[model card](https://huggingface.co/anime-song/SheetSage2-Refined).

## Results

SheetSage2 against SheetSage2-Refined on the same held-out songs; the largest gains are on bars and time
signatures. The beat tracker without the decoder ([below](#beats-downbeats-and-time-signatures-only-no-decoder))
gives beats and bars only. The evaluation sets and all scores are on the [model card](https://huggingface.co/anime-song/SheetSage2-Refined#results).

<!-- tables:summary -->
|  | SheetSage2 | SheetSage2-Refined | Beat tracker (no decoder) |
|---|---|---|---|
| Downbeat F, pop songs (mostly 4/4) | 0.966 | 0.965 | **0.976** |
| Downbeat F, odd-meter songs | 0.733 | 0.835 | **0.883** |
| Chords (tetrads), pop songs | 0.678 | **0.770** | — |
| Key, pop songs | 0.840 | **0.920** | — |
| Melody F, HookTheory | 0.664 | **0.682** | — |
<!-- /tables:summary -->

Bars whose two barlines are both found (±70 ms), by time signature, on 50 songs with odd meters and meter changes:

<!-- tables:summary-meters -->
| Bars found, odd-meter songs | SheetSage2 | SheetSage2-Refined | Beat tracker (no decoder) |
|---|---|---|---|
| 4/4 | 0.699 | 0.713 | **0.824** |
| 3/4 | 0.676 | 0.768 | 0.768 |
| 6/4 | 0.275 | 0.486 | **0.572** |
| 5/4 | 0.002 | 0.512 | **0.574** |
| 6/8 | 0.299 | 0.725 | **0.794** |
| 12/8 | 0.597 | **0.970** | 0.794 |
| 5/8 | 0.000 | 0.008 | **0.508** |
| 11/8 | 0.000 | **0.586** | 0.563 |
| 2/4 | 0.063 | **0.111** | 0.079 |
| 7/8 | 0.000 | 0.305 | **0.847** |
| 7/4 | 0.000 | 0.196 | **0.565** |
| **all** | 0.534 | 0.661 | **0.752** |
<!-- /tables:summary-meters -->

## Install

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install git+https://github.com/anime-song/SheetSage2-Refined
```

The weights (2.9 GB) are downloaded from Hugging Face on first use. A CUDA GPU is recommended.

## Usage

```bash
sheetsage2-refined song.mp3                 # -> outputs/transcriptions/song.json
sheetsage2-refined song.mp3 --key-mode      # keys with modes (dorian, mixolydian, ...)
python -m sheetsage2_refined.visualize outputs/transcriptions/song.json   # -> an HTML page with the audio
```

```python
from sheetsage2_refined import SheetSage2Refined, events_to_annotations

model = SheetSage2Refined.load()
events = model.transcribe(["song.mp3"])[0]
annotations = events_to_annotations(events, duration_seconds)
# annotations["beats"], ["downbeats"], ["chord_intervals"], ["chord_labels"], ["keys"]
```

### Beats, downbeats and time signatures only (no decoder)

A bar head on the same frozen encoder tracks beats, downbeats and each bar's time signature without the decoder,
in a few seconds per song on a GPU. Only SheetSage2's encoder weights and the 8 MB head are downloaded.

```bash
python -m sheetsage2_refined.beats song.mp3       # -> outputs/beats/song.beats.json
```

```python
from sheetsage2_refined.beats import HeadBeats

tracker = HeadBeats.load()
result = tracker.track("song.mp3")
result["beats"], result["downbeats"]   # times in seconds
result["bars"]                          # [{"start": 1.42, "beats": 7, "unit": 8, "meter": "7/8"}, ...]
```

Each bar's denominator is chosen so that the written tempo does not jump: the same beat keeps its denominator,
and a 7-beat bar after 4/4 at twice the beat speed is written 7/8, not 7/4.

## License

The code is MIT. The model weights are CC BY-NC 4.0 (non-commercial use only), as are SheetSage2 and
MERT-v2 they are based on.
