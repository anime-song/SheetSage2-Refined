"""Check the CPU engine against PyTorch on one song: the heads' logits, then the beats and notes the trackers make.

    python rust/scripts/check.py song.wav --weights cpu.safetensors --model MODEL_FOLDER [--threads N] [--keep DIR]

PyTorch runs SheetSage2's encoder and the model's beat and melody heads in float32 on the CPU
(``beats.frame_outputs``); the engine gets the very same 24 kHz waveform as a ``.f32`` file, so only the engine's
arithmetic differs.  Prints the largest logit differences per output, the agreement of the tracked beats, downbeats
(F, +-70 ms) and notes (onset +-50 ms and pitch), and both run times.
"""

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sheetsage2_refined import cpu  # noqa: E402
from sheetsage2_refined.beats import frame_outputs, load_encoder  # noqa: E402
from sheetsage2_refined.heads import load_head  # noqa: E402
from sheetsage2_refined.transcribe import load_waveform  # noqa: E402


def f_measure(ref, est, tolerance):
    ref, est = np.asarray(ref), np.asarray(est)
    if not len(ref) or not len(est):
        return float(len(ref) == len(est))
    used, hits = np.zeros(len(est), bool), 0
    for r in ref:
        d = np.abs(est - r)
        d[used] = np.inf
        j = int(np.argmin(d))
        if d[j] <= tolerance:
            used[j], hits = True, hits + 1
    return 2 * hits / (len(ref) + len(est))


def note_f(ref, est):
    used, hits = set(), 0
    for r in ref:
        for j, e in enumerate(est):
            if j not in used and e["pitch"] == r["pitch"] and abs(e["start"] - r["start"]) <= 0.05:
                used.add(j)
                hits += 1
                break
    return 2 * hits / max(1, len(ref) + len(est))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True, help="the model folder export_cpu read")
    parser.add_argument("--binary", type=Path, default=cpu.DEFAULT_BINARY)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--keep", type=Path, help="keep the reference logits here (reused when present)")
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    wave = load_waveform(args.audio)
    tmp = Path(tempfile.mkdtemp())
    raw = tmp / f"{args.audio.stem}.f32"
    wave.numpy().astype("<f4").tofile(raw)

    reference = (args.keep / f"{args.audio.stem}.torch.safetensors") if args.keep else tmp / "torch.safetensors"
    if not reference.exists():
        config = json.loads((args.model / "config.json").read_text())
        encoder = load_encoder(args.model / config["base_weights"])
        tensors, started = {}, time.perf_counter()
        for name, key in (("beat", "beat_tracker"), ("melody", "melody_tracker")):
            head = load_head(args.model / config[key]["head"])
            with torch.no_grad():
                out = frame_outputs(encoder, head, wave, torch.device("cpu"))
            layers = {"binary": torch.stack([out[n] for n in ("chord_boundary", "key_boundary", "beat", "downbeat", "group")], -1)}
            for layer, names in (("offset", ("beat_offset", "downbeat_offset")),):
                if names[0] in out:
                    layers[layer] = torch.stack([out[n] for n in names], -1)
            if "bar_beats" in out:
                layers["bar"] = torch.cat([out[n] for n in ("bar_beats", "beats_left", "bar_groups")], -1)
            if "bar_unit" in out:
                layers["unit"] = out["bar_unit"]
            if "melody_pitch" in out:
                layers["line"] = torch.cat([out[n] for n in ("melody_pitch", "note_steps", "steps_left", "onset_pos")]
                                           + [out[n][:, None] for n in ("melody_on", "melody_offset", "melody_vocal")], -1)  # fmt: skip
            tensors |= {f"{name}.{k}": v.contiguous().numpy() for k, v in layers.items()}
        print(f"torch (encoder twice, one per head): {time.perf_counter() - started:.1f} s")
        reference.parent.mkdir(parents=True, exist_ok=True)
        save_file(tensors, str(reference), metadata={"duration": str(len(wave) / 24000)})

    command = [str(args.binary), "--weights", str(args.weights), "--out-dir", str(tmp)]
    if args.threads:
        command += ["--threads", str(args.threads)]
    subprocess.run(command + [str(raw)], check=True)
    engine, _ = cpu.read_outputs(tmp / f"{args.audio.stem}.heads.safetensors")
    torch_out, _ = cpu.read_outputs(reference)
    for head in ("beat", "melody"):
        for name in torch_out[head]:
            a, b = torch_out[head][name], engine[head][name]
            if a.shape != b.shape:
                print(f"{head}.{name}: shapes {a.shape} vs {b.shape}")
                continue
            print(f"{head}.{name:16s} max |diff| {np.abs(a - b).max():.2e}  (logit range {np.abs(a).max():.1f})")
    beats, melody = cpu.trackers(args.weights)
    tb = beats.track_probabilities(cpu.beat_probabilities(torch_out["beat"]))
    eb = beats.track_probabilities(cpu.beat_probabilities(engine["beat"]))
    tn = melody.notes(cpu.melody_probabilities(torch_out["melody"]))
    en = melody.notes(cpu.melody_probabilities(engine["melody"]))
    print(f"beats F {f_measure(tb['beats'], eb['beats'], 0.07):.4f} ({len(tb['beats'])} / {len(eb['beats'])}), "
          f"downbeats F {f_measure(tb['downbeats'], eb['downbeats'], 0.07):.4f}, "
          f"meters equal {[b.get('meter') for b in tb['bars']] == [b.get('meter') for b in eb['bars']]}, "
          f"notes F {note_f(tn, en):.4f} ({len(tn)} / {len(en)})")  # fmt: skip
    truth = args.audio.with_suffix(".truth.json")
    if truth.exists():
        t = json.loads(truth.read_text())
        for label, b, n in (("torch", tb, tn), ("engine", eb, en)):
            print(f"vs the synthetic truth, {label}: beats F {f_measure(t['beats'], b['beats'], 0.07):.3f}, "
                  f"downbeats F {f_measure(t['downbeats'], b['downbeats'], 0.07):.3f}, notes F {note_f(t['notes'], n):.3f}")


if __name__ == "__main__":
    main()
