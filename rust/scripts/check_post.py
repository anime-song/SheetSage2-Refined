"""Check the engine's trackers (Rust) against the Python ones on the same logits.

    python rust/scripts/check_post.py --weights cpu.safetensors LOGITS.heads.safetensors ... [--noise S --copies N]

Each file (``sheetsage2-cpu --logits``, or ``check.py --keep``'s ``*.torch.safetensors``) goes through
``sheetsage2-cpu --from-logits`` and through ``sheetsage2_refined.cpu``'s trackers; the beats, downbeats, bars,
meters and notes must be equal.  ``--noise`` adds copies with Gaussian noise of that size on every logit, to try
more paths through the trackers' Viterbis.
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sheetsage2_refined import cpu  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("logits", type=Path, nargs="+")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--binary", type=Path, default=cpu.DEFAULT_BINARY)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--copies", type=int, default=3)
    parser.add_argument("--hold", type=float, default=0.0, help="the beat tracker's p_hold (fermatas)")
    args = parser.parse_args()
    tmp = Path(tempfile.mkdtemp())
    files = []
    for path in args.logits:
        with safe_open(str(path), "np") as f:
            meta, tensors = f.metadata(), {k: f.get_tensor(k) for k in f.keys()}
        stem = path.name.split(".")[0]
        files.append(tmp / f"{stem}.heads.safetensors")
        save_file(tensors, str(files[-1]), metadata=meta)
        rng = np.random.default_rng(0)
        for c in range(args.copies if args.noise else 0):
            noisy = {k: (v + args.noise * rng.standard_normal(v.shape)).astype(np.float32) for k, v in tensors.items()}
            files.append(tmp / f"{stem}-noise{c}.heads.safetensors")
            save_file(noisy, str(files[-1]), metadata=meta)
    out = tmp / "rust"
    subprocess.run([str(args.binary), "--weights", str(args.weights), "--out-dir", str(out), "--from-logits",
                    "--hold", str(args.hold), *map(str, files)], check=True, capture_output=True)  # fmt: skip
    beats, melody = cpu.trackers(args.weights, p_hold=args.hold)
    failures = 0
    for path in files:
        stem = path.name.removesuffix(".heads.safetensors")
        heads, duration = cpu.read_outputs(path)
        want_b = beats.track_probabilities(cpu.beat_probabilities(heads["beat"]))
        want_n = melody.notes(cpu.melody_probabilities(heads["melody"]))
        got_b = json.loads((out / f"{stem}.beats.json").read_text())
        got_n = json.loads((out / f"{stem}.melody.json").read_text())["notes"]
        same = {k: want_b[k] == got_b[k] for k in ("beats", "downbeats", "bars", "meters")} | {"notes": want_n == got_n}
        failures += not all(same.values())
        print(f"{stem}: {len(want_b['beats'])} beats, {len(want_b['bars'])} bars, meters {want_b['meters']}, "
              f"{len(want_n)} notes: {'equal' if all(same.values()) else 'DIFFERENT ' + str(same)}")  # fmt: skip
        if not same["notes"]:
            diff = next(i for i, (a, b) in enumerate(zip(want_n + [None], got_n + [None])) if a != b)
            print("  first differing note:", want_n[diff : diff + 1], got_n[diff : diff + 1])
        for k in ("beats", "downbeats"):
            if not same[k]:
                a, b = np.asarray(want_b[k]), np.asarray(got_b[k])
                print(f"  {k}: {len(a)} vs {len(b)}", np.setdiff1d(a, b)[:5], np.setdiff1d(b, a)[:5])
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
