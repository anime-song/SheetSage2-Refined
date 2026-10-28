"""Check the engine's section and key models (Rust) against the Python ones on the same logits and states.

    python rust/scripts/check_bars.py --weights cpu.safetensors [--model REPO_OR_FOLDER] LOGITS.heads.safetensors ...

Each file (``sheetsage2-cpu --heads section,key --logits``: the beat, melody and chord heads' logits and the encoder
states' block means) goes through ``sheetsage2-cpu --from-logits --heads section,key`` and through
``HeadSections.sections`` / ``HeadKeys.keys`` (the model folder's section and key models and decoding); the sections
and keys must be equal.
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sheetsage2_refined import cpu  # noqa: E402
from sheetsage2_refined.beats import HeadBeats  # noqa: E402
from sheetsage2_refined.keys import KEYS, HeadKeys, key_files, load_keys  # noqa: E402
from sheetsage2_refined.pipeline import DEFAULT_MODEL, mode_bias, model_folder  # noqa: E402
from sheetsage2_refined.sections import BLOCKS, WIDTH, HeadSections, load_sections  # noqa: E402
from sheetsage2_refined.theory import MODES  # noqa: E402


def frames(path: Path) -> dict:
    """``HeadSections.frames`` (with the chord head) from an engine's logits file."""

    heads, duration = cpu.read_outputs(path)
    with safe_open(str(path), "np") as f:
        states = torch.from_numpy(f.get_tensor("encoder.states")).reshape(-1, len(BLOCKS), WIDTH)
    melody = {k: torch.from_numpy(v) for k, v in heads["melody"].items()}
    pitch = torch.softmax(melody["melody_pitch"], -1)
    sung = (1 - pitch[:, 0]) * torch.sigmoid(melody["melody_vocal"])
    chord = {k: torch.from_numpy(v) for k, v in heads["chord"].items()}
    chord = torch.cat([torch.softmax(chord["root"], -1), torch.softmax(chord["bass"], -1), torch.sigmoid(chord["tones"])], -1)
    return {"probs": cpu.beat_probabilities(heads["beat"]), "states": states, "pitch": pitch.numpy(),
            "sung": sung.numpy(), "duration": duration, "chord": chord}  # fmt: skip


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("logits", type=Path, nargs="+")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="the model folder the weights were exported from")
    parser.add_argument("--binary", type=Path, default=cpu.DEFAULT_BINARY)
    args = parser.parse_args()
    out = Path(tempfile.mkdtemp())
    subprocess.run([str(args.binary), "--weights", str(args.weights), "--out-dir", str(out), "--from-logits",
                    "--heads", "section,key", *map(str, args.logits)], check=True, capture_output=True)  # fmt: skip
    folder = model_folder(args.model, None, key_files)
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    device = torch.device("cpu")
    model, priors = load_sections(folder / config["section_tracker"]["head"])
    sections = HeadSections(HeadBeats(None, None, device), None, model, priors, device,
                            **config["section_tracker"].get("decoding", {}))  # fmt: skip
    decoding = config["key_tracker"].get("decoding") or {}
    keys = load_keys(folder / config["key_tracker"]["head"])
    if decoding.get("local_weight") is not None:
        keys.config["local_weight"] = decoding["local_weight"]
    per_mode = mode_bias(decoding.get("mode_tau", 0.0), config.get("mode_counts"))
    bias = None if per_mode is None else np.array([per_mode[MODES.index(mode)] for _, mode in KEYS])
    keys = HeadKeys(sections, None, keys, device, decoding.get("boundary_weight", 1.0), bias)
    failures = 0
    for path in args.logits:
        stem = path.name.removesuffix(".heads.safetensors")
        f = frames(path)
        with torch.no_grad():
            want = {"sections": sections.sections(f)["sections"], "keys": keys.keys(f)["keys"]}
        got = {name: json.loads((out / f"{stem}.{name}.json").read_text())[name] for name in want}
        same = {name: want[name] == got[name] for name in want}
        failures += not all(same.values())
        print(f"{stem}: {len(want['sections'])} sections, {len(want['keys'])} keys: "
              f"{'equal' if all(same.values()) else 'DIFFERENT ' + str(same)}")  # fmt: skip
        for name in want:
            if not same[name]:
                print(f"  python {want[name]}\n  rust   {got[name]}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
