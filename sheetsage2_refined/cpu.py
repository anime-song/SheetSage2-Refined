"""Beats and melody from the Rust CPU engine (``rust/``): it runs the encoder and the heads, this module the trackers.

    python -m sheetsage2_refined.cpu song.mp3 [more.wav ...] --weights cpu.safetensors [--binary PATH] [--threads N]
                                     [--out-dir DIR]

writes ``<out-dir>/<song>.beats.json`` and ``<song>.melody.json`` as ``beats`` and ``melody`` do.  ``cpu.safetensors``
comes from ``python -m sheetsage2_refined.export_cpu``; the engine is built with ``cargo build --release`` in ``rust/``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
from safetensors import safe_open

from .beats import HeadBeats
from .heads import BAR, BINARY, LINE, MELODY_KEYS, OFFSETS
from .melody import HeadMelody

DEFAULT_BINARY = Path(__file__).resolve().parent.parent / "rust" / "target" / "release" / "sheetsage2-cpu"


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def head_outputs(layers: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """A head's output layers (``binary``, ``bar``, ``line``, ...: logits ``[frames, width]``) under the names
    ``heads.FrameHead.forward`` gives them."""

    out = {}
    if "binary" in layers:
        out |= {name: layers["binary"][:, i] for i, name in enumerate(BINARY[: layers["binary"].shape[1]])}
    if "bar" in layers:
        out |= dict(zip(BAR, np.split(layers["bar"], np.cumsum(list(BAR.values()))[:-1], axis=-1)))
    if "unit" in layers:
        out["bar_unit"] = layers["unit"]
    if "offset" in layers:
        out |= {name: layers["offset"][:, i] for i, name in enumerate(OFFSETS)}
    if "melody" in layers:
        m = layers["melody"]
        out |= {"melody": m[:, :MELODY_KEYS], "melody_on": m[:, MELODY_KEYS : 2 * MELODY_KEYS],
                "melody_offset": m[:, -2], "melody_vocal": m[:, -1]}  # fmt: skip
    if "line" in layers:
        line = layers["line"]
        out |= dict(zip(LINE, np.split(line[:, :-3], np.cumsum(list(LINE.values()))[:-1], axis=-1)))
        out |= {"melody_on": line[:, -3], "melody_offset": line[:, -2], "melody_vocal": line[:, -1]}
    return out


def read_outputs(path: Path) -> tuple[dict[str, dict[str, np.ndarray]], float]:
    """The engine's file: ``{head: {output: logits}}`` and the audio's duration in seconds."""

    heads: dict[str, dict[str, np.ndarray]] = {}
    with safe_open(str(path), "np") as f:
        duration = float(f.metadata()["duration"])
        for key in f.keys():
            head, layer = key.split(".", 1)
            heads.setdefault(head, {})[layer] = f.get_tensor(key)
    return {head: head_outputs(layers) for head, layers in heads.items()}, duration


def beat_probabilities(out: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """As ``HeadBeats.probabilities``."""

    probs = {k: _sigmoid(out[k]) for k in ("beat", "downbeat")}
    probs |= {k: _softmax(out[k]) for k in ("bar_beats", "beats_left", "bar_unit") if k in out}
    return probs | {k: out[k] for k in OFFSETS if k in out}


def melody_probabilities(out: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """As ``HeadMelody.probabilities``."""

    probs = {k: _sigmoid(out[k]) for k in ("melody", "melody_on", "melody_vocal") if k in out}
    probs |= {k: _softmax(out[k]) for k in LINE if k in out}
    return probs | {"melody_offset": out["melody_offset"]}


def trackers(weights: Path, tempo_weight: float = 4.0) -> tuple[HeadBeats, HeadMelody]:
    """The beat and melody trackers without their encoder and heads (the engine runs those), set up as the
    exported model's (``melody_tracker.decoding``)."""

    with safe_open(str(weights), "np") as f:
        meta = f.metadata()
    heads = json.loads(meta["heads"])
    decoding = json.loads(meta.get("melody_decoding", "{}"))
    beats = HeadBeats(None, None, None, tempo_weight=tempo_weight)
    melody = HeadMelody(None, None, None, line="line" in heads["melody"]["outputs"], **decoding)
    return beats, melody


def run_engine(audio: list[Path], weights: Path, out_dir: Path, binary: Path = DEFAULT_BINARY, threads: int | None = None) -> None:
    command = [str(binary), "--weights", str(weights), "--out-dir", str(out_dir)]
    if threads:
        command += ["--threads", str(threads)]
    subprocess.run(command + [str(a) for a in audio], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio", type=Path, nargs="+")
    parser.add_argument("--weights", type=Path, required=True, help="the engine's weights (export_cpu)")
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--threads", type=int, help="engine threads (default: one per logical CPU)")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs") / "cpu")
    args = parser.parse_args()
    beats, melody = trackers(args.weights)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        run_engine(args.audio, args.weights, Path(tmp), args.binary, args.threads)
        for audio in args.audio:
            started = time.perf_counter()
            heads, duration = read_outputs(Path(tmp) / f"{audio.stem}.heads.safetensors")
            result = {"audio": str(audio), "duration": duration}
            beat = result | beats.track_probabilities(beat_probabilities(heads["beat"]))
            notes = result | {"notes": melody.notes(melody_probabilities(heads["melody"]))}
            (args.out_dir / f"{audio.stem}.beats.json").write_text(json.dumps(beat, ensure_ascii=False), encoding="utf-8")
            (args.out_dir / f"{audio.stem}.melody.json").write_text(json.dumps(notes, ensure_ascii=False), encoding="utf-8")
            print(f"{audio.name}: {len(beat['beats'])} beats, {len(beat['bars'])} bars, {len(notes['notes'])} notes, "
                  f"trackers {time.perf_counter() - started:.1f} s -> {args.out_dir}", flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
