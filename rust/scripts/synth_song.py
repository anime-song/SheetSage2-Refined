"""A synthetic song for testing the CPU engine without real audio: drums, bass, chords and a sung-like melody on a
beat grid with meter changes.  Not music anyone would enjoy, but it has clear beats, bars and a melody line.

    python rust/scripts/synth_song.py out.wav [--seconds 330] [--seed 0] [--rate 44100]

writes the WAV (16-bit) and ``out.truth.json`` with its beats, downbeats and melody notes.
"""

import argparse
import json
import wave
from pathlib import Path

import numpy as np

SCALE = np.array([0, 2, 4, 7, 9])  # major pentatonic
PROGRESSIONS = [[0, 5, 3, 4], [0, 3, 4, 4], [5, 3, 0, 4]]
DEGREES = [0, 2, 4, 5, 7, 9, 11]


def tone(freq, length, rate, harmonics=(1.0,), vibrato=0.0, attack=0.01, release=0.05):
    t = np.arange(int(length * rate)) / rate
    phase = 2 * np.pi * np.cumsum(freq * (1 + vibrato * np.sin(2 * np.pi * 5.5 * t))) / rate
    y = sum(a * np.sin((k + 1) * phase) for k, a in enumerate(harmonics))
    env = np.minimum(1.0, t / attack) * np.minimum(1.0, (length - t) / release).clip(0)
    return y * env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("--seconds", type=float, default=330.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rate", type=int, default=44100)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    rate, total = args.rate, args.seconds
    out = np.zeros(int((total + 2) * rate))
    noise = rng.standard_normal(rate)

    def put(signal, start, gain):
        i = int(start * rate)
        n = min(len(signal), len(out) - i)
        if n > 0:
            out[i : i + n] += gain * signal[:n]

    beats, downbeats, notes = [], [], []
    t, key = 0.5, 0
    while t < total - 4:
        bpm = rng.uniform(85, 150)
        beat = 60 / bpm
        meter = rng.choice([4, 4, 4, 3, 6, 5, 7])
        progression = PROGRESSIONS[rng.integers(len(PROGRESSIONS))]
        for bar in range(int(rng.integers(8, 17))):
            if t > total - 4:
                break
            root = DEGREES[progression[bar % 4]] + key
            downbeats.append(t)
            chord = [root, root + (3 if progression[bar % 4] in (1, 2, 5) else 4), root + 7]
            for c in chord:
                put(tone(220 * 2 ** (c / 12), meter * beat, rate, (0.5, 0.25, 0.12), attack=0.05, release=0.2), t, 0.05)
            for b in range(meter):
                bt = t + b * beat
                beats.append(bt)
                kick = tone(55, 0.25, rate, attack=0.002, release=0.2) * np.exp(-np.arange(int(0.25 * rate)) / rate * 18)
                if b == 0 or (meter == 4 and b == 2):
                    put(kick, bt, 0.9)
                if (meter in (4, 6) and b % 2 == 1) or (meter in (3, 5, 7) and b in (meter - 1, 2)):
                    put(noise[: int(0.15 * rate)] * np.exp(-np.arange(int(0.15 * rate)) / rate * 25), bt, 0.4)
                for h in range(2):
                    put(noise[: int(0.04 * rate)] * np.exp(-np.arange(int(0.04 * rate)) / rate * 120), bt + h * beat / 2, 0.15)
                put(tone(55 * 2 ** (root / 12), beat * 0.9, rate, (1.0, 0.3)), bt, 0.35)
            pos = 0.0
            while pos < meter - 0.01:
                length = min(rng.choice([0.5, 1.0, 1.0, 1.5, 2.0]), meter - pos)
                if rng.random() < 0.85:
                    pitch = 60 + key + SCALE[rng.integers(5)] + 12 * rng.integers(0, 2)
                    start = t + pos * beat
                    put(tone(440 * 2 ** ((pitch - 69) / 12), length * beat * 0.95, rate, (1.0, 0.6, 0.3, 0.2, 0.1), vibrato=0.006), start, 0.25)
                    notes.append({"start": round(start, 4), "end": round(start + length * beat * 0.95, 4), "pitch": int(pitch)})
                pos += length
            t += meter * beat
        key = (key + int(rng.choice([0, 2, 5, 7]))) % 12
    out = out[: int(total * rate)]
    out = 0.9 * out / np.abs(out).max()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(args.out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((out * 32767).astype("<i2").tobytes())
    truth = {"beats": [round(b, 4) for b in beats], "downbeats": [round(d, 4) for d in downbeats], "notes": notes}
    args.out.with_suffix(".truth.json").write_text(json.dumps(truth))
    print(f"{args.out}: {total:.0f} s, {len(beats)} beats, {len(downbeats)} bars, {len(notes)} notes")


if __name__ == "__main__":
    main()
