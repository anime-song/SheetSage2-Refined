"""Export SheetSage2's encoder, the beat / melody heads and the section / key models (with the chord head the keys
read) for the Rust CPU engine (``rust/``).

    python -m sheetsage2_refined.export_cpu out.safetensors [--model REPO_OR_FOLDER] [--dtype f16|f32|i8]
                                            [--lora LORA.safetensors --lora-scale S]

writes one safetensors file the engine memory-maps.  Everything the engine would otherwise compute once per run is
folded here:

* a LayerNorm's scale and shift go into the linear layer it feeds (the engine only normalises),
* the Conformer's half-step ``0.5 * ffn`` goes into ``fc2``, attention's ``1/sqrt(head_dim)`` into ``q``,
* a ConvNeXt stage's downsampling LayerNorm goes into its stride-2 convolution, written as one linear layer over
  pairs of frames, and GRN's shift ``beta`` into ``fc2``'s bias (its per-channel scale depends on the song and is
  applied by the engine),
* the rotary tables (bfloat16-rounded, as ``encoder.rotary_tables``) are stored for the longest window,
* each head's layer mixes are stored as softmax weights, and its LayerNorm goes into its first linear layer,
* the section and key models' per-beat inputs (the three state groups' LayerNorms and linear layers, the melody's and
  the chord's) become one linear layer, and their attention blocks' and final LayerNorms go into the layers they
  feed (``--no-bar-models`` leaves the section and key models and the chord head out).

Linear weights are stored in panels of 16 output channels, ``[ceil(N / 16), K, 16]``, the order the engine's matrix
kernels read them in, as float16 (default), float32 or int8 with a scale per output channel.  The heads and the
section and key models stay float32.

``--lora`` merges a LoRA adapter into the encoder before folding (``W += scale * B @ A``), with keys named
``<module>.lora_A.weight`` / ``<module>.lora_B.weight`` as PEFT writes them, ``<module>`` an encoder module name such
as ``layers.3.attn.q`` (an ``encoder.`` or PEFT ``base_model.model.`` prefix is dropped).

``--adapter`` instead writes only the adapter, for the engine to apply at run time next to unchanged base weights
(``sheetsage2-cpu --adapter``), a few MiB in place of a new 1.2 GiB export:

    python -m sheetsage2_refined.export_cpu out.lora.safetensors --adapter LORA.safetensors [--lora-scale S]

The adapter is folded like the weights it changes: per engine linear layer ``name`` it holds ``name.a`` ``[R, K]``
(the norm's scale folded in), ``name.b.<row>`` ``[n, r]`` (``scale``, the FFN's 0.5 and q's ``1/sqrt(head_dim)``
folded in) for outputs ``row .. row + n``, and ``name.db`` ``[N]`` (what the norm's shift and GRN's ``beta`` add
through the adapter), so the engine adds ``(x a^T) b^T + db`` to the layer's output.  q, k and v share one
``attn.qkv`` layer: each adapter is one part (``b.0``, ``b.1024``, ``b.2048``), their rows of ``a`` stacked in that
order.  ``a`` and ``b`` are float16, the way the engine keeps them.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.numpy import save_file
from safetensors.torch import load_file

from .beats import beat_files
from .encoder import EncoderConfig, rotary_tables
from .heads import BAR, BINARY, LINE, MELODY_KEYS, OFFSETS, UNITS
from .melody import melody_files
from .model import WINDOW_SECONDS
from .pipeline import DEFAULT_MODEL, model_folder

PANEL = 16  # output channels per weight panel (the engine's micro-kernel width)
FORMAT = "sheetsage2-cpu/1"
ADAPTER_FORMAT = "sheetsage2-cpu-lora/1"


def pack(weight: np.ndarray, dtype: str) -> dict[str, np.ndarray]:
    """``weight`` ``[N, K]`` as ``{"w": [ceil(N/16), K, 16]}`` panels in ``dtype`` (int8 adds ``"s"``, ``[N]``)."""

    n, k = weight.shape
    padded = np.zeros(((n + PANEL - 1) // PANEL * PANEL, k), dtype=np.float32)
    padded[:n] = weight
    out = {}
    if dtype == "i8":
        scale = np.abs(padded).max(1) / 127.0
        scale[scale == 0] = 1.0
        padded = np.clip(np.rint(padded / scale[:, None]), -127, 127).astype(np.int8)
        out["s"] = scale[:n].astype(np.float32)
    elif dtype == "f16":
        padded = padded.astype(np.float16)
    panels = padded.reshape(-1, PANEL, k).transpose(0, 2, 1)
    out["w"] = np.ascontiguousarray(panels)
    return out


class Writer:
    def __init__(self, dtype: str):
        self.dtype, self.tensors = dtype, {}

    def linear(self, name: str, weight, bias=None, dtype: str | None = None) -> None:
        weight = np.asarray(weight, dtype=np.float32)
        bias = np.zeros(weight.shape[0], np.float32) if bias is None else np.asarray(bias, dtype=np.float32)
        for key, value in pack(weight, dtype or self.dtype).items():
            self.tensors[f"{name}.{key}"] = value
        self.tensors[f"{name}.b"] = bias.astype(np.float32)

    def vector(self, name: str, value) -> None:
        self.tensors[name] = np.ascontiguousarray(np.asarray(value, dtype=np.float32))


def fold_norm(weight, bias, gamma, beta):
    """``W (gamma * x + beta) + b`` as ``W' x + b'``."""

    bias = np.zeros(weight.shape[0], np.float32) if bias is None else bias
    return weight * gamma[None, :], bias + weight @ beta


def read_lora(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """A PEFT LoRA file as ``{encoder module: (A, B)}``."""

    lora = {k: v.float().numpy() for k, v in load_file(str(path)).items()}
    pairs = {}
    for key in lora:
        match = re.fullmatch(r"(?:base_model\.model\.)?(?:encoder\.)?(.+)\.lora_A(?:\.default)?\.weight", key)
        if match:
            pairs[match.group(1)] = (lora[key], lora[key.replace("lora_A", "lora_B")])
    if not pairs:
        raise ValueError(f"no lora_A / lora_B pairs in {path}")
    return pairs


def merge_lora(state: dict[str, np.ndarray], path: Path, scale: float) -> int:
    pairs = read_lora(path)
    for module, (down, up) in pairs.items():
        target = f"{module}.weight"
        if target not in state:
            raise KeyError(f"LoRA module {module} is not in the encoder")
        shape = state[target].shape
        delta = scale * up.reshape(shape[0], -1) @ down.reshape(len(down), -1)
        state[target] = (state[target].reshape(shape[0], -1) + delta).reshape(shape)
    return len(pairs)


def adapter_part(module: str, down: np.ndarray, up: np.ndarray, scale: float, s: dict[str, np.ndarray],
                 config: EncoderConfig) -> tuple[str, int, int, np.ndarray, np.ndarray, np.ndarray]:  # fmt: skip
    """One LoRA module as a part of an engine linear layer: ``(name, outputs, first output row, a, b, db)``."""

    rank = len(down)
    a, b = down.reshape(rank, -1), scale * up.reshape(len(up), -1)
    gamma = beta = None
    shift = None  # a vector the layer's input carries on top of ``a``'s input (GRN's beta)
    row, n = 0, len(b)
    if m := re.fullmatch(r"stages\.(\d+)\.down_conv", module):
        p = f"stages.{m.group(1)}"
        conv = a.reshape(rank, -1, 2)  # [r, in, 2] -> [r, 2 * in] over (frame 2t, frame 2t + 1), as the weights
        a = np.concatenate([conv[:, :, 0], conv[:, :, 1]], axis=1)
        gamma, beta = np.tile(s[f"{p}.down_norm.weight"], 2), np.tile(s[f"{p}.down_norm.bias"], 2)
        name = f"{p}.down"
    elif m := re.fullmatch(r"(stages\.\d+\.layers\.\d+)\.(fc1|fc2)", module):
        p, layer = m.groups()
        if layer == "fc1":
            gamma, beta = s[f"{p}.norm.weight"], s[f"{p}.norm.bias"]
        else:
            shift = s[f"{p}.grn_beta"]
        name = module
    elif m := re.fullmatch(r"(layers\.\d+)\.(ffn1|ffn2)\.(fc1|fc2)", module):
        p, ffn, layer = m.groups()
        if layer == "fc1":
            gamma, beta = s[f"{p}.{ffn}_norm.weight"], s[f"{p}.{ffn}_norm.bias"]
        else:
            b = 0.5 * b
        name = module
    elif m := re.fullmatch(r"(layers\.\d+)\.attn\.([qkv])", module):
        p, part = m.groups()
        gamma, beta = s[f"{p}.attn_norm.weight"], s[f"{p}.attn_norm.bias"]
        if part == "q":
            b = b / np.sqrt(config.width // config.heads)
        row, n = "qkv".index(part) * config.width, 3 * config.width
        name = f"{p}.attn.qkv"
    elif m := re.fullmatch(r"(layers\.\d+)\.attn\.out", module):
        name = module
    elif m := re.fullmatch(r"(layers\.\d+)\.conv\.pointwise_(in|out)", module):
        p, side = m.groups()
        if side == "in":
            gamma, beta = s[f"{p}.conv.norm.weight"], s[f"{p}.conv.norm.bias"]
        name = f"{p}.conv.pw_{side}"
    else:
        raise ValueError(f"LoRA module {module}: the engine applies adapters to linear layers only (merge it instead)")
    db = np.zeros(len(b), np.float32)
    if beta is not None:
        db = b @ (a @ beta)
        a = a * gamma[None, :]
    if shift is not None:
        db = b @ (a @ shift)
    return name, n, row, a, b, db


def export_adapter(path: Path, lora: Path, scale: float, state: dict[str, np.ndarray], config: EncoderConfig,
                   model: str) -> None:  # fmt: skip
    """The adapter alone, folded per engine linear layer (``a``, ``b.<row>``, ``db``)."""

    layers: dict[str, list] = {}
    for module, (down, up) in read_lora(lora).items():
        if f"{module}.weight" not in state:
            raise KeyError(f"LoRA module {module} is not in the encoder")
        name, n, row, a, b, db = adapter_part(module, down, up, scale, state, config)
        layers.setdefault(name, []).append((n, row, a, b, db))
    tensors = {}
    for name, parts in layers.items():
        parts.sort(key=lambda part: part[1])
        db_all = np.zeros(parts[0][0], np.float32)
        for _, row, a, b, db in parts:
            tensors[f"{name}.b.{row}"] = np.ascontiguousarray(b, dtype=np.float16)
            db_all[row : row + len(b)] += db
        tensors[f"{name}.a"] = np.ascontiguousarray(np.concatenate([a for _, _, a, _, _ in parts]), dtype=np.float16)
        tensors[f"{name}.db"] = db_all
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path), metadata={"format": ADAPTER_FORMAT, "model": model, "scale": str(scale)})
    size = sum(v.nbytes for v in tensors.values())
    print(f"{path}: adapter for {len(layers)} layers, {size / 2**20:.1f} MiB")


def export_encoder(out: Writer, state: dict[str, np.ndarray], config: EncoderConfig) -> None:
    s = state
    for name in ("mel_fb", "mel_mean", "mel_std", "window"):
        out.vector(f"frontend.{name}", s[f"frontend.{name}"])
    channels = (config.n_mels, *config.stage_channels)
    for i, depth in enumerate(config.stage_depths):
        prefix = f"stages.{i}"
        if channels[i] != channels[i + 1] or config.stage_strides[i] > 1:
            if config.stage_strides[i] != 2:
                raise ValueError("the engine assumes stride-2 downsampling")
            conv = s[f"{prefix}.down_conv.weight"]  # [out, in, 2] -> [out, 2 * in] over (frame 2t, frame 2t + 1)
            weight = np.concatenate([conv[:, :, 0], conv[:, :, 1]], axis=1)
            gamma, beta = s[f"{prefix}.down_norm.weight"], s[f"{prefix}.down_norm.bias"]
            out.linear(f"{prefix}.down", *fold_norm(weight, s[f"{prefix}.down_conv.bias"], np.tile(gamma, 2), np.tile(beta, 2)))
        for j in range(depth):
            p = f"{prefix}.layers.{j}"
            out.vector(f"{p}.dw.w", s[f"{p}.depthwise.weight"][:, 0, :])  # [C, 7]
            out.vector(f"{p}.dw.b", s[f"{p}.depthwise.bias"])
            out.linear(f"{p}.fc1", *fold_norm(s[f"{p}.fc1.weight"], s[f"{p}.fc1.bias"], s[f"{p}.norm.weight"], s[f"{p}.norm.bias"]))
            out.vector(f"{p}.grn_gamma", s[f"{p}.grn_gamma"])
            fc2 = s[f"{p}.fc2.weight"]
            out.linear(f"{p}.fc2", fc2, s[f"{p}.fc2.bias"] + fc2 @ s[f"{p}.grn_beta"])
    head_dim = config.width // config.heads
    for i in range(config.layers):
        p = f"layers.{i}"
        for ffn in ("ffn1", "ffn2"):
            gamma, beta = s[f"{p}.{ffn}_norm.weight"], s[f"{p}.{ffn}_norm.bias"]
            out.linear(f"{p}.{ffn}.fc1", *fold_norm(s[f"{p}.{ffn}.fc1.weight"], s[f"{p}.{ffn}.fc1.bias"], gamma, beta))
            out.linear(f"{p}.{ffn}.fc2", 0.5 * s[f"{p}.{ffn}.fc2.weight"], 0.5 * s[f"{p}.{ffn}.fc2.bias"])
        gamma, beta = s[f"{p}.attn_norm.weight"], s[f"{p}.attn_norm.bias"]
        scale = {"q": 1.0 / np.sqrt(head_dim), "k": 1.0, "v": 1.0}
        weights, biases = zip(*(fold_norm(s[f"{p}.attn.{n}.weight"], s[f"{p}.attn.{n}.bias"], gamma, beta) for n in "qkv"))
        out.linear(f"{p}.attn.qkv", np.concatenate([w * scale[n] for n, w in zip("qkv", weights)]),
                   np.concatenate([b * scale[n] for n, b in zip("qkv", biases)]))  # fmt: skip
        out.linear(f"{p}.attn.out", s[f"{p}.attn.out.weight"], s[f"{p}.attn.out.bias"])
        gamma, beta = s[f"{p}.conv.norm.weight"], s[f"{p}.conv.norm.bias"]
        out.linear(f"{p}.conv.pw_in", *fold_norm(s[f"{p}.conv.pointwise_in.weight"][:, :, 0], None, gamma, beta))
        out.vector(f"{p}.conv.dw.w", s[f"{p}.conv.depthwise.weight"][:, 0, :])  # [C, 31]
        out.vector(f"{p}.conv.dw_norm.w", s[f"{p}.conv.depthwise_norm.weight"])
        out.vector(f"{p}.conv.dw_norm.b", s[f"{p}.conv.depthwise_norm.bias"])
        out.linear(f"{p}.conv.pw_out", s[f"{p}.conv.pointwise_out.weight"][:, :, 0])
        out.vector(f"{p}.final_norm.w", s[f"{p}.final_norm.weight"])
        out.vector(f"{p}.final_norm.b", s[f"{p}.final_norm.bias"])
    frames = round(WINDOW_SECONDS * config.sample_rate) // config.samples_per_frame + 1
    cos, sin = rotary_tables(frames, head_dim, config.rope_base, "cpu", torch.float32)
    out.vector("rope.cos", cos[0, :, 0, : head_dim // 2].numpy())  # [frames, head_dim / 2]; both halves are equal
    out.vector("rope.sin", sin[0, :, 0, : head_dim // 2].numpy())


HEAD_OUTPUTS = ("binary", "bar", "offset", "unit", "melody", "line")  # what the beat and melody trackers read
CHORD_OUTPUTS = ("root", "bass", "tones")  # what the key model reads of the chord head


def export_head(out: Writer, name: str, state: dict[str, np.ndarray], outputs=HEAD_OUTPUTS) -> dict:
    """A ``heads.FrameHead`` with only the output layers in ``outputs``.  A head with several layer mixes reads
    them side by side, each normalised by the one LayerNorm."""

    s = state
    if "extra.weight" in s:
        raise ValueError(f"{name}: heads that read a beat head's bar facts are not supported yet")
    p = f"heads.{name}"
    mix = torch.softmax(torch.from_numpy(s["mix"]).reshape(-1, s["mix"].shape[-1]), -1).numpy()
    out.vector(f"{p}.mix", mix)  # [mixes, 25]
    gamma, beta = (np.tile(s[f"norm.{k}"], len(mix)) for k in ("weight", "bias"))
    out.linear(f"{p}.inp", *fold_norm(s["inp.weight"], s["inp.bias"], gamma, beta), dtype="f32")
    for i in range(2):
        out.vector(f"{p}.convs.{i}.w", s[f"convs.{i}.weight"])  # [C, C / groups, 9]
        out.vector(f"{p}.convs.{i}.b", s[f"convs.{i}.bias"])
    layers = sum(1 for k in s if k.startswith("rnn.weight_ih_l") and not k.endswith("_reverse"))
    for layer in range(layers):
        for suffix, direction in (("", "fwd"), ("_reverse", "bwd")):
            q = f"{p}.rnn.{layer}.{direction}"
            out.linear(f"{q}.ih", s[f"rnn.weight_ih_l{layer}{suffix}"], s[f"rnn.bias_ih_l{layer}{suffix}"], dtype="f32")
            out.vector(f"{q}.hh.w", s[f"rnn.weight_hh_l{layer}{suffix}"])  # [3 H, H], gates r, z, n
            out.vector(f"{q}.hh.b", s[f"rnn.bias_hh_l{layer}{suffix}"])
    written = {}
    for layer in outputs:
        if f"{layer}.weight" in s:
            out.linear(f"{p}.out.{layer}", s[f"{layer}.weight"], s[f"{layer}.bias"], dtype="f32")
            written[layer] = int(s[f"{layer}.weight"].shape[0])
    return {"mixes": len(mix), "rnn_layers": layers, "hidden": int(s["inp.weight"].shape[0]), "outputs": written}


def export_conv_block(out: Writer, p: str, s: dict[str, np.ndarray], q: str) -> None:
    """A ``sections.ConvBlock`` (its LayerNorm stays: the depthwise convolution after it pads with zeros)."""

    out.vector(f"{p}.norm.w", s[f"{q}.norm.weight"])
    out.vector(f"{p}.norm.b", s[f"{q}.norm.bias"])
    out.vector(f"{p}.depth.w", s[f"{q}.depth.weight"][:, 0])  # [C, kernel]
    out.vector(f"{p}.depth.b", s[f"{q}.depth.bias"])
    out.linear(f"{p}.point", s[f"{q}.point.weight"], s[f"{q}.point.bias"], dtype="f32")
    out.linear(f"{p}.out", s[f"{q}.out.weight"], s[f"{q}.out.bias"], dtype="f32")


def export_bar_model(out: Writer, p: str, s: dict[str, np.ndarray], config: dict, outputs: tuple[str, ...]) -> None:
    """A ``sections.SectionModel`` or ``keys.KeyModel``.  The per-beat inputs are one linear layer over the three
    normalised state groups, the melody (4 x 13), the sung share and the chord (38, keys only), side by side; the
    pre-LayerNorms of the attention blocks and the final one go into the linear layers they feed."""

    groups = sum(1 for k in s if re.fullmatch(r"states\.\d+\.1\.weight", k))
    weights, bias = [], 0.0
    for g in range(groups):
        w, b = fold_norm(s[f"states.{g}.1.weight"], s[f"states.{g}.1.bias"], s[f"states.{g}.0.weight"], s[f"states.{g}.0.bias"])
        weights.append(w)
        bias = bias + b
    for name in ("melody", "chord"):
        if f"{name}.weight" in s:
            weights.append(s[f"{name}.weight"])
            bias = bias + s[f"{name}.bias"]
    out.linear(f"{p}.beat_in", np.concatenate(weights, 1), bias, dtype="f32")
    out.vector(f"{p}.beat_position", s["beat_position.weight"])
    out.vector(f"{p}.bar_beats", s["bar_beats.weight"])
    out.linear(f"{p}.song_position", s["song_position.weight"], s["song_position.bias"], dtype="f32")
    for i in range(config["beat_layers"]):
        export_conv_block(out, f"{p}.beat_layers.{i}", s, f"beat_layers.{i}")
    for i in range(config["bar_layers"]):
        q, r = f"bar_layers.{i}", f"{p}.bar_layers.{i}"
        out.linear(f"{r}.qkv", *fold_norm(s[f"{q}.qkv.weight"], s[f"{q}.qkv.bias"], s[f"{q}.norm1.weight"], s[f"{q}.norm1.bias"]), dtype="f32")
        out.linear(f"{r}.proj", s[f"{q}.proj.weight"], s[f"{q}.proj.bias"], dtype="f32")
        out.linear(f"{r}.ff1", *fold_norm(s[f"{q}.ff.0.weight"], s[f"{q}.ff.0.bias"], s[f"{q}.norm2.weight"], s[f"{q}.norm2.bias"]), dtype="f32")
        out.linear(f"{r}.ff2", s[f"{q}.ff.3.weight"], s[f"{q}.ff.3.bias"], dtype="f32")
    out.vector(f"{p}.distance", s["distance"])  # [layers, heads, 2 * relative + 1]
    if "similar" in s:
        out.vector(f"{p}.similar", s["similar"])  # [layers, heads, 2]: melody, states
    for name in outputs:
        out.linear(f"{p}.out.{name}", *fold_norm(s[f"{name}.weight"], s[f"{name}.bias"], s["norm.weight"], s["norm.bias"]), dtype="f32")
    if "local.input.weight" in s:
        out.linear(f"{p}.local.input", s["local.input.weight"], s["local.input.bias"], dtype="f32")
        out.vector(f"{p}.local.beat_position", s["local.beat_position.weight"])
        for kind in ("beat_layers", "bar_layers"):
            for i in range(sum(1 for k in s if re.fullmatch(rf"local\.{kind}\.\d+\.point\.weight", k))):
                export_conv_block(out, f"{p}.local.{kind}.{i}", s, f"local.{kind}.{i}")
        w, b = fold_norm(s["local.head.1.weight"], s["local.head.1.bias"], s["local.head.0.weight"], s["local.head.0.bias"])
        out.linear(f"{p}.local.head", w, b, dtype="f32")
    for name in ("first", "transition", "length", "last"):
        if f"priors.{name}" in s:
            out.vector(f"{p}.priors.{name}", s[f"priors.{name}"])


def bar_model_files(config: dict) -> list[str]:
    """The section and key models of a model folder and the chord head the keys read, those it has."""

    names = [(config.get(k) or {}).get("head") for k in ("section_tracker", "key_tracker")]
    if (config.get("key_tracker") or {}).get("head"):
        names.append(config["rescoring"]["quality"]["head"])
    return [n for n in names if n]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face repo id or local model folder")
    parser.add_argument("--revision")
    parser.add_argument("--dtype", choices=("f16", "f32", "i8"), default="f16", help="the encoder's linear weights")
    parser.add_argument("--lora", type=Path, help="a LoRA adapter for the encoder, merged before export")
    parser.add_argument("--lora-scale", type=float, default=1.0, help="the adapter's alpha / rank")
    parser.add_argument("--adapter", type=Path, help="write only this LoRA adapter, for the engine's --adapter")
    parser.add_argument("--no-bar-models", action="store_true", help="leave out the section and key models")
    args = parser.parse_args()

    folder = model_folder(args.model, args.revision,
                          lambda config: [*beat_files(config), *melody_files(config), *bar_model_files(config)])
    model = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    weights = load_file(str(folder / model["base_weights"]))
    state = {k[len("encoder."):]: v.float().numpy() for k, v in weights.items() if k.startswith("encoder.")}
    del weights
    config = EncoderConfig()
    if args.adapter:
        export_adapter(args.out, args.adapter, args.lora_scale, state, config, model.get("name", str(args.model)))
        return
    if args.lora:
        print(f"merged {merge_lora(state, args.lora, args.lora_scale)} LoRA modules")
    out = Writer(args.dtype)
    export_encoder(out, state, config)
    heads = {}
    for name, key in (("beat", "beat_tracker"), ("melody", "melody_tracker")):
        head = {k: v.float().numpy() for k, v in load_file(str(folder / model[key]["head"])).items()}
        heads[name] = export_head(out, name, head) | {"file": model[key]["head"]}
    bar_models = {}
    for name, key, outputs in (("sections", "section_tracker", ("label", "boundary", "phrase")),
                               ("keys", "key_tracker", ("key", "boundary"))):  # fmt: skip
        spec = model.get(key) or {}
        if not spec.get("head") or args.no_bar_models:
            continue
        if name == "keys":
            chord = model["rescoring"]["quality"]["head"]
            state = {k: v.float().numpy() for k, v in load_file(str(folder / chord)).items()}
            heads["chord"] = export_head(out, "chord", state, CHORD_OUTPUTS) | {"file": chord}
        with safe_open(str(folder / spec["head"]), "pt") as handle:
            config_ = json.loads(handle.metadata()["config"])
        if config_.get("sections"):
            raise ValueError(f"{spec['head']}: key models that read the section model are not supported")
        state = {k: v.float().numpy() for k, v in load_file(str(folder / spec["head"])).items()}
        export_bar_model(out, name, state, config_, outputs)
        bar_models[name] = {"file": spec["head"], "config": config_, "decoding": spec.get("decoding") or {}}
    metadata = {
        "format": FORMAT,
        "dtype": args.dtype,
        "model": model.get("name", str(args.model)),
        "heads": json.dumps(heads),
        "melody_decoding": json.dumps((model.get("melody_tracker") or {}).get("decoding") or {}),
        "bar_models": json.dumps(bar_models),
        "mode_counts": json.dumps(model.get("mode_counts") or []),
        "names": json.dumps({"binary": BINARY, "bar": BAR, "offset": OFFSETS, "unit": UNITS, "line": LINE,
                             "melody_keys": MELODY_KEYS}),  # fmt: skip
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_file(out.tensors, str(args.out), metadata=metadata)
    size = sum(v.nbytes for v in out.tensors.values())
    print(f"{args.out}: {len(out.tensors)} tensors, {size / 2**20:.0f} MiB ({args.dtype})")


if __name__ == "__main__":
    main()
