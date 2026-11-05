# SPDX-License-Identifier: Apache-2.0. Copyright 2026 Toaster Club LLC.
"""Export LFM3-Omni's speech encoder (downsampler, LFM2 stack, projector) for public/engine/audio.js, byte-exact.

    python tools/audio_export.py --out work/audio --quant q8 q4g64

Writes <quant>.bin (every buffer of Audio.weightList(), back to back) and manifest_<quant>.json. Matrices as in
tools/vision_export.py; the small convolutions, norms and biases stay f32.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from vision_export import matrix  # noqa: E402

TYPES = ["conv", "conv", "attn", "conv", "conv", "attn", "conv", "conv", "attn", "conv", "conv", "attn", "conv", "conv",
         "attn", "attn", "attn"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="work/audio")
    ap.add_argument("--repo", default="LiquidAI/LFM3-Omni-2B", help="the Hugging Face repository (LiquidAI/d1-Omni-2B: the decision model)")
    ap.add_argument("--quant", nargs="+", default=["q8", "q4g64"])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    f = safe_open(hf_hub_download(args.repo, "model.safetensors"), "pt")
    g = lambda n: f.get_tensor(n)  # noqa: E731
    vec = lambda n: g(n).float().contiguous().numpy().astype(np.float32).tobytes()  # noqa: E731
    DS, E, PJ = "model.audio_encoder.downsample.", "model.audio_encoder.lfm.", "model.audio_projector."
    for quant in args.quant:
        entries = []

        def put(key, b):
            entries.append((key, b))

        def mat(key, w):
            q, s = matrix(w, quant)
            put(key + ".q", q)
            put(key + ".s", s)

        for key, name in [("convIn", "conv_in"), ("convMid", "conv_mid"), ("linMid", "lin_mid"), ("convEnd", "conv_end"),
                          ("linEnd", "lin_end")]:
            put(key + "W", vec(DS + name + ".weight"))
            put(key + "B", vec(DS + name + ".bias"))
        mat("linOut", g(DS + "lin_out.weight").reshape(768, -1))
        put("linOutB", vec(DS + "lin_out.bias"))
        for i, t in enumerate(TYPES):
            p = f"{E}layers.{i}."
            put(f"{i}.opNorm", vec(p + "operator_norm.weight"))
            put(f"{i}.ffnNorm", vec(p + "ffn_norm.weight"))
            if t == "conv":
                put(f"{i}.convW", vec(p + "conv.conv.weight"))
                mat(f"{i}.inProj", g(p + "conv.in_proj.weight"))
                mat(f"{i}.outProj", g(p + "conv.out_proj.weight"))
            else:
                a = p + "self_attn."
                put(f"{i}.qNorm", vec(a + "q_layernorm.weight"))
                put(f"{i}.kNorm", vec(a + "k_layernorm.weight"))
                mat(f"{i}.qkv", torch.cat([g(a + "q_proj.weight"), g(a + "k_proj.weight"), g(a + "v_proj.weight")]))
                mat(f"{i}.oProj", g(a + "out_proj.weight"))
            mat(f"{i}.w13", torch.cat([g(p + "feed_forward.w1.weight"), g(p + "feed_forward.w3.weight")]))
            mat(f"{i}.w2", g(p + "feed_forward.w2.weight"))
        put("finalNorm", vec(E + "embedding_norm.weight"))
        put("ln.w", vec(PJ + "layer_norm.weight"))
        put("ln.b", vec(PJ + "layer_norm.bias"))
        mat("l1", g(PJ + "linear_1.weight"))
        put("l1.b", vec(PJ + "linear_1.bias"))
        mat("l2", g(PJ + "linear_2.weight"))
        put("l2.b", vec(PJ + "linear_2.bias"))
        with open(os.path.join(args.out, f"{quant}.bin"), "wb") as o:
            for _, b in entries:
                o.write(b)
        man = {"format": quant, "part": "audio", "tensors": [{"key": k, "bytes": len(b)} for k, b in entries],
               "total_bytes": sum(len(b) for _, b in entries)}
        with open(os.path.join(args.out, f"manifest_{quant}.json"), "w") as o:
            json.dump(man, o, indent=1)
        print("AUDIO_EXPORT", quant, round(man["total_bytes"] / 2 ** 20, 1), "MiB", len(entries), "tensors", flush=True)


if __name__ == "__main__":
    main()
