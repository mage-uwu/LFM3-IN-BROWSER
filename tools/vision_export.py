# SPDX-License-Identifier: Apache-2.0. Copyright 2026 Toaster Club LLC.
"""Export LFM3-Omni's vision tower and projector for the WebGPU engine (public/engine/vision.js), byte-exact.

    python tools/vision_export.py --out work/vision --quant q8 q4g64

Writes <quant>.bin (every buffer of Vision.weightList(), back to back) and manifest_<quant>.json ([{key, bytes}]).
Matrices: q8 = groups of 32 along the input, symmetric int8 with an f32 scale (the engine's own GPU quantizer);
q4g64 = groups of 64 with an f16 scale and minimum, round-to-nearest with a clip search (tools/gptq_export.py).
Vectors (norms, the 3x3 depthwise kernels, biases) stay f32.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from gptq_export import pack, rtn  # noqa: E402

TYPES = ["conv", "conv", "attn", "conv", "attn", "conv", "attn", "conv", "attn", "conv", "attn", "conv", "attn", "conv"]


def q8(w: torch.Tensor):
    """[N, K] -> (codes bytes, scales bytes) exactly like quantizeSource('q8') in public/engine/kernels.js."""
    w = w.float().reshape(-1, 32)
    sc = (w.abs().amax(1) / 127.0).to(torch.float32)
    sc = torch.where(sc == 0, torch.ones_like(sc), sc)
    q = torch.clamp(torch.round(w / sc[:, None]), -127, 127).to(torch.int8)
    return q.cpu().numpy().tobytes(), sc.cpu().numpy().astype(np.float32).tobytes()


def matrix(w: torch.Tensor, quant: str):
    if quant == "q8":
        return q8(w)
    codes, S, Mn, _ = rtn(w.float().cuda() if torch.cuda.is_available() else w.float())
    return pack(codes, S, Mn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="work/vision")
    ap.add_argument("--repo", default="LiquidAI/LFM3-Omni-2B", help="the Hugging Face repository (LiquidAI/d1-Omni-2B: the decision model)")
    ap.add_argument("--quant", nargs="+", default=["q8", "q4g64"])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    f = safe_open(hf_hub_download(args.repo, "model.safetensors"), "pt")
    g = lambda n: f.get_tensor(n)  # noqa: E731
    vec = lambda n: g(n).float().contiguous().numpy().astype(np.float32).tobytes()  # noqa: E731
    V, PJ = "model.vision_tower.", "model.multi_modal_projector."
    for quant in args.quant:
        entries = []

        def put(key, b):
            entries.append((key, b))

        def mat(key, w):
            q, s = matrix(w, quant)
            put(key + ".q", q)
            put(key + ".s", s)

        mat("patch", g(V + "patchifier.weight"))
        put("embNorm", vec(V + "embedding_norm.weight"))
        for i, t in enumerate(TYPES):
            p = f"{V}layers.{i}."
            put(f"{i}.opNorm", vec(p + "operator_norm.weight"))
            put(f"{i}.ffnNorm", vec(p + "ffn_norm.weight"))
            if t == "conv":
                put(f"{i}.convW", vec(p + "conv.spatial_conv.weight"))
            else:
                put(f"{i}.qNorm", vec(p + "self_attn.q_layernorm.weight"))
                put(f"{i}.kNorm", vec(p + "self_attn.k_layernorm.weight"))
            if t == "conv":
                mat(f"{i}.inProj", g(p + "conv.in_proj.weight"))
                mat(f"{i}.outProj", g(p + "conv.out_proj.weight"))
            else:
                a = p + "self_attn."
                mat(f"{i}.qkv", torch.cat([g(a + "q_proj.weight"), g(a + "k_proj.weight"), g(a + "v_proj.weight")]))
                mat(f"{i}.oProj", g(a + "out_proj.weight"))
            mat(f"{i}.w13", torch.cat([g(p + "feed_forward.w1.weight"), g(p + "feed_forward.w3.weight")]))
            mat(f"{i}.w2", g(p + "feed_forward.w2.weight"))
        put("ln.w", vec(PJ + "layer_norm.weight"))
        put("ln.b", vec(PJ + "layer_norm.bias"))
        mat("l1", g(PJ + "linear_1.weight"))
        put("l1.b", vec(PJ + "linear_1.bias"))
        mat("l2", g(PJ + "linear_2.weight"))
        put("l2.b", vec(PJ + "linear_2.bias"))
        # the engine's order within the projector: ln.w, ln.b, l1.q, l1.s, l1.b, l2.q, l2.s, l2.b (as written above)
        with open(os.path.join(args.out, f"{quant}.bin"), "wb") as o:
            for _, b in entries:
                o.write(b)
        man = {"format": quant, "part": "vision", "tensors": [{"key": k, "bytes": len(b)} for k, b in entries],
               "total_bytes": sum(len(b) for _, b in entries)}
        with open(os.path.join(args.out, f"manifest_{quant}.json"), "w") as o:
            json.dump(man, o, indent=1)
        print("VISION_EXPORT", quant, round(man["total_bytes"] / 2 ** 20, 1), "MiB", len(entries), "tensors", flush=True)


if __name__ == "__main__":
    main()
