# SPDX-License-Identifier: Apache-2.0. Copyright 2026 Toaster Club LLC.
"""Export LFM3-Omni's speech output for public/engine/speech.js, byte-exact: two weight sets.

    python tools/speech_export.py --out work/speech --quant q8 q4g64

  speech_<quant>.bin  the depth decoder (depth_linear, mask embedding, the 12 codebooks' tables and output norms, the
                      3-layer trunk) and the codebook embedding the language model reads (DepthDecoder.weightList())
  codec_<quant>.bin   the Liquid codec's decoder: FSQ projections (f32), the 8-layer LFM3, the STFT head (Codec.weightList())
with manifest_speech_<quant>.json / manifest_codec_<quant>.json.
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

CODEC_TYPES = ["conv", "conv", "attn", "conv", "attn", "conv", "attn", "conv"]


def write(out, name, quant, entries):
    with open(os.path.join(out, f"{name}_{quant}.bin"), "wb") as o:
        for _, b in entries:
            o.write(b)
    man = {"format": quant, "part": name, "tensors": [{"key": k, "bytes": len(b)} for k, b in entries],
           "total_bytes": sum(len(b) for _, b in entries)}
    with open(os.path.join(out, f"manifest_{name}_{quant}.json"), "w") as o:
        json.dump(man, o, indent=1)
    print("SPEECH_EXPORT", name, quant, round(man["total_bytes"] / 2 ** 20, 1), "MiB", len(entries), "tensors", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="work/speech")
    ap.add_argument("--quant", nargs="+", default=["q8", "q4g64"])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    mid = "LiquidAI/LFM3-Omni-2B"
    f = safe_open(hf_hub_download(mid, "model.safetensors"), "pt")
    c = safe_open(hf_hub_download(mid, "audio_tokenizer/model.safetensors"), "pt")
    vec = lambda src, n: src.get_tensor(n).float().contiguous().numpy().astype(np.float32).tobytes()  # noqa: E731
    DD = "model.depth_decoder."
    for quant in args.quant:
        e = []

        def mat(key, w):
            q, s = matrix(w, quant)
            e.append((key + ".q", q))
            e.append((key + ".s", s))

        mat("dl", f.get_tensor(DD + "depth_linear.weight"))
        e.append(("dl.b", vec(f, DD + "depth_linear.bias")))
        e.append(("mask", vec(f, DD + "mask_embedding")))
        mat("tab", torch.cat([f.get_tensor(f"{DD}depth_embeddings.{i}.embedding.weight") for i in range(12)]))
        e.append(("tabNorm", b"".join(vec(f, f"{DD}depth_embeddings.{i}.embedding_norm.weight") for i in range(12))))
        for i in range(3):
            p = f"{DD}model.layers.{i}."
            a = p + "self_attn."
            e += [(f"{i}.opNorm", vec(f, p + "operator_norm.weight")), (f"{i}.ffnNorm", vec(f, p + "ffn_norm.weight")),
                  (f"{i}.qNorm", vec(f, a + "q_norm.weight")), (f"{i}.kNorm", vec(f, a + "k_norm.weight"))]
            mat(f"{i}.qkvg", torch.cat([f.get_tensor(a + n + ".weight") for n in ("q_proj", "k_proj", "v_proj", "gate_proj")]))
            mat(f"{i}.oProj", f.get_tensor(a + "o_proj.weight"))
            mat(f"{i}.w13", torch.cat([f.get_tensor(p + "feed_forward.w1.weight"), f.get_tensor(p + "feed_forward.w3.weight")]))
            mat(f"{i}.w2", f.get_tensor(p + "feed_forward.w2.weight"))
        e.append(("finalNorm", vec(f, DD + "model.embedding_norm.weight")))
        mat("cbe", f.get_tensor("model.codebook_embedding.weight"))
        write(args.out, "speech", quant, e)

        e = [("po.w", vec(c, "quantizer.project_out.weight")), ("po.b", vec(c, "quantizer.project_out.bias")),
             ("fq.w", vec(c, "from_quantizer.weight")), ("fq.b", vec(c, "from_quantizer.bias"))]
        for i, t in enumerate(CODEC_TYPES):
            p = f"decoder.lfm.layers.{i}."
            e += [(f"{i}.opNorm", vec(c, p + "operator_norm.weight")), (f"{i}.ffnNorm", vec(c, p + "ffn_norm.weight")),
                  (f"{i}.postOp", vec(c, p + "post_operator_norm.weight")), (f"{i}.postFfn", vec(c, p + "post_ffn_norm.weight"))]
            if t == "conv":
                e.append((f"{i}.convW", vec(c, p + "conv.conv.weight")))
                mat(f"{i}.inProj", c.get_tensor(p + "conv.in_proj.weight"))
                mat(f"{i}.outProj", c.get_tensor(p + "conv.out_proj.weight"))
            else:
                a = p + "self_attn."
                e += [(f"{i}.qNorm", vec(c, a + "q_norm.weight")), (f"{i}.kNorm", vec(c, a + "k_norm.weight"))]
                mat(f"{i}.qkvg", torch.cat([c.get_tensor(a + n + ".weight") for n in ("q_proj", "k_proj", "v_proj", "gate_proj")]))
                mat(f"{i}.oProj", c.get_tensor(a + "o_proj.weight"))
            mat(f"{i}.w13", torch.cat([c.get_tensor(p + "feed_forward.w1.weight"), c.get_tensor(p + "feed_forward.w3.weight")]))
            mat(f"{i}.w2", c.get_tensor(p + "feed_forward.w2.weight"))
        e.append(("finalNorm", vec(c, "decoder.lfm.embedding_norm.weight")))
        mat("lin", c.get_tensor("decoder.lin.weight"))
        e.append(("lin.b", vec(c, "decoder.lin.bias")))
        write(args.out, "codec", quant, e)


if __name__ == "__main__":
    main()
