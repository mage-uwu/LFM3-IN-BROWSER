# SPDX-License-Identifier: Apache-2.0. Copyright 2026 Toaster Club LLC.
"""PyTorch reference for the WebGPU speech-input path (public/engine/audio.js): a LibriSpeech clip through LFM3-Omni-2B.

    python tools/audio_ref.py --out work/audio

Writes into --out:
  ref.json      sizes, the reference transcript, the ASR prompt (text and ids), top-40 logits after it and the greedy
                transcription
  samples.f32   the 16 kHz clip;  mel.f32  the feature extractor's log-mel [128, frames]
  layers.f32    the encoder's features after every layer [17, T, 768];  emb.f32  the projector's output [T, 2048]
"""

from __future__ import annotations

import argparse
import io
import json
import os

import numpy as np
import torch


def clip():
    import soundfile as sf
    from datasets import Audio, load_dataset
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    ds = ds.cast_column("audio", Audio(decode=False))
    ex = ds[0]
    a, sr = sf.read(io.BytesIO(ex["audio"]["bytes"]), dtype="float32")
    assert sr == 16000, sr
    return a.astype(np.float32), ex["text"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="work/audio")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    mid = "LiquidAI/LFM3-Omni-2B"
    model = AutoModelForMultimodalLM.from_pretrained(mid, dtype=torch.bfloat16, device_map="cuda",
                                                     trust_remote_code=True).eval()
    proc = AutoProcessor.from_pretrained(mid, trust_remote_code=True)
    tok = proc.tokenizer
    samples, text_ref = clip()
    samples.tofile(os.path.join(args.out, "samples.f32"))
    feats = proc.feature_extractor(samples, sampling_rate=16000, return_tensors="pt")
    mel, mask = feats["input_features"].float(), feats["feature_attention_mask"]
    frames = mel.shape[-1]
    mel[0].numpy().astype(np.float32).tofile(os.path.join(args.out, "mel.f32"))

    enc = model.model.audio_encoder
    layers = []
    hooks = [layer.register_forward_hook(lambda m, a, o: layers.append((o[0] if isinstance(o, tuple) else o)[0].float().cpu()))
             for layer in enc.lfm.layers]
    with torch.no_grad():
        emb = model.model.get_audio_features(mel.cuda().to(torch.bfloat16), mask.cuda()).float().cpu()
    for hk in hooks:
        hk.remove()
    T = emb.shape[0]
    torch.stack(layers).numpy().astype(np.float32).tofile(os.path.join(args.out, "layers.f32"))
    emb.numpy().astype(np.float32).tofile(os.path.join(args.out, "emb.f32"))

    messages = [{"role": "system", "content": "Perform ASR in English."}, {"role": "user", "content": [{"type": "audio"}]}]
    text = proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    text_full = text.replace("<|audio|>", "<|audio|>" * T, 1)
    ids = tok(text_full, add_special_tokens=False)["input_ids"]
    x = torch.tensor([ids], device="cuda")
    kw = {"input_features": mel.cuda().to(torch.bfloat16), "feature_attention_mask": mask.cuda()}
    with torch.no_grad():
        logits = model(input_ids=x, use_cache=False, **kw).logits[0, -1].float()
        g = model.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=96, do_sample=False,
                           generation_mode="text", **kw)
    seq = g.sequences[0, x.shape[1]: g.sequence_lengths[0]].tolist()
    top = torch.topk(logits, 40)
    ref = {"samples": len(samples), "frames": frames, "T": T, "text_ref": text_ref, "template_text": text,
           "prompt_ids": ids, "audio_token": tok.convert_tokens_to_ids("<|audio|>"),
           "top_ids": top.indices.tolist(), "top_logits": [round(v, 4) for v in top.values.tolist()],
           "greedy": seq, "greedy_text": tok.decode(seq, skip_special_tokens=True)}
    with open(os.path.join(args.out, "ref.json"), "w") as o:
        json.dump(ref, o)
    print("AUDIO_REF", json.dumps({k: ref[k] for k in ("samples", "frames", "T", "text_ref", "greedy_text")}), repr(text[:300]), flush=True)


if __name__ == "__main__":
    main()
