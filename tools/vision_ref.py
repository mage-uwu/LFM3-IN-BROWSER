# SPDX-License-Identifier: Apache-2.0. Copyright 2026 Toaster Club LLC.
"""PyTorch reference for the WebGPU vision path (public/engine/vision.js): one synthetic picture through LFM3-Omni-2B.

    python tools/vision_ref.py --out work/vision

Writes into --out:
  ref.json         sizes, the chat prompt (text and ids, with the image placeholders), top-40 logits after it, a
                   48-token greedy description, and the vision position ids
  image.rgb        the picture, raw RGB bytes (the page's preprocessing is checked against the processor's)
  pixels.f32       the processor's patches [T, 1536]
  layers.f32       the tower's patch features after every layer [14, T, 1024]
  emb.f32          the projector's output [T/4, 2048]
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

PROMPT = "Describe this image in one sentence."


def picture():
    from PIL import Image, ImageDraw, ImageFont
    w, h = 640, 480
    img = Image.new("RGB", (w, h))
    d = ImageDraw.Draw(img)
    for y in range(h):  # sky to horizon
        c = int(120 + 120 * y / h)
        d.line([(0, y), (w, y)], fill=(c // 2, c - 40, 255))
    d.rectangle([0, 330, w, h], fill=(60, 160, 60))                       # grass
    d.ellipse([500, 40, 590, 130], fill=(255, 215, 0))                    # sun
    d.rectangle([150, 220, 330, 360], fill=(200, 40, 40))                 # house
    d.polygon([(130, 220), (240, 140), (350, 220)], fill=(110, 60, 30))   # roof
    d.rectangle([220, 290, 260, 360], fill=(90, 50, 20))                  # door
    try:
        font = ImageFont.load_default(size=48)
    except TypeError:
        font = ImageFont.load_default()
    d.text((380, 380), "HOME", fill=(255, 255, 255), font=font)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="work/vision")
    ap.add_argument("--max-pixels", type=int, default=200704)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    mid = "LiquidAI/LFM3-Omni-2B"
    model = AutoModelForMultimodalLM.from_pretrained(mid, dtype=torch.bfloat16, device_map="cuda",
                                                     trust_remote_code=True).eval()
    proc = AutoProcessor.from_pretrained(mid, trust_remote_code=True)
    tok = proc.tokenizer
    img = picture()
    img.save(os.path.join(args.out, "image.png"))
    np.asarray(img, dtype=np.uint8).tofile(os.path.join(args.out, "image.rgb"))

    ip = proc.image_processor(images=[img], size={"shortest_edge": 65536, "longest_edge": args.max_pixels}, return_tensors="pt")
    pv, grid = ip["pixel_values"].float(), ip["image_grid_thw"]
    T, gh, gw = pv.shape[0], int(grid[0, 1]), int(grid[0, 2])
    pv.numpy().astype(np.float32).tofile(os.path.join(args.out, "pixels.f32"))
    try:
        from transformers.vision_utils import get_vision_position_ids
        pos = get_vision_position_ids(grid, 2, kwargs={})
        pos = pos.tolist() if hasattr(pos, "tolist") else None
    except Exception as e:  # noqa: BLE001 - only informative
        pos = repr(e)

    vt = model.model.vision_tower
    layers = []
    hooks = [layer.register_forward_hook(lambda m, a, o: layers.append((o[0] if isinstance(o, tuple) else o).float().cpu()))
             for layer in vt.layers]
    with torch.no_grad():
        emb = model.model.get_image_features(pv.cuda().to(torch.bfloat16), grid.cuda()).float().cpu()
    for hk in hooks:
        hk.remove()
    torch.stack(layers).numpy().astype(np.float32).tofile(os.path.join(args.out, "layers.f32"))
    emb.numpy().astype(np.float32).tofile(os.path.join(args.out, "emb.f32"))

    messages = [{"role": "system", "content": "You are a helpful personal assistant."},
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
    text = proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    n = T // 4
    text_full = text.replace("<|image|>", "<|image|>" * n, 1)
    ids = tok(text_full, add_special_tokens=False)["input_ids"]
    x = torch.tensor([ids], device="cuda")
    with torch.no_grad():
        out = model(input_ids=x, pixel_values=pv.cuda().to(torch.bfloat16), image_grid_thw=grid.cuda(), use_cache=False)
        logits = out.logits[0, -1].float()
        g = model.generate(input_ids=x, attention_mask=torch.ones_like(x), pixel_values=pv.cuda().to(torch.bfloat16),
                           image_grid_thw=grid.cuda(), max_new_tokens=48, do_sample=False, generation_mode="text")
    seq = g.sequences[0, x.shape[1]: g.sequence_lengths[0]].tolist()
    top = torch.topk(logits, 40)
    ref = {"T": T, "gh": gh, "gw": gw, "rows": n, "image": [480, 640], "max_pixels": args.max_pixels,
           "template_text": text, "prompt_ids": ids, "image_token": tok.convert_tokens_to_ids("<|image|>"),
           "top_ids": top.indices.tolist(), "top_logits": [round(v, 4) for v in top.values.tolist()],
           "greedy": seq, "greedy_text": tok.decode(seq), "position_ids": pos}
    with open(os.path.join(args.out, "ref.json"), "w") as o:
        json.dump(ref, o)
    print("VISION_REF", json.dumps({k: ref[k] for k in ("T", "gh", "gw", "rows", "greedy_text")}), repr(text[:400]), flush=True)


if __name__ == "__main__":
    main()
