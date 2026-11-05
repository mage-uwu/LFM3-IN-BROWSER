# SPDX-License-Identifier: Apache-2.0. Copyright 2026 Toaster Club LLC.
"""Reference answers of LiquidAI/d1-Omni-2B, the decision model, from its own runner (the repository's System One
code, run as released), for checking the WebGPU port against:

    python tools/d1_ref.py --out work/d1_ref.json [--repo LiquidAI/d1-Omni-2B]

For each case it records what the port must reproduce: every sequence the runner builds (its token ids, the state's
length, each token's question, the answer slots), the log-probabilities at each slot over the ids the readout reads,
the probabilities after calibration, and the answers. Needs a GPU and the repository's remote code.
"""

from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np

TICKET = "I was charged twice and my parcel never arrived. Please refund one charge."
CASES = [
    ("text", TICKET, {
        "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
        "team": {"type": "choice", "instructions": "Which team should handle this?",
                 "criteria": {"billing": "Charges, refunds, invoices", "shipping": "Delivery and parcels",
                              "technical": "App or site faults"}},
        "topics": {"type": "choice", "multiple": True, "instructions": "Which topics does the message raise?",
                   "criteria": {"billing": "Payments and charges", "delivery": "Shipping and parcels",
                                "login": "Signing in to the account"}},
        "anger": {"type": "score", "instructions": "How upset is the customer?",
                  "criteria": ["Calm", "Annoyed", "Angry"]},
    }, None),
    ("json", {"order": {"id": 4411, "items": 3, "paid": True}, "message": "Where is my order? It is late."}, {
        "late": {"type": "noul", "instructions": "Does the customer say the order is late?"},
        "paid": {"type": "noul", "instructions": "Has the order been paid?"},
    }, None),
    ("audio", None, {
        "beeps": {"type": "number", "instructions": "How many beeps are there?", "range": [0, 10], "step": 1},
        "speech": {"type": "noul", "instructions": "Does someone speak in the recording?"},
    }, "beeps"),
    ("image", None, {
        "red": {"type": "noul", "instructions": "Is there a red circle in the picture?"},
        "shape": {"type": "choice", "instructions": "What shape is shown?",
                  "criteria": {"circle": None, "square": None, "triangle": None}},
    }, "circle"),
]


def beeps(rate=16000):
    clip = np.random.default_rng(0).normal(0, 0.003, 4 * rate).astype(np.float32)
    for start in (0.5, 1.5, 2.5):
        clip[int(start * rate):int(start * rate) + 3200] += 0.3 * np.sin(2 * np.pi * 880 * np.arange(3200) / rate)
    return clip


def circle(size=448):
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (size, size), "white")
    r = size * 0.33
    ImageDraw.Draw(im).ellipse((size / 2 - r, size / 2 - r, size / 2 + r, size / 2 + r), fill=(224, 38, 43))
    return im


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.generic,)):
        return x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if hasattr(x, "tolist"):
        return x.tolist()
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return repr(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="LiquidAI/d1-Omni-2B")
    ap.add_argument("--out", default="work/d1_ref.json")
    args = ap.parse_args()
    import torch
    from transformers import AutoModel

    model = AutoModel.from_pretrained(args.repo, trust_remote_code=True, dtype=torch.bfloat16).to("cuda").eval()
    eng = model.engine
    tok = eng.tokenizer
    mod = __import__(type(eng).__module__.rsplit(".", 1)[0] + ".prompt", fromlist=["x"])
    res = {"repo": args.repo, "config": {k: getattr(model.config, k, None) for k in
                                         ("d1_codes", "d1_attention", "d1_set_size", "d1_numbers", "d1_vision_max_pixels")},
           "calibration": jsonable(getattr(eng.calibration, "__dict__", None)), "cases": []}
    captured = []
    orig = eng._readouts

    def spy(qs, logz, tempered):  # the slot log-probabilities the readout reads
        out = orig(qs, logz, tempered)
        for q, z, t in zip(qs, logz, tempered):
            groups = mod.readout_ids(tok, q)
            captured.append({"question": repr(q), "tempered": bool(t), "ids": groups,
                             "logz": [[float(z[i]) for i in g] for g in groups],
                             "top": [[int(i), float(z[i])] for i in torch.topk(z.float(), 8).indices.tolist()]})
        return out
    eng._readouts = spy
    for name, state, questions, media in CASES:
        med = None
        if media == "beeps":
            med = [{"type": "audio", "audio": beeps(), "sampling_rate": 16000}]
        elif media == "circle":
            med = [circle()]
        named = eng.named(questions)
        qs = list(named.values())
        seqs = eng._sequences([(state, qs, med or [])])
        captured.clear()
        out = model.system_one(state, questions, media=med)
        probs = eng.probabilities(state, questions, med)
        case = {"name": name, "state": state, "questions": questions, "media": media, "answers": jsonable(out["answers"]),
                "usage": jsonable(out.get("usage")), "probabilities": jsonable(probs), "slots": captured[:len(captured) // 2 or None],
                "sequences": [{"first": k, "ids": ids, "text": tok.decode(ids), "state_len": jsonable(sl),
                               "question": jsonable(qq), "ends": jsonable(ends), "media_keys": sorted(inp.keys()) if inp else []}
                              for (_, k, ids, sl, qq, ends, inp) in seqs],
                "suffixes": [mod.suffix_layout(tok, q, eng.lead, eng.option_style)[0] for q in qs]}
        res["cases"].append(case)
        print(name, json.dumps(jsonable(out["answers"]))[:400], flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(jsonable(res), f, indent=1)
    print("D1_REF", args.out, flush=True)


if __name__ == "__main__":
    main()
