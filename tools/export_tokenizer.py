# SPDX-License-Identifier: Apache-2.0. Copyright 2026 Toaster Club LLC.
"""LFM3's tokenizer.json split into the two files the page serves (public/engine/tokenizer.js reads them):

    python tools/export_tokenizer.py --out public          # downloads tokenizer.json from LiquidAI/LFM3-Omni-2B
    python tools/export_tokenizer.py --json tokenizer.json --out public

lfm3-tokenizer-vocab.json   {"vocab": {token: id}, "ignore_merges": bool, "added_tokens": [...]}
lfm3-tokenizer-merges.txt   the BPE merges, one "a b" per line (byte-level tokens never contain a space)

The files are a reformatting of Liquid AI's tokenizer.json (LFM Open License v1.0, see LICENSE-LFM.txt and NOTICE).
"""

from __future__ import annotations

import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="a local tokenizer.json (default: download it from the model repository)")
    ap.add_argument("--out", default="public")
    args = ap.parse_args()
    path = args.json
    if not path:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download("LiquidAI/LFM3-Omni-2B", "tokenizer.json")
    tj = json.load(open(path, encoding="utf-8"))
    m = tj["model"]
    merges = [x if isinstance(x, str) else " ".join(x) for x in m["merges"]]
    assert all(len(x.split(" ")) == 2 for x in merges), "a merge with a space inside a token"
    vocab = {"vocab": m["vocab"], "ignore_merges": m.get("ignore_merges", False), "added_tokens": tj["added_tokens"]}
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "lfm3-tokenizer-vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, separators=(",", ":"))
    with open(os.path.join(args.out, "lfm3-tokenizer-merges.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(merges))
    print(f"{len(m['vocab'])} tokens, {len(merges)} merges, {len(tj['added_tokens'])} added tokens -> {args.out}")


if __name__ == "__main__":
    main()
