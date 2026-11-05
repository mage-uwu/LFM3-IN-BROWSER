# SPDX-License-Identifier: Apache-2.0. Copyright 2026 Toaster Club LLC.
"""Reference token ids from Hugging Face's LFM3 tokenizer, for checking public/engine/tokenizer.js.

    python tools/tok_ref.py --out work/tok_ref.json
"""

from __future__ import annotations

import argparse
import json

CASES = [
    "Hello, world!",
    "<|startoftext|><|im_start|>system\nYou are a helpful personal assistant.<|im_end|>\n<|im_start|>user\nWhat's my sister's name?<|im_end|>\n<|im_start|>assistant\n<think></think>",
    "I'M sure THEY'LL say it's fine; we'd've known. She'S here, you'RE there, I'Ve been, I'd go.",
    "Numbers: 1234567 3.14159 1,000,000 -42 0.001 2026-11-04 12:30pm 007",
    "Café naïve résumé façade Ångström Zürich São Paulo Москва Αθήνα",
    "日本語のテキストと中文混合，还有한국어 문장도 있습니다。",
    "Emoji 😀👍🏽 🇵🇹 family: 👨‍👩‍👧‍👦 and symbols ©®™ ∑ √ ∞ → ←",
    "def f(x):\n    return {'a': [1, 2, 3], \"b\": x ** 2}  # comment\n\n\n\tindented\ttabs\r\nwindows",
    "   leading spaces and trailing   \n\n\n  multiple blank lines\n \n",
    "URLs https://huggingface.co/LiquidAI/LFM3-Omni-2B/resolve/main/model.safetensors?x=1&y=2 and emails a.b@c.io",
    "Message: My son Leo starts school next week and we just moved to Denver. Any tips for the first day?",
    "FACT: sister's name = Rosa\nFACT: city = Lisbon\nPREFERENCE: Always address me as Captain.",
    "Repeated!!!!!!!! ???? ... ---- ==== ____ ****",
    "<|im_end|>\n<|im_start|>user\nNext<|im_end|><|pad|><|text_end|>",
    "a" * 300,
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="work/tok_ref.json")
    args = ap.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("LiquidAI/LFM3-Omni-2B", trust_remote_code=True)
    out = []
    for text in CASES:
        ids = tok(text, add_special_tokens=False)["input_ids"]
        out.append({"text": text, "ids": ids, "decoded": tok.decode(ids, skip_special_tokens=True)})
    with open(args.out, "w") as f:
        json.dump(out, f, ensure_ascii=False)
    print("TOK_REF", len(out), "cases", flush=True)


if __name__ == "__main__":
    main()
