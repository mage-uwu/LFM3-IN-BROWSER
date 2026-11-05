# SPDX-License-Identifier: Apache-2.0. Copyright 2026 Toaster Club LLC.
"""Reference outputs of vanilla LFM3-Omni-2B (PyTorch, bf16) for the WebGPU engine's correctness test.

    python tools/lfm3_ref.py --out work/test_ref.json

Cases:
  text: a ~300-token passage as a chat turn (crosses the 256-token sliding windows, several 256-key attention
        chunks and many 16-token prefill chunks): last-position logits (top 40), the residual stream after every
        layer at the last position, and a 48-token greedy reply;
  chat: a chat prompt in LFM3's template: the greedy reply up to <|im_end|>, and the exact prompt text (the
        browser builds the same text with its own template code).
"""

from __future__ import annotations

import argparse
import json

import torch

PASSAGE = (
    "The lighthouse keeper's log for the winter of that year is unusually detailed. Each morning he recorded the "
    "wind, the height of the swell, and the colour of the sky before dawn, and each evening he noted which ships "
    "had passed and whether their lamps were lit. In February he begins to write about his daughter, who had come "
    "to stay after her husband was lost at sea. She took over the evening watch, and her handwriting appears in the "
    "log for eleven nights in a row, neater than his and more precise about the times.\n\n"
    "On the night of the twelfth of March a cargo steamer lost her rudder in a gale and drifted toward the rocks. "
    "The log records that the light was kept burning through the night, that the fog signal was sounded every "
    "thirty seconds for nine hours, and that at first light the steamer was seen at anchor in the lee of the "
    "headland, her crew safe. The shipping company later sent a silver tray, which thanks the keeper by name and "
    "does not mention his daughter. The log itself was found in a box of parish records, water-stained but "
    "complete. Its last entry for that winter is a single line: \"Calm sea, clear sky, all lamps in order.\" Below "
    "it, in the second hand, someone has added: \"We did well.\"\n\n"
    "In two sentences: what does the log tell us that the tray does not?"
)
CHAT = [{"role": "system", "content": "You are a helpful personal assistant."},
        {"role": "user", "content": "What is the capital of France? Answer in one sentence."}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="work/test_ref.json")
    args = ap.parse_args()
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    mid = "LiquidAI/LFM3-Omni-2B"
    model = AutoModelForMultimodalLM.from_pretrained(mid, dtype=torch.bfloat16, device_map="cuda",
                                                     trust_remote_code=True).eval()
    proc = AutoProcessor.from_pretrained(mid, trust_remote_code=True)
    tok = proc.tokenizer
    lm, head = model.model.language_model, model.lm_head
    cases = {}

    def run(name, ids, n_new, text=None):
        x = torch.tensor([ids], device="cuda")
        with torch.no_grad():
            out = lm(input_ids=x, output_hidden_states=True, use_cache=False)
            logits = head(out.last_hidden_state[:, -1]).float()[0]
            hs = out.hidden_states  # embeddings, then each layer (the last may already be normalised)
            g = model.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=n_new, do_sample=False,
                               generation_mode="text")
        top = torch.topk(logits, 40)
        seq = g.sequences[0, x.shape[1] : g.sequence_lengths[0]].tolist()
        cases[name] = {
            "ids": ids, "text": text, "top_ids": top.indices.tolist(), "top_logits": [round(v, 4) for v in top.values.tolist()],
            "hidden": [[round(v, 5) for v in h[0, -1].float().tolist()] for h in hs[1:]],
            "hidden_count": len(hs), "greedy": seq, "greedy_text": tok.decode(seq),
        }
        print(name, len(ids), "tokens; top", top.indices[:5].tolist(), "greedy:", repr(tok.decode(seq))[:200], flush=True)

    long_chat = [{"role": "system", "content": "You are a helpful personal assistant."}, {"role": "user", "content": PASSAGE}]
    text = proc.apply_chat_template(long_chat, add_generation_prompt=True, tokenize=False)
    run("text", tok(text, add_special_tokens=False)["input_ids"], 32, text=text)
    chat_text = proc.apply_chat_template(CHAT, add_generation_prompt=True, tokenize=False)
    run("chat", tok(chat_text, add_special_tokens=False)["input_ids"], 64, text=chat_text)
    with open(args.out, "w") as f:
        json.dump({"model": mid, "passage": text, "cases": cases}, f)
    print("REF_DONE", args.out, flush=True)


if __name__ == "__main__":
    main()
