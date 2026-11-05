# SPDX-License-Identifier: Apache-2.0. Copyright 2026 Toaster Club LLC.
"""PyTorch reference for the WebGPU speech output (public/engine/speech.js, Chat.speak).

    python tools/speech_ref.py --out work/speech

The model's interleaved loop (blocks of 6 text tokens and 12 speech frames; after <|text_end|> speech until the
end-of-audio frame), made deterministic so the browser engine can be compared step by step: greedy text, and every
speech frame revealed greedily in the fixed codebook order 0..11 over the 3 passes. Writes into --out:
  ref.json      the prompt ids, the text ids, the frames (12 codes each), the transcript the model's own ASR gives
                for the audio, and sizes
  hidden.f32    the language model's hidden state before the first frame;  logits0.f32  the depth decoder's first-pass
                logits for it [12, 2017]
  audio.f32     the codec's decoding of all frames at once (24 kHz)
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

SYSTEM = "Respond with interleaved text and audio."
QUESTION = "What is the capital of France? Answer in one short sentence."
BANNED = [125012, 125015, 125029, 125030, 125031, 125032]  # as the WebGPU argmax


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="work/speech")
    ap.add_argument("--max-steps", type=int, default=600)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    mid = "LiquidAI/LFM3-Omni-2B"
    model = AutoModelForMultimodalLM.from_pretrained(mid, dtype=torch.bfloat16, device_map="cuda",
                                                     trust_remote_code=True).eval()
    proc = AutoProcessor.from_pretrained(mid, trust_remote_code=True)
    tok, cfg, M = proc.tokenizer, model.config, model.model
    codec = proc.audio_tokenizer.to(device="cuda").eval()
    text = proc.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": QUESTION}],
                                    add_generation_prompt=True, tokenize=False) + "<|mixed_start|>"
    ids = tok(text, add_special_tokens=False)["input_ids"]
    lm, head, emb = M.language_model, model.lm_head, M.language_model.get_input_embeddings()

    with torch.no_grad():
        out = lm(input_ids=torch.tensor([ids], device="cuda"), use_cache=True)
        cache, hidden = out.past_key_values, out.last_hidden_state[:, -1]
        text_ids, frames, first_hidden, first_logits = [], [], None, None
        speaking, slots, text_done, interleaved, pos = False, cfg.interleaved_n_text, False, True, len(ids)
        for _ in range(args.max_steps):
            if speaking:
                codes = torch.zeros((1, 12), dtype=torch.long, device="cuda")
                revealed = torch.zeros_like(codes, dtype=torch.bool)
                for pick in ([0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]):
                    logits = M.depth_decoder(hidden, codes, ~revealed)
                    if first_logits is None:
                        first_hidden, first_logits = hidden.float().cpu(), logits[0].float().cpu()
                    for i in pick:
                        codes[0, i] = logits[0, i].float().argmax()
                    revealed[0, pick] = True
                if bool((codes == cfg.audio_eos_code).any()):
                    codes = torch.full_like(codes, cfg.audio_eos_code)
                    interleaved, speaking = False, False
                else:
                    frames.append(codes[0].tolist())
                    if not text_done:
                        slots -= 1
                        if slots == 0:
                            speaking, slots = False, cfg.interleaved_n_text
                nxt = M.get_audio_out_embeds(codes)   # (frames, 12) -> (frames, 2048)
            else:
                logits = head(hidden).float()[0]
                logits[BANNED] = -float("inf")
                t = int(logits.argmax())
                text_ids.append(t)
                if t == cfg.text_end_token_id and interleaved:
                    text_done, speaking = True, True
                elif t in (124900,) or (t == cfg.text_end_token_id and not interleaved):
                    break
                elif interleaved:
                    slots -= 1
                    if slots == 0:
                        speaking, slots = True, cfg.interleaved_n_audio
                nxt = emb(torch.tensor([t], device="cuda"))
            nxt = nxt.reshape(1, 1, -1).to(torch.bfloat16)
            o = lm(inputs_embeds=nxt, past_key_values=cache, use_cache=True,
                   position_ids=torch.tensor([[pos]], device="cuda"))
            cache, hidden, pos = o.past_key_values, o.last_hidden_state[:, -1], pos + 1
        audio = codec.decode(torch.tensor(frames, device="cuda").T[None]).audio_values[0].float().cpu().numpy() if frames else np.zeros(0, np.float32)
    first_hidden.numpy().astype(np.float32).tofile(os.path.join(args.out, "hidden.f32"))
    first_logits.numpy().astype(np.float32).tofile(os.path.join(args.out, "logits0.f32"))
    audio.astype(np.float32).tofile(os.path.join(args.out, "audio.f32"))
    # the model's own ASR on its speech (24 kHz -> 16 kHz)
    from scipy.signal import resample_poly
    a16 = resample_poly(audio, 2, 3).astype(np.float32)
    feats = proc.feature_extractor(a16, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        n = M.get_audio_features(feats["input_features"].cuda().to(torch.bfloat16), feats["feature_attention_mask"].cuda()).shape[0]
        asr = proc.apply_chat_template([{"role": "system", "content": "Perform ASR in English."}, {"role": "user", "content": [{"type": "audio"}]}],
                                       add_generation_prompt=True, tokenize=False).replace("<|audio|>", "<|audio|>" * n, 1)
        x = torch.tensor([tok(asr, add_special_tokens=False)["input_ids"]], device="cuda")
        g = model.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=96, do_sample=False, generation_mode="text",
                           input_features=feats["input_features"].cuda().to(torch.bfloat16), feature_attention_mask=feats["feature_attention_mask"].cuda())
    transcript = tok.decode(g.sequences[0, x.shape[1]: g.sequence_lengths[0]].tolist(), skip_special_tokens=True)
    ref = {"system": SYSTEM, "question": QUESTION, "prompt_ids": ids, "text_ids": text_ids,
           "text": tok.decode(text_ids, skip_special_tokens=True), "frames": frames, "samples": int(audio.shape[0]),
           "transcript": transcript}
    with open(os.path.join(args.out, "ref.json"), "w") as o:
        json.dump(ref, o)
    print("SPEECH_REF", json.dumps({"text": ref["text"], "frames": len(frames), "seconds": round(audio.shape[0] / 24000, 2),
                                    "transcript": transcript}), flush=True)


if __name__ == "__main__":
    main()
