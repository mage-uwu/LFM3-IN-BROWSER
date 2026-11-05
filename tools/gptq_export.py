# SPDX-License-Identifier: Apache-2.0. Copyright 2026 Toaster Club LLC.
"""GPTQ 4-bit export of vanilla LFM3-Omni-2B's text backbone for the WebGPU engine (public/engine/lfm3.js, "q4g64").

    python tools/gptq_export.py --out work/gptq [--repo LiquidAI/d1-Omni-2B] [--rtn]

Every matrix (feed-forward, conv projections, attention, and the tied embedding / output head) becomes 4-bit codes in
groups of 64 along the input dimension with an f16 scale and an f16 minimum per group (4.5 bits per weight, about
975 MB in all). The layers are quantized one after another with GPTQ (Frantar et al., 2022): each matrix's rounding
error is compensated within its row using the second moment of its real inputs, and every later matrix is
calibrated on the outputs of the already-quantized ones (true-sequential). The calibration text is chat-formatted
conversations (the app's use) plus educational web text.

Writes into --out:
  embed_rtn.bin    embed.q + embed.s: the tied table, round-to-nearest with a clip search (best for lookups)
  embed_gptq.bin   the same table with GPTQ on the output head's inputs (best for the logits)
  rest.bin         everything after the table, in the engine's weightList() order, byte-exact
  manifest.json    the tensor list [{key, bytes}] and the held-out quality: perplexity of the original model, of
                   plain round-to-nearest 4-bit, and of GPTQ with either table
A model file for the engine is embed_<variant>.bin followed by rest.bin (tools/png_chunks.py packs it for the page).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch

GS = 64          # weights per group
CLIP = (1.0, 0.96, 0.92, 0.88, 0.84, 0.8)  # range shrink factors tried per group (min squared error wins)
F16_TINY = 2.0 ** -24


class Stop(Exception):
    pass


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---- the group quantizer (exactly what the engine dequantizes: code * f16(scale) + f16(min)) -----------------
def group_params(Wg):
    """Wg: [..., GS] f32 -> scale, min [...] (f16-representable f32), chosen by a clip search on squared error."""
    lo, hi = Wg.amin(-1), Wg.amax(-1)
    best = None
    for r in CLIP:
        l, h = lo * r, hi * r
        s = ((h - l) / 15).half().float()
        s = torch.where(s <= 0, torch.full_like(s, F16_TINY), s)
        m = l.half().float()
        q = torch.clamp(torch.round((Wg - m[..., None]) / s[..., None]), 0, 15)
        e = ((q * s[..., None] + m[..., None] - Wg) ** 2).sum(-1)
        if best is None:
            best = [e, s, m]
        else:
            k = e < best[0]
            best = [torch.where(k, e, best[0]), torch.where(k, s, best[1]), torch.where(k, m, best[2])]
    return best[1], best[2]


def rtn(W, rows=8192):
    """Round-to-nearest with the clip search: codes [N, K] u8, scale/min [N, K/GS], dequantized [N, K]."""
    N, K = W.shape
    codes = torch.empty(N, K, dtype=torch.uint8, device=W.device)
    S = torch.empty(N, K // GS, device=W.device)
    Mn = torch.empty_like(S)
    deq = torch.empty(N, K, device=W.device)
    for r0 in range(0, N, rows):
        Wg = W[r0:r0 + rows].float().reshape(-1, K // GS, GS)
        s, m = group_params(Wg)
        q = torch.clamp(torch.round((Wg - m[..., None]) / s[..., None]), 0, 15)
        codes[r0:r0 + rows] = q.reshape(-1, K).to(torch.uint8)
        S[r0:r0 + rows], Mn[r0:r0 + rows] = s, m
        deq[r0:r0 + rows] = (q * s[..., None] + m[..., None]).reshape(-1, K)
    return codes, S, Mn, deq


def gptq(W, H, block=128, damp=0.01):
    """GPTQ with groups of GS (group parameters from the error-updated weights as each group is reached)."""
    assert block % GS == 0
    W = W.float().clone()
    N, K = W.shape
    H = H.clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    for attempt in range(6):
        try:
            Hd = H + (damp * 10 ** attempt) * torch.mean(torch.diag(H)) * torch.eye(K, device=H.device)
            Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(Hd)), upper=True)
            break
        except RuntimeError:
            continue
    else:
        raise RuntimeError("Hessian is not positive definite")
    codes = torch.empty(N, K, dtype=torch.uint8, device=W.device)
    S = torch.empty(N, K // GS, device=W.device)
    Mn = torch.empty_like(S)
    for i1 in range(0, K, block):
        i2 = min(i1 + block, K)
        W1 = W[:, i1:i2].clone()
        Err = torch.zeros_like(W1)
        Hi = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            c = i1 + i
            if c % GS == 0:
                s, m = group_params(W1[:, i:i + GS])
                S[:, c // GS], Mn[:, c // GS] = s, m
            w = W1[:, i]
            q = torch.clamp(torch.round((w - m) / s), 0, 15)
            codes[:, c] = q.to(torch.uint8)
            err = (w - (q * s + m)) / Hi[i, i]
            W1[:, i:] -= err[:, None] * Hi[i, i:][None, :]
            Err[:, i] = err
        W[:, i2:] -= Err @ Hinv[i1:i2, i2:]
    deq = codes.float() * S.repeat_interleave(GS, 1) + Mn.repeat_interleave(GS, 1)
    return codes, S, Mn, deq


def pack(codes, S, Mn):
    """The engine's bytes: two codes per byte (element 2j in the low nibble), then [f16 scale, f16 min] per group."""
    c = codes.cpu().numpy()
    q = (c[:, 0::2] | (c[:, 1::2] << 4)).astype(np.uint8).tobytes()
    sm = np.stack([S.cpu().numpy().astype(np.float16), Mn.cpu().numpy().astype(np.float16)], -1)
    return q, sm.tobytes()


# ---- calibration and held-out text ---------------------------------------------------------------------------
def token_windows(tok, texts, n, seqlen, bos):
    """Concatenate tokenized texts and cut n windows of seqlen tokens, each starting with BOS."""
    out, buf = [], []
    for t in texts:
        ids = tok(t, add_special_tokens=False)["input_ids"]
        if ids and ids[0] == bos:
            ids = ids[1:]
        buf.extend(ids)
        while len(buf) >= seqlen - 1 and len(out) < n:
            out.append([bos] + buf[:seqlen - 1])
            buf = buf[seqlen - 1:]
        if len(out) >= n:
            break
    return torch.tensor(out)


def texts_chat(proc, split, skip=0):
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split=split, streaming=True)
    for i, ex in enumerate(ds):
        if i < skip:
            continue
        msgs = [{"role": "system", "content": "You are a helpful personal assistant."}] + \
               [{"role": m["role"], "content": m["content"]} for m in ex["messages"]]
        yield proc.apply_chat_template(msgs, tokenize=False)


def texts_web(skip=0):
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True).skip(skip)
    for ex in ds:
        yield ex["text"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="work/gptq")
    ap.add_argument("--repo", default="LiquidAI/LFM3-Omni-2B", help="the Hugging Face repository (LiquidAI/d1-Omni-2B: the decision model)")
    ap.add_argument("--chat", type=int, default=96, help="calibration windows of chat text")
    ap.add_argument("--web", type=int, default=32, help="calibration windows of web text")
    ap.add_argument("--eval", type=int, default=24, help="held-out windows of each kind")
    ap.add_argument("--seqlen", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--rtn", action="store_true", help="plain round-to-nearest only: no calibration, GPTQ or perplexity")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    from safetensors import safe_open
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    mid = args.repo
    base = "LiquidAI/LFM3-Omni-2B"
    # the architecture (and the processor, for the calibration text) from LFM3-Omni-2B; a release built on it with its
    # own classes (d1-Omni-2B, the decision model) has the same tensors, which replace the base model's here
    model = AutoModelForMultimodalLM.from_pretrained(base, dtype=torch.bfloat16, device_map="cuda",
                                                     trust_remote_code=True).eval()
    proc = AutoProcessor.from_pretrained(base, trust_remote_code=True)
    if mid != base:
        from safetensors.torch import load_file
        weights = load_file(hf_hub_download(mid, "model.safetensors"))
        own = model.state_dict()
        lm_keys = [k for k in own if ".language_model." in k]
        missing = [k for k in lm_keys if k not in weights]
        assert not missing, f"{mid} lacks {len(missing)} language-model tensors, e.g. {missing[:3]}"
        model.load_state_dict({k: v for k, v in weights.items() if k in own}, strict=False)
        model.tie_weights()
        log(f"weights from {mid}: {sum(k in own for k in weights)} of {len(weights)} tensors, all {len(lm_keys)} language-model ones")
        del weights
    tok = proc.tokenizer
    lm, head = model.model.language_model, model.lm_head
    bos = tok.convert_tokens_to_ids("<|startoftext|>")
    tied = head.weight.data_ptr() == lm.embed_tokens.weight.data_ptr()
    log("model loaded; head tied to the embedding:", tied)
    assert tied, "the engine stores one table for the embedding and the head"

    t0 = time.time()
    calib = held = torch.zeros(0, args.seqlen, dtype=torch.long)
    if not args.rtn:
        calib = torch.cat([token_windows(tok, texts_chat(proc, "train_sft"), args.chat, args.seqlen, bos),
                           token_windows(tok, texts_web(), args.web, args.seqlen, bos)])
        held = torch.cat([token_windows(tok, texts_chat(proc, "test_sft"), args.eval, args.seqlen, bos),
                          token_windows(tok, texts_web(skip=5000), args.eval, args.seqlen, bos)])
        log(f"calibration {tuple(calib.shape)}, held-out {tuple(held.shape)} ({time.time() - t0:.0f}s)")
    batches = [calib[i:i + args.batch].cuda() for i in range(0, len(calib), args.batch)]

    @torch.no_grad()
    def perplexity():
        tot, cnt = 0.0, 0
        for row in held:
            x = row[None].cuda()
            h = lm(input_ids=x, use_cache=False).last_hidden_state[0, :-1]
            for c0 in range(0, h.shape[0], 256):
                logits = head(h[c0:c0 + 256]).float()
                tot += torch.nn.functional.cross_entropy(logits, x[0, 1 + c0:1 + c0 + 256], reduction="sum").item()
            cnt += h.shape[0]
        return round(math.exp(tot / cnt), 4)

    @torch.no_grad()
    def hessian(module):
        """Second moment of `module`'s inputs over the calibration set; the forward stops at that module."""
        H = None
        n = 0

        def pre(mod, a, kw):
            nonlocal H, n
            x = a[0] if a else next(iter(kw.values()))
            x = x.reshape(-1, x.shape[-1]).float()
            H = x.t() @ x if H is None else H.addmm_(x.t(), x)
            n += x.shape[0]
            raise Stop

        hk = module.register_forward_pre_hook(pre, with_kwargs=True)
        try:
            for b in batches:
                try:
                    lm(input_ids=b, use_cache=False)
                except Stop:
                    pass
        finally:
            hk.remove()
        return H / n

    layers = lm.layers
    L = len(layers)
    orig = {k: v.detach().to("cpu", copy=True) for k, v in lm.named_parameters()}
    name_of = {v: k for k, v in lm.named_parameters()}

    def groups_of(i):
        """Each step: the module whose input is captured, and the matrices sharing that input (engine order)."""
        ly = layers[i]
        if hasattr(ly, "conv"):
            return [("inProj", ly.conv.in_proj, [ly.conv.in_proj]), ("outProj", ly.conv.out_proj, [ly.conv.out_proj]),
                    ("w13", ly.feed_forward.w1, [ly.feed_forward.w1, ly.feed_forward.w3]),
                    ("w2", ly.feed_forward.w2, [ly.feed_forward.w2])]
        a = ly.self_attn
        return [("qkvg", a.q_proj, [a.q_proj, a.k_proj, a.v_proj, a.gate_proj]), ("oProj", a.o_proj, [a.o_proj]),
                ("w13", ly.feed_forward.w1, [ly.feed_forward.w1, ly.feed_forward.w3]),
                ("w2", ly.feed_forward.w2, [ly.feed_forward.w2])]

    for i in range(L):
        assert [k for k, _, _ in groups_of(i)][0] in ("inProj", "qkvg")
    if args.rtn:  # plain round-to-nearest (same clip search), no calibration text or perplexity
        quality = {"method": "round-to-nearest 4-bit, groups of 64, clip search; no calibration"}
        packed = {}
        with torch.no_grad():
            for i in range(L):
                for key, _, mats in groups_of(i):
                    codes, S, Mn, _ = rtn(torch.cat([orig[name_of[m.weight]].to("cuda").float() for m in mats]))
                    packed[f"{i}.{key}"] = pack(codes, S, Mn)
            emb_rtn = [t.cpu() for t in rtn(orig[name_of[lm.embed_tokens.weight]].to("cuda").float())[:3]]
        emb_g = emb_rtn
        log("round-to-nearest done")
    else:
        quality = {}
        quality["ppl_bf16"] = perplexity()
        log("perplexity, original bf16:", quality["ppl_bf16"])

        # baseline: plain round-to-nearest 4-bit (groups of 64, same clip search) on every matrix
        with torch.no_grad():
            for i in range(L):
                for _, _, mats in groups_of(i):
                    for m in mats:
                        m.weight.copy_(rtn(m.weight.float())[3].to(m.weight.dtype))
            emb_rtn = rtn(lm.embed_tokens.weight.float())
            emb_rtn_deq = emb_rtn[3].to(torch.bfloat16)
            emb_rtn = [t.cpu() for t in emb_rtn[:3]]
            lm.embed_tokens.weight.copy_(emb_rtn_deq)
        quality["ppl_rtn_q4g64"] = perplexity()
        log("perplexity, round-to-nearest 4-bit:", quality["ppl_rtn_q4g64"])
        with torch.no_grad():
            for k, v in lm.named_parameters():
                v.copy_(orig[k].to(v.device))
            lm.embed_tokens.weight.copy_(emb_rtn_deq)  # the layers are calibrated on the 4-bit table
            del emb_rtn_deq

        # GPTQ, true-sequential: each step sees the outputs of every matrix quantized before it
        packed = {}
        t0 = time.time()
        with torch.no_grad():
            for i in range(L):
                for key, cap, mats in groups_of(i):
                    H = hessian(cap)
                    W = torch.cat([orig[name_of[m.weight]].to("cuda").float() for m in mats])
                    codes, S, Mn, deq = gptq(W, H)
                    r = 0
                    for m in mats:
                        n = m.weight.shape[0]
                        m.weight.copy_(deq[r:r + n].to(m.weight.dtype))
                        r += n
                    packed[f"{i}.{key}"] = pack(codes, S, Mn)
                    del H, W, codes, S, Mn, deq
                log(f"layer {i} ({'conv' if hasattr(layers[i], 'conv') else 'attn'}) done, {time.time() - t0:.0f}s")
        quality["ppl_gptq_q4g64_rtn_table"] = perplexity()
        log("perplexity, GPTQ 4-bit (table round-to-nearest):", quality["ppl_gptq_q4g64_rtn_table"])

        # the table as the output head: GPTQ on the final hidden states (the lookups then use the same values)
        with torch.no_grad():
            H = None
            n = 0
            for b in batches:
                x = lm(input_ids=b, use_cache=False).last_hidden_state
                x = x.reshape(-1, x.shape[-1]).float()
                H = x.t() @ x if H is None else H.addmm_(x.t(), x)
                n += x.shape[0]
            emb_g = gptq(orig[name_of[lm.embed_tokens.weight]].to("cuda").float(), H / n)
            del H
            lm.embed_tokens.weight.copy_(emb_g[3].to(torch.bfloat16))
            emb_g = [t.cpu() for t in emb_g[:3]]
        quality["ppl_gptq_q4g64_gptq_table"] = perplexity()
        log("perplexity, GPTQ 4-bit (table GPTQ):", quality["ppl_gptq_q4g64_gptq_table"])

    # ---- write the engine's bytes ----
    path = hf_hub_download(mid, "model.safetensors")
    f = safe_open(path, "pt")
    vec = lambda n: f.get_tensor("model.language_model." + n).float().contiguous().numpy().tobytes()  # noqa: E731
    tensors = []
    for name, (codes, S, Mn) in (("rtn", emb_rtn), ("gptq", emb_g)):
        q, s = pack(codes, S, Mn)
        with open(os.path.join(args.out, f"embed_{name}.bin"), "wb") as o:
            o.write(q)
            o.write(s)
        if name == "rtn":
            tensors += [{"key": "embed.q", "bytes": len(q)}, {"key": "embed.s", "bytes": len(s)}]
    with open(os.path.join(args.out, "rest.bin"), "wb") as o:
        def put(key, b):
            o.write(b)
            tensors.append({"key": key, "bytes": len(b)})

        put("finalNorm", vec("embedding_norm.weight"))
        for i in range(L):
            p = f"layers.{i}."
            put(f"{i}.opNorm", vec(p + "operator_norm.weight"))
            put(f"{i}.ffnNorm", vec(p + "ffn_norm.weight"))
            put(f"{i}.postOp", vec(p + "post_operator_norm.weight"))
            put(f"{i}.postFfn", vec(p + "post_ffn_norm.weight"))
            if hasattr(layers[i], "conv"):
                put(f"{i}.convW", vec(p + "conv.conv.weight"))
            else:
                put(f"{i}.qNorm", vec(p + "self_attn.q_norm.weight"))
                put(f"{i}.kNorm", vec(p + "self_attn.k_norm.weight"))
            for key in ("w13", "w2", "inProj", "outProj", "qkvg", "oProj"):  # weightList() order
                if f"{i}.{key}" in packed:
                    q, s = packed[f"{i}.{key}"]
                    put(f"{i}.{key}.q", q)
                    put(f"{i}.{key}.s", s)
    total = sum(t["bytes"] for t in tensors)
    man = {"format": "q4g64", "group": GS, "model": mid, "tensors": tensors, "total_bytes": total, "quality": quality,
           "calibration": {"chat_windows": args.chat, "web_windows": args.web, "seqlen": args.seqlen},
           "held_out": {"windows": len(held), "seqlen": args.seqlen}}
    with open(os.path.join(args.out, "manifest.json"), "w") as o:
        json.dump(man, o, indent=1)
    log("GPTQ_DONE", json.dumps({"total_mb": round(total / 2 ** 20, 1), **quality}))


if __name__ == "__main__":
    main()
