// SPDX-License-Identifier: Apache-2.0. Copyright 2026 Toaster Club LLC.
// The happy path: load the model into any page with one call.
//
//   import { loadLFM3 } from "./engine/loader.js";
//   const { chat } = await loadLFM3({ onProgress: (p) => console.log(p.got, p.total) });
//   const { reply } = await chat.send("Hello!");
//
// The weight sets come from the site (weights.js lists them; each part is a PNG byte container). Every part is
// kept in this browser's IndexedDB as it arrives, so the next load reads the device's copy, and an interrupted
// first load resumes with the parts it already has. The tokenizer is kept beside it. loadVision(), loadAudio()
// and loadSpeech() load the image model, the speech encoder and the spoken-reply models the same way, each the
// first time it is needed. loadPair() loads Voice Desk: LFM3-Omni-2B and d1-Omni-2B together (engine/pair.js).

import { initDevice, LFM3 } from "./lfm3.js";
import { Chat } from "./chat.js";
import { Tokenizer } from "./tokenizer.js";
import { Vision } from "./vision.js";
import { Audio } from "./audio.js";
import { Codec, DepthDecoder } from "./speech.js";
import { pngPayload, prefetched } from "./pngbytes.js";
import { persist, room, store } from "./store.js";
import { Decider } from "./d1.js";
import { Pair } from "./pair.js";

const siteRoot = () => new URL("./", document.baseURI);

// one part's bytes, retried with backoff
export async function fetchBytes(url) {
  for (let a = 0; ; a++) {
    try {
      const r = await fetch(url);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return new Uint8Array(await r.arrayBuffer());
    } catch (e) {
      if (a >= 3) throw new Error(`could not fetch the model's weights (${e.message})`);
      await new Promise((res) => setTimeout(res, 1000 * 2 ** a));
    }
  }
}

// weights.js, the list of weight sets (window.LFM3_WEIGHTS, LFM3_VISION, LFM3_AUDIO, LFM3_TALK, LFM3_CODEC), if the
// page did not load it already with <script src="weights.js">
export async function ensureWeights(base = siteRoot()) {
  if (!globalThis.LFM3_WEIGHTS) await import(new URL("weights.js", base).href);
  if (!globalThis.LFM3_WEIGHTS) throw new Error("weights.js lists no language model");
}

// ---- weights on this device (IndexedDB): each decoded part of a weight set is kept as it streams in, under
// <family>/<id>/<index>; <family>/<id>/complete marks a full copy. Later loads read the parts back instead of
// downloading, and an interrupted first load resumes with the parts it already kept. Families: w/ the language
// model, v/ the image model, a/ the speech encoder, s/ the speech decoder, c/ the voice codec.
export function weightSet(W, family, { base = siteRoot() } = {}) {
  const prefix = `${family}${W.id}/`, n = W.chunks.length;
  return {
    W, prefix, n,
    async saved() { // parts on this device and whether the copy is complete (null: no storage)
      try {
        const keys = (await store.keys()).map(String).filter((k) => k.startsWith(prefix));
        return { n: keys.filter((k) => !k.endsWith("complete")).length, complete: keys.includes(prefix + "complete") };
      } catch { return null; }
    },
    source() {
      const src = { fromDevice: 0, saved: 0, saving: true, saveError: null };
      const loadOne = async (c, i) => {
        try {
          const ab = await store.get(prefix + i);
          if (ab && ab.byteLength === c.bytes) { src.fromDevice++; return new Uint8Array(ab); }
        } catch { /* not on this device: download it */ }
        const data = await pngPayload(await fetchBytes(new URL(c.url, base)), c.bytes);
        if (src.saving) {
          try { await store.put(prefix + i, data.buffer); src.saved++; } catch (e) { src.saving = false; src.saveError = e; }
        }
        return data;
      };
      src.prequant = { id: W.id, manifest: W.tensors, chunks: () => prefetched(W.chunks, loadOne) };
      return src;
    },
    async prepare(src) { // keep a copy only if the browser has storage and room for the parts still missing
      const s = await this.saved();
      if (s?.complete) return s;
      await persist();
      const free = await room(), need = W.total * (1 - (s?.n || 0) / n);
      if (s === null) { src.saving = false; src.saveError = new Error("storage is not available here"); }
      else if (free !== null && free < need * 1.05) { src.saving = false; src.saveError = new Error("not enough room"); }
      return s;
    },
    async finish(src) { // mark a full copy, or drop a partial one that cannot be finished
      try {
        if (src.fromDevice + src.saved === n) {
          await store.put(prefix + "complete", { total: W.total, at: new Date().toISOString() });
          await store.delWhere((k) => k.startsWith(family) && !k.startsWith(prefix));  // older versions
          return "saved";
        }
        if (src.saveError) await store.delWhere((k) => k.startsWith(prefix));
      } catch { /* storage unavailable */ }
      return src.saveError ? "failed" : "partial";
    },
  };
}

// the tokenizer, kept on the device beside the weights
export async function cachedTokenizer(base = siteRoot()) {
  const KEY = "tok/lfm3-v1";
  try {
    const t = await store.get(KEY);
    if (t) return Tokenizer.fromParts(JSON.parse(t.vocab), t.merges);
  } catch { /* fetch it */ }
  const get = async (n) => { const r = await fetch(new URL(n, base)); if (!r.ok) throw new Error(`tokenizer: HTTP ${r.status}`); return r.text(); };
  const [vocab, merges] = await Promise.all([get("lfm3-tokenizer-vocab.json"), get("lfm3-tokenizer-merges.txt")]);
  try { await store.put(KEY, { vocab, merges }); } catch { /* keep going without */ }
  return Tokenizer.fromParts(JSON.parse(vocab), merges);
}

// one weight set into one model class (Vision, Audio, DepthDecoder, Codec), through the device's copy
async function loadSet(device, W, family, Cls, { base, onProgress = () => {} }) {
  const set = weightSet(W, family, { base });
  const src = set.source();
  await set.prepare(src);
  const m = new Cls(device, { quant: W.format, prequant: src.prequant });
  await m.load(onProgress);
  await set.finish(src);
  return m;
}

// The language model, its tokenizer and a chat that has prefilled its system prompt.
//   base: the site's root (where weights.js, weights/ and the tokenizer are); default: this page's folder
//   ctx: the context length in tokens; onProgress({ got, total }): bytes uploaded to the GPU
//   device: a GPUDevice to share (default: a new one from initDevice())
// Returns { device, info, model, tokenizer, chat, fromDevice } (fromDevice: every part came from this device).
export async function loadLFM3({ base = siteRoot(), ctx = 4096, onProgress = () => {}, device = null } = {}) {
  await ensureWeights(base);
  const gpu = device ? { device, info: {} } : await initDevice();
  const W = globalThis.LFM3_WEIGHTS, set = weightSet(W, "w/", { base });
  const src = set.source();
  await set.prepare(src);
  const model = new LFM3(gpu.device, { ctx, prequant: src.prequant });
  const tokP = cachedTokenizer(base);
  await model.load(onProgress);
  await set.finish(src);
  const tokenizer = await tokP;
  const chat = new Chat(model, tokenizer);
  await chat.newChat();
  return { device: gpu.device, info: gpu.info, model, tokenizer, chat, fromDevice: src.fromDevice === set.n };
}

// The image encoder: photos into the chat (see docs/API.md).
export async function loadVision(device, { base = siteRoot(), onProgress } = {}) {
  await ensureWeights(base);
  if (!globalThis.LFM3_VISION) throw new Error("weights.js lists no image model");
  return loadSet(device, globalThis.LFM3_VISION, "v/", Vision, { base, onProgress });
}

// The speech encoder: spoken questions into the chat.
export async function loadAudio(device, { base = siteRoot(), onProgress } = {}) {
  await ensureWeights(base);
  if (!globalThis.LFM3_AUDIO) throw new Error("weights.js lists no speech encoder");
  return loadSet(device, globalThis.LFM3_AUDIO, "a/", Audio, { base, onProgress });
}

// The spoken replies: { depth, codec } for Chat.speak() and Chat.closeTurn().
export async function loadSpeech(device, { base = siteRoot(), onProgress } = {}) {
  await ensureWeights(base);
  if (!globalThis.LFM3_TALK || !globalThis.LFM3_CODEC) throw new Error("weights.js lists no spoken-reply models");
  const depth = await loadSet(device, globalThis.LFM3_TALK, "s/", DepthDecoder, { base, onProgress });
  const codec = await loadSet(device, globalThis.LFM3_CODEC, "c/", Codec, { base, onProgress });
  return { depth, codec };
}

// ---- Voice Desk: both models on every turn (pair.js) -------------------------------------------------------------
// d1-weights.js, the decision model's weight sets (window.D1_WEIGHTS, D1_VISION, D1_AUDIO), if the page did not load
// it already with <script src="d1-weights.js">
export async function ensureD1Weights(base = siteRoot()) {
  if (!globalThis.D1_WEIGHTS) await import(new URL("d1-weights.js", base).href);
  if (!globalThis.D1_WEIGHTS) throw new Error("d1-weights.js lists no decision model");
}

// LFM3-Omni-2B and d1-Omni-2B, each with its speech encoder, and LFM3's spoken-reply models: a Pair (engine/pair.js)
// ready for its first turn, about 2.1 GB on the GPU. Each set goes through this device's copy, under the same storage
// families as the chat (w/ a/ s/ c/) and Decide (d1w/ d1a/) pages, so either page's saved copy counts.
//   base: the site's root (weights.js, d1-weights.js, weights/, d1/calibration.json, the tokenizer)
//   system: the voice's instructions (a profile's system); onProgress({ got, total, label }) in bytes
//   pictures: true loads both image encoders now (0.4 GB); otherwise call loadPictures() before the first picture
// Returns { device, info, pair, decider, chat, tokenizer, calibration, fromDevice, loadPictures }.
export async function loadPair({ base = siteRoot(), ctx = 4096, system = null, onProgress = () => {}, device = null, pictures = false } = {}) {
  await ensureWeights(base);
  await ensureD1Weights(base);
  const G = globalThis;
  const gpu = device ? { device, info: {} } : await initDevice();
  const sets = [["LFM3_WEIGHTS", "w/", LFM3, "LFM3-Omni-2B"], ["D1_WEIGHTS", "d1w/", LFM3, "d1-Omni-2B"],
    ["LFM3_AUDIO", "a/", Audio, "LFM3's speech encoder"], ["D1_AUDIO", "d1a/", Audio, "d1's speech encoder"],
    ["LFM3_TALK", "s/", DepthDecoder, "the speech decoder"], ["LFM3_CODEC", "c/", Codec, "the voice codec"]];
  for (const [g] of sets) if (!G[g]) throw new Error(`the site lists no ${g}`);
  const load = async (list, report) => {
    const total = list.reduce((s, [g]) => s + G[g].total, 0), out = {};
    let before = 0, fromDevice = 0, parts = 0;
    for (const [g, family, Cls, label] of list) {
      const W = G[g], set = weightSet(W, family, { base }), src = set.source();
      await set.prepare(src);
      const m = Cls === LFM3 ? new LFM3(gpu.device, { ctx, prequant: src.prequant }) : new Cls(gpu.device, { quant: W.format, prequant: src.prequant });
      await m.load((p) => report({ got: before + p.got * (W.total / p.total), total, label }));
      await set.finish(src);
      before += W.total; fromDevice += src.fromDevice; parts += set.n;
      out[g] = m;
    }
    out.fromDevice = fromDevice === parts;
    return out;
  };
  const m = await load(sets, onProgress);
  const tokenizer = await cachedTokenizer(base);
  const r = await fetch(new URL("d1/calibration.json", base));
  if (!r.ok) throw new Error(`d1/calibration.json: HTTP ${r.status}`);
  const calibration = await r.json();
  const decider = new Decider(m.D1_WEIGHTS, tokenizer, { calibration, setSize: calibration.set_size || 16, lead: "<think></think>" });
  const chat = new Chat(m.LFM3_WEIGHTS, tokenizer);
  const pair = new Pair({ decider, d1Audio: m.D1_AUDIO, chat, audio: m.LFM3_AUDIO, depth: m.LFM3_TALK, codec: m.LFM3_CODEC });
  let loading = null;
  const loadPictures = (report = onProgress) => { // both image encoders, once
    if (pair.vision && pair.d1Vision) return Promise.resolve();
    loading ||= load([["LFM3_VISION", "v/", Vision, "LFM3's image encoder"], ["D1_VISION", "d1v/", Vision, "d1's image encoder"]], report)
      .then((v) => { pair.vision = v.LFM3_VISION; pair.d1Vision = v.D1_VISION; })
      .finally(() => { loading = null; });
    return loading;
  };
  if (pictures) await loadPictures();
  await pair.newChat(system);
  return { device: gpu.device, info: gpu.info, pair, decider, chat, tokenizer, calibration, fromDevice: m.fromDevice, loadPictures };
}
