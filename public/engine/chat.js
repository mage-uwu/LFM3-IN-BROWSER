// SPDX-License-Identifier: Apache-2.0. Copyright 2026 Toaster Club LLC.
// Chat on top of the WebGPU engine: LFM3's chat template, the official tokenizer (engine/tokenizer.js, from the
// repository's tokenizer.json shipped beside the page), spoken turns and speech recognition. Retrieval for a chat
// (its documents and older turns) is engine/rag.js; each chat starts from the system prompt alone.
import { CFG } from "./lfm3.js";
import { Tokenizer } from "./tokenizer.js";
import { SCFG } from "./speech.js";

export const SYSTEM = "You are a helpful personal assistant.";
// spoken replies: the model's interleaved text-and-speech mode. The model card's instruction for it ends the system
// prompt ("append the corresponding instructions to the end of your system prompt").
export const SPEECH_INSTRUCTION = "Respond with interleaved text and audio.";
export const SPEECH_SYSTEM = `${SYSTEM} ${SPEECH_INSTRUCTION}`;
export async function loadTokenizer(base = new URL("./", document.baseURI).href) {
  return Tokenizer.fromFiles(base);
}

const BOS = "<|startoftext|>";
// media: { token: "<|image|>" (or "<|audio|>"), rows } placed before the text, as the official template does; the
// placeholders become the rows of the model's external embeddings (LFM3.setExternalEmbeddings)
// (one item or a list, e.g. a photo and a spoken question: their rows sit back to back in that order)
const mediaList = (m) => (!m ? [] : Array.isArray(m) ? m : [m]);
const mediaBlock = (m) => mediaList(m).map((x) => `<|vision_start|>${x.token.repeat(x.rows)}<|vision_end|>`).join("");
export const turnUser = (text, media = null) => `<|im_start|>user\n${mediaBlock(media)}${text}<|im_end|>\n<|im_start|>assistant\n<think></think>`;
// prompt ids with each media placeholder replaced by 0x80000000 + its row
export function mediaIds(ids, tok, media) {
  const items = mediaList(media);
  if (!items.length) return ids;
  const next = new Map();   // placeholder id -> next row (each kind's rows follow the earlier items' rows)
  let off = 0;
  for (const x of items) { next.set(tok.added.get(x.token), off); off += x.rows || 0; }
  return ids.map((i) => {
    if (!next.has(i)) return i;
    const r = next.get(i);
    next.set(i, r + 1);
    return 0x80000000 + r;
  });
}
export const systemBlock = (text) => `${BOS}<|im_start|>system\n${text}<|im_end|>\n`;

// ---- the chat session ----------------------------------------------------------------------------------------
export class Chat {
  constructor(model, tok) {
    this.model = model;
    this.tok = tok;
    this.messages = [];
    this.pendingEnd = false;  // the last reply's <|im_end|> is not in the cache yet
    this.prefix = null;       // snapshot after the system prompt
    this.system = SYSTEM;     // the base system prompt: the user's own instructions, when they set some
  }
  ids(text) { return Array.from(this.tok.encode(text, { add_special_tokens: false })); }
  decode(ids) { return this.tok.decode(ids.filter((i) => !CFG.stopIds.includes(i)), { skip_special_tokens: true }); }

  // prefill the system prompt once and keep a snapshot: each new chat restores it
  async newChat() {
    const m = this.model;
    m.reset();
    const base = (this.system || SYSTEM).trim();
    const sys = this.spoken ? `${base} ${SPEECH_INSTRUCTION}` : base;
    const ids = this.ids(systemBlock(sys));
    const t0 = performance.now();
    await m.prefill(ids, { sample: false });
    this.prefixInfo = { tokens: ids.length, ms: performance.now() - t0 };
    if (this.prefix) this.prefix.layers.forEach((s) => Object.values(s).forEach((b) => b.destroy?.()));
    this.prefix = await m.snapshot();
    this.messages = [];
    this.pendingEnd = false;
    return this.prefixInfo;
  }
  restartFromPrefix() { this.model.restore(this.prefix); this.model.nout = 0; this.messages = []; this.pendingEnd = false; }

  async send(text, { onText, maxNew = 512, shouldStop, media = null, audio = null } = {}) {
    const m = this.model;
    const turn = (this.pendingEnd ? "<|im_end|>\n" : "") + turnUser(text, media);
    const ids = mediaIds(this.ids(turn), this.tok, media);
    const t0 = performance.now();
    await m.prefill(ids, { sample: true });
    const t1 = performance.now();
    const out = [];
    let first = 0;
    await m.generate(maxNew, {
      onToken: (id) => { if (!first) first = performance.now(); out.push(id); onText?.(this.decode(out)); },
      shouldStop,
    });
    const t2 = performance.now();
    const reply = this.decode(out);
    this.messages.push({ role: "user", content: text, media: mediaList(media).map((x) => x.token), audio }, { role: "assistant", content: reply });
    this.pendingEnd = true;
    const n = out.length;
    return { reply, stats: { promptTokens: ids.length, prefillMs: t1 - t0, prefillTokS: ids.length / ((t1 - t0) / 1000),
      newTokens: n, decodeTokS: n > 1 ? (n - 1) / ((t2 - first) / 1000) : 0, pos: m.pos } };
  }

  // A spoken reply (the model's interleaved mode): blocks of 6 text tokens and 12 speech frames; after <|text_end|>
  // speech only until the end-of-audio frame, then text to the end of the turn. depth: a DepthDecoder, codec: a Codec
  // (engine/speech.js). onText(text) streams the reply's text, onAudio(Float32Array, 24 kHz) its sound, about every
  // 320 ms. greedy: argmax codes in the fixed codebook order (tests).
  async speak(text, opts = {}) {
    const media = opts.media || null;
    const turn = (this.pendingEnd ? "<|im_end|>\n" : "") + turnUser(text, media) + "<|mixed_start|>";
    const ids = mediaIds(this.ids(turn), this.tok, media);
    return this.spokenReply(ids, { text, media: mediaList(media).map((x) => x.token), promptTokens: ids.length, audio: opts.audio }, opts);
  }

  // A spoken turn streamed in while the user talks (push to talk), as the official prefill_audio does: openTurn()
  // prefills the turn's opening and any picture at the press, feedAudio() the speech encoder's rows as they become
  // final, and closeTurn() the closing tokens at the release, then answers like speak(). The model ends up in the
  // same state as speak() with the whole clip, so the answer starts as soon as the user lets go.
  async openTurn(picture = null) { // picture: { emb, rows } from the vision tower
    const m = this.model;
    let ids = this.ids((this.pendingEnd ? "<|im_end|>\n" : "") + "<|im_start|>user\n");
    if (picture) {
      m.setExternalEmbeddings(picture.emb);
      ids = ids.concat(this.ids("<|vision_start|>"), Array.from({ length: picture.rows }, (_, r) => 0x80000000 + r), this.ids("<|vision_end|>"));
    }
    ids = ids.concat(this.ids("<|vision_start|>"));
    await m.prefill(ids, { sample: false });
    this.pendingEnd = false;
    this.turn = { rows: 0, media: picture ? ["<|image|>", "<|audio|>"] : ["<|audio|>"], tokens: ids.length };
  }
  async feedAudio(emb, rows) { // rows [0, rows) of emb are final: those not in the model yet go in
    const t = this.turn;
    if (!t || rows <= t.rows) return;
    this.model.setExternalEmbeddings(emb);
    const ids = [];
    for (let r = t.rows; r < rows; r++) ids.push(0x80000000 + r);
    await this.model.prefill(ids, { sample: false });
    t.rows = rows;
    t.tokens += ids.length;
  }
  async closeTurn(text, opts = {}) {
    const t = this.turn, ids = this.ids(`<|vision_end|>${text}<|im_end|>\n<|im_start|>assistant\n<think></think><|mixed_start|>`);
    this.turn = null;
    return this.spokenReply(ids, { text, media: t.media, promptTokens: t.tokens + ids.length, streamed: t.rows, audio: opts.audio }, opts);
  }

  async spokenReply(ids, info, { onText, onAudio, shouldStop, depth, codec, greedy = false, maxFrames = 750, maxText = 400 } = {}) {
    const m = this.model, { nText, nAudio, EOS, textEnd, imEnd } = SCFG;
    const t0 = performance.now();
    await m.prefill(ids, { sample: true });
    const t1 = performance.now();
    codec.reset();
    const textIds = [], frames = [], waiting = [];
    let speaking = false, slots = nText, textDone = false, interleaved = true, ended = false, samples = 0;
    const play = async (final) => {
      const a = await codec.decode(waiting.splice(0), { final });
      samples += a.length;
      if (a.length) onAudio?.(a);
    };
    while (!ended && !shouldStop?.()) {
      if (!speaking) { // text: the token the last step sampled, then the rest of the block
        const toks = await m.generate(interleaved ? slots : maxText, { batch: 8 });
        let stop = 0;
        for (const t of toks) { textIds.push(t); if (t === textEnd || t === imEnd) { stop = t; break; } }
        onText?.(this.decode(textIds));
        if (stop === imEnd || !interleaved || textIds.length > maxText) { ended = true; break; }
        if (stop === textEnd) textDone = true; else slots = nAudio;
        speaking = true;
        await m.prefill([textIds[textIds.length - 1]], { sample: false, hidden: true });   // feed the block's last token
        continue;
      }
      // speech: one frame from the hidden state, fed back as the next input
      let codes = await depth.sampleFrame(m.hlast, frames.slice(-SCFG.history), { greedy });
      const eos = codes.some((c) => c === EOS);
      let toText = false;
      if (eos) { codes = new Array(12).fill(EOS); interleaved = false; speaking = false; toText = true; }
      else {
        frames.push(codes); waiting.push(codes);
        if (waiting.length >= 4) await play(false);
        if (!textDone && --slots === 0) { speaking = false; slots = nText; toText = true; }
      }
      m.setExternalEmbeddings(depth.embedFrame(codes));
      await m.prefill([0x80000000], toText ? { sample: true } : { sample: false, hidden: true });
      if (frames.length >= maxFrames) break;
    }
    await play(true);
    const t2 = performance.now();
    const reply = this.decode(textIds);
    this.messages.push({ role: "user", content: info.text, media: info.media, audio: info.audio || null }, { role: "assistant", content: reply });
    this.pendingEnd = true;
    return { reply, textIds, frames, seconds: samples / 24000,
      stats: { promptTokens: info.promptTokens, streamed: info.streamed || 0, prefillMs: t1 - t0, frames: frames.length, ms: t2 - t1, pos: m.pos } };
  }

  // The user's speech as text, by the model itself (the model card's "Perform ASR in <language>." system prompt):
  // emb holds the speech encoder's rows for the clip. It resets the model's state, so it runs between chats.
  async asr(emb, rows, language = "English") {
    const m = this.model, media = { token: "<|audio|>", rows };
    m.reset();
    m.setExternalEmbeddings(emb);
    const ids = mediaIds(this.ids(systemBlock(`Perform ASR in ${language}.`) + turnUser("", media)), this.tok, media);
    await m.prefill(ids, { sample: true });
    return this.decode(await m.generate(160)).trim();
  }
}
