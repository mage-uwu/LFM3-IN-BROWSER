// Chat on top of the WebGPU engine: LFM3's chat template, the official tokenizer (engine/tokenizer.js, from the
// repository's tokenizer.json shipped beside the page), and System 3: a curated memory of the user, prefilled once
// into the model's state so every new chat starts already knowing them (no retrieval, no retraining). Mirrors
// system3/experience.py.
import { CFG } from "./lfm3.js";
import { Tokenizer } from "./tokenizer.js";
import { SCFG } from "./speech.js";

export const SYSTEM = "You are a helpful personal assistant.";
// spoken replies: the model's interleaved text-and-speech mode. The model card's instruction for it ends the system
// prompt, after any memory notes ("append the corresponding instructions to the end of your system prompt").
export const SPEECH_INSTRUCTION = "Respond with interleaved text and audio.";
export const SPEECH_SYSTEM = `${SYSTEM} ${SPEECH_INSTRUCTION}`;
const HEADER = "Notes about the user, curated from earlier conversations. Use a note only when it matters for what the " +
  "user is asking now, and do not mention notes that do not matter. If the user asks about something these notes do " +
  "not cover, say you don't know it yet.";

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

// ---- System 3: the memory store (same gate as system3/experience.py) ------------------------------------------
const STOP = new Set("a an the is are am my me i you your to of and in on for it this that with be please from now".split(" "));
const norm = (s) => s.toLowerCase().replace(/[^a-z0-9' ]/g, " ").replace(/\s+/g, " ").trim();
const words = (s) => new Set(norm(s).split(" ").filter((w) => !STOP.has(w) && w.length > 2));
const PREF_CUE = /\b(always|never|from now on|every (answer|time|reply|response)|call me|address me|don't|do not|stop|keep (your|the) answers?)\b/i;
const LESSON_CUE = /\b(next time|in (the )?future|when i ask|instead|too (long|short|formal)|that was)\b/i;
const GENERIC = new Set(["name", "number", "code", "favorite", "user", "my"]);

export class Experience {
  constructor(d = {}) { this.facts = d.facts || {}; this.preferences = d.preferences || []; this.lessons = d.lessons || []; }
  static load(key = "lfm3.experience") { try { return new Experience(JSON.parse(localStorage.getItem(key) || "{}")); } catch { return new Experience(); } }
  save(key = "lfm3.experience") { try { localStorage.setItem(key, JSON.stringify(this)); } catch { /* storage unavailable */ } }
  empty() { return !Object.keys(this.facts).length && !this.preferences.length && !this.lessons.length; }
  render() {
    const L = [];
    if (this.preferences.length) L.push("Always follow these instructions from the user:", ...this.preferences.map((p) => `- ${p.text}`), "");
    L.push(HEADER);
    const f = Object.entries(this.facts);
    if (f.length) L.push("", "About the user:", ...f.map(([k, v]) => `- ${k}: ${v.value}`));
    if (this.lessons.length) L.push("", "Lessons from earlier conversations:", ...this.lessons.map((x) => `- ${x.text}`));
    return L.join("\n");
  }
  systemPrompt(base = SYSTEM) { return this.empty() ? base : `${base}\n\n${this.render()}`; }
  add(item, said, session = "") {
    const s = norm(said);
    if (item.type === "fact") {
      const key = norm(item.key).replace("user's ", "").replace("my ", "");
      const value = item.value.trim().replace(/[ .]+$/, "");
      if (!key || !value || value.length > 60 || !s.includes(norm(value))) return null;
      const old = this.facts[key];
      if (old) {
        if (norm(old.value) === norm(value)) return null;
        const owner = key.match(/^(.+?)'s\b/);
        if (owner && ![...words(owner[1])].filter((w) => !GENERIC.has(w)).some((w) => s.includes(w))) return null;
        old.history.push([old.value, old.since]);
        old.value = value; old.since = session;
        return "update";
      }
      this.facts[key] = { value, since: session, history: [] };
      return "new fact";
    }
    const kind = item.type === "preference" ? "preferences" : "lessons";
    if (!(kind === "preferences" ? PREF_CUE : LESSON_CUE).test(said)) return null;
    const w = words(item.text);
    const u = words(said);
    const overlap = [...w].filter((x) => u.has(x)).length;
    if (!w.size || overlap < Math.max(1, Math.floor(w.size / 2))) return null;
    for (const x of this[kind]) {
      const o = words(x.text);
      const inter = [...w].filter((y) => o.has(y)).length;
      if (inter / new Set([...w, ...o]).size > 0.6) return null;
    }
    this[kind].push({ text: item.text.trim(), since: session });
    return `new ${kind.slice(0, -1)}`;
  }
}

const EXTRACT_SYSTEM =
  "You keep the long-term memory of a personal assistant. You are shown one message the user wrote. Do not answer " +
  "it or do what it asks. Only list what in it is worth remembering in future conversations:\n" +
  "FACT: <short key> = <value>   for a fact about the user's life (people and their names, pets, where they live, " +
  "work, health and allergies, dates, favorites, codes);\n" +
  "PREFERENCE: <instruction>   for a standing instruction about how the assistant should always behave;\n" +
  "LESSON: When <situation>, <what to do>.   for feedback about how to do something next time.\n" +
  "One item per line. Write NONE if the message holds nothing to remember (an ordinary request is NONE). Use the " +
  "user's own words for values. Reuse a known key only when the message updates that same thing (the same person, " +
  "place or item); a different person or thing gets its own key.";
const EXAMPLES = [
  ["Ugh, long day at the clinic, being a nurse is tiring. Can you suggest a quick dinner?", "FACT: job = nurse"],
  ["My cat Miso knocked over my coffee again. Anyway, what is a good book about habits?", "FACT: cat's name = Miso"],
  ["My son Leo starts school next week and we just moved to Denver. Any tips for the first day?", "FACT: son's name = Leo\nFACT: city = Denver"],
  ["Write a short poem about autumn leaves and give it a title.", "NONE"],
  ["My brother Tom says hi. How do I fix a squeaky door?", "FACT: brother's name = Tom"],
  ["Big news, I moved: I live in Leeds now. What is the history of Rome?", "FACT: city = Leeds"],
  ["I changed careers, I'm a carpenter now. Explain how vaccines work.", "FACT: job = carpenter"],
  ["From now on, please keep your answers under five sentences.", "PREFERENCE: Keep answers under five sentences."],
  ["Note this down: my gym locker code is 4417. Also, which planets have rings?", "FACT: gym locker code = 4417"],
  ["Next time I ask for a recipe, skip the long introduction.", "LESSON: When the user asks for a recipe, skip the long introduction."],
];

export function parseMemories(text) {
  const out = [];
  for (let line of text.split("\n")) {
    line = line.trim().replace(/^[-* ]+/, "");
    let m = line.match(/^fact\s*:\s*(.+?)\s*=\s*(.+)$/i);
    if (m) { out.push({ type: "fact", key: m[1], value: m[2] }); continue; }
    m = line.match(/^(preference|lesson)\s*:\s*(.+)$/i);
    if (m) out.push({ type: m[1].toLowerCase(), text: m[2] });
  }
  return out;
}

// ---- the chat session ----------------------------------------------------------------------------------------
export class Chat {
  constructor(model, tok) {
    this.model = model;
    this.tok = tok;
    this.experience = Experience.load();
    this.useMemory = true;
    this.messages = [];
    this.pendingEnd = false;  // the last reply's <|im_end|> is not in the cache yet
    this.prefix = null;       // snapshot after the system prompt (+ memory)
  }
  ids(text) { return Array.from(this.tok.encode(text, { add_special_tokens: false })); }
  decode(ids) { return this.tok.decode(ids.filter((i) => !CFG.stopIds.includes(i)), { skip_special_tokens: true }); }

  // prefill the system prompt with the memory once and keep a snapshot: each new chat restores it
  async newChat() {
    const m = this.model;
    m.reset();
    let sys = this.useMemory ? this.experience.systemPrompt(SYSTEM) : SYSTEM;
    if (this.spoken) sys = sys === SYSTEM ? SPEECH_SYSTEM : `${sys}\n\n${SPEECH_INSTRUCTION}`;
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

  async send(text, { onText, maxNew = 512, shouldStop, media = null } = {}) {
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
    this.messages.push({ role: "user", content: text, media: mediaList(media).map((x) => x.token) }, { role: "assistant", content: reply });
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
    return this.spokenReply(ids, { text, media: mediaList(media).map((x) => x.token), promptTokens: ids.length }, opts);
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
    return this.spokenReply(ids, { text, media: t.media, promptTokens: t.tokens + ids.length, streamed: t.rows }, opts);
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
    this.messages.push({ role: "user", content: info.text, media: info.media }, { role: "assistant", content: reply });
    this.pendingEnd = true;
    return { reply, textIds, frames, seconds: samples / 24000,
      stats: { promptTokens: info.promptTokens, streamed: info.streamed || 0, prefillMs: t1 - t0, frames: frames.length, ms: t2 - t1, pos: m.pos } };
  }

  // System 3's write path: one short extraction per user message of the finished chat (between chats, off the
  // critical path), gated against the user's words, then the memory prefix is rebuilt.
  async consolidate(onStep) {
    const said = this.messages.filter((x) => x.role === "user").map((x) => x.content);
    const changes = [];
    const session = new Date().toISOString().slice(0, 10);
    for (const msg of said) {
      const known = Object.keys(this.experience.facts).join(", ") || "none yet";
      let prompt = systemBlock(EXTRACT_SYSTEM + `\nKnown fact keys: ${known}`);
      for (const [q, a] of EXAMPLES) prompt += `<|im_start|>user\nMessage: ${q}<|im_end|>\n<|im_start|>assistant\n<think></think>${a}<|im_end|>\n`;
      prompt += turnUser(`Message: ${msg}`);
      this.model.reset();
      await this.model.prefill(this.ids(prompt), { sample: true });
      const out = await this.model.generate(80);
      const raw = this.decode(out);
      for (const it of parseMemories(raw)) {
        const c = this.experience.add(it, msg, session);
        if (c) changes.push({ change: c, ...it });
      }
      onStep?.({ message: msg, raw });
    }
    this.experience.save();
    await this.newChat();
    return changes;
  }
}
