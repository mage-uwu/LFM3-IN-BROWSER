// SPDX-License-Identifier: Apache-2.0. Copyright 2026 Toaster Club LLC.
// Two models on one turn, with any of the inputs LFM3-Omni-2B takes. LiquidAI's d1-Omni-2B reads the turn first
// (typed text, a picture, speech, or a camera frame and speech together) and answers named questions about it
// (engine/d1.js); then LFM3-Omni-2B takes the same turn and answers aloud, with d1's decision written into that turn
// as a short note it is told to act on without reading out. Each model has its own image and speech encoders (d1's
// release changed them too); both read the same preprocessed pixels and log-mel features.
//
//   const pair = new Pair({ decider, d1Audio, d1Vision, chat, audio, vision, depth, codec });
//   await pair.newChat(instructions);
//   await pair.turn({ text, image, samples }, questions, { onDecision, onText, onAudio });   // any of the three
//
// Live, as the user talks: the camera frame and the speech go into LFM3 while they are spoken; at the release d1
// decides on the frame and the whole clip, and its note closes LFM3's turn, so the answer starts about a second after
// the user lets go.
//   await pair.liveOpen(image);                 // the frame as the user starts to speak (or null)
//   await pair.liveFeed(melStream);             // every few hundred ms while they talk
//   const r = await pair.liveClose(melStream, questions, { onDecision, onText, onAudio });   // after melStream.finish()
import { CFG, buf } from "./lfm3.js";
import { logMel } from "./audio.js";
import { preprocess } from "./vision.js";

// How the reply model is told about the notes (the end of its system prompt, before the speech instruction)
export const NOTES_INSTRUCTION = "Each message from the user may come with notes from a decision model that read it first. " +
  "Use them to shape your answer, but do not read them out or mention the decision model.";
export const LIVE_PIXELS = 102400;   // a camera frame as the chat's Live reads it: about 320x320, 100 image tokens

const pc = (p) => `${Math.round(100 * p)}%`;
// d1's answers as one line of text: each question's name and verdict, with its probability
export function decisionNote(questions, answers) {
  const parts = [];
  for (const [name, a] of Object.entries(answers)) {
    const q = questions[name] || {};
    const label = name.replace(/_/g, " ");
    if (a.type === "noul") parts.push(`${label}: ${a.noul >= 0.5 ? "yes" : "no"} (${pc(a.noul >= 0.5 ? a.noul : 1 - a.noul)})`);
    else if (a.choices) parts.push(`${label}: ${a.choices.length ? a.choices.join(", ") : "none"}`);
    else if (a.type === "choice") parts.push(`${label}: ${a.choice} (${pc(a.confidence)})`);
    else if (a.type === "score") parts.push(`${label}: ${q.criteria?.[a.score] ?? a.score} (level ${a.score} of ${(q.criteria?.length || 1) - 1})`);
  }
  return `[Notes from the decision model on this message: ${parts.join("; ")}.]`;
}

export class Pair {
  // decider: a Decider on d1's language model; d1Audio, d1Vision: d1's encoders; chat: a Chat on LFM3-Omni-2B;
  // audio, vision: its encoders; depth, codec: its spoken-reply models (engine/speech.js). The vision encoders may be
  // left out (null) until a picture is needed; set pair.d1Vision and pair.vision when they load.
  constructor({ decider, d1Audio, d1Vision = null, chat, audio, vision = null, depth, codec }) {
    Object.assign(this, { decider, d1Audio, d1Vision, chat, audio, vision, depth, codec });
    this.device = chat.model.device;
    this.joined = new Map();   // per model: one buffer holding a picture's and a clip's rows back to back
    this.live = null;
  }

  // a new conversation: the reply model's system prompt explains the notes and asks for spoken replies
  async newChat(system = null) {
    this.live = null;
    this.chat.spoken = true;
    this.chat.system = `${(system || this.chat.system || "You are a helpful personal assistant.").replace(NOTES_INSTRUCTION, "").trim()} ${NOTES_INSTRUCTION}`;
    await this.chat.newChat();
  }

  // one model's rows for a turn's inputs (a preprocessed picture, then log-mel), in the order the turn holds them
  async encode(side, pp, lm) {
    const parts = [], items = [];
    if (pp) {
      const v = side === "d1" ? this.d1Vision : this.vision;
      if (!v) throw new Error("the image encoders are not loaded");
      const o = await v.encode(pp.patches, pp.gh, pp.gw);
      parts.push(o); items.push({ token: "<|image|>", rows: o.rows });
    }
    if (lm) {
      const o = await (side === "d1" ? this.d1Audio : this.audio).encode(lm.mel, lm.frames);
      parts.push(o); items.push({ token: "<|audio|>", rows: o.rows });
    }
    return { items, emb: this.join(side, parts) };
  }
  join(side, parts) { // the parts' rows in one buffer (a single part is used as it is)
    if (!parts.length) return null;
    if (parts.length === 1) return parts[0].emb;
    const d = this.device, rows = parts.reduce((a, p) => a + p.rows, 0), row = CFG.D * 4;
    const out = buf(d, rows * row);
    const enc = d.createCommandEncoder();
    let off = 0;
    for (const p of parts) { enc.copyBufferToBuffer(p.emb, 0, out, off, p.rows * row); off += p.rows * row; }
    d.queue.submit([enc.finish()]);
    this.joined.get(side)?.destroy();
    this.joined.set(side, out);
    return out;
  }
  // d1's decision on a turn: the text is its state; the picture and the clip are its media
  async decide({ text = null, pp = null, lm = null, state = null }, questions) {
    const m = await this.encode("d1", pp, lm);
    const st = state ?? (text && text.trim() ? text.trim() : null);
    return this.decider.ask(st, questions, { media: m.items.length ? { items: m.items, emb: m.emb } : null });
  }
  static picture(image, maxPixels) { // { rgba, w, h } -> the encoders' patches
    return image ? preprocess(image.rgba, image.w, image.h, maxPixels ? { maxPixels } : {}) : null;
  }
  // the reply model's text for a turn: what the user typed, then the note
  static replyText(text, note) { return [text?.trim() || "", note || ""].filter(Boolean).join("\n\n"); }

  // one turn, whole. input: { text, image: { rgba, w, h }, samples (16 kHz mono, up to 30 s) }, any of them.
  // withNote: false answers without d1's decision (a comparison); note: a note to give instead of d1's (tests);
  // state: what d1 reads in place of the text; maxPixels: the picture's size for both encoders.
  async turn(input, questions, { onDecision, onText, onAudio, shouldStop, maxFrames = 750, withNote = true, note: given = null,
    state = null, greedy = false, maxPixels = 0 } = {}) {
    const { text = "", image = null, samples = null } = input;
    if (!text.trim() && !image && !samples) throw new Error("a turn needs text, a picture or speech");
    const t0 = performance.now();
    const pp = Pair.picture(image, maxPixels), lm = samples ? logMel(samples) : null;
    const decision = await this.decide({ text, pp, lm, state }, questions);
    const t1 = performance.now();
    const note = given ?? decisionNote(questions, decision.answers);
    onDecision?.(decision, note);
    const m = await this.encode("lfm3", pp, lm);
    if (m.emb) this.chat.model.setExternalEmbeddings(m.emb);
    const res = await this.chat.speak(Pair.replyText(text, withNote ? note : ""), { media: m.items.length ? m.items : null,
      depth: this.depth, codec: this.codec, onText, onAudio, shouldStop, maxFrames, greedy, audio: samples });
    return { decision, note: withNote ? note : null, ...res, decideMs: t1 - t0, heard: samples ? samples.length / 16000 : 0,
      imageRows: m.items.find((x) => x.token === "<|image|>")?.rows || 0 };
  }

  // ---- Live: the turn streamed into LFM3 while the user talks ----------------------------------------------------
  async liveOpen(image = null, { maxPixels = LIVE_PIXELS } = {}) {
    const pp = Pair.picture(image, maxPixels);
    let pic = null;
    if (pp) {
      if (!this.vision) throw new Error("the image encoders are not loaded");
      pic = await this.vision.encode(pp.patches, pp.gh, pp.gw);
    }
    await this.chat.openTurn(pic ? { emb: pic.emb, rows: pic.rows } : null);
    this.live = { pp, rows: pic?.rows || 0 };
  }
  // the speech that became final since the last call (mel: an engine/audio.js MelStream fed as the user talks)
  async liveFeed(mel) {
    if (!this.live) return;
    const final = mel.complete();
    if (final <= this.chat.turn.rows) return;
    const width = Math.min(3000, Math.ceil((mel.frames + 1) / 64) * 64);   // a few widths, so encoder plans are reused
    const o = await this.audio.encode(mel.layout(width), width);
    await this.chat.feedAudio(o.emb, Math.min(final, o.rows));
  }
  // at the release, after mel.finish(): the clip's last rows into LFM3, d1's decision on the frame and the whole clip,
  // and LFM3's spoken answer with the note
  async liveClose(mel, questions, { onDecision, onText, onAudio, shouldStop, maxFrames = 750, greedy = false, withNote = true, note: given = null } = {}) {
    const L = this.live;
    if (!L) throw new Error("no live turn is open");
    this.live = null;
    const t0 = performance.now();
    const o = await this.audio.encode(mel.layout(), mel.frames);
    const streamed = this.chat.turn.rows;
    await this.chat.feedAudio(o.emb, o.rows);
    const decision = await this.decide({ pp: L.pp, lm: { mel: mel.layout(), frames: mel.frames } }, questions);
    const t1 = performance.now();
    const note = given ?? decisionNote(questions, decision.answers);
    onDecision?.(decision, note);
    const samples = mel.x.slice(0, mel.n);
    const res = await this.chat.closeTurn(withNote ? note : "", { depth: this.depth, codec: this.codec, onText, onAudio, shouldStop, maxFrames, greedy, audio: samples });
    return { decision, note: withNote ? note : null, ...res, decideMs: t1 - t0, heard: samples.length / 16000, imageRows: L.rows, streamed };
  }
  liveCancel() { // a live turn that will not be answered: back to the start of the conversation
    if (!this.live) return;
    this.live = null;
    this.chat.turn = null;
    this.chat.restartFromPrefix();
  }
}
