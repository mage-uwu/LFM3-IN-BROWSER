// SPDX-License-Identifier: Apache-2.0. Copyright 2026 Toaster Club LLC.
// Two models on one spoken turn. LiquidAI's d1-Omni-2B hears the clip first and answers named questions about it
// (engine/d1.js); then LFM3-Omni-2B hears the same clip and answers aloud, with d1's decision written into that turn
// as a short note it is told to act on without reading out. Each model has its own speech encoder (d1's release
// changed its encoder too); both read the same log-mel features.
//
//   const pair = new Pair({ decider, d1Audio, chat, audio, depth, codec });
//   await pair.newChat();                                     // the spoken system prompt, with the notes explained
//   const r = await pair.turn(samples16k, questions, { onDecision, onText, onAudio });
//   // r.decision.answers: d1's answers; r.reply: what LFM3 said; r.note: the note it was given
import { logMel } from "./audio.js";

// How the reply model is told about the notes (the end of its system prompt, before the speech instruction)
export const NOTES_INSTRUCTION = "Each message from the user may come with notes from a decision model that heard it first. " +
  "Use them to shape your answer, but do not read them out or mention the decision model.";

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
  // decider: a Decider on d1's language model; d1Audio: d1's speech encoder; chat: a Chat on LFM3-Omni-2B;
  // audio: LFM3-Omni-2B's speech encoder; depth, codec: its spoken-reply models (engine/speech.js)
  constructor({ decider, d1Audio, chat, audio, depth, codec }) {
    Object.assign(this, { decider, d1Audio, chat, audio, depth, codec });
  }

  // a new conversation: the reply model's system prompt explains the notes and asks for spoken replies
  async newChat(system = null) {
    this.chat.spoken = true;
    this.chat.system = `${(system || this.chat.system || "You are a helpful personal assistant.").replace(NOTES_INSTRUCTION, "").trim()} ${NOTES_INSTRUCTION}`;
    await this.chat.newChat();
  }

  // one spoken turn: samples are 16 kHz mono (up to 30 s). questions: as Decider.ask takes them. withNote: false
  // answers without d1's decision (a comparison). state: anything d1 should read beside the clip (text or JSON).
  // note: a note to give instead of d1's (to check that the reply follows the note rather than the clip).
  async turn(samples, questions, { onDecision, onText, onAudio, shouldStop, maxFrames = 750, withNote = true, state = null, greedy = false, note: given = null } = {}) {
    const t0 = performance.now();
    const lm = logMel(samples);
    const a1 = await this.d1Audio.encode(lm.mel, lm.frames);
    const decision = await this.decider.ask(state, questions, { media: { items: [{ token: "<|audio|>", rows: a1.rows }], emb: a1.emb } });
    const t1 = performance.now();
    const note = given ?? decisionNote(questions, decision.answers);   // given: a note of the caller's own (tests)
    onDecision?.(decision, note);
    const a2 = await this.audio.encode(lm.mel, lm.frames);
    this.chat.model.setExternalEmbeddings(a2.emb);
    const res = await this.chat.speak(withNote ? note : "", { media: { token: "<|audio|>", rows: a2.rows }, depth: this.depth, codec: this.codec,
      onText, onAudio, shouldStop, maxFrames, greedy, audio: samples });
    return { decision, note: withNote ? note : null, ...res, decideMs: t1 - t0, seconds: res.seconds, heard: samples.length / 16000 };
  }
}
