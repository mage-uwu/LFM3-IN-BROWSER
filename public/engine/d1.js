// SPDX-License-Identifier: Apache-2.0. Copyright 2026 Toaster Club LLC.
// LiquidAI's d1 decision models in the browser: named questions over a state, answered with calibrated
// probabilities in one forward pass and no generated tokens. A port of the release's own System One runner
// (LiquidAI/d1-Omni-2B: prompt.py, attention.py, calibration.py, api.py) for the question types that read the
// vocabulary at an answer slot:
//   noul      yes or no: { noul: P(yes) }
//   choice    one of named options: { choice, confidence, probabilities }; with multiple: true, each option yes
//             or no: { choices, probabilities }
//   score     a level on a rubric of 2 to 10, lowest first: { score (the median level), confidence, probabilities }
// The prompt is the release's ChatML (no system turn, the state as JSON or text, "QUESTION:" then each question and
// its reply format); a state's questions are packed after it, at most d1_set_size at a time, and read under d1's
// "set" attention (LFM3.decide). The probabilities go through the release's calibration.json.
//
//   const d1 = new Decider(model, tokenizer, { calibration });          // model: an LFM3 with d1's weights
//   const { answers } = await d1.ask("I was charged twice.", {
//     refund: { type: "noul", instructions: "Is the customer asking for a refund?" },
//     team: { type: "choice", instructions: "Which team?", criteria: { billing: "Charges", shipping: "Parcels" } },
//   });
//   // answers.refund.noul -> 0.94; answers.team.choice -> "billing"

import { mediaIds } from "./chat.js";

const IM_START = "<|im_start|>", IM_END = "<|im_end|>";
const YES_FORMS = ["yes", "Yes", "YES"], NO_FORMS = ["no", "No", "NO"];
const LETTERS = Array.from({ length: 26 }, (_, i) => String.fromCharCode(65 + i));
const BREAK = "​";   // after a special token's first character, so the caller's text cannot be a special token
const SPECIAL = /<\|[A-Za-z0-9_]{1,40}\|>/g;

export const plain = (text) => text.replace(SPECIAL, (m) => m[0] + BREAK + m.slice(1));

// the state as the block before "QUESTION:" (the release's default "json_only" style): text as it is, any other
// value as indented JSON
export function stateBlock(state) {
  return typeof state === "string" ? `${state}\n\n` : `${JSON.stringify(state, null, 2)}\n\n`;
}

export function asQuestion(q) {
  if (!q || typeof q !== "object" || typeof q.instructions !== "string") throw new Error("a question is { type, instructions, ... } with text instructions");
  const type = q.type || "choice";
  if (type === "noul") return { type, instructions: q.instructions, criteria: q.criteria || null };
  if (type === "score") {
    if (!Array.isArray(q.criteria) || q.criteria.length < 2 || q.criteria.length > 10) throw new Error('a score\'s "criteria" lists 2 to 10 levels, lowest first');
    return { type, instructions: q.instructions, criteria: q.criteria.map(String) };
  }
  if (type === "choice") {
    if (!q.criteria || typeof q.criteria !== "object" || Array.isArray(q.criteria) || !Object.keys(q.criteria).length) throw new Error('a choice\'s "criteria" names its options: { name: description }');
    return { type, instructions: q.instructions, criteria: q.criteria, multiple: !!q.multiple };
  }
  throw new Error(`question type "${type}" is not in this port yet (noul, choice, score)`);
}

// letters when every label is one, else A..Z (the release's scheme 1 for up to 26 options)
function optionCodes(labels) {
  const labs = labels.map((x) => String(x).trim());
  if (labs.length && labs.every((k) => k.length === 1 && /\p{L}/u.test(k))) return labs;
  if (labs.length <= 26) return LETTERS.slice(0, labs.length);
  return labs.map((_, i) => String(i).padStart(2, "0"));
}

const TEMPERATURE_BUCKET = (q) => {
  if (q.type === "noul") return "noul";
  if (q.type === "score") return `score:${q.criteria.length}`;
  const k = Object.keys(q.criteria).length;
  return k <= 3 ? "choice:small" : k <= 8 ? "choice:mid" : k <= 26 ? "choice:large" : "choice:inventory";
};

// a value through a calibration table's increasing (raw, calibrated) knots, linear between them, flat past the ends
export function through(value, knots) {
  if (!knots?.length) return value;
  let i = 0;
  while (i < knots.length && knots[i][0] < value) i++;
  if (i === 0) return knots[0][1];
  if (i === knots.length) return knots[knots.length - 1][1];
  const [a, pa] = knots[i - 1], [b, pb] = knots[i];
  return pa + ((value - a) * (pb - pa)) / (b - a);
}

export class Decider {
  // model: an LFM3 holding a d1 release's language model; tok: its tokenizer (engine/tokenizer.js);
  // calibration: the release's calibration.json ({ default, temperatures, tables }); setSize: its d1_set_size;
  // lead: what its chat template writes after "<|im_start|>assistant\n"; bos: the sequence's first token
  constructor(model, tok, { calibration = null, setSize = 16, lead = "", bos = "<|startoftext|>", optionStyle = "desc" } = {}) {
    Object.assign(this, { model, tok, calibration, setSize, lead, bos, optionStyle });
    this.single = new Map();
  }
  ids(text) { return Array.from(this.tok.encode(text, { add_special_tokens: false })); }
  // the one token a text is, or null
  token(text) {
    if (!this.single.has(text)) { const ids = this.ids(text); this.single.set(text, ids.length === 1 ? ids[0] : null); }
    return this.single.get(text);
  }
  tokens(texts) { return [...new Set(texts.map((t) => this.token(t)).filter((x) => x !== null))]; }
  // each option's code and its single token (the release's aliases(): a code that is not one token takes the next
  // free one from a fallback pool)
  aliases(labels) {
    const used = new Set(), out = [];
    const pool = [...LETTERS, ...Array.from({ length: 100 }, (_, i) => String(i).padStart(2, "0")),
      ...Array.from({ length: 26 }, (_, i) => String.fromCharCode(97 + i))];
    const take = (raw) => { const t = this.token(raw); if (t === null || used.has(t)) return false; out.push([raw, t]); used.add(t); return true; };
    for (const code of optionCodes(labels)) if (!take(code) && !pool.some(take)) throw new Error(`no single-token code left for ${labels.length} options`);
    return out;
  }

  questionBlock(q) {
    if (q.type === "choice" && q.multiple) {
      const lines = Object.entries(q.criteria).map(([name, d]) => `${JSON.stringify(name)}: ${d || name}`).join("\n");
      return `${q.instructions}\n\nOptions:\n${lines}\n\nReply with yes or no for each option.`;
    }
    if (q.type === "choice") {
      const labels = Object.keys(q.criteria);
      const lines = this.aliases(labels).map(([code], i) => {
        const label = labels[i], desc = q.criteria[label], text = desc || label.replaceAll("_", " ");
        return this.optionStyle === "name_desc" && !/^opt\d+$/.test(label) && label !== text ? `${code} ${label}: ${text}` : `${code} ${text}`;
      }).join("\n");
      return `${q.instructions}\n\nOptions:\n${lines}\n\nReply with the option code only.`;
    }
    if (q.type === "noul") {
      const extra = q.criteria ? `\nYes: ${q.criteria.true}\nNo: ${q.criteria.false}` : "";
      return `${q.instructions}${extra}\n\nReply with yes or no only.`;
    }
    const legend = q.criteria.map((name, i) => `${i} ${name}`).join("\n");
    return `${q.instructions}\n\n${legend}\n\nReply with a single digit 0-${q.criteria.length - 1} only.`;
  }

  // a question's text after the state up to its answer slots, and each slot's token offset from its last token
  suffix(q) {
    const head = `${plain(this.questionBlock(q))}${IM_END}\n${IM_START}assistant\n${this.lead}`;
    if (!(q.type === "choice" && q.multiple)) return { text: head, offsets: [0] };
    // a multiple choice answers "name": yes/no at each option, a slot at each name's colon
    let text = head;
    const colons = [];
    Object.keys(q.criteria).forEach((name, i) => { text += (i ? "\n" : "") + plain(JSON.stringify(name)) + ":"; colons.push(text.length - 1); });
    const ids = this.ids(text);
    // each token's end in characters (decoding token by token: the release reads the tokenizer's offset mapping)
    const ends = [];
    let at = 0;
    for (let i = 0; i < ids.length; i++) { at += this.tok.decode([ids[i]], { skip_special_tokens: false }).length; ends.push(at); }
    if (at !== text.length) throw new Error("could not place a multiple choice's answer slots on its tokens");
    const offsets = colons.map((c) => ends.findIndex((e) => e > c) - ids.length + 1);
    return { text, offsets };
  }

  // the readout of each slot: one group of token ids per option, max-pooled (prompt.py readout_ids)
  readoutIds(q) {
    if (q.type === "noul" || (q.type === "choice" && q.multiple)) return [this.tokens(YES_FORMS), this.tokens(NO_FORMS)];
    if (q.type === "score") return q.criteria.map((_, i) => this.tokens([String(i)]));
    return this.aliases(Object.keys(q.criteria)).map(([code, tid]) => [tid, ...this.tokens([` ${code}`]).filter((x) => x !== tid)]);
  }

  // the sequences for one state: its prefix (its media, the state, "QUESTION:") and up to setSize questions packed
  // after it, each a group of its own (attention.pack_set). media: [{ token: "<|image|>" | "<|audio|>", rows }] in the
  // order their rows sit in the external embeddings; their placeholders become those rows (chat.js mediaIds)
  sequences(state, qs, media = []) {
    const header = state === null || state === undefined ? "" : "QUESTION:\n";
    const body = header ? `${plain(stateBlock(state))}\n${header}` : "";
    const markup = media.map((m) => `<|vision_start|>${m.token.repeat(m.rows)}<|vision_end|>`).join("");
    const prefix = `${this.bos}${IM_START}user\n${markup}${body}`;
    const suffixes = qs.map((q) => this.suffix(q));
    const out = [];
    for (let k = 0; k < qs.length; k += this.setSize) {
      const part = suffixes.slice(k, k + this.setSize);
      let ids = this.ids(prefix + part[0].text);
      const state0 = ids.length - this.ids(part[0].text).length;
      let groups = [...Array(state0).fill(0), ...Array(ids.length - state0).fill(1)];
      const slots = [part[0].offsets.map((o) => ids.length - 1 + o)];
      part.slice(1).forEach((s, j) => {
        const own = this.ids(header + s.text);
        ids = ids.concat(own);
        groups = groups.concat(Array(own.length).fill(j + 2));
        slots.push(s.offsets.map((o) => ids.length - 1 + o));
      });
      out.push({ first: k, ids: media.length ? mediaIds(ids, this.tok, media) : ids, state: state0, groups, slots });
    }
    return out;
  }

  temperature(q) { return this.calibration ? this.calibration.temperatures?.[TEMPERATURE_BUCKET(q)] ?? this.calibration.default ?? 1 : 1; }
  // option probabilities from a slot's logits: each option's best form's log-probability, tempered unless the
  // question is read through a table (a multiple choice), then a softmax
  probabilities(q, logits, tempered) {
    let mx = -Infinity;
    for (let i = 0; i < logits.length; i++) if (logits[i] > mx) mx = logits[i];
    let sum = 0;
    for (let i = 0; i < logits.length; i++) sum += Math.exp(logits[i] - mx);
    const lse = mx + Math.log(sum);
    const t = tempered ? this.temperature(q) : 1;
    const scores = this.readoutIds(q).map((g) => Math.max(...g.map((i) => logits[i] - lse)) / t);
    const m = Math.max(...scores), e = scores.map((s) => Math.exp(s - m)), z = e.reduce((a, b) => a + b, 0);
    return e.map((x) => x / z);
  }

  answer(q, probs) {
    const table = (name, v) => through(v, this.calibration?.tables?.[name]);
    if (q.type === "noul") return { type: "noul", noul: probs[0] };
    const names = q.type === "choice" ? Object.keys(q.criteria) : null;
    if (q.type === "choice" && q.multiple) {
      const p = probs.map((x) => table("multiple", x[0]));
      return { type: "choice", choices: names.filter((_, i) => p[i] >= 0.5), probabilities: Object.fromEntries(names.map((n, i) => [n, p[i]])) };
    }
    const best = probs.indexOf(Math.max(...probs));
    if (q.type === "choice") return { type: "choice", choice: names[best], confidence: probs[best], probabilities: Object.fromEntries(names.map((n, i) => [n, probs[i]])) };
    let mass = 0, level = probs.length - 1;
    for (let i = 0; i < probs.length; i++) { mass += probs[i]; if (mass >= 0.5) { level = i; break; } }
    return { type: "score", score: level, confidence: probs[best], probabilities: Object.fromEntries(probs.map((p, i) => [String(i), p])),
      legend: Object.fromEntries(q.criteria.map((t, i) => [String(i), t])) };
  }

  // named questions over a state (text, any JSON value, or null with media alone). media: { items, emb } — the
  // state's pictures and clips ([{ token, rows }], as Chat.send takes them) and the buffer holding their rows
  async ask(state, questions, { media = null } = {}) {
    const names = Object.keys(questions), qs = names.map((n) => asQuestion(questions[n]));
    if (media?.emb) this.model.setExternalEmbeddings(media.emb);
    const answers = {}, raw = {};
    let tokens = 0;
    for (const seq of this.sequences(state ?? null, qs, media?.items || [])) {
      const flat = seq.slots.flat();
      const logits = await this.model.decide(seq.ids, { state: seq.state, groups: seq.groups, slots: flat });
      tokens += seq.ids.length;
      let at = 0;
      seq.slots.forEach((own, j) => {
        const q = qs[seq.first + j], name = names[seq.first + j];
        const per = own.map(() => this.probabilities(q, logits[at++], !(q.type === "choice" && q.multiple)));
        raw[name] = q.type === "choice" && q.multiple ? per : per[0];
        answers[name] = this.answer(q, raw[name]);
      });
    }
    this.model.reset();
    return { answers, probabilities: raw, usage: { input_tokens: tokens, output_tokens: 0 } };
  }
}
