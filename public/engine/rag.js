// SPDX-License-Identifier: Apache-2.0. Copyright 2026 Toaster Club LLC.
// Retrieval for one chat (RAG). Each chat has its own index, kept in memory and dropped with the chat, so nothing
// carries over from one chat to the next. It holds the documents attached to the chat and the chat's own turns
// that no longer fit in the model's context. Passages are ranked with BM25, a standard keyword ranking that needs no
// extra model, and the best ones go into the user's turn as context.
//
//   const index = new ChatIndex();
//   index.add(text, "notes.txt");                 // a document, cut into overlapping passages
//   const hits = index.search("when is the launch?");
//   const prompt = withContext(question, hits);   // the question with its passages
const STOP = new Set(("a an and are as at be but by for from has have he her his i if in into is it its me my no not " +
  "of on or our she so than that the their them then there these they this to was we were what when where which who " +
  "will with you your").split(" "));
const terms = (s) => s.toLowerCase().normalize("NFKD").replace(/[̀-ͯ]/g, "").match(/[\p{L}\p{N}]+/gu)
  ?.filter((w) => w.length > 1 && !STOP.has(w)) || [];

export class ChatIndex {
  constructor({ words = 120, overlap = 30 } = {}) {
    this.words = words; this.overlap = overlap;
    this.passages = [];       // { text, source, tf: Map, len }
    this.df = new Map();      // term -> number of passages that hold it
    this.sources = new Map(); // source -> passage count
  }
  get size() { return this.passages.length; }
  clear() { this.passages = []; this.df.clear(); this.sources.clear(); }
  // text is cut into passages of about `words` words that overlap by `overlap`, so a sentence at an edge stays whole
  add(text, source = "this chat") {
    const w = text.replace(/\r/g, "").split(/[ \t]+|\n+/).filter(Boolean);
    let n = 0;
    for (let i = 0; i < w.length; i += this.words - this.overlap) {
      const chunk = w.slice(i, i + this.words).join(" ");
      const t = terms(chunk);
      if (t.length) {
        const tf = new Map();
        for (const x of t) tf.set(x, (tf.get(x) || 0) + 1);
        for (const x of tf.keys()) this.df.set(x, (this.df.get(x) || 0) + 1);
        this.passages.push({ text: chunk, source, tf, len: t.length });
        n++;
      }
      if (i + this.words >= w.length) break;
    }
    if (n) this.sources.set(source, (this.sources.get(source) || 0) + n);
    return n;
  }
  remove(source) {
    const keep = this.passages.filter((p) => p.source !== source);
    this.clear();
    for (const p of keep) {
      for (const x of p.tf.keys()) this.df.set(x, (this.df.get(x) || 0) + 1);
      this.sources.set(p.source, (this.sources.get(p.source) || 0) + 1);
    }
    this.passages = keep;
  }
  // the k best passages for a query by BM25 (k1 1.2, b 0.75); passages that share no term with it are left out
  search(query, k = 3, { k1 = 1.2, b = 0.75 } = {}) {
    const q = [...new Set(terms(query))], N = this.passages.length;
    if (!q.length || !N) return [];
    const avg = this.passages.reduce((a, p) => a + p.len, 0) / N;
    const scored = [];
    for (const p of this.passages) {
      let s = 0;
      for (const x of q) {
        const f = p.tf.get(x);
        if (!f) continue;
        const df = this.df.get(x);
        s += Math.log(1 + (N - df + 0.5) / (df + 0.5)) * (f * (k1 + 1)) / (f + k1 * (1 - b + (b * p.len) / avg));
      }
      if (s > 0) scored.push({ score: s, text: p.text, source: p.source });
    }
    return scored.sort((a, c) => c.score - a.score).slice(0, k);
  }
}

// the user's message with the passages found for it, labelled so the model treats them as reference, not as orders
export function withContext(question, hits) {
  if (!hits.length) return question;
  const ctx = hits.map((h, i) => `[${i + 1}] (${h.source}) ${h.text}`).join("\n");
  return `Use these passages if they help answer. They come from this chat's documents and earlier turns.\n${ctx}\n\n${question}`;
}
