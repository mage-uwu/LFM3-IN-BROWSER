// SPDX-License-Identifier: Apache-2.0. Copyright 2026 Toaster Club LLC.
// LFM3's tokenizer in plain JavaScript, built from the official tokenizer.json: added tokens split first, then the
// GPT-style pre-split regex, byte-level mapping, and BPE with ignore_merges (a piece that is already in the vocabulary
// is one token). No dependencies, no network: the app ships tokenizer.json, split in two files, beside the page.

// (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
// with the case-insensitive group spelled out (inline flag groups are too new for some browsers)
const PRE = /'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD]|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+/gu;

function byteMaps() { // GPT-2's reversible byte <-> printable-character map
  const bs = [];
  for (let b = 33; b <= 126; b++) bs.push(b);
  for (let b = 161; b <= 172; b++) bs.push(b);
  for (let b = 174; b <= 255; b++) bs.push(b);
  const cs = bs.slice();
  let n = 0;
  for (let b = 0; b < 256; b++) if (!bs.includes(b)) { bs.push(b); cs.push(256 + n++); }
  const enc = new Array(256), dec = new Map();
  bs.forEach((b, i) => { enc[b] = String.fromCodePoint(cs[i]); dec.set(String.fromCodePoint(cs[i]), b); });
  return { enc, dec };
}

export class Tokenizer {
  constructor(tj) {
    const m = tj.model;
    this.vocab = new Map(Object.entries(m.vocab));
    this.inv = [];
    for (const [t, i] of this.vocab) this.inv[i] = t;
    this.ignoreMerges = !!m.ignore_merges;
    this.ranks = new Map();
    m.merges.forEach((mg, r) => this.ranks.set(Array.isArray(mg) ? `${mg[0]} ${mg[1]}` : mg, r));
    this.added = new Map();
    this.special = new Set();
    for (const a of tj.added_tokens) {
      this.added.set(a.content, a.id);
      this.inv[a.id] = a.content;
      if (a.special) this.special.add(a.id);
    }
    this.addedIds = new Set(this.added.values());
    const esc = [...this.added.keys()].sort((a, b) => b.length - a.length).map((s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
    this.addedRe = new RegExp(`(${esc.join("|")})`);
    const { enc, dec } = byteMaps();
    this.byteEnc = enc;
    this.byteDec = dec;
    this.cache = new Map();
    this.utf8 = new TextEncoder();
  }

  // the app's form: the vocabulary (+ added tokens) as JSON and the merges as text, one "a b" per line (byte-level
  // tokens never contain a space), each small enough to publish beside the page
  static async fromFiles(base) {
    const get = async (name) => { const r = await fetch(new URL(name, base)); if (!r.ok) throw new Error(`tokenizer: HTTP ${r.status}`); return r; };
    const [v, m] = await Promise.all([get("lfm3-tokenizer-vocab.json").then((r) => r.json()), get("lfm3-tokenizer-merges.txt").then((r) => r.text())]);
    return Tokenizer.fromParts(v, m);
  }
  static fromParts(v, mergesText) {
    return new Tokenizer({ model: { vocab: v.vocab, merges: mergesText.split("\n"), ignore_merges: v.ignore_merges }, added_tokens: v.added_tokens });
  }

  static async fromUrl(url) {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`tokenizer: HTTP ${r.status}`);
    let bytes = new Uint8Array(await r.arrayBuffer());
    if (bytes[0] === 0x1f && bytes[1] === 0x8b) { // gzip (unless the server already decoded it)
      const ds = new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip")));
      bytes = new Uint8Array(await ds.arrayBuffer());
    }
    return new Tokenizer(JSON.parse(new TextDecoder().decode(bytes)));
  }

  bpe(piece) {
    const hit = this.cache.get(piece);
    if (hit) return hit;
    const mapped = Array.from(this.utf8.encode(piece), (b) => this.byteEnc[b]).join("");
    let ids;
    if (this.ignoreMerges && this.vocab.has(mapped)) {
      ids = [this.vocab.get(mapped)];
    } else {
      let sym = Array.from(mapped);
      while (sym.length > 1) {
        let best = Infinity, at = -1;
        for (let i = 0; i < sym.length - 1; i++) {
          const r = this.ranks.get(`${sym[i]} ${sym[i + 1]}`);
          if (r !== undefined && r < best) { best = r; at = i; }
        }
        if (at < 0) break;
        const a = sym[at], b = sym[at + 1], out = [];
        for (let i = 0; i < sym.length; i++) {
          if (i < sym.length - 1 && sym[i] === a && sym[i + 1] === b) { out.push(a + b); i++; } else out.push(sym[i]);
        }
        sym = out;
      }
      ids = sym.map((s) => this.vocab.get(s));
    }
    if (this.cache.size < 50000) this.cache.set(piece, ids);
    return ids;
  }

  encode(text) {
    const ids = [];
    for (const part of text.split(this.addedRe)) {
      if (!part) continue;
      const a = this.added.get(part);
      if (a !== undefined) { ids.push(a); continue; }
      for (const m of part.matchAll(PRE)) ids.push(...this.bpe(m[0]));
    }
    return ids;
  }

  decode(ids, { skip_special_tokens = false } = {}) {
    let out = "", buf = [];
    const flush = () => {
      if (!buf.length) return;
      const bytes = [];
      for (const ch of buf.join("")) { const b = this.byteDec.get(ch); if (b !== undefined) bytes.push(b); }
      out += new TextDecoder().decode(new Uint8Array(bytes));
      buf = [];
    };
    for (const id of ids) {
      if (this.addedIds.has(id)) {
        flush();
        if (!(skip_special_tokens && this.special.has(id))) out += this.inv[id];
      } else if (this.inv[id] !== undefined) {
        buf.push(this.inv[id]);
      }
    }
    flush();
    return out;
  }
}
