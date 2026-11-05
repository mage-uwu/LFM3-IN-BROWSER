// SPDX-License-Identifier: Apache-2.0. Copyright 2026 Toaster Club LLC.
// public/engine/tokenizer.js against Hugging Face's tokenizer (tools/tok_ref.py): ids and decoded text per case.
// Built exactly as the app builds it, from the two files it ships (public/lfm3-tokenizer-vocab.json, -merges.txt).
//   node tools/tok_test.mjs work/tok_ref.json public
import { readFileSync } from "node:fs";
import { Tokenizer } from "../public/engine/tokenizer.js";

const ref = JSON.parse(readFileSync(process.argv[2], "utf8"));
const dir = process.argv[3] || "public";
const tok = Tokenizer.fromParts(JSON.parse(readFileSync(`${dir}/lfm3-tokenizer-vocab.json`, "utf8")),
  readFileSync(`${dir}/lfm3-tokenizer-merges.txt`, "utf8"));
let ok = 0;
const fails = [];
for (const c of ref) {
  const ids = tok.encode(c.text);
  const same = JSON.stringify(ids) === JSON.stringify(c.ids);
  const dec = tok.decode(c.ids, { skip_special_tokens: true });
  const decSame = dec === c.decoded;
  if (same && decSame) ok++;
  else fails.push({ text: c.text.slice(0, 80), same, decSame, ours: ids.slice(0, 30), ref: c.ids.slice(0, 30) });
}
console.log("TOK_TEST " + JSON.stringify({ cases: ref.length, ok, fails }));
