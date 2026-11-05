// SPDX-License-Identifier: Apache-2.0. Copyright 2026 Toaster Club LLC.
// A chat regression check on a real GPU through Dawn: the same greedy turns through one engine directory, token ids
// written out, so two engines (before and after a change to lfm3.js) can be compared token for token.
//   node tools/node_chat_check.mjs --engine public/engine --tokenizer public --lm-png dir --lm-manifest m.json --out ids.json
import { readFileSync, writeFileSync } from "node:fs";
import { create, globals } from "webgpu";

Object.assign(globalThis, globals);
const arg = (k, d) => { const i = process.argv.indexOf(`--${k}`); return i > 0 ? process.argv[i + 1] : d; };
const gpu = create([]);
if (globalThis.navigator) Object.defineProperty(globalThis.navigator, "gpu", { value: gpu, configurable: true });
else Object.defineProperty(globalThis, "navigator", { value: { gpu, userAgent: "node" }, configurable: true });
const root = arg("engine", "../public/engine");
const { initDevice, LFM3 } = await import(`${root}/lfm3.js`);
const { Tokenizer } = await import(`${root}/tokenizer.js`);
const { Chat } = await import(`${root}/chat.js`);
const { pngPayload } = await import(`${root}/pngbytes.js`);
const tdir = arg("tokenizer", "../public");
const tok = Tokenizer.fromParts(JSON.parse(readFileSync(`${tdir}/lfm3-tokenizer-vocab.json`, "utf8")), readFileSync(`${tdir}/lfm3-tokenizer-merges.txt`, "utf8"));
const dir = arg("lm-png");
const prequant = { id: "check", manifest: JSON.parse(readFileSync(arg("lm-manifest"), "utf8")).tensors,
  chunks: async function* () { for (const c of JSON.parse(readFileSync(`${dir}/chunks.json`, "utf8")).chunks) yield await pngPayload(new Uint8Array(readFileSync(`${dir}/${c.file}`)), c.bytes); } };
const { device } = await initDevice();
let gpuErrors = 0;
device.onuncapturederror = (e) => { gpuErrors++; if (gpuErrors < 5) console.log("GPU ERROR", e.error.message); };
const lm = new LFM3(device, { ctx: 4096, prequant });
await lm.load(() => {});
const chat = new Chat(lm, tok);
const para = "The lighthouse keeper logged the weather every hour: wind direction, wind speed, visibility, and the state of the sea. ";
const turns = ["What is the capital of France? Answer in one sentence.",
  para.repeat(40) + "\nWhat did the keeper record? List the four things.",
  "Now say it again in five words."];
const out = [];
await chat.newChat();
for (const t of turns) {
  const r = await chat.send(t, { maxNew: 64 });
  const ids = tok.encode ? tok.encode(r.reply) : [];
  out.push({ reply: r.reply, prompt_tokens: r.stats.promptTokens, ids });
  console.log("TURN", JSON.stringify({ prompt_tokens: r.stats.promptTokens, reply: r.reply.slice(0, 200) }));
}
writeFileSync(arg("out", "chat_ids.json"), JSON.stringify({ turns: out, gpu_errors: gpuErrors }, null, 1));
console.log("CHAT_CHECK", JSON.stringify({ gpu_errors: gpuErrors }));
process.exit(0);
