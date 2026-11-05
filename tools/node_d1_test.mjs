// SPDX-License-Identifier: Apache-2.0. Copyright 2026 Toaster Club LLC.
// The d1 decision model on a real GPU through Dawn, against the release's own runner (tools/d1_ref.py): the same
// cases through engine/d1.js (the prompt, the packing, d1's "set" attention in LFM3.decide, the readout and the
// calibration), compared question by question with the reference's probabilities.
//   node tools/node_d1_test.mjs --ref work/d1_ref.json --src URL-of-model.safetensors --vision work/d1vision
//        --audio work/d1audio --media work/d1media [--lm work/d1lm.bin --lm-manifest work/d1gptq/manifest.json]
// --src converts the release's bf16 weights on the GPU (--quant q8, the default); --lm loads a prequantized export
// (the GPTQ 4-bit the site serves) instead. --media holds circle.rgb (448x448 RGB) and beeps.f32 (16 kHz), the
// reference's picture and clip, written by d1_ref.py's circle() and beeps().
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { create, globals } from "webgpu";

Object.assign(globalThis, globals);
const arg = (k, d) => { const i = process.argv.indexOf(`--${k}`); return i > 0 ? process.argv[i + 1] : d; };
const gpu = create([]);
if (globalThis.navigator) Object.defineProperty(globalThis.navigator, "gpu", { value: gpu, configurable: true });
else Object.defineProperty(globalThis, "navigator", { value: { gpu, userAgent: "node" }, configurable: true });

const root = arg("engine", "../public/engine");
const { initDevice, LFM3 } = await import(`${root}/lfm3.js`);
const { Vision, preprocess } = await import(`${root}/vision.js`);
const { Audio, logMel } = await import(`${root}/audio.js`);
const { Tokenizer } = await import(`${root}/tokenizer.js`);
const { Decider } = await import(`${root}/d1.js`);
const { pngPayload } = await import(`${root}/pngbytes.js`);

const out = arg("out", "out/d1");
mkdirSync(out, { recursive: true });
const f32 = (p) => { const b = readFileSync(p); return new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)); };
const pre = (path, man) => ({ manifest: JSON.parse(readFileSync(man, "utf8")).tensors,
  chunks: async function* () { const b = readFileSync(path); for (let o = 0; o < b.length; o += 16 << 20) yield new Uint8Array(b.buffer, b.byteOffset + o, Math.min(16 << 20, b.length - o)); } });
const tdir = arg("tokenizer", "../public");
const tok = Tokenizer.fromParts(JSON.parse(readFileSync(`${tdir}/lfm3-tokenizer-vocab.json`, "utf8")), readFileSync(`${tdir}/lfm3-tokenizer-merges.txt`, "utf8"));
const ref = JSON.parse(readFileSync(arg("ref", "work/d1_ref.json"), "utf8"));
const res = { weights: arg("lm-png") ? "GPTQ 4-bit (PNG parts)" : arg("lm") ? "prequantized" : arg("quant", "q8"), cases: [] };
const { device, info } = await initDevice();
res.adapter = info;
let gpuErrors = 0;
device.onuncapturederror = (e) => { gpuErrors++; if (gpuErrors < 5) console.log("GPU ERROR", e.error.message); };

// a prequantized export as PNG parts (png_chunks.py: chunks.json beside them), as the site serves it
const prePng = (dir, man) => ({ id: "test", manifest: JSON.parse(readFileSync(man, "utf8")).tensors,
  chunks: async function* () { for (const c of JSON.parse(readFileSync(`${dir}/chunks.json`, "utf8")).chunks) yield await pngPayload(new Uint8Array(readFileSync(`${dir}/${c.file}`)), c.bytes); } });
const lm = arg("lm-png") ? new LFM3(device, { ctx: 4096, prequant: prePng(arg("lm-png"), arg("lm-manifest")) })
  : arg("lm") ? new LFM3(device, { ctx: 4096, prequant: pre(arg("lm"), arg("lm-manifest")) })
  : new LFM3(device, { quant: arg("quant", "q8"), url: arg("src"), ctx: 4096 });
await lm.load(() => {});
const V = arg("vision"), A = arg("audio");
const vision = V ? new Vision(device, { quant: "q8", prequant: pre(`${V}/q8.bin`, `${V}/manifest_q8.json`) }) : null;
if (vision) await vision.load();
const audio = A ? new Audio(device, { quant: "q8", prequant: pre(`${A}/q8.bin`, `${A}/manifest_q8.json`) }) : null;
if (audio) await audio.load();
const d1 = new Decider(lm, tok, { lead: "<think></think>", calibration: ref.calibration, setSize: ref.config.d1_set_size || 16 });
const M = arg("media", "work/d1media");

const SUPPORTED = new Set(["noul", "choice", "score"]);
for (const c of ref.cases) {
  const names = Object.keys(c.questions).filter((n) => SUPPORTED.has(c.questions[n].type) && !(c.questions[n].type === "number"));
  const qs = Object.fromEntries(names.map((n) => [n, c.questions[n]]));
  let media = null;
  if (c.media === "circle" && vision) {
    const rgb = new Uint8Array(readFileSync(`${M}/circle.rgb`)), rgba = new Uint8ClampedArray(448 * 448 * 4);
    for (let i = 0; i < 448 * 448; i++) { rgba[4 * i] = rgb[3 * i]; rgba[4 * i + 1] = rgb[3 * i + 1]; rgba[4 * i + 2] = rgb[3 * i + 2]; rgba[4 * i + 3] = 255; }
    const pp = preprocess(rgba, 448, 448, { maxPixels: ref.config.d1_vision_max_pixels });
    const o = await vision.encode(pp.patches, pp.gh, pp.gw);
    media = { items: [{ token: "<|image|>", rows: o.rows }], emb: o.emb };
  } else if (c.media === "beeps" && audio) {
    const lmel = logMel(f32(`${M}/beeps.f32`));
    const o = await audio.encode(lmel.mel, lmel.frames);
    media = { items: [{ token: "<|audio|>", rows: o.rows }], emb: o.emb };
  } else if (c.media) { console.log("skip", c.name, "(no encoder)"); continue; }
  const t0 = performance.now();
  const got = await d1.ask(c.state, qs, { media });
  const ms = performance.now() - t0;
  const all = Object.keys(c.questions), row = { name: c.name, ms: Math.round(ms), tokens: got.usage.input_tokens, questions: {} };
  for (const n of names) {
    const want = c.probabilities[all.indexOf(n)];
    const mine = got.probabilities[n];
    const a = c.questions[n].type === "choice" && c.questions[n].multiple ? mine.map((p) => p[0]) : mine;
    const diff = Math.max(...a.map((x, i) => Math.abs(x - want[i])));
    row.questions[n] = { diff: +diff.toFixed(4), mine: a.map((x) => +x.toFixed(4)), reference: want.map((x) => +x.toFixed(4)),
      answer: got.answers[n], reference_answer: c.answers[n] };
  }
  res.cases.push(row);
  console.log("CASE", JSON.stringify(row).slice(0, 1500));
}
res.max_diff = Math.max(...res.cases.flatMap((c) => Object.values(c.questions).map((q) => q.diff)));
res.gpu_errors = gpuErrors;
writeFileSync(`${out}/result_d1.json`, JSON.stringify(res, null, 1));
console.log("D1_TEST", JSON.stringify({ max_diff: res.max_diff, gpu_errors: gpuErrors, weights: res.weights }));
