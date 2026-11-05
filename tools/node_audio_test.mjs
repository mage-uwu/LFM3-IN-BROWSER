// SPDX-License-Identifier: Apache-2.0. Copyright 2026 Toaster Club LLC.
// The WebGPU speech-input path (public/engine/audio.js) on a real GPU through Dawn, against tools/audio_ref.py: the
// page's log-mel vs the feature extractor's, the encoder after every layer, the projector, and the transcription the
// language model writes (ASR prompt) with 8-bit and 4-bit encoder weights.
//   node tools/node_audio_test.mjs --dir work/audio --src http://127.0.0.1:8080/model/model.safetensors --out r.json
import { closeSync, openSync, readFileSync, readSync, writeFileSync } from "node:fs";
import { create, globals } from "webgpu";

Object.assign(globalThis, globals);
const arg = (k, d) => { const i = process.argv.indexOf(`--${k}`); return i > 0 ? process.argv[i + 1] : d; };
const gpu = create([]);
if (globalThis.navigator) Object.defineProperty(globalThis.navigator, "gpu", { value: gpu, configurable: true });
else Object.defineProperty(globalThis, "navigator", { value: { gpu, userAgent: "node" }, configurable: true });

const { CFG, initDevice, LFM3 } = await import("../public/engine/lfm3.js");
const { Audio, ACFG, logMel } = await import("../public/engine/audio.js");
const { Tokenizer } = await import("../public/engine/tokenizer.js");
const { systemBlock, turnUser, mediaIds } = await import("../public/engine/chat.js");

const dir = arg("dir", "work/audio");
const ref = JSON.parse(readFileSync(`${dir}/ref.json`, "utf8"));
const f32 = (name) => { const b = readFileSync(`${dir}/${name}`); return new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)); };
const tok = Tokenizer.fromParts(JSON.parse(readFileSync("public/lfm3-tokenizer-vocab.json", "utf8")), readFileSync("public/lfm3-tokenizer-merges.txt", "utf8"));
const res = { T: ref.T, frames: ref.frames, text_ref: ref.text_ref, ref_text: ref.greedy_text };
const { device, info } = await initDevice();
res.adapter = info;
let gpuErrors = 0;
device.onuncapturederror = (e) => { gpuErrors++; if (gpuErrors < 5) console.log("GPU ERROR", e.error.message); };
const relErr = (a, b) => { let n = 0, d = 0; for (let i = 0; i < b.length; i++) { n += (a[i] - b[i]) ** 2; d += b[i] ** 2; } return +Math.sqrt(n / d).toFixed(4); };
const fileChunks = (path) => async function* () {
  const fd = openSync(path, "r");
  for (;;) { const b = Buffer.allocUnsafe(16 << 20); const n = readSync(fd, b, 0, b.length, null); if (!n) break; yield new Uint8Array(b.buffer, b.byteOffset, n); }
  closeSync(fd);
};

// 1. the page's log-mel vs the feature extractor
const samples = f32("samples.f32"), refMel = f32("mel.f32");
let t0 = performance.now();
const lm0 = logMel(samples);
res.logmel_ms = +(performance.now() - t0).toFixed(0);
let maxd = 0, sumd = 0;
if (lm0.frames === ref.frames) for (let i = 0; i < refMel.length; i++) { const d = Math.abs(lm0.mel[i] - refMel[i]); maxd = Math.max(maxd, d); sumd += d; }
res.logmel = { frames: lm0.frames, same_frames: lm0.frames === ref.frames, tokens: lm0.tokens, max_abs: +maxd.toFixed(5), mean_abs: +(sumd / refMel.length).toFixed(6) };
console.log("logmel", JSON.stringify(res.logmel), res.logmel_ms, "ms");

// 2. the ASR prompt the app builds vs the processor's
const text = systemBlock("Perform ASR in English.") + turnUser("", { token: "<|audio|>", rows: ref.T });
const ids = Array.from(tok.encode(text));
res.prompt_ids_match = ids.length === ref.prompt_ids.length && ids.every((v, i) => v === ref.prompt_ids[i]);
console.log("prompt ids match", res.prompt_ids_match, ids.length, ref.prompt_ids.length);
const pids = mediaIds(ref.prompt_ids, tok, { token: "<|audio|>" });

// 3. encoder and projector, then the transcription, per encoder precision
const refLayers = f32("layers.f32"), refEmb = f32("emb.f32"), TD = ref.T * ACFG.D;
const lm = new LFM3(device, { quant: "q8", url: arg("src"), ctx: 4096 });
await lm.load(() => {});
res.audio = {};
for (const quant of arg("quants", "q8,q4g64").split(",")) {
  const man = JSON.parse(readFileSync(`${dir}/manifest_${quant}.json`, "utf8")).tensors;
  const a = new Audio(device, { quant, maxTokens: ref.T, prequant: { manifest: man, chunks: fileChunks(`${dir}/${quant}.bin`) } });
  await a.load();
  const r = {};
  const out = await a.encode(refMel, ref.frames, { debug: true });
  r.rows = out.rows;
  r.layer_rel_err = ACFG.types.map((_, l) => relErr(out.layers.subarray(l * TD, (l + 1) * TD), refLayers.subarray(l * TD, (l + 1) * TD)));
  r.emb_rel_err = relErr(out.emb32, refEmb);
  const times = [];
  for (let k = 0; k < 3; k++) { t0 = performance.now(); await a.encode(refMel, ref.frames); times.push(performance.now() - t0); }
  r.encode_ms = +Math.min(...times).toFixed(1);
  for (const [name, mel] of [["extractor_mel", refMel], ["page_mel", lm0.mel]]) {
    if (mel.length !== refMel.length) continue;
    const o = await a.encode(mel, ref.frames);
    lm.setExternalEmbeddings(o.emb);
    lm.reset();
    const logits = new Float32Array(CFG.V);
    await lm.prefill(pids, { sample: true, logitsOut: logits });
    let top1 = 0;
    for (let i = 1; i < CFG.V; i++) if (logits[i] > logits[top1]) top1 = i;
    const gen = await lm.generate(ref.greedy.length + 4);
    let match = 0;
    while (match < Math.min(gen.length, ref.greedy.length) && gen[match] === ref.greedy[match]) match++;
    r[name] = { top1_match: top1 === ref.top_ids[0], greedy_leading_match: match, greedy_ref: ref.greedy.length,
      text: tok.decode(gen.filter((i) => !CFG.stopIds.includes(i)), { skip_special_tokens: true }) };
  }
  res.audio[quant] = r;
  console.log(`AUDIO_${quant.toUpperCase()}`, JSON.stringify(r));
  for (const [, b] of a.weightList()) b.destroy();
}
res.gpu_errors = gpuErrors;
writeFileSync(arg("out", "result_audio.json"), JSON.stringify(res, null, 1));
console.log("NODE_AUDIO " + JSON.stringify(res).slice(0, 3000));
process.exit(0);
