// SPDX-License-Identifier: Apache-2.0. Copyright 2026 Toaster Club LLC.
// The WebGPU vision path (public/engine/vision.js) on a real GPU through Dawn, against tools/vision_ref.py:
// the page's preprocessing vs the processor's patches, the tower after every layer, the projector, and the language
// model's answer about the picture (top logits and the greedy description) with 8-bit and 4-bit vision weights.
//   node tools/node_vision_test.mjs --dir work/vision --src http://127.0.0.1:8080/model/model.safetensors --out r.json
import { closeSync, openSync, readFileSync, readSync, writeFileSync } from "node:fs";
import { create, globals } from "webgpu";

Object.assign(globalThis, globals);
const arg = (k, d) => { const i = process.argv.indexOf(`--${k}`); return i > 0 ? process.argv[i + 1] : d; };
const gpu = create([]);
if (globalThis.navigator) Object.defineProperty(globalThis.navigator, "gpu", { value: gpu, configurable: true });
else Object.defineProperty(globalThis, "navigator", { value: { gpu, userAgent: "node" }, configurable: true });

const { CFG, initDevice, LFM3 } = await import("../public/engine/lfm3.js");
const { Vision, VCFG, preprocess } = await import("../public/engine/vision.js");
const { Tokenizer } = await import("../public/engine/tokenizer.js");
const { SYSTEM, systemBlock, turnUser, mediaIds } = await import("../public/engine/chat.js");

const dir = arg("dir", "work/vision");
const ref = JSON.parse(readFileSync(`${dir}/ref.json`, "utf8"));
const f32 = (name) => { const b = readFileSync(`${dir}/${name}`); return new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)); };
const tok = Tokenizer.fromParts(JSON.parse(readFileSync("public/lfm3-tokenizer-vocab.json", "utf8")), readFileSync("public/lfm3-tokenizer-merges.txt", "utf8"));
const res = { T: ref.T, grid: [ref.gh, ref.gw], ref_text: ref.greedy_text };
const { device, info } = await initDevice();
res.adapter = info;
let gpuErrors = 0;
device.onuncapturederror = (e) => { gpuErrors++; if (gpuErrors < 5) console.log("GPU ERROR", e.error.message); };

const relErr = (a, b) => { let n = 0, d = 0; for (let i = 0; i < b.length; i++) { n += (a[i] - b[i]) ** 2; d += b[i] ** 2; } return +Math.sqrt(n / d).toFixed(4); };
function fileChunks(path) {
  return async function* () {
    const fd = openSync(path, "r");
    for (;;) {
      const b = Buffer.allocUnsafe(16 << 20);
      const n = readSync(fd, b, 0, b.length, null);
      if (!n) break;
      yield new Uint8Array(b.buffer, b.byteOffset, n);
    }
    closeSync(fd);
  };
}

// 1. the page's preprocessing (smart resize, bicubic, normalize, patchify) vs the processor
const [H, W] = ref.image;
const rgb = readFileSync(`${dir}/image.rgb`);
const rgba = new Uint8Array(W * H * 4);
for (let i = 0; i < W * H; i++) { rgba[4 * i] = rgb[3 * i]; rgba[4 * i + 1] = rgb[3 * i + 1]; rgba[4 * i + 2] = rgb[3 * i + 2]; rgba[4 * i + 3] = 255; }
const pp = preprocess(rgba, W, H, { maxPixels: ref.max_pixels });
const pv = f32("pixels.f32");
let maxd = 0, sumd = 0;
if (pp.T === ref.T) for (let i = 0; i < pv.length; i++) { const dd = Math.abs(pp.patches[i] - pv[i]); maxd = Math.max(maxd, dd); sumd += dd; }
res.preprocess = { grid: [pp.gh, pp.gw], same_grid: pp.gh === ref.gh && pp.gw === ref.gw,
  max_abs_8bit_steps: +(maxd * 127.5).toFixed(2), mean_abs_8bit_steps: +((sumd / pv.length) * 127.5).toFixed(4) };
console.log("preprocess", JSON.stringify(res.preprocess));

// 2. the prompt the app builds vs the processor's
const text = systemBlock(SYSTEM) + turnUser("Describe this image in one sentence.", { token: "<|image|>", rows: ref.rows });
const ids = Array.from(tok.encode(text));
res.prompt_ids_match = ids.length === ref.prompt_ids.length && ids.every((v, i) => v === ref.prompt_ids[i]);
console.log("prompt ids match", res.prompt_ids_match, ids.length, ref.prompt_ids.length);
const pids = mediaIds(ref.prompt_ids, tok, { token: "<|image|>" });

// 3. the tower and projector, then the answer, per vision precision
const refLayers = f32("layers.f32"), refEmb = f32("emb.f32"), TD = ref.T * VCFG.D;
const lm = new LFM3(device, { quant: "q8", url: arg("src"), ctx: 4096 });
let t0 = performance.now();
await lm.load(() => {});
res.lm_load_s = +((performance.now() - t0) / 1000).toFixed(1);
res.vision = {};
for (const quant of (arg("quants", "q8,q4g64")).split(",")) {
  const man = JSON.parse(readFileSync(`${dir}/manifest_${quant}.json`, "utf8")).tensors;
  const v = new Vision(device, { quant, maxTokens: ref.T, prequant: { manifest: man, chunks: fileChunks(`${dir}/${quant}.bin`) } });
  await v.load();
  const r = {};
  const out = await v.encode(pv, ref.gh, ref.gw, { debug: true });
  r.layer_rel_err = VCFG.types.map((_, l) => relErr(out.layers.subarray(l * TD, (l + 1) * TD), refLayers.subarray(l * TD, (l + 1) * TD)));
  r.emb_rel_err = relErr(out.emb32, refEmb);
  const times = [];
  for (let k = 0; k < 3; k++) { t0 = performance.now(); await v.encode(pv, ref.gh, ref.gw); times.push(performance.now() - t0); }
  r.encode_ms = +Math.min(...times).toFixed(1);
  // the answer with the processor's patches, then with the page's own preprocessing
  for (const [name, patches] of [["processor_pixels", pv], ["page_pixels", pp.patches]]) {
    if (patches.length !== pv.length) continue;
    const o = await v.encode(patches, ref.gh, ref.gw);
    lm.setExternalEmbeddings(o.emb);
    lm.reset();
    const logits = new Float32Array(CFG.V);
    t0 = performance.now();
    await lm.prefill(pids, { sample: true, logitsOut: logits });
    const prefillMs = performance.now() - t0;
    let top1 = 0;
    for (let i = 1; i < CFG.V; i++) if (logits[i] > logits[top1]) top1 = i;
    const order = Array.from(logits.keys()).sort((a, b) => logits[b] - logits[a]).slice(0, 10);
    const refTop = new Set(ref.top_ids.slice(0, 10));
    const gen = await lm.generate(ref.greedy.length);
    let match = 0;
    while (match < Math.min(gen.length, ref.greedy.length) && gen[match] === ref.greedy[match]) match++;
    r[name] = { top1_match: top1 === ref.top_ids[0], top10_overlap: order.filter((i) => refTop.has(i)).length,
      greedy_leading_match: match, greedy_ref: ref.greedy.length, prefill_ms: +prefillMs.toFixed(0),
      text: tok.decode(gen.filter((i) => !CFG.stopIds.includes(i)), { skip_special_tokens: true }) };
  }
  res.vision[quant] = r;
  console.log(`VISION_${quant.toUpperCase()}`, JSON.stringify(r));
  for (const [, b] of v.weightList()) b.destroy();
}
res.gpu_errors = gpuErrors;
writeFileSync(arg("out", "result_vision.json"), JSON.stringify(res, null, 1));
console.log("NODE_VISION " + JSON.stringify(res).slice(0, 4000));
process.exit(0);
