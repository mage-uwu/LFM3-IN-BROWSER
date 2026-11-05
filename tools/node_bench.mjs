// SPDX-License-Identifier: Apache-2.0. Copyright 2026 Toaster Club LLC.
// The WebGPU engine (public/engine/lfm3.js, unchanged) on a real GPU through Dawn's Node.js bindings (npm "webgpu"),
// which use Vulkan directly: the correctness checks against the PyTorch reference, plus
// prefill and decode speed.
//   node tools/node_bench.mjs --ref work/test_ref.json --src http://127.0.0.1:8080/model/model.safetensors --quant q4
import { closeSync, openSync, readFileSync, readSync, writeFileSync } from "node:fs";
import { create, globals } from "webgpu";

Object.assign(globalThis, globals);
const arg = (k, d) => { const i = process.argv.indexOf(`--${k}`); return i > 0 ? process.argv[i + 1] : d; };
// Node 21+ has a read-only global navigator: attach gpu to it (or define one)
const gpu = create([]);
if (globalThis.navigator) Object.defineProperty(globalThis.navigator, "gpu", { value: gpu, configurable: true });
else Object.defineProperty(globalThis, "navigator", { value: { gpu, userAgent: "node" }, configurable: true });

const { CFG, initDevice, LFM3 } = await import("../public/engine/lfm3.js");
const { pngChunks } = await import("../public/engine/pngbytes.js");
// --quant gptq: pre-quantized weights (tools/gptq_export.py) from raw files (--parts a.bin,b.bin) or from the PNG
// containers the page loads (--png dir/chunks.json), through the page's own decoder
function prequant() {
  const dir = arg("pre", "work/gptq");
  const manifest = JSON.parse(readFileSync(`${dir}/manifest.json`, "utf8")).tensors;
  const png = arg("png");
  if (png) {
    const c = JSON.parse(readFileSync(png, "utf8"));
    const base = png.slice(0, png.lastIndexOf("/") + 1);
    return { manifest, chunks: () => pngChunks(c.chunks, async (x) => readFileSync(base + x.file)) };
  }
  const parts = arg("parts", "embed_rtn.bin,rest.bin").split(",");
  return { manifest, chunks: async function* () {
    for (const f of parts) {
      const fd = openSync(`${dir}/${f}`, "r");
      for (;;) {
        const b = Buffer.allocUnsafe(16 << 20);
        const n = readSync(fd, b, 0, b.length, null);
        if (!n) break;
        yield new Uint8Array(b.buffer, b.byteOffset, n);
      }
      closeSync(fd);
    }
  } };
}
const ref = JSON.parse(readFileSync(arg("ref", "work/test_ref.json"), "utf8"));
const quant = arg("quant", "q4");
const res = { quant, runtime: "dawn-node" };
const { device, info } = await initDevice();
res.adapter = info;
console.log("adapter", JSON.stringify(info));
let gpuErrors = 0;
device.onuncapturederror = (e) => { gpuErrors++; if (gpuErrors < 5) console.log("GPU ERROR", e.error.message); };

function compare(c, logits, hidden) {
  let top1 = 0;
  for (let i = 1; i < CFG.V; i++) if (logits[i] > logits[top1]) top1 = i;
  const order = Array.from(logits.keys()).sort((a, b) => logits[b] - logits[a]).slice(0, 10);
  const refTop10 = new Set(c.top_ids.slice(0, 10));
  const diffs = c.top_ids.map((i, j) => Math.abs(logits[i] - c.top_logits[j]));
  const r = { top1_match: top1 === c.top_ids[0], top10_overlap: order.filter((i) => refTop10.has(i)).length,
    max_abs_diff_top40: +Math.max(...diffs).toFixed(4), mean_abs_diff_top40: +(diffs.reduce((a, b) => a + b, 0) / diffs.length).toFixed(4) };
  const layers = [];
  for (let l = 0; l < CFG.layers - 1; l++) {
    const a = hidden[0].subarray(l * CFG.D, (l + 1) * CFG.D), b = c.hidden[l];
    let num = 0, den = 0;
    for (let i = 0; i < CFG.D; i++) { num += (a[i] - b[i]) ** 2; den += b[i] ** 2; }
    layers.push(+Math.sqrt(num / den).toFixed(4));
  }
  r.hidden_rel_err_max = Math.max(...layers);
  r.hidden_rel_err_by_layer = layers;
  return r;
}

const SWEEP = {
  q4: [{}],
  sweep: [{ convQuant: "q4", headQuant: "q8" }, { convQuant: "q8" }, { convQuant: "q4", mlpQuant: "q8" },
    { convQuant: "q4", attnQuant: "q8" }, { convQuant: "q8", attnQuant: "q8" }, { convQuant: "q4", mlpQuant: "q8", attnQuant: "q8" }],
}[arg("sweep", "q4")] || [{}];
const results = [];
for (const cfg of quant === "q4" ? SWEEP : [{}]) {
const model = quant === "gptq" ? new LFM3(device, { prequant: prequant(), ctx: 4096 })
  : new LFM3(device, { quant, url: arg("src"), ctx: 4096, ...cfg });
res.config = { quant, head: model.headQuant, conv: model.convQuant, mlp: model.mlpQuant, attn: model.attnQuant };
let t0 = performance.now();
await model.load(() => {});
res.load_s = +((performance.now() - t0) / 1000).toFixed(1);
res.cases = {};
for (const [name, c] of Object.entries(ref.cases)) {
  model.reset();
  const logits = new Float32Array(CFG.V), hidden = [];
  await model.prefill(c.ids, { sample: true, logitsOut: logits, debugHidden: hidden });
  const r = compare(c, logits, hidden);
  const gen = await model.generate(c.greedy.length);
  let match = 0;
  while (match < Math.min(gen.length, c.greedy.length) && gen[match] === c.greedy[match]) match++;
  Object.assign(r, { greedy_leading_match: match, greedy_ref: c.greedy.length, gen_ids: gen });
  res.cases[name] = r;
  console.log(name, JSON.stringify(r));
}
// speed: warm prefill of the long prompt, then decoding (no debug copies, no logits read-back)
const ids = ref.cases.text.ids;
for (let rep = 0; rep < 2; rep++) {
  model.reset();
  t0 = performance.now();
  await model.prefill(ids, { sample: true });
  res.prefill_tok_s = +(ids.length / ((performance.now() - t0) / 1000)).toFixed(1);
  t0 = performance.now();
  const g = await model.generate(129, { batch: 16, noStop: true });  // always 128 decode steps
  res.decode_tok_s = +((g.length - 1) / ((performance.now() - t0) / 1000)).toFixed(1);
}
res.prompt_tokens = ids.length;
res.gpu_errors = gpuErrors;
res.weight_mb = Math.round(model.weightList().reduce((a, [, b]) => a + b.size, 0) / 2 ** 20);
results.push(JSON.parse(JSON.stringify(res)));
console.log(`NODE_${quant.toUpperCase()} ` + JSON.stringify({ ...res, cases: Object.fromEntries(Object.entries(res.cases).map(([k, v]) => [k, { ...v, gen_ids: undefined, hidden_rel_err_by_layer: undefined }])) }));
console.log("LAYERS " + JSON.stringify(res.config) + " " + JSON.stringify(res.cases.text.hidden_rel_err_by_layer));
for (const b of model.weightList()) b[1].destroy();
}
writeFileSync(arg("out", `result_node_${quant}.json`), JSON.stringify(results.length === 1 ? results[0] : results, null, 1));
process.exit(0);
