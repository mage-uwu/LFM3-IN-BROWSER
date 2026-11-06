// SPDX-License-Identifier: Apache-2.0. Copyright 2026 Toaster Club LLC.
// The integration path of docs/VOICE_DESK_HANDOFF.md on a real GPU through Dawn: engine/loader.js loadPair() against
// a deployed site (its weights.js, d1-weights.js, weight parts, tokenizer and calibration, fetched as a page would),
// then the "Trades dispatch" profile (engine/profiles.js) on typed, spoken and photo turns. The spoken turn is made
// by LFM3-Omni-2B's own voice. Prints each decision and spoken answer, and DESK_API_TEST.
//   node tools/node_desk_api_test.mjs --engine public/engine --base https://lfm.toasterclub.com/ --out dir
import { mkdirSync, writeFileSync } from "node:fs";
import vm from "node:vm";
import { create, globals } from "webgpu";

Object.assign(globalThis, globals);
const arg = (k, d) => { const i = process.argv.indexOf(`--${k}`); return i > 0 ? process.argv[i + 1] : d; };
const gpu = create([]);
if (globalThis.navigator) Object.defineProperty(globalThis.navigator, "gpu", { value: gpu, configurable: true });
else Object.defineProperty(globalThis, "navigator", { value: { gpu, userAgent: "node" }, configurable: true });
const base = arg("base", "https://lfm.toasterclub.com/"), root = arg("engine", "../public/engine"), out = arg("out", "out/desk");
mkdirSync(out, { recursive: true });
// what a page has that Node lacks: its address, and the weight lists a <script> tag would have run (Node cannot
// import a classic script over https). There is no IndexedDB here, so every part is downloaded, as on a first visit.
globalThis.document = { baseURI: base };
globalThis.window = globalThis;
for (const f of ["weights.js", "d1-weights.js"]) {
  const r = await fetch(new URL(f, base));
  if (!r.ok) throw new Error(`${f}: HTTP ${r.status}`);
  vm.runInThisContext(await r.text());
}

const { loadPair } = await import(`${root}/loader.js`);
const { PROFILES } = await import(`${root}/profiles.js`);
const { logMel } = await import(`${root}/audio.js`);
const { resample16k } = await import(`${root}/listen.js`);
const trades = PROFILES["Trades dispatch"];
const res = { base, profile: "Trades dispatch" };
const t0 = performance.now();
let last = "";
const desk = await loadPair({ base, system: trades.system, pictures: true,
  onProgress: (p) => { if (p.label !== last) { last = p.label; console.log("LOADING", p.label); } } });
await desk.loadPictures();
res.load_s = +((performance.now() - t0) / 1000).toFixed(1);
res.adapter = desk.info;
let gpuErrors = 0;
desk.device.onuncapturederror = (e) => { gpuErrors++; if (gpuErrors < 5) console.log("GPU ERROR", e.error.message); };
console.log("LOADED", res.load_s, "s");
const { pair, chat } = desk;

async function say(line) { // a spoken message in LFM3-Omni-2B's own voice, at 16 kHz, and what its ASR hears
  chat.spoken = true;
  chat.system = "You are a voice actor. Say exactly the line you are given, word for word, and nothing else.";
  await chat.newChat();
  const parts = [];
  await chat.speak(`Say this line: "${line}"`, { depth: pair.depth, codec: pair.codec, greedy: true, maxFrames: 300, onAudio: (x) => parts.push(x) });
  const x24 = new Float32Array(parts.reduce((s, p) => s + p.length, 0));
  let o = 0;
  for (const p of parts) { x24.set(p, o); o += p.length; }
  const x16 = resample16k(x24, 24000).slice(0, 30 * 16000);
  const l = logMel(x16), enc = await pair.audio.encode(l.mel, l.frames);
  return { samples: x16, heard: await chat.asr(enc.emb, enc.rows) };
}
function pipe(size = 448) { // a plain grey pipe on white: a photo with no damage in it
  const rgba = new Uint8ClampedArray(size * size * 4);
  for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
    const i = 4 * (y * size + x), inPipe = y > size * 0.42 && y < size * 0.58 && x > size * 0.1 && x < size * 0.9;
    const v = inPipe ? 150 + Math.round(40 * Math.sin((y - size * 0.42) / (size * 0.16) * Math.PI)) : 250;
    rgba[i] = v; rgba[i + 1] = v; rgba[i + 2] = inPipe ? v + 8 : v; rgba[i + 3] = 255;
  }
  return { rgba, w: size, h: size };
}
const short = (a) => Object.fromEntries(Object.entries(a).map(([k, v]) => [k, v.type === "noul" ? +v.noul.toFixed(3) : v.choice ?? (v.type === "score" ? trades.questions[k].criteria[v.score] : v.choices)]));
res.turns = [];
async function run(label, input) {
  await pair.newChat(trades.system);
  const t = performance.now();
  const r = await pair.turn(input, trades.questions, { greedy: true, maxFrames: 450 });
  const row = { label, decision: short(r.decision.answers), reply: r.reply, decide_ms: Math.round(r.decideMs), total_ms: Math.round(performance.now() - t), speech_s: +r.seconds.toFixed(1) };
  res.turns.push(row);
  console.log("TURN", JSON.stringify(row));
}
await run("typed: a leak", { text: "Hi, water is pouring out from under my kitchen sink and it's spreading across the floor. Can someone come today?" });
await run("typed: a gas smell", { text: "There's a strong smell of gas near my boiler in the basement." });
const spoken = await say("I'd like a quote to put two new ceiling lights in my living room. There's no rush.");
console.log("SAID", JSON.stringify(spoken.heard));
await run("spoken: a quote", { samples: spoken.samples });
await run("photo + typed", { text: "This is the pipe under my sink. It drips a little when the tap runs.", image: pipe() });
res.spoken_heard = spoken.heard;
res.gpu_errors = gpuErrors;
writeFileSync(`${out}/result_desk_api.json`, JSON.stringify(res, null, 1));
console.log("DESK_API_TEST", JSON.stringify({ gpu_errors: gpuErrors, load_s: res.load_s, decisions: res.turns.map((t) => t.decision) }));
process.exit(0);
