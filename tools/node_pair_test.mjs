// SPDX-License-Identifier: Apache-2.0. Copyright 2026 Toaster Club LLC.
// The two-model voice turn (engine/pair.js) on a real GPU through Dawn, with the site's own weight sets: d1-Omni-2B
// hears a spoken message and decides, then LFM3-Omni-2B answers aloud with the decision as a note. The spoken
// messages are made by LFM3-Omni-2B's own voice (greedy), and its ASR checks what they say. Each message is answered
// with and without the note, so the effect of the decision shows in the replies.
//   node tools/node_pair_test.mjs --site dir --out dir
// --site holds weights.js, d1-weights.js, the tokenizer and weights/<set-id>/ as the site serves them.
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import vm from "node:vm";
import { create, globals } from "webgpu";

Object.assign(globalThis, globals);
const arg = (k, d) => { const i = process.argv.indexOf(`--${k}`); return i > 0 ? process.argv[i + 1] : d; };
const gpu = create([]);
if (globalThis.navigator) Object.defineProperty(globalThis.navigator, "gpu", { value: gpu, configurable: true });
else Object.defineProperty(globalThis, "navigator", { value: { gpu, userAgent: "node" }, configurable: true });

const root = arg("engine", "../public/engine"), site = arg("site", "public"), out = arg("out", "out/pair");
mkdirSync(out, { recursive: true });
const { initDevice, LFM3 } = await import(`${root}/lfm3.js`);
const { Audio } = await import(`${root}/audio.js`);
const { DepthDecoder, Codec } = await import(`${root}/speech.js`);
const { Tokenizer } = await import(`${root}/tokenizer.js`);
const { Chat } = await import(`${root}/chat.js`);
const { Decider } = await import(`${root}/d1.js`);
const { Pair } = await import(`${root}/pair.js`);
const { resample16k } = await import(`${root}/listen.js`);
const { logMel } = await import(`${root}/audio.js`);
const { pngPayload } = await import(`${root}/pngbytes.js`);

const win = {};
for (const f of ["weights.js", "d1-weights.js"]) vm.runInNewContext(readFileSync(`${site}/${f}`, "utf8"), { window: win });
const pre = (W) => ({ id: W.id, manifest: W.tensors,
  chunks: async function* () { for (const c of W.chunks) yield await pngPayload(new Uint8Array(readFileSync(`${site}/${c.url}`)), c.bytes); } });
const tok = Tokenizer.fromParts(JSON.parse(readFileSync(`${site}/lfm3-tokenizer-vocab.json`, "utf8")), readFileSync(`${site}/lfm3-tokenizer-merges.txt`, "utf8"));
const calibration = JSON.parse(readFileSync(`${site}/d1/calibration.json`, "utf8"));

const { device, info } = await initDevice();
const res = { adapter: info, sets: {} };
let gpuErrors = 0;
device.onuncapturederror = (e) => { gpuErrors++; if (gpuErrors < 5) console.log("GPU ERROR", e.error.message); };
const t0 = performance.now();
const omni = new LFM3(device, { ctx: 4096, prequant: pre(win.LFM3_WEIGHTS) });
await omni.load(() => {});
const d1lm = new LFM3(device, { ctx: 4096, prequant: pre(win.D1_WEIGHTS) });
await d1lm.load(() => {});
const load = async (Cls, W) => { const m = new Cls(device, { quant: W.format, prequant: pre(W) }); await m.load(() => {}); return m; };
const audio = await load(Audio, win.LFM3_AUDIO), d1Audio = await load(Audio, win.D1_AUDIO);
const depth = await load(DepthDecoder, win.LFM3_TALK), codec = await load(Codec, win.LFM3_CODEC);
for (const k of ["LFM3_WEIGHTS", "D1_WEIGHTS", "LFM3_AUDIO", "D1_AUDIO", "LFM3_TALK", "LFM3_CODEC"]) res.sets[k] = win[k].id;
res.load_s = +((performance.now() - t0) / 1000).toFixed(1);
console.log("LOADED", res.load_s, "s", JSON.stringify(res.sets));

const chat = new Chat(omni, tok);
const decider = new Decider(d1lm, tok, { calibration, setSize: calibration.set_size || 16, lead: "<think></think>" });
const pair = new Pair({ decider, d1Audio, chat, audio, depth, codec });

// a spoken message in LFM3-Omni-2B's own voice, at 16 kHz, and what its ASR hears in it
async function say(line) {
  chat.spoken = true;
  chat.system = "You are a voice actor. Say exactly the line you are given, word for word, and nothing else.";
  await chat.newChat();
  const parts = [];
  await chat.speak(`Say this line: "${line}"`, { depth, codec, greedy: true, maxFrames: 300, onAudio: (x) => parts.push(x) });
  const n = parts.reduce((s, p) => s + p.length, 0), x24 = new Float32Array(n);
  let o = 0;
  for (const p of parts) { x24.set(p, o); o += p.length; }
  const x16 = resample16k(x24, 24000).slice(0, 30 * 16000);
  const lm = logMel(x16), enc = await audio.encode(lm.mel, lm.frames);
  const heard = await chat.asr(enc.emb, enc.rows);
  return { samples: x16, heard };
}

const QUESTIONS = {
  intent: { type: "choice", instructions: "What does the speaker want?", criteria: { refund: "Money back for a charge", question: "An answer to a question", complaint: "To report a problem", smalltalk: "Just chatting" } },
  urgent: { type: "noul", instructions: "Does the speaker say it is urgent?" },
  mood: { type: "score", instructions: "How upset is the speaker?", criteria: ["Calm", "Annoyed", "Upset"] },
  escalate: { type: "noul", instructions: "Should this go to a human agent?" },
};
const SYSTEM = "You are the voice of a shop's customer line. Keep answers to two or three short spoken sentences. Act on the notes: if escalate is yes, tell the caller you are passing them to a human agent now; if urgent is yes, say you are treating it as a priority; if the mood is Upset, apologize first.";
const MESSAGES = [
  "I was charged twice for my order and the parcel never arrived. Refund one of the charges, this is urgent!",
  "Hi there, I was just wondering what time your shop opens on Sundays.",
];
res.cases = [];
const clips = [];
for (const line of MESSAGES) {
  const clip = await say(line);
  const row = { line, heard: clip.heard, seconds: +(clip.samples.length / 16000).toFixed(2) };
  console.log("SAID", JSON.stringify(row));
  for (const withNote of [true, false]) {
    await pair.newChat(SYSTEM);
    const t = performance.now();
    const r = await pair.turn({ samples: clip.samples }, QUESTIONS, { greedy: true, withNote, maxFrames: 400 });
    const key = withNote ? "with_note" : "without_note";
    row[key] = { reply: r.reply, speech_s: +r.seconds.toFixed(1), total_ms: Math.round(performance.now() - t), decide_ms: Math.round(r.decideMs) };
    if (withNote) {
      row.note = r.note;
      row.decision = Object.fromEntries(Object.entries(r.decision.answers).map(([k, a]) => [k, a.type === "noul" ? +a.noul.toFixed(3) : a.choice ?? a.score]));
      row.d1_tokens = r.decision.usage.input_tokens;
    }
    console.log("TURN", key, JSON.stringify(row[key]));
  }
  res.cases.push(row);
  clips.push(clip);
}
// the note against the clip: the calm question with a note that says the opposite. A reply that follows the note
// (passing a calm caller to a person) shows the voice acts on d1's decision, not only on what it heard itself.
{
  const note = "[Notes from the decision model on this message: intent: complaint (90%); urgent: yes (95%); mood: Upset (level 2 of 2); escalate: yes (93%).]";
  await pair.newChat(SYSTEM);
  const r = await pair.turn({ samples: clips[1].samples }, QUESTIONS, { greedy: true, maxFrames: 400, note });
  res.counterfactual = { line: MESSAGES[1], note, reply: r.reply };
  console.log("COUNTER", JSON.stringify(res.counterfactual));
}
res.gpu_errors = gpuErrors;
writeFileSync(`${out}/result_pair.json`, JSON.stringify(res, null, 1));
console.log("PAIR_TEST", JSON.stringify({ gpu_errors: gpuErrors, decisions: res.cases.map((c) => c.decision), notes: res.cases.map((c) => c.note) }));
process.exit(0);
