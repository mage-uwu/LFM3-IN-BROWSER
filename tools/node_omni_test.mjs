// SPDX-License-Identifier: Apache-2.0. Copyright 2026 Toaster Club LLC.
// Voice Desk with every input (engine/pair.js) on a real GPU through Dawn, with the site's own weight sets. d1-Omni-2B
// decides, then LFM3-Omni-2B answers aloud with the decision as a note, for:
//   text        a typed message
//   photo+text  a picture with a typed question
//   frame+speech, whole     a camera frame and a spoken question as one turn
//   frame+speech, live      the same frame and speech streamed in as they are "spoken" (48 kHz capture blocks ->
//                           Resampler -> MelStream -> LFM3 every 320 ms), with d1 deciding at the release
// The two frame+speech turns build the same prompt, so with greedy decoding their replies must match token for
// token. The spoken question is made by LFM3-Omni-2B's own voice, and its ASR checks what it says.
//   node tools/node_omni_test.mjs --site dir --out dir
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import vm from "node:vm";
import { create, globals } from "webgpu";

Object.assign(globalThis, globals);
const arg = (k, d) => { const i = process.argv.indexOf(`--${k}`); return i > 0 ? process.argv[i + 1] : d; };
const gpu = create([]);
if (globalThis.navigator) Object.defineProperty(globalThis.navigator, "gpu", { value: gpu, configurable: true });
else Object.defineProperty(globalThis, "navigator", { value: { gpu, userAgent: "node" }, configurable: true });

const root = arg("engine", "../public/engine"), site = arg("site", "public"), out = arg("out", "out/omni");
mkdirSync(out, { recursive: true });
const { initDevice, LFM3 } = await import(`${root}/lfm3.js`);
const { Audio, logMel, MelStream } = await import(`${root}/audio.js`);
const { Vision } = await import(`${root}/vision.js`);
const { DepthDecoder, Codec } = await import(`${root}/speech.js`);
const { Tokenizer } = await import(`${root}/tokenizer.js`);
const { Chat } = await import(`${root}/chat.js`);
const { Decider } = await import(`${root}/d1.js`);
const { Pair, LIVE_PIXELS } = await import(`${root}/pair.js`);
const { Resampler, resample16k } = await import(`${root}/listen.js`);
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
const lm = async (W) => { const m = new LFM3(device, { ctx: 4096, prequant: pre(W) }); await m.load(() => {}); return m; };
const load = async (Cls, W) => { const m = new Cls(device, { quant: W.format, prequant: pre(W) }); await m.load(() => {}); return m; };
const omni = await lm(win.LFM3_WEIGHTS), d1lm = await lm(win.D1_WEIGHTS);
const audio = await load(Audio, win.LFM3_AUDIO), d1Audio = await load(Audio, win.D1_AUDIO);
const vision = await load(Vision, win.LFM3_VISION), d1Vision = await load(Vision, win.D1_VISION);
const depth = await load(DepthDecoder, win.LFM3_TALK), codec = await load(Codec, win.LFM3_CODEC);
for (const k of ["LFM3_WEIGHTS", "D1_WEIGHTS", "LFM3_AUDIO", "D1_AUDIO", "LFM3_VISION", "D1_VISION", "LFM3_TALK", "LFM3_CODEC"]) res.sets[k] = win[k].id;
res.load_s = +((performance.now() - t0) / 1000).toFixed(1);
console.log("LOADED", res.load_s, "s");

const chat = new Chat(omni, tok);
const decider = new Decider(d1lm, tok, { calibration, setSize: calibration.set_size || 16, lead: "<think></think>" });
const pair = new Pair({ decider, d1Audio, d1Vision, chat, audio, vision, depth, codec });

// the red circle on white the d1 and LFM3 references use, as RGBA
function circle(size = 448) {
  const rgba = new Uint8ClampedArray(size * size * 4), r = size * 0.33, c = size / 2;
  for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
    const i = 4 * (y * size + x), inside = (x + 0.5 - c) ** 2 + (y + 0.5 - c) ** 2 <= r * r;
    rgba[i] = inside ? 224 : 255; rgba[i + 1] = inside ? 38 : 255; rgba[i + 2] = inside ? 43 : 255; rgba[i + 3] = 255;
  }
  return { rgba, w: size, h: size };
}
async function say(line) { // a spoken line in LFM3-Omni-2B's own voice at 16 kHz, and what its ASR hears in it
  chat.spoken = true;
  chat.system = "You are a voice actor. Say exactly the line you are given, word for word, and nothing else.";
  await chat.newChat();
  const parts = [];
  await chat.speak(`Say this line: "${line}"`, { depth, codec, greedy: true, maxFrames: 300, onAudio: (x) => parts.push(x) });
  const x24 = new Float32Array(parts.reduce((s, p) => s + p.length, 0));
  let o = 0;
  for (const p of parts) { x24.set(p, o); o += p.length; }
  const x16 = resample16k(x24, 24000).slice(0, 30 * 16000);
  const l = logMel(x16), enc = await audio.encode(l.mel, l.frames);
  return { samples: x16, heard: await chat.asr(enc.emb, enc.rows) };
}

const CUSTOMER = {
  system: "You are the voice of a shop's customer line. Keep answers to two or three short spoken sentences. Act on the notes: if escalate is yes, tell the caller you are passing them to a human agent now; if urgent is yes, say you are treating it as a priority; if the mood is Upset, apologize first.",
  questions: {
    intent: { type: "choice", instructions: "What does the customer want?", criteria: { refund: "Money back for a charge", order_status: "Where an order is", question: "An answer to a question", complaint: "To report a problem" } },
    urgent: { type: "noul", instructions: "Does the customer say it is urgent?" },
    mood: { type: "score", instructions: "How upset is the customer?", criteria: ["Calm", "Annoyed", "Upset"] },
    escalate: { type: "noul", instructions: "Should this go to a human agent?" },
  },
};
const HELPER = {
  system: "You are a helpful assistant who can see through the user's camera. Keep answers to one or two short spoken sentences. Act on the notes: if about_view is yes, answer from what you see.",
  questions: {
    intent: { type: "choice", instructions: "What does the user want?", criteria: { question: "An answer", request: "Something done", chat: "Just talking" } },
    about_view: { type: "noul", instructions: "Is the user asking about something they can see?" },
    seen: { type: "choice", instructions: "What is shown?", criteria: { shape: "A simple shape", person: "A person", text: "Writing", scene: "A room or a place", nothing: "No picture" } },
  },
};
const short = (a) => Object.fromEntries(Object.entries(a).map(([k, v]) => [k, v.type === "noul" ? +v.noul.toFixed(3) : v.choice ?? v.score]));
res.turns = {};
async function run(name, profile, fn) {
  await pair.newChat(profile.system);
  const t = performance.now();
  const r = await fn();
  res.turns[name] = { decision: short(r.decision.answers), d1_tokens: r.decision.usage.input_tokens, note: r.note, reply: r.reply,
    image_rows: r.imageRows, streamed: r.streamed, decide_ms: Math.round(r.decideMs), total_ms: Math.round(performance.now() - t),
    speech_s: +r.seconds.toFixed(1), textIds: r.textIds };
  console.log("TURN", name, JSON.stringify({ ...res.turns[name], textIds: undefined }));
  return r;
}

// 1. typed text
await run("text", CUSTOMER, () => pair.turn({ text: "I was charged twice for my order and the parcel never arrived. Refund one of the charges, this is urgent!" },
  CUSTOMER.questions, { greedy: true, maxFrames: 400 }));
// 2. a photo with a typed question
await run("photo+text", HELPER, () => pair.turn({ text: "What is this?", image: circle() }, HELPER.questions, { greedy: true, maxFrames: 400, maxPixels: LIVE_PIXELS }));
// 3 and 4: a camera frame and a spoken question, whole and streamed
const spoken = await say("What colour is the thing in front of me?");
res.spoken = { heard: spoken.heard, seconds: +(spoken.samples.length / 16000).toFixed(2) };
console.log("SAID", JSON.stringify(res.spoken));
const whole = await run("frame+speech whole", HELPER, () => pair.turn({ image: circle(), samples: spoken.samples }, HELPER.questions, { greedy: true, maxFrames: 400, maxPixels: LIVE_PIXELS }));
let firstAudio = 0, released = 0;
const live = await run("frame+speech live", HELPER, async () => {
  // a 48 kHz "capture" of the clip, 2048-sample blocks; every 320 ms of audio the final rows go into LFM3
  const up = new Resampler(16000, 48000), a = up.push(spoken.samples), b = up.finish();
  const x48 = new Float32Array(a.length + b.length); x48.set(a); x48.set(b, a.length);
  const rs = new Resampler(48000, 16000), mel = new MelStream(30), per = Math.round(0.32 * 48000);
  await pair.liveOpen(circle());
  let pending = 0;
  for (let o = 0; o < x48.length; o += 2048) {
    const blk = x48.subarray(o, Math.min(x48.length, o + 2048));
    mel.push(rs.push(blk));
    pending += blk.length;
    if (pending >= per) { pending = 0; await pair.liveFeed(mel); }
  }
  released = performance.now();
  mel.push(rs.finish()); mel.finish();
  return pair.liveClose(mel, HELPER.questions, { greedy: true, maxFrames: 400, onAudio: () => { if (!firstAudio) firstAudio = performance.now(); } });
});
res.live = { first_audio_ms_after_release: Math.round(firstAudio - released), streamed_rows: live.streamed,
  same_reply: whole.reply === live.reply, same_text_ids: JSON.stringify(whole.textIds) === JSON.stringify(live.textIds),
  same_frames: whole.frames.length === live.frames.length && whole.frames.every((f, i) => JSON.stringify(f) === JSON.stringify(live.frames[i])),
  same_decision: JSON.stringify(res.turns["frame+speech whole"].decision) === JSON.stringify(res.turns["frame+speech live"].decision) };
for (const t of Object.values(res.turns)) delete t.textIds;
res.gpu_errors = gpuErrors;
writeFileSync(`${out}/result_omni.json`, JSON.stringify(res, null, 1));
console.log("OMNI_TEST", JSON.stringify({ gpu_errors: gpuErrors, live: res.live }));
process.exit(0);
