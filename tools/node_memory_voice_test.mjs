// SPDX-License-Identifier: Apache-2.0. Copyright 2026 Toaster Club LLC.
// System 3 from speech, on a real GPU through Dawn: spoken turns teach the memory. The model's own speech says two
// sentences with facts and a standing instruction; those recordings (24 kHz, resampled to 16 kHz) are the user's
// spoken turns in a voice chat. Chat.consolidate then hears each turn with the model's own speech recognition
// (Chat.asr) and extracts memories from what it heard. Checks: the transcripts, the memories, and a new chat that
// starts from the memory answering from it. A control runs the old path (no transcription) on the same turns.
//   node tools/node_memory_voice_test.mjs --audio work/audio --speech work/speech --src URL --out dir
import { mkdirSync, writeFileSync } from "node:fs";
import { readFileSync } from "node:fs";
import { create, globals } from "webgpu";

Object.assign(globalThis, globals);
const arg = (k, d) => { const i = process.argv.indexOf(`--${k}`); return i > 0 ? process.argv[i + 1] : d; };
const gpu = create([]);
if (globalThis.navigator) Object.defineProperty(globalThis.navigator, "gpu", { value: gpu, configurable: true });
else Object.defineProperty(globalThis, "navigator", { value: { gpu, userAgent: "node" }, configurable: true });

const { initDevice, LFM3 } = await import("../public/engine/lfm3.js");
const { DepthDecoder, Codec } = await import("../public/engine/speech.js");
const { Audio, logMel } = await import("../public/engine/audio.js");
const { resample16k } = await import("../public/engine/listen.js");
const { Tokenizer } = await import("../public/engine/tokenizer.js");
const { Chat, Experience } = await import("../public/engine/chat.js");

const outDir = arg("out", "out/memvoice");
mkdirSync(outDir, { recursive: true });
const pre = (path, man) => ({ manifest: JSON.parse(readFileSync(man, "utf8")).tensors,
  chunks: async function* () { const b = readFileSync(path); for (let o = 0; o < b.length; o += 16 << 20) yield new Uint8Array(b.buffer, b.byteOffset + o, Math.min(16 << 20, b.length - o)); } });
const tok = Tokenizer.fromParts(JSON.parse(readFileSync("public/lfm3-tokenizer-vocab.json", "utf8")), readFileSync("public/lfm3-tokenizer-merges.txt", "utf8"));
const res = {};
const { device, info } = await initDevice();
res.adapter = info;
let gpuErrors = 0;
device.onuncapturederror = (e) => { gpuErrors++; if (gpuErrors < 5) console.log("GPU ERROR", e.error.message); };

const lm = new LFM3(device, { quant: "q8", url: arg("src"), ctx: 4096 });
await lm.load(() => {});
const A = arg("audio", "work/audio"), S = arg("speech", "work/speech");
const enc = new Audio(device, { quant: "q8", prequant: pre(`${A}/q8.bin`, `${A}/manifest_q8.json`) });
await enc.load();
const depth = new DepthDecoder(device, { quant: "q8", prequant: pre(`${S}/speech_q8.bin`, `${S}/manifest_speech_q8.json`) });
await depth.load();
const codec = new Codec(device, { quant: "q8", prequant: pre(`${S}/codec_q8.bin`, `${S}/manifest_codec_q8.json`) });
await codec.load();
const chat = new Chat(lm, tok);

// 1. the user's voice: the model reads each sentence aloud (greedy), recorded and resampled to 16 kHz
const SENTENCES = ["My sister's name is Rosa and I live in Lisbon.", "From now on, always address me as Captain."];
chat.useMemory = false;
chat.spoken = true;
const clips = [];
for (const s of SENTENCES) {
  await chat.newChat();
  const parts = [];
  const r = await chat.speak(`Read this sentence aloud, word for word, and say nothing else: "${s}"`, { depth, codec, greedy: true, maxFrames: 220, maxText: 60, onAudio: (a) => parts.push(a) });
  const a24 = new Float32Array(parts.reduce((n, p) => n + p.length, 0));
  let o = 0; for (const p of parts) { a24.set(p, o); o += p.length; }
  clips.push({ sentence: s, said_text: r.reply, seconds: +(a24.length / 24000).toFixed(2), samples: resample16k(a24, 24000) });
}
res.voices = clips.map(({ sentence, said_text, seconds }) => ({ sentence, said_text, seconds }));
console.log("VOICES", JSON.stringify(res.voices));

// 2. a voice chat: each clip is a spoken user turn (the speech encoder's rows), answered aloud, its audio kept
const hear = async (x) => { const m = logMel(x.audio); const o = await enc.encode(m.mel, m.frames); return chat.asr(o.emb, o.rows, "English"); };
async function voiceChat() {
  chat.experience = new Experience();
  chat.useMemory = true;
  chat.spoken = true;
  await chat.newChat();
  for (const c of clips) {
    const m = logMel(c.samples);
    const o = await enc.encode(m.mel, m.frames);
    lm.setExternalEmbeddings(o.emb);
    await chat.speak("", { media: { token: "<|audio|>", rows: o.rows }, audio: c.samples, depth, codec, greedy: true, maxFrames: 120, maxText: 80 });
  }
  return chat.messages.map((x) => ({ ...x }));
}

// 3. control: the old path, without transcription (a spoken turn has no text)
const msgsControl = await voiceChat();
const control = await chat.consolidate(null, { messages: msgsControl });
res.control_without_transcription = { memories: control.length };

// 4. the fix: each spoken turn is heard first
const msgs = await voiceChat();
const t0 = performance.now();
const changes = await chat.consolidate(null, { messages: msgs, transcribe: hear });
res.consolidate_ms = Math.round(performance.now() - t0);
res.heard = msgs.filter((x) => x.role === "user").map((x) => x.heard);
res.memories = changes.map((c) => (c.type === "fact" ? `fact: ${c.key} = ${c.value}` : `${c.type}: ${c.text}`));
res.facts = Object.fromEntries(Object.entries(chat.experience.facts).map(([k, v]) => [k, v.value]));
res.preferences = chat.experience.preferences.map((p) => p.text);
console.log("LEARNED", JSON.stringify({ heard: res.heard, memories: res.memories }));

// 5. a new chat starts from the memory (consolidate rebuilt the prefix): does it know?
const ask = await chat.send("What is my sister's name, and which city do I live in? Answer in one short sentence.", { maxNew: 60 });
res.new_chat_answer = ask.reply;
const low = ask.reply.toLowerCase();
res.checks = {
  heard_rosa: res.heard.some((h) => /rosa/i.test(h || "")),
  heard_captain: res.heard.some((h) => /captain/i.test(h || "")),
  fact_rosa: Object.values(res.facts).some((v) => /rosa/i.test(v)),
  fact_lisbon: Object.values(res.facts).some((v) => /lisbon/i.test(v)),
  preference_captain: res.preferences.some((p) => /captain/i.test(p)),
  answer_rosa: low.includes("rosa"),
  answer_lisbon: low.includes("lisbon"),
  answer_captain: low.includes("captain"),
  control_learned_nothing: control.length === 0,
};
res.gpu_errors = gpuErrors;
writeFileSync(`${outDir}/result_memory_voice.json`, JSON.stringify(res, null, 1));
console.log("NODE_MEMVOICE " + JSON.stringify(res).slice(0, 3000));
process.exit(0);
