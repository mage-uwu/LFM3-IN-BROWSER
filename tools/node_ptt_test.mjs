// SPDX-License-Identifier: Apache-2.0. Copyright 2026 Toaster Club LLC.
// Push to talk on a real GPU through Dawn: the same spoken question as one turn (the whole clip, Chat.speak) and
// as a turn streamed in while it is "spoken" (48 kHz capture blocks -> Resampler -> MelStream -> the speech encoder
// every 320 ms of audio -> Chat.openTurn / feedAudio / closeTurn), both greedy. Checks that the streamed 16 kHz
// samples and log-mel equal the one-shot ones, that both turns give the same text and speech frames, and that a
// second turn after each continues the same way; reports how many speech tokens were in the model before the
// release and how soon the first audio came after it.
//   node tools/node_ptt_test.mjs --clip work/voice16k.f32 --audio work/audio --speech work/speech --src URL --out dir
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { create, globals } from "webgpu";

Object.assign(globalThis, globals);
const arg = (k, d) => { const i = process.argv.indexOf(`--${k}`); return i > 0 ? process.argv[i + 1] : d; };
const gpu = create([]);
if (globalThis.navigator) Object.defineProperty(globalThis.navigator, "gpu", { value: gpu, configurable: true });
else Object.defineProperty(globalThis, "navigator", { value: { gpu, userAgent: "node" }, configurable: true });

const { initDevice, LFM3 } = await import("../public/engine/lfm3.js");
const { DepthDecoder, Codec } = await import("../public/engine/speech.js");
const { Audio, logMel, MelStream } = await import("../public/engine/audio.js");
const { Resampler, resample16k } = await import("../public/engine/listen.js");
const { Tokenizer } = await import("../public/engine/tokenizer.js");
const { Chat } = await import("../public/engine/chat.js");

const outDir = arg("out", "out/ptt");
mkdirSync(outDir, { recursive: true });
const f32 = (p) => { const b = readFileSync(p); return new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)); };
const pre = (path, man) => ({ manifest: JSON.parse(readFileSync(man, "utf8")).tensors,
  chunks: async function* () { const b = readFileSync(path); for (let o = 0; o < b.length; o += 16 << 20) yield new Uint8Array(b.buffer, b.byteOffset + o, Math.min(16 << 20, b.length - o)); } });
const tok = Tokenizer.fromParts(JSON.parse(readFileSync("public/lfm3-tokenizer-vocab.json", "utf8")), readFileSync("public/lfm3-tokenizer-merges.txt", "utf8"));
const res = {};
const { device, info } = await initDevice();
res.adapter = info;
let gpuErrors = 0;
device.onuncapturederror = (e) => { gpuErrors++; if (gpuErrors < 5) console.log("GPU ERROR", e.error.message); };

// 1. the front end: a 48 kHz "capture" of the clip, resampled and turned into log-mel as it arrives vs in one go
const clip16 = f32(arg("clip", "work/voice16k.f32"));
const up = new Resampler(16000, 48000), a1 = up.push(clip16), a2 = up.finish();
const x48 = new Float32Array(a1.length + a2.length); x48.set(a1); x48.set(a2, a1.length);
const once16 = resample16k(x48, 48000), once = logMel(once16);
const rs = new Resampler(48000, 16000), ms = new MelStream(30);
const blocks = [];
for (let o = 0; o < x48.length; o += 2048) blocks.push(x48.subarray(o, Math.min(x48.length, o + 2048)));
const streamed16 = [];
for (const b of blocks) { const y = rs.push(b); streamed16.push(y); ms.push(y); }
const tail = rs.finish(); streamed16.push(tail); ms.push(tail); ms.finish();
const s16 = new Float32Array(streamed16.reduce((a, b) => a + b.length, 0));
{ let o = 0; for (const b of streamed16) { s16.set(b, o); o += b.length; } }
let maxD = 0; for (let i = 0; i < Math.min(s16.length, once16.length); i++) maxD = Math.max(maxD, Math.abs(s16[i] - once16[i]));
const melS = ms.layout();
let melD = 0; for (let i = 0; i < Math.min(melS.length, once.mel.length); i++) melD = Math.max(melD, Math.abs(melS[i] - once.mel[i]));
res.frontend = { seconds: +(clip16.length / 16000).toFixed(2), samples16k: [s16.length, once16.length], max_sample_diff: maxD,
  mel_frames: [ms.frames, once.frames], max_mel_diff: melD };
console.log("FRONTEND", JSON.stringify(res.frontend));

// 2. the models
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
chat.spoken = true;
const opts = (log) => ({ depth, codec, greedy: true, maxFrames: 160, maxText: 120, onAudio: () => { if (!log.first) log.first = performance.now(); } });
const second = "Say that again in five words or fewer.";

// 3. A: the whole clip as one turn, then a typed follow-up
await chat.newChat();
const encA = await enc.encode(once.mel, once.frames);
lm.setExternalEmbeddings(encA.emb);
const logA = {}, tA = performance.now();
const A1 = await chat.speak("", { media: { token: "<|audio|>", rows: encA.rows }, ...opts(logA) });
const A2 = await chat.speak(second, opts({}));
res.whole = { rows: encA.rows, text: A1.reply, frames: A1.frames.length, first_audio_ms: Math.round(logA.first - tA), followup: A2.reply };
console.log("WHOLE", JSON.stringify(res.whole));

// 4. B: the same clip streamed in, 320 ms of audio at a time, then the same follow-up
await chat.newChat();
await chat.openTurn(null);
const rsB = new Resampler(48000, 16000), mB = new MelStream(30);
const perTick = Math.round(0.32 * 48000);
let pending = 0, ticks = 0;
for (const b of blocks) {
  mB.push(rsB.push(b));
  pending += b.length;
  if (pending >= perTick) { // what the page does every 320 ms while the button is held
    pending = 0; ticks++;
    const final = mB.complete();
    if (final > chat.turn.rows) {
      const width = Math.min(3000, Math.ceil((mB.frames + 1) / 64) * 64);
      const o = await enc.encode(mB.layout(width), width);
      await chat.feedAudio(o.emb, Math.min(final, o.rows));
    }
  }
}
const streamedRows = chat.turn.rows;
const tB = performance.now();   // the release
mB.push(rsB.finish()); mB.finish();
const encB = await enc.encode(mB.layout(), mB.frames);
await chat.feedAudio(encB.emb, encB.rows);
const logB = {};
const B1 = await chat.closeTurn("", opts(logB));
const B2 = await chat.speak(second, opts({}));
res.streamed = { rows: encB.rows, streamed_before_release: streamedRows, ticks, text: B1.reply, frames: B1.frames.length,
  first_audio_ms_after_release: Math.round(logB.first - tB), followup: B2.reply };
console.log("STREAMED", JSON.stringify(res.streamed));

const lead = (a, b) => { let i = 0; while (i < Math.min(a.length, b.length) && JSON.stringify(a[i]) === JSON.stringify(b[i])) i++; return i; };
res.same = {
  text_ids: [lead(A1.textIds, B1.textIds), A1.textIds.length, B1.textIds.length],
  frames: [lead(A1.frames, B1.frames), A1.frames.length, B1.frames.length],
  followup_text_ids: [lead(A2.textIds, B2.textIds), A2.textIds.length, B2.textIds.length],
  followup_frames: [lead(A2.frames, B2.frames), A2.frames.length, B2.frames.length],
};
res.identical = res.same.text_ids[0] === A1.textIds.length && A1.textIds.length === B1.textIds.length &&
  res.same.frames[0] === A1.frames.length && A1.frames.length === B1.frames.length &&
  res.same.followup_text_ids[0] === A2.textIds.length && A2.textIds.length === B2.textIds.length;
res.gpu_errors = gpuErrors;
writeFileSync(`${outDir}/result_ptt.json`, JSON.stringify(res, null, 1));
console.log("NODE_PTT " + JSON.stringify(res).slice(0, 3000));
process.exit(0);
