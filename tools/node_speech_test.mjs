// SPDX-License-Identifier: Apache-2.0. Copyright 2026 Toaster Club LLC.
// The WebGPU speech output (public/engine/speech.js, Chat.speak) on a real GPU through Dawn, against
// tools/speech_ref.py: the depth decoder's logits for the same hidden state, the codec's audio for the reference
// frames, the deterministic interleaved loop (text and frames), and a round trip: our speech transcribed by our own
// speech encoder and language model. Writes WAV files of the reference and of our speech.
//   node tools/node_speech_test.mjs --dir work/speech --audio work/audio --src http://127.0.0.1:8080/model/model.safetensors --out dir
import { closeSync, mkdirSync, openSync, readFileSync, readSync, writeFileSync } from "node:fs";
import { create, globals } from "webgpu";

Object.assign(globalThis, globals);
const arg = (k, d) => { const i = process.argv.indexOf(`--${k}`); return i > 0 ? process.argv[i + 1] : d; };
const gpu = create([]);
if (globalThis.navigator) Object.defineProperty(globalThis.navigator, "gpu", { value: gpu, configurable: true });
else Object.defineProperty(globalThis, "navigator", { value: { gpu, userAgent: "node" }, configurable: true });

const { CFG, initDevice, LFM3 } = await import("../public/engine/lfm3.js");
const { DepthDecoder, Codec, SCFG } = await import("../public/engine/speech.js");
const { Audio, logMel } = await import("../public/engine/audio.js");
const { Tokenizer } = await import("../public/engine/tokenizer.js");
const { Chat, systemBlock, turnUser, mediaIds } = await import("../public/engine/chat.js");

const dir = arg("dir", "work/speech"), outDir = arg("out", "out/speech");
mkdirSync(outDir, { recursive: true });
const ref = JSON.parse(readFileSync(`${dir}/ref.json`, "utf8"));
const f32 = (p) => { const b = readFileSync(p); return new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)); };
const tok = Tokenizer.fromParts(JSON.parse(readFileSync("public/lfm3-tokenizer-vocab.json", "utf8")), readFileSync("public/lfm3-tokenizer-merges.txt", "utf8"));
const res = { ref_text: ref.text, ref_frames: ref.frames.length, ref_transcript: ref.transcript };
const { device, info } = await initDevice();
res.adapter = info;
let gpuErrors = 0;
device.onuncapturederror = (e) => { gpuErrors++; if (gpuErrors < 5) console.log("GPU ERROR", e.error.message); };
const relErr = (a, b) => { let n = 0, d = 0; for (let i = 0; i < b.length; i++) { n += (a[i] - b[i]) ** 2; d += b[i] ** 2; } return +Math.sqrt(n / d).toFixed(4); };
const snr = (a, b) => { const n = Math.min(a.length, b.length); let s = 0, e = 0; for (let i = 0; i < n; i++) { s += b[i] ** 2; e += (a[i] - b[i]) ** 2; } return +(10 * Math.log10(s / e)).toFixed(2); };
const fileChunks = (path) => async function* () {
  const fd = openSync(path, "r");
  for (;;) { const b = Buffer.allocUnsafe(16 << 20); const n = readSync(fd, b, 0, b.length, null); if (!n) break; yield new Uint8Array(b.buffer, b.byteOffset, n); }
  closeSync(fd);
};
const pre = (d, name, q) => ({ manifest: JSON.parse(readFileSync(`${d}/manifest_${name}_${q}.json`, "utf8")).tensors, chunks: fileChunks(`${d}/${name}_${q}.bin`) });
function wav(path, x, rate = 24000) {
  const b = Buffer.alloc(44 + x.length * 2);
  b.write("RIFF", 0); b.writeUInt32LE(36 + x.length * 2, 4); b.write("WAVEfmt ", 8); b.writeUInt32LE(16, 16); b.writeUInt16LE(1, 20); b.writeUInt16LE(1, 22);
  b.writeUInt32LE(rate, 24); b.writeUInt32LE(rate * 2, 28); b.writeUInt16LE(2, 32); b.writeUInt16LE(16, 34); b.write("data", 36); b.writeUInt32LE(x.length * 2, 40);
  for (let i = 0; i < x.length; i++) b.writeInt16LE(Math.max(-32768, Math.min(32767, Math.round(x[i] * 32767))), 44 + 2 * i);
  writeFileSync(path, b);
}
function to16k(x) { // 24 kHz -> 16 kHz: windowed-sinc low-pass at 7.2 kHz, then 2 of every 3 samples (resample 2/3)
  const taps = 48, fc = 7200 / 48000, h = [];
  for (let i = -taps; i <= taps; i++) { const w = 0.5 + 0.5 * Math.cos((Math.PI * i) / (taps + 1)); h.push(i === 0 ? 2 * fc : (Math.sin(2 * Math.PI * fc * i) / (Math.PI * i)) * w); }
  const n = Math.floor((x.length * 2) / 3), y = new Float32Array(n);
  for (let o = 0; o < n; o++) { // output o sits at 48 kHz index 3o on the zero-stuffed (x2) signal
    let s = 0;
    for (let k = -taps; k <= taps; k++) { const j = 3 * o - k; if (j >= 0 && (j & 1) === 0 && j / 2 < x.length) s += 2 * h[k + taps] * x[j / 2]; }
    y[o] = s;
  }
  return y;
}
wav(`${outDir}/reference.wav`, f32(`${dir}/audio.f32`));

const lm = new LFM3(device, { quant: "q8", url: arg("src"), ctx: 4096 });
await lm.load(() => {});
const audioEnc = new Audio(device, { quant: "q8", prequant: { manifest: JSON.parse(readFileSync(`${arg("audio", "work/audio")}/manifest_q8.json`, "utf8")).tensors, chunks: fileChunks(`${arg("audio", "work/audio")}/q8.bin`) } });
await audioEnc.load();
async function transcribe(a24) {
  const lmel = logMel(to16k(a24));
  const o = await audioEnc.encode(lmel.mel, lmel.frames);
  lm.setExternalEmbeddings(o.emb);
  lm.reset();
  const ids = mediaIds(Array.from(tok.encode(systemBlock("Perform ASR in English.") + turnUser("", { token: "<|audio|>", rows: o.rows }))), tok, { token: "<|audio|>" });
  await lm.prefill(ids, { sample: true });
  const gen = await lm.generate(96);
  return tok.decode(gen.filter((i) => !CFG.stopIds.includes(i)), { skip_special_tokens: true });
}
res.transcript_of_reference_audio = await transcribe(f32(`${dir}/audio.f32`));
console.log("ASR(reference audio):", res.transcript_of_reference_audio);

res.quants = {};
for (const q of arg("quants", "q8,q4g64").split(",")) {
  const r = {};
  const depth = new DepthDecoder(device, { quant: q, prequant: pre(dir, "speech", q) });
  await depth.load();
  const codec = new Codec(device, { quant: q, prequant: pre(dir, "codec", q) });
  await codec.load();
  // 1. the depth decoder on the reference hidden state
  const hb = device.createBuffer({ size: 2048 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(hb, 0, f32(`${dir}/hidden.f32`));
  const first = [];
  const codes0 = await depth.sampleFrame(hb, [], { greedy: true, firstLogits: first });
  r.depth_logits_rel_err = relErr(first[0], f32(`${dir}/logits0.f32`));
  r.depth_frame0_codes_match = codes0.filter((c, i) => c === ref.frames[0]?.[i]).length;
  // 2. the codec on the reference frames, in streaming chunks
  codec.reset();
  let t0 = performance.now();
  const ours = await codec.decode(ref.frames, { final: true });
  r.codec_ms_per_second = +((performance.now() - t0) / (ours.length / 24000)).toFixed(1);
  r.codec_samples = [ours.length, ref.samples];
  r.codec_snr_db = snr(ours, f32(`${dir}/audio.f32`));
  wav(`${outDir}/codec_${q}_of_reference_frames.wav`, ours);
  // 3. the whole interleaved loop, deterministic
  const chat = new Chat(lm, tok);
  chat.spoken = true;
  lm.reset();
  const promptText = systemBlock(ref.system) + turnUser(ref.question) + "<|mixed_start|>";
  const pids = Array.from(tok.encode(promptText));
  r.prompt_ids_match = pids.length === ref.prompt_ids.length && pids.every((v, i) => v === ref.prompt_ids[i]);
  const sys = Array.from(tok.encode(systemBlock(ref.system)));
  await lm.prefill(sys, { sample: false });
  const audio = [];
  t0 = performance.now();
  const sp = await chat.speak(ref.question, { depth, codec, greedy: true, onAudio: (a) => audio.push(a), maxFrames: 400 });
  const ms = performance.now() - t0;
  const all = new Float32Array(audio.reduce((a, x) => a + x.length, 0));
  let o = 0; for (const x of audio) { all.set(x, o); o += x.length; }
  wav(`${outDir}/speech_${q}.wav`, all);
  let tm = 0; while (tm < Math.min(sp.textIds.length, ref.text_ids.length) && sp.textIds[tm] === ref.text_ids[tm]) tm++;
  let fm = 0; while (fm < Math.min(sp.frames.length, ref.frames.length) && sp.frames[fm].every((c, i) => c === ref.frames[fm][i])) fm++;
  Object.assign(r, { text: sp.reply, text_ids_leading_match: tm, text_ids: [sp.textIds.length, ref.text_ids.length],
    frames_leading_match: fm, frames: [sp.frames.length, ref.frames.length], seconds: +sp.seconds.toFixed(2),
    loop_ms: +ms.toFixed(0), frames_per_s: +(sp.frames.length / (ms / 1000)).toFixed(1), realtime_factor: +(sp.seconds / (ms / 1000)).toFixed(2) });
  // 4. round trip: our speech, transcribed by our own speech encoder and language model
  r.transcript_of_our_speech = await transcribe(all);
  res.quants[q] = r;
  console.log(`SPEECH_${q.toUpperCase()}`, JSON.stringify(r));
  for (const [, b] of depth.weightList()) b.destroy();
  for (const [, b] of codec.weightList()) b.destroy();
}
res.gpu_errors = gpuErrors;
writeFileSync(`${outDir}/result_speech.json`, JSON.stringify(res, null, 1));
console.log("NODE_SPEECH " + JSON.stringify(res).slice(0, 3000));
process.exit(0);
