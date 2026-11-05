// SPDX-License-Identifier: Apache-2.0. Copyright 2026 Toaster Club LLC.
// The smoke test in Settings: every workflow once, with fixed inputs, measured on this device's GPU. Each row checks
// that the workflow works (a known answer, a known color, words heard back) and records its speed:
//   Text           the time to the first token of a short answer; prefill and decode tokens per second
//   Photo          a drawn picture through the image encoder, then a question about it
//   Spoken reply   the model's interleaved text-and-speech mode: time to the first sound, speed against real time
//   Speech in      that spoken reply heard back by the speech encoder and the model's own speech recognition
//   Push to talk   the same speech streamed in at the pace of someone talking, with a camera picture, as in Live:
//                  the time from letting go to the answer's first sound
// It runs with the default instructions, so results compare across devices, and leaves the chat
// for the app to restart. Nothing leaves the device: the results are only shown, copied or saved by the user.
import { SYSTEM, turnUser } from "./engine/chat.js";
import { preprocess } from "./engine/vision.js";
import { logMel, MelStream } from "./engine/audio.js";
import { resample16k } from "./engine/listen.js";

export const SMOKE_VERSION = 1;
const Q_TEXT = "What is the capital of France? Answer in one word.";
const Q_PHOTO = "What color is the circle in this picture? Answer in one word.";
const Q_SPEAK = "Introduce yourself in one short sentence.";
const PASSAGE = "WebGPU lets a web page use the graphics processor of the device it runs on. A language model does two " +
  "kinds of work: reading a prompt, where many tokens are processed at once, and writing an answer, where each new " +
  "token depends on the one before it. Reading is limited by arithmetic, writing by how fast the weights can be read " +
  "from memory. A model with two billion parameters stored in four bits per weight reads a little over a gigabyte for " +
  "each token it writes, so the speed of the memory sets the pace. Phones, laptops and desktop graphics cards differ " +
  "mostly in that speed, and in how many arithmetic units they can keep busy at the same time.";
const passage = (short) => `${Array.from({ length: short ? 1 : 4 }, (_, i) => `Part ${i + 1}. ${PASSAGE}`).join("\n\n")}\n\nSummarize the text above in detail.`;

const now = () => performance.now();
const sleep = (ms) => new Promise((ok) => setTimeout(ok, ms));
const num = (v) => (!Number.isFinite(v) ? "–" : v >= 100 ? v.toFixed(0) : v >= 10 ? v.toFixed(1) : v.toFixed(2));
const ms = (v) => (v < 1000 ? `${Math.round(v)} ms` : `${(v / 1000).toFixed(v < 10000 ? 2 : 1)} s`);
const quote = (s, n = 48) => { s = s.replace(/\s+/g, " ").trim(); return `“${s.length > n ? s.slice(0, n - 1) + "…" : s}”`; };
function concat(chunks) {
  const x = new Float32Array(chunks.reduce((a, c) => a + c.length, 0));
  let o = 0;
  for (const c of chunks) { x.set(c, o); o += c.length; }
  return x;
}
// the share of the reference's words heard, in order (1 - word error rate, at least 0)
export function wordAccuracy(ref, hyp) {
  const w = (s) => s.toLowerCase().replace(/[^\p{L}\p{N}' ]+/gu, " ").split(/\s+/).filter(Boolean);
  const a = w(ref), b = w(hyp);
  if (!a.length) return 0;
  let prev = Array.from({ length: b.length + 1 }, (_, j) => j);
  for (let i = 1; i <= a.length; i++) {
    const cur = [i];
    for (let j = 1; j <= b.length; j++) cur[j] = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
    prev = cur;
  }
  return Math.max(0, 1 - prev[b.length] / a.length);
}
// the picture the photo and Live rows look at: a red circle on white
function picture(size = 448) {
  const cv = document.createElement("canvas");
  cv.width = cv.height = size;
  const cx = cv.getContext("2d");
  cx.fillStyle = "#ffffff"; cx.fillRect(0, 0, size, size);
  cx.fillStyle = "#e0262b"; cx.beginPath(); cx.arc(size / 2, size / 2, size * 0.33, 0, Math.PI * 2); cx.fill();
  return { rgba: cx.getImageData(0, 0, size, size).data, w: size, h: size };
}

// what the results are from: the GPU as WebGPU names it, the browser and the system
export function deviceInfo(gpu = {}) {
  const ua = navigator.userAgent || "";
  const m = (re) => ua.match(re)?.[1];
  const browser = m(/Edg\/(\d+)/) ? `Edge ${m(/Edg\/(\d+)/)}` : m(/Firefox\/(\d+)/) ? `Firefox ${m(/Firefox\/(\d+)/)}`
    : m(/CriOS\/(\d+)/) ? `Chrome ${m(/CriOS\/(\d+)/)}` : m(/Chrome\/(\d+)/) ? `Chrome ${m(/Chrome\/(\d+)/)}`
    : m(/Version\/(\d+(?:\.\d+)?).*Safari/) ? `Safari ${m(/Version\/(\d+(?:\.\d+)?).*Safari/)}` : "unknown browser";
  const os = /iPhone/.test(ua) ? "iOS" : /iPad/.test(ua) ? "iPadOS" : /Android/.test(ua) ? "Android" : /CrOS/.test(ua) ? "ChromeOS"
    : /Mac OS X/.test(ua) ? (navigator.maxTouchPoints > 1 ? "iPadOS" : "macOS") : /Windows/.test(ua) ? "Windows" : /Linux/.test(ua) ? "Linux" : "unknown system";
  const name = [gpu.vendor, gpu.architecture].filter(Boolean).join(" ") || gpu.description || "unknown GPU";
  return { gpu: name, description: gpu.description || "", browser, os, cores: navigator.hardwareConcurrency || null };
}

// app: { device, model, chat, info (the adapter's), load: { ms, local }, weights: { lm, vision, ... } (set ids),
//   vision(), audio(), talk(): the models (each loads the first time; null where the build has none),
//   play: { begin(), play(Float32Array 24 kHz), end() } or null }
// opts: { short (smaller inputs, for tests), onRow(row, result), onStatus(text), shouldStop() }
export async function runSmoke(app, opts = {}) {
  const { model, chat, device } = app, short = !!opts.short;
  const rows = [];
  const res = { version: SMOKE_VERSION, at: new Date().toISOString(), short, site: location.host, device: deviceInfo(app.info),
    weights: { format: model.quant === "q8" ? "8-bit" : "4-bit", ...app.weights }, context: model.ctx, rows };
  const show = (row) => opts.onRow?.(row, res);
  const step = async (id, name, fn, need = true) => {
    const row = { id, name, state: "run", result: "", numbers: "", data: {} };
    rows.push(row);
    if (need !== true || opts.shouldStop?.()) { row.state = "skip"; row.result = need === true ? "stopped" : need; show(row); return; }
    show(row);
    const t0 = now();
    try {
      await fn(row);
      if (row.state === "run") row.state = "pass";
    } catch (e) {
      console.error(e);
      row.state = "fail";
      row.result = e?.message || String(e);
    }
    row.data.totalMs = Math.round(now() - t0);
    show(row);
  };
  const status = (t) => opts.onStatus?.(t);
  const saved = { system: chat.system, spoken: chat.spoken };
  const t00 = now();
  const pic = picture();
  let spoken = null;
  try {
    chat.system = SYSTEM; chat.spoken = false;

    await step("load", "Models", async (row) => {
      const got = [`language model ${ms(app.load?.ms ?? NaN)}${app.load?.local ? " from this device" : ""}`];
      for (const [k, label] of [["vision", "image"], ["audio", "speech in"], ["talk", "voice"]]) {
        if (!app[k]) continue;
        status(`Loading the ${label} model…`);
        const t = now(), r = await app[k]();
        row.data[k] = { ms: Math.round(now() - t), how: r.how };
        got.push(`${label} ${r.how === "loaded" ? "already loaded" : `${ms(now() - t)}${r.how === "device" ? " from this device" : ""}`}`);
      }
      row.result = "ready";
      row.numbers = got.join(" · ");
    });

    await step("text", "Text", async (row) => {
      status("Text…");
      await chat.newChat();
      let first = 0;
      const t0 = now();
      const a = await chat.send(Q_TEXT, { maxNew: 24, onText: () => { first ||= now(); }, shouldStop: opts.shouldStop });
      const ttft = first - t0;
      // throughput: a long prompt read at once, then a fixed number of new tokens (an end of turn does not stop it)
      chat.restartFromPrefix();
      const ids = chat.ids(turnUser(passage(short)));
      const t1 = now();
      await model.prefill(ids, { sample: true });
      const t2 = now();
      let f = 0;
      const out = await model.generate(short ? 32 : 128, { noStop: true, onToken: () => { f ||= now(); } });
      const t3 = now();
      const prefillTokS = ids.length / ((t2 - t1) / 1000), decodeTokS = (out.length - 1) / ((t3 - f) / 1000);
      chat.restartFromPrefix();
      row.data = { answer: a.reply, firstTokenMs: Math.round(ttft), promptTokens: ids.length, prefillMs: Math.round(t2 - t1),
        prefillTokS: +prefillTokS.toFixed(1), newTokens: out.length, decodeTokS: +decodeTokS.toFixed(2) };
      const ok = /paris/i.test(a.reply);
      row.state = ok ? "pass" : "fail";
      row.result = ok ? quote(a.reply, 24) : `expected Paris, got ${quote(a.reply, 32)}`;
      row.numbers = `first token ${ms(ttft)} · prefill ${num(prefillTokS)} tok/s (${ids.length} tokens) · decode ${num(decodeTokS)} tok/s (${out.length} tokens)`;
    });

    await step("photo", "Photo", async (row) => {
      const v = (await app.vision()).model;
      status("Photo…");
      await chat.newChat();
      const t0 = now();
      const pp = preprocess(pic.rgba, pic.w, pic.h, short ? { maxPixels: 50176 } : {});
      const o = await v.encode(pp.patches, pp.gh, pp.gw);
      await device.queue.onSubmittedWorkDone();
      const t1 = now();
      model.setExternalEmbeddings(o.emb);
      let first = 0;
      const a = await chat.send(Q_PHOTO, { media: { token: "<|image|>", rows: o.rows }, maxNew: 24, onText: () => { first ||= now(); }, shouldStop: opts.shouldStop });
      chat.restartFromPrefix();
      row.data = { answer: a.reply, size: `${pp.width}x${pp.height}`, imageTokens: o.rows, encodeMs: Math.round(t1 - t0),
        prefillTokS: +a.stats.prefillTokS.toFixed(1), firstTokenMs: Math.round(first - t0) };
      const ok = /red/i.test(a.reply);
      row.state = ok ? "pass" : "fail";
      row.result = ok ? quote(a.reply, 24) : `expected red, got ${quote(a.reply, 32)}`;
      row.numbers = `${o.rows} image tokens (${pp.width}×${pp.height}) encoded in ${ms(t1 - t0)} · prefill ${num(a.stats.prefillTokS)} tok/s · ` +
        `first token ${ms(first - t0)} after the photo`;
    }, app.vision ? true : "not in this build");

    await step("speak", "Spoken reply", async (row) => {
      const { depth, codec } = (await app.talk()).model;
      status("Spoken reply…");
      chat.spoken = true;
      await chat.newChat();
      const chunks = [], maxFrames = short ? 100 : 300;
      let first = 0;
      const t0 = now();
      app.play?.begin();
      const r = await chat.speak(Q_SPEAK, { depth, codec, maxFrames, shouldStop: opts.shouldStop,
        onAudio: (x) => { first ||= now(); chunks.push(x); app.play?.play(x); } });
      app.play?.end();
      const gen = r.stats.ms / 1000;
      spoken = { audio: concat(chunks), text: r.reply, cut: r.frames.length >= maxFrames };
      row.data = { reply: r.reply, speechS: +r.seconds.toFixed(2), firstAudioMs: Math.round(first - t0), frames: r.stats.frames,
        framesPerS: +(r.stats.frames / gen).toFixed(2), realTime: +(r.seconds / gen).toFixed(2), textTokens: r.textIds.length };
      const ok = r.seconds >= 0.5 && !!r.reply.trim();
      row.state = ok ? "pass" : "fail";
      row.result = ok ? `${r.seconds.toFixed(1)} s of speech` : `no speech (${r.seconds.toFixed(1)} s)`;
      row.numbers = `first sound ${ms(first - t0)} · ${num(r.stats.frames / gen)} frames/s (12.5 is real time) · ${num(r.seconds / gen)}× real time · ${quote(r.reply)}`;
    }, app.talk ? true : "not in this build");

    let x16 = null;
    await step("asr", "Speech in", async (row) => {
      const enc = (await app.audio()).model;
      status("Speech in…");
      x16 = resample16k(spoken.audio, 24000);
      const secs = x16.length / 16000, lm = logMel(x16);
      const t0 = now();
      const o = await enc.encode(lm.mel, lm.frames);
      await device.queue.onSubmittedWorkDone();
      const t1 = now();
      const heard = await chat.asr(o.emb, o.rows, "English");
      const t2 = now();
      // a reply cut at the frame limit is compared with as many of its words as were heard
      const ref = spoken.cut ? spoken.text.split(/\s+/).slice(0, heard.split(/\s+/).length).join(" ") : spoken.text;
      const acc = wordAccuracy(ref, heard);
      row.data = { heard, speechS: +secs.toFixed(2), audioTokens: o.rows, encodeMs: Math.round(t1 - t0), asrMs: Math.round(t2 - t1),
        realTime: +(secs / ((t2 - t0) / 1000)).toFixed(2), wordAccuracy: +acc.toFixed(3) };
      row.state = acc >= 0.6 ? "pass" : "fail";
      row.result = `${Math.round(acc * 100)}% of words heard`;
      row.numbers = `${secs.toFixed(1)} s of speech heard in ${ms(t2 - t0)} (${num(secs / ((t2 - t0) / 1000))}× real time) · encoder ${ms(t1 - t0)} · ${quote(heard)}`;
    }, !app.audio ? "not in this build" : spoken ? true : "needs the spoken reply");

    await step("ptt", "Push to talk + camera", async (row) => {
      const enc = (await app.audio()).model, { depth, codec } = (await app.talk()).model, v = app.vision ? (await app.vision()).model : null;
      status("Push to talk…");
      x16 ||= resample16k(spoken.audio, 24000);
      await chat.newChat();
      let cam = null;
      if (v) { // the camera's picture at Live's size, taken as the user starts to speak
        const pp = preprocess(pic.rgba, pic.w, pic.h, { maxPixels: short ? 50176 : 102400 });
        cam = await v.encode(pp.patches, pp.gh, pp.gw);
      }
      await chat.openTurn(cam);
      // the speech arrives as fast as someone says it; every 320 ms the encoder runs on what has arrived and its
      // final tokens go into the model, as the app does while the button is held
      const mel = new MelStream(30), STEP = 5120, t0 = now();
      for (let i = 0; i < x16.length && !opts.shouldStop?.(); i += STEP) {
        mel.push(x16.subarray(i, i + STEP));
        const final = mel.complete();
        if (final > chat.turn.rows) {
          const width = Math.min(3000, Math.ceil((mel.frames + 1) / 64) * 64);
          const o = await enc.encode(mel.layout(width), width);
          await chat.feedAudio(o.emb, Math.min(final, o.rows));
        }
        const wait = t0 + ((i + STEP) / 16000) * 1000 - now();
        if (wait > 0) await sleep(wait);
      }
      const release = now();
      mel.finish();
      const o = await enc.encode(mel.layout(), mel.frames);
      const streamed = chat.turn.rows;
      await chat.feedAudio(o.emb, o.rows);
      let first = 0;
      app.play?.begin();
      const r = await chat.closeTurn("", { depth, codec, maxFrames: short ? 40 : 250, shouldStop: opts.shouldStop,
        onAudio: (y) => { first ||= now(); app.play?.play(y); } });
      app.play?.end();
      row.data = { speechS: +(x16.length / 16000).toFixed(2), audioTokens: o.rows, streamedTokens: streamed, cameraTokens: cam?.rows || 0,
        firstAudioAfterReleaseMs: first ? Math.round(first - release) : null, replyS: +r.seconds.toFixed(2), reply: r.reply };
      row.state = first ? "pass" : "fail";
      row.result = first ? `voice ${ms(first - release)} after letting go` : "no answer";
      row.numbers = `${streamed} of ${o.rows} speech tokens taken in while speaking${cam ? ` · camera ${cam.rows} tokens` : ""} · ` +
        `${r.seconds.toFixed(1)} s answer · ${quote(r.reply)}`;
    }, !(app.audio && app.talk) ? "not in this build" : spoken ? true : "needs the spoken reply");
  } finally {
    Object.assign(chat, saved);
  }
  res.ms = Math.round(now() - t00);
  res.passed = rows.filter((r) => r.state === "pass").length;
  res.failed = rows.filter((r) => r.state === "fail").length;
  res.total = rows.filter((r) => r.state !== "skip").length;
  status("");
  return res;
}

// the results as plain text: for a post, an issue or a message
export function smokeText(res) {
  const d = res.device, L = [
    `LFM3-Omni-2B in the browser (WebGPU) · smoke test v${res.version}${res.short ? " (short)" : ""} · ${res.passed} of ${res.total} passed`,
    `${d.gpu}${d.description && !d.gpu.includes(d.description) ? ` (${d.description})` : ""} · ${d.os} · ${d.browser} · ${res.weights.format} weights`,
  ];
  for (const r of res.rows) {
    L.push(`${r.name}: ${r.state === "pass" ? "" : r.state === "skip" ? "skipped, " : "FAILED, "}${r.result}${r.numbers ? ` · ${r.numbers}` : ""}`);
  }
  L.push(`Run it yourself: https://${res.site || "lfm.toasterclub.com"} → Settings → Smoke test`);
  return L.join("\n");
}
