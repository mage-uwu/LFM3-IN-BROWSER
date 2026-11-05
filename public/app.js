// SPDX-License-Identifier: Apache-2.0. Copyright 2026 Toaster Club LLC.
import { initDevice, LFM3 } from "./engine/lfm3.js";
import { Chat, Experience, loadTokenizer, SYSTEM } from "./engine/chat.js";
import { store } from "./engine/store.js";
import { preprocess, Vision } from "./engine/vision.js";
import { Audio, logMel, MelStream } from "./engine/audio.js";
import { Codec, DepthDecoder } from "./engine/speech.js";
import { buf } from "./engine/lfm3.js";
import { MicCapture, Resampler, resample16k } from "./engine/listen.js";
import { cachedTokenizer, weightSet } from "./engine/loader.js";   // the device copy of each weight set
import { runSmoke, smokeText } from "./smoke.js";                    // Settings: the smoke test

const $ = (id) => document.getElementById(id);
// inside a claude.ai artifact the page cannot reach other hosts: the weights come from a picked file
const ARTIFACT = !!globalThis.LFM3_ARTIFACT;
// the artifact build carries the model itself: GPTQ 4-bit weights in the artifact's own asset store, as PNG byte
// containers (tools/gptq_export.py, tools/png_chunks.py) that the page streams into the GPU
const WEIGHTS = globalThis.LFM3_WEIGHTS || null;
let gpu, model, chat, busy = false, stopFlag = false, file = null;
let answering = null;   // the reply being generated: a new spoken turn stops it and waits for the model

const LM = WEIGHTS ? weightSet(WEIGHTS, "w/") : null;
const VISION = globalThis.LFM3_VISION || null;   // the image model's weights, when the build carries them
const VIS = VISION ? weightSet(VISION, "v/") : null;
const SPEECH = globalThis.LFM3_AUDIO || null;    // the speech encoder's weights, when the build carries them
const AUD = SPEECH ? weightSet(SPEECH, "a/") : null;
const TALK = globalThis.LFM3_TALK || null, CODEC = globalThis.LFM3_CODEC || null;   // spoken replies' weights
const TLK = TALK ? weightSet(TALK, "s/") : null, CDC = CODEC ? weightSet(CODEC, "c/") : null;
async function showSaved() {
  const s = await LM.saved();
  const n = WEIGHTS.chunks.length;
  $("dropBtn").hidden = !s || !s.n;
  $("loadBtn").textContent = s?.complete ? "Load the model (saved on this device)"
    : s?.n ? `Continue loading the model (${Math.round((100 * s.n) / n)}% saved)` : "Load the model";
}
function bubble(role, text) {
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  div.textContent = text;
  $("msgs").append(div);
  $("main").scrollTop = $("main").scrollHeight;
  syncHello();
  return div;
}
const note = (text) => bubble("assistant notice", text);   // the app's own remarks, quieter than the model's replies
// the greeting shows until the first message of a chat
function syncHello() { $("hello").hidden = !chat || !!$("msgs").querySelector(".msg.user"); }
// the composer's round button: Live while there is nothing to send, Send (or Stop while answering) otherwise
const LIVE = !!(SPEECH && TALK && CODEC);
function syncComposer() { // the round button: push to talk while there is nothing typed, Send (Stop while answering)
  const has = !!($("input").value.trim() || photo || voice), talkable = LIVE && !!chat;
  $("talkBtn").hidden = !talkable || has;
  $("send").hidden = talkable && !has && !busy;
  $("liveBtn").hidden = !talkable;
  $("send").classList.toggle("busy", busy);
  $("send").setAttribute("aria-label", busy ? "Stop" : "Send");
}
const status = (html) => { $("loadMsg").innerHTML = html; };

async function boot() {
  $("hubBtn").hidden = ARTIFACT || !!WEIGHTS;   // a page that carries its weights does not need the bf16 original
  $("photoBtn").hidden = !VISION;
  $("speakBtn").hidden = !(TALK && CODEC);
  $("hintLive").hidden = !LIVE; $("hintPhoto").hidden = !VISION; $("hintVoice").hidden = !LIVE;
  $("liveCam").hidden = $("liveFlip").hidden = !VISION;
  if (WEIGHTS) {
    $("steps").hidden = true;
    $("quantWrap").hidden = true;
    $("assetNote").hidden = false;
    $("assetNote").textContent = `The model's weights (4-bit, ${(WEIGHTS.total / 1e9).toFixed(2)} GB) are part of this page. ` +
      "The first load streams them into the GPU and keeps a copy in this browser's storage on the device (not in " +
      "Files), so later loads skip the download.";
  }
  try {
    gpu = await initDevice();
    const i = gpu.info;
    $("chip").textContent = [i.vendor, i.architecture || i.description].filter(Boolean).join(" ") || "WebGPU ready";
    $("chip").title = i.description || $("chip").textContent;
    if (WEIGHTS) { $("bits").textContent = "4-bit"; $("bits").hidden = false; }   // the site's weights: GPTQ 4-bit
    $("hubBtn").disabled = false;
    if (WEIGHTS) {
      $("loadBtn").disabled = false;
      await showSaved();
      try {  // an earlier version kept a GPU read-back copy in Cache Storage: the parts in IndexedDB replace it
        for (const k of await caches.keys()) if (k.startsWith("lfm3-webgpu-q4g64")) await caches.delete(k);
      } catch { /* no Cache Storage here */ }
    } else if (await LFM3.cached($("quant").value)) $("cachedBtn").hidden = false;
  } catch (e) {
    $("chip").textContent = "no WebGPU";
    $("hubBtn").disabled = true;
    status(`<span class="err">${e.message}. On iPhone, open this page in Safari (iOS 26 or later).</span>`);
  }
}

$("file").onchange = () => {
  file = $("file").files[0] || null;
  $("fname").textContent = file ? `${file.name} · ${(file.size / 1e9).toFixed(2)} GB` : "No file chosen";
  $("loadBtn").disabled = !file || !gpu;
};
$("quant").onchange = async () => { $("cachedBtn").hidden = !!WEIGHTS || !(gpu && await LFM3.cached($("quant").value)); };

async function load(source) {
  for (const b of ["loadBtn", "cachedBtn", "hubBtn", "dropBtn"]) $(b).disabled = true;
  $("prog").hidden = false;
  const t0 = performance.now();
  let src = null, saved = "";
  try {
    if (WEIGHTS) {
      src = LM.source();
      source = { prequant: src.prequant };
      await LM.prepare(src);
    }
    model = new LFM3(gpu.device, { quant: $("quant").value, ctx: +$("ctx").value, ...source });
    const tokP = WEIGHTS ? cachedTokenizer() : loadTokenizer();
    await model.load((p) => {
      $("prog").value = p.got / p.total;
      const gb = `${(p.got / 1e9).toFixed(2)} of ${(p.total / 1e9).toFixed(2)} GB`;
      const how = src && src.fromDevice ? (src.fromDevice === WEIGHTS.chunks.length || !src.saved ? "from this device" : "from this device and the page")
        : `into the page${src?.saving ? ", keeping a copy on this device" : ""}`;
      status(p.phase === "download" ? (WEIGHTS ? `loading the model ${how}: ${gb}` : `converting ${gb} on the GPU`)
        : p.phase === "cache" ? `loading the saved copy (${p.got}/${p.total})` : `saving a copy in this browser (${p.got}/${p.total})`);
    }, { cache: !WEIGHTS, cacheOnly: !!source.cacheOnly });
    if (src) saved = await LM.finish(src);
    chat = new Chat(model, await tokP);
    applySystem();   // the user's own instructions, if they set some
    const info = await chat.newChat();
    $("setup").hidden = true;
    $("input").disabled = false; $("send").disabled = false; $("memBtn").disabled = false; $("newChatTop").disabled = false; $("setBtn").disabled = false;
    $("photoBtn").disabled = !VISION;
    $("talkBtn").disabled = !LIVE;
    $("speakBtn").disabled = !(TALK && CODEC);
    $("liveBtn").disabled = !LIVE;
    $("input").placeholder = "Ask anything";
    syncComposer();
    const s = ((performance.now() - t0) / 1000).toFixed(0);
    $("bits").textContent = model.quant === "q8" ? "8-bit" : "4-bit";
    $("bits").hidden = false;
    const mem = chat.experience.empty() ? "" : ` Memory loaded: ${Object.keys(chat.experience.facts).length} facts (${info.tokens} tokens in ${info.ms.toFixed(0)} ms).`;
    const kept = !src ? (model.cached === false ? " This browser had no room to save a copy, so next time choose the file again." : "")
      : src.fromDevice === WEIGHTS.chunks.length ? ""
      : saved === "saved" ? " A copy is now saved on this device, so the next load skips the download."
      : ` This browser could not keep a copy (${src.saveError?.message || "storage unavailable"}), so the next load downloads it again.`;
    const local = model.fromCache || (src && src.fromDevice === WEIGHTS.chunks.length);
    loadStats = { ms: performance.now() - t0, local: !!local };
    note(`Ready in ${s} s${local ? ", from this device" : ""}.${kept}${mem}`);
    await learnLastVisit();
  } catch (e) {
    console.error(e);
    const oom = /memory|allocat|lost/i.test(String(e.message));
    status(`<span class="err">${e.message}${oom && !WEIGHTS && $("quant").value === "q8" ? ". Try 4-bit." : ""}</span>`);
    $("cachedBtn").disabled = false; $("hubBtn").disabled = false; $("dropBtn").disabled = false;
    $("loadBtn").disabled = !file && !WEIGHTS;
    if (WEIGHTS) await showSaved();  // the parts kept so far stay: the next try resumes from them
  }
}
$("loadBtn").onclick = () => load(WEIGHTS ? {} : { file });
let dropArmed = false;
$("dropBtn").onclick = async () => {  // two taps: remove the weights kept on this device
  if (!dropArmed) { dropArmed = true; $("dropBtn").textContent = "Tap again to remove"; setTimeout(() => { dropArmed = false; $("dropBtn").textContent = "Remove the saved copy"; }, 4000); return; }
  dropArmed = false;
  try { await store.delWhere((k) => k.startsWith("w/")); } catch { /* nothing kept */ }
  $("dropBtn").textContent = "Remove the saved copy";
  status("The saved copy is removed from this device.");
  await showSaved();
};
$("cachedBtn").onclick = () => load({ cacheOnly: true });
$("hubBtn").onclick = () => load({});

// ---- image mode: a photo goes through the vision tower (loaded the first time it is needed) into the prompt ------
let vision = null, visionLoading = null, photo = null;
const MAXPX = +(new URLSearchParams(location.search).get("maxpx") || 0);  // tests: fewer image tokens
async function ensureVision(onStatus) {
  if (vision) return vision;
  visionLoading ||= (async () => {
    const src = VIS.source();
    await VIS.prepare(src);
    const v = new Vision(gpu.device, { quant: VISION.format, prequant: src.prequant });
    await v.load((p) => onStatus(`Loading the image model${src.fromDevice ? " from this device" : ""}: ` +
      `${(p.got / 1e6).toFixed(0)} of ${(p.total / 1e6).toFixed(0)} MB`));
    await VIS.finish(src);
    vision = v;
    return v;
  })().finally(() => { visionLoading = null; });
  return visionLoading;
}
function thumbnail(cv, max = 180) { // a small copy of the picture for the transcript (no image URLs needed)
  const s = Math.min(1, max / Math.max(cv.width, cv.height));
  const t = document.createElement("canvas");
  t.width = Math.round(cv.width * s); t.height = Math.round(cv.height * s);
  t.getContext("2d").drawImage(cv, 0, 0, t.width, t.height);
  t.className = "thumb";
  return t;
}
$("photoBtn").onclick = () => $("photo").click();
$("photo").onchange = async () => {
  const file = $("photo").files[0];
  $("photo").value = "";
  if (!file) return;
  try {
    const bmp = await createImageBitmap(file);
    const s = Math.min(1, 1024 / Math.max(bmp.width, bmp.height));  // the model sees at most 448x448 worth of pixels
    const cv = document.createElement("canvas");
    cv.width = Math.max(32, Math.round(bmp.width * s)); cv.height = Math.max(32, Math.round(bmp.height * s));
    const cx = cv.getContext("2d");
    cx.imageSmoothingQuality = "high";
    cx.drawImage(bmp, 0, 0, cv.width, cv.height);
    bmp.close?.();
    photo = { rgba: cx.getImageData(0, 0, cv.width, cv.height).data, w: cv.width, h: cv.height, canvas: cv };
    $("attachThumb").replaceChildren(thumbnail(cv, 48));
    $("attach").hidden = false;
  } catch (e) {
    photo = null;
    $("attach").hidden = true;
    note(`Could not read that picture (${e.message}).`);
  }
  syncComposer();
};
$("attachDrop").onclick = () => { photo = null; $("attach").hidden = true; syncComposer(); };

// ---- speech input: a recording goes through the speech encoder (loaded the first time) into the prompt ---------
let audioEnc = null, audioLoading = null, voice = null;
async function ensureAudio(onStatus) {
  if (audioEnc) return audioEnc;
  audioLoading ||= (async () => {
    const src = AUD.source();
    await AUD.prepare(src);
    const a = new Audio(gpu.device, { quant: SPEECH.format, prequant: src.prequant });
    await a.load((p) => onStatus(`Loading the speech model${src.fromDevice ? " from this device" : ""}: ` +
      `${(p.got / 1e6).toFixed(0)} of ${(p.total / 1e6).toFixed(0)} MB`));
    await AUD.finish(src);
    audioEnc = a;
    return a;
  })().finally(() => { audioLoading = null; });
  return audioLoading;
}
async function toMono16k(data) { // any recording the browser can decode -> 16 kHz mono samples
  const AC = globalThis.AudioContext || globalThis.webkitAudioContext;
  const ctx = new AC();
  const decoded = await ctx.decodeAudioData(data);
  ctx.close?.();
  const n = Math.max(1, Math.ceil(decoded.duration * 16000));
  const off = new OfflineAudioContext(1, n, 16000);
  const src = off.createBufferSource();
  src.buffer = decoded;
  src.connect(off.destination);
  src.start();
  return (await off.startRendering()).getChannelData(0).slice();
}
function showVoice() {
  $("voiceChip").hidden = !voice;
  if (voice) $("voiceLen").textContent = `Voice message · ${voice.seconds.toFixed(1)} s`;
  syncComposer();
}
async function attachVoice(blob) {
  try {
    let samples = await toMono16k(await blob.arrayBuffer());
    if (samples.length > 30 * 16000) samples = samples.slice(0, 30 * 16000);   // the encoder takes up to 30 s
    voice = { samples, seconds: samples.length / 16000 };
  } catch (e) {
    voice = null;
    note(`Could not read that recording (${e.message}).`);
  }
  showVoice();
}
$("voiceFileBtn").onclick = () => $("voiceFile").click();
$("voiceFile").onchange = async () => {
  const file = $("voiceFile").files[0];
  $("voiceFile").value = "";
  if (file) { $("voiceFileRow").hidden = true; await attachVoice(file); }
};
$("voiceDrop").onclick = () => { voice = null; showVoice(); };
// ---- spoken replies: the depth decoder and the codec (loaded the first time), played as they are decoded ---------
let talk = null, talkLoading = null, player = null;
async function loadSet(set, W, Cls, label, onStatus) {
  const src = set.source();
  await set.prepare(src);
  const x = new Cls(gpu.device, { quant: W.format, prequant: src.prequant });
  await x.load((p) => onStatus(`Loading the ${label}${src.fromDevice ? " from this device" : ""}: ` +
    `${(p.got / 1e6).toFixed(0)} of ${(p.total / 1e6).toFixed(0)} MB`));
  await set.finish(src);
  return x;
}
async function ensureTalk(onStatus) {
  if (talk) return talk;
  talkLoading ||= (async () => {
    const depth = await loadSet(TLK, TALK, DepthDecoder, "speech model", onStatus);
    const codec = await loadSet(CDC, CODEC, Codec, "voice codec", onStatus);
    talk = { depth, codec };
    return talk;
  })().finally(() => { talkLoading = null; });
  return talkLoading;
}
// The answer's sound: 24 kHz chunks played back to back (the context is created in a tap, as iOS requires). When
// the device makes speech slower than it plays, a chunk arrives after the last one ended; playback then waits for
// a longer stretch (doubling each time, up to 4 s), so the answer pauses between phrases instead of stuttering.
function makePlayer() {
  const AC = globalThis.AudioContext || globalThis.webkitAudioContext;
  const ctx = new AC();
  let queue = [], queued = 0, playing = false, next = 0, ended = true, need = 0.3;
  const live = new Set();
  const out = ctx.createGain(), an = ctx.createAnalyser(), wave = new Float32Array(512);   // the sound, for the bubble's mouth
  an.fftSize = 512;
  out.connect(an);
  an.connect(ctx.destination);
  function schedule(x) {
    globalThis.LFM3_PLAYED = (globalThis.LFM3_PLAYED || 0) + x.length;   // samples handed to the speaker (tests)
    const b = ctx.createBuffer(1, x.length, 24000);
    b.getChannelData(0).set(x);
    const src = ctx.createBufferSource();
    src.buffer = b;
    src.connect(out);
    src.start(next);
    next += b.duration;
    live.add(src);
    src.onended = () => live.delete(src);
  }
  function pump() {
    if (!playing) {
      if (!queue.length || (queued < need && !ended)) return;
      playing = true;
      next = ctx.currentTime + 0.05;
    }
    while (queue.length) {
      if (next < ctx.currentTime) { playing = false; need = Math.min(4, need * 2 + 0.25); pump(); return; }   // ran dry
      const x = queue.shift();
      queued -= x.length / 24000;
      schedule(x);
    }
  }
  const P = {
    ctx,
    begin() { P.stop(); ended = false; need = 0.3; },   // a new answer
    play(x) { if (!x.length) return; queue.push(x); queued += x.length / 24000; pump(); },
    end() { ended = true; pump(); },                     // the answer is complete: play what is left
    stop() {
      for (const s of live) { try { s.stop(); } catch { /* ended */ } }
      live.clear();
      queue = []; queued = 0; playing = false; next = 0; ended = true;
    },
    left() { return Math.max(0, next - ctx.currentTime) + queued; },   // seconds not yet heard
    level() { // loudness of what is playing now, 0..1
      an.getFloatTimeDomainData(wave);
      let s = 0;
      for (let i = 0; i < wave.length; i++) s += wave[i] * wave[i];
      return Math.min(1, Math.sqrt(s / wave.length) * 7);
    },
  };
  return P;
}
function setSpeak(on) {
  chat.spoken = on;
  document.body.classList.toggle("voice", on);   // voice mode: the bubble above the composer
  $("dock").hidden = !on;
  $("speakBtn").setAttribute("aria-pressed", String(on));
  $("speakBtn").classList.toggle("on", on);
}
$("speakBtn").onclick = async () => {
  if (busy) return;
  const on = !chat.spoken;
  if (on) { player ||= makePlayer(); player.ctx.resume?.(); }
  setSpeak(on);
  const n = note(on ? "Spoken replies on. Starting a new chat…" : "Spoken replies off. Starting a new chat…");
  if (on) {
    try { await ensureTalk((m) => { n.textContent = m; }); } catch (e) {
      setSpeak(false);
      n.innerHTML = `<span class="err">${e.message}</span>`;
      return;
    }
  }
  const msgs = [...earlier, ...chat.messages];
  busy = true;
  syncComposer();
  let learned = "";
  try { const ch = await learnFrom(msgs, (t) => { n.textContent = t; }); if (ch?.length) learned = ` Remembered: ${told(ch)}.`; }
  catch (e) { console.error(e); await chat.newChat(); }
  busy = false;
  syncComposer();
  n.textContent = (on ? "Spoken replies on: the answers are read aloud as they are written. New chat started." : "Spoken replies off. New chat started.") + learned;
};

// several media items in one turn: their rows go back to back into one buffer, in the turn's order
function joinRows(parts) {
  if (parts.length === 1) return parts[0].emb;
  const d = gpu.device, rows = parts.reduce((a, p) => a + p.rows, 0);
  const out = buf(d, rows * 2048 * 4);
  const enc = d.createCommandEncoder();
  let off = 0;
  for (const p of parts) { enc.copyBufferToBuffer(p.emb, 0, out, off, p.rows * 2048 * 4); off += p.rows * 2048 * 4; }
  d.queue.submit([enc.finish()]);
  joinRows.last?.destroy();
  joinRows.last = out;
  return out;
}

async function send() {
  if (busy) { stopFlag = true; if (smokeRun) smokeStop = true; player?.stop(); return; }
  const text = $("input").value.trim();
  const pic = photo, vc = voice;
  if (!text && !pic && !vc) return;
  $("input").value = "";
  $("input").style.height = "auto";
  const ub = bubble("user", text);
  if (pic) { ub.prepend(thumbnail(pic.canvas)); photo = null; $("attach").hidden = true; }
  if (vc) {
    const tag = document.createElement("div");
    tag.className = "voicetag";
    tag.textContent = `Voice message · ${vc.seconds.toFixed(1)} s`;
    ub.prepend(tag);
    voice = null;
    showVoice();
  }
  const out = bubble("assistant", "…");
  busy = true; stopFlag = false;
  let done;
  answering = new Promise((ok) => { done = ok; });
  syncComposer();
  try {
    const media = [], parts = [];
    let seen = "";
    if (pic) {
      const t0 = performance.now();
      out.textContent = "Looking at the picture…";
      const v = await ensureVision((m) => { out.textContent = m; });
      const pp = preprocess(pic.rgba, pic.w, pic.h, MAXPX ? { maxPixels: MAXPX } : {});
      const o = await v.encode(pp.patches, pp.gh, pp.gw);
      parts.push(o);
      media.push({ token: "<|image|>", rows: o.rows });
      seen = `image ${pp.width}x${pp.height} · ${o.rows} tokens in ${((performance.now() - t0) / 1000).toFixed(1)} s  |  `;
      if (vc) await gpu.device.queue.onSubmittedWorkDone();
    }
    if (vc) {
      const t0 = performance.now();
      out.textContent = "Listening…";
      const a = await ensureAudio((m) => { out.textContent = m; });
      const lm = logMel(vc.samples);
      const o = await a.encode(lm.mel, lm.frames);
      parts.push(o);
      media.push({ token: "<|audio|>", rows: o.rows });
      seen += `speech ${vc.seconds.toFixed(1)} s · ${o.rows} tokens in ${((performance.now() - t0) / 1000).toFixed(1)} s  |  `;
    }
    if (parts.length) { model.setExternalEmbeddings(joinRows(parts)); out.textContent = "…"; }
    const r = (v) => (v >= 10 ? v.toFixed(0) : v.toFixed(1));
    const onText = (t) => { out.textContent = t; $("main").scrollTop = $("main").scrollHeight; };
    let line;
    if (chat.spoken) {
      const t = await ensureTalk((m) => { out.textContent = m; });
      player ||= makePlayer();
      player.begin();
      face("thinking");
      const res = await chat.speak(text || (vc ? "" : "Describe this image."), { media, depth: t.depth, codec: t.codec, onText, audio: vc?.samples || null,
        onAudio: (a) => { if (FACE.state !== "speaking") face("speaking"); player.play(a); }, shouldStop: () => stopFlag });
      player.end();
      faceAfterPlayback();
      out.textContent = res.reply;
      const s = res.stats;
      line = `${seen}${s.promptTokens} prompt tokens  |  ${res.seconds.toFixed(1)} s of speech in ${(s.ms / 1000).toFixed(1)} s ` +
        `(${r(s.frames / (s.ms / 1000))} frames/s, 12.5 is real time)  |  context ${s.pos}`;
    } else {
      const { reply, stats } = await chat.send(text || (vc ? "" : "Describe this image."), { media, onText, shouldStop: () => stopFlag, audio: vc?.samples || null });
      out.textContent = reply;
      line = `${seen}${stats.promptTokens} prompt tokens · ${r(stats.prefillTokS)} tok/s  |  ${stats.newTokens} tokens · ${r(stats.decodeTokS)} tok/s  |  context ${stats.pos}`;
    }
    const st = document.createElement("div");
    st.className = "stats";
    st.textContent = line;
    out.append(st);
    savePending();
  } catch (e) {
    out.innerHTML = `<span class="err">${e.message}</span>`;
  }
  busy = false;
  answering = null;
  done();
  syncComposer();
}
$("send").onclick = send;
$("input").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } });
$("input").addEventListener("input", () => { const t = $("input"); t.style.height = "auto"; t.style.height = t.scrollHeight + "px"; syncComposer(); });

// ---- push to talk: one button in the composer, and the orb in Live. Hold it and speak, let go and the answer
// starts; a short tap starts and a second tap ends, for one hand. While the user speaks, the speech encoder runs
// every 320 ms on what has arrived and its final tokens go into the model (Chat.openTurn / feedAudio, as the
// official prefill_audio), so at the release only the last tokens and the turn's closing are left. The microphone
// opens at the press and closes at the release, so a phone plays the answer through its speaker, not the earpiece.
// Pressing while an answer plays stops it and starts the next turn.
const sleep = (ms) => new Promise((ok) => setTimeout(ok, ms));
const LIVE_PX = MAXPX || 102400;   // the camera picture the model sees: about 320x320, 100 image tokens
const TICK = 320, MIN_S = 0.35, TAP_MS = 300;
const PT = { state: "idle", hold: false, live: false, pressT: 0, endT: 0, cap: null, rs: null, mel: null, opened: null, feeding: null, limit: 0, frame: null };
const L = { on: false, video: null, facing: "environment", camOn: true, lock: null };
const hook = globalThis.LFM3_LIVE = { state: "", heard: 0, turns: 0, reply: "", frames: 0, streamed: 0, firstAudioMs: 0 };   // tests
const caption = (t) => { $("liveCaption").textContent = t.length > 220 ? "…" + t.slice(-220) : t; };
// the bubble's face, the same for the chat's bubble and Live's: idle (smiles, blinks), listening (eyes wide, swells
// with the voice), thinking (looks up), speaking (the mouth follows the sound going to the speaker, frame by frame)
const FACE = { state: "idle", talk: 0, loop: false };
const FACE_LABEL = { idle: "Hold to talk", listening: "Listening…", thinking: "Thinking…", speaking: "Hold to interrupt" };
function face(s) {
  FACE.state = s;
  for (const el of [$("buddy"), $("orb")]) el.dataset.state = s;
  $("buddyState").textContent = s === "listening" && !PT.hold ? "Listening · tap to send" : FACE_LABEL[s];
  if (s === "speaking" && !FACE.loop) { FACE.loop = true; requestAnimationFrame(mouth); }
}
function mouth() {
  const v = FACE.state === "speaking" && player ? player.level() : 0;
  FACE.talk = FACE.talk * 0.45 + v * 0.55;
  for (const el of [$("buddy"), $("orb")]) el.style.setProperty("--talk", FACE.talk.toFixed(3));
  if (FACE.state === "speaking") requestAnimationFrame(mouth); else FACE.loop = false;
}
async function faceAfterPlayback() { // speaking until the last of the answer has played, then idle
  while (player && player.left() > 0.05 && !busy && PT.state === "idle") await sleep(100);
  if (!busy && PT.state === "idle") face("idle");
}
const LIVE_FACE = { loading: "thinking", ready: "idle", listening: "listening", thinking: "thinking", speaking: "speaking", fallback: "idle", error: "idle" };
function liveState(s, text) {
  hook.state = s;
  $("liveState").textContent = text;
  face(LIVE_FACE[s]);
}
function talkUI() { // the composer's button and Live's orb follow the turn
  const rec = PT.state === "recording" || PT.state === "starting";
  $("talkBtn").classList.toggle("rec", rec);
  $("talkBtn").setAttribute("aria-label", rec ? (PT.hold ? "Release to send" : "Tap to send") : "Hold to talk");
  $("talkBtn").setAttribute("aria-pressed", String(rec));
  if (rec) face("listening");
  else if (PT.state === "finishing") face("thinking");
  else if (!busy && FACE.state !== "speaking") face("idle");
  if (!L.on) return;
  if (rec) liveState("listening", PT.hold ? "Listening · let go to send" : "Listening · tap to send");
  else if (PT.state === "finishing") liveState("thinking", "Thinking…");
  else if (!busy) liveState("ready", "Hold to talk");
}
const level = (v) => { for (const id of ["orb", "buddy", "talkBtn"]) $(id).style.setProperty("--level", v.toFixed(3)); };
async function stopAnswer() { // the answer playing, if any, stops; resolves when the model is free
  if (!answering) return;
  stopFlag = true;
  player?.stop();
  await answering;
}
async function spokenChat() { // speech in, speech out: the spoken system prompt when the chat is still empty
  if (chat.spoken) return;
  if (chat.messages.length) return;   // a typed chat stays as it is; the turn still asks for speech (<|mixed_start|>)
  setSpeak(true);
  await chat.newChat();
}
function frameCanvas(src, w, h) { // a picture at about the size the model sees
  const s = Math.min(1, Math.sqrt((LIVE_PX * 1.3) / (w * h)));
  const cv = document.createElement("canvas");
  cv.width = Math.max(32, Math.round(w * s)); cv.height = Math.max(32, Math.round(h * s));
  const cx = cv.getContext("2d");
  cx.imageSmoothingQuality = "high";
  cx.drawImage(src, 0, 0, cv.width, cv.height);
  return cv;
}
function liveFrame() { // the camera's picture now, or null
  const v = $("liveVideo");
  return L.on && L.camOn && L.video && v.videoWidth ? frameCanvas(v, v.videoWidth, v.videoHeight) : null;
}
async function encodePicture(cv) {
  const px = cv.getContext("2d").getImageData(0, 0, cv.width, cv.height).data;
  const pp = preprocess(px, cv.width, cv.height, { maxPixels: LIVE_PX });
  const o = await vision.encode(pp.patches, pp.gh, pp.gw);
  hook.frames++;
  return { ...o, size: `${pp.width}x${pp.height}` };
}
function noMicrophone(e, live) { // the page may not open the microphone (a claude.ai artifact): a recording instead
  if (live) { liveFallback(e); return; }
  $("voiceFileRow").hidden = false;
  $("voiceErr").textContent = `The microphone is not available here (${e.name || e.message}).`;
}
async function talkStart(live) {
  if (PT.state !== "idle" || !chat || !LIVE || smokeRun) return;
  Object.assign(PT, { state: "starting", live, frame: null, pic: null, opened: null, feeding: null, cancel: false });
  player ||= makePlayer();
  player.ctx.resume?.();   // in the press: iOS starts audio only from a user's gesture
  const stopping = stopAnswer();
  talkUI();
  try { if (navigator.audioSession) navigator.audioSession.type = "play-and-record"; } catch { /* older Safari */ }
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true } });
  } catch (e) {
    PT.state = "idle"; talkUI(); noMicrophone(e, live);
    return;
  }
  if (PT.state !== "starting") { stream.getTracks().forEach((t) => t.stop()); return; }
  PT.rs = new Resampler(player.ctx.sampleRate, 16000);
  PT.mel = new MelStream(30);
  PT.cap = new MicCapture(player.ctx, stream, {
    onLevel: level,
    onSamples: (x) => { if (PT.state !== "recording") return; PT.mel.push(PT.rs.push(x)); if (PT.mel.full) talkEnd(); },
  });
  PT.cap.start();
  PT.state = "recording";
  PT.limit = setTimeout(() => talkEnd(), 30500);
  talkUI();
  if (live) { const f = liveFrame(); if (f) PT.frame = f; }   // the picture as the user starts to speak
  const status = live ? caption : (m) => { $("input").placeholder = m; };
  PT.opened = (async () => { // the model's side, while the user talks
    await stopping;
    await ensureAudio(status);
    await ensureTalk(status);
    const pic = PT.frame && VISION ? await encodePicture(PT.frame) : null;
    if (!live) $("input").placeholder = "Ask anything";
    await spokenChat();
    if (model.pos + (pic?.rows || 0) + 375 + 1000 > model.ctx) contextFull();
    while (PT.state === "recording" && PT.mel.seconds < MIN_S) await sleep(40);   // a stray tap opens no turn
    if (PT.mel.seconds < MIN_S) return false;
    await chat.openTurn(pic);
    PT.pic = pic;
    return true;
  })();
  PT.feeding = (async () => { // every 320 ms: the tokens that became final go into the model
    if (!(await PT.opened)) return false;
    while (PT.state === "recording") {
      const final = PT.mel.complete();
      if (final > chat.turn.rows) {
        const width = Math.min(3000, Math.ceil((PT.mel.frames + 1) / 64) * 64);   // a few widths, so encoder plans are reused
        const o = await audioEnc.encode(PT.mel.layout(width), width);
        await chat.feedAudio(o.emb, Math.min(final, o.rows));
      }
      await sleep(TICK);
    }
    return true;
  })();
  PT.feeding.catch(() => {});
}
async function talkEnd() {
  if (PT.state === "starting") { PT.hold = false; talkUI(); return; }   // still opening the microphone: tap to send
  if (PT.state !== "recording") return;
  PT.state = "finishing";
  PT.endT = performance.now();
  clearTimeout(PT.limit);
  PT.cap.stop();
  try { if (navigator.audioSession) navigator.audioSession.type = "playback"; } catch { /* older Safari */ }
  level(0);
  PT.mel.push(PT.rs.finish());
  PT.mel.finish();
  const secs = PT.mel.seconds, live = PT.live;
  talkUI();
  let ok = false;
  try { ok = await PT.feeding; } catch (e) {
    console.error(e);
    PT.state = "idle"; talkUI();
    if (chat.turn) { // a half-taken turn would confuse the model: back to the start of the chat
      chat.turn = null;
      chat.restartFromPrefix();
      note(`Could not take that in (${e.message}). The conversation starts over from here.`);
    } else note(`Could not listen (${e.message}).`);
    return;
  }
  if (!ok) { PT.state = "idle"; talkUI(); if (live) caption("Hold the orb while you speak."); else note("Hold the button while you speak, then let go."); return; }
  const o = await audioEnc.encode(PT.mel.layout(), PT.mel.frames);   // the exact clip: its last tokens
  const streamed = chat.turn.rows;
  await chat.feedAudio(o.emb, o.rows);
  hook.heard++;
  hook.streamed = streamed;
  PT.state = "idle";
  // the transcript
  const ub = bubble("user", "");
  if (PT.frame && PT.pic) ub.append(thumbnail(PT.frame));
  const tag = document.createElement("div");
  tag.className = "voicetag";
  tag.textContent = `${live ? "Live" : "Voice"} · ${secs.toFixed(1)} s`;
  ub.append(tag);
  const out = bubble("assistant", "…");
  busy = true; stopFlag = false;
  let done;
  answering = new Promise((res) => { done = res; });
  syncComposer();
  talkUI();
  if (live) liveState("thinking", "Thinking…"); else face("thinking");
  let first = 0;
  try {
    player.begin();
    const res = await chat.closeTurn("", { depth: talk.depth, codec: talk.codec, maxFrames: 750, audio: PT.mel.x.slice(0, PT.mel.n),
      onText: (t) => { out.textContent = t; if (live) caption(t); $("main").scrollTop = $("main").scrollHeight; },
      onAudio: (x) => {
        if (!first) { first = performance.now() - PT.endT; hook.firstAudioMs = Math.round(first); if (live) liveState("speaking", "Speaking · hold to interrupt"); else face("speaking"); }
        player.play(x);
      },
      shouldStop: () => stopFlag || PT.cancel });
    player.end();
    out.textContent = res.reply;
    const s = res.stats, st = document.createElement("div");
    st.className = "stats";
    st.textContent = `${PT.pic ? `camera ${PT.pic.size} · ${PT.pic.rows} tokens  |  ` : ""}speech ${secs.toFixed(1)} s · ${o.rows} tokens, ` +
      `${streamed} taken in while you spoke  |  voice ${first ? (first / 1000).toFixed(1) + " s" : "–"} after you let go  |  ` +
      `${res.seconds.toFixed(1)} s of speech in ${(s.ms / 1000).toFixed(1)} s  |  context ${s.pos}`;
    out.append(st);
    if (live) caption(res.reply);
    hook.turns++;
    hook.reply = res.reply;
    savePending();
  } catch (e) {
    console.error(e);
    out.innerHTML = `<span class="err">${e.message}</span>`;
  }
  busy = false;
  answering = null;
  done();
  syncComposer();
  await faceAfterPlayback();   // the bubble speaks until the answer has played
  if (L.on && live && PT.state === "idle" && !busy) liveState("ready", "Hold to talk");
}
// the button: hold to talk, or tap to start and tap again to send
function pushToTalk(el, live) {
  el.addEventListener("pointerdown", (e) => {
    if (e.button > 0) return;
    e.preventDefault();
    try { el.setPointerCapture(e.pointerId); } catch { /* not capturable */ }
    if (PT.state === "recording" && !PT.hold) { PT.tapEnd = true; return; }   // the second tap ends it on release
    if (PT.state !== "idle") return;
    PT.hold = true; PT.tapEnd = false; PT.pressT = performance.now();
    talkStart(live);
  });
  const up = () => {
    if (PT.tapEnd) { PT.tapEnd = false; talkEnd(); return; }
    if (!PT.hold) return;
    if (performance.now() - PT.pressT < TAP_MS) { PT.hold = false; talkUI(); return; }   // a tap: keep listening
    PT.hold = false;
    talkEnd();
  };
  el.addEventListener("pointerup", up);
  el.addEventListener("pointercancel", up);
  el.addEventListener("contextmenu", (e) => e.preventDefault());
  el.addEventListener("keydown", (e) => { // Space or Enter: start, then again to send
    if (e.key !== " " && e.key !== "Enter") return;
    e.preventDefault();
    if (e.repeat) return;
    if (PT.state === "idle") { PT.hold = false; talkStart(live); } else talkEnd();
  });
}
pushToTalk($("talkBtn"), false);
pushToTalk($("orb"), true);
pushToTalk($("buddy"), false);

// ---- Live: the camera full screen, the orb to talk. The picture at the moment the user starts speaking goes
// with the speech. Where the page may not open the camera and microphone (a claude.ai artifact), a video recorded
// with the camera app takes their place.
function syncLiveButtons() {
  $("liveCam").classList.toggle("off", !L.camOn);
  $("liveCam").setAttribute("aria-pressed", String(L.camOn));
  $("liveCam").setAttribute("aria-label", L.camOn ? "Turn the camera off" : "Turn the camera on");
  $("liveFlip").disabled = !L.camOn;
}
async function liveCamera() { // (re)open the camera facing the chosen way, or show nothing
  L.video?.getTracks().forEach((t) => t.stop());
  L.video = null;
  const v = $("liveVideo");
  v.srcObject = null;
  v.hidden = true;
  if (!L.camOn || !VISION || !L.on) return;
  try {
    L.video = await navigator.mediaDevices.getUserMedia({ video: { facingMode: L.facing, width: { ideal: 640 }, height: { ideal: 480 } } });
    if (!L.on) { L.video.getTracks().forEach((t) => t.stop()); L.video = null; return; }
    v.srcObject = L.video;
    v.classList.toggle("front", L.facing === "user");
    v.hidden = false;
    await v.play().catch(() => {});
  } catch (e) {
    L.camOn = false;
    syncLiveButtons();
    caption(`The camera is not available (${e.name || e.message}), so this is voice only.`);
  }
}
async function liveOpen() {
  if (L.on || !LIVE || !chat || smokeRun) return;
  L.on = true;
  player ||= makePlayer();
  player.ctx.resume?.();
  $("live").hidden = false;
  $("liveFallback").hidden = true;
  $("orb").hidden = false;
  $("liveCam").hidden = $("liveFlip").hidden = !VISION;
  caption("");
  syncLiveButtons();
  liveState("loading", "Starting…");
  if (PT.state === "recording" || PT.state === "starting") { PT.cancel = true; talkEnd(); }   // a turn being spoken in the chat closes
  await stopAnswer();   // an answer still being made stops, as a press to talk would stop it
  try { L.lock = await navigator.wakeLock?.request("screen"); } catch { /* the screen may sleep */ }
  try { // the microphone's permission now, so the first press starts at once (it is opened per turn)
    const probe = await navigator.mediaDevices.getUserMedia({ audio: true });
    probe.getTracks().forEach((t) => t.stop());
  } catch (e) {
    liveFallback(e);
  }
  try {
    await ensureAudio(caption);
    await ensureTalk(caption);
    if (VISION) await ensureVision(caption);
    caption("");
  } catch (e) {
    liveState("error", "Live could not start");
    caption(e.message);
    return;
  }
  if (!L.on || hook.state === "fallback") return;
  await liveCamera();
  if (PT.state === "idle" && !busy) liveState("ready", "Hold to talk");
}
function liveFallback(e) { // no microphone for the page: a video from the camera app instead
  liveState("fallback", ARTIFACT ? "Record a video message" : "No microphone here");
  $("liveWhy").textContent = (ARTIFACT ? "Pages on claude.ai can't open the camera or microphone directly."
    : `The microphone is not available here (${e?.name || e?.message || "refused"}).`) +
    " Record a short video instead: point the camera, ask your question, then stop. The answer is spoken back.";
  $("liveFallback").hidden = false;
  $("orb").hidden = true;
  $("liveCam").hidden = $("liveFlip").hidden = true;
}
function liveClose() {
  L.on = false;
  if (PT.state === "recording" || PT.state === "starting") { PT.cancel = true; talkEnd(); }   // the turn closes, unanswered
  stopAnswer();
  L.video?.getTracks().forEach((t) => t.stop());
  L.video = null;
  $("liveVideo").srcObject = null;
  $("live").hidden = true;
  L.lock?.release?.().catch?.(() => {});
  L.lock = null;
  syncComposer();
}
// a whole recorded clip as one spoken turn (the video fallback)
async function clipTurn(samples, frame) {
  const parts = [], media = [];
  let seen = "";
  if (frame && vision) {
    const o = await encodePicture(frame);
    parts.push(o);
    media.push({ token: "<|image|>", rows: o.rows });
    seen = `camera ${o.size} · ${o.rows} tokens  |  `;
    await gpu.device.queue.onSubmittedWorkDone();
  }
  const lm = logMel(samples);
  const a = await audioEnc.encode(lm.mel, lm.frames);
  parts.push(a);
  media.push({ token: "<|audio|>", rows: a.rows });
  const secs = samples.length / 16000;
  seen += `speech ${secs.toFixed(1)} s · ${a.rows} tokens  |  `;
  hook.heard++;
  await spokenChat();
  const rows = parts.reduce((s, p) => s + p.rows, 0);
  if (model.pos + rows + 1000 > model.ctx) contextFull();
  model.setExternalEmbeddings(joinRows(parts));
  const ub = bubble("user", "");
  if (frame) ub.append(thumbnail(frame));
  const tag = document.createElement("div");
  tag.className = "voicetag";
  tag.textContent = `Video · ${secs.toFixed(1)} s`;
  ub.append(tag);
  const out = bubble("assistant", "…");
  player.begin();
  const res = await chat.speak("", { media, depth: talk.depth, codec: talk.codec, maxFrames: 750, audio: samples,
    onText: (t) => { caption(t); out.textContent = t; },
    onAudio: (x) => { player.play(x); liveState("speaking", "Speaking"); },
    shouldStop: () => stopFlag || !L.on });
  player.end();
  out.textContent = res.reply;
  const st = document.createElement("div");
  st.className = "stats";
  st.textContent = `${seen}${res.seconds.toFixed(1)} s of speech in ${(res.stats.ms / 1000).toFixed(1)} s  |  context ${res.stats.pos}`;
  out.append(st);
  caption(res.reply);
  hook.turns++;
  hook.reply = res.reply;
  savePending();
}
async function clipFrame(file) { // the clip's middle frame
  const url = URL.createObjectURL(file);
  try {
    const v = document.createElement("video");
    Object.assign(v, { muted: true, playsInline: true, preload: "auto", src: url });
    const until = (ev, ms) => new Promise((ok, no) => { v.addEventListener(ev, ok, { once: true }); v.onerror = () => no(new Error("unreadable video")); setTimeout(() => no(new Error("timeout")), ms); });
    await until("loadedmetadata", 6000);
    await v.play().catch(() => {});
    v.pause();
    if (Number.isFinite(v.duration) && v.duration > 0.2) { v.currentTime = v.duration / 2; await until("seeked", 6000); }
    else if (v.readyState < 2) await until("loadeddata", 6000);
    return v.videoWidth ? frameCanvas(v, v.videoWidth, v.videoHeight) : null;
  } finally { URL.revokeObjectURL(url); }
}
async function clipSound(file) { // decode the clip's audio, or, where that fails, play it silently and record it
  try { return await toMono16k(await file.arrayBuffer()); } catch { /* not decodable as audio here */ }
  const url = URL.createObjectURL(file), ctx = player.ctx;
  const v = document.createElement("video");
  Object.assign(v, { playsInline: true, src: url });
  const src = ctx.createMediaElementSource(v), proc = ctx.createScriptProcessor(4096, 1, 1), g = ctx.createGain();
  g.gain.value = 0;
  const got = [];
  proc.onaudioprocess = (e) => got.push(new Float32Array(e.inputBuffer.getChannelData(0)));
  src.connect(proc); proc.connect(g); g.connect(ctx.destination);
  try {
    await v.play();
    await new Promise((ok) => { v.onended = ok; setTimeout(ok, 31000); });
  } finally { proc.onaudioprocess = null; src.disconnect(); proc.disconnect(); g.disconnect(); URL.revokeObjectURL(url); }
  const x = new Float32Array(got.reduce((a, b) => a + b.length, 0));
  let o = 0;
  for (const b of got) { x.set(b, o); o += b.length; }
  return resample16k(x, ctx.sampleRate);
}
$("liveClip").onchange = async () => {
  const f = $("liveClip").files[0];
  $("liveClip").value = "";
  if (!f || busy) return;
  busy = true; stopFlag = false;
  let done;
  answering = new Promise((ok) => { done = ok; });
  $("liveFallback").hidden = true;
  $("orb").hidden = false;
  liveState("thinking", "Watching the clip…");
  try {
    let samples = await clipSound(f);
    if (samples.length > 30 * 16000) samples = samples.slice(0, 30 * 16000);
    const frame = VISION ? await clipFrame(f).catch(() => null) : null;
    liveState("thinking", "Thinking…");
    await clipTurn(samples, frame);
  } catch (e) { console.error(e); caption(`Could not use that video (${e.message}).`); }
  busy = false;
  answering = null;
  done();
  while (L.on && !busy && player.left() > 0.05) await sleep(100);
  if (L.on) { $("liveFallback").hidden = false; $("orb").hidden = true; liveState("fallback", "Record another, or end Live"); }
};
$("liveBtn").onclick = liveOpen;
$("hintLive").onclick = liveOpen;
$("hintPhoto").onclick = () => $("photo").click();
$("hintVoice").onclick = () => { PT.hold = false; talkStart(false); };   // starts listening; tap the button to send
$("liveEnd").onclick = liveClose;
$("liveCam").onclick = async () => { L.camOn = !L.camOn; syncLiveButtons(); await liveCamera(); };
$("liveFlip").onclick = async () => { L.facing = L.facing === "user" ? "environment" : "user"; await liveCamera(); };
document.addEventListener("keydown", (e) => { if (e.key === "Escape" && L.on) liveClose(); });

// ---- memory (System 3) ---------------------------------------------------------------------------------------
// The memory learns from each conversation by itself: when a new chat starts (the button, or turning Speak on or
// off), and, for a conversation that simply ended with the tab, at the next visit. The conversation so far is kept
// on the device as it goes ("chat/pending", with each spoken turn's 16 kHz audio), and turns the model's context no
// longer holds wait in `earlier`. A spoken turn is heard first: the speech encoder and the model's own speech
// recognition turn it into text, which the extraction reads (Chat.consolidate).
const ASR_LANG = { en: "English", de: "German", fr: "French", es: "Spanish", it: "Italian", pt: "Portuguese", ja: "Japanese", ko: "Korean",
  zh: "Chinese", nl: "Dutch", pl: "Polish", sv: "Swedish", da: "Danish", fi: "Finnish", el: "Greek", he: "Hebrew", tr: "Turkish",
  ar: "Arabic" }[(navigator.language || "en").slice(0, 2).toLowerCase()] || "English";
let earlier = [];   // this conversation's turns that the model's context no longer holds, still to be learned
const learnable = (msgs) => msgs.filter((x) => x.role === "user" && ((x.content || "").trim() || x.audio)).length;
async function savePending() { // after each turn: the conversation so far, kept on the device
  const msgs = [...earlier, ...chat.messages];
  let keep = 24;   // the last 24 spoken turns keep their audio (about 2 MB each at most)
  const rows = msgs.slice().reverse().map(({ role, content, media, audio }) => ({ role, content, media, audio: audio && keep-- > 0 ? audio : null })).reverse();
  try { await store.put("chat/pending", { at: new Date().toISOString(), messages: rows }); } catch { /* no storage: learned only at New chat */ }
}
async function dropPending() { try { await store.del("chat/pending"); } catch { /* none */ } }
async function transcribe(x) { // a spoken turn's words, by the model's own speech recognition
  const enc = await ensureAudio(() => {});
  const lm = logMel(x.audio);
  const o = await enc.encode(lm.mel, lm.frames);
  return chat.asr(o.emb, o.rows, ASR_LANG);
}
const told = (changes) => changes.map((c) => (c.type === "fact" ? `${c.key} = ${c.value}` : c.text)).join("; ");
// learn from a finished conversation, then start the next chat with the updated memory. report(text) shows progress.
async function learnFrom(msgs, report = () => {}) {
  if (!chat.useMemory || !learnable(msgs)) { await chat.newChat(); earlier = []; await dropPending(); return null; }
  const changes = await chat.consolidate((s) => {
    if (s.phase === "hear") report(`Listening back to what you said (${s.n + 1} of ${s.of})…`);
    else if (s.phase === "read") report(`Reading what you said (${s.n + 1} of ${s.of})${s.message ? `: “${s.message.slice(0, 48)}${s.message.length > 48 ? "…" : ""}”` : ""}`);
  }, { messages: msgs, transcribe: SPEECH ? transcribe : null });
  earlier = [];
  await dropPending();
  return changes;
}
// the user goes on while the model's context starts over: the turns it held are learned later
function contextFull() {
  earlier.push(...chat.messages);
  chat.restartFromPrefix();
  note("The conversation filled the model's context, so it starts over from here. What you said is still remembered when this chat ends.");
}
async function learnLastVisit() { // a conversation from the last visit that ended with the tab
  let pend = null;
  try { pend = await store.get("chat/pending"); } catch { /* no storage */ }
  if (!pend?.messages || !learnable(pend.messages)) return;
  busy = true;
  $("input").disabled = true;
  syncComposer();
  const n = note("Remembering your last conversation…");
  try {
    const changes = await learnFrom(pend.messages, (t) => { n.textContent = t; });
    n.textContent = changes?.length ? `From your last conversation, I remembered: ${told(changes)}.` : "Your last conversation had nothing new to remember.";
  } catch (e) { console.error(e); n.textContent = `Could not learn from your last conversation (${e.message}).`; await chat.newChat(); }
  busy = false;
  $("input").disabled = false;
  syncComposer();
}
function renderNotes() {
  const e = chat.experience;
  const box = $("notes");
  box.innerHTML = "";
  const add = (label, del) => {
    const d = document.createElement("div");
    d.className = "note";
    const s = document.createElement("span"); s.textContent = label;
    const b = document.createElement("button"); b.textContent = "Remove"; b.onclick = () => { del(); e.save(); renderNotes(); };
    d.append(s, b); box.append(d);
  };
  e.preferences.forEach((p, i) => add(`Instruction: ${p.text}`, () => e.preferences.splice(i, 1)));
  for (const [k, v] of Object.entries(e.facts)) add(`${k}: ${v.value}${v.history.length ? ` (was ${v.history.map((h) => h[0]).join(", ")})` : ""}`, () => delete e.facts[k]);
  e.lessons.forEach((p, i) => add(`Lesson: ${p.text}`, () => e.lessons.splice(i, 1)));
  if (e.empty()) box.innerHTML = `<p class="muted">Nothing yet. Talk or chat; when a chat ends, what you said is read for things worth remembering.</p>`;
}
$("memBtn").onclick = () => { renderNotes(); $("memMsg").textContent = ""; $("memory").classList.add("open"); };
$("memClose").onclick = () => $("memory").classList.remove("open");
$("useMem").onchange = () => { chat.useMemory = $("useMem").checked; };
// a new chat: the finished one is learned first, then the next starts with the updated memory
async function freshChat({ report = null } = {}) {
  if (busy || PT.state !== "idle") return;
  const msgs = [...earlier, ...chat.messages];
  busy = true;
  syncComposer();
  $("memory").classList.remove("open");
  $("msgs").innerHTML = "";
  const n = learnable(msgs) && chat.useMemory ? note("Remembering this conversation…") : null;
  const show = (t) => { if (n) n.textContent = t; report?.(t); };
  try {
    const changes = await learnFrom(msgs, show);
    if (n) n.textContent = changes?.length ? `Remembered: ${told(changes)}.` : "Nothing new to remember from the last conversation.";
    report?.(changes?.length ? `Learned ${changes.length}: ${told(changes)}` : "Nothing new to remember.");
  } catch (e) {
    console.error(e);
    if (n) n.textContent = `Could not learn from that conversation (${e.message}).`;
    await chat.newChat();
  }
  busy = false;
  syncComposer();
  syncHello();
}
$("learn").onclick = async () => {
  if (busy) return;
  if (!learnable([...earlier, ...chat.messages])) { $("memMsg").textContent = "Talk or chat first: memories are learned from what you say."; return; }
  await freshChat({ report: (t) => { $("memMsg").textContent = t; renderNotes(); } });
  renderNotes();
};
$("newChat").onclick = () => freshChat();
$("newChatTop").onclick = () => freshChat();
// ---- settings: the assistant's instructions (its base system prompt), kept on this device. They take effect
// from the next chat, so saving starts one (after learning from the current one).
const SYS_KEY = "lfm3.system", SYS_MAX = 2000;
function savedSystem() { try { return (localStorage.getItem(SYS_KEY) || "").slice(0, SYS_MAX); } catch { return ""; } }
function applySystem() { chat.system = savedSystem().trim() || SYSTEM; }
const sysCount = () => { $("sysCount").textContent = `${$("sysPrompt").value.length} / ${SYS_MAX}`; };
$("setBtn").onclick = () => {
  $("sysPrompt").value = savedSystem();
  sysCount();
  $("sysMsg").textContent = "";
  if (!smokeRun) { smokeLast ||= storedSmoke(); renderSmoke(smokeLast); }
  $("settings").classList.add("open");
};
$("setClose").onclick = () => $("settings").classList.remove("open");
$("sysPrompt").addEventListener("input", sysCount);
$("sysReset").onclick = () => { $("sysPrompt").value = ""; sysCount(); $("sysPrompt").focus(); };
$("sysSave").onclick = async () => {
  if (busy || PT.state !== "idle") { $("sysMsg").textContent = "Wait for the answer to finish, then save."; return; }
  const v = $("sysPrompt").value.trim().slice(0, SYS_MAX);
  try { if (v && v !== SYSTEM) localStorage.setItem(SYS_KEY, v); else localStorage.removeItem(SYS_KEY); }
  catch { $("sysMsg").textContent = "This browser could not keep them, so they last until the page is closed."; }
  chat.system = v || SYSTEM;
  $("settings").classList.remove("open");
  await freshChat();
  note(v && v !== SYSTEM ? "New chat with your instructions." : "New chat with the default instructions.");
};
$("copyMem").onclick = async () => {
  const text = JSON.stringify(chat.experience, null, 1);
  try { await navigator.clipboard.writeText(text); $("memMsg").textContent = "Copied."; }
  catch {
    $("memMsg").textContent = "Copying is blocked here. Select the text below:";
    const pre = document.createElement("pre"); pre.className = "url"; pre.textContent = text; $("memMsg").append(pre);
  }
};
let forgetArmed = false;
$("clearMem").onclick = async () => {
  if (!forgetArmed) {
    forgetArmed = true;
    $("clearMem").textContent = "Tap again to forget everything";
    setTimeout(() => { forgetArmed = false; $("clearMem").textContent = "Forget everything"; }, 4000);
    return;
  }
  forgetArmed = false;
  $("clearMem").textContent = "Forget everything";
  chat.experience = new Experience();
  chat.experience.save();
  earlier = [];
  await dropPending();
  renderNotes();
  await chat.newChat();
  $("memMsg").textContent = "Forgotten.";
};

// ---- settings: the smoke test (smoke.js). Every workflow once on this device, checked and measured; the results
// stay on the device unless the user copies, shares or saves them. The last run is kept to show next time.
const SMOKE_SHORT = new URLSearchParams(location.search).get("smoke") === "short";   // tests: smaller inputs
const SMOKE_KEY = "lfm3.smoke";
let smokeRun = null, smokeStop = false, smokeLast = null, loadStats = null;
const smokeHook = globalThis.LFM3_SMOKE = { running: false, done: 0, result: null };   // tests
const smokeStatus = (t) => { $("smokeMsg").textContent = t; };
function el(tag, cls, text) { const e = document.createElement(tag); e.className = cls; if (text != null) e.textContent = text; return e; }
function storedSmoke() { try { return JSON.parse(localStorage.getItem(SMOKE_KEY) || "null"); } catch { return null; } }
function renderSmoke(res) {
  const box = $("smokeOut");
  box.replaceChildren();
  const done = !!res && res.total != null;
  $("smokeCopy").hidden = $("smokeSave").hidden = !done;
  $("smokeShare").hidden = !done || !navigator.share;
  if (!res) return;
  const d = res.device;
  box.append(el("p", "smokehead", [d.gpu, d.os, d.browser, `${res.weights.format} weights`, res.ms ? `${Math.round(res.ms / 1000)} s` : "",
    done ? new Date(res.at).toLocaleString() : ""].filter(Boolean).join(" · ")));
  for (const r of res.rows) {
    const row = el("div", `srow ${r.state}`);
    row.append(el("span", "sname", r.name), el("span", "sres", r.state === "run" ? "running…" : r.result));
    if (r.numbers) row.append(el("div", "snum", r.numbers));
    box.append(row);
  }
  if (done && !smokeRun) smokeStatus(`${res.passed} of ${res.total} passed${res.short ? " (short inputs)" : ""}.`);
}
const how = async (loaded, ...sets) => (loaded ? "loaded" : (await Promise.all(sets.map((x) => x.saved()))).every((x) => x?.complete) ? "device" : "network");
$("smokeRun").onclick = async () => {
  if (smokeRun) { smokeStop = true; $("smokeRun").textContent = "Stopping…"; return; }
  if (!chat || busy || PT.state !== "idle" || L.on) { smokeStatus("Wait for the answer to finish, then run it."); return; }
  player ||= makePlayer();
  player.ctx.resume?.();   // in the tap, so the spoken replies can play
  smokeStop = false;
  smokeRun = true;
  $("smokeRun").textContent = "Stop";
  smokeStatus("Starting a new chat…");
  await freshChat();   // the chat so far is learned from first, as New chat does
  busy = true;
  $("input").disabled = true;
  syncComposer();
  smokeHook.running = true;
  const api = {
    device: gpu.device, model, chat, info: gpu.info, load: loadStats,
    weights: { lm: WEIGHTS?.id, vision: VISION?.id, audio: SPEECH?.id, talk: TALK?.id, codec: CODEC?.id },
    vision: VISION ? async () => ({ how: await how(vision, VIS), model: await ensureVision(smokeStatus) }) : null,
    audio: SPEECH ? async () => ({ how: await how(audioEnc, AUD), model: await ensureAudio(smokeStatus) }) : null,
    talk: TALK && CODEC ? async () => ({ how: await how(talk, TLK, CDC), model: await ensureTalk(smokeStatus) }) : null,
    play: { begin: () => player.begin(), play: (x) => player.play(x), end: () => player.end() },
  };
  let res = null;
  try {
    smokeRun = runSmoke(api, { short: SMOKE_SHORT, onRow: (_, r) => renderSmoke(r), onStatus: smokeStatus, shouldStop: () => smokeStop });
    res = await smokeRun;
    smokeLast = res;
    try { localStorage.setItem(SMOKE_KEY, JSON.stringify(res)); } catch { /* shown only */ }
  } catch (e) {
    console.error(e);
    smokeStatus(`The smoke test stopped: ${e.message}`);
  }
  smokeRun = null;
  try { await chat.newChat(); } catch (e) { console.error(e); }   // the user's own chat again: their instructions, memory, voice
  busy = false;
  $("input").disabled = false;
  syncComposer();
  $("smokeRun").textContent = "Run the smoke test";
  if (res) {
    renderSmoke(res);
    note(`Smoke test: ${res.passed} of ${res.total} passed${res.failed ? `, ${res.failed} failed` : ""}. The results are in Settings.`);
  }
  Object.assign(smokeHook, { running: false, result: res, text: res ? smokeText(res) : "" });
  smokeHook.done++;
};
$("smokeCopy").onclick = async () => {
  const text = smokeText(smokeLast);
  try { await navigator.clipboard.writeText(text); smokeStatus("Copied."); }
  catch {
    smokeStatus("Copying is blocked here. Select the text below:");
    $("smokeMsg").append(el("pre", "url", text));
  }
};
$("smokeShare").onclick = async () => {
  try { await navigator.share({ title: "LFM3-Omni-2B smoke test", text: smokeText(smokeLast) }); } catch { /* cancelled */ }
};
$("smokeSave").onclick = () => {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([JSON.stringify(smokeLast, null, 1)], { type: "application/json" }));
  a.download = `lfm3-smoke-${(smokeLast.at || "").slice(0, 10)}.json`;
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
};

boot();
