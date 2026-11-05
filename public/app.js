import { initDevice, LFM3 } from "./engine/lfm3.js";
import { Chat, Experience, loadTokenizer } from "./engine/chat.js";
import { pngPayload, prefetched } from "./engine/pngbytes.js";
import { persist, room, store } from "./engine/store.js";
import { Tokenizer } from "./engine/tokenizer.js";
import { preprocess, Vision } from "./engine/vision.js";
import { Audio, logMel } from "./engine/audio.js";
import { Codec, DepthDecoder } from "./engine/speech.js";
import { buf } from "./engine/lfm3.js";

const $ = (id) => document.getElementById(id);
// inside a claude.ai artifact the page cannot reach other hosts: the weights come from a picked file
const ARTIFACT = !!globalThis.LFM3_ARTIFACT;
// the artifact build carries the model itself: GPTQ 4-bit weights in the artifact's own asset store, as PNG byte
// containers (tools/gptq_export.py, tools/png_chunks.py) that the page streams into the GPU
const WEIGHTS = globalThis.LFM3_WEIGHTS || null;
let gpu, model, chat, busy = false, stopFlag = false, file = null;

async function fetchBytes(c) {
  for (let a = 0; ; a++) {
    try {
      const r = await fetch(c.url);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return new Uint8Array(await r.arrayBuffer());
    } catch (e) {
      if (a >= 3) throw new Error(`could not fetch the model's weights (${e.message})`);
      await new Promise((res) => setTimeout(res, 1000 * 2 ** a));
    }
  }
}
// ---- weights on this device (IndexedDB): each decoded part of a weight set is kept as it streams in, under
// <family>/<id>/<index>; <family>/<id>/complete marks a full copy. Later loads read the parts back instead of
// downloading, and an interrupted first load resumes with the parts it already kept. Families: w/ the language
// model, v/ the image model.
function weightSet(W, family) {
  const prefix = `${family}${W.id}/`, n = W.chunks.length;
  return {
    W, prefix, n,
    async saved() { // parts on this device and whether the copy is complete (null: no storage)
      try {
        const keys = (await store.keys()).map(String).filter((k) => k.startsWith(prefix));
        return { n: keys.filter((k) => !k.endsWith("complete")).length, complete: keys.includes(prefix + "complete") };
      } catch { return null; }
    },
    source() {
      const src = { fromDevice: 0, saved: 0, saving: true, saveError: null };
      const loadOne = async (c, i) => {
        try {
          const ab = await store.get(prefix + i);
          if (ab && ab.byteLength === c.bytes) { src.fromDevice++; return new Uint8Array(ab); }
        } catch { /* not on this device: download it */ }
        const data = await pngPayload(await fetchBytes(c), c.bytes);
        if (src.saving) {
          try { await store.put(prefix + i, data.buffer); src.saved++; } catch (e) { src.saving = false; src.saveError = e; }
        }
        return data;
      };
      src.prequant = { id: W.id, manifest: W.tensors, chunks: () => prefetched(W.chunks, loadOne) };
      return src;
    },
    async prepare(src) { // keep a copy only if the browser has storage and room for the parts still missing
      const s = await this.saved();
      if (s?.complete) return s;
      await persist();
      const free = await room(), need = W.total * (1 - (s?.n || 0) / n);
      if (s === null) { src.saving = false; src.saveError = new Error("storage is not available here"); }
      else if (free !== null && free < need * 1.05) { src.saving = false; src.saveError = new Error("not enough room"); }
      return s;
    },
    async finish(src) { // mark a full copy, or drop a partial one that cannot be finished
      try {
        if (src.fromDevice + src.saved === n) {
          await store.put(prefix + "complete", { total: W.total, at: new Date().toISOString() });
          await store.delWhere((k) => k.startsWith(family) && !k.startsWith(prefix));  // older versions
          return "saved";
        }
        if (src.saveError) await store.delWhere((k) => k.startsWith(prefix));
      } catch { /* storage unavailable */ }
      return src.saveError ? "failed" : "partial";
    },
  };
}
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
// the tokenizer, kept on the device beside the weights
async function tokenizerFromStore() {
  const KEY = "tok/lfm3-v1";
  try {
    const t = await store.get(KEY);
    if (t) return Tokenizer.fromParts(JSON.parse(t.vocab), t.merges);
  } catch { /* fetch it */ }
  const base = new URL("./", document.baseURI);
  const get = async (n) => { const r = await fetch(new URL(n, base)); if (!r.ok) throw new Error(`tokenizer: HTTP ${r.status}`); return r.text(); };
  const [vocab, merges] = await Promise.all([get("lfm3-tokenizer-vocab.json"), get("lfm3-tokenizer-merges.txt")]);
  try { await store.put(KEY, { vocab, merges }); } catch { /* keep going without */ }
  return Tokenizer.fromParts(JSON.parse(vocab), merges);
}

function bubble(role, text) {
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  div.textContent = text;
  $("msgs").append(div);
  $("main").scrollTop = $("main").scrollHeight;
  return div;
}
const status = (html) => { $("loadMsg").innerHTML = html; };

async function boot() {
  $("hubBtn").hidden = ARTIFACT || !!WEIGHTS;   // a page that carries its weights does not need the bf16 original
  $("photoBtn").hidden = !VISION;
  $("micBtn").hidden = !SPEECH;
  $("speakBtn").hidden = !(TALK && CODEC);
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
    const tokP = WEIGHTS ? tokenizerFromStore() : loadTokenizer();
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
    const info = await chat.newChat();
    $("setup").hidden = true;
    $("input").disabled = false; $("send").disabled = false; $("memBtn").disabled = false;
    $("photoBtn").disabled = !VISION;
    $("micBtn").disabled = !SPEECH;
    $("speakBtn").disabled = !(TALK && CODEC);
    $("input").placeholder = "Message";
    const s = ((performance.now() - t0) / 1000).toFixed(0);
    $("chip").textContent += ` · ${model.quant === "q8" ? "8-bit" : "4-bit"}`;
    const mem = chat.experience.empty() ? "" : ` Memory loaded: ${Object.keys(chat.experience.facts).length} facts (${info.tokens} tokens in ${info.ms.toFixed(0)} ms).`;
    const kept = !src ? (model.cached === false ? " This browser had no room to save a copy, so next time choose the file again." : "")
      : src.fromDevice === WEIGHTS.chunks.length ? ""
      : saved === "saved" ? " A copy is now saved on this device, so the next load skips the download."
      : ` This browser could not keep a copy (${src.saveError?.message || "storage unavailable"}), so the next load downloads it again.`;
    const local = model.fromCache || (src && src.fromDevice === WEIGHTS.chunks.length);
    bubble("assistant", `Ready in ${s} s${local ? ", from this device" : ""}.${kept}${mem}`);
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
    bubble("assistant", `Could not read that picture (${e.message}).`);
  }
};
$("attachDrop").onclick = () => { photo = null; $("attach").hidden = true; };

// ---- speech input: a recording goes through the speech encoder (loaded the first time) into the prompt ---------
let audioEnc = null, audioLoading = null, voice = null, rec = null;
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
}
async function attachVoice(blob) {
  try {
    let samples = await toMono16k(await blob.arrayBuffer());
    if (samples.length > 30 * 16000) samples = samples.slice(0, 30 * 16000);   // the encoder takes up to 30 s
    voice = { samples, seconds: samples.length / 16000 };
  } catch (e) {
    voice = null;
    bubble("assistant", `Could not read that recording (${e.message}).`);
  }
  showVoice();
}
$("micBtn").onclick = async () => {
  if (rec) { rec.stop(); return; }
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (e) { // the page's frame may not be allowed the microphone: offer a recording from the device instead
    $("voiceFileRow").hidden = false;
    $("voiceErr").textContent = `The microphone is not available here (${e.name || e.message}).`;
    return;
  }
  const parts = [];
  const mr = new MediaRecorder(stream);
  mr.ondataavailable = (e) => { if (e.data.size) parts.push(e.data); };
  mr.onstop = async () => {
    stream.getTracks().forEach((t) => t.stop());
    clearTimeout(mr.limit);
    rec = null;
    $("micBtn").textContent = "Voice";
    $("micBtn").classList.remove("rec");
    await attachVoice(new Blob(parts, { type: mr.mimeType }));
  };
  mr.start();
  mr.limit = setTimeout(() => mr.state === "recording" && mr.stop(), 30000);
  rec = mr;
  $("micBtn").textContent = "Stop";
  $("micBtn").classList.add("rec");
};
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
function makePlayer() { // 24 kHz chunks queued back to back (created in a tap, as iOS requires)
  const AC = globalThis.AudioContext || globalThis.webkitAudioContext;
  const ctx = new AC();
  let next = 0;
  const live = new Set();
  return {
    ctx,
    play(x) {
      globalThis.LFM3_PLAYED = (globalThis.LFM3_PLAYED || 0) + x.length;   // samples handed to the speaker (tests)
      const b = ctx.createBuffer(1, x.length, 24000);
      b.getChannelData(0).set(x);
      const src = ctx.createBufferSource();
      src.buffer = b;
      src.connect(ctx.destination);
      next = Math.max(next, ctx.currentTime + 0.08);
      src.start(next);
      next += b.duration;
      live.add(src);
      src.onended = () => live.delete(src);
    },
    stop() { for (const s of live) { try { s.stop(); } catch { /* ended */ } } live.clear(); next = 0; },
  };
}
$("speakBtn").onclick = async () => {
  const on = !chat.spoken;
  if (on) { player ||= makePlayer(); player.ctx.resume?.(); }
  $("speakBtn").setAttribute("aria-pressed", String(on));
  $("speakBtn").classList.toggle("on", on);
  chat.spoken = on;
  const note = bubble("assistant", on ? "Spoken replies on. Starting a new chat…" : "Spoken replies off. Starting a new chat…");
  if (on) {
    try { await ensureTalk((m) => { note.textContent = m; }); } catch (e) {
      chat.spoken = false; $("speakBtn").setAttribute("aria-pressed", "false"); $("speakBtn").classList.remove("on");
      note.innerHTML = `<span class="err">${e.message}</span>`;
      return;
    }
  }
  await chat.newChat();
  note.textContent = on ? "Spoken replies on: the answers are read aloud as they are written. New chat started." : "Spoken replies off. New chat started.";
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
  if (busy) { stopFlag = true; player?.stop(); return; }
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
  $("send").textContent = "Stop";
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
      const res = await chat.speak(text || (vc ? "" : "Describe this image."), { media, depth: t.depth, codec: t.codec, onText,
        onAudio: (a) => player.play(a), shouldStop: () => stopFlag });
      out.textContent = res.reply;
      const s = res.stats;
      line = `${seen}${s.promptTokens} prompt tokens  |  ${res.seconds.toFixed(1)} s of speech in ${(s.ms / 1000).toFixed(1)} s ` +
        `(${r(s.frames / (s.ms / 1000))} frames/s, 12.5 is real time)  |  context ${s.pos}`;
    } else {
      const { reply, stats } = await chat.send(text || (vc ? "" : "Describe this image."), { media, onText, shouldStop: () => stopFlag });
      out.textContent = reply;
      line = `${seen}${stats.promptTokens} prompt tokens · ${r(stats.prefillTokS)} tok/s  |  ${stats.newTokens} tokens · ${r(stats.decodeTokS)} tok/s  |  context ${stats.pos}`;
    }
    const st = document.createElement("div");
    st.className = "stats";
    st.textContent = line;
    out.append(st);
  } catch (e) {
    out.innerHTML = `<span class="err">${e.message}</span>`;
  }
  busy = false;
  $("send").textContent = "Send";
}
$("send").onclick = send;
$("input").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } });
$("input").addEventListener("input", () => { const t = $("input"); t.style.height = "auto"; t.style.height = t.scrollHeight + "px"; });

// ---- memory (System 3) ---------------------------------------------------------------------------------------
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
  if (e.empty()) box.innerHTML = `<p class="muted">Nothing yet. Chat, then tap “Learn from this chat”.</p>`;
}
$("memBtn").onclick = () => { renderNotes(); $("memMsg").textContent = ""; $("memory").classList.add("open"); };
$("memClose").onclick = () => $("memory").classList.remove("open");
$("useMem").onchange = () => { chat.useMemory = $("useMem").checked; };
$("learn").onclick = async () => {
  if (busy) return;
  if (!chat.messages.length) { $("memMsg").textContent = "Chat first: memories are learned from what you write."; return; }
  busy = true;
  $("memMsg").textContent = "Reading your messages…";
  const changes = await chat.consolidate((s) => { $("memMsg").textContent = `Read: “${s.message.slice(0, 60)}”`; });
  busy = false;
  $("memMsg").textContent = changes.length ? `Learned ${changes.length}: ` + changes.map((c) => c.type === "fact" ? `${c.key} = ${c.value}` : c.text).join("; ") : "Nothing new to remember.";
  renderNotes();
  $("msgs").innerHTML = "";
  bubble("assistant", "New chat, with what I learned from the last one.");
};
$("newChat").onclick = async () => {
  if (busy) return;
  await chat.newChat();
  $("msgs").innerHTML = "";
  $("memory").classList.remove("open");
  bubble("assistant", "New chat.");
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
  renderNotes();
  await chat.newChat();
  $("memMsg").textContent = "Forgotten.";
};

boot();
