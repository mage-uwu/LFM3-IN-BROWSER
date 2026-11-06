// SPDX-License-Identifier: Apache-2.0. Copyright 2026 Toaster Club LLC.
// Voice Desk: two of LiquidAI's models on every turn (engine/pair.js). d1-Omni-2B reads the turn first (what you type,
// a photo, what you say, or the camera and your voice in Live) and answers the profile's questions; LFM3-Omni-2B takes
// the same turn and answers aloud, with d1's decision as a note in that turn. Spoken turns stream into LFM3 while you
// talk, as the chat's push to talk does; at the release d1 decides on the whole clip (and the camera frame) and its
// note closes the turn. Weights: weights.js and d1-weights.js, kept in this browser's storage under the same families
// as the chat and Decide pages; the image encoders load the first time a picture is needed.
import { initDevice, LFM3 } from "./engine/lfm3.js";
import { Chat } from "./engine/chat.js";
import { Decider } from "./engine/d1.js";
import { Pair, LIVE_PIXELS } from "./engine/pair.js";
import { Audio, MelStream } from "./engine/audio.js";
import { Vision } from "./engine/vision.js";
import { Codec, DepthDecoder } from "./engine/speech.js";
import { MicCapture, Resampler, resample16k } from "./engine/listen.js";
import { cachedTokenizer, weightSet } from "./engine/loader.js";
import { makePlayer } from "./engine/play.js";
import { PROFILES } from "./engine/profiles.js";   // the ready-made profiles (Live helper, Trades dispatch, ...)

const $ = (id) => document.getElementById(id);
const G = globalThis;
const SETS = [ // [the global, its storage family, a label]: the chat page's families for LFM3, Decide's for d1
  ["LFM3_WEIGHTS", "w/", "LFM3-Omni-2B"], ["D1_WEIGHTS", "d1w/", "d1-Omni-2B"],
  ["LFM3_AUDIO", "a/", "LFM3's speech encoder"], ["D1_AUDIO", "d1a/", "d1's speech encoder"],
  ["LFM3_TALK", "s/", "the speech decoder"], ["LFM3_CODEC", "c/", "the voice codec"],
];
const PICTURE_SETS = [["LFM3_VISION", "v/", "LFM3's image encoder"], ["D1_VISION", "d1v/", "d1's image encoder"]];
const MAX_SECONDS = 30, TICK = 320, MIN_S = 0.35;
let gpu = null, pair = null, player = null, busy = false, stop = false, photo = null;

// ---- profiles: d1's questions and the voice's instructions (engine/profiles.js) ------------------------------------
let profile = "Live helper", { questions, system } = PROFILES[profile];

function showProfile() {
  $("qlist").replaceChildren(...Object.entries(questions).map(([n, q]) => Object.assign(document.createElement("span"),
    { textContent: `${n} · ${q.type === "noul" ? "yes/no" : q.type === "score" ? "score" : q.multiple ? "any of" : "one of"}` })));
  $("qjson").value = JSON.stringify(questions, null, 2);
  $("sys").value = system;
}
for (const name of [...Object.keys(PROFILES), "Custom"]) $("profile").append(new Option(name, name));
$("profile").value = profile;
$("profile").onchange = async () => {
  if ($("profile").value === "Custom") { $("custom").open = true; return; }
  profile = $("profile").value;
  ({ questions, system } = PROFILES[profile]);
  showProfile();
  if (pair && !busy && !T) await fresh();
};
$("applyQ").onclick = async () => {
  try {
    const q = JSON.parse($("qjson").value);
    if (!q || typeof q !== "object" || Array.isArray(q) || !Object.keys(q).length) throw new Error("give an object of named questions");
    questions = q; system = $("sys").value.trim() || PROFILES["Live helper"].system;
    profile = "Custom"; $("profile").value = "Custom";
    showProfile();
    $("qmsg").textContent = "In use from the next message.";
    if (pair && !busy && !T) await fresh();
  } catch (e) { $("qmsg").textContent = `Not valid: ${e.message}`; }
};

// ---- loading ----------------------------------------------------------------------------------------------------
const status = (t, err = false) => { $("loadMsg").textContent = t; $("loadMsg").classList.toggle("err", err); };
const say = (t, err = false) => { $("status").textContent = t; $("status").classList.toggle("err", err); };
const classOf = (g) => (g.endsWith("WEIGHTS") ? LFM3 : g.endsWith("AUDIO") ? Audio : g.endsWith("VISION") ? Vision : g === "LFM3_TALK" ? DepthDecoder : Codec);

async function loadSets(sets, onProgress) { // weight sets into the GPU through the device's copy; returns { global: model }
  const total = sets.reduce((s, [g]) => s + G[g].total, 0), loaded = {};
  let before = 0, fromDevice = 0, parts = 0;
  for (const [g, fam, label] of sets) {
    const W = G[g], set = weightSet(W, fam), src = set.source();
    await set.prepare(src);
    const Cls = classOf(g);
    const m = Cls === LFM3 ? new LFM3(gpu.device, { ctx: 4096, prequant: src.prequant }) : new Cls(gpu.device, { quant: W.format, prequant: src.prequant });
    await m.load((p) => onProgress((before + p.got * (W.total / p.total)) / total,
      `loading ${label}${src.fromDevice ? " from this device" : ""}: ${((before + p.got * (W.total / p.total)) / 1e9).toFixed(2)} of ${(total / 1e9).toFixed(2)} GB`));
    await set.finish(src);
    before += W.total; fromDevice += src.fromDevice; parts += set.n;
    loaded[g] = m;
  }
  loaded.allFromDevice = fromDevice === parts;
  return loaded;
}
async function load() {
  $("loadBtn").disabled = true;
  $("prog").hidden = false;
  const t0 = performance.now();
  try {
    const m = await loadSets(SETS, (f, t) => { $("prog").value = f; status(t); });
    const tok = await cachedTokenizer();
    const calibration = await (await fetch("./d1/calibration.json")).json();
    const decider = new Decider(m.D1_WEIGHTS, tok, { calibration, setSize: calibration.set_size || 16, lead: "<think></think>" });
    pair = new Pair({ decider, d1Audio: m.D1_AUDIO, chat: new Chat(m.LFM3_WEIGHTS, tok), audio: m.LFM3_AUDIO, depth: m.LFM3_TALK, codec: m.LFM3_CODEC });
    await fresh();
    $("setup").hidden = true;
    for (const b of ["talk", "plusBtn", "sampleBtn", "newBtn", "liveBtn", "input"]) $(b).disabled = false;
    $("input").placeholder = "Message, or tap the mic";
    say(`Ready in ${((performance.now() - t0) / 1000).toFixed(0)} s${m.allFromDevice ? ", from this device" : ""}.`);
  } catch (e) {
    console.error(e);
    const oom = /memory|allocat|lost/i.test(String(e.message));
    status(`${e.message}${oom ? ". This device may not have room for both models; the chat and Decide pages each need one." : ""}`, true);
    $("loadBtn").disabled = false;
  }
}
$("loadBtn").onclick = load;
let picturesLoading = null;
async function ensurePictures() { // the two image encoders, the first time a photo or Live needs them
  if (pair.vision && pair.d1Vision) return;
  picturesLoading ||= (async () => {
    const m = await loadSets(PICTURE_SETS, (f, t) => say(t));
    pair.vision = m.LFM3_VISION; pair.d1Vision = m.D1_VISION;
    say("");
  })().finally(() => { picturesLoading = null; });
  return picturesLoading;
}

async function fresh() { // a new conversation
  await pair.newChat(system);
  $("turns").replaceChildren();
  $("hello").hidden = L.on;
}
$("newBtn").onclick = async () => { if (!busy && !T) { await fresh(); say("New conversation."); } };

// ---- the composer: the round button sends what is typed (or the photo), or talks -----------------------------------
let T = null;   // the turn being spoken: { cap, rs, mel, frame, opened, feeding, done }
const hasDraft = () => !!($("input").value.trim() || photo);
function syncTalk() {
  $("talk").classList.toggle("has", hasDraft() && !T);
  $("talk").classList.toggle("busy", busy);
  $("talk").classList.toggle("rec", !!T);
  $("talk").setAttribute("aria-label", busy ? "Stop" : T ? "Send what you said" : hasDraft() ? "Send" : "Talk");
}
$("input").oninput = () => { $("input").style.height = "auto"; $("input").style.height = `${Math.min($("input").scrollHeight, 136)}px`; syncTalk(); };
$("input").onkeydown = (e) => { if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); if (hasDraft() && !busy) $("talk").click(); } };
$("talk").onclick = async () => {
  if (busy) { stop = true; player?.stop(); return; }   // a tap while answering stops the answer
  if (T) { talkEnd(); return; }
  if (hasDraft()) { await sendTyped(); return; }
  talkStart();
};

// a photo from the + button (an audio file goes in as a spoken turn)
async function pictureFrom(file, name, max = 1024) { // an image file -> RGBA at most `max` px on a side
  const bmp = await createImageBitmap(file);
  const s = Math.min(1, max / Math.max(bmp.width, bmp.height));
  const w = Math.max(32, Math.round(bmp.width * s)), h = Math.max(32, Math.round(bmp.height * s));
  const cv = document.createElement("canvas");
  cv.width = w; cv.height = h;
  const cx = cv.getContext("2d");
  cx.imageSmoothingQuality = "high";
  cx.drawImage(bmp, 0, 0, w, h);
  bmp.close?.();
  return { rgba: cx.getImageData(0, 0, w, h).data, w, h, canvas: cv, name };
}
function showAttach() {
  $("attach").hidden = !photo;
  if (photo) {
    const cv = $("attachPic"), cx = cv.getContext("2d"), s = Math.max(40 / photo.w, 40 / photo.h);
    cx.clearRect(0, 0, 40, 40);
    cx.drawImage(photo.canvas, (40 - photo.w * s) / 2, (40 - photo.h * s) / 2, photo.w * s, photo.h * s);
    $("attachName").textContent = photo.name;
  }
  syncTalk();
}
$("plusBtn").onclick = () => $("file").click();
$("file").onchange = async () => {
  const f = $("file").files[0];
  $("file").value = "";
  if (!f) return;
  try {
    if (f.type.startsWith("audio/")) { await answer({ samples: await toMono16k(await f.arrayBuffer()) }, f.name || "Audio file"); return; }
    photo = await pictureFrom(f, f.name || "Photo");
    showAttach();
    ensurePictures().catch((e) => say(`The image encoders could not load (${e.message}).`, true));
  } catch (e) { say(`Could not read that file (${e.message}).`, true); }
};
$("attachDrop").onclick = () => { photo = null; showAttach(); };

async function sendTyped() {
  const text = $("input").value.trim(), pic = photo;
  $("input").value = ""; $("input").style.height = "auto";
  photo = null; showAttach();
  await answer({ text, image: pic }, null);
}

// ---- a whole turn (typed text, a photo, an audio file, the sample) ------------------------------------------------
const pc = (p) => `${Math.round(100 * p)}%`;
function youBubble({ text, image, seconds, label }) {
  const you = Object.assign(document.createElement("div"), { className: "you" });
  if (image?.canvas) {
    const t = document.createElement("canvas"), s = Math.min(1, 160 / Math.max(image.w, image.h));
    t.width = Math.round(image.w * s); t.height = Math.round(image.h * s);
    t.getContext("2d").drawImage(image.canvas, 0, 0, t.width, t.height);
    you.append(t);
  }
  if (text) you.append(Object.assign(document.createElement("span"), { textContent: text }));
  if (seconds) you.append(Object.assign(document.createElement("small"), { textContent: `${label || "You said"} · ${seconds.toFixed(1)} s` }));
  return you;
}
function decisionCard(card, decision, note) {
  card.classList.remove("waiting");
  card.querySelector(".hd span").textContent = `${decision.usage.input_tokens} tokens`;
  for (const [name, a] of Object.entries(decision.answers)) {
    const q = questions[name] || {};
    let verdict, p;
    if (a.type === "noul") { verdict = a.noul >= 0.5 ? "yes" : "no"; p = a.noul >= 0.5 ? a.noul : 1 - a.noul; }
    else if (a.choices) { verdict = a.choices.join(", ") || "none"; p = Math.max(...Object.values(a.probabilities)); }
    else if (a.type === "choice") { verdict = a.choice.replace(/_/g, " "); p = a.confidence; }
    else { verdict = q.criteria?.[a.score] ?? String(a.score); p = a.confidence; }
    const row = Object.assign(document.createElement("div"), { className: "row" });
    row.innerHTML = '<span class="k"></span><span class="v"><span></span><span class="track"><span class="fill"></span></span></span><span class="p"></span>';
    row.querySelector(".k").textContent = name.replace(/_/g, " ");
    row.querySelector(".v span").textContent = verdict;
    row.querySelector(".p").textContent = pc(p);
    card.append(row);
    requestAnimationFrame(() => { row.querySelector(".fill").style.width = `${Math.max(2, 100 * p)}%`; });
  }
  const det = document.createElement("details");
  det.innerHTML = '<summary>The note LFM3 was given</summary><div class="note"></div>';
  det.querySelector(".note").textContent = note;
  card.append(det);
}
function turnView(you) { // the turn's three parts: what you gave, d1's card, LFM3's answer
  $("hello").hidden = true;
  const t = Object.assign(document.createElement("div"), { className: "turn" });
  const card = Object.assign(document.createElement("div"), { className: "card waiting" });
  card.innerHTML = '<div class="hd"><b>d1 decided</b><span>reading</span></div>';
  const reply = Object.assign(document.createElement("div"), { className: "reply waiting", textContent: "…" });
  t.append(you, card, reply);
  $("turns").append(t);
  t.scrollIntoView({ behavior: "smooth", block: "end" });
  return { t, card, reply };
}
function handlers(v, t0) {
  let first = 0;
  return {
    onDecision: (d, note) => { decisionCard(v.card, d, note); say("LFM3 is answering"); },
    onText: (x) => { v.reply.textContent = x; v.reply.classList.remove("waiting"); },
    onAudio: (x) => { if (!first) first = performance.now() - t0; player.play(x); say("Speaking · tap ■ to stop"); },
    shouldStop: () => stop,
    first: () => first,
  };
}
function finish(v, r, extra = "") {
  player.end();
  v.reply.textContent = r.reply;
  v.reply.classList.remove("waiting");
  const st = Object.assign(document.createElement("div"), { className: "stats" });
  st.textContent = `${extra}decided in ${(r.decideMs / 1000).toFixed(1)} s · ${r.seconds.toFixed(1)} s of speech in ${(r.stats.ms / 1000).toFixed(1)} s · context ${r.stats.pos}`;
  v.t.append(st);
  G.VOICE_LAST = r;   // tests
}
async function room() { // keep the conversation inside the model's context
  if (pair.chat.model.pos + 1500 > pair.chat.model.ctx) { await pair.newChat(system); say("The conversation was long, so it started over."); }
}
async function answer(input, label) {
  if (busy || T || !pair) return;
  busy = true; stop = false; syncTalk();
  player ||= makePlayer();
  player.ctx.resume?.();
  const v = turnView(youBubble({ text: input.text, image: input.image, seconds: input.samples ? input.samples.length / 16000 : 0, label }));
  try {
    await room();
    if (input.image) { say("Loading the image encoders"); await ensurePictures(); }
    say("d1 is deciding");
    player.begin();
    const r = await pair.turn(input, questions, handlers(v, performance.now()));
    finish(v, r, r.imageRows ? `picture ${r.imageRows} tokens · ` : "");
    say("");
  } catch (e) {
    console.error(e);
    v.reply.textContent = `Something went wrong: ${e.message}`;
    v.reply.classList.add("err");
    say("");
  } finally { busy = false; syncTalk(); }
}

// ---- a spoken turn, streamed into LFM3 while you talk (with the camera frame in Live) ----------------------------
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
async function talkStart() {
  if (busy || T || !pair) return;
  player ||= makePlayer();
  player.ctx.resume?.();   // in the tap: iOS starts audio only from a user's gesture
  let stream;
  try {
    try { if (navigator.audioSession) navigator.audioSession.type = "play-and-record"; } catch { /* older Safari */ }
    stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true } });
  } catch (e) { say(`The microphone is not available (${e.name || e.message}). Type instead, or add an audio file with +.`, true); return; }
  const frame = L.on ? liveFrame() : null;
  const t = T = { rs: new Resampler(player.ctx.sampleRate, 16000), mel: new MelStream(MAX_SECONDS), frame, done: false };
  t.cap = new MicCapture(player.ctx, stream, { onSamples: (x) => { if (t.done) return; t.mel.push(t.rs.push(x)); if (t.mel.full) talkEnd(); } });
  t.cap.start();
  t.limit = setTimeout(() => talkEnd(), (MAX_SECONDS + 0.5) * 1000);
  t.tick = setInterval(() => say(`Listening ${t.mel.seconds.toFixed(0)} s${frame ? " · with the camera" : ""} · tap to send`), 250);
  syncTalk();
  t.opened = (async () => { // LFM3's side, while you talk: the frame, then your speech as it becomes final
    if (frame) await ensurePictures();
    await room();
    while (!t.done && t.mel.seconds < MIN_S) await sleep(40);   // a stray tap opens no turn
    if (t.mel.seconds < MIN_S) return false;
    await pair.liveOpen(frame);
    return true;
  })();
  t.feeding = (async () => {
    if (!(await t.opened)) return false;
    while (!t.done) { await pair.liveFeed(t.mel); await sleep(TICK); }
    return true;
  })();
  t.feeding.catch(() => {});
}
async function talkEnd() {
  const t = T;
  if (!t || t.done) return;
  t.done = true;
  clearTimeout(t.limit); clearInterval(t.tick);
  t.cap.stop();
  try { if (navigator.audioSession) navigator.audioSession.type = "playback"; } catch { /* older Safari */ }
  t.mel.push(t.rs.finish());
  t.mel.finish();
  const released = performance.now(), secs = t.mel.seconds;
  let ok = false;
  try { ok = await t.feeding; } catch (e) {
    console.error(e);
    pair.liveCancel();
    T = null; syncTalk();
    say(`Could not take that in (${e.message}). The conversation starts over from here.`, true);
    return;
  }
  T = null;
  if (!ok) { syncTalk(); say("That was too short. Tap the mic, speak, then tap again."); return; }
  busy = true; stop = false; syncTalk();
  const v = turnView(youBubble({ image: t.frame, seconds: secs, label: t.frame ? "Live" : "You said" }));
  try {
    say("d1 is deciding");
    player.begin();
    const h = handlers(v, released);
    const r = await pair.liveClose(t.mel, questions, h);
    finish(v, r, `${r.imageRows ? `camera ${r.imageRows} tokens · ` : ""}${r.streamed} speech tokens taken in while you spoke · voice ${h.first() ? (h.first() / 1000).toFixed(1) : "–"} s after you let go · `);
    say("");
  } catch (e) {
    console.error(e);
    v.reply.textContent = `Something went wrong: ${e.message}`;
    v.reply.classList.add("err");
    say("");
  } finally { busy = false; syncTalk(); }
}

// ---- Live: the camera above the conversation ---------------------------------------------------------------------
const L = { on: false, facing: "environment", stream: null, lock: null };
function liveFrame() { // the camera's picture now, at about the size the model reads it
  const v = $("liveVideo");
  if (!L.on || !v.videoWidth) return null;
  const s = Math.min(1, Math.sqrt((LIVE_PIXELS * 1.3) / (v.videoWidth * v.videoHeight)));
  const cv = document.createElement("canvas");
  cv.width = Math.max(32, Math.round(v.videoWidth * s)); cv.height = Math.max(32, Math.round(v.videoHeight * s));
  const cx = cv.getContext("2d");
  cx.drawImage(v, 0, 0, cv.width, cv.height);
  return { rgba: cx.getImageData(0, 0, cv.width, cv.height).data, w: cv.width, h: cv.height, canvas: cv, name: "Camera" };
}
async function camera() {
  L.stream?.getTracks().forEach((t) => t.stop());
  L.stream = null;
  if (!L.on) return;
  try {
    L.stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: L.facing, width: { ideal: 640 }, height: { ideal: 480 } } });
    const v = $("liveVideo");
    v.srcObject = L.stream;
    v.classList.toggle("front", L.facing === "user");
    await v.play().catch(() => {});
  } catch (e) {
    say(`The camera is not available (${e.name || e.message}).`, true);
    liveOff();
  }
}
async function liveOn() {
  L.on = true;
  $("live").hidden = false;
  $("liveBtn").classList.add("on");
  $("hello").hidden = true;
  player ||= makePlayer();
  player.ctx.resume?.();
  try { L.lock = await navigator.wakeLock?.request("screen"); } catch { /* the screen may sleep */ }
  await camera();
  if (!L.on) return;
  say("Loading the image encoders");
  try { await ensurePictures(); say("Live · tap the mic and talk about what you see"); }
  catch (e) { say(`The image encoders could not load (${e.message}).`, true); }
}
function liveOff() {
  L.on = false;
  L.stream?.getTracks().forEach((t) => t.stop());
  L.stream = null;
  $("liveVideo").srcObject = null;
  $("live").hidden = true;
  $("liveBtn").classList.remove("on");
  L.lock?.release?.().catch(() => {});
  L.lock = null;
}
$("liveBtn").onclick = () => (L.on ? liveOff() : liveOn());
$("liveClose").onclick = liveOff;
$("flipBtn").onclick = async () => { L.facing = L.facing === "user" ? "environment" : "user"; await camera(); };

// ---- a sample message for the profile, spoken by LFM3-Omni-2B's own voice (it starts a new conversation) -------------
async function toMono16k(data) {
  const AC = G.AudioContext || G.webkitAudioContext;
  const ctx = new AC();
  const decoded = await ctx.decodeAudioData(data);
  ctx.close?.();
  const n = Math.max(1, Math.ceil(Math.min(decoded.duration, MAX_SECONDS) * 16000));
  const off = new OfflineAudioContext(1, n, 16000);
  const src = off.createBufferSource();
  src.buffer = decoded; src.connect(off.destination); src.start();
  return (await off.startRendering()).getChannelData(0).slice();
}
$("sampleBtn").onclick = async () => {
  if (busy || T || !pair) return;
  busy = true; syncTalk();
  player ||= makePlayer();
  player.ctx.resume?.();
  const line = PROFILES[profile]?.sample || PROFILES["Live helper"].sample;
  const frame = L.on ? liveFrame() : null;
  try {
    say("Making a sample message with the model's voice");
    const chat = pair.chat;
    chat.spoken = true;
    chat.system = "You are a voice actor. Say exactly the line you are given, word for word, and nothing else.";
    await chat.newChat();
    const parts = [];
    await chat.speak(`Say this line: "${line}"`, { depth: pair.depth, codec: pair.codec, maxFrames: 300, onAudio: (x) => parts.push(x) });
    const x24 = new Float32Array(parts.reduce((s, p) => s + p.length, 0));
    let o = 0;
    for (const p of parts) { x24.set(p, o); o += p.length; }
    await fresh();
    busy = false;
    await answer({ samples: resample16k(x24, 24000).slice(0, MAX_SECONDS * 16000), image: frame }, `Sample: "${line}"`);
  } catch (e) {
    console.error(e);
    say(`The sample failed (${e.message}).`, true);
  } finally { busy = false; syncTalk(); }
};

// ---- start ------------------------------------------------------------------------------------------------------
showProfile();
syncTalk();
(async () => {
  const missing = [...SETS, ...PICTURE_SETS].filter(([g]) => !G[g]).map(([g]) => g);
  if (missing.length) { status(`This build lacks ${missing.join(", ")}.`, true); return; }
  try {
    gpu = await initDevice();
    const i = gpu.info;
    $("chip").textContent = [i.vendor, i.architecture || i.description].filter(Boolean).join(" ") || "WebGPU ready";
    $("loadBtn").disabled = false;
    let saved = 0, n = 0;
    for (const [g, fam] of SETS) { const s = await weightSet(G[g], fam).saved(); saved += s?.complete ? G[g].total : 0; n += G[g].total; }
    if (saved) $("loadBtn").textContent = saved >= n ? "Load both models (saved on this device)" : `Load both models (${Math.round((100 * saved) / n)}% saved)`;
  } catch (e) {
    $("chip").textContent = "no WebGPU";
    status(`WebGPU is not available here (${e.message}).`, true);
  }
})();
G.VOICE_PAGE = { answer, get pair() { return pair; }, load, PROFILES, ensurePictures };   // tests
