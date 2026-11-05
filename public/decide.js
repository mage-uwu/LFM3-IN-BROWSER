// SPDX-License-Identifier: Apache-2.0. Copyright 2026 Toaster Club LLC.
// The decision page: LiquidAI's d1-Omni-2B on this device. What the model looks at (text, JSON, a photo or a
// recording) and named questions go through engine/d1.js, which answers them all in one forward pass with
// calibrated probabilities. The weights come from d1-weights.js (the language model, and the image and speech
// encoders, each loaded the first time it is needed) and are kept in this browser's storage like the chat's.
import { initDevice, LFM3 } from "./engine/lfm3.js";
import { Decider } from "./engine/d1.js";
import { cachedTokenizer, weightSet } from "./engine/loader.js";
import { store } from "./engine/store.js";
import { preprocess, Vision } from "./engine/vision.js";
import { Audio, logMel } from "./engine/audio.js";

const $ = (id) => document.getElementById(id);
const W = globalThis.D1_WEIGHTS, WV = globalThis.D1_VISION, WA = globalThis.D1_AUDIO;
const LM = weightSet(W, "d1w/"), VIS = WV ? weightSet(WV, "d1v/") : null, AUD = WA ? weightSet(WA, "d1a/") : null;
const SAVED = /^d1[wva]\//;   // this page's families in the browser's storage (the chat's are w/ v/ a/ s/ c/)
const MAX_SECONDS = 30;       // the speech encoder takes up to 30 s
let gpu = null, d1 = null, vision = null, audio = null, busy = false, last = null;

// ---- questions --------------------------------------------------------------------------------------------------
// a question as the page edits it: { name, type: noul | choice | multiple | score, text, opts } (opts: one option
// per line, "name: description" for a choice, the levels lowest first for a score)
const TYPES = [["noul", "Yes / no"], ["choice", "One of"], ["multiple", "Any of"], ["score", "Score"]];
let qs = [];

const EXAMPLES = [
  { label: "Support ticket", state: "I was charged twice and my parcel never arrived. Please refund one charge.", media: null, qs: [
    { name: "refund", type: "noul", text: "Is the customer asking for a refund?" },
    { name: "team", type: "choice", text: "Which team should handle this?", opts: "billing: Charges, refunds, invoices\nshipping: Delivery and parcels\ntechnical: App or site faults" },
    { name: "topics", type: "multiple", text: "Which topics does the message raise?", opts: "billing: Payments and charges\ndelivery: Shipping and parcels\nlogin: Signing in to the account" },
    { name: "anger", type: "score", text: "How upset is the customer?", opts: "Calm\nAnnoyed\nAngry" },
  ] },
  { label: "Order (JSON)", state: JSON.stringify({ order: { id: 4411, items: 3, paid: true }, message: "Where is my order? It is late." }, null, 2), media: null, qs: [
    { name: "late", type: "noul", text: "Does the customer say the order is late?" },
    { name: "paid", type: "noul", text: "Has the order been paid?" },
  ] },
  { label: "Product review", state: "The battery easily lasts two days, but the screen scratched in the first week. For the price I would still buy it again.", media: null, qs: [
    { name: "sentiment", type: "score", text: "How positive is the review overall?", opts: "Negative\nMixed\nPositive" },
    { name: "mentions", type: "multiple", text: "What does the review talk about?", opts: "battery\nscreen\nprice\nshipping\ncamera" },
    { name: "recommend", type: "noul", text: "Would the reviewer recommend the product?" },
  ] },
  { label: "Photo (sample)", state: "", media: "circle", qs: [
    { name: "red", type: "noul", text: "Is there a red circle in the picture?" },
    { name: "shape", type: "choice", text: "What shape is shown?", opts: "circle\nsquare\ntriangle" },
    { name: "people", type: "noul", text: "Are there any people in the picture?" },
  ] },
  { label: "Recording (sample)", state: "", media: "beeps", qs: [
    { name: "speech", type: "noul", text: "Does someone speak in the recording?" },
    { name: "kind", type: "choice", text: "What is in the recording?", opts: "tones: Beeps or tones\nspeech: Someone talking\nmusic: Music\nsilence: Nothing" },
  ] },
];

function slug(s) { return String(s).trim().replace(/\s+/g, "_").replace(/[^\p{L}\p{N}_-]/gu, "").slice(0, 40); }
function lines(s) { return String(s || "").split("\n").map((x) => x.trim()).filter(Boolean); }

// the page's questions as engine/d1.js takes them (throws with a message the page shows)
function toQuestions() {
  const out = {};
  for (const q of qs) {
    const name = slug(q.name);
    if (!name) throw new Error("every question needs a name");
    if (out[name]) throw new Error(`two questions are named "${name}"`);
    if (!q.text.trim()) throw new Error(`"${name}" has no question`);
    if (q.type === "noul") out[name] = { type: "noul", instructions: q.text.trim() };
    else if (q.type === "score") {
      const levels = lines(q.opts);
      if (levels.length < 2 || levels.length > 10) throw new Error(`"${name}" needs 2 to 10 levels, one per line, lowest first`);
      out[name] = { type: "score", instructions: q.text.trim(), criteria: levels };
    } else {
      const criteria = {};
      for (const l of lines(q.opts)) {
        const i = l.indexOf(":");
        const key = slug(i > 0 ? l.slice(0, i) : l), desc = i > 0 ? l.slice(i + 1).trim() : "";
        if (key) criteria[key] = desc || null;
      }
      if (Object.keys(criteria).length < 2) throw new Error(`"${name}" needs at least two options, one per line`);
      out[name] = { type: "choice", instructions: q.text.trim(), criteria, ...(q.type === "multiple" ? { multiple: true } : {}) };
    }
  }
  if (!Object.keys(out).length) throw new Error("add a question");
  return out;
}

function renderQs() {
  const box = $("qs");
  box.replaceChildren();
  qs.forEach((q, i) => {
    const card = document.createElement("div");
    card.className = "q";
    const top = document.createElement("div");
    top.className = "top";
    const name = Object.assign(document.createElement("input"), { className: "name", value: q.name, placeholder: "name", spellcheck: false, autocapitalize: "off" });
    name.setAttribute("aria-label", "Question name");
    name.oninput = () => { q.name = name.value; clearAnswer(i); syncCode(); };
    const drop = Object.assign(document.createElement("button"), { className: "icon", title: "Remove this question" });
    drop.setAttribute("aria-label", "Remove this question");
    drop.innerHTML = '<svg viewBox="0 0 24 24"><path d="M6 6l12 12M18 6 6 18"/></svg>';
    drop.onclick = () => { qs.splice(i, 1); renderQs(); syncCode(); };
    top.append(name, drop);
    const types = document.createElement("div");
    types.className = "types";
    for (const [t, label] of TYPES) {
      const b = Object.assign(document.createElement("button"), { textContent: label, className: q.type === t ? "on" : "" });
      b.onclick = () => { q.type = t; renderQs(); syncCode(); };
      types.append(b);
    }
    const text = Object.assign(document.createElement("input"), { value: q.text, placeholder: "The question, e.g. Is the customer asking for a refund?" });
    text.setAttribute("aria-label", "Question");
    text.oninput = () => { q.text = text.value; clearAnswer(i); syncCode(); };
    card.append(top, types, text);
    if (q.type !== "noul") {
      const lab = Object.assign(document.createElement("div"), { className: "opts-label",
        textContent: q.type === "score" ? "Levels, lowest first, one per line" : "Options, one per line (name: description)" });
      const opts = Object.assign(document.createElement("textarea"), { className: "opts", value: q.opts || "", rows: Math.max(2, lines(q.opts).length) });
      opts.placeholder = q.type === "score" ? "Calm\nAnnoyed\nAngry" : "billing: Charges and refunds\nshipping: Delivery";
      opts.oninput = () => { q.opts = opts.value; clearAnswer(i); syncCode(); };
      card.append(lab, opts);
    }
    const ans = Object.assign(document.createElement("div"), { className: "ans" });
    ans.hidden = true;
    card.append(ans);
    q.el = ans;
    box.append(card);
  });
  $("addQ").disabled = qs.length >= 32;
}
function clearAnswer(i) { if (qs[i]?.el) qs[i].el.hidden = true; }

// ---- what the model looks at --------------------------------------------------------------------------------------
let media = null;   // { kind: "photo", rgba, w, h, name } | { kind: "audio", samples, name }; encoded: { items, emb }

function stateValue() {
  const t = $("state").value;
  if (!t.trim()) return null;
  const s = t.trim();
  if (s[0] === "{" || s[0] === "[") {
    try { return JSON.parse(s); } catch { /* text that starts with a bracket */ }
  }
  return t;
}
function syncKind() {
  const v = stateValue();
  const json = v !== null && typeof v !== "string";
  $("state").classList.toggle("json", json);
  $("kind").textContent = v === null ? "" : json ? "JSON" : `${t3(v.length)} characters`;
}
const t3 = (n) => n.toLocaleString("en-US");

function showMedia() {
  $("media").hidden = !media;
  if (!media) return;
  const pic = media.kind === "photo";
  $("mediaPic").hidden = !pic; $("mediaWave").hidden = pic;
  $("mediaName").textContent = media.name;
  if (pic) {
    const cv = $("mediaPic"), cx = cv.getContext("2d");
    cv.width = 112; cv.height = 112;
    const tmp = new OffscreenCanvas(media.w, media.h);
    tmp.getContext("2d").putImageData(new ImageData(media.rgba, media.w, media.h), 0, 0);
    const s = Math.max(112 / media.w, 112 / media.h);
    cx.drawImage(tmp, (112 - media.w * s) / 2, (112 - media.h * s) / 2, media.w * s, media.h * s);
    $("mediaInfo").textContent = `${media.w} x ${media.h} px, read at up to 448 x 448`;
  } else {
    $("mediaInfo").textContent = `${(media.samples.length / 16000).toFixed(1)} s of audio`;
  }
}
function setMedia(m) { media = m; showMedia(); for (const q of qs) if (q.el) q.el.hidden = true; syncCode(); }

async function pictureFrom(source, name) { // any image -> RGBA at most 1024 px on a side (the encoder reads 448x448 worth)
  const bmp = source instanceof ImageBitmap ? source : await createImageBitmap(source);
  const s = Math.min(1, 1024 / Math.max(bmp.width, bmp.height));
  const w = Math.max(32, Math.round(bmp.width * s)), h = Math.max(32, Math.round(bmp.height * s));
  const cv = new OffscreenCanvas(w, h), cx = cv.getContext("2d");
  cx.imageSmoothingQuality = "high";
  cx.drawImage(bmp, 0, 0, w, h);
  bmp.close?.();
  return { kind: "photo", rgba: cx.getImageData(0, 0, w, h).data, w, h, name };
}
async function toMono16k(data) { // any recording the browser can decode -> 16 kHz mono, at most 30 s
  const AC = globalThis.AudioContext || globalThis.webkitAudioContext;
  const ctx = new AC();
  const decoded = await ctx.decodeAudioData(data);
  ctx.close?.();
  const n = Math.max(1, Math.ceil(Math.min(decoded.duration, MAX_SECONDS) * 16000));
  const off = new OfflineAudioContext(1, n, 16000);
  const src = off.createBufferSource();
  src.buffer = decoded;
  src.connect(off.destination);
  src.start();
  return (await off.startRendering()).getChannelData(0).slice();
}
// the release's own test media: a red circle on white, and three 880 Hz beeps in faint noise
function sampleCircle(size = 448) {
  const cv = new OffscreenCanvas(size, size), cx = cv.getContext("2d");
  cx.fillStyle = "#fff"; cx.fillRect(0, 0, size, size);
  cx.fillStyle = "rgb(224, 38, 43)";
  cx.beginPath(); cx.arc(size / 2, size / 2, size * 0.33, 0, 2 * Math.PI); cx.fill();
  return { kind: "photo", rgba: cx.getImageData(0, 0, size, size).data, w: size, h: size, name: "Sample photo: a red circle" };
}
function sampleBeeps(rate = 16000) {
  const x = new Float32Array(4 * rate);
  let seed = 1;
  const rnd = () => ((seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648);
  for (let i = 0; i < x.length; i++) x[i] = 0.003 * Math.sqrt(-2 * Math.log(rnd() || 1e-9)) * Math.cos(2 * Math.PI * rnd());
  for (const t of [0.5, 1.5, 2.5]) for (let i = 0; i < 3200; i++) x[Math.round(t * rate) + i] += 0.3 * Math.sin((2 * Math.PI * 880 * i) / rate);
  return { kind: "audio", samples: x, name: "Sample recording: three beeps" };
}

$("photoBtn").onclick = () => $("photo").click();
$("photo").onchange = async () => {
  const f = $("photo").files[0];
  $("photo").value = "";
  if (!f) return;
  try { setMedia(await pictureFrom(f, f.name || "Photo")); } catch (e) { say(`Could not read that picture (${e.message}).`, true); }
};
$("clipBtn").onclick = () => $("clip").click();
$("clip").onchange = async () => {
  const f = $("clip").files[0];
  $("clip").value = "";
  if (!f) return;
  try { setMedia({ kind: "audio", samples: await toMono16k(await f.arrayBuffer()), name: f.name || "Recording" }); }
  catch (e) { say(`Could not read that recording (${e.message}).`, true); }
};
$("mediaDrop").onclick = () => setMedia(null);

// recording from the microphone: tap to start, tap to stop (stops by itself after 30 s)
let rec = null;
$("recBtn").onclick = async () => {
  if (rec) { rec.stop(); return; }
  let stream;
  try { stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } }); }
  catch (e) { say(`The microphone is not available (${e.message}).`, true); return; }
  const parts = [], t0 = performance.now();
  rec = new MediaRecorder(stream);
  rec.ondataavailable = (e) => { if (e.data.size) parts.push(e.data); };
  const tick = setInterval(() => { $("recBtn").querySelector("span").textContent = `Stop ${((performance.now() - t0) / 1000).toFixed(0)} s`; }, 250);
  const limit = setTimeout(() => rec?.stop(), MAX_SECONDS * 1000);
  rec.onstop = async () => {
    clearInterval(tick); clearTimeout(limit);
    stream.getTracks().forEach((t) => t.stop());
    rec = null;
    $("recBtn").classList.remove("rec");
    $("recBtn").querySelector("span").textContent = "Record";
    try {
      const blob = new Blob(parts, { type: parts[0]?.type || "audio/webm" });
      setMedia({ kind: "audio", samples: await toMono16k(await blob.arrayBuffer()), name: "Recording" });
    } catch (e) { say(`Could not read the recording (${e.message}).`, true); }
  };
  rec.start(250);
  $("recBtn").classList.add("rec");
};

// ---- the model --------------------------------------------------------------------------------------------------
const status = (t, err = false) => { $("loadMsg").textContent = t; $("loadMsg").classList.toggle("err", err); };
const say = (t, err = false) => { $("runMsg").textContent = t; $("runMsg").classList.toggle("err", err); };
const mb = (n) => (n / 1e6).toFixed(0);

async function showSaved() {
  const s = await LM.saved();
  $("loadBtn").textContent = s?.complete ? "Load the model (saved on this device)"
    : s?.n ? `Continue loading the model (${Math.round((100 * s.n) / LM.n)}% saved)` : "Load the model";
}

async function load() {
  $("loadBtn").disabled = true;
  $("prog").hidden = false;
  const t0 = performance.now();
  try {
    const src = LM.source();
    await LM.prepare(src);
    const model = new LFM3(gpu.device, { ctx: 4096, prequant: src.prequant });
    const tokP = cachedTokenizer(), calP = fetch("./d1/calibration.json").then((r) => { if (!r.ok) throw new Error(`calibration: HTTP ${r.status}`); return r.json(); });
    await model.load((p) => {
      $("prog").value = p.got / p.total;
      status(`loading the model ${src.fromDevice ? "from this device" : `into the page${src.saving ? ", keeping a copy on this device" : ""}`}: ` +
        `${(p.got / 1e9).toFixed(2)} of ${(p.total / 1e9).toFixed(2)} GB`);
    });
    const kept = await LM.finish(src);
    const calibration = await calP;
    d1 = new Decider(model, await tokP, { calibration, setSize: calibration.set_size || 16, lead: "<think></think>" });
    $("setup").hidden = true;
    $("decide").disabled = false;
    const s = ((performance.now() - t0) / 1000).toFixed(0);
    say(`Ready in ${s} s${src.fromDevice === LM.n ? ", from this device" : kept === "saved" ? ". A copy is saved on this device" : ""}.`);
    showStorage();
  } catch (e) {
    console.error(e);
    status(e.message, true);
    $("loadBtn").disabled = false;
    await showSaved();
  }
}
$("loadBtn").onclick = load;

async function ensure(kind) { // the image or speech encoder, the first time it is needed
  const [set, W2, Cls, label] = kind === "photo" ? [VIS, WV, Vision, "image"] : [AUD, WA, Audio, "speech"];
  if (kind === "photo" ? vision : audio) return kind === "photo" ? vision : audio;
  if (!set) throw new Error(`this build has no ${label} model`);
  const src = set.source();
  await set.prepare(src);
  const m = new Cls(gpu.device, { quant: W2.format, prequant: src.prequant });
  await m.load((p) => say(`Loading the ${label} model${src.fromDevice ? " from this device" : ""}: ${mb(p.got)} of ${mb(p.total)} MB`));
  await set.finish(src);
  if (kind === "photo") vision = m; else audio = m;
  return m;
}
async function encodeMedia() {
  if (!media) return null;
  if (media.kind === "photo") {
    const v = await ensure("photo");
    const pp = preprocess(media.rgba, media.w, media.h);
    const o = await v.encode(pp.patches, pp.gh, pp.gw);
    return { items: [{ token: "<|image|>", rows: o.rows }], emb: o.emb };
  }
  const a = await ensure("audio");
  const lm = logMel(media.samples);
  const o = await a.encode(lm.mel, lm.frames);
  return { items: [{ token: "<|audio|>", rows: o.rows }], emb: o.emb };
}

// ---- answers ----------------------------------------------------------------------------------------------------
const pct = (p) => `${(100 * p).toFixed(p >= 0.995 || p < 0.005 ? 0 : 1)}%`;
function bars(el, rows, best) { // rows: [[label, p]] in the question's order; best: the label shown strongest
  for (const [label, p] of rows) {
    const r = Object.assign(document.createElement("div"), { className: "bar" });
    const top = best === null ? p >= 0.5 : label === best;
    r.innerHTML = `<span class="lab${top ? " top" : ""}"></span><span class="track"><span class="fill${top ? " top" : ""}"></span></span><span class="pct"></span>`;
    r.querySelector(".lab").textContent = label;
    r.querySelector(".lab").title = label;
    r.querySelector(".pct").textContent = pct(p);
    el.append(r);
    requestAnimationFrame(() => { r.querySelector(".fill").style.width = `${Math.max(1, 100 * p)}%`; });
  }
}
function render(q, a, spec) {
  const el = q.el;
  el.replaceChildren();
  el.hidden = false;
  const head = Object.assign(document.createElement("div"), { className: "head" });
  const verdict = Object.assign(document.createElement("span"), { className: "verdict" });
  const conf = Object.assign(document.createElement("span"), { className: "conf" });
  head.append(verdict, conf);
  el.append(head);
  if (a.type === "noul") {
    verdict.textContent = a.noul >= 0.5 ? "Yes" : "No";
    conf.textContent = `P(yes) ${a.noul.toFixed(3)}`;
    bars(el, [["yes", a.noul], ["no", 1 - a.noul]], a.noul >= 0.5 ? "yes" : "no");
  } else if (a.choices) {
    verdict.textContent = a.choices.length ? a.choices.join(", ") : "None of them";
    conf.textContent = "each option yes or no";
    bars(el, Object.entries(a.probabilities), null);
  } else if (a.type === "choice") {
    verdict.textContent = a.choice;
    conf.textContent = `confidence ${a.confidence.toFixed(3)}`;
    bars(el, Object.entries(a.probabilities), a.choice);
  } else {
    const levels = spec.criteria;
    verdict.textContent = levels[a.score];
    conf.textContent = `level ${a.score} of ${levels.length - 1}`;
    bars(el, levels.map((t, i) => [t, a.probabilities[String(i)]]), levels[a.score]);
  }
}

$("decide").onclick = async () => {
  if (busy || !d1) return;
  let questions;
  try { questions = toQuestions(); } catch (e) { say(e.message, true); return; }
  const state = stateValue();
  if (state === null && !media) { say("Give it something to look at: text, JSON, a photo or a recording.", true); return; }
  busy = true;
  $("decide").disabled = true;
  $("decide").textContent = "Deciding";
  say("");
  try {
    const t0 = performance.now();
    const m = await encodeMedia();
    const t1 = performance.now();
    say("Deciding");
    const { answers, usage } = await d1.ask(state, questions, { media: m });
    const t2 = performance.now();
    const names = Object.keys(questions);
    names.forEach((n, i) => render(qs[i], answers[n], questions[n]));
    last = { state, questions, answers };
    $("copyBtn").disabled = false;
    const passes = Math.ceil(names.length / (d1.setSize || 16));
    say(`${t3(usage.input_tokens)} tokens in ${Math.round(t2 - t1)} ms` + (m ? ` · ${media.kind === "photo" ? "image" : "audio"} encoded in ${Math.round(t1 - t0)} ms` : "") +
      ` · ${passes} pass${passes > 1 ? "es" : ""}, no text generated`);
    qs[0]?.el?.closest(".q")?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  } catch (e) {
    console.error(e);
    say(e.message, true);
  } finally {
    busy = false;
    $("decide").disabled = false;
    $("decide").textContent = "Decide";
  }
};
$("copyBtn").onclick = async () => {
  if (!last) return;
  const text = JSON.stringify(last.answers, null, 2);
  try { await navigator.clipboard.writeText(text); say("Answers copied as JSON."); }
  catch { say("Could not copy here."); }
};

// ---- the code view: the same call ------------------------------------------------------------------------------
function syncCode() {
  let q;
  try { q = toQuestions(); } catch { q = Object.fromEntries(qs.map((x) => [slug(x.name) || "name", { type: x.type === "multiple" ? "choice" : x.type, instructions: x.text }])); }
  const st = stateValue();
  const stateSrc = st === null ? "null" : typeof st === "string" ? JSON.stringify(st) : JSON.stringify(st, null, 2);
  $("code").textContent =
`import { Decider } from "./engine/d1.js";
// model: an LFM3 loaded with d1-weights.js; calibration: d1/calibration.json
const d1 = new Decider(model, tokenizer, { calibration, lead: "<think></think>" });
const { answers } = await d1.ask(${stateSrc}, ${JSON.stringify(q, null, 2)}${media ? `,
  { media })  // { items: [{ token: "${media.kind === "photo" ? "<|image|>" : "<|audio|>"}", rows }], emb } from the encoder` : ")"};`;
}

function useExample(ex) {
  $("state").value = ex.state;
  syncKind();
  qs = ex.qs.map((q) => ({ ...q, opts: q.opts || "" }));
  renderQs();
  setMedia(ex.media === "circle" ? sampleCircle() : ex.media === "beeps" ? sampleBeeps() : null);
  last = null;
  $("copyBtn").disabled = true;
  say("");
}
for (const ex of EXAMPLES) {
  const b = Object.assign(document.createElement("button"), { className: "chip", textContent: ex.label });
  b.onclick = () => useExample(ex);
  $("examples").append(b);
}
$("addQ").onclick = () => {
  qs.push({ name: `q${qs.length + 1}`, type: "noul", text: "", opts: "" });
  renderQs();
  syncCode();
  $("qs").lastElementChild?.querySelector("input:not(.name)")?.focus();
};
$("state").oninput = () => { syncKind(); for (const q of qs) if (q.el) q.el.hidden = true; syncCode(); };

// ---- storage ----------------------------------------------------------------------------------------------------
let dropArmed = false;
async function showStorage() {
  let n = 0;
  try { n = (await store.keys()).map(String).filter((k) => SAVED.test(k)).length; } catch { /* no storage */ }
  $("dropBtn").hidden = !n;
  $("dropInfo").textContent = n ? "A copy of d1's weights is kept in this browser's storage on this device (not in Files), so later visits skip the download. The chat's copy is separate."
    : "No copy of d1's weights is kept on this device. The next visit downloads them again.";
}
$("dropBtn").onclick = async () => {
  if (!dropArmed) { dropArmed = true; $("dropBtn").textContent = "Tap again to remove"; setTimeout(() => { dropArmed = false; $("dropBtn").textContent = "Remove the saved copy"; }, 4000); return; }
  dropArmed = false;
  try { await store.delWhere((k) => SAVED.test(String(k))); } catch { /* nothing kept */ }
  $("dropBtn").textContent = "Remove the saved copy";
  $("dropMsg").textContent = "Removed. The model keeps running until you close this page.";
  await showStorage();
};

addEventListener("scroll", () => $("top").classList.toggle("scrolled", scrollY > 4), { passive: true });

// ---- start ------------------------------------------------------------------------------------------------------
useExample(EXAMPLES[0]);
showStorage();
(async () => {
  if (!W) { status("This build carries no d1 weights.", true); return; }
  try {
    gpu = await initDevice();
    const i = gpu.info;
    $("chip").textContent = [i.vendor, i.architecture || i.description].filter(Boolean).join(" ") || "WebGPU ready";
    $("chip").title = i.description || $("chip").textContent;
    $("bits").hidden = false;
    $("loadBtn").disabled = false;
    await showSaved();
  } catch (e) {
    $("chip").textContent = "no WebGPU";
    status(`WebGPU is not available here (${e.message}). Try Safari 26, a current Chrome or Edge, or Firefox with WebGPU on.`, true);
  }
})();
globalThis.D1_PAGE = { useExample, EXAMPLES, get d1() { return d1; }, get last() { return last; }, load };   // tests
