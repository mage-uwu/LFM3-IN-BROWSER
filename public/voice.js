// SPDX-License-Identifier: Apache-2.0. Copyright 2026 Toaster Club LLC.
// Voice Desk: two of LiquidAI's models on one spoken turn (engine/pair.js). d1-Omni-2B hears the message and answers
// the profile's questions; LFM3-Omni-2B hears it too and answers aloud, with d1's decision as a note in that turn.
// Weights: weights.js (LFM3-Omni-2B, its speech encoder and spoken-reply models) and d1-weights.js (d1's language
// model and speech encoder), kept in this browser's storage under the same families as the chat and Decide pages.
import { initDevice, LFM3 } from "./engine/lfm3.js";
import { Chat } from "./engine/chat.js";
import { Decider } from "./engine/d1.js";
import { Pair } from "./engine/pair.js";
import { Audio } from "./engine/audio.js";
import { Codec, DepthDecoder } from "./engine/speech.js";
import { resample16k } from "./engine/listen.js";
import { cachedTokenizer, weightSet } from "./engine/loader.js";
import { makePlayer } from "./engine/play.js";

const $ = (id) => document.getElementById(id);
const G = globalThis;
const SETS = [ // [the global, its storage family, a label]: the chat page's families for LFM3, Decide's for d1
  ["LFM3_WEIGHTS", "w/", "LFM3-Omni-2B"], ["D1_WEIGHTS", "d1w/", "d1-Omni-2B"],
  ["LFM3_AUDIO", "a/", "LFM3's speech encoder"], ["D1_AUDIO", "d1a/", "d1's speech encoder"],
  ["LFM3_TALK", "s/", "the speech decoder"], ["LFM3_CODEC", "c/", "the voice codec"],
];
const MAX_SECONDS = 30;
let gpu = null, pair = null, player = null, busy = false, rec = null, stop = false;

// ---- profiles: d1's questions and the voice's instructions -------------------------------------------------------
const PROFILES = {
  "Customer line": {
    system: "You are the voice of a shop's customer line. Keep answers to two or three short spoken sentences. Act on the notes: if escalate is yes, tell the caller you are passing them to a human agent now; if urgent is yes, say you are treating it as a priority; if the mood is Upset, apologize first.",
    questions: {
      intent: { type: "choice", instructions: "What does the speaker want?", criteria: { refund: "Money back for a charge", order_status: "Where an order is", question: "An answer to a question", complaint: "To report a problem", smalltalk: "Just chatting" } },
      urgent: { type: "noul", instructions: "Does the speaker say it is urgent?" },
      mood: { type: "score", instructions: "How upset is the speaker?", criteria: ["Calm", "Annoyed", "Upset"] },
      escalate: { type: "noul", instructions: "Should this go to a human agent?" },
    },
    sample: "I was charged twice for my order and the parcel never arrived. Refund one of the charges, this is urgent!",
  },
  "Personal assistant": {
    system: "You are a warm personal assistant who talks. Keep answers short and natural, like speech. Act on the notes: if details is yes, ask for the one thing you most need before helping; if the mood is Stressed, be brief and reassuring.",
    questions: {
      intent: { type: "choice", instructions: "What does the speaker want?", criteria: { question: "An answer", task: "Something done", plan: "Help planning", chat: "Just talking" } },
      details: { type: "noul", instructions: "Is something important missing from the request?" },
      mood: { type: "score", instructions: "How is the speaker feeling?", criteria: ["Relaxed", "Busy", "Stressed"] },
    },
    sample: "Can you help me plan dinner for six people tomorrow? One of them is vegetarian and I have no time.",
  },
  "Language tutor": {
    system: "You are a friendly English tutor. Answer what the learner says, and if the notes say they made mistakes, gently show a better way to say it.",
    questions: {
      mistakes: { type: "noul", instructions: "Does the speaker make grammar mistakes?" },
      level: { type: "score", instructions: "How fluent is the speaker's English?", criteria: ["Beginner", "Intermediate", "Advanced"] },
      topic: { type: "choice", instructions: "What is the speaker talking about?", criteria: { daily_life: null, work: null, travel: null, food: null, other: null } },
    },
    sample: "Yesterday I go to the market and I buyed many fruits, but the apples was not so good.",
  },
};
let profile = "Customer line", questions = PROFILES[profile].questions, system = PROFILES[profile].system;

function showProfile() {
  $("qlist").replaceChildren(...Object.entries(questions).map(([n, q]) => Object.assign(document.createElement("span"),
    { textContent: `${n} · ${q.type === "noul" ? "yes/no" : q.type === "score" ? "score" : q.multiple ? "any of" : "one of"}` })));
  $("qjson").value = JSON.stringify(questions, null, 2);
  $("sys").value = system;
}
for (const name of [...Object.keys(PROFILES), "Custom"]) $("profile").append(new Option(name, name));
$("profile").onchange = async () => {
  if ($("profile").value === "Custom") { $("custom").open = true; return; }
  profile = $("profile").value;
  ({ questions, system } = PROFILES[profile]);
  showProfile();
  if (pair && !busy) await fresh();
};
$("applyQ").onclick = async () => {
  try {
    const q = JSON.parse($("qjson").value);
    if (!q || typeof q !== "object" || Array.isArray(q) || !Object.keys(q).length) throw new Error("give an object of named questions");
    questions = q; system = $("sys").value.trim() || PROFILES["Customer line"].system;
    profile = "Custom"; $("profile").value = "Custom";
    showProfile();
    $("qmsg").textContent = "In use from the next message.";
    if (pair && !busy) await fresh();
  } catch (e) { $("qmsg").textContent = `Not valid: ${e.message}`; }
};

// ---- loading ----------------------------------------------------------------------------------------------------
const status = (t, err = false) => { $("loadMsg").textContent = t; $("loadMsg").classList.toggle("err", err); };
const say = (t, err = false) => { $("status").textContent = t; $("status").classList.toggle("err", err); };

async function load() {
  $("loadBtn").disabled = true;
  $("prog").hidden = false;
  const t0 = performance.now();
  try {
    const total = SETS.reduce((s, [g]) => s + G[g].total, 0);
    let before = 0, fromDevice = 0, parts = 0;
    const loaded = {};
    for (const [g, fam, label] of SETS) {
      const W = G[g], set = weightSet(W, fam), src = set.source();
      await set.prepare(src);
      const Cls = g.endsWith("WEIGHTS") ? LFM3 : g.endsWith("AUDIO") ? Audio : g === "LFM3_TALK" ? DepthDecoder : Codec;
      const m = Cls === LFM3 ? new LFM3(gpu.device, { ctx: 4096, prequant: src.prequant }) : new Cls(gpu.device, { quant: W.format, prequant: src.prequant });
      await m.load((p) => {
        $("prog").value = (before + p.got * (W.total / p.total)) / total;
        status(`loading ${label}${src.fromDevice ? " from this device" : ""}: ${((before + p.got * (W.total / p.total)) / 1e9).toFixed(2)} of ${(total / 1e9).toFixed(2)} GB`);
      });
      await set.finish(src);
      before += W.total; fromDevice += src.fromDevice; parts += set.n;
      loaded[g] = m;
    }
    const tok = await cachedTokenizer();
    const calibration = await (await fetch("./d1/calibration.json")).json();
    const decider = new Decider(loaded.D1_WEIGHTS, tok, { calibration, setSize: calibration.set_size || 16, lead: "<think></think>" });
    pair = new Pair({ decider, d1Audio: loaded.D1_AUDIO, chat: new Chat(loaded.LFM3_WEIGHTS, tok), audio: loaded.LFM3_AUDIO,
      depth: loaded.LFM3_TALK, codec: loaded.LFM3_CODEC });
    await fresh();
    $("setup").hidden = true;
    for (const b of ["talk", "fileBtn", "sampleBtn", "newBtn"]) $(b).disabled = false;
    say(`Ready in ${((performance.now() - t0) / 1000).toFixed(0)} s${fromDevice === parts ? ", from this device" : ""}. Tap to talk.`);
  } catch (e) {
    console.error(e);
    const oom = /memory|allocat|lost/i.test(String(e.message));
    status(`${e.message}${oom ? ". This device may not have room for both models; the chat and Decide pages each need one." : ""}`, true);
    $("loadBtn").disabled = false;
  }
}
$("loadBtn").onclick = load;

async function fresh() { // a new conversation
  await pair.newChat(system);
  $("turns").replaceChildren();
  $("hello").hidden = false;
}
$("newBtn").onclick = async () => { if (!busy) { await fresh(); say("New conversation."); } };

// ---- a turn -----------------------------------------------------------------------------------------------------
const pc = (p) => `${Math.round(100 * p)}%`;
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

async function turn(samples, label) {
  if (busy || !pair) return;
  busy = true; stop = false;
  $("talk").classList.add("busy");
  $("hello").hidden = true;
  player ||= makePlayer();
  player.ctx.resume?.();
  const t = Object.assign(document.createElement("div"), { className: "turn" });
  const you = Object.assign(document.createElement("div"), { className: "you" });
  you.textContent = `${label} · ${(samples.length / 16000).toFixed(1)} s`;
  const card = Object.assign(document.createElement("div"), { className: "card waiting" });
  card.innerHTML = '<div class="hd"><b>d1 decided</b><span>listening</span></div>';
  const reply = Object.assign(document.createElement("div"), { className: "reply waiting", textContent: "…" });
  t.append(you, card, reply);
  $("turns").append(t);
  t.scrollIntoView({ behavior: "smooth", block: "end" });
  try {
    const chat = pair.chat;
    if (chat.model.pos + 1200 > chat.model.ctx) { await pair.newChat(system); say("The conversation was long, so it started over."); }
    say("d1 is deciding");
    player.begin();
    const r = await pair.turn(samples, questions, {
      onDecision: (d, note) => { decisionCard(card, d, note); say("LFM3 is answering"); },
      onText: (x) => { reply.textContent = x; reply.classList.remove("waiting"); },
      onAudio: (x) => { player.play(x); say("Speaking"); },
      shouldStop: () => stop,
    });
    player.end();
    reply.textContent = r.reply;
    reply.classList.remove("waiting");
    const st = Object.assign(document.createElement("div"), { className: "stats" });
    st.textContent = `decided in ${(r.decideMs / 1000).toFixed(1)} s · ${r.seconds.toFixed(1)} s of speech in ${(r.stats.ms / 1000).toFixed(1)} s · context ${r.stats.pos}`;
    t.append(st);
    say("Tap to talk.");
    G.VOICE_LAST = r;   // tests
  } catch (e) {
    console.error(e);
    reply.textContent = `Something went wrong: ${e.message}`;
    reply.classList.add("err");
    say("");
  } finally {
    busy = false;
    $("talk").classList.remove("busy");
  }
}

// ---- input: the microphone, a file, or a sample spoken by the model -------------------------------------------
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
$("talk").onclick = async () => {
  if (busy) { stop = true; player?.stop(); return; }   // a tap while answering stops the answer
  if (rec) { rec.stop(); return; }
  player ||= makePlayer();
  player.ctx.resume?.();
  let stream;
  try { stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } }); }
  catch (e) { say(`The microphone is not available (${e.message}).`, true); return; }
  const parts = [], t0 = performance.now();
  rec = new MediaRecorder(stream);
  rec.ondataavailable = (e) => { if (e.data.size) parts.push(e.data); };
  const tick = setInterval(() => say(`Listening ${((performance.now() - t0) / 1000).toFixed(0)} s · tap to send`), 250);
  const limit = setTimeout(() => rec?.stop(), MAX_SECONDS * 1000);
  rec.onstop = async () => {
    clearInterval(tick); clearTimeout(limit);
    stream.getTracks().forEach((t) => t.stop());
    rec = null;
    $("talk").classList.remove("rec");
    try {
      const samples = await toMono16k(await new Blob(parts, { type: parts[0]?.type || "audio/webm" }).arrayBuffer());
      if (samples.length < 8000) { say("That was too short. Tap, speak, then tap again."); return; }
      await turn(samples, "You said");
    } catch (e) { say(`Could not read the recording (${e.message}).`, true); }
  };
  rec.start(250);
  $("talk").classList.add("rec");
};
$("fileBtn").onclick = () => $("file").click();
$("file").onchange = async () => {
  const f = $("file").files[0];
  $("file").value = "";
  if (!f) return;
  try { await turn(await toMono16k(await f.arrayBuffer()), f.name || "Audio file"); }
  catch (e) { say(`Could not read that file (${e.message}).`, true); }
};
// a sample message for the profile, spoken by LFM3-Omni-2B's own voice (it starts a new conversation)
$("sampleBtn").onclick = async () => {
  if (busy || !pair) return;
  busy = true;
  player ||= makePlayer();
  player.ctx.resume?.();
  const line = PROFILES[profile]?.sample || PROFILES["Customer line"].sample;
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
    await turn(resample16k(x24, 24000).slice(0, MAX_SECONDS * 16000), `Sample: "${line}"`);
  } catch (e) {
    console.error(e);
    say(`The sample failed (${e.message}).`, true);
  } finally { busy = false; }
};

// ---- start ------------------------------------------------------------------------------------------------------
showProfile();
(async () => {
  const missing = SETS.filter(([g]) => !G[g]).map(([g]) => g);
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
G.VOICE_PAGE = { turn, get pair() { return pair; }, load, PROFILES };   // tests
