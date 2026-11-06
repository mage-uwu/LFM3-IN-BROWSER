# Voice Desk handoff: two on-device models in your app

For an engineer (or an AI coding agent) adding Voice Desk to another product, such as a trades or field-service
SaaS. It covers what Voice Desk does, what to copy, the one call that loads it, how to design questions for your
use case, how to get decisions into your backend, and what to watch out for. Read [AGENTS.md](../AGENTS.md) for this
repository's rules and [API.md](API.md) for the full engine API.

Live demo: <https://lfm.toasterclub.com/voice>. Smallest working page: [`public/examples/voice-desk.html`](../public/examples/voice-desk.html)
(about 100 lines, the "Trades dispatch" profile).

## 1. What it is

Every turn (typed text, a photo, speech, or Live: the camera and a voice together) goes through two of Liquid AI's
models, both running on the user's GPU in the browser through WebGPU:

```
  customer ──► d1-Omni-2B (decides) ──► answers to your questions, as JSON, with probabilities
                     │                         │
                     │                         ├──► your backend (job ticket, triage, CRM)  ← you send this
                     ▼                         ▼
               LFM3-Omni-2B ◄── a one-line note of the decision ──► answers the customer aloud (and as text)
```

- **d1-Omni-2B** answers named questions in one pass, with calibrated probabilities and no generated text:
  yes/no (`noul`), one of several options (`choice`), any of several (`choice` + `multiple`), a level on a scale
  (`score`). It reads text, JSON, pictures and audio.
- **LFM3-Omni-2B** hears or reads the same turn and answers in speech and text. d1's decision goes into its turn as
  a note (`[Notes from the decision model on this message: urgency: Emergency (level 3 of 3); hazard: yes (97%) ...]`),
  and the profile's instructions tell it what to do with each answer.
- Nothing leaves the device unless your code sends it. No API key, no inference server, no per-call cost.

## 2. Requirements

| | |
| --- | --- |
| Browser | WebGPU: Safari 26 (iOS, iPadOS, macOS), Chrome or Edge 113+ (desktop, Android 121+), Firefox with WebGPU. HTTPS (or localhost) for the microphone and camera. |
| GPU memory | About 2.1 GB for both models with speech; 2.5 GB with pictures. Desktop GPUs and Apple-silicon Macs have room; on phones it needs a recent high-memory model (8 GB of RAM is a reasonable bar), and phones with less will likely fail to load. It has been measured on an NVIDIA A40, not yet on phones: test the devices your customers use. |
| First download | 2.2 GB (both language models, both speech encoders, the voice), and 0.43 GB more for the two image encoders the first time a picture is used. Kept in IndexedDB afterwards; later loads read it from the device. |
| Speed (NVIDIA A40) | d1's decision 0.6 to 2.4 s (more with a picture or a long clip); a whole turn with its spoken answer 2.7 to 5.2 s; in Live the voice starts about 1.6 s after the user lets go. Phones are slower; the page shows real numbers per turn. |

Have a fallback for devices without WebGPU or without the memory (a plain form, or your server's own pipeline).
`initDevice()` throws when WebGPU is missing; a load that runs out of memory throws an error whose message mentions
memory, allocation or a lost device.

## 3. Put the files in your app

The engine is plain ES modules with no dependencies and no build step. Copy these from `public/` into one folder
your app serves as static files (here `/lfm/`):

```
/lfm/engine/                    all of public/engine/*.js (about 260 KB)
/lfm/weights.js                 LFM3-Omni-2B's weight list
/lfm/d1-weights.js              d1-Omni-2B's weight list
/lfm/weights/<set-id>/          every set folder both lists name (8 sets, 141 PNG parts of at most 19 MB, 2.6 GB)
/lfm/weights/NOTICE.txt
/lfm/d1/calibration.json        d1's calibration (required)
/lfm/lfm3-tokenizer-vocab.json  the tokenizer (shared by both models)
/lfm/lfm3-tokenizer-merges.txt
/lfm/licenses/                  the licenses and NOTICE (required by the model license, see section 8)
```

- Serve `/lfm/weights/*` with `Cache-Control: public, max-age=31536000, immutable` (set folders are content
  hashes, so a new export gets a new folder). See `public/_headers`.
- The weights are PNG files on purpose: hosts that refuse large binaries accept images. Any static host or CDN
  works (Cloudflare Workers assets, S3 + CloudFront, Vercel, Netlify; note Cloudflare's 25 MiB per-file limit, which
  the parts already respect).
- To load the weights from another origin (a CDN), that origin must send `Access-Control-Allow-Origin` for your app.
  lfm.toasterclub.com does not, so host your own copy.
- Do not run the engine through a bundler's transforms. It loads `weights.js` and `d1-weights.js` with
  `import(new URL(...))` at run time; import the engine itself from the served folder:
  - Vite: `const { loadPair } = await import(/* @vite-ignore */ "/lfm/engine/loader.js");`
  - webpack / Next.js: `await import(/* webpackIgnore: true */ "/lfm/engine/loader.js")`, in client code only (it needs
    `navigator.gpu`).
- Inside an iframe, the frame needs `allow="microphone; camera"` for speech and Live.
- Verify the parts after copying: `python3 tools/pack_weights.py --site <your copy> --verify`.

## 4. Load it and run a turn

```js
const { loadPair } = await import("/lfm/engine/loader.js");
const { makePlayer } = await import("/lfm/engine/play.js");
const { PROFILES } = await import("/lfm/engine/profiles.js");
const profile = PROFILES["Trades dispatch"];        // or your own, see section 5

const player = makePlayer();                         // create it in a click or tap: iOS only plays audio after one
const desk = await loadPair({ base: "/lfm/", system: profile.system,
  onProgress: ({ got, total, label }) => showProgress(got / total, label) });

const r = await desk.pair.turn({ text: "Water is pouring out under my sink, can someone come today?" }, profile.questions, {
  onDecision: (decision, note) => saveTriage(decision.answers),   // d1's answers, before the voice starts
  onText: (t) => showReply(t),                                    // the spoken answer as text, as it is made
  onAudio: (chunk) => player.play(chunk),                         // 24 kHz audio, about every 320 ms
});
player.end();
```

Before each answer call `player.begin()` (and `player.ctx.resume()` in the user's tap). A turn's input is any of:

| Input | Shape | Notes |
| --- | --- | --- |
| `text` | string | d1 reads it as the state; the voice gets it, then the note |
| `image` | `{ rgba: Uint8ClampedArray, w, h }` | from a canvas `getImageData`; call `await desk.loadPictures()` once first |
| `samples` | `Float32Array`, 16 kHz mono, up to 30 s | a recording; `resample16k()` in `engine/listen.js` converts other rates |

Streaming a spoken turn while the user talks (lower latency, what `voice.html` does):

```js
await desk.pair.liveOpen(frameOrNull);               // at the press; frame: the camera picture, or null for voice only
// feed microphone audio: MicCapture -> Resampler(ctx.sampleRate, 16000) -> MelStream(30), and every ~320 ms:
await desk.pair.liveFeed(melStream);
// at the release: melStream.finish(), then
const r = await desk.pair.liveClose(melStream, profile.questions, { onDecision, onText, onAudio });
```

See `talkStart()` and `talkEnd()` in `public/voice.js` for the full microphone code. `pair.newChat(system)` starts a
new conversation; the context holds about 4,000 tokens, so start a new one when `desk.chat.model.pos` nears
`desk.chat.model.ctx` (the page does this at 1,500 tokens left).

What a turn returns:

```js
r.decision.answers   // { trade: { type: "choice", choice: "plumbing", confidence: 0.93, probabilities: {...} },
                     //   urgency: { type: "score", score: 2, confidence, probabilities: { "0": ..., "3": ... }, legend },
                     //   hazard: { type: "noul", noul: 0.88 }, ... }   noul is P(yes); a score's level indexes criteria
r.note               // the line the voice was given
r.reply              // what the voice said (text); r.seconds of speech; r.textIds; r.frames
r.decideMs           // d1's time, including the encoders
```

## 5. Design the questions for your product

A profile is `{ system, questions }` (see `public/engine/profiles.js`, "Trades dispatch").

**Questions (d1)**
- Use as few as decide what you need; up to 16 run in one pass (more run as extra passes).
- `choice` options take a short name and a concrete description: `{ plumbing: "Water, pipes, drains, toilets, water heaters" }`.
  The names become your field values, so use your system's own codes.
- `score` levels go lowest first: `["Routine", "This week", "Today", "Emergency"]`. The answer is the median level;
  `probabilities` gives the full spread.
- `noul` for yes/no flags: `hazard`, `needs_visit`, `damage_shown`. Ask about one thing per question.
- Questions about a picture only mean something when there is one. Mark them `needs: "image"`: they are asked only
  on turns with a picture, and are missing from `answers` otherwise (without a picture d1 answers them near 0.5).
- The ported types are `noul`, `choice` and `score`. The release's `number`, `date` and box types are not ported
  yet (they throw).

**Instructions (the voice)**
- The voice only acts on a note it has been told how to use. Name the answers and the action for each:
  "if hazard is yes, give the one safety step that matters most first ...". Without that it mostly ignores them.
- Keep spoken answers short ("two or three short spoken sentences"); long speech takes long to make on a phone.
- Say what it must never do ("Never promise a price or an arrival time").
- Measured (results/pair): with instructions that only said the notes exist, the voice did not act on
  "escalate: yes"; with an action per answer it said "I'm passing you to a human agent now", and a calm question
  given a note that said urgent made it say "I'm treating this as a priority".

**Thresholds in your code, not in the model**
- Treat d1's answers as triage signals with probabilities. Decide in your code what happens at each level, for
  example `hazard.noul >= 0.5` opens the emergency flow, and a `confidence` under 0.6 on `trade` asks a person to
  confirm. Log the probabilities with the ticket so you can tune these later.
- Safety-critical calls (a gas smell, electrical danger) should still reach a human and your normal emergency
  guidance. d1 is a 2B model; it is good at routing, not a guarantee.
- Escalate on the flag, not on the level: in the test below a gas smell got `hazard` 0.94 but `urgency` "Today",
  not "Emergency". And show your own safety text when `hazard` is yes: the voice gave the safety step for the gas
  smell but skipped it for a flooding leak (it went straight to booking).

**What the Trades dispatch profile did** (NVIDIA A40, `tools/node_desk_api_test.mjs`, greedy, results/desk):

| Turn | d1's decision | Spoken answer (start) |
| --- | --- | --- |
| Typed: water pouring from under the sink, spreading | plumbing, repair, Today, hazard 0.83, needs visit 0.74, Frustrated | "I'm prioritising your plumbing job today. I can book a visit ... address" |
| Typed: strong smell of gas near the boiler | hvac, repair, Today, hazard 0.94 | "... shut off the gas supply to the boiler immediately, and a technician is being sent as an emergency" |
| Spoken: a quote for two ceiling lights, no rush | electrical, quote, Routine, hazard 0.09, Calm | "... could you tell me the size of the room and the type of lights" |
| Photo of a pipe + typed: drips a little | plumbing, repair, This week, hazard 0.43 | "I'm prioritising the job. Please book a visit ..." |

d1 took 1.2 to 2.4 s per decision and whole turns 3.5 to 5.2 s; loading everything took 70 s from the network.

## 6. Getting decisions into your backend

The models run in the browser, so your backend only sees what you send. A typical flow for a trades intake:

```js
onDecision: async (d) => {
  const a = d.answers;
  await fetch("/api/intake", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({
    trade: a.trade.choice, request: a.request.choice,
    urgency: profile.questions.urgency.criteria[a.urgency.score],
    hazard: a.hazard.noul, needsVisit: a.needs_visit.noul, damageShown: a.damage_shown?.noul ?? null,   // photo turns only
    confidence: { trade: a.trade.confidence, urgency: a.urgency.confidence },
    transcript: typedTextOrNull,          // send only what the customer agreed to share
  }) });
}
```

- The decision is a few hundred bytes. Audio, photos and the spoken reply stay on the device unless you upload them.
- For a written record of a spoken turn, `desk.chat.asr(emb, rows)` transcribes it with the model's own speech
  recognition, but it resets the conversation, so run it between conversations, not mid-call.
- To give d1 context from your system (the customer's address on file, open jobs), pass it as `state`:
  `pair.turn(input, questions, { state: { open_jobs: 1, last_visit: "2026-10-02", message: text } })`. d1 reads JSON.

## 7. Pitfalls

- **Audio on iOS** plays only after a user gesture: create the player (or `player.ctx.resume()`) inside the tap.
  Set `navigator.audioSession.type = "play-and-record"` while listening, and `"playback"` after (as voice.js does),
  or the iPhone plays the answer through the earpiece.
- **Memory**: load once and keep the `desk` for the session; a second `loadPair()` on the same page doubles the
  memory. On failure tell the user the device lacks room, and offer your fallback.
- **First load** is 2.2 GB. Say so before the user taps Load, and prefer Wi-Fi. `weightSet(...).saved()` tells
  whether a copy is already on the device (voice.js shows "saved on this device").
- **Storage**: browsers may evict the copy under pressure; `persist()` asks to keep it. Clearing site data removes it.
- **One turn at a time**: do not start a turn while another is answering; pass `shouldStop: () => flag` to stop a
  reply early.
- **The camera frame** in Live is read at about 320x320 (100 image tokens, `LIVE_PIXELS`); photos at up to 448x448.
  Larger pictures are scaled down.

## 8. Licence terms you must keep

The weights and tokenizer are under the **LFM Open License v1.0** (`LICENSE-LFM.txt`); the code is Apache-2.0.

- **Commercial use is licensed only below US$10,000,000 in annual revenue** (Section 5). A SaaS over that needs a
  separate agreement with Liquid AI before shipping this.
- Ship the license and NOTICE with the weights (`/lfm/licenses/`, `weights/NOTICE.txt`) and keep the attribution.
  Modified weights must say they were changed.
- Liquid AI's names may only describe where the models come from; say your product is not affiliated with or
  endorsed by Liquid AI.
- Credit for this implementation: Mage / Toaster Club LLC (Apache-2.0, copyright Toaster Club LLC).

## 9. Test it

All tests run on a GPU (an NVIDIA A40 through Dawn in Node.js); see [TESTING.md](TESTING.md).

| What | Test | Result |
| --- | --- | --- |
| This handoff's path: `loadPair()` against a deployed site, the Trades dispatch profile on typed, spoken and photo turns | `node tools/node_desk_api_test.mjs --engine public/engine --base https://your-host/lfm/ --out out` | [results/desk](../results/desk) |
| Every input, and Live streamed against the whole turn (token for token) | `node tools/node_omni_test.mjs --site <copy> --out out` | [results/omni](../results/omni) |
| The note changes the answer (with, without, and a contradicting note) | `node tools/node_pair_test.mjs --site <copy> --out out` | [results/pair](../results/pair) |
| d1 itself against Liquid's runner | `node tools/node_d1_test.mjs ...` | every answer matches, probabilities within 0.043 |

## 10. Checklist

1. Copy the files in section 3 to a folder your app serves; verify the parts.
2. Show a load step with the size; call `loadPair()` from a tap; handle "no WebGPU" and "out of memory".
3. Write your profile: questions with your field codes, instructions with an action per answer.
4. Run turns: text first, then photos (`loadPictures()`), then speech (whole clips, then streamed Live).
5. Send `onDecision`'s answers to your backend; set thresholds and a human fallback in your code.
6. Keep the licenses, the NOTICE and the "not affiliated" line; check the revenue cap.
7. Test on the phones your customers use: load time, memory, first sound after release.

## Where things are

| File | What |
| --- | --- |
| `public/engine/loader.js` | `loadPair()`: loads both models through the device's copy |
| `public/engine/pair.js` | `Pair`: `turn`, `liveOpen` / `liveFeed` / `liveClose`, `decisionNote`, `NOTES_INSTRUCTION` |
| `public/engine/d1.js` | `Decider`: d1's prompt, readout and calibration |
| `public/engine/profiles.js` | ready-made profiles, including "Trades dispatch" |
| `public/engine/play.js` | `makePlayer()`: plays the spoken answer as it is made |
| `public/examples/voice-desk.html` | the smallest working page |
| `public/voice.html`, `public/voice.js` | the full app: composer, streamed speech, Live camera, profiles, storage |
