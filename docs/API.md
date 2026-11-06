# Engine API

The engine is plain ES modules in `public/engine/`. Import them from your page; there are no dependencies and no
build step. `public/engine/index.js` exports everything from one module.

## The happy path: `loader.js`

```js
import { loadLFM3, loadVision, loadAudio, loadSpeech } from "./engine/loader.js";

const { device, model, chat } = await loadLFM3({ onProgress: ({ got, total }) => {} });
const { reply } = await chat.send("Hello!", { onText: (t) => render(t) });
```

| Call | What it does |
| --- | --- |
| `loadLFM3({ base, ctx = 4096, onProgress, device })` | Opens WebGPU, or uses `device`. Loads `weights.js` if the page has not, then the language model, the tokenizer and a `Chat` with its system prompt prefilled. Returns `{ device, info, model, tokenizer, chat, fromDevice }`. |
| `loadVision(device, { base, onProgress })` | The image encoder, as a `Vision`. |
| `loadAudio(device, { base, onProgress })` | The speech encoder, as an `Audio`. |
| `loadSpeech(device, { base, onProgress })` | The spoken-reply models, as `{ depth, codec }`. |
| `weightSet(W, family, { base })` | The device cache for one weight set: `saved()`, `source()`, `prepare()`, `finish()`. The app uses it to show "Continue loading (40% saved)". |
| `cachedTokenizer(base)` | The tokenizer, kept in IndexedDB after the first fetch. |

`base` is the site's root, where `weights.js`, `weights/` and the `lfm3-tokenizer-*` files are. It defaults to the
page's folder; a page in a subfolder passes `new URL("../", location.href)`. Every weight part is stored in
IndexedDB as it arrives, under the set's id, so:
- later loads read the device's copy;
- an interrupted download resumes;
- pages on the same site share one copy.

## The weight sets

`weights.js` defines one global per weight set:

| Global | Model | Format | Size |
| --- | --- | --- | --- |
| `LFM3_WEIGHTS` | language model (needed for everything) | GPTQ `q4g64` | 0.97 GB |
| `LFM3_VISION` | image encoder + projector | `q8` | 0.19 GB |
| `LFM3_AUDIO` | speech encoder + projector | `q4g64` | 0.06 GB |
| `LFM3_TALK` | speech decoder (depth decoder + code embeddings) | `q4g64` | 0.07 GB |
| `LFM3_CODEC` | voice codec decoder | `q8` | 0.04 GB |

Each is `{ id, format, total, tensors, chunks: [{ url, bytes }] }`. The URLs are relative to the site's root.

### Loading by hand

Every model class takes `{ prequant }`: the tensor list, plus a function returning an async iterator of the weight
bytes in order. Without the loader, and without the device cache:

```js
import { pngPayload, prefetched } from "./engine/pngbytes.js";

const site = new URL("./", location.href);
async function part(c) {
  const r = await fetch(new URL(c.url, site));
  if (!r.ok) throw new Error(`${c.url}: HTTP ${r.status}`);
  return pngPayload(new Uint8Array(await r.arrayBuffer()), c.bytes);   // inflates the PNG, checks the length
}
const source = (W) => ({ id: W.id, manifest: W.tensors, chunks: () => prefetched(W.chunks, part) });
```

The examples below use `source(...)`. With the loader, use `loadVision()` and the like instead.

## `lfm3.js`: the language model

```js
import { initDevice, LFM3 } from "./engine/lfm3.js";

const { device, info } = await initDevice();            // throws if WebGPU is missing; info: { vendor, architecture, ... }
// (loadLFM3() above does all of this, with the device cache)
const model = new LFM3(device, { ctx: 4096, prequant: source(LFM3_WEIGHTS) });
await model.load(({ got, total }) => {});               // streams the weights into GPU buffers
```

| Member | What it does |
| --- | --- |
| `new LFM3(device, { ctx, prequant })` | `ctx`: the context length in tokens (default 8192). Without `prequant`, the model streams the original bf16 `model.safetensors` from `url` (default: Hugging Face) or from `file` (a picked File), and quantizes it on the GPU to `quant` (`"q8"` or `"q4"`). |
| `await load(onProgress)` | Allocates and uploads the weights. |
| `await prefill(ids, { sample = true })` | Runs the token ids through the model at the current position. With `sample`, it also picks the next token on the GPU, which `generate()` continues from. |
| `await generate(maxNew, { onToken, shouldStop, batch = 8 })` | Greedy decoding. Returns the new token ids. It stops at `<|im_end|>`, after `maxNew` tokens, or when `shouldStop()` returns true. |
| `reset()` | An empty context. |
| `await snapshot()` / `restore(snap)` | Copies of the full model state, for example after a system prompt. |
| `setExternalEmbeddings(buffer)` | Rows that ids `0x80000000 + r` read instead of the embedding table. Images, speech and speech frames enter this way. |
| `pos` | The current position, which must stay below `ctx`. |
| `hlast` | The last hidden state, written by `prefill(ids, { sample: false, hidden: true })`. The speech decoder reads it. |

## `tokenizer.js` and `chat.js`: conversations

```js
import { Tokenizer } from "./engine/tokenizer.js";
import { Chat } from "./engine/chat.js";

const chat = new Chat(model, await Tokenizer.fromFiles(site.href));   // reads lfm3-tokenizer-vocab.json and -merges.txt
await chat.newChat();                                                  // prefills the system prompt, keeps a snapshot
const { reply, stats } = await chat.send("Hello!", { onText: (t) => render(t) });
```

| Member | What it does |
| --- | --- |
| `chat.system` | The base system prompt (default `SYSTEM`, "You are a helpful personal assistant."). It takes effect at `newChat()`. |
| `chat.spoken` | Voice mode: `newChat()` ends the system prompt with "Respond with interleaved text and audio.". |
| `await newChat()` | Starts a conversation: the system prompt is prefilled once and snapshotted. |
| `await send(text, { onText, media, maxNew, shouldStop })` | One turn with a text answer. `stats` gives the prompt tokens, the prefill and decode speed, and the position. |
| `await speak(text, { depth, codec, onText, onAudio, media, shouldStop })` | One turn with a spoken answer. `onAudio(Float32Array)` receives 24 kHz audio about every 320 ms. Returns `{ reply, frames, seconds, stats }`. |
| `await openTurn(picture)`, `feedAudio(emb, rows)`, `closeTurn(text, opts)` | A spoken turn streamed in while the user talks; see below. |
| `await asr(emb, rows, language)` | Speech recognition by the model itself ("Perform ASR in <language>."). It resets the context, so run it between chats. |
| `messages` | The turns of the conversation: `{ role, content, media, audio }`. |

`media` is one item or a list of `{ token: "<|image|>" | "<|audio|>", rows }`. Their rows must already be in the
buffer given to `model.setExternalEmbeddings()`, back to back in the same order.

## `rag.js`: retrieval for one chat

Standard RAG, kept separate for each chat. Make one index per chat, add the chat's documents (and, if you like,
turns that no longer fit in the context), and put the best passages in front of each question:

```js
import { ChatIndex, withContext } from "./engine/rag.js";
const index = new ChatIndex();                    // one per chat; drop it when the chat ends
index.add(await file.text(), file.name);          // cut into ~120-word passages that overlap by 30
const hits = index.search(question, 3);           // BM25: [{ score, text, source }], best first
const { reply } = await chat.send(withContext(question, hits));
```

| Member | What it does |
| --- | --- |
| `new ChatIndex({ words, overlap })` | An empty index. Passages are `words` long (120) and overlap by `overlap` (30). |
| `add(text, source)` | Indexes a text under a source name; returns the number of passages. |
| `search(query, k)` | The `k` best passages by BM25 (k1 1.2, b 0.75). Passages that share no term with the query are left out. |
| `remove(source)`, `clear()` | Removes one source's passages, or all of them. |
| `size`, `sources` | The passage count, and a `Map` of source name to passage count. |
| `withContext(question, hits)` | The question with its passages, numbered and labelled with their source. |

## `vision.js`: photos

```js
import { preprocess, Vision } from "./engine/vision.js";

const vision = new Vision(device, { quant: LFM3_VISION.format, prequant: source(LFM3_VISION) });
await vision.load();
const pp = preprocess(imageData.data, imageData.width, imageData.height);   // RGBA bytes -> patches
const o = await vision.encode(pp.patches, pp.gh, pp.gw);                    // -> { emb, rows }
model.setExternalEmbeddings(o.emb);
await chat.send("What is in this picture?", { media: { token: "<|image|>", rows: o.rows } });
```

`preprocess(rgba, w, h, { maxPixels })` resizes like the model's processor. The default `maxPixels` is 448×448
worth; fewer pixels mean fewer image tokens and a faster answer.

## `audio.js`: speech input

```js
import { Audio, logMel } from "./engine/audio.js";
import { resample16k } from "./engine/listen.js";

const audio = new Audio(device, { quant: LFM3_AUDIO.format, prequant: source(LFM3_AUDIO) });
await audio.load();
const m = logMel(resample16k(samples, sampleRate));   // mono Float32Array -> { mel, frames }
const o = await audio.encode(m.mel, m.frames);        // -> { emb, rows }, 12.5 rows per second, up to 30 s
model.setExternalEmbeddings(o.emb);
await chat.send("", { media: { token: "<|audio|>", rows: o.rows } });   // a spoken question
```

## `speech.js`: spoken answers

```js
import { Codec, DepthDecoder } from "./engine/speech.js";

const depth = new DepthDecoder(device, { quant: LFM3_TALK.format, prequant: source(LFM3_TALK) });
const codec = new Codec(device, { quant: LFM3_CODEC.format, prequant: source(LFM3_CODEC) });
await depth.load(); await codec.load();
chat.spoken = true;
await chat.newChat();
await chat.speak("Tell me a short joke.", { depth, codec, onText: (t) => render(t), onAudio: (pcm24k) => play(pcm24k) });
```

To play the chunks back to back, see `makePlayer()` in `public/app.js`. It creates its `AudioContext` in a tap, as
iOS requires, and rebuffers when speech is made slower than real time.

## Streamed push to talk (`listen.js`, `audio.js`, `chat.js`)

```js
import { MicCapture, Resampler } from "./engine/listen.js";
import { MelStream } from "./engine/audio.js";

// at the press (inside the tap's handler: the AudioContext and getUserMedia)
const stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } });
const rs = new Resampler(ctx.sampleRate, 16000), mel = new MelStream(30);
const mic = new MicCapture(ctx, stream, { onSamples: (x) => mel.push(rs.push(x)) });
mic.start();
await chat.openTurn(null);   // or a picture's { emb, rows }

// every ~320 ms while held: the tokens that are final go into the model
const final = mel.complete();
if (final > chat.turn.rows) {
  const width = Math.min(3000, Math.ceil((mel.frames + 1) / 64) * 64);   // a few widths, so encoder plans are reused
  const o = await audio.encode(mel.layout(width), width);
  await chat.feedAudio(o.emb, Math.min(final, o.rows));
}

// at the release
mic.stop();                                    // releases the microphone
mel.push(rs.finish()); mel.finish();
const o = await audio.encode(mel.layout(), mel.frames);
await chat.feedAudio(o.emb, o.rows);
await chat.closeTurn("", { depth, codec, onAudio: play, audio: mel.x.slice(0, mel.n) });   // audio: kept in chat.messages
```

The model ends up in exactly the state that `speak()` with the whole clip would give.

## `d1.js`: decisions (LiquidAI/d1-*)

A port of the d1 release's own runner. Named questions over a state (text, any JSON value, or nothing but media)
are answered with calibrated probabilities in one forward pass of `LFM3.decide()`, with no generated tokens. The
model is an `LFM3` loaded with d1's weights (`d1-weights.js`: `D1_WEIGHTS`, `D1_VISION`, `D1_AUDIO`).

```js
import { Decider } from "./engine/d1.js";
const calibration = await (await fetch("d1/calibration.json")).json();
const d1 = new Decider(model, tokenizer, { calibration, setSize: calibration.set_size, lead: "<think></think>" });
const { answers, probabilities, usage } = await d1.ask("I was charged twice. Please refund one charge.", {
  refund: { type: "noul", instructions: "Is the customer asking for a refund?" },
  team: { type: "choice", instructions: "Which team?", criteria: { billing: "Charges", shipping: "Parcels" } },
  topics: { type: "choice", multiple: true, instructions: "Which topics?", criteria: { billing: null, delivery: null } },
  anger: { type: "score", instructions: "How upset?", criteria: ["Calm", "Annoyed", "Angry"] },
});
// answers.refund   -> { type: "noul", noul: 0.94 }                          P(yes)
// answers.team     -> { type: "choice", choice: "billing", confidence, probabilities }
// answers.topics   -> { type: "choice", choices: ["billing"], probabilities }  each option yes or no
// answers.anger    -> { type: "score", score: 1, confidence, probabilities, legend }  the median level
```

- Question types: `noul` (yes or no), `choice` (one of named options; with `multiple: true`, each option yes or
  no), `score` (2 to 10 levels, lowest first). The release's other types (`number`, `date`, boxes and so on) are not
  ported yet and throw.
- Media: `ask(state, questions, { media: { items: [{ token: "<|image|>" | "<|audio|>", rows }], emb } })`, with
  `rows` and `emb` from `Vision.encode()` or `Audio.encode()` run on d1's own encoder weights.
- Up to `setSize` (16) questions share one pass; more run as further passes over the same state. `ask()` resets the
  model's position when it is done.
- Lower level: `LFM3.decide(ids, { state, groups, slots })` runs a packed sequence under d1's "set" attention (the
  state sees every token, each question sees the state and itself) and returns the logits at `slots`.

## `pair.js`: a decision, then a spoken answer

`Pair` runs one spoken turn through both models. d1-Omni-2B hears the clip through its own speech encoder and answers
the questions; LFM3-Omni-2B hears the same clip through its encoder and answers aloud, with the decision written into
its turn as a one-line note (`decisionNote`). `newChat()` puts `NOTES_INSTRUCTION` in the voice's system prompt: use
the notes, do not read them out. A profile's own instructions can say what to do with each answer ("if escalate is
yes, tell the caller you are passing them to a person").

```js
import { Pair } from "./engine/pair.js";
const pair = new Pair({ decider, d1Audio, chat, audio, depth, codec });   // d1Audio, audio: the two speech encoders
await pair.newChat("You are the voice of a shop's customer line. ...");
const r = await pair.turn(samples16k, questions, { onDecision: (d, note) => {}, onText, onAudio });
// r.decision.answers (d1), r.note (what the voice was given), r.reply, r.seconds (of speech), r.decideMs
```

Options: `withNote: false` answers without the decision (a comparison), `note` gives a note of your own, `state` adds
text or JSON for d1 to read beside the clip, `greedy` for repeatable tests. `makePlayer()` in `play.js` plays the
reply's 24 kHz chunks as they come.

## Other modules

| Module | Exports |
| --- | --- |
| `kernels.js` | The WGSL sources: `gemvSource(quant)`, `embedSource`, the norms, attention, RoPE, arg-max, and on-GPU quantization (`quantizeSource`, `BF16_TO_F32`). |
| `pngbytes.js` | `pngPayload(bytes, n)`, and `prefetched(chunks, loadOne, { ahead })`. |
| `store.js` | `store` (IndexedDB key-value: `get`, `put`, `del`, `keys`, `delWhere`), `persist()`, `room()`. |
| `pair.js` | `Pair`, `decisionNote(questions, answers)`, `NOTES_INSTRUCTION`. |
| `play.js` | `makePlayer()`: `begin`, `play(chunk)`, `end`, `stop`, `left`. |
| `d1.js` | `Decider`, and `asQuestion`, `stateBlock`, `plain`, `through` (the prompt and calibration helpers). |
| `loader.js` | `loadLFM3`, `loadVision`, `loadAudio`, `loadSpeech`, `weightSet`, `cachedTokenizer`, `ensureWeights`, `fetchBytes`. |
| `index.js` | Re-exports the public API of every module above. |
| `listen.js` | `MicCapture`, `Resampler`, `resample16k`. |
