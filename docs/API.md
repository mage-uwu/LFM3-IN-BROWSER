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
| `chat.useMemory` | Whether `newChat()` adds the memory notes (default true). |
| `chat.spoken` | Voice mode: `newChat()` ends the system prompt with "Respond with interleaved text and audio.". |
| `await newChat()` | Starts a conversation: the system prompt (+ memory) is prefilled once and snapshotted. |
| `await send(text, { onText, media, maxNew, shouldStop })` | One turn with a text answer. `stats` gives the prompt tokens, the prefill and decode speed, and the position. |
| `await speak(text, { depth, codec, onText, onAudio, media, shouldStop })` | One turn with a spoken answer. `onAudio(Float32Array)` receives 24 kHz audio about every 320 ms. Returns `{ reply, frames, seconds, stats }`. |
| `await openTurn(picture)`, `feedAudio(emb, rows)`, `closeTurn(text, opts)` | A spoken turn streamed in while the user talks; see below. |
| `await asr(emb, rows, language)` | Speech recognition by the model itself ("Perform ASR in <language>."). It resets the context, so run it between chats. |
| `await consolidate(onStep, { messages, transcribe })` | The memory's write path: reads the user's messages, transcribing spoken ones with `transcribe(message)`, stores what is worth keeping, then calls `newChat()`. |
| `chat.experience` | The memory (`Experience`): `facts`, `preferences`, `lessons`, `save()`, `systemPrompt(base)`. |
| `messages` | The turns of the conversation: `{ role, content, media, audio }`. |

`media` is one item or a list of `{ token: "<|image|>" | "<|audio|>", rows }`. Their rows must already be in the
buffer given to `model.setExternalEmbeddings()`, back to back in the same order.

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
await chat.closeTurn("", { depth, codec, onAudio: play, audio: mel.x.slice(0, mel.n) });   // audio: kept for the memory
```

The model ends up in exactly the state that `speak()` with the whole clip would give.

## Other modules

| Module | Exports |
| --- | --- |
| `kernels.js` | The WGSL sources: `gemvSource(quant)`, `embedSource`, the norms, attention, RoPE, arg-max, and on-GPU quantization (`quantizeSource`, `BF16_TO_F32`). |
| `pngbytes.js` | `pngPayload(bytes, n)`, and `prefetched(chunks, loadOne, { ahead })`. |
| `store.js` | `store` (IndexedDB key-value: `get`, `put`, `del`, `keys`, `delWhere`), `persist()`, `room()`. |
| `loader.js` | `loadLFM3`, `loadVision`, `loadAudio`, `loadSpeech`, `weightSet`, `cachedTokenizer`, `ensureWeights`, `fetchBytes`. |
| `index.js` | Re-exports the public API of every module above. |
| `listen.js` | `MicCapture`, `Resampler`, `resample16k`. |
