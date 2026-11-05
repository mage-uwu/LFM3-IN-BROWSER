# Architecture

LFM3 in the Browser runs every part of LiquidAI's LFM3-Omni-2B on the user's GPU through WebGPU: the language
model, the image encoder, the speech encoder, the speech decoder and the voice codec. The engine is hand-written
WGSL in `public/engine/`, with no ML framework, no dependencies and no build step.

```
text ──────────► tokenizer.js ─┐
photo ─► vision.js (14-layer   │                         ┌─► text tokens ─► tokenizer.js ─► text
          encoder + projector) ├─► lfm3.js (24 layers) ──┤
speech ─► audio.js (log-mel,   │                         └─► hidden state ─► speech.js: depth decoder (12 codes per
          17-layer encoder) ───┘                                              frame) ─► codec ─► 24 kHz audio
```

## Modules

| Module | Role |
| --- | --- |
| `loader.js` | The happy path: one call loads a model through the device's IndexedDB copy (`loadLFM3`, `loadVision`, `loadAudio`, `loadSpeech`). |
| `lfm3.js`, `kernels.js` | The language model; the shared WGSL kernels (GEMV/GEMM per weight format, norms, attention, RoPE, arg-max, on-GPU quantization). |
| `chat.js`, `tokenizer.js` | The chat template, conversations, spoken turns, streamed push to talk, speech recognition and the memory; the tokenizer. |
| `vision.js` | Image preprocessing, the image encoder and its projector. |
| `audio.js` | Log-mel (whole clips, and streamed while recording), the speech encoder and its projector. |
| `speech.js` | Spoken replies: the depth decoder and the voice codec decoder. |
| `listen.js` | Microphone capture, and the resampler to 16 kHz. |
| `pngbytes.js`, `store.js` | Weight part containers; IndexedDB storage. |
| `index.js` | Re-exports the public API of all of the above. |

Each module imports only its siblings, so a page can take just the ones it needs.

## Constraints

- **No WebGPU extensions.** f16 values are packed into u32 words and unpacked in the shaders, so everything runs
  on baseline WebGPU: Safari on iOS 26+, Chrome, Edge and desktop Safari. Dawn in Node.js runs the same code for
  testing.
- **Phones.** Buffers stay under 1 GiB each, and binding counts stay within the baseline limits (`initDevice()` in
  `lfm3.js`).
- **The weights are static files.** The page needs nothing but a static host; see [WEIGHTS.md](WEIGHTS.md).

## The language model (`lfm3.js`, `kernels.js`)

LFM3's backbone has 24 layers of width 2048:

- 13 gated short-convolution layers;
- 5 sliding-window attention layers, with a 256-token window;
- 6 full-attention layers;
- a feed-forward block of width 7680 in every layer.

Attention has 16 query heads of 128, sharing 4 key/value heads. The output head is tied to the 128K-row embedding
table.

| Part | How it runs |
| --- | --- |
| Matrices | One GEMV kernel per weight format (`gemvSource`). For prefill, the same kernel takes up to 16 tokens at a time, so each weight is read once per chunk. |
| Short-conv layers | GEMV `in_proj`. Then one kernel (`CONV_MIX`) does the B·x gate, the width-7 causal convolution with its carried state, and the C gate. Then GEMV `out_proj`. |
| Sliding-window attention | q, k, v and the per-head gate come from one GEMV. One kernel (`QK_ROPE_STORE`) applies the QK RMSNorm and RoPE and stores keys and values into a 512-slot ring of packed f16. |
| Full attention | The same, with a linear cache up to the context length. Split-chunk attention (`ATTN_PARTIAL` + `ATTN_COMBINE`, flash-decoding) keeps long contexts parallel. |
| Feed-forward | `w1` and `w3` in one GEMV, then SiLU·mul, then `w2`. The sandwich norm, the residual add and the next layer's norm are fused (`POST_ADD_NORM`). |
| Output | The final norm, then the tied head, then an arg-max on the GPU (`ARGMAX1`/`ARGMAX2`). The arg-max skips the media placeholders and the speech-mode markers. |
| Decoding | GPU-driven. The sampler writes the next token and advances the position on the GPU, so 8 to 16 tokens run per submit with no CPU round trip. A stop flag set on the GPU freezes all state at `<|im_end|>`. |
| State | `snapshot()` and `restore()` copy the caches. The app prefills the system prompt once and restores it for every new chat. |

Media enter the model as **external embedding rows**. A prompt id of `0x80000000 + r` reads row `r` of a buffer set
with `setExternalEmbeddings()`. The image encoder, the speech encoder and the speech decoder all write such rows,
and the chat code puts their ids where the chat template has `<|image|>` or `<|audio|>` placeholders.

### Weight formats

| Format | Layout | Used for |
| --- | --- | --- |
| `q8` | int8 codes in groups of 32 along the input dimension, one f32 scale per group | image encoder, voice codec, and the in-browser conversion of the original weights |
| `q4g64` | 4-bit codes in groups of 64 along the input dimension, an f16 scale and an f16 minimum per group (4.5 bits per weight); nibble i of word j is element 8j+i | the language model (GPTQ), speech encoder, speech decoder |

The language model's 4-bit weights come from GPTQ (`tools/gptq_export.py`). Each matrix's rounding error is
compensated using the second moment of its real inputs, layer after layer on the already-quantized outputs, with
chat-formatted and web text as calibration. Held-out perplexity is 12.73, against 12.62 for bf16; plain
round-to-nearest 4-bit gives 13.05.

## Images (`vision.js`)

1. **Preprocessing** (JavaScript, matching the model's image processor). Smart resize to multiples of 32 pixels
   (the app allows up to 448×448 worth), then antialiased bicubic resampling and normalization to [-1, 1]. The
   image is cut into 16×16 patches in 2×2 merge-block order.
2. **The LFM3-VL encoder** has 14 layers of width 1024. Gated depthwise 3×3 convolutions alternate with
   bidirectional attention: 16 heads of 64 with 8 key/value heads, QK RMSNorm, and axial 2D RoPE (rows and columns
   on separate dimensions).
3. **The projector** applies a LayerNorm, merges each 2×2 block (giving width 4096), then GELU, giving 2048-wide
   rows. One row is one image token.

## Speech input (`audio.js`)

1. **Log-mel** (JavaScript, matching the model's feature extractor): 16 kHz audio, a periodic Hann window, a
   400-point DFT with a 160-sample hop, and 128 Slaney mel bins. Values are log10, clamped at −12, then mapped
   through (x + 4) / 4. Each frame depends only on its own 400 samples.
2. **The downsampler** applies three stride-2 convolutions over the (mel, time) plane, so 8 frames become one token.
   That is 12.5 tokens per second.
3. **The encoder** has 17 LFM2 layers of width 768: gated short convolutions of width 3, and causal attention over
   a 32-token window (8 heads of 96, 4 key/value heads).
4. **The projector** maps each token to a 2048-wide row.

Token t reads only frames up to 8t + 7, and every layer is causal. So the first ⌊frames/8⌋ tokens of a partial
clip are already final, and `MelStream` with `Chat.feedAudio` streams speech into the model while the user is
still talking (see below).

## Speech output (`speech.js`, `Chat.speak`)

This is the model's interleaved mode. The system prompt ends with "Respond with interleaved text and audio.", and
the turn opens with `<|mixed_start|>`. Generation alternates blocks of 6 text tokens and 12 speech frames. After
`<|text_end|>` only speech is generated, up to the end-of-audio frame, then text again up to `<|im_end|>`.

- **The depth decoder** turns the language model's hidden state into one frame of 12 codebook codes. It is a
  3-layer bidirectional LFM3 over the 12 codebook slots. The codes are revealed MaskGIT-style over 3 passes of 4,
  in a random order. Each code is sampled with a repetition penalty of 1.2 over that codebook's last 64 frames, then
  temperature 0.4, the model card's settings. The frame's next input to the language model is the sum of its 12
  codebook embeddings.
- **The Liquid codec decoder** does FSQ decoding, then an 8-layer causal LFM3 at 6× the frame rate (convolutions
  and sliding-window attention, with state carried between chunks), then a linear head to magnitude and phase, then
  an inverse STFT with overlap-add. Each frame is 1920 samples at 24 kHz. Audio is decoded and played about every
  4 frames (320 ms).

Text is greedy. These are the same sampling settings and mode switches as the model's own `generate()`, which was
checked against a deterministic PyTorch run.

## Push to talk, streamed (`listen.js`, `audio.js`, `chat.js`)

This follows the official `prefill_audio` flow.

1. **At the press,** `Chat.openTurn()` prefills the turn's opening, plus any camera picture.
2. **While the button is held,** `MicCapture` delivers the microphone samples. `Resampler`, a windowed sinc,
   converts them to 16 kHz exactly as it would resample the whole clip. `MelStream` computes each log-mel frame
   once its samples have arrived. Every 320 ms, the speech encoder runs on what has arrived (at a few fixed widths,
   so its plans are reused), and `Chat.feedAudio()` prefills the tokens that are final.
3. **At the release,** only the last tokens and the turn's closing remain. `Chat.closeTurn()` adds them and the
   spoken answer starts.

The result is identical to sending the whole clip at once: 24 of 24 text tokens and 89 of 89 speech frames, on the
GPU. The first audio came 164 ms after the release, against 389 ms without streaming.

The microphone is opened at the press and released after it, so phones play the answer through the speaker rather
than the earpiece. When the device makes speech slower than real time, the player waits for a longer stretch
before continuing, doubling the wait each time up to 4 s. The answer then pauses between phrases instead of
stuttering.

## Memory (`chat.js`: `Experience`, `consolidate`, `asr`)

- **What it stores:** short notes about the user, as facts (`key = value`), standing instructions, and lessons.
  The notes are kept in `localStorage`.
- **How it is used:** the notes are rendered into the system prompt, which is prefilled once and snapshotted.
  Nothing is retrieved per message and the weights never change.
- **When it learns:**
  - a new chat starts: the finished one is read first;
  - the next visit: for a conversation that ended with the tab, kept in IndexedDB as it went;
  - "Learn from this chat" is tapped.
- **How it learns:**
  1. Each spoken turn keeps its 16 kHz audio. It is transcribed first by the model's own speech recognition, with
     the model card's system prompt "Perform ASR in <language>.".
  2. Then, for every user message, the model proposes notes.
  3. A gate keeps a fact only if its value appears in what the user said. A new value for a known key replaces the
     old one and keeps it in a history.
- **Your own instructions:** an optional system prompt (Settings) replaces the default base prompt. The memory
  comes after it, and in voice mode the speech instruction always goes last.

## The app (`app.js`, `index.html`)

The app is one page. Its parts:

- the setup card;
- the conversation;
- a composer whose round button is push to talk;
- a soap-bubble character in voice mode, whose mouth follows the output audio through an `AnalyserNode`;
- Live, a full-screen camera mode;
- the memory and settings sheets.

The weight sets download the first time each is needed, and their parts are stored in IndexedDB as they stream in.
An interrupted load resumes from the parts already stored.
