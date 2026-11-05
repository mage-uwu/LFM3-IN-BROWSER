# LFM3 in the browser

LiquidAI's [LFM3-Omni-2B](https://huggingface.co/LiquidAI/LFM3-Omni-2B) running on the device's own GPU with
WebGPU, iPhone included (Safari 26 or later). It reads text, photos and speech, and can answer out loud. Nothing
you send leaves the device: the server only hands out static files.

Live at <https://lfm3-in-browser.toasterclub.workers.dev>.

* **Talk:** hold the round mic button, speak, and let go. The answer is spoken and written as it is made. Or tap
  once to start and tap again to send. Your speech goes into the model while you are still talking, so the answer
  starts right after you let go. Press again during an answer to interrupt it.
* **Voice mode** shows a soap bubble with a small face above the message box. It smiles when idle, swells with
  your voice while you hold to talk, looks up while thinking, and moves its mouth with the answer's sound. You
  can hold the bubble itself to talk.
* **Live:** the camera button opens the camera full screen. Hold the orb to talk; the picture from when you start
  talking goes in with your words.
* **Chat:** type, or add a photo. Turn on **Speak** in the top bar to hear typed answers too.
* **On iPhone:** in Safari, Share → **Add to Home Screen** turns the site into an app with its own icon. The
  weights it has downloaded stay on the phone.

This repository is a Cloudflare Workers project with static assets only. `public/` is the whole site:

| Path | What it is | Size |
| --- | --- | --- |
| `index.html`, `app.js`, `engine/*.js` | the chat app and its hand-written WGSL engine, served as ES modules | 0.2 MB |
| `lfm3-tokenizer-*` | the tokenizer | 5.7 MB |
| `weights.js` | the weight sets' tensor lists and part URLs | 35 KB |
| `weights/gptq-…/` | the language model, GPTQ 4-bit (perplexity +0.9% over bf16) | 871 MB, 49 parts |
| `weights/vision-…/` | the image encoder, 8-bit | 182 MB, 15 parts |
| `weights/audio-…/` | the speech encoder, 4-bit | 59 MB, 5 parts |
| `weights/talk-…/` | the spoken replies' depth decoder and code embeddings, 4-bit | 67 MB, 5 parts |
| `weights/codec-…/` | the voice codec's decoder, 8-bit | 36 MB, 3 parts |

The weights are PNG byte containers: grayscale images whose pixels are the raw bytes. The page inflates them with
`DecompressionStream`, checks each part's length and streams it into GPU buffers. Every part is under 20 MB,
below Workers' limit of 25 MiB per file. Each set's folder is named after a hash of its contents, so a path
never changes and `_headers` lets browsers cache the parts for a year.

On a device, the first load downloads the language model, about 0.9 GB. The image, speech and voice models
download the first time each is used. Each part is also stored in the browser's IndexedDB as it arrives, so
later loads read the device's copy and an interrupted load resumes where it stopped.

## Deploy

From the Cloudflare dashboard: **Workers & Pages → Create → Import a repository**, pick this repository, and
keep the defaults: no build command, deploy command `npx wrangler deploy`. Every push to `main` then redeploys.

Or from a computer with Node.js:

```bash
npm install
```

```bash
npx wrangler deploy
```

The site is then at `https://lfm3-in-browser.<your-subdomain>.workers.dev`. Requests for static assets are free
and don't count against the Workers request limits. WebGPU, the camera and the microphone need a secure context;
`workers.dev` and custom domains are HTTPS.

To try it locally, run `npx wrangler dev` and open `http://localhost:8787` in a desktop Chrome, Edge or Safari.
`localhost` counts as secure.

## Where it comes from

The app, the engine and the exporters live in [mage-uwu/tinyopt](https://github.com/mage-uwu/tinyopt) under
`web/` and `tools/`. `tools/build_site.py` writes this `public/` folder, and it checks each weight part's sha256
before copying it. The engine was tested part by part against the PyTorch model on a GPU:

- The 4-bit language model's perplexity is within 0.9% of bf16.
- The image encoder matches 48 of 48 reference tokens.
- Speech input transcribes exactly.
- Spoken replies give the same text, and the 8-bit codec reaches 33.6 dB SNR against PyTorch.

## License

The model weights under `public/weights/` are derived from LiquidAI/LFM3-Omni-2B and are distributed under the
LFM Open License v1.0 (`LICENSE-LFM.txt`). They were changed from the original: quantized (GPTQ 4-bit for the
language model; 4-bit or 8-bit round-to-nearest with per-group scales for the other parts), regrouped into the
engine's tensor layout, and repacked into PNG byte containers.
