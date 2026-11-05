# Weights

The weights in `public/weights/` are derived from Liquid AI's LFM3-Omni-2B under the
[LFM Open License v1.0](../LICENSE-LFM.txt); see [NOTICE](../NOTICE). We quantized them, reordered them into the
engine's tensor layout, and packed them into PNG byte containers.

| Set | Folder | Format | Payload | Parts |
| --- | --- | --- | --- | --- |
| Language model | `weights/gptq-522fbf23560c/` | GPTQ 4-bit (`q4g64`) | 974 MB | 49 |
| Image encoder + projector | `weights/vision-df571b74252a/` | 8-bit (`q8`) | 191 MB | 15 |
| Speech encoder + projector | `weights/audio-1e6edcd9339f/` | 4-bit (`q4g64`) | 61 MB | 5 |
| Speech decoder (depth decoder + code embeddings) | `weights/talk-6d397a45176c/` | 4-bit (`q4g64`) | 70 MB | 5 |
| Voice codec decoder | `weights/codec-b3eb35fe8be9/` | 8-bit (`q8`) | 37 MB | 3 |

Only the language model is needed for text chat. The page downloads the others the first time a photo, speech or
a spoken answer needs them.

## The PNG byte containers

Each part is a grayscale 8-bit PNG, 4096 pixels wide. Every row is one filter byte (always 0, "none") followed by
4096 bytes of payload. The payload of all parts, concatenated, is the weight stream in the engine's
`weightList()` order. Why PNG:
- every static host and CDN serves it;
- some hosts accept images but not binary files;
- deflate is lossless, and the browser inflates it natively (`DecompressionStream`).

Each set's folder is named after a hash of its parts' payload hashes, so a file at a given path never changes, and
`public/_headers` lets browsers cache the parts for a year. Each folder's `chunks.json` lists every part's payload
size and sha256.

Checking all parts:

```bash
python3 tools/pack_weights.py --verify
```

## Regenerating the weights

You need Python 3.10+ with `torch`, `transformers`, `safetensors`, `huggingface_hub` and `numpy`. The exporters
download LiquidAI/LFM3-Omni-2B from Hugging Face. GPTQ needs a CUDA GPU; the other exporters run anywhere.

```bash
python3 tools/gptq_export.py --out work/gptq
```

```bash
python3 tools/vision_export.py --out work/vision --quant q8
```

```bash
python3 tools/audio_export.py --out work/audio --quant q4g64
```

```bash
python3 tools/speech_export.py --out work/speech --quant q8 q4g64
```

Decide's sets come from LiquidAI/d1-Omni-2B with the same exporters (`--repo LiquidAI/d1-Omni-2B`). Its language
model is plain round-to-nearest 4-bit (`--rtn`: no calibration text, about a minute on a GPU), which matched the
release's runner better than GPTQ:

```bash
python3 tools/gptq_export.py --repo LiquidAI/d1-Omni-2B --rtn --out work/d1rtn
```

```bash
python3 tools/vision_export.py --repo LiquidAI/d1-Omni-2B --quant q8 --out work/d1vision
```

```bash
python3 tools/audio_export.py --repo LiquidAI/d1-Omni-2B --quant q4g64 --out work/d1audio
```

They are packed with `pack_weights.py --js d1-weights.js --set d1 … --set d1vision … --set d1audio …`, which
leaves the chat's sets and `weights.js` alone. The calibration (`public/d1/calibration.json`) is the release's
`calibration.json` with its `d1_set_size`.

Pack each set into PNG parts of 14.5 MiB, below the 25 MiB that Cloudflare Workers serve per file. For the
language model, `gptq_export.py` writes two versions of the embedding table: `embed_gptq.bin` and `embed_rtn.bin`.
Its `manifest.json` reports their held-out perplexities, 12.73 and 12.94 for the shipped weights, and the shipped
weights use the better one, `embed_gptq.bin`.

```bash
python3 tools/png_chunks.py --parts work/gptq/embed_gptq.bin work/gptq/rest.bin --out work/gptq/png --mib 14.5
```

```bash
python3 tools/png_chunks.py --parts work/vision/q8.bin --out work/vw --mib 14.5
```

```bash
python3 tools/png_chunks.py --parts work/audio/q4g64.bin --out work/aw --mib 14.5
```

```bash
python3 tools/png_chunks.py --parts work/speech/speech_q4g64.bin --out work/sw --mib 14.5
```

```bash
python3 tools/png_chunks.py --parts work/speech/codec_q8.bin --out work/cw --mib 14.5
```

Put them into the site. This also writes `public/weights.js` and removes sets the site no longer lists:

```bash
python3 tools/pack_weights.py --set lm work/gptq/manifest.json work/gptq/png/chunks.json work/gptq/png --set vision work/vision/manifest_q8.json work/vw/chunks.json work/vw --set audio work/audio/manifest_q4g64.json work/aw/chunks.json work/aw --set talk work/speech/manifest_speech_q4g64.json work/sw/chunks.json work/sw --set codec work/speech/manifest_codec_q8.json work/cw/chunks.json work/cw
```

The tokenizer files come from the model's `tokenizer.json`:

```bash
python3 tools/export_tokenizer.py --out public
```

Exports are not always bit-for-bit identical across machines, because floating-point rounding differs between
devices. So a regenerated set can get a different id (folder name). That is expected; `weights.js` points the page
at whatever ids you pack.

## Without the prequantized weights

Without `weights.js` sets, the app falls back to the original checkpoint. It streams the bf16 `model.safetensors`
from Hugging Face (or reads a file the user picks) with HTTP range requests, and quantizes each tensor on the GPU
as it arrives, to 8-bit (1.9 GB, closest to the original) or 4-bit round-to-nearest. This only covers the language
model; the photo and speech features need the exported sets.
