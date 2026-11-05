# Testing

Every part of the engine was checked against LiquidAI/LFM3-Omni-2B in PyTorch, on the same inputs. The PyTorch
references are Python scripts that need a CUDA GPU. The engine side runs on a real GPU through **Dawn**, Chrome's
WebGPU implementation, which the npm package `webgpu` exposes to Node.js. It is the same engine code the page runs,
unchanged. The app itself is tested in headless Chromium with Playwright. Results from an NVIDIA A40 are in
[`results/`](../results/).

## Setup

```bash
pip install torch transformers safetensors huggingface_hub datasets soundfile scipy pillow numpy
```

```bash
npm install --no-save webgpu
```

Some tests read the original checkpoint over HTTP with range requests. Serve it locally:

```bash
python3 tools/serve.py --root . --port 8080 --map /model/model.safetensors=$(python3 -c "from huggingface_hub import hf_hub_download as d; print(d('LiquidAI/LFM3-Omni-2B', 'model.safetensors'))")
```

## Engine tests (Node.js + Dawn on a GPU)

Each test has a PyTorch reference script that writes into `work/`, and a Node test that compares against it.

| What | Reference (PyTorch) | Engine test | Result file |
| --- | --- | --- | --- |
| Language model: logits, greedy text, speed; GPTQ weights | `python3 tools/lfm3_ref.py`; `python3 tools/gptq_export.py --out work/gptq` | `node tools/node_bench.mjs --ref work/test_ref.json --quant gptq --pre work/gptq --parts embed_gptq.bin,rest.bin --src http://127.0.0.1:8080/model/model.safetensors` | `result_node_gptq.json` |
| Image encoder | `python3 tools/vision_ref.py --out work/vision` and `vision_export.py` | `node tools/node_vision_test.mjs --dir work/vision --src …` | `result_vision.json` |
| Speech encoder | `python3 tools/audio_ref.py --out work/audio` and `audio_export.py` | `node tools/node_audio_test.mjs --dir work/audio --src …` | `result_audio.json` |
| Spoken replies: depth decoder, codec, the interleaved loop; our speech transcribed back | `python3 tools/speech_ref.py --out work/speech` and `speech_export.py` | `node tools/node_speech_test.mjs --dir work/speech --audio work/audio --src …` | `result_speech.json` |
| Push to talk: a streamed turn against the whole clip | the speech and audio exports | `node tools/node_ptt_test.mjs --clip work/voice16k.f32 --audio work/audio --speech work/speech --src …` | `result_ptt.json` |
| Memory from speech: spoken turns heard, learned and recalled | the speech and audio exports | `node tools/node_memory_voice_test.mjs --audio work/audio --speech work/speech --src …` | `result_memory_voice.json` |
| Tokenizer and chat template | `python3 tools/tok_ref.py --out work/tok_ref.json` | `node tools/tok_test.mjs work/tok_ref.json public` | (printed) |

The push-to-talk test needs a 16 kHz clip as raw float32 (`work/voice16k.f32`). This writes the LibriSpeech
sample the other tests use:

```bash
python3 -c "import sys; sys.path.insert(0, 'tools'); from audio_ref import clip; a, t = clip(); a.tofile('work/voice16k.f32')"
```

## The app in a browser

`tools/ui_test.py` drives the page as a user would, on an emulated iPhone viewport in headless Chromium:
loading, chatting, photos, voice messages, spoken replies, push to talk, Live with Chromium's fake camera and
microphone, and the settings. It saves screenshots and a `ui_test.json`. Without a GPU, Chromium runs WebGPU on
SwiftShader (the CPU): it is correct but slow, so allow tens of minutes per answer.

```bash
pip install playwright
```

```bash
python3 -m playwright install --with-deps chromium
```

```bash
python3 tools/serve.py --root public --port 8080
```

```bash
python3 tools/fake_media.py --wav work/live.wav --y4m work/house.y4m
```

```bash
python3 tools/ui_test.py --page "/index.html?maxpx=65536" --assets --faces --talk --live --fake-audio work/live.wav --fake-video work/house.y4m --out out/ui
```

## A deployed site

`tools/site_test.py` checks a live deployment. It runs the minimal example, loads the app, checks the credits and
licenses in Settings, sets custom instructions and asks a question, and checks that the license files are served:

```bash
python3 tools/site_test.py --base https://lfm.toasterclub.com --out out/site
```

## What was found this way

These tests caught real bugs before users did:

- A streamed log-mel that wrote each frame into the wrong column. The 16 kHz samples matched; the mel did not.
- A codec that was too lossy at 4 bits: 13 dB SNR, against 33.6 dB at 8 bits. The codec ships at 8 bits.
