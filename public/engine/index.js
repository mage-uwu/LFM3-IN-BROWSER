// SPDX-License-Identifier: Apache-2.0. Copyright 2026 Toaster Club LLC.
// The engine in one import:  import { loadLFM3, Chat, ... } from "./engine/index.js";
// Each module also works on its own; see docs/API.md for what each does.

// the happy path: one call loads a model through this device's copy (loader.js)
export { cachedTokenizer, ensureD1Weights, ensureWeights, loadAudio, loadLFM3, loadPair, loadSpeech, loadVision, weightSet } from "./loader.js";
// the language model on WebGPU, and the chat format and spoken turns on top of it
export { CFG, initDevice, LFM3 } from "./lfm3.js";
export { Chat, SPEECH_INSTRUCTION, SYSTEM, mediaIds, systemBlock, turnUser } from "./chat.js";
export { Tokenizer } from "./tokenizer.js";
// photos, speech in, speech out
export { preprocess, smartResize, VCFG, Vision } from "./vision.js";
export { ACFG, Audio, logMel, MelStream } from "./audio.js";
export { CCFG, Codec, DepthDecoder, SCFG } from "./speech.js";
// the microphone, weight containers and storage
export { MicCapture, Resampler, resample16k } from "./listen.js";
export { pngPayload, prefetched } from "./pngbytes.js";
export { persist, room, store } from "./store.js";
export { ChatIndex, withContext } from "./rag.js";
// d1 decision models, and Voice Desk: both models on every turn, its player and ready-made profiles
export { Decider, asQuestion, plain, stateBlock, through } from "./d1.js";
export { LIVE_PIXELS, NOTES_INSTRUCTION, Pair, decisionNote } from "./pair.js";
export { makePlayer } from "./play.js";
export { PROFILES } from "./profiles.js";
