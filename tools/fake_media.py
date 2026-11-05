# SPDX-License-Identifier: Apache-2.0. Copyright 2026 Toaster Club LLC.
"""Inputs for Chromium's fake media devices (the Live-mode browser test, tools/ui_test.py --live):

    python tools/fake_media.py --wav work/live.wav --y4m work/house.y4m

live.wav   the LibriSpeech sample (tools/audio_ref.py clip()) then 3 s of silence, 16-bit PCM at 16 kHz; the fake
           microphone loops it, so the page hears a whole utterance and its end whenever it starts listening
house.y4m  tools/vision_ref.py's synthetic picture, 640x480, as two YUV 4:2:0 frames for the fake camera
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))


def y4m(path: str, rgb: np.ndarray, frames: int = 2):
    h, w, _ = rgb.shape
    r, g, b = (rgb[..., i].astype(np.float32) for i in range(3))
    y = 0.299 * r + 0.587 * g + 0.114 * b                      # BT.601 full range ("C420jpeg")
    u = (b - y) * 0.564 + 128
    v = (r - y) * 0.713 + 128
    sub = lambda c: c.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))  # noqa: E731
    planes = [np.clip(p, 0, 255).round().astype(np.uint8).tobytes() for p in (y, sub(u), sub(v))]
    with open(path, "wb") as f:
        f.write(f"YUV4MPEG2 W{w} H{h} F30:1 Ip A1:1 C420jpeg\n".encode())
        for _ in range(frames):
            f.write(b"FRAME\n" + b"".join(planes))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", default="work/live.wav")
    ap.add_argument("--y4m", default="work/house.y4m")
    args = ap.parse_args()
    import soundfile as sf
    from audio_ref import clip
    from vision_ref import picture
    a, text = clip()
    sf.write(args.wav, np.concatenate([a, np.zeros(3 * 16000, np.float32)]), 16000, subtype="PCM_16")
    y4m(args.y4m, np.asarray(picture().convert("RGB")))
    print("FAKE_MEDIA", args.wav, f"{len(a) / 16000:.1f} s + 3 s silence:", text, "|", args.y4m, flush=True)


if __name__ == "__main__":
    main()
