# SPDX-License-Identifier: Apache-2.0. Copyright 2026 Toaster Club LLC.
"""The site's images: home-screen icons (icon-180.png, icon-512.png) and the share card for links (og.png, 1200x630),
which shows the app's soap bubble with its face.

    python tools/make_icons.py --site public
"""

from __future__ import annotations

import argparse
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


def icon(path: str, size: int):
    """White, two faint halos, the push-to-talk waveform in near-black (drawn 4x and scaled down)."""
    S = size * 4
    img = Image.new("RGB", (S, S), (255, 255, 255))
    halo = Image.new("RGB", (S, S), (255, 255, 255))
    d = ImageDraw.Draw(halo)
    d.ellipse((-0.25 * S, -0.25 * S, 0.65 * S, 0.6 * S), fill=(214, 226, 255))
    d.ellipse((0.4 * S, 0.45 * S, 1.25 * S, 1.25 * S), fill=(236, 222, 255))
    img = Image.blend(img, halo.filter(ImageFilter.GaussianBlur(S * 0.12)), 0.9)
    d = ImageDraw.Draw(img)
    bars, w, gap = [0.22, 0.42, 0.62, 0.36, 0.16], 0.075 * S, 0.055 * S
    x = (S - (len(bars) * w + (len(bars) - 1) * gap)) / 2
    for h in bars:
        d.rounded_rectangle((x, (S - h * S) / 2, x + w, (S + h * S) / 2), radius=w / 2, fill=(14, 15, 18))
        x += w + gap
    img.resize((size, size), Image.LANCZOS).save(path, optimize=True)


def bubble(size: int) -> Image.Image:
    """The soap bubble with its face, RGBA: a milky film, an iridescent rim, two highlights, eyes and a smile."""
    S = size * 2
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
    c = (S - 1) / 2
    r = np.hypot(xx - c, yy - c) / (S / 2)
    ang = (np.arctan2(yy - c, xx - c) / (2 * np.pi) + 1) % 1
    hues = np.array([[143, 224, 255], [201, 166, 255], [255, 179, 220], [255, 233, 163], [169, 245, 214], [143, 224, 255]], np.float32)
    t = ang * 5
    i = np.floor(t).astype(int)
    f = (t - i)[..., None]
    film = hues[i] * (1 - f) + hues[np.minimum(i + 1, 5)] * f                  # the rim's colours around the circle
    rim = np.clip((r - 0.62) / 0.3, 0, 1) ** 1.6                               # strongest at the edge
    base = np.array([246, 249, 255], np.float32)
    rgb = base * (1 - rim[..., None] * 0.85) + film * (rim[..., None] * 0.85)
    alpha = np.where(r <= 1, 0.72 + 0.28 * rim, 0) * np.clip((1 - r) * S / 3, 0, 1)
    img = Image.fromarray(np.dstack([np.clip(rgb, 0, 255), alpha * 255]).astype(np.uint8), "RGBA")
    img = img.filter(ImageFilter.GaussianBlur(S * 0.004))
    d = ImageDraw.Draw(img)
    hl = lambda cx, cy, rr, a: d.ellipse((cx - rr, cy - rr, cx + rr, cy + rr), fill=(255, 255, 255, a))  # noqa: E731
    hl(S * 0.31, S * 0.25, S * 0.05, 245)
    hl(S * 0.71, S * 0.79, S * 0.022, 210)
    ink = (36, 38, 43, 255)
    for ex in (0.39, 0.61):
        d.ellipse((S * (ex - 0.036), S * (0.45 - 0.054), S * (ex + 0.036), S * (0.45 + 0.054)), fill=ink)
    d.arc((S * 0.43, S * 0.50, S * 0.57, S * 0.64), start=20, end=160, fill=ink, width=max(2, int(S * 0.024)))
    return img.resize((size, size), Image.LANCZOS)


def card(path: str):
    """The 1200x630 share card: white, halos, the bubble, the name and one line."""
    W, H = 1200, 630
    img = Image.new("RGB", (W, H), (255, 255, 255))
    halo = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(halo)
    d.ellipse((-260, -320, 520, 300), fill=(220, 230, 255))
    d.ellipse((760, 330, 1460, 900), fill=(240, 228, 255))
    img = Image.blend(img, halo.filter(ImageFilter.GaussianBlur(90)), 0.9)
    b = bubble(300)
    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).ellipse((120, 210, 400, 520), fill=(100, 120, 190, 50))
    img.paste(shadow.filter(ImageFilter.GaussianBlur(28)), (0, 0), shadow.filter(ImageFilter.GaussianBlur(28)))
    img.paste(b, (110, 165), b)

    def font(size, bold=False):
        for name in (["DejaVuSans-Bold.ttf", "Arial Bold.ttf"] if bold else ["DejaVuSans.ttf", "Arial.ttf"]):
            try:
                return ImageFont.truetype(name, size)
            except OSError:
                continue
        return ImageFont.load_default()
    d = ImageDraw.Draw(img)
    room = W - 472 - 48

    def fit(text, size, bold=False):  # the largest size up to `size` at which the line fits beside the bubble
        while size > 12 and d.textlength(text, font=font(size, bold)) > room:
            size -= 1
        return font(size, bold)
    d.text((470, 200), "LFM3 in the Browser", font=fit("LFM3 in the Browser", 58, True), fill=(17, 18, 20))
    for k, line in enumerate(("An omni model on your own GPU: text,", "photos, speech and spoken replies.", "Nothing leaves the device.")):
        d.text((472, 288 + 40 * k), line, font=fit(line, 28), fill=(90, 94, 102))
    foot = "WebGPU · LFM3-Omni-2B by Liquid AI · Toaster Club LLC"
    d.text((472, 428), foot, font=fit(foot, 22), fill=(140, 144, 152))
    img.save(path, optimize=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="public")
    args = ap.parse_args()
    for n in (180, 512):
        icon(os.path.join(args.site, f"icon-{n}.png"), n)
    card(os.path.join(args.site, "og.png"))
    print("icons and og.png ->", args.site)


if __name__ == "__main__":
    main()
