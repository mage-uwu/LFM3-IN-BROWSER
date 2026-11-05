# SPDX-License-Identifier: Apache-2.0. Copyright 2026 Toaster Club LLC.
"""Put exported weight sets into the site: each set's PNG parts under public/weights/<id>/ (with its chunks.json, the
payload sha256 of every part), and public/weights.js, the list app.js reads. A set's id is a hash of its parts, so a
path never changes and the parts can be cached for good (public/_headers). Every payload is checked before copying.

    python tools/pack_weights.py \
      --set lm     work/gptq/manifest.json                work/gptq/png/chunks.json   work/gptq/png \
      --set vision work/vision/manifest_q8.json           work/vw/chunks.json         work/vw \
      --set audio  work/audio/manifest_q4g64.json         work/aw/chunks.json         work/aw \
      --set talk   work/speech/manifest_speech_q4g64.json work/sw/chunks.json         work/sw \
      --set codec  work/speech/manifest_codec_q8.json     work/cw/chunks.json         work/cw
    python tools/pack_weights.py --js d1-weights.js \
      --set d1       work/d1lm/manifest.json     work/d1lm/chunks.json     work/d1lm \
      --set d1vision work/d1vision/manifest.json work/d1vision/chunks.json work/d1vision \
      --set d1audio  work/d1audio/manifest.json  work/d1audio/chunks.json  work/d1audio
    python tools/pack_weights.py --verify            # re-check the parts already in public/weights

--js names the list written (weights.js, the chat page's, by default); a run replaces only the sets of the kinds it
packs, so the chat page's sets and the decision model's are packed separately.

Each part must stay under Cloudflare Workers' 25 MiB per static file (png_chunks.py --mib 14.5 or 18 does).
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import shutil
import struct
import zlib

# set name -> (window global the app reads, id tag)
SETS = {"lm": ("LFM3_WEIGHTS", "gptq"), "vision": ("LFM3_VISION", "vision"), "audio": ("LFM3_AUDIO", "audio"),
        "talk": ("LFM3_TALK", "talk"), "codec": ("LFM3_CODEC", "codec"),
        # LiquidAI/d1-Omni-2B, the decision model (public/decide.html reads d1-weights.js): its own language model and
        # encoders (the release changed them too)
        "d1": ("D1_WEIGHTS", "d1lm"), "d1vision": ("D1_VISION", "d1vision"), "d1audio": ("D1_AUDIO", "d1audio")}
MAX_FILE = 25 * 2**20


def png_payload(path: str, n: int) -> bytes:
    """The bytes a grayscale 8-bit PNG container carries (png_chunks.py): IDAT inflated, row filter bytes removed."""
    data = open(path, "rb").read()
    pos, width, idat = 8, 0, []
    while pos < len(data):
        ln, typ = struct.unpack(">I4s", data[pos:pos + 8])
        body = data[pos + 8:pos + 8 + ln]
        if typ == b"IHDR":
            width = struct.unpack(">I", body[:4])[0]
        elif typ == b"IDAT":
            idat.append(body)
        pos += 12 + ln
    raw = zlib.decompress(b"".join(idat))
    assert all(raw[i] == 0 for i in range(0, len(raw), width + 1)), f"{path}: a row filter other than none"
    return b"".join(raw[i + 1:i + 1 + width] for i in range(0, len(raw), width + 1))[:n]


def set_id(tag: str, chunks: list) -> str:
    return f"{tag}-" + hashlib.sha256("".join(c["sha256"] for c in chunks).encode()).hexdigest()[:12]


def check(path: str, c: dict) -> None:
    assert os.path.getsize(path) == c["png_bytes"] and c["png_bytes"] <= MAX_FILE, f"{path}: size"
    got = hashlib.sha256(png_payload(path, c["bytes"])).hexdigest()
    assert got == c["sha256"], f"{path}: payload sha256 {got} is not {c['sha256']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="public", help="the site directory (the Worker's assets.directory)")
    ap.add_argument("--set", nargs=4, action="append", default=[], metavar=("NAME", "MANIFEST", "CHUNKS", "DIR"),
                    help=f"a weight set: one of {', '.join(SETS)}, its manifest, its chunks.json, the folder of its PNGs")
    ap.add_argument("--js", default="weights.js", help="the list to write, under the site directory")
    ap.add_argument("--verify", action="store_true", help="only re-check every part under <site>/weights")
    args = ap.parse_args()
    if args.verify:
        for cj in sorted(glob.glob(os.path.join(args.site, "weights", "*", "chunks.json"))):
            d, ch = os.path.dirname(cj), json.load(open(cj))
            for c in ch["chunks"]:
                check(os.path.join(d, c["file"]), c)
            assert os.path.basename(d) == set_id(os.path.basename(d).split("-")[0], ch["chunks"]), f"{d}: id"
            print(f"{os.path.basename(d)}: {len(ch['chunks'])} parts verified")
        return
    js = [f"// The model's weight sets for the page: tensor lists and the URLs of their parts (tools/pack_weights.py)."]
    keep, tags = set(), set()
    for name, manifest, chunks_json, src in args.set:
        glob_name, tag = SETS[name]
        man, ch = json.load(open(manifest)), json.load(open(chunks_json))
        assert sum(t["bytes"] for t in man["tensors"]) == ch["total_bytes"], f"{name}: manifest and chunks disagree"
        wid = set_id(tag, ch["chunks"])
        keep.add(wid)
        tags.add(tag)
        rel = f"weights/{wid}/"
        os.makedirs(os.path.join(args.site, rel), exist_ok=True)
        for c in ch["chunks"]:
            s, d = os.path.join(src, c["file"]), os.path.join(args.site, rel, c["file"])
            if not (os.path.exists(d) and os.path.getsize(d) == c["png_bytes"]):
                check(s, c)
                shutil.copyfile(s, d)
        shutil.copyfile(chunks_json, os.path.join(args.site, rel, "chunks.json"))
        ws = {"id": wid, "format": man["format"], "total": ch["total_bytes"], "tensors": man["tensors"],
              "chunks": [{"url": rel + c["file"], "bytes": c["bytes"]} for c in ch["chunks"]]}
        js.append(f"window.{glob_name} = {json.dumps(ws, separators=(',', ':'))};")
        print(f"{name}: {wid}, {len(ch['chunks'])} parts, {sum(c['png_bytes'] for c in ch['chunks']) / 1e6:.0f} MB")
    for d in glob.glob(os.path.join(args.site, "weights", "*")):  # older sets of the kinds just packed
        if os.path.isdir(d) and os.path.basename(d).split("-")[0] in tags and os.path.basename(d) not in keep:
            shutil.rmtree(d)
    with open(os.path.join(args.site, args.js), "w") as f:
        f.write("\n".join(js) + "\n")


if __name__ == "__main__":
    main()
