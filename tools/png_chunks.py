# SPDX-License-Identifier: Apache-2.0. Copyright 2026 Toaster Club LLC.
"""Pack the engine's weight bytes into lossless PNG byte containers (parts the page streams and checks).

    python tools/png_chunks.py --parts work/gptq/embed_gptq.bin work/gptq/rest.bin --out work/gptq/png

Some hosts take images but not binary files (a claude.ai artifact's asset store), so the bytes travel as 8-bit grayscale PNGs:
each image row is one filter byte (0, none) and WIDTH payload bytes, deflate-compressed as PNG requires. The page
(public/engine/pngbytes.js) inflates the image data with the browser's DecompressionStream and drops the filter bytes.
Writes w000.png, w001.png, ... and chunks.json: [{file, bytes, png_bytes, sha256}] (sha256 of the payload).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import zlib

WIDTH = 4096


def chunk(tag: bytes, body: bytes) -> bytes:
    return struct.pack(">I", len(body)) + tag + body + struct.pack(">I", zlib.crc32(tag + body) & 0xFFFFFFFF)


def png(payload: bytes) -> bytes:
    rows = -(-len(payload) // WIDTH)
    data = payload + bytes(rows * WIDTH - len(payload))
    raw = b"".join(b"\x00" + data[r * WIDTH:(r + 1) * WIDTH] for r in range(rows))
    ihdr = struct.pack(">IIBBBBB", WIDTH, rows, 8, 0, 0, 0, 0)  # 8-bit grayscale, no interlace
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", required=True, help="files concatenated in order")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mib", type=float, default=19.0, help="payload per image (the store takes 20 MiB per file)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    size = int(args.mib * 2 ** 20) // WIDTH * WIDTH
    out, buf, n = [], b"", 0

    def flush(b):
        nonlocal n
        name = f"w{n:03d}.png"
        img = png(b)
        with open(os.path.join(args.out, name), "wb") as f:
            f.write(img)
        out.append({"file": name, "bytes": len(b), "png_bytes": len(img), "sha256": hashlib.sha256(b).hexdigest()})
        n += 1
        print(name, len(b), len(img), flush=True)

    for p in args.parts:
        with open(p, "rb") as f:
            while True:
                piece = f.read(size - len(buf))
                if not piece:
                    break
                buf += piece
                if len(buf) == size:
                    flush(buf)
                    buf = b""
    if buf:
        flush(buf)
    total = sum(c["bytes"] for c in out)
    stored = sum(c["png_bytes"] for c in out)
    with open(os.path.join(args.out, "chunks.json"), "w") as f:
        json.dump({"width": WIDTH, "total_bytes": total, "stored_bytes": stored, "chunks": out}, f, indent=1)
    print("PNG_DONE", json.dumps({"chunks": len(out), "payload_mb": round(total / 2 ** 20, 1),
                                  "stored_mb": round(stored / 2 ** 20, 1)}), flush=True)


if __name__ == "__main__":
    main()
