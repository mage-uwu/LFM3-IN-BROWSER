# SPDX-License-Identifier: Apache-2.0. Copyright 2026 Toaster Club LLC.
"""Static file server with HTTP Range support (the WebGPU engine streams the checkpoint with Range requests).

    python tools/serve.py --root . --port 8080 --map /model/model.safetensors=/path/to/model.safetensors
"""

from __future__ import annotations

import argparse
import mimetypes
import os
import re
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

MAPS: dict[str, str] = {}
CSP = ""
# close to what a claude.ai artifact allows: scripts from a few CDNs, fonts from Google, no other hosts
ARTIFACT_CSP = ("default-src 'self'; script-src 'self' 'unsafe-inline' https://cdnjs.cloudflare.com https://cdn.jsdelivr.net "
                "https://unpkg.com; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; font-src https://fonts.gstatic.com "
                "data:; connect-src 'self'; img-src 'self' data: blob:; worker-src 'self' blob:")


class Handler(SimpleHTTPRequestHandler):
    def translate_path(self, path):
        p = path.split("?", 1)[0]
        return MAPS.get(p) or super().translate_path(path)

    def end_headers(self):
        if CSP:
            self.send_header("Content-Security-Policy", CSP)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Expose-Headers", "Content-Range, Accept-Ranges, Content-Length")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self):  # noqa: N802
        path = self.translate_path(self.path)
        rng = self.headers.get("Range")
        if not rng or not os.path.isfile(path):
            return super().do_GET()
        m = re.match(r"bytes=(\d+)-(\d*)", rng)
        size = os.path.getsize(path)
        start = int(m.group(1))
        end = int(m.group(2)) if m.group(2) else size - 1
        end = min(end, size - 1)
        self.send_response(206)
        self.send_header("Content-Type", mimetypes.guess_type(path)[0] or "application/octet-stream")
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            left = end - start + 1
            while left > 0:
                chunk = f.read(min(left, 1 << 20))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return
                left -= len(chunk)

    def log_message(self, *a):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--map", action="append", default=[], help="/url/path=/file/path")
    ap.add_argument("--artifact-csp", action="store_true", help="send an artifact-like Content-Security-Policy")
    args = ap.parse_args()
    global CSP
    CSP = ARTIFACT_CSP if args.artifact_csp else ""
    for m in args.map:
        k, v = m.split("=", 1)
        MAPS[k] = v
    os.chdir(args.root)
    mimetypes.add_type("text/javascript", ".js")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
