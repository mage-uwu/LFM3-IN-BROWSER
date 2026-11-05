# SPDX-License-Identifier: Apache-2.0. Copyright 2026 Toaster Club LLC.
"""Check a deployed site end to end in headless Chromium (WebGPU on SwiftShader when there is no GPU):

    python tools/site_test.py --base https://lfm.toasterclub.com --out out/site

1. examples/chat.html loads the language model from the site's weights and answers a question;
2. the app loads, its Settings show the credits and licenses, custom instructions ("Always answer in French.")
   start a new chat, and the next answer follows them;
3. the licenses, notices and share image are served.
Writes site_test.json and screenshots into --out. On SwiftShader each answer takes minutes.

    python tools/site_test.py --smoke [--short] [--overlay public]

runs only the app's smoke test instead (Settings → Smoke test): every workflow once, checked and measured, and
records its results and the header's tags. --short uses its smaller inputs (?smoke=short), for SwiftShader.
--overlay DIR serves the page's own files (index.html, *.js, ...) from DIR instead of the site, so a change can be
checked against the deployed weights before it is deployed.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import time
import urllib.request

SWIFTSHADER_FLAGS = ["--enable-unsafe-webgpu", "--use-webgpu-adapter=swiftshader", "--enable-unsafe-swiftshader", "--no-sandbox",
                     "--disable-gpu-watchdog"]   # a software GPU can take longer per submit than the watchdog allows
FILES = ["licenses/LFM-Open-License-v1.0.txt", "licenses/Apache-2.0.txt", "licenses/NOTICE.txt", "weights/NOTICE.txt", "og.png",
         "manifest.webmanifest", "weights/gptq-522fbf23560c/chunks.json"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="https://lfm.toasterclub.com")
    ap.add_argument("--out", default="out/site")
    ap.add_argument("--minutes", type=float, default=40, help="per wait for a model load or an answer")
    ap.add_argument("--smoke", action="store_true", help="only the app's smoke test (Settings)")
    ap.add_argument("--short", action="store_true", help="the smoke test's smaller inputs (?smoke=short)")
    ap.add_argument("--overlay", help="serve these files (a copy of public/) instead of the site's own")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    base = args.base.rstrip("/")
    res = {"base": base, "files": {}, "console_errors": [], "page_errors": []}
    for f in FILES:
        try:
            req = urllib.request.Request(f"{base}/{f}", headers={"User-Agent": "Mozilla/5.0 (lfm3-in-browser site test)"})
            with urllib.request.urlopen(req, timeout=60) as r:   # Cloudflare may refuse Python's default user agent
                res["files"][f] = [r.status, len(r.read())]
        except Exception as e:  # noqa: BLE001 - record any failure
            res["files"][f] = str(e)[:120]
    from playwright.sync_api import sync_playwright

    def wait(page, cond, minutes):
        t0 = time.time()
        while time.time() - t0 < minutes * 60:
            if page.evaluate(f"() => !!({cond})"):
                return round(time.time() - t0, 1)
            time.sleep(3)
        raise TimeoutError(cond)

    def save():
        with open(os.path.join(args.out, "site_test.json"), "w") as f:
            json.dump(res, f, indent=1)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=SWIFTSHADER_FLAGS)
        ctx = browser.new_context(**{k: v for k, v in p.devices["iPhone 15 Pro"].items() if k != "default_browser_type"})
        if args.overlay:  # the page's files from disk, everything else (the weights) from the site
            files = {}
            for root, _, names in os.walk(args.overlay):
                for n in names:
                    rel = os.path.relpath(os.path.join(root, n), args.overlay).replace(os.sep, "/")
                    if not rel.startswith("weights/"):
                        files[rel] = os.path.join(root, n)
            if "index.html" in files:
                files[""] = files["index.html"]
            res["overlay"] = sorted(k for k in files if k)
            pat = re.compile("^" + re.escape(base + "/") + "(" + "|".join(re.escape(k) for k in files) + r")(\?.*)?$")

            def fulfil(route):
                rel = route.request.url[len(base) + 1:].split("?")[0]
                with open(files[rel], "rb") as f:
                    body = f.read()
                kind = mimetypes.guess_type(files[rel])[0] or "application/octet-stream"
                route.fulfill(status=200, body=body, content_type="text/javascript" if rel.endswith((".js", ".mjs")) else kind)
            ctx.route(pat, fulfil)
        page = ctx.new_page()
        page.on("console", lambda m: m.type == "error" and res["console_errors"].append(m.text[:300]))
        page.on("pageerror", lambda e: res["page_errors"].append(str(e)[:300]))
        if args.smoke:
            smoke(page, base, args, res, wait, save)
            browser.close()
            save()
            print("SITE_TEST " + json.dumps(res)[:6000], flush=True)
            return
        # 1. the minimal example
        page.goto(f"{base}/examples/chat.html")
        page.click("#load")
        try:
            res["example_load_s"] = wait(page, "document.getElementById('status').textContent === 'Ready.' || /Error|error|HTTP/.test(document.getElementById('status').textContent)", args.minutes)
            res["example_status"] = page.inner_text("#status")
            page.click("#ask")
            res["example_answer_s"] = wait(page, "document.getElementById('out').textContent.includes('tokens at')", args.minutes)
            res["example_answer"] = page.inner_text("#out")
        except TimeoutError as e:
            res["example_timeout"] = str(e)[:160]
            res["example_status"] = page.inner_text("#status")
        page.screenshot(path=os.path.join(args.out, "example.png"))
        save()
        # 2. the app: load, About, instructions, an answer that follows them
        page.goto(f"{base}/")
        wait(page, "document.getElementById('chip').textContent !== 'checking WebGPU…'", 2)
        res["title"] = page.title()
        page.click("#loadBtn")
        try:
            res["app_load_s"] = wait(page, "!document.getElementById('input').disabled || document.querySelector('#loadMsg .err')", args.minutes)
            page.click("#setBtn")
            page.wait_for_timeout(500)
            about = page.inner_text("#settings")
            res["about_credits"] = all(s in about for s in ("Mage / Toaster Club LLC", "Liquid AI", "LFM Open License v1.0", "Apache License 2.0", "github.com/mage-uwu/LFM3-IN-BROWSER"))
            res["about_links"] = page.evaluate("() => [...document.querySelectorAll('#settings a')].map((a) => a.getAttribute('href'))")
            page.screenshot(path=os.path.join(args.out, "settings.png"))
            page.fill("#sysPrompt", "Always answer in French.")
            page.click("#sysSave")
            wait(page, "[...document.querySelectorAll('.msg.notice')].some((n) => n.textContent.includes('your instructions'))", args.minutes)
            res["instructions_stored"] = page.evaluate("() => localStorage.getItem('lfm3.system')")
            page.fill("#input", "What is the capital of Italy? Answer in one short sentence.")
            page.click("#send")
            res["app_answer_s"] = wait(page, "document.querySelectorAll('.msg.assistant .stats').length >= 1 || !!document.querySelector('.msg .err')", args.minutes)
            res["app_answer"] = page.inner_text(".msg.assistant:last-child")[:500]
        except TimeoutError as e:
            res["app_timeout"] = str(e)[:160]
        page.screenshot(path=os.path.join(args.out, "app.png"))
        browser.close()
    save()
    print("SITE_TEST " + json.dumps(res)[:3000], flush=True)


def smoke(page, base, args, res, wait, save):
    """The app's smoke test: load, Settings → Run the smoke test, then its results (window.LFM3_SMOKE)."""
    page.goto(f"{base}/" + ("?smoke=short" if args.short else ""))
    wait(page, "document.getElementById('chip').textContent !== 'checking WebGPU…'", 2)
    res["title"] = page.title()
    res["header"] = page.inner_text("header .brand")
    res["tags"] = page.evaluate("() => [...document.querySelectorAll('header .tag')].filter((t) => !t.hidden).map((t) => t.textContent)")
    page.click("#loadBtn")
    try:
        res["app_load_s"] = wait(page, "!document.getElementById('input').disabled || document.querySelector('#loadMsg .err')", args.minutes)
        res["tags_loaded"] = page.evaluate("() => [...document.querySelectorAll('header .tag')].filter((t) => !t.hidden).map((t) => t.textContent)")
        page.screenshot(path=os.path.join(args.out, "header.png"), clip={"x": 0, "y": 0, "width": page.viewport_size["width"], "height": 64})
        page.click("#setBtn")
        page.wait_for_timeout(500)
        page.click("#smokeRun")
        t0 = time.time()
        while time.time() - t0 < args.minutes * 60 * 4:
            if page.evaluate("() => (window.LFM3_SMOKE?.done || 0) >= 1"):
                break
            res["smoke_progress"] = page.inner_text("#smokeOut")[:2000]
            res["smoke_status"] = page.inner_text("#smokeMsg")[:300]
            save()
            time.sleep(20)
        res["smoke_s"] = round(time.time() - t0, 1)
        res["smoke"] = page.evaluate("() => window.LFM3_SMOKE?.result")
        res["smoke_text"] = page.evaluate("() => window.LFM3_SMOKE?.text")
        res["smoke_status"] = page.inner_text("#smokeMsg")[:300]
        res["smoke_progress"] = page.inner_text("#smokeOut")[:2000]
        res["smoke_buttons"] = page.evaluate("() => ['smokeCopy', 'smokeShare', 'smokeSave'].map((id) => [id, !document.getElementById(id).hidden])")
        page.locator("#settings .sheet").screenshot(path=os.path.join(args.out, "smoke.png"))
        page.click("#setClose")
        page.wait_for_timeout(300)
        res["notice"] = page.evaluate("() => [...document.querySelectorAll('.msg.notice')].map((n) => n.textContent).slice(-1)[0] || ''")
        res["input_enabled"] = page.evaluate("() => !document.getElementById('input').disabled")
        page.screenshot(path=os.path.join(args.out, "app.png"))
    except TimeoutError as e:
        res["app_timeout"] = str(e)[:160]
        page.screenshot(path=os.path.join(args.out, "app.png"))


if __name__ == "__main__":
    main()
