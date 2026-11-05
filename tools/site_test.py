# SPDX-License-Identifier: Apache-2.0. Copyright 2026 Toaster Club LLC.
"""Check a deployed site end to end in headless Chromium (WebGPU on SwiftShader when there is no GPU):

    python tools/site_test.py --base https://lfm3-in-browser.toasterclub.workers.dev --out out/site

1. examples/chat.html loads the language model from the site's weights and answers a question;
2. the app loads, its Settings show the credits and licenses, custom instructions ("Always answer in French.")
   start a new chat, and the next answer follows them;
3. the licenses, notices and share image are served.
Writes site_test.json and screenshots into --out. On SwiftShader each answer takes minutes.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request

SWIFTSHADER_FLAGS = ["--enable-unsafe-webgpu", "--use-webgpu-adapter=swiftshader", "--enable-unsafe-swiftshader", "--no-sandbox"]
FILES = ["licenses/LFM-Open-License-v1.0.txt", "licenses/Apache-2.0.txt", "licenses/NOTICE.txt", "weights/NOTICE.txt", "og.png",
         "manifest.webmanifest", "weights/gptq-522fbf23560c/chunks.json"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="https://lfm3-in-browser.toasterclub.workers.dev")
    ap.add_argument("--out", default="out/site")
    ap.add_argument("--minutes", type=float, default=40, help="per wait for a model load or an answer")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    base = args.base.rstrip("/")
    res = {"base": base, "files": {}, "console_errors": [], "page_errors": []}
    for f in FILES:
        try:
            with urllib.request.urlopen(f"{base}/{f}", timeout=60) as r:
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
        page = ctx.new_page()
        page.on("console", lambda m: m.type == "error" and res["console_errors"].append(m.text[:300]))
        page.on("pageerror", lambda e: res["page_errors"].append(str(e)[:300]))
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
            res["about_credits"] = all(s in about for s in ("Mage / Toaster Club LLC", "Liquid AI", "LFM Open License v1.0", "Apache License 2.0"))
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


if __name__ == "__main__":
    main()
