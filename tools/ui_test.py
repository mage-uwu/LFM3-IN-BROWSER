# SPDX-License-Identifier: Apache-2.0. Copyright 2026 Toaster Club LLC.
"""Drive the web app (public/index.html) as a user would, on an emulated iPhone 15 Pro viewport in headless Chromium:
load the model, send a message, wait for the streamed reply and its speed line; save console errors and
screenshots.

    python tools/ui_test.py --base http://127.0.0.1:8080 --out out/web
"""

from __future__ import annotations

import argparse
import json
import os
import time

# headless Chromium with WebGPU on SwiftShader (the CPU), for machines without a GPU Chromium will use
SWIFTSHADER_FLAGS = ["--enable-unsafe-webgpu", "--use-webgpu-adapter=swiftshader", "--enable-unsafe-swiftshader", "--no-sandbox",
                     "--disable-gpu-watchdog"]   # a software GPU can take longer per submit than the watchdog allows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8080")
    ap.add_argument("--out", default="out/web")
    ap.add_argument("--quant", default="q4")
    ap.add_argument("--message", default="Hi! In one short sentence: what is WebGPU?")
    ap.add_argument("--page", default="/public/index.html")
    ap.add_argument("--file", default="", help="choose this model.safetensors in the page's file picker")
    ap.add_argument("--assets", action="store_true", help="the page carries its weights (public/weights.js)")
    ap.add_argument("--reply-min", type=float, default=40, help="minutes to wait for the reply (0: do not chat)")
    ap.add_argument("--profile", default="", help="an on-disk browser profile (normal storage quota, kept across reloads)")
    ap.add_argument("--reload", action="store_true", help="then reload the page and load again (from the device's copy)")
    ap.add_argument("--photo", default="", help="then attach this picture and send it (image mode)")
    ap.add_argument("--voice", default="", help="then attach this recording (the no-microphone path) and send it")
    ap.add_argument("--speak", action="store_true", help="then turn on spoken replies and ask a question")
    ap.add_argument("--talk", action="store_true", help="then hold the push-to-talk button (Chromium's fake microphone)")
    ap.add_argument("--live", action="store_true", help="then start Live with Chromium's fake camera and microphone")
    ap.add_argument("--hold-s", type=float, default=8, help="seconds to hold push to talk")
    ap.add_argument("--faces", action="store_true", help="first, screenshots of the voice bubble's face in each state")
    ap.add_argument("--instructions", default="", help="after loading, set these as the system prompt in Settings")
    ap.add_argument("--fake-audio", default="", help="Live: a WAV the fake microphone plays (looped)")
    ap.add_argument("--fake-video", default="", help="Live: a .y4m the fake camera shows")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    from playwright.sync_api import sync_playwright

    res = {"console_errors": [], "page_errors": []}

    def wait(page, cond, timeout_s):
        # poll with a function evaluation (Playwright's string predicates use eval, which an artifact-like CSP refuses)
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            if page.evaluate(f"() => !!({cond})"):
                return
            time.sleep(2)
        raise TimeoutError(cond)
    flags = list(SWIFTSHADER_FLAGS)
    if args.live or args.talk:  # fake media devices; the permission prompts answered yes
        flags += ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream", "--autoplay-policy=no-user-gesture-required"]
        if args.fake_audio:
            flags.append(f"--use-file-for-fake-audio-capture={args.fake_audio}")
        if args.fake_video:
            flags.append(f"--use-file-for-fake-video-capture={args.fake_video}")
    perms = {"permissions": ["camera", "microphone"]} if args.live or args.talk else {}
    with sync_playwright() as p:
        if args.profile:  # a persistent profile: IndexedDB gets the normal quota, unlike a throwaway context
            dev = {k: v for k, v in p.devices["iPhone 15 Pro"].items() if k != "default_browser_type"}
            ctx = p.chromium.launch_persistent_context(args.profile, channel="chromium", headless=True, args=flags,
                                                       **dev, **perms)
            browser = ctx
            page = ctx.pages[0] if ctx.pages else ctx.new_page()
        else:
            browser = p.chromium.launch(channel="chromium", headless=True, args=flags)
            ctx = browser.new_context(**p.devices["iPhone 15 Pro"], **perms)
            page = ctx.new_page()
        png_requests = []
        page.on("request", lambda r: ".png" in r.url and png_requests.append(r.url))
        page.on("console", lambda m: (print("[page]", m.type, m.text[:500], flush=True),
                                      m.type == "error" and res["console_errors"].append(m.text[:500])))
        page.on("pageerror", lambda e: (print("[pageerror]", e, flush=True), res["page_errors"].append(str(e)[:500])))
        page.goto(f"{args.base}{args.page}")
        wait(page, "document.getElementById('chip').textContent !== 'checking WebGPU…'", 60)
        if args.faces:  # the bubble alone, in each state (the page's CSS; no model needed)
            for st, talk in (("idle", 0), ("listening", 0), ("thinking", 0), ("speaking", 0.15), ("speaking", 0.85)):
                page.evaluate(f"""() => {{ document.body.classList.add('voice'); const d = document.getElementById('dock'); d.hidden = false;
                  const b = document.getElementById('buddy'); b.dataset.state = '{st}'; b.style.setProperty('--talk', '{talk}');
                  b.style.setProperty('--level', '{0.6 if st == "listening" else 0}'); }}""")
                page.wait_for_timeout(700)
                page.locator("footer").screenshot(path=os.path.join(args.out, f"face_{st}_{talk}.png"))
            page.evaluate("() => { document.body.classList.remove('voice'); document.getElementById('dock').hidden = true; }")
        if args.assets:
            pass
        elif args.file:
            page.set_input_files("#file", args.file)
        else:
            page.evaluate("document.getElementById('hubBtn').hidden = false")
        wait(page, "!document.getElementById('loadBtn').disabled || !document.getElementById('hubBtn').hidden", 60)
        page.screenshot(path=os.path.join(args.out, "ui_setup.png"))
        res["chip_before"] = page.inner_text("#chip")
        if not args.assets:
            page.select_option("#quant", args.quant)
        page.click("#loadBtn" if args.file or args.assets else "#hubBtn")
        t_load = time.time()
        wait(page, "!document.getElementById('input').disabled || document.querySelector('#loadMsg .err')", 30 * 60)
        res["load_message"] = page.inner_text("#loadMsg")
        res["load_s"] = round(time.time() - t_load, 1)
        res["chip_after"] = page.inner_text("#chip")
        res["ready_bubble"] = page.inner_text("#msgs")
        res["png_requests"] = len(png_requests)
        if args.instructions and not page.is_disabled("#input"):  # Settings: the user's own system prompt
            page.click("#setBtn")
            page.wait_for_timeout(400)
            page.fill("#sysPrompt", args.instructions)
            res["instructions_count"] = page.inner_text("#sysCount")
            page.screenshot(path=os.path.join(args.out, "ui_settings.png"))
            page.click("#sysSave")
            wait(page, "[...document.querySelectorAll('.msg.notice')].some((n) => n.textContent.includes('your instructions'))", 600)
            res["instructions_stored"] = page.evaluate("() => localStorage.getItem('lfm3.system')")
        if args.reload and not page.is_disabled("#input"):  # the second visit: the parts come from IndexedDB
            res["stored_parts"] = page.evaluate("""() => new Promise((ok) => { const r = indexedDB.open('lfm3');
              r.onsuccess = () => { const q = r.result.transaction('kv').objectStore('kv').getAllKeys();
                q.onsuccess = () => ok(q.result.filter((k) => String(k).startsWith('w/')).length); }; r.onerror = () => ok(-1); })""")
            png_requests.clear()
            page.reload()
            wait(page, "!document.getElementById('loadBtn').disabled", 120)
            res["second_button"] = page.inner_text("#loadBtn")
            page.click("#loadBtn")
            t2 = time.time()
            wait(page, "!document.getElementById('input').disabled || document.querySelector('#loadMsg .err')", 30 * 60)
            res["second_load_s"] = round(time.time() - t2, 1)
            res["second_load_message"] = page.inner_text("#loadMsg")
            res["second_ready_bubble"] = page.inner_text("#msgs")
            res["second_png_requests"] = len(png_requests)
            page.screenshot(path=os.path.join(args.out, "ui_second_load.png"))
        if args.photo and not page.is_disabled("#input"):  # image mode: the photo picker, the image model, the answer
            page.set_input_files("#photo", args.photo)
            wait(page, "!document.getElementById('attach').hidden", 60)
            page.screenshot(path=os.path.join(args.out, "ui_photo_attached.png"))
            page.click("#send")
            t3 = time.time()
            busy = "['…', 'Looking at the picture…'].includes(t) || t.startsWith('Loading the image model')"
            try:
                wait(page, f"(() => {{ const t = document.querySelector('.msg.assistant:last-child').textContent; return !({busy}); }})()",
                     args.reply_min * 60)
                res["photo_first_text_s"] = round(time.time() - t3, 1)
            except TimeoutError:
                res["photo_timeout_min"] = args.reply_min
            res["photo_reply"] = page.inner_text(".msg.assistant:last-child")
            res["photo_user_bubble_has_thumb"] = page.evaluate("() => !!document.querySelector('.msg.user .thumb')")
            page.screenshot(path=os.path.join(args.out, "ui_photo.png"))
        elif args.speak and not page.is_disabled("#input"):  # spoken replies: the toggle, both weight sets, the first audio
            page.click("#speakBtn")
            wait(page, "(() => { const t = document.querySelector('.msg.assistant:last-child').textContent; "
                       "return t.includes('New chat started') || !!document.querySelector('.msg .err'); })()", args.reply_min * 60)
            res["speak_toggle"] = page.inner_text(".msg.assistant:last-child")
            page.fill("#input", "Say hello in one short sentence.")
            page.click("#send")
            t3 = time.time()
            try:
                wait(page, "(globalThis.LFM3_PLAYED || 0) > 0 || !!document.querySelector('.msg .err')", args.reply_min * 60)
                res["speak_first_audio_s"] = round(time.time() - t3, 1)
            except TimeoutError:
                res["speak_timeout_min"] = args.reply_min
            res["speak_played_samples"] = page.evaluate("() => globalThis.LFM3_PLAYED || 0")
            res["speak_reply"] = page.inner_text(".msg.assistant:last-child")
            page.screenshot(path=os.path.join(args.out, "ui_speak.png"))
            if page.is_visible("#send"):
                page.click("#send")   # Stop
        elif (args.live or args.talk) and not page.is_disabled("#input"):  # push to talk, then Live
            page.screenshot(path=os.path.join(args.out, "ui_ready.png"))
            err = "!!document.querySelector('.msg .err') || ['error', 'fallback'].includes((globalThis.LFM3_LIVE || {}).state)"
            hook = "(globalThis.LFM3_LIVE || {})"

            def hold(sel, name):  # press, speak (the fake microphone loops a clip), let go; then the spoken answer
                turns = page.evaluate(f"() => {hook}.turns || 0")
                played = page.evaluate("() => globalThis.LFM3_PLAYED || 0")
                box = page.locator(sel).bounding_box()
                page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
                page.mouse.down()
                page.wait_for_timeout(1500)
                page.screenshot(path=os.path.join(args.out, f"ui_{name}_holding.png"))
                page.wait_for_timeout(int(args.hold_s * 1000) - 1500)
                page.mouse.up()
                t = time.time()
                try:
                    wait(page, f"(globalThis.LFM3_PLAYED || 0) > {played} || {err}", args.reply_min * 60)
                    res[f"{name}_first_audio_s"] = round(time.time() - t, 1)
                    page.screenshot(path=os.path.join(args.out, f"ui_{name}_speaking.png"))
                    wait(page, f"{hook}.turns > {turns} || {err}", args.reply_min * 60)
                    res[f"{name}_turn_s"] = round(time.time() - t, 1)
                except TimeoutError as e:
                    res[f"{name}_timeout"] = str(e)[:200]
                res[name] = page.evaluate(f"() => ({{ ...{hook}, played: globalThis.LFM3_PLAYED || 0 }})")
                res[f"{name}_reply_bubble"] = page.inner_text(".msg.assistant:last-child")[:600]
                page.screenshot(path=os.path.join(args.out, f"ui_{name}_after.png"))
            if args.talk:
                res["talk_button_visible"] = page.is_visible("#talkBtn")
                hold("#talkBtn", "talk")
            if args.live:
                page.click("#liveBtn")
                t3 = time.time()
                try:
                    wait(page, f"{hook}.state === 'ready' || {err}", args.reply_min * 60)
                    res["live_ready_s"] = round(time.time() - t3, 1)
                    page.screenshot(path=os.path.join(args.out, "ui_live_ready.png"))
                    hold("#orb", "live")
                except TimeoutError as e:
                    res["live_timeout"] = str(e)[:200]
                res["live_video"] = page.evaluate("() => document.getElementById('liveVideo').videoWidth + 'x' + document.getElementById('liveVideo').videoHeight")
                res["live_caption"] = page.inner_text("#liveCaption")[:400]
                page.click("#liveEnd")
                page.wait_for_timeout(500)
                res["live_closed"] = page.evaluate("() => document.getElementById('live').hidden")
            res["transcript"] = page.inner_text("#msgs")[:2000]
            res["user_bubble_has_thumb"] = page.evaluate("() => !!document.querySelector('.msg.user .thumb')")
            page.screenshot(path=os.path.join(args.out, "ui_transcript.png"))
        elif args.voice and not page.is_disabled("#input"):  # speech input through "Choose a recording"
            page.evaluate("() => { document.getElementById('voiceFileRow').hidden = false; }")
            page.set_input_files("#voiceFile", args.voice)
            wait(page, "!document.getElementById('voiceChip').hidden", 60)
            res["voice_chip"] = page.inner_text("#voiceChip")
            page.click("#send")
            t3 = time.time()
            busy = "['…', 'Listening…'].includes(t) || t.startsWith('Loading the speech model')"
            try:
                wait(page, f"(() => {{ const t = document.querySelector('.msg.assistant:last-child').textContent; return !({busy}); }})()",
                     args.reply_min * 60)
                res["voice_first_text_s"] = round(time.time() - t3, 1)
            except TimeoutError:
                res["voice_timeout_min"] = args.reply_min
            res["voice_reply"] = page.inner_text(".msg.assistant:last-child")
            res["voice_user_tag"] = page.evaluate("() => document.querySelector('.msg.user .voicetag')?.textContent || null")
            page.screenshot(path=os.path.join(args.out, "ui_voice.png"))
        elif args.reply_min > 0 and not page.is_disabled("#input"):
            page.fill("#input", args.message)
            page.click("#send")
            try:
                wait(page, "document.querySelectorAll('.msg.assistant .stats').length >= 1", args.reply_min * 60)
            except TimeoutError:  # keep what there is (a CPU-emulated GPU can be very slow)
                res["reply_timeout_min"] = args.reply_min
            res["reply"] = page.inner_text(".msg.assistant:last-child")
            page.click("#memBtn")
            page.wait_for_timeout(500)
            page.screenshot(path=os.path.join(args.out, "ui_memory.png"))
            page.click("#memClose")
            page.screenshot(path=os.path.join(args.out, "ui_chat.png"))
        browser.close()
    with open(os.path.join(args.out, "ui_test.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("UI_TEST " + json.dumps(res)[:3000], flush=True)


if __name__ == "__main__":
    main()
