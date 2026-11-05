# SPDX-License-Identifier: Apache-2.0. Copyright 2026 Toaster Club LLC.
"""The decision page (decide.html) in a real Chromium on a phone-sized screen: load d1-Omni-2B, run every example,
and check the answers of the release's own test cases against its runner (tools/d1_ref.py's output).

    python tools/decide_test.py --ref work/d1_ref.json [--base https://lfm.toasterclub.com] [--out out/decide]

It asks for a hardware WebGPU adapter first (Vulkan) and falls back to SwiftShader. Writes result_decide.json and
screenshots; prints DECIDE_TEST with the largest difference from the reference.
"""

from __future__ import annotations

import argparse
import json
import os
import time

GPU_FLAGS = ["--enable-unsafe-webgpu", "--enable-features=Vulkan,WebGPU", "--use-angle=vulkan", "--enable-vulkan",
             "--ignore-gpu-blocklist", "--disable-vulkan-surface", "--no-sandbox"]
SWIFTSHADER_FLAGS = ["--enable-unsafe-webgpu", "--use-webgpu-adapter=swiftshader", "--enable-unsafe-swiftshader", "--no-sandbox",
                     "--disable-gpu-watchdog"]
# example label on the page -> the reference case it reproduces
REF = {"Support ticket": "text", "Order (JSON)": "json", "Photo (sample)": "image", "Recording (sample)": "audio"}


def compare(answers, case):
    """Largest absolute difference between the page's probabilities and the reference's, per question."""
    out = {}
    for i, (name, q) in enumerate(case["questions"].items()):
        if name not in answers:
            continue
        a, want = answers[name], case["probabilities"][i]
        if a["type"] == "noul":
            mine = [a["noul"], 1 - a["noul"]]
        elif "choices" in a:  # several: each option's P(yes) after the release's "multiple" table, as its answer gives them
            mine = list(a["probabilities"].values())
            want = list(case["answers"][name]["probabilities"].values())
        else:
            mine = list(a["probabilities"].values())
        out[name] = round(max(abs(x - y) for x, y in zip(mine, want)), 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="https://lfm.toasterclub.com")
    ap.add_argument("--ref", default="work/d1_ref.json")
    ap.add_argument("--out", default="out/decide")
    ap.add_argument("--minutes", type=float, default=30)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    ref = {c["name"]: c for c in json.load(open(args.ref))["cases"]}
    from playwright.sync_api import sync_playwright

    res = {"base": args.base, "examples": []}
    with sync_playwright() as p:
        for mode, flags in (("gpu", GPU_FLAGS), ("swiftshader", SWIFTSHADER_FLAGS)):
            browser = p.chromium.launch(headless=True, args=flags)
            ctx = browser.new_context(viewport={"width": 390, "height": 844}, device_scale_factor=2, is_mobile=True, has_touch=True)
            page = ctx.new_page()
            logs = []
            page.on("console", lambda m: logs.append(f"{m.type}: {m.text}"[:400]))
            page.goto(f"{args.base}/decide.html", wait_until="load")
            info = page.evaluate("""async () => { const a = await navigator.gpu?.requestAdapter();
                return a ? { vendor: a.info?.vendor, arch: a.info?.architecture, desc: a.info?.description, fallback: a.isFallbackAdapter } : null; }""")
            if info and (mode == "swiftshader" or not info.get("fallback")):
                # the adapter Chromium gave, whatever the flags asked for
                res["adapter"] = info
                res["mode"] = "swiftshader" if "swiftshader" in (info.get("arch") or "") else "gpu"
                break
            print("no hardware adapter with", mode, info, flush=True)
            browser.close()
        print("ADAPTER", json.dumps(res.get("adapter")), res.get("mode"), flush=True)
        page.screenshot(path=f"{args.out}/0_setup.png", full_page=True)
        t0 = time.time()
        page.click("#loadBtn")
        page.wait_for_function("globalThis.D1_PAGE?.d1 || document.querySelector('#loadMsg.err')", timeout=args.minutes * 60e3)
        err = page.evaluate("document.querySelector('#loadMsg.err')?.textContent || ''")
        if err:
            raise SystemExit(f"load failed: {err}")
        res["load_s"] = round(time.time() - t0, 1)
        res["ready"] = page.inner_text("#runMsg")
        print("LOADED", res["load_s"], "s", res["ready"], flush=True)
        labels = page.evaluate("D1_PAGE.EXAMPLES.map((e) => e.label)")
        worst = 0.0
        for i, label in enumerate(labels):
            page.evaluate(f"D1_PAGE.useExample(D1_PAGE.EXAMPLES[{i}])")
            t1 = time.time()
            page.click("#decide")
            page.wait_for_function("D1_PAGE.last !== null && document.querySelector('#decide').textContent === 'Decide' || document.querySelector('#runMsg.err')",
                                   timeout=args.minutes * 60e3)
            ex = {"label": label, "s": round(time.time() - t1, 2), "msg": page.inner_text("#runMsg")}
            last = page.evaluate("D1_PAGE.last && D1_PAGE.last.answers")
            if last is None:
                ex["error"] = ex["msg"]
            else:
                ex["answers"] = last
                if label in REF:
                    ex["diff"] = compare(last, ref[REF[label]])
                    worst = max([worst, *ex["diff"].values()])
                    ex["reference_answers"] = ref[REF[label]]["answers"]
            res["examples"].append(ex)
            print("EXAMPLE", json.dumps(ex)[:900], flush=True)
            page.screenshot(path=f"{args.out}/{i + 1}_{label.split()[0].lower()}.png", full_page=True)
        res["max_diff"] = worst
        res["console_errors"] = [l for l in logs if l.startswith("error")][:20]
        browser.close()
    with open(f"{args.out}/result_decide.json", "w") as f:
        json.dump(res, f, indent=1)
    print("DECIDE_TEST", json.dumps({"max_diff": res["max_diff"], "mode": res.get("mode"), "load_s": res.get("load_s"),
                                     "errors": sum("error" in e for e in res["examples"])}), flush=True)


if __name__ == "__main__":
    main()
