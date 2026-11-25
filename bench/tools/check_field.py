"""P5.1.5: the diffusion field's frame rate and its cost to generation, in a real (headed) Edge so WebGL uses the GPU.
  check_field.py gpu   (rope serve from the CUDA build on :4003): fps while a GPU job denoises (target 30) and while the
                        finished image develops (target 60, pass >= 50)
  check_field.py cpu   (rope serve from the CPU build on :4003): fps while a CPU-only job runs (pass <= 5) and the job time
                        with the app open vs no app (6 interleaved rounds, one warm browser, pass: at most 3% longer)
The app counts rendered frames in window.__ropeFrames."""
import json, statistics, sys, time, urllib.request
from playwright.sync_api import sync_playwright

URL = "http://127.0.0.1:4003"
MODE = sys.argv[1] if len(sys.argv) > 1 else "gpu"
PROMPT = "a red bicycle leaning on a brick wall, photo"
ok = True


def api_job():
    """One SD-Turbo job through the API with no page open; returns the server's job seconds."""
    req = urllib.request.Request(f"{URL}/api/jobs", data=json.dumps({"model": "sd-turbo", "prompt": PROMPT, "quality": "balanced", "seed": 1}).encode(),
                                 headers={"Content-Type": "application/json", "Origin": URL}, method="POST")
    jid = json.load(urllib.request.urlopen(req))["id"]
    for line in urllib.request.urlopen(f"{URL}/api/jobs/{jid}/events"):
        line = line.decode().strip()
        if line.startswith("data: "):
            e = json.loads(line[6:])
            if e["type"] == "done": return e["seconds"]
            if e["type"] == "error": raise RuntimeError(e["message"])


def ui_job(pg):
    """One SD-Turbo job through the app; returns (job seconds, fps per stage)."""
    pg.goto(f"{URL}/#/create"); pg.reload()          # same URL after a job: reload, or the done screen stays
    pg.wait_for_selector(".model[aria-checked='true']", timeout=60000)
    pg.fill("#prompt", PROMPT); pg.click("button.btn.primary")
    samples, t0 = [], time.time()
    while time.time() - t0 < 300:
        st = pg.evaluate("(() => ({f: window.__ropeFrames || 0, s: document.querySelector('.progress-row strong')?.textContent || '', img: !!document.querySelector('.stage img'), dev: !!document.querySelector('.stage canvas')}))()")
        samples.append((time.time(), st))
        if st["img"] and not st["dev"]: break
        pg.wait_for_timeout(100)
    # seconds of the job, as the server measured it
    secs = float(pg.evaluate("document.querySelector('.progress-wrap .badge.ok')?.textContent || '0'").split()[0] or 0)
    def fps(pred):
        win = [(t, s["f"]) for t, s in samples if pred(s)]
        return (win[-1][1] - win[0][1]) / (win[-1][0] - win[0][0]) if len(win) > 2 and win[-1][0] > win[0][0] else float("nan")
    return secs, fps(lambda s: s["s"].startswith("Shaping")), fps(lambda s: not s["s"] and s["dev"])


with sync_playwright() as p:
    b = p.chromium.launch(channel="msedge", headless=False, args=["--window-size=1280,900"])
    pg = b.new_page(viewport={"width": 1280, "height": 860})
    renderer = pg.evaluate("(() => { const g = document.createElement('canvas').getContext('webgl'); const d = g.getExtension('WEBGL_debug_renderer_info'); return d ? g.getParameter(d.UNMASKED_RENDERER_WEBGL) : 'unknown' })()")
    print(f"WebGL renderer: {renderer}")
    if MODE == "gpu":
        secs, f_denoise, f_develop = ui_job(pg)
        print(f"GPU job {secs:.0f} s: field {f_denoise:.1f} fps while denoising (target 30), {f_develop:.1f} fps while developing (target 60)")
        if not f_develop >= 50: ok = False; print("  FAIL develop fps < 50")
    else:
        # fair method (found while measuring): one warm browser for all runs; runs interleaved; while a job runs the
        # harness makes no browser calls (they cost CPU too), completion is read from the server's gallery
        def gallery(): return json.load(urllib.request.urlopen(f"{URL}/api/gallery"))
        pg.goto("about:blank"); time.sleep(20)
        app, none, fps = [], [], []
        for r in range(6):
            none.append(api_job()); time.sleep(3)
            pg.goto(f"{URL}/#/create"); pg.reload(); pg.wait_for_selector(".model[aria-checked='true']", timeout=60000); pg.wait_for_timeout(3000)
            f0, n0 = pg.evaluate("window.__ropeFrames || 0"), len(gallery())
            pg.fill("#prompt", PROMPT); pg.click("button.btn.primary")
            t0 = time.time()
            while len(gallery()) == n0 and time.time() - t0 < 300: time.sleep(1)
            secs = gallery()[0]["seconds"]
            fps.append((pg.evaluate("window.__ropeFrames || 0") - f0) / secs)   # frames drawn during the job
            app.append(secs); pg.wait_for_timeout(1500); pg.goto("about:blank"); time.sleep(3)
            print(f"  round {r}: no app {none[-1]:.1f} s, app {secs:.1f} s, field {fps[-1]:.2f} fps", flush=True)
        a, w = statistics.mean(app), statistics.mean(none)
        print(f"CPU-only job: app open {a:.2f} s vs no app {w:.2f} s: {100 * (a - w) / w:+.1f}% (run spread {100 * statistics.stdev(none) / w:.1f}%); field max {max(fps):.2f} fps")
        if not max(fps) <= 5: ok = False; print("  FAIL fps > 5 during a CPU-only job")
        if (a - w) / w > 0.03: ok = False; print("  FAIL job time more than 3% longer with the app open")
    if b: b.close()
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
