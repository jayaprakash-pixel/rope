"""Measure how smoothly the hero film renders (headed Edge, real GPU): frames per second and the share of slow
frames (> 25 ms, a visible stutter) at the intro and at three scroll positions. Also with the CPU slowed 4x, like a
weak laptop. Usage: python perf_hero.py http://127.0.0.1:4182"""
import sys
from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:4182"
JS = """() => new Promise(r => { const t = []; let last = performance.now(); const end = last + 4000;
  const f = now => { t.push(now - last); last = now; now < end ? requestAnimationFrame(f) : r(t) }; requestAnimationFrame(f) })"""

def measure(pg, label):
    t = pg.evaluate(JS)[3:]
    fps = 1000 * len(t) / sum(t)
    slow = 100 * sum(1 for x in t if x > 25) / len(t)
    print(f"  {label:10} {fps:5.1f} fps   slow frames {slow:4.1f}%   worst {max(t):5.1f} ms")

with sync_playwright() as p:
    b = p.chromium.launch(channel="msedge", headless=False)
    for throttle in (1, 4):
        pg = b.new_page(viewport={"width": 1440, "height": 900}, color_scheme="dark")
        if throttle > 1:
            cdp = pg.context.new_cdp_session(pg); cdp.send("Emulation.setCPUThrottlingRate", {"rate": throttle})
        print(f"CPU x{throttle}:")
        pg.goto(URL + "/", wait_until="domcontentloaded"); pg.wait_for_timeout(1500)
        measure(pg, "intro")
        for f in (0.36, 0.62, 0.96):
            pg.evaluate(f"scrollTo(0, 900*3.3*{f})"); pg.wait_for_timeout(1500)
            measure(pg, f"scroll {f}")
        pg.close()
    b.close()
