"""Walk the built site like a visitor (headless Edge): no console errors, WebGL canvas drawn, every section visible after
scrolling, the planner switches laptops, the FAQ opens, the docs page renders, light theme, phone width without sideways
scroll, reduced motion. Screenshots go to ../build/site-shots/.  Usage: python check_site.py http://127.0.0.1:4180"""
import os, sys
from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:4180"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "build", "site-shots")
os.makedirs(OUT, exist_ok=True)
fails = []
def check(ok, what):
    print(("PASS " if ok else "FAIL ") + what)
    if not ok: fails.append(what)

def walk(pg):
    h = pg.evaluate("document.body.scrollHeight")
    y = 0
    while y < h:
        pg.mouse.wheel(0, 500); pg.wait_for_timeout(160); y += 500
        h = pg.evaluate("document.body.scrollHeight")
    pg.wait_for_timeout(1200)

with sync_playwright() as p:
    b = p.chromium.launch(channel="msedge", args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
    for name, vp, scheme, motion in [("desktop-dark", {"width": 1440, "height": 900}, "dark", "no-preference"),
                                      ("phone", {"width": 390, "height": 844}, "dark", "no-preference"),
                                      ("reduced", {"width": 1280, "height": 800}, "dark", "reduce")]:
        ctx = b.new_context(viewport=vp, color_scheme=scheme, reduced_motion=motion)
        pg = ctx.new_page()
        errs = []
        pg.on("console", lambda m: errs.append(m.text[:200]) if m.type == "error" else None)
        pg.on("pageerror", lambda e: errs.append(str(e)[:200]))
        pg.goto(URL + "/", wait_until="networkidle")
        pg.wait_for_timeout(2500)
        canvas = pg.query_selector(".rope-scene canvas")
        check(canvas is not None, f"{name}: the WebGL rope canvas is there")
        pg.screenshot(path=os.path.join(OUT, f"{name}-hero.png"))
        walk(pg)
        hidden = pg.evaluate("[...document.querySelectorAll('.reveal')].filter(e => getComputedStyle(e).opacity < 0.9).length")
        check(hidden == 0, f"{name}: every section is visible after scrolling ({hidden} hidden)")
        over = pg.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth")
        check(over <= 1, f"{name}: no sideways scroll ({over}px)")
        if name == "desktop-dark":
            pg.click("button.profile:has-text('4 GB laptop')"); pg.wait_for_timeout(900)
            t4 = pg.inner_text(".plan-list .plan-row >> nth=0")
            pg.click("button.profile:has-text('GTX 1650')"); pg.wait_for_timeout(900)
            tg = pg.inner_text(".plan-list .plan-row >> nth=0")
            check(t4 != tg and "GPU" in tg, f"planner switches laptops: 4 GB first row [{t4.splitlines()[-1]}], GPU first row [{tg.splitlines()[-1]}]")
            pg.click("#faq button >> nth=2"); pg.wait_for_timeout(600)
            check("processor" in pg.inner_text("#faq"), "FAQ answer opens")
            dl = pg.get_attribute("a.btn-primary", "href")
            check(dl.endswith("/releases/latest/download/Rope-Setup.exe"), f"download link: {dl}")
            pg.screenshot(path=os.path.join(OUT, f"{name}-full.png"), full_page=True)
            pg.click("button[aria-label^='Switch to light']"); pg.wait_for_timeout(500)
            pg.screenshot(path=os.path.join(OUT, "desktop-light-full.png"), full_page=True)
            pg.goto(URL + "/docs.html", wait_until="networkidle"); pg.wait_for_timeout(800)
            check("Using" in pg.inner_text("h1") and pg.query_selector("#uninstall") is not None, "docs page renders with all sections")
            pg.screenshot(path=os.path.join(OUT, "docs.png"), full_page=True)
        if name == "phone":
            pg.screenshot(path=os.path.join(OUT, "phone-full.png"), full_page=True)
        check(not errs, f"{name}: no console errors {errs[:3]}")
        ctx.close()
    b.close()
print("RESULT", "PASS" if not fails else f"FAIL ({len(fails)})")
sys.exit(1 if fails else 0)
