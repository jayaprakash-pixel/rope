"""P5.1.4 / P5.1.1: the app in a real browser (Playwright driving the Edge that ships with Windows; nothing downloaded).
Needs `rope serve --static app/dist` on 127.0.0.1:4003 and SD-Turbo on disk.
 1. contrast of every text / background token pair in dark and light (from app/src/styles.css): >= 4.5:1
 2. keyboard-only walk: prompt, quality, model, Ctrl+Enter, progress seen, image shown; gallery opened with Enter,
    dialog closed with Esc
 3. every request goes to 127.0.0.1:4003; 0 console errors; prefers-reduced-motion -> transitions 0 ms
 4. screenshots of every screen in dark and light (bench/diffusion/out/app_*.png) for review
Usage: check_app.py [--quick]   (--quick: skip the generation)"""
import os, re, sys, time
from playwright.sync_api import sync_playwright

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
URL = "http://127.0.0.1:4003"
OUT = os.path.join(R, "bench", "diffusion", "out")
os.makedirs(OUT, exist_ok=True)
ok = True


def fail(msg):
    global ok
    ok = False
    print("  FAIL", msg)


# ---- 1. contrast ----
css = open(os.path.join(R, "app", "src", "styles.css"), encoding="utf-8").read()
def tokens(block):
    return dict(re.findall(r"--([a-z0-9-]+):\s*(#[0-9a-fA-F]{6})", block))
dark = tokens(css[css.index(":root {"):css.index("@media (prefers-color-scheme: light)")])
light = {**dark, **tokens(css[css.index(":root[data-theme='light']"):css.index("* { box-sizing")])}
def lum(h):
    c = [int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    c = [x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
def ratio(a, b):
    la, lb = sorted([lum(a), lum(b)], reverse=True)
    return (la + 0.05) / (lb + 0.05)
pairs = [("text", "bg"), ("text", "surface"), ("text", "surface-2"), ("text-muted", "bg"), ("text-muted", "surface"), ("text-muted", "surface-2"),
         ("on-primary", "primary-strong"), ("success", "surface-2"), ("warning", "surface-2"), ("danger", "surface-2"), ("primary", "surface")]
print("contrast (>= 4.5:1; 'primary on surface' is a 3:1 UI-component border/ring)")
for name, t in [("dark", dark), ("light", light)]:
    for fg, bg in pairs:
        r = ratio(t[fg], t[bg])
        need = 3.0 if (fg, bg) == ("primary", "surface") else 4.5
        if r < need: fail(f"{name}: {fg} on {bg} = {r:.2f}")
    worst = min(ratio(t[f], t[b]) for f, b in pairs if (f, b) != ("primary", "surface"))
    print(f"  {name}: worst text pair {worst:.2f}:1")

quick = "--quick" in sys.argv
with sync_playwright() as p:
    b = p.chromium.launch(channel="msedge", headless=True)
    for scheme in ["dark", "light"]:
        ctx = b.new_context(color_scheme=scheme, viewport={"width": 1280, "height": 860})
        pg = ctx.new_page()
        errors, foreign = [], []
        pg.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.on("request", lambda r: foreign.append(r.url) if not r.url.startswith(URL) and not r.url.startswith("data:") and not r.url.startswith("blob:") else None)
        for route in ["device", "create", "gallery"]:
            pg.goto(f"{URL}/#/{route}")
            pg.wait_for_timeout(1500 if route != "create" else 4000)
            pg.screenshot(path=os.path.join(OUT, f"app_{route}_{scheme}.png"))
        if scheme == "dark" and not quick:
            # ---- 2. keyboard-only walk ----
            pg.goto(f"{URL}/#/create")
            pg.wait_for_selector(".model[aria-checked='true']", timeout=60000)
            for _ in range(40):                                   # Tab until the prompt has focus
                pg.keyboard.press("Tab")
                if pg.evaluate("document.activeElement && document.activeElement.id") == "prompt": break
            else: fail("prompt never reached with Tab")
            pg.keyboard.type("a red bicycle leaning on a brick wall, photo")
            pg.keyboard.press("Tab"); pg.keyboard.press("Tab")    # quality: Balanced
            if pg.evaluate("document.activeElement.textContent") != "Balanced": fail("Tab order: quality")
            pg.keyboard.press("Enter")
            pg.keyboard.press("Shift+Tab"); pg.keyboard.press("Shift+Tab")
            if pg.evaluate("document.activeElement.id") != "prompt": fail("Shift+Tab back to prompt")
            pg.keyboard.press("Control+Enter")
            seen, t0 = set(), time.time()
            while time.time() - t0 < 180:
                if pg.query_selector(".stage img"): break
                txt = pg.evaluate("document.querySelector('.progress-row strong')?.textContent || ''")
                if txt: seen.add(txt)
                pg.wait_for_timeout(200)
            else: fail("no image within 180 s")
            print(f"  generation via keyboard: {time.time() - t0:.1f} s; stages seen: {sorted(seen)}")
            if not seen: fail("no progress text seen")
            # the image must really load and decode, not only exist as an element
            # polled with evaluate: wait_for_function needs eval, which the app's CSP (rightly) blocks
            t1 = time.time()
            while not pg.evaluate("(() => { const i = document.querySelector('.stage img'); return !!(i && i.complete && i.naturalWidth > 0) })()"):
                if time.time() - t1 > 15: fail("finished image did not load"); break
                pg.wait_for_timeout(100)
            pg.wait_for_timeout(400)
            pg.screenshot(path=os.path.join(OUT, f"app_done_{scheme}.png"))
            # gallery: keyboard to the nav link, Enter, first tile with Enter, Esc closes the dialog
            pg.goto(f"{URL}/#/create")
            pg.focus("a[href='#/gallery']"); pg.keyboard.press("Enter")
            pg.wait_for_selector(".tile", timeout=10000)
            pg.focus(".tile"); pg.keyboard.press("Enter")
            pg.wait_for_timeout(500)
            if not pg.evaluate("document.querySelector('dialog').open"): fail("gallery dialog did not open with Enter")
            pg.screenshot(path=os.path.join(OUT, "app_lightbox_dark.png"))
            pg.keyboard.press("Escape"); pg.wait_for_timeout(300)
            if pg.evaluate("document.querySelector('dialog').open"): fail("Esc did not close the dialog")
        print(f"  {scheme}: console errors {len(errors)}, requests outside 127.0.0.1:4003 {len(foreign)}")
        for e in errors[:5]: fail(f"console: {e[:160]}")
        for u in foreign[:5]: fail(f"foreign request: {u}")
        ctx.close()
    # ---- 3. reduced motion ----
    ctx = b.new_context(reduced_motion="reduce")
    pg = ctx.new_page(); pg.goto(f"{URL}/#/create"); pg.wait_for_timeout(1500)
    d = pg.evaluate("getComputedStyle(document.querySelector('.btn')).transitionDuration")
    print(f"  reduced motion: .btn transition-duration {d}")
    if any(float(x.strip().rstrip('s')) > 0 for x in d.split(",")): fail("transitions not removed")
    ctx.close(); b.close()
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
