"""P2.3.4 in the app (headed Edge). Needs `rope serve --port 4013 --static app/dist --ram-gib 1.6 --no-gpu` (a simulated small laptop):
  - the size picker of SDXL-Turbo offers sizes up to its ceiling on this laptop (832), not up to the model's 1024
  - the plan card says "This laptop can go up to 832x832" and gives the reason the next size is refused (memory, in plain words)
  - picking a size refreshes the plan card for that size (memory and time change)
  - a size that fits is generated end to end (SDXL-Turbo at 640, 1 step: Fast quality)
  0 console errors; screenshot bench/diffusion/out/app_size.png"""
import os, sys, time
from playwright.sync_api import sync_playwright

URL = "http://127.0.0.1:4013"
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
OUT = os.path.join(R, "bench", "diffusion", "out")
os.makedirs(OUT, exist_ok=True)
ok = True


def fail(m):
    global ok
    ok = False
    print("  FAIL", m)


with sync_playwright() as p:
    b = p.chromium.launch(channel="msedge", headless=False)
    pg = b.new_page(viewport={"width": 1180, "height": 1000})
    errors = []
    pg.on("console", lambda m: errors.append(m.text[:160]) if m.type == "error" else None)
    pg.on("pageerror", lambda e: errors.append(str(e)[:160]))
    pg.goto(f"{URL}/#/create")
    pg.wait_for_selector(".model[aria-checked='true']", timeout=60000)
    pg.click(".model:has-text('SDXL-Turbo')")
    pg.wait_for_selector("#size-label", timeout=20000)
    sizes = pg.evaluate("[...document.querySelectorAll('[aria-labelledby=size-label] button')].map(b => Number(b.textContent))")
    print("size picker:", sizes)
    if not sizes or max(sizes) != 832: fail(f"the largest size offered should be this laptop's ceiling (832), got {max(sizes) if sizes else None}")
    t1 = time.time()
    while time.time() - t1 < 15 and "can go up to" not in pg.evaluate("document.querySelector('aside.plan').innerText"): pg.wait_for_timeout(200)
    card = pg.evaluate("document.querySelector('aside.plan').innerText")
    print("plan card:", " | ".join(l for l in card.split("\n") if "up to" in l or "needs" in l or "Memory" in l)[:400])
    if "can go up to 832" not in card: fail("the plan card does not name the ceiling")
    if "needs about" not in card or "kept for Windows" not in card: fail("the plan card does not say why the next size is refused")
    mem0 = pg.evaluate("[...document.querySelectorAll('aside.plan dt')].findIndex(d => d.textContent.startsWith('Memory'))")
    # pick 640: the card refreshes
    pg.click("[aria-labelledby=size-label] button:has-text('640')")
    pg.wait_for_timeout(2500)
    size_line = pg.evaluate("(() => { const d = [...document.querySelectorAll('aside.plan dt')].find(d => d.textContent === 'Size'); return d && d.nextElementSibling.textContent })()")
    print("size after picking 640:", size_line)
    if "640" not in (size_line or ""): fail("the plan card did not refresh for 640")
    pg.screenshot(path=os.path.join(OUT, "app_size.png"))
    # generate it (Fast, 1 step)
    pg.click("text=Fast"); pg.wait_for_timeout(1500)
    pg.click(".model:has-text('SDXL-Turbo')"); pg.wait_for_timeout(800)
    pg.click("[aria-labelledby=size-label] button:has-text('640')"); pg.wait_for_timeout(1500)
    pg.fill("#prompt", "a red bicycle leaning on a brick wall, photo"); pg.click("button.btn.primary")
    t0 = time.time()
    while time.time() - t0 < 300 and not pg.evaluate("!!document.querySelector('.progress-wrap .badge.ok')"): pg.wait_for_timeout(300)
    pg.wait_for_timeout(2500)                                  # the particles develop into the image first
    wh = pg.evaluate("(() => { const i = [...document.querySelectorAll('.stage img')].find(i => !i.classList.contains('preview')); return i ? [i.naturalWidth, i.naturalHeight] : null })()")
    print(f"generated at {time.time() - t0:.0f} s, image {wh}")
    if wh != [640, 640]: fail(f"the image should be 640x640, got {wh}")
    print(f"console errors {len(errors)}")
    for e in errors[:5]: fail("console: " + e)
    b.close()
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
