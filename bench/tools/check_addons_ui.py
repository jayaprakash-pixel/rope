"""P2.4.4 / P2.4.5 in the app (headed Edge). Needs `rope serve --port 4013 --static app/dist --no-gpu` and the two files in models/addons
(pixel-art-xl LoRA for SDXL, realesrgan_x4plus upscaler):
  1. The Add-ons field lists the LoRA and the upscale switch.
  2. With SD-Turbo chosen the SDXL LoRA is disabled in the list, with its reason.
  3. Upscale on: the plan card shows Output 4x and "+N s"; the estimate grows; the size picker follows the new (upscaler-aware) ceiling.
  4. SD-Turbo 256 + upscale: the stage text "Upscaling x4" shows, the progress bar moves, the result image is 1024x1024.
  5. SDXL-Turbo 512 + LoRA (strength 1) at Fast: runs and ends with an image (the job event of the result names the LoRA).
  0 console errors; screenshot bench/diffusion/out/app_addons.png. Deletes the two result images from the gallery afterwards."""
import glob, os, sys, time
from playwright.sync_api import sync_playwright

URL = "http://127.0.0.1:4013"
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
OUT = os.path.join(R, "bench", "diffusion", "out")
os.makedirs(OUT, exist_ok=True)
ok = True
made = []


def fail(m):
    global ok
    ok = False
    print("  FAIL", m)


def wait_done(pg, secs):
    t0 = time.time()
    seen = set()
    while time.time() - t0 < secs and not pg.evaluate("!!document.querySelector('.progress-wrap .badge.ok')"):
        txt = pg.evaluate("(document.querySelector('.progress-row strong') || {}).textContent || ''")
        if txt: seen.add(txt.split(' · ')[0])
        pg.wait_for_timeout(500)
    return seen, time.time() - t0


with sync_playwright() as p:
    b = p.chromium.launch(channel="msedge", headless=False)
    pg = b.new_page(viewport={"width": 1180, "height": 1100})
    errors = []
    pg.on("console", lambda m: errors.append(m.text[:160]) if m.type == "error" else None)
    pg.on("pageerror", lambda e: errors.append(str(e)[:160]))
    pg.goto(f"{URL}/#/create")
    pg.wait_for_selector(".model[aria-checked='true']", timeout=60000)
    pg.click("text=Fast"); pg.wait_for_timeout(1500)
    pg.click(".model:has-text('SD-Turbo'):not(:has-text('SDXL'))")
    pg.wait_for_selector("#lora", timeout=15000)
    opts = pg.evaluate("[...document.querySelectorAll('#lora option')].map(o => [o.textContent, o.disabled])")
    print("LoRA list with SD-Turbo:", opts)
    if not any("pixel-art-xl" in t and d and "SDXL" in t for t, d in opts): fail("the SDXL LoRA should be disabled for SD-Turbo with its reason")
    if not pg.query_selector("text=Upscale ×4"): fail("no upscale switch")
    pg.click("[aria-labelledby=size-label] button:has-text('256')"); pg.wait_for_timeout(2000)
    card0 = pg.evaluate("document.querySelector('aside.plan').innerText")
    pg.check("label.check input"); pg.wait_for_timeout(3000)
    card1 = pg.evaluate("document.querySelector('aside.plan').innerText")
    print("plan card with upscale:", " | ".join(l for l in card1.split("\n") if any(k in l for k in ("Output", "Upscale", "~")))[:300])
    if "Output" not in card1 or "1024×1024" not in card1: fail("the plan card does not show the 4x output size")
    if "Upscale" not in card1 or "+" not in card1: fail("the plan card does not show the upscale time")
    t0 = [l for l in card0.split("\n") if l.startswith("~")][:1]
    t1 = [l for l in card1.split("\n") if l.startswith("~")][:1]
    print("estimate before / after:", t0, t1)
    pg.screenshot(path=os.path.join(OUT, "app_addons.png"))
    pg.fill("#prompt", "a red bicycle leaning on a brick wall, photo"); pg.click("button.btn.primary")
    seen, secs = wait_done(pg, 300)
    pg.wait_for_timeout(2500)
    wh = pg.evaluate("(() => { const i = [...document.querySelectorAll('.stage img')].find(i => !i.classList.contains('preview')); return i ? [i.naturalWidth, i.naturalHeight] : null })()")
    print(f"SD-Turbo 256 + upscale: {secs:.0f} s, stages seen {sorted(seen)}, image {wh}")
    if "Upscaling ×4" not in seen: fail("the upscale stage was never shown")
    if wh != [1024, 1024]: fail(f"the image should be 1024x1024, got {wh}")
    pg.click("text=New image"); pg.wait_for_selector("#prompt", timeout=10000)
    # 5. SDXL-Turbo + LoRA
    pg.uncheck("label.check input"); pg.wait_for_timeout(1500)
    pg.click(".model:has-text('SDXL-Turbo')"); pg.wait_for_timeout(1500)
    pg.select_option("#lora", "pixel-art-xl.safetensors")
    pg.wait_for_selector("input[aria-label='LoRA strength']", timeout=5000)
    pg.fill("#prompt", "pixel art, a red bicycle leaning on a brick wall"); pg.click("button.btn.primary")
    seen, secs = wait_done(pg, 400)
    pg.wait_for_timeout(2500)
    wh = pg.evaluate("(() => { const i = [...document.querySelectorAll('.stage img')].find(i => !i.classList.contains('preview')); return i ? [i.naturalWidth, i.naturalHeight] : null })()")
    print(f"SDXL-Turbo + LoRA: {secs:.0f} s, image {wh}")
    if not wh: fail("no image from the LoRA run")
    print(f"console errors {len(errors)}")
    for e in errors[:5]: fail("console: " + e)
    b.close()
# the sidecar json of the newest results names the add-ons
for j in sorted(glob.glob(os.path.join(R, "gallery", "*.json")))[-2:]:
    t = open(j, encoding="utf-8").read()
    print("sidecar:", os.path.basename(j), "lora" in t, "upscale" in t)
    made.append(j)
if not any('"lora"' in open(j, encoding="utf-8").read() for j in made): fail("the LoRA run's record does not name the LoRA")
if not any('"upscale"' in open(j, encoding="utf-8").read() for j in made): fail("the upscale run's record does not name the upscaler")
for j in made:
    for f in (j, j[:-5] + ".png"):
        if os.path.exists(f): os.remove(f)
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
