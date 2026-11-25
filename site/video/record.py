"""Record the demo scenes (headed Edge, 1920x1080) from the live site and the local app. Each scene is its own video;
cuts.json holds where each scene's useful part starts and how long it is. Run after narrate.py."""
import json, os, shutil, sys, time
from playwright.sync_api import sync_playwright

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "build", "demo")
SITE = "https://rope-ai.vercel.app"
APP = "http://127.0.0.1:4003"
MINE = r"C:\Models\sd_xl_turbo_1.0_fp16.safetensors"   # a neutral path for the "your own model" scene (a hard link, real file name)
D = json.load(open(os.path.join(OUT, "durations.json")))
VP = {"width": 1920, "height": 1080}
ONLY = sys.argv[1:]                                     # scene names to record again; empty = all
want = lambda n: not ONLY or n in ONLY
cuts = json.load(open(os.path.join(OUT, "cuts.json"))) if ONLY else {}

def scene(p, name):
    d = os.path.join(OUT, "rec_" + name)
    shutil.rmtree(d, ignore_errors=True)
    ctx = p.chromium.launch(channel="msedge", headless=False, args=["--window-size=1920,1080", "--hide-scrollbars"]).new_context(
        viewport=VP, record_video_dir=d, record_video_size=VP, color_scheme="dark")
    return ctx, d, time.time()

def finish(ctx, d, name, starts):
    page = ctx.pages[0]
    vid = page.video.path()
    ctx.close(); ctx.browser.close()
    shutil.move(vid, os.path.join(OUT, f"clip_{name}.webm")); shutil.rmtree(d, ignore_errors=True)
    cuts[name] = starts

def glide(pg, y, ms=1600):
    pg.evaluate(f"""new Promise(r => {{ const s = scrollY, t0 = performance.now();
      const f = t => {{ const k = Math.min(1, (t - t0) / {ms}); const e = k < .5 ? 4*k*k*k : 1 - Math.pow(-2*k + 2, 3) / 2;
        scrollTo(0, s + ({y} - s) * e); k < 1 ? requestAnimationFrame(f) : r(); }}; requestAnimationFrame(f); }})""")

with sync_playwright() as p:
    # 1. hero: the rope, the mouse drifts across it
  if want("hero"):
    ctx, d, t0 = scene(p, "hero"); pg = ctx.new_page()
    pg.goto(SITE + "/", wait_until="networkidle"); pg.wait_for_timeout(2200)
    s = time.time() - t0
    for i in range(60):
        pg.mouse.move(400 + i * 20, 300 + i * 6); pg.wait_for_timeout(int((D["hero"] + 1.4) * 1000 / 60))
    finish(ctx, d, "hero", {"hero": [s, D["hero"] + 1.4]})

    # 2 + 3. plan, then make the picture (the app on this laptop, GPU)
  if want("app"):
    ctx, d, t0 = scene(p, "app"); pg = ctx.new_page()
    pg.goto(APP + "/#/create"); pg.wait_for_selector(".model[aria-checked='true']", timeout=90000)
    pg.click("text=Best"); pg.wait_for_timeout(1500)
    pg.click(".model:has-text('SD-Turbo'):not(:has-text('SDXL'))"); pg.wait_for_timeout(1200)
    s_plan = time.time() - t0
    pg.click("textarea")
    pg.keyboard.type("a lighthouse on a cliff at dusk, violet sky, crashing waves, cinematic photo", delay=55)
    pg.wait_for_timeout(int(max(0.5, D["plan"] + 1.2 - (time.time() - t0 - s_plan)) * 1000))
    s_make = time.time() - t0
    pg.click("button:has-text('Generate')")
    pg.wait_for_selector(".progress-wrap .badge.ok", timeout=300000)
    pg.wait_for_timeout(3200)
    l_make = min(18.0, max(D["make"] + 1.2, time.time() - t0 - s_make))
    # the video clock drifts from the wall clock: cut the make scene from the END of the clip (negative start)
    finish(ctx, d, "app", {"plan": [s_plan, D["plan"] + 1.2], "make": [-l_make, l_make]})

    # 4. laptops: the planner on the site, switching laptops
  if want("laptops"):
    ctx, d, t0 = scene(p, "laptops"); pg = ctx.new_page()
    pg.goto(SITE + "/", wait_until="networkidle"); pg.wait_for_timeout(800)
    y = pg.evaluate("document.getElementById('planner').getBoundingClientRect().top + scrollY - 70")
    pg.evaluate(f"scrollTo(0, {y} - 900)"); pg.wait_for_timeout(1500)
    s = time.time() - t0
    glide(pg, y, 1500); pg.wait_for_timeout(900)
    for prof in ["4 GB laptop", "16 GB laptop", "GTX 1650"]:
        pg.click(f"button.profile:has-text('{prof}')"); pg.wait_for_timeout(int((D["laptops"] - 1.2) * 1000 / 3))
    pg.wait_for_timeout(900)
    finish(ctx, d, "laptops", {"laptops": [s, D["laptops"] + 1.4]})

    # 5. your own model file
  if want("mine"):
    ctx, d, t0 = scene(p, "mine"); pg = ctx.new_page()
    pg.goto(APP + "/#/create"); pg.wait_for_selector("text=Your own model files", timeout=90000); pg.wait_for_timeout(800)
    pg.evaluate("document.querySelector('[aria-labelledby=mine-label]').scrollIntoView({block: 'center'})"); pg.wait_for_timeout(1200)
    s = time.time() - t0
    pg.click("input[aria-labelledby=mine-label]")
    pg.keyboard.type(MINE, delay=45)
    pg.click("button:has-text('Add')")
    pg.wait_for_selector(".mine-row", timeout=120000)
    pg.evaluate("document.querySelector('.mine-row').scrollIntoView({block: 'center', behavior: 'smooth'})")
    pg.wait_for_timeout(2500)
    pg.wait_for_timeout(int(max(0, D["mine"] + 2.0 - (time.time() - t0 - s)) * 1000))
    finish(ctx, d, "mine", {"mine": [s, max(D["mine"] + 1.4, time.time() - t0 - s)]})

    # 6. end: the final call to action
  if want("end"):
    ctx, d, t0 = scene(p, "end"); pg = ctx.new_page()
    pg.goto(SITE + "/", wait_until="networkidle"); pg.wait_for_timeout(600)
    y = pg.evaluate("document.querySelector('.final').getBoundingClientRect().top + scrollY - 140")
    pg.evaluate(f"scrollTo(0, {y} - 700)"); pg.wait_for_timeout(1500)
    s = time.time() - t0
    glide(pg, y, 1400); pg.wait_for_timeout(700)
    b = pg.query_selector(".final .btn-primary").bounding_box()
    pg.mouse.move(b["x"] + b["width"] / 2, b["y"] + b["height"] / 2, steps=30)
    pg.wait_for_timeout(int(D["end"] * 1000))
    finish(ctx, d, "end", {"end": [s, D["end"] + 1.8]})

json.dump(cuts, open(os.path.join(OUT, "cuts.json"), "w"), indent=1)
print(json.dumps(cuts, indent=1))
