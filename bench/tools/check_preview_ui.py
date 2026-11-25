"""P5.1.6 / P5.1.7 in the browser (headed Edge). Needs `rope serve --port 4013 --static app/dist` from the CPU build:
a CPU-only SD-Turbo job at quality Best (4 steps, about 30 s) is long enough for previews to fit their time budget.
  preview mode:  a preview image appears in the stage before the job ends, it really loads (naturalWidth > 0), the stage then
                 shows the final image, 0 console errors
  steer mode (P5.1.7, `check_preview_ui.py steer`): Pause from the keyboard-reachable button, the heading says "Paused before step N"
                 and the clock stands still, change the prompt in the field and Apply (note "Sent"), Resume, the job finishes with a
                 "Prompt changed before step N" note; the final image shows; 0 console errors
Polling uses evaluate (the app's CSP blocks wait_for_function)."""
import os, sys, time
from playwright.sync_api import sync_playwright

URL = "http://127.0.0.1:4013"
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
OUT = os.path.join(R, "bench", "diffusion", "out")
os.makedirs(OUT, exist_ok=True)
ok = True
STEER = len(sys.argv) > 1 and sys.argv[1] == "steer"


def fail(m):
    global ok
    ok = False
    print("  FAIL", m)


with sync_playwright() as p:
    b = p.chromium.launch(channel="msedge", headless=False)
    pg = b.new_page(viewport={"width": 1100, "height": 860})
    errors = []
    pg.on("console", lambda m: errors.append(m.text[:160]) if m.type == "error" else None)
    pg.on("pageerror", lambda e: errors.append(str(e)[:160]))
    pg.goto(f"{URL}/#/create")
    pg.wait_for_selector(".model[aria-checked='true']", timeout=60000)
    pg.click("text=Best")                                      # 4 steps
    pg.wait_for_timeout(1500)
    pg.fill("#prompt", "a red bicycle leaning on a brick wall, photo")
    pg.click("button.btn.primary")
    t0, first, shots = time.time(), None, 0
    steered = False
    while time.time() - t0 < 240:
        st = pg.evaluate("""(() => {
            const pv = document.querySelector('.stage img.preview');
            const fin = [...document.querySelectorAll('.stage img')].find(i => !i.classList.contains('preview'));
            return {preview: !!pv, loaded: !!(pv && pv.complete && pv.naturalWidth > 0), final: !!fin, done: !!document.querySelector('.progress-wrap .badge.ok'),
                    step: document.querySelector('.progress-row strong')?.textContent || ''}
        })()""")
        if STEER and st["loaded"] and not steered:
            steered = True
            print("steering: pausing")
            pg.click("button:has-text('Pause')")
            tp = time.time()                                    # taken at the next step boundary: up to one step (about 8 s here)
            while time.time() - tp < 25 and not pg.evaluate("(document.querySelector('.progress-row strong')?.textContent || '').startsWith('Paused')"): pg.wait_for_timeout(150)
            head = pg.evaluate("document.querySelector('.progress-row strong')?.textContent || ''")
            c1 = pg.evaluate("document.querySelector('.progress-row .mono')?.textContent || ''")
            pg.wait_for_timeout(3000)
            c2 = pg.evaluate("document.querySelector('.progress-row .mono')?.textContent || ''")
            print(f"  heading {head!r}; clock {c1!r} -> {c2!r}")
            if not head.startswith("Paused before step"): fail("heading does not say paused")
            paused_step = head.split("step ")[1].split(" ")[0]
            if c1 != c2: fail("the clock kept running while paused")
            pg.fill("#steer", "a tall white lighthouse on a cliff at dusk, photo"); pg.click("button:has-text('Apply')")
            pg.wait_for_timeout(800)
            note = pg.evaluate("document.querySelector('#steer-hint')?.textContent || ''")
            print(f"  note after Apply: {note[-60:]!r}")
            if "Sent" not in note: fail("no confirmation after Apply")
            pg.screenshot(path=os.path.join(OUT, "app_steer_paused.png"))
            pg.click("button:has-text('Resume')")
            tp = time.time()
            while time.time() - tp < 30 and "Prompt changed before step" not in pg.evaluate("document.querySelector('#steer-hint')?.textContent || ''"): pg.wait_for_timeout(200)
            note = pg.evaluate("document.querySelector('#steer-hint')?.textContent || ''")
            print(f"  after Resume: {note[-45:]!r}")
            if "Prompt changed before step" not in note: fail("the app never showed that the prompt was applied")
            elif f"before step {paused_step}" not in note: fail(f"the pause said step {paused_step}, the prompt note says: {note[-30:]}")
        if st["loaded"] and first is None:
            first = time.time() - t0
            print(f"first preview after {first:.1f} s while: {st['step']!r}")
            pg.wait_for_timeout(300); pg.screenshot(path=os.path.join(OUT, "app_preview.png")); shots += 1
        if st["done"]: break
        pg.wait_for_timeout(200)
    total = time.time() - t0
    if first is None: fail("no preview image appeared during the job")
    elif first > total - 3: fail("the preview came only at the very end")
    pg.wait_for_timeout(1200)
    st = pg.evaluate("(() => { const i = [...document.querySelectorAll('.stage img')].find(i => !i.classList.contains('preview')); return !!(i && i.complete && i.naturalWidth > 0) })()")
    if not st: fail("the final image is not shown")
    pg.screenshot(path=os.path.join(OUT, "app_preview_done.png"))
    print(f"job {total:.1f} s; console errors {len(errors)}")
    for e in errors[:5]: fail("console: " + e)
    b.close()
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
