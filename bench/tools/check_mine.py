"""Your own model files: a model file anywhere on the laptop, added by its path, checked and run by rope.

    python bench/tools/check_mine.py

Starts its own `rope serve` (port 4014, CPU only) with a temp models folder (this repo's models hard-linked: no copy,
no download), then checks:
  API  1. a whole model (SD-Turbo, full path) is accepted: runs, an estimate, the source rule's choice
       2. wrong inputs: a relative path (400), no file (404); a VAE file, a LoRA and a .ckpt are listed with the reason
       3. a big file on a small laptop (SDXL-Turbo, server with --ram-gib 2.5): rope finds a way (a smaller copy made once)
       4. a picture made with the added file (POST /api/jobs with its id), PNG in the gallery
       5. another web page cannot add a path (Origin rule: 403); remove takes it off the list, the file stays
  UI   6. headed Edge: paste the path, Add, the entry shows with its time; choose it, the plan card shows it; Remove.
Exit 1 on any failure.
"""
import json, os, shutil, subprocess, sys, tempfile, time, urllib.error, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXE = os.path.join(ROOT, "build", "rust-cpu", "release", "rope.exe")
PORT = 4014
URL = f"http://127.0.0.1:{PORT}"
fails = []

def check(ok, what):
    print(("PASS " if ok else "FAIL ") + what)
    if not ok: fails.append(what)

def call(path, body=None, origin=None):
    h = {"Content-Type": "application/json"}
    if origin: h["Origin"] = origin
    req = urllib.request.Request(URL + path, data=None if body is None else json.dumps(body).encode(), headers=h)
    try:
        with urllib.request.urlopen(req, timeout=300) as r: return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e: return e.code, json.loads(e.read() or b"{}")

def serve(models, extra=()):
    p = subprocess.Popen([EXE, "serve", "--port", str(PORT), "--static", os.path.join(ROOT, "app", "dist"), "--models-dir", models,
                          "--catalog", os.path.join(ROOT, "catalog", "models.json"), "--gallery", os.path.join(models, "..", "gallery"), "--no-gpu", *extra],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(100):
        try: urllib.request.urlopen(URL + "/api/health", timeout=1); return p
        except OSError: time.sleep(0.1)
    raise SystemExit("server did not start")

def main():
    tmp = tempfile.mkdtemp(prefix="rope-mine-")
    models = os.path.join(tmp, "models")
    src = os.path.join(ROOT, "models")
    for d, dirs, files in os.walk(src):
        dirs[:] = [x for x in dirs if not x.startswith(".")]
        for f in files:
            dst = os.path.join(models, os.path.relpath(os.path.join(d, f), src))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.link(os.path.join(d, f), dst)
    turbo = os.path.join(src, "sd_turbo.safetensors")           # the user's file stays in place: the repo copy
    xl = os.path.join(src, "sd_xl_turbo_1.0_fp16.safetensors")
    ckpt = os.path.join(tmp, "old-model.ckpt"); open(ckpt, "wb").write(b"not really a pickle")
    srv = serve(models)
    try:
        st, m = call("/api/mine", {"path": turbo})
        check(st == 200 and m.get("runs") and m.get("estimate_s", 0) > 0 and m.get("source") in ("as-is", "stream", "quantize"),
              f"1. SD-Turbo by full path: runs, ~{m.get('estimate_s')} s, source {m.get('source')}")
        tid = m.get("id", "")
        st, e = call("/api/mine", {"path": "models/sd_turbo.safetensors"}); check(st == 400, f"2. a relative path is refused ({st}: {e.get('error')})")
        st, e = call("/api/mine", {"path": os.path.join(tmp, "nothing.safetensors")}); check(st == 404, f"2. a missing file is refused ({st})")
        for f, want in [(os.path.join(src, "ae.safetensors"), "part of a model"), (os.path.join(src, "addons", "pixel-art-xl.safetensors"), "LoRA"), (ckpt, "pickle")]:
            st, r = call("/api/mine", {"path": f})
            check(st == 200 and r.get("runs") is False and want in (r.get("why") or ""), f"2. {os.path.basename(f)}: listed, cannot run: {r.get('why', '')[:90]}")
        st, lst = call("/api/mine?quality=fast")
        check(st == 200 and len(lst.get("models", [])) == 4, f"2. the list holds the 4 added files ({len(lst.get('models', []))})")
        # 4. a picture with the added file
        st, j = call("/api/jobs", {"model": tid, "prompt": "a red bicycle", "quality": "fast", "size": 256, "seed": 1})
        ok = st == 202
        if ok:
            t0 = time.time(); done = None
            with urllib.request.urlopen(f"{URL}/api/jobs/{j['id']}/events", timeout=900) as r:
                for line in r:
                    if line.startswith(b"data: "):
                        ev = json.loads(line[6:])
                        if ev["type"] in ("done", "error"): done = ev; break
            ok = bool(done and done["type"] == "done")
            png = os.path.join(tmp, "gallery", os.path.basename(done.get("image", ""))) if done else ""
            ok = ok and os.path.exists(png) and open(png, "rb").read(8) == b"\x89PNG\r\n\x1a\n"
            check(ok, f"4. a picture made with the added file in {time.time() - t0:.0f} s" + ("" if ok else f": {done}"))
        else: check(False, f"4. job not started: {st} {j}")
        # 5. Origin rule and remove
        st, _ = call("/api/mine", {"path": xl}, origin="http://evil.example")
        check(st == 403, f"5. another web page cannot add a path ({st})")
        st, _ = call("/api/mine/remove", {"id": tid})
        st2, lst = call("/api/mine")
        check(st == 200 and all(x["id"] != tid for x in lst["models"]) and os.path.exists(turbo), "5. remove takes it off the list; the file stays")
    finally:
        srv.kill(); srv.wait()
    # 3. a small laptop: the server plans for 2.5 GiB RAM
    os.remove(os.path.join(models, "mine.json"))
    srv = serve(models, ["--ram-gib", "2.5"])
    try:
        st, m = call("/api/mine", {"path": xl})
        notes = " ".join(m.get("notes", []))
        check(st == 200 and m.get("runs") and m.get("source") == "quantize" and "copy" in notes,
              f"3. SDXL-Turbo (6.5 GB) on a 2.5 GiB laptop: runs via {m.get('source')} ({notes[:110]}...)")
        # 6. the app
        os.remove(os.path.join(models, "mine.json"))
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            b = p.chromium.launch(channel="msedge", headless=False)
            pg = b.new_page(viewport={"width": 1180, "height": 1300})
            errors = []
            pg.on("pageerror", lambda e: errors.append(str(e)[:160]))
            pg.on("console", lambda m: errors.append(m.text[:160]) if m.type == "error" else None)
            pg.goto(f"{URL}/#/create")
            pg.wait_for_selector("text=Your own model files", timeout=60000)
            pg.fill("input[aria-labelledby=mine-label]", turbo)
            pg.click("button:has-text('Add')")
            pg.wait_for_selector(".mine-row .model:has-text('sd_turbo.safetensors')", timeout=120000)
            row = pg.inner_text(".mine-row")
            check("~" in row and "Stable Diffusion 2" in row, f"6. the entry shows with its time: {row.splitlines()[:3]}")
            check(pg.get_attribute(".mine-row .model", "aria-checked") == "true", "6. the added file is chosen")
            card = pg.inner_text("aside.plan")
            check("~" in card and "Steps" in card, "6. the plan card shows the added file's plan")
            pg.screenshot(path=os.path.join(ROOT, "bench", "diffusion", "out", "app_mine.png"), full_page=True)
            pg.click(".mine-row button:has-text('Remove')"); pg.wait_for_timeout(1000)
            check(pg.query_selector(".mine-row") is None, "6. Remove takes it off the list")
            check(not errors, f"6. no console errors {errors[:3]}")
            b.close()
    finally:
        srv.kill(); srv.wait()
        shutil.rmtree(tmp, ignore_errors=True)
    print("RESULT", "PASS" if not fails else f"FAIL ({len(fails)})")
    return 1 if fails else 0

if __name__ == "__main__":
    sys.exit(main())
