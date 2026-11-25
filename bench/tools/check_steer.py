"""P5.1.7: steering a running job through rope serve's HTTP API (no browser). Needs `rope serve --port 4013` running.
  check_steer.py cpu   CPU-build server: SD-Turbo, quality Best (4 steps)
  check_steer.py gpu   CUDA-build server: SDXL-Turbo (4 steps) and FLUX (4 steps); optional model ids after the mode
For each model, with the same seed:
  A  unsteered job                                     -> image A
  B  Pause and a prompt change taken at the boundary of step 2 (events stop for >= 4 s), then Resume
                                                       -> `paused`, `resumed`, `prompt` events; image B differs from A (mean abs > 5;
                                                          an unsteered rerun differs by exactly 0);
                                                          the ETA after the resume does not include the pause
  C  Cancel at step 2                                  -> the job ends (error event) within 1.5 step times
The pass/fail rules are written next to each check."""
import json, os, sys, threading, time, urllib.request
import numpy as np
from PIL import Image

URL = "http://127.0.0.1:4013"
MODE = sys.argv[1] if len(sys.argv) > 1 else "cpu"
P1 = "a red bicycle leaning on a brick wall, photo"
P2 = "a tall white lighthouse on a cliff at dusk, photo"
ok = True


def fail(m):
    global ok
    ok = False
    print("  FAIL", m)


def post(path, body=None):
    req = urllib.request.Request(URL + path, data=json.dumps(body or {}).encode(), headers={"Content-Type": "application/json", "Origin": URL}, method="POST")
    try:
        return json.load(urllib.request.urlopen(req))
    except urllib.error.HTTPError as e:
        return {"error": json.load(e).get("error"), "status": e.code}


class Job:
    """A job and its event stream, collected on a thread."""
    def __init__(self, model, prompt, seed=5):
        r = post("/api/jobs", {"model": model, "prompt": prompt, "quality": "best", "seed": seed})
        assert "id" in r, r
        self.id, self.events, self.t0 = r["id"], [], time.time()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in urllib.request.urlopen(f"{URL}/api/jobs/{self.id}/events"):
            line = line.decode().strip()
            if line.startswith("data: "):
                self.events.append((time.time() - self.t0, json.loads(line[6:])))

    def wait_event(self, pred, timeout=400):
        t = time.time()
        while time.time() - t < timeout:
            for ts, e in list(self.events):
                if pred(e): return ts, e
            time.sleep(0.1)
        raise TimeoutError("event not seen")

    def end(self, timeout=600):
        return self.wait_event(lambda e: e["type"] in ("done", "error"), timeout)

    def image(self, path):
        urllib.request.urlretrieve(f"{URL}/api/jobs/{self.id}/image", path)
        return np.asarray(Image.open(path).convert("RGB"), float)


def at_step(k):
    return lambda e: e["type"] == "progress" and e["stage"] == "denoise" and e["i"] >= k


def run_model(model):
    print(f"== {model}")
    tmp = os.path.join(os.environ.get("TEMP", "."), f"rope_steer_{model}")
    # A: unsteered
    a = Job(model, P1); ts_a, e = a.end(); assert e["type"] == "done", e
    img_a = a.image(tmp + "_a.png")
    step_s = (ts_a) / 4.0
    print(f"  A unsteered: {ts_a:.1f} s")

    # B: seen at the start of step 1, so the commands are taken at the boundary of step 2 (two steps run on the new prompt)
    b = Job(model, P1)
    b.wait_event(at_step(1))
    print(post(f"/api/jobs/{b.id}/control", {"op": "pause"}))
    time.sleep(0.2)
    n0 = len(b.events); t_pause = time.time()
    r = post(f"/api/jobs/{b.id}/control", {"op": "prompt", "text": P2}); assert r.get("ok"), r
    # the pause is taken at the NEXT step boundary: the step running now finishes first; then no event for >= 4 s
    pz = b.wait_event(lambda e: e["type"] == "paused")[1]
    time.sleep(4.5)
    n1 = len([1 for _, ev in b.events if ev["type"] in ("progress", "preview")])
    time.sleep(0.5)
    n2 = len([1 for _, ev in b.events if ev["type"] in ("progress", "preview")])
    held = n1 == n2
    print(f"  B paused at step {pz['step']}: no progress while paused: {held}")
    if not held: fail("progress events kept coming while paused")
    t_resume = time.time()
    print(post(f"/api/jobs/{b.id}/control", {"op": "resume"}))
    ts_b, e = b.end(); assert e["type"] == "done", e
    img_b = b.image(tmp + "_b.png")
    kinds = [ev["type"] for _, ev in b.events]
    for k in ("paused", "resumed", "prompt"):
        if k not in kinds: fail(f"no {k} event")
    diff = float(np.abs(img_a - img_b).mean())
    print(f"  B steered: {ts_b:.1f} s (A took {ts_a:.1f} s); image differs from A by {diff:.1f} (mean abs, 0-255)")
    if diff <= 5: fail("the prompt change did not change the final image (mean abs difference <= 5; an unsteered rerun differs by exactly 0)")
    # the ETA must not count the pause: the last progress after the resume has elapsed ~ real time minus the pause
    paused_s = t_resume - (b.t0 + [ts for ts, ev in b.events if ev["type"] == "paused"][0])
    last = [ev for _, ev in b.events if ev["type"] == "progress"][-1]
    print(f"  pause held {paused_s:.1f} s; last reported elapsed {last['elapsed_s']:.1f} s of {ts_b:.1f} s wall")
    if last["elapsed_s"] > ts_b - paused_s + 2.0: fail("elapsed in the events still includes the pause")

    # B0 (information only): the same prompt change taken at the boundary of step 1 (three steps on the new prompt)
    b0 = Job(model, P1)
    b0.wait_event(at_step(0))
    post(f"/api/jobs/{b0.id}/control", {"op": "prompt", "text": P2})
    b0.end()
    d0 = float(np.abs(img_a - b0.image(tmp + "_b0.png")).mean())
    print(f"  B0 prompt change one step earlier (step 1): image differs from A by {d0:.1f}  (the earlier, the stronger the effect)")

    # C: cancel at step 2
    c = Job(model, P1)
    c.wait_event(at_step(2)); t_c = time.time()
    post(f"/api/jobs/{c.id}/cancel")
    ts_c, e = c.end(60)
    took = time.time() - t_c
    print(f"  C cancel: ended with {e['type']} after {took:.1f} s (one step is about {step_s:.1f} s)")
    if e["type"] != "error" or took > 1.5 * step_s + 3: fail("cancel was not quick")
    # a command to a job that has ended is refused
    r = post(f"/api/jobs/{c.id}/control", {"op": "pause"})
    if r.get("status") != 409: fail(f"a command to an ended job should be refused (409), got {r}")


models = sys.argv[2:] or (["sd-turbo"] if MODE == "cpu" else ["sdxl-turbo", "flux-schnell-q4k"])
for m in models:
    run_model(m)
# validation of bad commands
r = post("/api/jobs/none/control", {"op": "explode"}); print("bad op:", r)
if r.get("status") != 400: fail("an unknown op should be refused with 400")
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
