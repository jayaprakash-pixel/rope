"""P5.1.8: the HTTP API as documented in docs/HTTP_API.md.
Starts `rope serve --port 4013 --no-gpu` itself and checks, with plain HTTP only:
  - every route of the document answers as written (fields, status codes), the errors (400, 403, 404, 409) too,
  - examples/client.py (another program, stdlib only) makes an image through the documented API only,
  - every route listed in the document's table exists (no route in the table that answers "no such route").
Deletes the gallery records it made."""
import http.client, json, os, re, subprocess, sys, time

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
ROPE = os.path.abspath(os.environ.get("ROPE_EXE", os.path.join(R, "build", "rust-cpu", "release", "rope.exe")))
PORT = 4013
ok = True


def fail(m):
    global ok
    ok = False
    print("  FAIL", m)


def req(method, path, body=None, headers=None, host=None):
    c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=120)
    h = dict(headers or {})
    if body is not None: h["Content-Type"] = "application/json"
    c.putrequest(method, path, skip_host=True)
    c.putheader("Host", host or f"127.0.0.1:{PORT}")
    for k, v in h.items(): c.putheader(k, v)
    data = None if body is None else json.dumps(body).encode()
    if data is not None: c.putheader("Content-Length", str(len(data)))
    c.endheaders(data)
    r = c.getresponse()
    return r.status, r.read(), r.getheader("Content-Type")


def js(method, path, body=None, **kw):
    st, b, _ = req(method, path, body, **kw)
    try: return st, json.loads(b)
    except Exception: return st, None


def expect(label, got, want):
    if got != want: fail(f"{label}: got {got}, expected {want}")


proc = subprocess.Popen([ROPE, "serve", "--port", str(PORT), "--static", os.path.join(R, "app", "dist"), "--no-gpu"], cwd=R, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
made = []
try:
    for _ in range(50):
        try: js("GET", "/api/health"); break
        except OSError: time.sleep(0.2)
    st, h = js("GET", "/api/health")
    print("health:", h)
    expect("health status", st, 200); expect("api version", h.get("api"), 1)
    st, d = js("GET", "/api/device")
    expect("device", st, 200)
    for k in ("os", "cpu", "cores", "threads", "cpu_path", "ram_usable_gib", "ram_free_gib"):
        if k not in d: fail(f"device has no {k}")
    st, m = js("GET", "/api/models?quality=fast")
    expect("models", st, 200)
    sd = next((x for x in m["models"] if x["id"] == "sd-turbo"), None)
    for k in ("id", "kind", "title", "file", "on_disk", "runs"):
        if not sd or k not in sd: fail(f"model has no {k}")
    for k in ("size", "steps", "guidance", "decoder", "estimate_s", "upscale_s", "peak_gib", "vae_tile", "max_size", "max_native", "sizes", "limit", "notes", "on_gpu"):
        if sd and k not in sd: fail(f"runnable model has no {k}")
    nr = next((x for x in m["models"] if not x["runs"]), None)
    if nr and not ("why" in nr and "supported" in nr): fail("a model that does not run needs why and supported")
    st, mu = js("GET", "/api/models?quality=fast&upscale=1&only=sd-turbo&size=256")
    expect("models upscale", st, 200)
    if not mu["models"] or mu["models"][0].get("upscale_s", 0) <= 0: fail("upscale=1 gives no upscale_s")
    expect("bad quality", js("GET", "/api/models?quality=x")[0], 400)
    expect("bad size", js("GET", "/api/models?size=100")[0], 400)
    st, a = js("GET", "/api/addons")
    expect("addons", st, 200)
    for x in a["addons"]:
        for k in ("name", "kind", "title", "bytes"):
            if k not in x: fail(f"add-on has no {k}")
    print("add-ons:", [x["name"] for x in a["addons"]])
    # security rules
    expect("bad host", js("GET", "/api/health", host="evil.example:4013")[0], 403)
    expect("bad origin", js("POST", "/api/jobs", {"model": "sd-turbo", "prompt": "x"}, headers={"Origin": "http://evil.example"})[0], 403)
    # job validation
    expect("not json", req("POST", "/api/jobs", None)[0], 400)
    expect("empty prompt", js("POST", "/api/jobs", {"model": "sd-turbo", "prompt": " "})[0], 400)
    expect("bad quality", js("POST", "/api/jobs", {"model": "sd-turbo", "prompt": "x", "quality": "x"})[0], 400)
    expect("unknown model", js("POST", "/api/jobs", {"model": "nope", "prompt": "x"})[0], 400)
    expect("bad add-on name", js("POST", "/api/jobs", {"model": "sd-turbo", "prompt": "x", "lora": "../x"})[0], 400)
    expect("no such add-on", js("POST", "/api/jobs", {"model": "sd-turbo", "prompt": "x", "lora": "nothing.safetensors"})[0], 400)
    expect("bad lora_scale", js("POST", "/api/jobs", {"model": "sd-turbo", "prompt": "x", "lora_scale": 9})[0], 400)
    expect("no such job (events)", js("GET", "/api/jobs/0/events")[0], 404)
    expect("no such job (image)", js("GET", "/api/jobs/0/image")[0], 404)
    expect("no such job (preview)", js("GET", "/api/jobs/0/preview")[0], 404)
    expect("no such job (control)", js("POST", "/api/jobs/0/control", {"op": "pause"})[0], 404)
    expect("gallery bad name", js("GET", "/api/gallery/..%2Fx.png")[0], 400)
    expect("gallery missing", js("GET", "/api/gallery/nothing.png")[0], 404)
    expect("unknown route", js("GET", "/api/nothing")[0], 404)
    # one job: 409 while it runs, the event stream, control after the end
    st, j = js("POST", "/api/jobs", {"model": "sd-turbo", "prompt": "a red bicycle", "quality": "fast", "size": 256, "steps": 1, "seed": 7})
    expect("start", st, 202)
    if st == 202:
        expect("second job while running", js("POST", "/api/jobs", {"model": "sd-turbo", "prompt": "y"})[0], 409)
        expect("bad control op", js("POST", f"/api/jobs/{j['id']}/control", {"op": "dance"})[0], 400)
        expect("bad cfg", js("POST", f"/api/jobs/{j['id']}/control", {"op": "cfg", "value": 99})[0], 400)
        st, raw, ct = req("GET", f"/api/jobs/{j['id']}/events")
        expect("events", st, 200)
        if not ct or "text/event-stream" not in ct: fail("events content type")
        ev = [json.loads(l[6:]) for l in raw.decode().splitlines() if l.startswith("data: ")]
        types = [e["type"] for e in ev]
        print("events:", sorted(set(types)))
        for t in ("start", "log", "estimate", "progress", "done"):
            if t not in types: fail(f"no {t} event")
        if types[-1] != "done": fail("the stream does not end with done")
        st, png, ct = req("GET", f"/api/jobs/{j['id']}/image")
        if st != 200 or not png.startswith(b"\x89PNG") or ct != "image/png": fail("image route")
        expect("control after the end", js("POST", f"/api/jobs/{j['id']}/control", {"op": "pause"})[0], 409)
        expect("cancel", js("POST", f"/api/jobs/{j['id']}/cancel")[0], 200)
        done = [e for e in ev if e["type"] == "done"][0]
        made.append(done["image"].rsplit("/", 1)[1])
        st, g = js("GET", "/api/gallery")
        expect("gallery", st, 200)
        if not g or g[0].get("id") != j["id"] or g[0].get("seed") != 7: fail("gallery record is not the new job")
        st, png, ct = req("GET", done["image"])
        if st != 200 or ct != "image/png": fail("gallery image")
    # the example client: another program, the documented API only
    out = os.path.join(R, "bench", "diffusion", "out", "client_api.png")
    r = subprocess.run([sys.executable, os.path.join(R, "examples", "client.py"), "a red bicycle", out, "--size", "256", "--steps", "1", "--port", str(PORT)], capture_output=True, text=True)
    print("client.py:", (r.stdout.strip().splitlines() or [""])[-1], r.stderr.strip()[-200:])
    if r.returncode != 0 or not os.path.exists(out) or not open(out, "rb").read(8).startswith(b"\x89PNG"): fail("examples/client.py did not make an image")
    if os.path.exists(out): os.remove(out)
    st, g = js("GET", "/api/gallery")
    if g: made.append(g[0]["image"].rsplit("/", 1)[1])
    # every route of the document's table exists
    doc = open(os.path.join(R, "docs", "HTTP_API.md"), encoding="utf-8").read()
    for meth, path in re.findall(r"^\| `(GET|POST) (/api/[^`]*)` \|", doc, re.M):
        p = path.replace("{id}", "0").replace("{name}", "x")
        st, body = js(meth, p, {} if meth == "POST" else None)
        if isinstance(body, dict) and body.get("error") == "no such route": fail(f"documented route {meth} {path} does not exist")
    print("documented routes checked")
finally:
    proc.kill()
    time.sleep(1)
    for n in made:
        for f in (os.path.join(R, "gallery", n), os.path.join(R, "gallery", n[:-4] + ".json")):
            if os.path.exists(f): os.remove(f)
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
