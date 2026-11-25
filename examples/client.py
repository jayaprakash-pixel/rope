"""Make one image with the Rope HTTP API (docs/HTTP_API.md) and nothing else. Python standard library only.

    python examples/client.py "a red bicycle leaning on a brick wall" out.png [--model sd-turbo] [--size 256] [--steps 1] [--port 4003]

Start the server first:  rope serve
"""
import argparse, json, sys, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("prompt")
ap.add_argument("out")
ap.add_argument("--model", default="sd-turbo")
ap.add_argument("--size", type=int)
ap.add_argument("--steps", type=int)
ap.add_argument("--quality", default="fast")
ap.add_argument("--seed", type=int)
ap.add_argument("--port", type=int, default=4003)
a = ap.parse_args()
base = f"http://127.0.0.1:{a.port}"


def call(path, body=None):
    req = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        return urllib.request.urlopen(req)
    except urllib.error.HTTPError as e:
        sys.exit(f"{path}: {e.code} {json.load(e).get('error')}")


health = json.load(call("/api/health"))
if health.get("api") != 1:
    sys.exit(f"this client speaks API 1, the server speaks {health.get('api')}")
body = {"model": a.model, "prompt": a.prompt, "quality": a.quality}
for k in ("size", "steps", "seed"):
    if getattr(a, k) is not None:
        body[k] = getattr(a, k)
job = json.load(call("/api/jobs", body))
print("job", job["id"], "seed", job["seed"])
for raw in call(f"/api/jobs/{job['id']}/events"):             # server-sent events: lines "data: {json}"
    line = raw.decode().strip()
    if not line.startswith("data: "):
        continue
    e = json.loads(line[6:])
    if e["type"] == "progress":
        print(f"  {e['stage']} {int(e['i'])}/{int(e['n'])}  about {e['eta_s']:.0f} s left")
    elif e["type"] == "error":
        sys.exit("failed: " + e["message"])
    elif e["type"] == "done":
        open(a.out, "wb").write(call(f"/api/jobs/{job['id']}/image").read())
        print(f"saved {a.out} ({e['seconds']:.1f} s)")
        break
