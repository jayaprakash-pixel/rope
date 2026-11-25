"""P2.3.4: real peak memory of `rope image` at several image sizes, on the CPU path of the 4 GB profile
(--ram-gib 3.4 --no-gpu), one step (memory does not depend on the step count). Appends one JSON line per run to the file given.
  measure_memory.py OUT.jsonl MODEL_FILE SIZE [SIZE ...]
A run records: the plan's own numbers (plan: line, commit), the peak commit and peak working set the process reached
(the `memory:` line), the engine's total time."""
import json, os, re, subprocess, sys, time

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
ROPE = os.environ.get("ROPE_EXE", os.path.join(R, "build", "rust-cpu", "release", "rope.exe"))
out, model, sizes = sys.argv[1], sys.argv[2], [int(s) for s in sys.argv[3:]]
for size in sizes:
    cmd = [ROPE, "image", "--model", os.path.join(R, "models", model), "--prompt", "a red bicycle leaning on a brick wall, photo",
           "--quality", "fast", "--steps", "1", "--size", str(size), "--seed", "1", "--models-dir", os.path.join(R, "models"),
           "--catalog", os.path.join(R, "catalog", "models.json"), "--ram-gib", "3.4", "--no-gpu", "--disk-mb", "0",
           "--out", os.path.join(os.environ.get("TEMP", "."), f"rope_mem_{size}.png")]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    rec = {"model": model, "size": size, "rc": p.returncode, "wall_s": round(time.time() - t0, 1)}
    for l in p.stderr.splitlines():
        m = re.match(r"plan: .*decoder (\w+), buffer cache ([\d.]+) GiB", l)
        if m: rec.update(decoder=m.group(1), buffer_cache_gib=float(m.group(2)))
        m = re.match(r"memory: peak commit ([\d.]+) GiB, peak working set ([\d.]+) GiB", l)
        if m: rec.update(peak_commit_gib=float(m.group(1)), peak_ws_gib=float(m.group(2)))
        m = re.search(r"unet ([\d.]+) s", l)
        if m and l.startswith("image:"): rec["unet_s"] = float(m.group(1))
    if p.returncode: rec["error"] = p.stderr.strip().splitlines()[-1][:200]
    open(out, "a", encoding="utf-8").write(json.dumps(rec) + "\n")
    print(rec, flush=True)
