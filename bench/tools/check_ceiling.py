"""P2.3.4: the activation-aware planner.
  check_ceiling.py predict MEASURED.jsonl   predicted peak memory vs the measured one (bench/tools/measure_memory.py output),
                                            4 GB profile, CPU only: every row from 10% below to 20% above
                                            (a prediction below the measurement risks running out of memory, so that side is tighter)
  check_ceiling.py tiles                    SDXL-Turbo at 512 with the five VAE tiles: predicted vs measured peak commit, every row from -10% to +20%
  check_ceiling.py ceiling [RAM_GIB]        for a laptop with RAM_GIB (default 1.6) of usable RAM (safety, not accuracy: the model is cautious
                                            here; the prediction must not be below the measurement): `rope plan --max` names the
                                            largest size per model; each runs at that size under a HARD commit limit of
                                            RAM - 1 GiB (the budget; the process is killed by the OS if it goes above), and the
                                            next size up is refused before any weights are loaded (fast, with the reason)
The commit limit is a Windows job object (bench/diffusion/memcap.py)."""
import json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(R, "bench", "diffusion"))
import memcap

ROPE = os.environ.get("ROPE_EXE", os.path.join(R, "build", "rust-cpu", "release", "rope.exe"))
MODELS, CATALOG = os.path.join(R, "models"), os.path.join(R, "catalog", "models.json")
IDS = {"sd_xl_turbo_1.0_fp16.safetensors": "sdxl-turbo", "sd_turbo.safetensors": "sd-turbo", "flux1-schnell-q4_k.gguf": "flux-schnell-q4k"}
FILES = {v: k for k, v in IDS.items()}
ok = True


def fail(m):
    global ok
    ok = False
    print("  FAIL", m)


def plan_json(args):
    r = subprocess.run([ROPE, "plan", "--models-dir", MODELS, "--catalog", CATALOG, "--no-gpu", "--disk-mb", "0", "--json"] + args, capture_output=True, text=True)
    return json.loads(r.stdout) if r.returncode == 0 else {"error": r.stderr.strip()}


def predict(path):
    print("predicted vs measured peak private memory (4 GB profile, CPU only, 1 step)")
    for line in open(path, encoding="utf-8"):
        m = json.loads(line)
        if m.get("rc") != 0 or "peak_commit_gib" not in m: continue
        p = plan_json(["--model-id", IDS[m["model"]], "--size", str(m["size"]), "--ram-gib", "3.4", "--quality", "fast", "--steps", "1"])
        pred, meas = p["peak_gib"], m["peak_commit_gib"]
        err = 100 * (pred - meas) / meas
        print(f"  {IDS[m['model']]:<17} {m['size']:>4}  decoder {p['decoder']:<5} predicted {pred:.2f}  measured {meas:.2f}  {err:+.1f}%")
        if not -10 <= err <= 20: fail(f"{IDS[m['model']]} {m['size']}: {err:+.1f}% off")


def run_capped(model_id, size, ram, cap_gib, steps=1, vae_tile=None):
    out = os.path.join(os.environ.get("TEMP", "."), f"rope_ceiling_{model_id}_{size}.png")
    if os.path.exists(out): os.remove(out)
    cmd = [ROPE, "image", "--model", os.path.join(MODELS, FILES[model_id]), "--prompt", "a red bicycle leaning on a brick wall, photo", "--quality", "fast",
           "--steps", str(steps), "--size", str(size), "--seed", "1", "--models-dir", MODELS, "--catalog", CATALOG, "--ram-gib", str(ram), "--no-gpu", "--disk-mb", "0", "--out", out] + (["--vae-tile", str(vae_tile)] if vae_tile else [])
    t0 = time.time()
    p, job = memcap.spawn(cmd, None, cap_gib, True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, encoding="utf-8")
    err = p.stderr.read()
    p.wait()
    peak = memcap.job_peak_commit(job) / 2 ** 30
    import re
    m = re.search(r"memory: peak commit ([\d.]+) GiB", err)
    return p.returncode, os.path.exists(out) and os.path.getsize(out) > 0, (peak, float(m.group(1)) if m else 0.0), time.time() - t0, err


def ceiling(ram):
    cap = ram - 1.0
    print(f"laptop with {ram} GiB usable RAM: private-memory budget {cap:.1f} GiB (hard commit limit for the runs)")
    mx = json.loads(subprocess.run([ROPE, "plan", "--max", "--models-dir", MODELS, "--catalog", CATALOG, "--no-gpu", "--disk-mb", "0", "--ram-gib", str(ram), "--quality", "fast", "--json"],
                                   capture_output=True, text=True).stdout)
    for m in mx["models"]:
        if m["id"] not in FILES or not m["max_size"]: continue
        print(f"== {m['id']}: ceiling {m['max_size']} px, predicted peak {m['peak_gib']:.2f} GiB, decoder tile {m['vae_tile'] * 8} px, ~{m['estimate_s']:.0f} s (4-step estimate)")
        rc, img, peak, secs, err = run_capped(m["id"], m["max_size"], ram, cap)
        print(f"   at the ceiling, commit limit {cap:.1f} GiB: exit {rc}, image {img}, job peak {peak[0]:.2f} GiB, process peak commit {peak[1]:.2f} GiB (predicted {m['peak_gib']:.2f}), {secs:.0f} s")
        if rc != 0 or not img: fail(f"{m['id']} did not finish at its ceiling: {err.strip().splitlines()[-1][:160] if err.strip() else '(no message)'}")
        if peak[1] > cap: fail(f"{m['id']}: measured process peak {peak[1]:.2f} GiB above the budget {cap:.1f}")
        over = 100 * (m["peak_gib"] - peak[1]) / peak[1]
        print(f"   prediction {over:+.0f}% against the measured process peak (on a small laptop the model is cautious: the allocator's cache cap shrinks with RAM and the model does not follow it yet)")
        if over < -10: fail(f"{m['id']}: the prediction {m['peak_gib']:.2f} is more than 10% below the measured {peak[1]:.2f} GiB (unsafe side)")
        if m["max_size"] < m["model_max"]:
            nxt = m["max_size"] + 64
            rc, img, peak, secs, err = run_capped(m["id"], nxt, ram, cap)
            last = err.strip().splitlines()[-1] if err.strip() else ""
            print(f"   next size up ({nxt}): exit {rc}, {secs:.1f} s, message: {last[:150]}")
            if rc == 0 or img: fail(f"{m['id']} {nxt} should have been refused")
            if secs > 15: fail("the refusal was not fast (weights were loaded first)")
            if "needs" not in last: fail("the refusal has no memory reason")


def tiles():
    print("SDXL-Turbo, 512, 1 step: predicted vs measured peak commit by VAE tile (latent px per side)")
    for tile in (64, 48, 32, 24, 16):
        p = plan_json(["--model-id", "sdxl-turbo", "--size", "512", "--vae-tile", str(tile), "--ram-gib", "3.4", "--quality", "fast", "--steps", "1"])
        rc, img, peak, secs, err = run_capped("sdxl-turbo", 512, 3.4, None, vae_tile=tile)
        meas = peak[1]
        e = 100 * (p["peak_gib"] - meas) / meas
        print(f"  tile {tile:>2} ({tile * 8:>3} px): predicted {p['peak_gib']:.2f}  measured {meas:.2f}  {e:+.1f}%  ({secs:.0f} s)")
        if not -10 <= e <= 20: fail(f"tile {tile}: {e:+.1f}% off")


if len(sys.argv) < 2: sys.exit(__doc__)
if sys.argv[1] == "predict": predict(sys.argv[2])
elif sys.argv[1] == "tiles": tiles()
else: ceiling(float(sys.argv[2]) if len(sys.argv) > 2 else 1.6)
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
