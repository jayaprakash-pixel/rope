"""P1.2.2: planner time estimates vs measured rope runs that were NOT used for calibration.
Calibration point (catalog): 512², 1 step, uncapped, 6 threads. Test cases change steps, resolution, decoder,
CPU speed (ROPE_THREADS) and the 4 GB memory cap. Pass: every estimate within +/-50% of the measured wall time."""
import json, os, subprocess, sys, time
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
ROPE = os.path.join(R, "build", "rust-cpu", "release", "rope.exe")
M = os.path.join(R, "models")
MEAS = os.path.join(R, "bench", "diffusion", "measure.py")
CASES = [  # (size, steps, decoder, threads, profile)
    (512, 2, "taesd", 6, None), (512, 4, "vae", 6, None), (384, 1, "taesd", 6, None), (256, 2, "vae", 6, None),
    (448, 3, "taesd", 6, None), (512, 1, "taesd", 3, None), (384, 2, "vae", 2, None), (512, 1, "vae", 6, "4gb"),
]
rows, ok = [], True
for size, steps, dec, th, prof in CASES:
    env = dict(os.environ, ROPE_THREADS=str(th))
    plan = json.loads(subprocess.run([ROPE, "plan", "--json", "--size", str(size), "--steps", str(steps), "--decoder", dec,
                                      "--models-dir", M, "--catalog", os.path.join(R, "catalog", "models.json"), "--disk-mb", "0"],
                                     capture_output=True, text=True, env=env, cwd=R, check=True).stdout)
    est = plan["estimate_s"]["total"]
    cmd = [ROPE, "image", "--model", os.path.join(M, "sd_turbo.safetensors"), "--tokenizer", os.path.join(M, "sd-turbo-config", "tokenizer"),
           "--prompt", "a red bicycle leaning on a brick wall, photo", "--size", str(size), "--steps", str(steps), "--out", os.path.join(os.environ.get("TEMP", "."), "rope_plan_check.png")]
    if dec == "taesd":
        cmd += ["--taesd", os.path.join(M, "taesd.safetensors")]
    if prof:
        r = subprocess.run([sys.executable, "-I", MEAS, "--label", f"plancheck_{size}_{steps}_{dec}_{th}_{prof}", "--profile", prof, "--no-gpu", "--"] + cmd,
                           capture_output=True, text=True, env=env, check=True)
        wall = json.loads(r.stdout)["wall_s"]
    else:
        t0 = time.time(); subprocess.run(cmd, capture_output=True, env=env, check=True); wall = time.time() - t0
    err = (est - wall) / wall
    ok &= abs(err) <= 0.5
    rows.append((size, steps, dec, th, prof or "-", plan["device_gflops"], est, wall, err))
    print(f"{size:4d}² {steps} step(s) {dec:5s} threads {th} cap {prof or '-':3s} | device {plan['device_gflops']:6.1f} GFLOPS | estimate {est:6.1f} s | measured {wall:6.1f} s | error {err:+6.1%}", flush=True)
worst = max(abs(r[-1]) for r in rows)
print(f"worst error {worst:.1%} (limit 50%) -> {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
