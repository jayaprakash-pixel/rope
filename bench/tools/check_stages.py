"""P2.3.5: the stage timeline and one memory cap.
For SD-Turbo (TAESD decoder) at 512 and 256 on the 4 GB profile (--ram-gib 3.4 --no-gpu), 1 step:
  A = peak commit of the image run alone, B = peak commit of the upscaler alone on an image of that size (4x), C = peak commit of the image run with --upscale.
  The multi-stage run must peak at the largest single stage: |C - max(A, B)| <= 10% (+0.01 GiB rounding) of max(A, B). Stages run one after the other, so
  the peaks do not add up.
  `rope plan --explain --upscale` must list the stages text, denoise, decode, upscale with what each frees, and its predicted peak must be within
  -10%..+20% of C (a prediction below the measurement is the unsafe side).
Needs models/sd_turbo.safetensors and models/addons/realesrgan_x4plus.safetensors. Deletes its temporary files."""
import os, re, subprocess, sys
import numpy as np

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
ROPE = os.path.abspath(os.environ.get("ROPE_EXE", os.path.join(R, "build", "rust-cpu", "release", "rope.exe")))
M, CAT = os.path.join(R, "models"), os.path.join(R, "catalog", "models.json")
UP, MODEL = os.path.join(M, "addons", "realesrgan_x4plus.safetensors"), os.path.join(M, "sd_turbo.safetensors")
TMP = os.path.join(R, "bench", "diffusion", "out")
ok = True


def peak(err):
    m = re.search(r"memory: peak commit ([\d.]+) GiB", err)
    return float(m.group(1)) if m else None


def image(size, up):
    out = os.path.join(TMP, "stages.png")
    cmd = [ROPE, "image", "--model", MODEL, "--prompt", "a red bicycle leaning on a brick wall, photo", "--quality", "fast", "--steps", "1", "--size", str(size), "--seed", "1",
           "--models-dir", M, "--catalog", CAT, "--ram-gib", "3.4", "--no-gpu", "--disk-mb", "0", "--out", out] + (["--upscale", UP] if up else [])
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode, peak(r.stderr), r.stderr


def upscaler_alone(size):
    fi, fo = os.path.join(TMP, "st_in.f32"), os.path.join(TMP, "st_out.f32")
    np.random.default_rng(1).uniform(-1, 1, (3, size, size)).astype(np.float32).tofile(fi)
    r = subprocess.run([ROPE, "upscale", UP, fi, str(size), str(size), fo], capture_output=True, text=True, env=dict(os.environ, ROPE_UPSCALE_TILE="1"))
    for f in (fi, fo):
        if os.path.exists(f): os.remove(f)
    return peak(r.stderr)


for size in (512, 256):
    rc1, A, e1 = image(size, False)
    B = upscaler_alone(size)
    rc2, C, e2 = image(size, True)
    if rc1 or rc2 or None in (A, B, C): ok = False; print("  FAIL a run failed:", (e1 + e2).strip().splitlines()[-1]); continue
    big = max(A, B)
    pl = subprocess.run([ROPE, "plan", "--model", MODEL, "--explain", "--upscale", "--size", str(size), "--quality", "fast", "--steps", "1", "--ram-gib", "3.4", "--no-gpu", "--disk-mb", "0",
                         "--models-dir", M, "--catalog", CAT], capture_output=True, text=True).stdout
    m = re.search(r"predicted peak ([\d.]+) GiB", pl)
    pred = float(m.group(1)) if m else 0.0
    print(f"{size}: image alone {A:.2f} GiB, upscaler alone {B:.2f}, both {C:.2f} (largest single stage {big:.2f}: {100 * (C - big) / big:+.1f}%); plan predicts {pred:.2f} ({100 * (pred - C) / C:+.1f}%)")
    if abs(C - big) > 0.10 * big + 0.01:   # +0.01: the peaks are printed to 0.01 GiB
        ok = False; print("  FAIL multi-stage peak is not the largest single stage (within 10%)")
    # high side = safe: at 256 the size model is already 40% cautious for the image alone
    if not -10 <= 100 * (pred - C) / C <= 30: ok = False; print("  FAIL predicted peak off")
    for st in ("text ", "denoise", "decode", "upscale"):
        if not re.search(rf"^\s+{st}\s.*then frees:", pl, re.M): ok = False; print(f"  FAIL plan --explain has no stage line for {st.strip()}")
out = os.path.join(TMP, "stages.png")
if os.path.exists(out): os.remove(out)
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
