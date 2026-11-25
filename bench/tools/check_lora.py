"""P2.4.4: LoRA applied as a low-rank add equals the merged-weight reference.
Makes a synthetic kohya-style LoRA for SD-Turbo (F32 base, so the merge is exact): rank 8, random alpha, every linear layer of the UNet,
strength 0.8. Writes a merged copy of the checkpoint (W + s*alpha/r * up@down), runs both
  rope image --model sd_turbo.safetensors --lora L --lora-scale 0.8     and     rope image --model merged.safetensors
with the same seed/noise, and compares the two PNGs: PSNR must be >= 75 dB. Also checks that the LoRA changes the image
against the plain model (PSNR < 45 dB, so the test can fail), and that a LoRA for another model is refused.
Deletes the merged copy and the LoRA file at the end (storage rule)."""
import os, subprocess, sys
import numpy as np
from PIL import Image
from safetensors import safe_open
from safetensors.numpy import save_file

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
ROPE = os.environ.get("ROPE_EXE", os.path.join(R, "build", "rust-cpu", "release", "rope.exe"))
M = os.path.join(R, "models")
BASE = os.path.join(M, "sd_turbo.safetensors")
TMP = os.path.join(R, "bench", "diffusion", "out")
os.makedirs(TMP, exist_ok=True)
LORA, MERGED = os.path.join(TMP, "synthetic_lora.safetensors"), os.path.join(TMP, "sd_turbo_merged.safetensors")
S, RANK = 0.8, 8
rng = np.random.default_rng(3)
ok = True


def psnr(a, b):
    a, b = [np.asarray(Image.open(p).convert("RGB"), dtype=np.float64) for p in (a, b)]
    mse = ((a - b) ** 2).mean()
    return 99.0 if mse == 0 else 10 * np.log10(255 ** 2 / mse)


def run(model, out, extra=()):
    cmd = [ROPE, "image", "--model", model, "--prompt", "a red bicycle leaning on a brick wall", "--quality", "fast", "--steps", "1",
           "--size", "256", "--seed", "1", "--models-dir", M, "--catalog", os.path.join(R, "catalog", "models.json"), "--no-gpu", "--out", out, *extra]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode, r.stderr


f = safe_open(BASE, "np")
lora, merged = {}, {}
for k in f.keys():
    w = f.get_tensor(k)
    if k.startswith("model.diffusion_model.") and k.endswith(".weight") and w.ndim == 2:
        n, kk = w.shape
        down = (rng.standard_normal((RANK, kk)) * 0.03).astype(np.float32)
        up = (rng.standard_normal((n, RANK)) * 0.03).astype(np.float32)
        alpha = float(rng.integers(4, 17))
        base = k[len("model.diffusion_model."):-len(".weight")].replace(".", "_")
        lora[f"lora_unet_{base}.lora_down.weight"], lora[f"lora_unet_{base}.lora_up.weight"] = down, up
        lora[f"lora_unet_{base}.alpha"] = np.array(alpha, dtype=np.float32)
        w = (w.astype(np.float64) + S * alpha / RANK * (up.astype(np.float64) @ down.astype(np.float64))).astype(np.float32)
    merged[k] = w
save_file(lora, LORA)
save_file(merged, MERGED)
del merged
print(f"synthetic LoRA: {len(lora) // 3} layers, rank {RANK}")

a, b, c = [os.path.join(TMP, n) for n in ("lora_ref_plain.png", "lora_ref_lowrank.png", "lora_ref_merged.png")]
for name, model, out, extra in (("plain", BASE, a, ()), ("low-rank", BASE, b, ("--lora", LORA, "--lora-scale", str(S))), ("merged", MERGED, c, ())):
    rc, err = run(model, out, extra)
    print(f"  {name}: exit {rc}")
    if rc != 0:
        ok = False
        print("  FAIL", err.strip().splitlines()[-1] if err.strip() else "")
if ok:
    p_eq, p_chg = psnr(b, c), psnr(a, c)
    print(f"low-rank vs merged: {p_eq:.1f} dB (need >= 75); plain vs merged: {p_chg:.1f} dB (need < 45: the LoRA must change the image)")
    if p_eq < 75: ok = False; print("  FAIL low-rank result differs from the merged reference")
    if p_chg >= 45: ok = False; print("  FAIL the LoRA changes nothing")
    # a LoRA for another model is refused, not half applied
    real = os.path.join(M, "addons", "pixel-art-xl.safetensors")
    if os.path.exists(real):
        rc, err = run(BASE, os.path.join(TMP, "lora_refused.png"), ("--lora", real))
        print(f"SDXL LoRA on SD-Turbo: exit {rc}: {err.strip().splitlines()[-1][:140] if err.strip() else ''}")
        if rc == 0 or "does not fit" not in err: ok = False; print("  FAIL a LoRA of another model should be refused")
    # detection names the kinds; an outlier-injected add-on is refused (same scan as for models)
    d = subprocess.run([ROPE, "detect", LORA], capture_output=True, text=True).stdout
    print("detect synthetic LoRA:", d.splitlines()[0])
    if "family   lora" not in d: ok = False; print("  FAIL rope detect does not name the LoRA")
    bad = {k: v.copy() for k, v in list(lora.items())[:3]}
    k0 = next(k for k in bad if k.endswith("lora_down.weight"))
    bad[k0][:, ::32] = 4096.0 * 0.03                                  # one huge value in every group of 32
    INJ = os.path.join(TMP, "injected_lora.safetensors")
    save_file(bad, INJ)
    rc, err = run(BASE, os.path.join(TMP, "lora_inj.png"), ("--lora", INJ))
    print(f"outlier-injected LoRA: exit {rc}: {err.strip().splitlines()[-1][:120] if err.strip() else ''}")
    if rc == 0 or "SUSPICIOUS" not in err: ok = False; print("  FAIL an outlier-injected LoRA should be refused")
    os.remove(INJ)
for p in (LORA, MERGED):
    if os.path.exists(p): os.remove(p)
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
