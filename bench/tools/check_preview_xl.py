"""P5.1.10: live previews for SDXL (TAESDXL) and FLUX (TAEF1) through `rope image` auto mode (the tiny decoder is found by the catalog).
  check_preview_xl.py sdxl   SDXL-Turbo 512, 4 steps, CPU build with --no-gpu
  check_preview_xl.py flux   FLUX schnell Q4_K 512, 4 steps, CUDA build
Checks, as for SD (check_preview.py):
 1. unlimited budget: one preview after every step but the last (3 of 4), each a valid PNG of half size (256x256);
 2. the previews show the final layout and colors: Pearson correlation >= 0.9 at 16x16 for the last one, and each at least as close as the one before;
 3. default budget (3% of the plan's estimate): previews cost <= 3% of the image time (the engine's own `previews: N in X s` line), >= 1 preview;
 4. (sdxl) previews do not change the image: the final PNG with previews equals the final PNG without (same seed, CPU).
Needs models/taesdxl.safetensors / models/taef1.safetensors (downloaded with the user's yes)."""
import hashlib, os, shutil, subprocess, sys, tempfile
import numpy as np
from PIL import Image

R = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
fam = sys.argv[1] if len(sys.argv) > 1 else "sdxl"
BUILD = "rust-cpu" if fam == "sdxl" else "rust-cuda"
ROPE = os.path.join(R, "build", BUILD, "release", "rope.exe")
MODELS = os.path.join(R, "models")
MODEL = os.path.join(MODELS, "sd_xl_turbo_1.0_fp16.safetensors" if fam == "sdxl" else "flux1-schnell-q4_k.gguf")
TMP = tempfile.mkdtemp(prefix="rope_prevxl_")
ok = True


def run(tag, preview, budget=None):
    out, pdir = os.path.join(TMP, f"{tag}.png"), os.path.join(TMP, f"pv_{tag}")
    cmd = [ROPE, "image", "--model", MODEL, "--prompt", "a red bicycle leaning on a brick wall, photo", "--quality", "fast", "--steps", "4", "--size", "512", "--seed", "7",
           "--models-dir", MODELS, "--catalog", os.path.join(R, "catalog", "models.json"), "--disk-mb", "0", "--out", out]
    cmd += ["--no-gpu"] if fam == "sdxl" else []
    cmd += (["--preview-dir", pdir] if preview else []) + (["--preview-budget-s", str(budget)] if budget else [])
    p = subprocess.Popen(cmd, env=dict(os.environ, ROPE_PROGRESS="1"), stderr=subprocess.PIPE, text=True, encoding="utf-8", cwd=R)
    copies, total, spent, n = [], None, 0.0, 0
    for line in p.stderr:
        if line.startswith("progress: preview"):
            dst = os.path.join(TMP, f"{tag}_prev{len(copies) + 1}.png")
            shutil.copy(os.path.join(pdir, "preview.png"), dst); copies.append(dst)
        if line.startswith("image:"): total = float(line.rsplit(" total ", 1)[1].split()[0])
        if line.startswith("previews:"): n, spent = int(line.split()[1]), float(line.split(" in ")[1].split()[0])
    p.wait()
    assert p.returncode == 0 and total, f"run failed ({tag})"
    return out, copies, total, n, spent


out, copies, t, _, _ = run("a", True, budget=999)
print(f"{fam}: previews written {len(copies)} (expected 3 for 4 steps), image time {t:.1f} s")
if len(copies) != 3: ok = False; print("  FAIL count")
sizes = {Image.open(c).size for c in copies}
print("preview sizes:", sorted(sizes))
if sizes != {(256, 256)}: ok = False; print("  FAIL size")
final = Image.open(out).convert("RGB")
low = lambda im, n=16: np.asarray(im.resize((n, n), Image.BOX), float).ravel()
corr = [float(np.corrcoef(low(Image.open(c).convert("RGB")), low(final))[0, 1]) for c in copies]
print("16x16 correlation of each preview with the final image: " + ", ".join(f"{x:.2f}" for x in corr))
if not corr or not corr[-1] >= 0.9: ok = False; print("  FAIL last preview correlation < 0.9")
if not all(y >= x - 0.02 for x, y in zip(corr, corr[1:])): ok = False; print("  FAIL previews do not get closer to the final image")
shutil.copy(copies[-1], os.path.join(R, "bench", "diffusion", "out", f"preview_{fam}_last.png")) if copies else None
shutil.copy(out, os.path.join(R, "bench", "diffusion", "out", f"preview_{fam}_final.png"))

out2, _, t2, n2, spent2 = run("b", True)
print(f"default budget: {n2} previews, {spent2:.2f} s of {t2:.1f} s = {100 * spent2 / t2:.1f}%")
if n2 < 1: ok = False; print("  FAIL no preview within the default budget")
if spent2 / t2 > 0.03: ok = False; print("  FAIL previews take more than 3% of the image time")
if fam == "sdxl":
    out3, _, _, _, _ = run("c", False)
    h = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
    print("final image with previews / without:", h(out), h(out3))
    if h(out) != h(out3): ok = False; print("  FAIL previews changed the image")
shutil.rmtree(TMP, ignore_errors=True)
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
