"""P5.1.6: per-step live preview of rope image (SD-Turbo, 4 steps, tiny decoder TAESD).
 1. a preview exists after every step but the last (3 of 4): the file is copied the moment its `progress: preview k/n`
    line appears; each is a valid PNG of half size (256x256 for a 512 image). Run with an unlimited budget: one per step.
 2. the previews show the final layout and colors: Pearson correlation >= 0.9 at 16x16 for the last one, and each is at least
    as close as the one before. (A preview is a blurry denoised estimate: full-size SSIM is about 0.1 and is only reported.)
 3. budget: with the default budget (3% of the plan's estimate) the engine skips previews that would not fit; the engine's own
    `previews: N in X s` line over the image time (exact; run-to-run spread is about 5%) must be <= 3%, with >= 1 preview.
    The interleaved wall-time difference is printed for the record.
Usage: check_preview.py [rounds]   (default 3; CPU path: --no-gpu)"""
import os, shutil, statistics, subprocess, sys, tempfile, time
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
ROPE = os.path.join(R, "build", "rust-cpu", "release", "rope.exe")
MODELS = os.path.join(R, "models")
ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
TMP = tempfile.mkdtemp(prefix="rope_prev_")
ok = True
spent, npv = [], []


def run(tag, preview, budget=None):
    out = os.path.join(TMP, f"{tag}.png")
    pdir = os.path.join(TMP, f"pv_{tag}")
    cmd = [ROPE, "image", "--model", os.path.join(MODELS, "sd_turbo.safetensors"), "--prompt", "a red bicycle leaning on a brick wall, photo",
           "--quality", "best", "--seed", "7", "--models-dir", MODELS, "--catalog", os.path.join(R, "catalog", "models.json"),
           "--no-gpu", "--disk-mb", "0", "--out", out] + (["--preview-dir", pdir] if preview else []) + (["--preview-budget-s", str(budget)] if budget else [])
    env = dict(os.environ, ROPE_PROGRESS="1")
    p = subprocess.Popen(cmd, env=env, stderr=subprocess.PIPE, text=True, encoding="utf-8")
    copies, total = [], None
    global spent
    for line in p.stderr:
        if line.startswith("progress: preview"):
            dst = os.path.join(TMP, f"{tag}_prev{len(copies) + 1}.png")
            shutil.copy(os.path.join(pdir, "preview.png"), dst); copies.append(dst)
        if line.startswith("image:"): total = float(line.rsplit(" total ", 1)[1].split()[0])
        if line.startswith("previews:"): spent.append(float(line.split(" in ")[1].split()[0])); npv.append(int(line.split()[1]))
    p.wait()
    assert p.returncode == 0 and total, f"run failed ({tag})"
    return out, copies, total


# 1 + 2
out, copies, t = run("a", True, budget=99)
print(f"previews written: {len(copies)} (expected 3 for 4 steps)")
if len(copies) != 3: ok = False; print("  FAIL count")
sizes = {Image.open(c).size for c in copies}
print(f"preview sizes: {sorted(sizes)}")
if sizes != {(256, 256)}: ok = False; print("  FAIL size")
final_img = Image.open(out).convert("RGB")
lowres = lambda im, n=16: np.asarray(im.resize((n, n), Image.BOX), float).ravel()
fin = lowres(final_img)
corr, full = [], []
for c in copies:
    pv = Image.open(c).convert("RGB")
    corr.append(float(np.corrcoef(lowres(pv), fin)[0, 1]))
    full.append(structural_similarity(np.asarray(pv.resize(final_img.size, Image.BICUBIC)), np.asarray(final_img), channel_axis=2))
print("layout + color correlation (16x16) of each preview vs the final image: " + ", ".join(f"{x:.2f}" for x in corr))
print("full-size SSIM, for the record (texture is not expected to match): " + ", ".join(f"{x:.2f}" for x in full))
if not corr[-1] >= 0.9: ok = False; print("  FAIL last preview correlation < 0.9")
if not all(y >= x - 0.02 for x, y in zip(corr, corr[1:])): ok = False; print("  FAIL previews do not get closer to the final image")

# 3 budget
spent.clear(); npv.clear()
with_p, without = [], []
for r in range(ROUNDS):
    without.append(run(f"n{r}", False)[2]); time.sleep(2)
    with_p.append(run(f"p{r}", True)[2]); time.sleep(2)
    print(f"  round {r}: no preview {without[-1]:.2f} s, preview {with_p[-1]:.2f} s", flush=True)
a, b = statistics.mean(with_p), statistics.mean(without)
cost = statistics.mean(spent) / a
print(f"default budget: {statistics.mean(npv):.1f} previews, {statistics.mean(spent):.2f} s of {a:.1f} s = {100 * cost:.1f}%")
if min(npv) < 1: ok = False; print("  FAIL no preview within the budget")
print(f"wall time, for the record: {a:.2f} s vs {b:.2f} s without = {100 * (a - b) / b:+.1f}% (run spread {100 * statistics.stdev(without) / b:.1f}%)")
if cost > 0.03: ok = False; print("  FAIL previews take more than 3% of the image time")
shutil.rmtree(TMP, ignore_errors=True)
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
