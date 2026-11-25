"""P2.4.5: the upscaler add-on (Real-ESRGAN x4, RRDBNet).
  1. One tile (64x64 test image) through rope (`rope upscale`, ROPE_UPSCALE_TILE=0) vs a torch RRDBNet written here from the same
     file: max difference in the [0,1] output must be <= 1e-3.
  2. Tiled run (tile 128, pad 16) of a 160x144 image vs the one-piece run: PSNR (information; the pad is a known approximation).
  3. 512 -> 2048 on the 4 GB profile: `rope image --upscale` (SD-Turbo 512, 1 step, --ram-gib 3.4 --no-gpu): the image is 2048x2048 and the
     peak commit stays under the profile's private-memory budget (RAM - 1 GiB = 2.4 GiB).
Deletes its temporary files at the end."""
import os, re, subprocess, sys, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
ROPE = os.environ.get("ROPE_EXE", os.path.join(R, "build", "rust-cpu", "release", "rope.exe"))
UP = os.path.join(R, "models", "addons", "realesrgan_x4plus.safetensors")
TMP = os.path.join(R, "bench", "diffusion", "out")
os.makedirs(TMP, exist_ok=True)
ok = True


def fail(m):
    global ok
    ok = False
    print("  FAIL", m)


class RDB(nn.Module):
    def __init__(s, nf=64, gc=32):
        super().__init__()
        s.conv1, s.conv2, s.conv3 = nn.Conv2d(nf, gc, 3, 1, 1), nn.Conv2d(nf + gc, gc, 3, 1, 1), nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
        s.conv4, s.conv5 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1), nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)

    def forward(s, x):
        a = lambda t: F.leaky_relu(t, 0.2)
        x1 = a(s.conv1(x)); x2 = a(s.conv2(torch.cat((x, x1), 1))); x3 = a(s.conv3(torch.cat((x, x1, x2), 1)))
        x4 = a(s.conv4(torch.cat((x, x1, x2, x3), 1)))
        return s.conv5(torch.cat((x, x1, x2, x3, x4), 1)) * 0.2 + x


class RRDB(nn.Module):
    def __init__(s):
        super().__init__()
        s.rdb1, s.rdb2, s.rdb3 = RDB(), RDB(), RDB()

    def forward(s, x): return s.rdb3(s.rdb2(s.rdb1(x))) * 0.2 + x


class Net(nn.Module):
    def __init__(s, n=23):
        super().__init__()
        s.conv_first = nn.Conv2d(3, 64, 3, 1, 1)
        s.body = nn.Sequential(*[RRDB() for _ in range(n)])
        s.conv_body, s.conv_up1, s.conv_up2 = nn.Conv2d(64, 64, 3, 1, 1), nn.Conv2d(64, 64, 3, 1, 1), nn.Conv2d(64, 64, 3, 1, 1)
        s.conv_hr, s.conv_last = nn.Conv2d(64, 64, 3, 1, 1), nn.Conv2d(64, 3, 3, 1, 1)

    def forward(s, x):
        a = lambda t: F.leaky_relu(t, 0.2)
        fea = s.conv_first(x)
        fea = fea + s.conv_body(s.body(fea))
        fea = a(s.conv_up1(F.interpolate(fea, scale_factor=2, mode="nearest")))
        fea = a(s.conv_up2(F.interpolate(fea, scale_factor=2, mode="nearest")))
        return s.conv_last(a(s.conv_hr(fea)))


def test_image(h, w):
    rng = np.random.default_rng(5)
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.stack([np.sin(x / 9) * 0.5 + 0.3, np.cos(y / 7) * 0.4, np.sin((x + y) / 13) * 0.6]) + rng.standard_normal((3, h, w)).astype(np.float32) * 0.05
    return np.clip(img, -1, 1).astype(np.float32)                 # [3,h,w] in [-1,1]


def rope_up(img, tiled):
    c, h, w = img.shape
    fi, fo = os.path.join(TMP, "up_in.f32"), os.path.join(TMP, "up_out.f32")
    img.tofile(fi)
    env = dict(os.environ, ROPE_UPSCALE_TILE="1" if tiled else "0")
    t = time.time()
    r = subprocess.run([ROPE, "upscale", UP, fi, str(h), str(w), fo], capture_output=True, text=True, env=env)
    if r.returncode != 0: fail("rope upscale: " + r.stderr.strip()); return None
    print(f"  rope {'tiled' if tiled else 'one piece'}: {time.time() - t:.1f} s")
    return np.fromfile(fo, dtype=np.float32).reshape(3, h * 4, w * 4)


# 1. one tile against torch
net = Net().eval()
sd = load_file(UP)
print("torch reference: load", net.load_state_dict({k: v for k, v in sd.items()}, strict=True))
img = test_image(64, 64)
with torch.no_grad(): ref = net(torch.from_numpy((img + 1) / 2)[None])[0].numpy() * 2 - 1
mine = rope_up(img, False)
if mine is not None:
    d = np.abs(mine - ref).max() / 2                            # in the [0,1] domain
    print(f"one tile 64x64 -> 256x256: max difference {d:.2e} in [0,1] (need <= 1e-3); output range {mine.min():.2f}..{mine.max():.2f}")
    if d > 1e-3: fail("one tile differs from the reference")
# 2. tiled vs one piece
img2 = test_image(144, 160)
a, b = rope_up(img2, False), rope_up(img2, True)
if a is not None and b is not None:
    mse = (((a - b) / 2 * 255) ** 2).mean()
    print(f"tiled (128 + pad 16) vs one piece, 160x144 -> 640x576: PSNR {10 * np.log10(255 ** 2 / mse):.1f} dB, max difference {np.abs(a - b).max() / 2:.2e}")
# 3. 512 -> 2048 on the 4 GB profile
out = os.path.join(TMP, "up_2048.png")
cmd = [ROPE, "image", "--model", os.path.join(R, "models", "sd_turbo.safetensors"), "--prompt", "a red bicycle leaning on a brick wall, photo", "--quality", "fast",
       "--steps", "1", "--size", "512", "--seed", "1", "--models-dir", os.path.join(R, "models"), "--catalog", os.path.join(R, "catalog", "models.json"),
       "--ram-gib", "3.4", "--no-gpu", "--disk-mb", "0", "--upscale", UP, "--out", out]
t = time.time()
r = subprocess.run(cmd, capture_output=True, text=True)
m = re.search(r"memory: peak commit ([\d.]+) GiB", r.stderr)
up = [l for l in r.stderr.splitlines() if l.startswith("upscale:")]
print(f"512 -> 2048 run: exit {r.returncode}, {time.time() - t:.0f} s, {up[0] if up else ''}, peak commit {m.group(1) if m else '?'} GiB (budget 2.4)")
if r.returncode != 0 or not os.path.exists(out): fail("the 2048 run failed: " + (r.stderr.strip().splitlines() or [''])[-1])
else:
    wh = Image.open(out).size
    if wh != (2048, 2048): fail(f"image is {wh}, expected 2048x2048")
    if not m or float(m.group(1)) > 2.4: fail("peak commit above the 4 GB profile budget")
for f in ("up_in.f32", "up_out.f32"):
    p = os.path.join(TMP, f)
    if os.path.exists(p): os.remove(p)
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
