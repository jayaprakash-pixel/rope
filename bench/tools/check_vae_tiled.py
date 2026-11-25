"""P1.5.5.7: tiled VAE decode at 1024x1024 (FLUX VAE, 16-channel latent [16,128,128]).
Latent: rope's real 512x512 FLUX result (bench/rope/flux_x0_512.f32) upsampled 2x (bilinear) -> smooth, image-like.
  1. untiled decode, uncapped (reference)          rope vae ae.safetensors lat out 0
  2. tiled decode, uncapped (time)                 rope vae ae.safetensors lat out 64
  3. tiled decode under the 4 GB profile (memory)  measure.py --profile 4gb --no-gpu
Pass: PSNR(tiled, untiled) >= 35 dB; capped peak commit <= 1.5 GB; capped time within 15% of uncapped tiled time."""
import json, os, subprocess, sys, tempfile, time
import numpy as np, torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "..", "..")
ROPE = os.path.join(R, "build", "rust-cpu", "release", "rope.exe")
VAE = os.path.join(R, "models", "ae.safetensors")
DIFF = os.path.join(R, "bench", "diffusion")
TMP = tempfile.gettempdir()
f = lambda n: os.path.join(TMP, f"rope_tiled_{n}")
u8 = lambda x: ((x / 2 + 0.5).clip(0, 1) * 255).round().astype(np.uint8)

x0 = torch.from_numpy(np.fromfile(os.path.join(R, "bench", "rope", "flux_x0_512.f32"), dtype="<f4").reshape(1, 16, 64, 64))
lat = torch.nn.functional.interpolate(x0, scale_factor=2, mode="bilinear", align_corners=False)[0].numpy().astype("<f4")
lat.tofile(f("lat.f32"))

def decode(tile, out):
    t0 = time.time()
    r = subprocess.run([ROPE, "vae", VAE, f("lat.f32"), out, str(tile)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return time.time() - t0, r.stderr.strip()

t_full, log_full = decode(0, f("full.f32"))
t_tile, log_tile = decode(64, f("tile.f32"))
full = np.fromfile(f("full.f32"), dtype="<f4").reshape(3, 1024, 1024)
tile = np.fromfile(f("tile.f32"), dtype="<f4").reshape(3, 1024, 1024)
a, b = u8(full).astype(np.float64), u8(tile).astype(np.float64)
p = 10 * np.log10(255 ** 2 / np.mean((a - b) ** 2)) if np.any(a != b) else float("inf")
Image.fromarray(u8(full).transpose(1, 2, 0)).save(f("full.png"))
Image.fromarray(u8(tile).transpose(1, 2, 0)).save(f("tile.png"))
# seam probe: error in the 128-px blend bands vs elsewhere
band = np.zeros((1024, 1024), bool)
for s in (48, 64):  # tile starts 0/48/64 latent -> blends around pixels 384..512 and 512..640
    band[s * 8:s * 8 + 128, :] = True; band[:, s * 8:s * 8 + 128] = True
err = np.abs(a - b).mean(0)
print(f"untiled: {log_full} ({t_full:.1f} s wall)\ntiled:   {log_tile} ({t_tile:.1f} s wall)")
print(f"PSNR tiled vs untiled {p:.1f} dB (limit 35); mean abs err in blend bands {err[band].mean():.2f}, elsewhere {err[~band].mean():.2f} (8-bit levels)")

subprocess.run([sys.executable, "-I", os.path.join(DIFF, "measure.py"), "--label", "p15557_vae_tiled_1024_4gb", "--profile", "4gb", "--no-gpu", "--",
                ROPE, "vae", VAE, f("lat.f32"), f("cap.f32"), "64"], capture_output=True, text=True, cwd=DIFF)
rec = json.loads(open(os.path.join(DIFF, "results.jsonl"), encoding="utf-8").read().strip().splitlines()[-1])
assert rec["label"] == "p15557_vae_tiled_1024_4gb" and rec["rc"] == 0, rec
slow = rec["wall_s"] / t_tile - 1
print(f"4 GB profile: {rec['wall_s']:.1f} s ({slow:+.0%} vs uncapped tiled), peak commit {rec['peak_commit_gb']:.2f} GB (limit 1.5), WS {rec['peak_ws_gb']:.2f} GB, disk {rec['disk_read_gb']:.2f} GB")
ok = p >= 35 and rec["peak_commit_gb"] <= 1.5 and slow <= 0.15
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
