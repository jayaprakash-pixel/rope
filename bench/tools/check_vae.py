"""P1.5.5.6: rope's full SD VAE decoder vs diffusers AutoencoderKL (weights mapped by diffusers'
convert_ldm_vae_checkpoint). Same latent -> PSNR >= 35 dB."""
import os, subprocess, sys, tempfile
import numpy as np, torch
from diffusers import AutoencoderKL
from diffusers.loaders.single_file_utils import convert_ldm_vae_checkpoint

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
ROPE = os.path.join(R, "build", "rust-cpu", "release", "rope.exe")
MODEL, CFG = os.path.join(R, "models", "sd_turbo.safetensors"), os.path.join(R, "models", "sd-turbo-config")
TMP = tempfile.gettempdir()
lat_f, out_f = os.path.join(TMP, "rope_vae_lat.f32"), os.path.join(TMP, "rope_vae_out.f32")
# a realistic latent: rope's own 1-step SD-Turbo result (bench/rope/first_x0.f32) if present, else seeded noise
src = os.path.join(R, "bench", "rope", "first_x0.f32")
lat = np.fromfile(src, dtype="<f4") if os.path.exists(src) else np.random.default_rng(3).standard_normal(4 * 64 * 64).astype("<f4")
lat.tofile(lat_f)
r = subprocess.run([ROPE, "vae", MODEL, lat_f, out_f], capture_output=True, text=True)
assert r.returncode == 0, r.stderr
got = np.fromfile(out_f, dtype="<f4").reshape(3, 512, 512)

lines = subprocess.run([ROPE, "inspect", MODEL], capture_output=True, text=True, encoding="utf-8", check=True).stdout.splitlines()[1:]
mm = np.memmap(MODEL, dtype=np.uint8, mode="c")
ck = {}
for ln in lines:
    name, dt, shape, off, nb = ln.split("\t")
    if name.startswith("first_stage_model.") and dt == "F32":
        shp = tuple(int(s) for s in shape.split(",")) if shape else ()
        ck[name] = torch.from_numpy(mm[int(off):int(off) + int(nb)].view(np.float32).reshape(shp))
cfg = AutoencoderKL.load_config(os.path.join(CFG, "vae"))
vae = AutoencoderKL.from_config(cfg)
missing, unexpected = vae.load_state_dict(convert_ldm_vae_checkpoint(ck, cfg), strict=False)
assert not [m for m in missing if m.startswith("decoder") or m.startswith("post_quant")], missing[:5]
vae.eval()
with torch.no_grad():
    ref = vae.decode(torch.from_numpy(lat.reshape(1, 4, 64, 64)) / cfg["scaling_factor"]).sample[0].numpy()
u8 = lambda x: ((x / 2 + 0.5).clip(0, 1) * 255).round()
p = 10 * np.log10(255 ** 2 / np.mean((u8(got) - u8(ref)) ** 2)) if np.any(u8(got) != u8(ref)) else float("inf")
print(f"VAE: PSNR {p:.1f} dB (limit 35), max abs diff {np.abs(got - ref).max():.2e}   [{r.stderr.strip()}]")
sys.exit(0 if p >= 35 else 1)
