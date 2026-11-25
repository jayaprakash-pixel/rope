"""P2.0.5: rope's VAE decoder on FLUX's ae.safetensors vs diffusers AutoencoderKL (FLUX config, weights mapped by
diffusers' convert_ldm_vae_checkpoint). Same latent [16,64,64] -> 512x512; decode(x / 0.3611 + 0.1159). PSNR >= 40 dB."""
import os, subprocess, sys, tempfile
import numpy as np, torch
from safetensors.torch import load_file
from diffusers import AutoencoderKL
from diffusers.loaders.single_file_utils import convert_ldm_vae_checkpoint

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "..", "..")
ROPE = os.path.join(R, "build", "rust-cpu", "release", "rope.exe")
MODEL = os.path.join(R, "models", "ae.safetensors")
TMP = tempfile.gettempdir()
lat_f, out_f = os.path.join(TMP, "rope_fvae_lat.f32"), os.path.join(TMP, "rope_fvae_out.f32")
lat = np.random.default_rng(5).standard_normal((16, 64, 64)).astype("<f4")
lat.tofile(lat_f)
r = subprocess.run([ROPE, "vae", MODEL, lat_f, out_f], capture_output=True, text=True)
assert r.returncode == 0, r.stderr
got = np.fromfile(out_f, dtype="<f4").reshape(3, 512, 512)

cfg = AutoencoderKL.load_config(os.path.join(HERE, "flux_vae_config.json"))
vae = AutoencoderKL.from_config(cfg)
missing, unexpected = vae.load_state_dict(convert_ldm_vae_checkpoint(load_file(MODEL), cfg), strict=False)
assert not [m for m in missing if m.startswith("decoder")], missing[:5]
vae.eval()
with torch.no_grad():
    ref = vae.decode(torch.from_numpy(lat)[None] / cfg["scaling_factor"] + cfg["shift_factor"]).sample[0].numpy()
u8 = lambda x: ((x / 2 + 0.5).clip(0, 1) * 255).round()
p = 10 * np.log10(255 ** 2 / np.mean((u8(got) - u8(ref)) ** 2)) if np.any(u8(got) != u8(ref)) else float("inf")
print(f"FLUX VAE: PSNR {p:.1f} dB (limit 40), max abs diff {np.abs(got - ref).max():.2e}, missing (encoder side ok): {len(missing)}   [{r.stderr.strip()}]")
sys.exit(0 if p >= 40 else 1)
