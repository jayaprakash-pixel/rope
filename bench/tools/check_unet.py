"""P1.5.5.3 / P1.7.3.3 (arg "sd15": SD 1.5): rope's SD UNet vs diffusers UNet2DConditionModel (weights mapped by diffusers' own
convert_ldm_unet_checkpoint). One call: fixed seeded latent [4,64,64], t=999, text context from rope encode-text
(verified in P1.5.5.2). Weights are zero-copy numpy memmap views (this laptop has ~3.5 GB free RAM).
Pass: max |rope - ref| / max |ref| <= 2e-2."""
import os, subprocess, sys, tempfile, time
import numpy as np, torch
from diffusers import UNet2DConditionModel
from diffusers.loaders.single_file_utils import convert_ldm_unet_checkpoint

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
ROPE = os.path.join(R, "build", "rust-cpu", "release", "rope.exe")
SD15 = len(sys.argv) > 1 and sys.argv[1] == "sd15"
MODEL = os.path.join(R, "models", "v1-5-pruned-emaonly.safetensors" if SD15 else "sd_turbo.safetensors")
CFG = os.path.join(R, "models", "sd15-config" if SD15 else "sd-turbo-config")
TMP = tempfile.gettempdir()
lat_f, ctx_f, ref_f, out_f = (os.path.join(TMP, f"rope_unet_{n}.f32") for n in ("lat", "ctx", "ref", "out"))

# inputs: seeded latent, text context from rope (already verified against transformers)
np.random.default_rng(42).standard_normal((4, 64, 64)).astype("<f4").tofile(lat_f)
r = subprocess.run([ROPE, "encode-text", MODEL, os.path.join(CFG, "tokenizer"), "a red bicycle leaning on a brick wall, photo", ctx_f], capture_output=True, text=True)
assert r.returncode == 0, r.stderr

# reference UNet on zero-copy views of the mapped file
inspect = subprocess.run([ROPE, "inspect", MODEL], capture_output=True, text=True, encoding="utf-8", check=True).stdout.splitlines()[1:]
mm = np.memmap(MODEL, dtype=np.uint8, mode="c")
ckpt = {}
for ln in inspect:
    name, dt, shape, off, nb = ln.split("\t")
    if name.startswith("model.diffusion_model.") and dt == "F32":
        shp = tuple(int(s) for s in shape.split(",")) if shape else ()
        ckpt[name] = torch.from_numpy(mm[int(off):int(off) + int(nb)].view(np.float32).reshape(shp))
config = UNet2DConditionModel.load_config(os.path.join(CFG, "unet"))
with torch.device("meta"):
    unet = UNet2DConditionModel.from_config(config)
sd = convert_ldm_unet_checkpoint(ckpt, config)
missing, unexpected = unet.load_state_dict(sd, strict=False, assign=True)
assert not missing and not unexpected, (missing[:5], unexpected[:5])
unet.eval()
lat = torch.from_numpy(np.fromfile(lat_f, dtype="<f4").reshape(1, 4, 64, 64))
ctx = torch.from_numpy(np.fromfile(ctx_f, dtype="<f4").reshape(1, 77, -1))
t0 = time.time()
with torch.no_grad():
    ref = unet(lat, torch.tensor([999.0]), encoder_hidden_states=ctx).sample[0].numpy()
t_ref = time.time() - t0
ref.astype("<f4").tofile(ref_f)
del unet, sd, ckpt, mm

r = subprocess.run([ROPE, "unet", MODEL, lat_f, "999", ctx_f, out_f], capture_output=True, text=True)
assert r.returncode == 0, r.stderr
got = np.fromfile(out_f, dtype="<f4").reshape(4, 64, 64).astype(np.float64)
ref = ref.astype(np.float64)
err = np.abs(got - ref).max() / np.abs(ref).max()
print(f"reference: all diffusers UNet tensors loaded, 0 missing/unexpected; torch CPU {t_ref:.2f} s")
print(f"rope: {r.stderr.strip()}")
print(f"max rel err {err:.2e}, mean abs err {np.abs(got - ref).mean():.2e}, ref std {ref.std():.3f} -> {'PASS' if err <= 2e-2 else 'FAIL'}")
sys.exit(0 if err <= 2e-2 else 1)
