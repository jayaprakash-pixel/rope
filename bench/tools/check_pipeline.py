"""P1.5.5.4 + P1.5.5.5: rope's TAESD decoder and the whole SD-Turbo pipeline vs diffusers.
  P1.5.5.4: same latent -> rope taesd vs diffusers AutoencoderTiny: PSNR >= 40 dB.
  P1.5.5.5: same noise -> diffusers EulerDiscreteScheduler (official, 1 step) + diffusers UNet + AutoencoderTiny,
            text from transformers CLIPTextModel; vs `rope image --latent-in`: PSNR >= 30 dB.
Weights: zero-copy memmap views (this laptop has ~3.5 GB free RAM)."""
import os, subprocess, sys, tempfile
import numpy as np, torch
from PIL import Image
from safetensors.torch import load_file
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer
from diffusers import UNet2DConditionModel, AutoencoderTiny, EulerDiscreteScheduler
from diffusers.loaders.single_file_utils import convert_ldm_unet_checkpoint, convert_open_clip_checkpoint

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
ROPE = os.path.join(R, "build", "rust-cpu", "release", "rope.exe")
M = os.path.join(R, "models")
MODEL, TAESD, CFG = os.path.join(M, "sd_turbo.safetensors"), os.path.join(M, "taesd.safetensors"), os.path.join(M, "sd-turbo-config")
TOK = os.path.join(CFG, "tokenizer")
TMP = tempfile.gettempdir()
PROMPT = "a red bicycle leaning on a brick wall, photo"
psnr = lambda a, b: 10 * np.log10(255.0 ** 2 / np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
to_u8 = lambda x: ((x / 2 + 0.5).clip(0, 1) * 255).round().astype(np.uint8)   # [3,H,W] in [-1,1] -> HWC u8

taesd = AutoencoderTiny.from_config(AutoencoderTiny.load_config(os.path.join(M, "taesd-config")))
missing, unexpected = taesd.load_state_dict(load_file(TAESD), strict=False)
assert not [k for k in missing if k.startswith("decoder")], missing[:5]
taesd.eval()

# ---- P1.5.5.4: TAESD alone on a fixed latent ----
lat = np.random.default_rng(7).standard_normal((4, 64, 64)).astype("<f4")
lat_f, out_f = os.path.join(TMP, "rope_taesd_lat.f32"), os.path.join(TMP, "rope_taesd_out.f32")
lat.tofile(lat_f)
r = subprocess.run([ROPE, "taesd", TAESD, lat_f, out_f], capture_output=True, text=True); assert r.returncode == 0, r.stderr
got = np.fromfile(out_f, dtype="<f4").reshape(3, 512, 512)
with torch.no_grad():
    ref = taesd.decode(torch.from_numpy(lat)[None]).sample[0].numpy()
p_taesd = psnr(to_u8(got), to_u8(ref))
print(f"P1.5.5.4 TAESD: PSNR {p_taesd:.1f} dB (limit 40), max abs diff {np.abs(got - ref).max():.2e}   [{r.stderr.strip()}]")

# ---- P1.5.5.5: whole pipeline, same noise ----
noise = np.random.default_rng(42).standard_normal((4, 64, 64)).astype("<f4")
noise_f, png_f = os.path.join(TMP, "rope_noise.f32"), os.path.join(TMP, "rope_pipe.png")
noise.tofile(noise_f)
r = subprocess.run([ROPE, "image", "--model", MODEL, "--taesd", TAESD, "--tokenizer", TOK, "--prompt", PROMPT,
                    "--latent-in", noise_f, "--out", png_f], capture_output=True, text=True)
assert r.returncode == 0, r.stderr
rope_img = np.asarray(Image.open(png_f).convert("RGB"))

lines = subprocess.run([ROPE, "inspect", MODEL], capture_output=True, text=True, encoding="utf-8", check=True).stdout.splitlines()[1:]
mm = np.memmap(MODEL, dtype=np.uint8, mode="c")
view = {}
for ln in lines:
    name, dt, shape, off, nb = ln.split("\t")
    if dt == "F32" and (name.startswith("model.diffusion_model.") or name.startswith("conditioner.embedders.0.model.")):
        shp = tuple(int(s) for s in shape.split(",")) if shape else ()
        view[name] = torch.from_numpy(mm[int(off):int(off) + int(nb)].view(np.float32).reshape(shp))
te = CLIPTextModel(CLIPTextConfig.from_pretrained(os.path.join(CFG, "text_encoder")))
own = set(te.state_dict().keys())
sd = convert_open_clip_checkpoint(te, {k: v for k, v in view.items() if k.startswith("conditioner.")}, prefix="conditioner.embedders.0.model.")
te.load_state_dict({(k if k in own else k.removeprefix("text_model.")): v for k, v in sd.items()}, strict=False)
te.eval()
ucfg = UNet2DConditionModel.load_config(os.path.join(CFG, "unet"))
with torch.device("meta"):
    unet = UNet2DConditionModel.from_config(ucfg)
unet.load_state_dict(convert_ldm_unet_checkpoint({k: v for k, v in view.items() if k.startswith("model.")}, ucfg), strict=True, assign=True)
unet.eval()
sched = EulerDiscreteScheduler.from_pretrained(os.path.join(CFG, "scheduler"))
sched.set_timesteps(1)
with torch.no_grad():
    ids = CLIPTokenizer.from_pretrained(TOK)(PROMPT, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids
    ctx = te(ids).last_hidden_state
    x = torch.from_numpy(noise)[None] * sched.init_noise_sigma
    for t in sched.timesteps:
        eps = unet(sched.scale_model_input(x, t), t, encoder_hidden_states=ctx).sample
        x = sched.step(eps, t, x).prev_sample
    ref_img = to_u8(taesd.decode(x).sample[0].numpy()).transpose(1, 2, 0)
Image.fromarray(ref_img).save(os.path.join(TMP, "ref_pipe.png"))
p_pipe = psnr(rope_img, ref_img)
print(f"P1.5.5.5 pipeline: PSNR {p_pipe:.1f} dB (limit 30); timesteps {sched.timesteps.tolist()}, init sigma {float(sched.init_noise_sigma):.4f}   [{r.stderr.strip()}]")
ok = p_taesd >= 40 and p_pipe >= 30
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
