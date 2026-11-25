"""P1.7.3.4: rope's SD 1.5 pipeline vs diffusers, same noise.
Reference: transformers CLIPTextModel (CLIP ViT-L) + diffusers UNet2DConditionModel + EulerDiscreteScheduler
(trailing spacing, as rope) with classifier-free guidance 7.5 + AutoencoderKL; vs `rope image --latent-in --cfg 7.5`.
Pass: PSNR >= 30 dB. Weights: zero-copy memmap views (this laptop has ~3.5 GB free RAM).
Usage: check_pipeline_sd15.py [steps]   (default 20)"""
import os, subprocess, sys, tempfile, time
import numpy as np, torch
from PIL import Image
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer
from diffusers import UNet2DConditionModel, AutoencoderKL, EulerDiscreteScheduler
from diffusers.loaders.single_file_utils import convert_ldm_unet_checkpoint, convert_ldm_vae_checkpoint

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
ROPE = os.path.join(R, "build", "rust-cpu", "release", "rope.exe")
MODEL = os.path.join(R, "models", "v1-5-pruned-emaonly.safetensors")
CFG = os.path.join(R, "models", "sd15-config")
TOK = os.path.join(CFG, "tokenizer")
TMP = tempfile.gettempdir()
PROMPT, G = "a red bicycle leaning on a brick wall, photo", 7.5
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
psnr = lambda a, b: 10 * np.log10(255.0 ** 2 / np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
to_u8 = lambda x: ((x / 2 + 0.5).clip(0, 1) * 255).round().astype(np.uint8)

noise = np.random.default_rng(42).standard_normal((4, 64, 64)).astype("<f4")
noise_f, png_f = os.path.join(TMP, "rope_noise15.f32"), os.path.join(TMP, "rope_pipe15.png")
noise.tofile(noise_f)
r = subprocess.run([ROPE, "image", "--model", MODEL, "--tokenizer", TOK, "--prompt", PROMPT, "--steps", str(STEPS),
                    "--cfg", str(G), "--latent-in", noise_f, "--out", png_f], capture_output=True, text=True)
assert r.returncode == 0, r.stderr
rope_img = np.asarray(Image.open(png_f).convert("RGB"))

lines = subprocess.run([ROPE, "inspect", MODEL], capture_output=True, text=True, encoding="utf-8", check=True).stdout.splitlines()[1:]
mm = np.memmap(MODEL, dtype=np.uint8, mode="c")
view = {}
for ln in lines:
    name, dt, shape, off, nb = ln.split("\t")
    if dt == "F32":
        shp = tuple(int(s) for s in shape.split(",")) if shape else ()
        view[name] = torch.from_numpy(mm[int(off):int(off) + int(nb)].view(np.float32).reshape(shp))
pick = lambda p: {k: v for k, v in view.items() if k.startswith(p)}

te = CLIPTextModel(CLIPTextConfig.from_pretrained(os.path.join(CFG, "text_encoder")))
own = set(te.state_dict().keys())
tsd = {k.removeprefix("cond_stage_model.transformer."): v for k, v in pick("cond_stage_model.transformer.").items()}
missing, _ = te.load_state_dict({(k if k in own else k.removeprefix("text_model.")): v for k, v in tsd.items()}, strict=False)
assert not [m for m in missing if "position_ids" not in m], missing[:5]
te.eval()
ucfg = UNet2DConditionModel.load_config(os.path.join(CFG, "unet"))
with torch.device("meta"):
    unet = UNet2DConditionModel.from_config(ucfg)
unet.load_state_dict(convert_ldm_unet_checkpoint(pick("model."), ucfg), strict=True, assign=True)
unet.eval()
vcfg = AutoencoderKL.load_config(os.path.join(CFG, "vae"))
with torch.device("meta"):
    vae = AutoencoderKL.from_config(vcfg)
vae.load_state_dict(convert_ldm_vae_checkpoint(pick("first_stage_model."), vcfg), strict=True, assign=True)
vae.eval()
sched = EulerDiscreteScheduler.from_pretrained(os.path.join(CFG, "scheduler"), timestep_spacing="trailing")
sched.set_timesteps(STEPS)
t0 = time.time()
with torch.no_grad():
    tok = CLIPTokenizer.from_pretrained(TOK)
    enc = lambda p: te(tok(p, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids).last_hidden_state
    ctx, uctx = enc(PROMPT), enc("")
    x = torch.from_numpy(noise)[None] * sched.init_noise_sigma
    for t in sched.timesteps:
        xi = sched.scale_model_input(x, t)
        ec, eu = unet(xi, t, encoder_hidden_states=ctx).sample, unet(xi, t, encoder_hidden_states=uctx).sample
        x = sched.step(eu + G * (ec - eu), t, x).prev_sample
    ref_img = to_u8(vae.decode(x / 0.18215).sample[0].numpy()).transpose(1, 2, 0)
Image.fromarray(ref_img).save(os.path.join(TMP, "ref_pipe15.png"))
p = psnr(rope_img, ref_img)
print(f"reference (torch CPU) {time.time() - t0:.0f} s; timesteps {sched.timesteps.tolist()[:3]}..., init sigma {float(sched.init_noise_sigma):.4f}")
print(f"rope: {r.stderr.strip()}")
print(f"SD 1.5, {STEPS} steps, CFG {G}: PSNR {p:.1f} dB (limit 30) -> {'PASS' if p >= 30 else 'FAIL'}")
sys.exit(0 if p >= 30 else 1)
