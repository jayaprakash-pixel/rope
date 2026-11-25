"""P2.4.2: rope's SDXL pipeline vs diffusers, same noise (SDXL-Turbo single file, FP16).
Reference: transformers CLIP-L + CLIP-G (penultimate states, pooled x projection) + diffusers UNet2DConditionModel
(text_time conditioning) + EulerDiscreteScheduler (trailing, as rope) + AutoencoderKL (scale 0.13025);
vs `rope image --latent-in`. Weights stay FP16; each layer runs in FP32 only while it runs (the FP32 model does not
fit 8 GB). Pass: PSNR >= 30 dB.   Usage: check_pipeline_sdxl.py [steps]   (default 1)"""
import os, subprocess, sys, tempfile, time
import numpy as np, torch
from PIL import Image
from safetensors import safe_open
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from diffusers import UNet2DConditionModel, AutoencoderKL, EulerDiscreteScheduler
from diffusers.loaders.single_file_utils import convert_ldm_unet_checkpoint, convert_ldm_vae_checkpoint, convert_open_clip_checkpoint

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
ROPE = os.path.join(R, "build", "rust-cpu", "release", "rope.exe")
MODEL = os.path.join(R, "models", "sd_xl_turbo_1.0_fp16.safetensors")
CFG = os.path.join(R, "models", "sdxl-turbo-config")
TMP = tempfile.gettempdir()
PROMPT = "a red bicycle leaning on a brick wall, photo"
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 1
psnr = lambda a, b: 10 * np.log10(255.0 ** 2 / np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
to_u8 = lambda x: ((x / 2 + 0.5).clip(0, 1) * 255).round().astype(np.uint8)


def fp32_while_running(model):
    def pre(m, *_):
        m._keep = [(p, p.data) for p in m.parameters(recurse=False)]
        for p, d in m._keep: p.data = d.float()
    def post(m, *_):
        for p, d in m._keep: p.data = d
        del m._keep
    for m in model.modules():
        if any(True for _ in m.parameters(recurse=False)):
            m.register_forward_pre_hook(pre); m.register_forward_hook(post)


noise = np.random.default_rng(42).standard_normal((4, 64, 64)).astype("<f4")
noise_f, png_f = os.path.join(TMP, "rope_noise_xl.f32"), os.path.join(TMP, "rope_pipe_xl.png")
noise.tofile(noise_f)
r = subprocess.run([ROPE, "image", "--model", MODEL, "--tokenizer", os.path.join(CFG, "tokenizer"), "--clip-g-tokenizer",
                    os.path.join(CFG, "tokenizer_2"), "--prompt", PROMPT, "--steps", str(STEPS), "--latent-in", noise_f, "--out", png_f],
                   capture_output=True, text=True)
assert r.returncode == 0, r.stderr
rope_img = np.asarray(Image.open(png_f).convert("RGB"))

# text (copies, freed after encoding)
PL, PG = "conditioner.embedders.0.transformer.", "conditioner.embedders.1.model."
with safe_open(MODEL, "pt") as f:
    cl = {k.removeprefix(PL): f.get_tensor(k) for k in f.keys() if k.startswith(PL)}
    cg = {k: f.get_tensor(k) for k in f.keys() if k.startswith(PG)}
L = CLIPTextModel(CLIPTextConfig.from_pretrained(os.path.join(CFG, "text_encoder"))).half()
G = CLIPTextModelWithProjection(CLIPTextConfig.from_pretrained(os.path.join(CFG, "text_encoder_2"))).half()
for m, sd in [(L, cl), (G, convert_open_clip_checkpoint(G, cg, prefix=PG))]:
    own = set(m.state_dict().keys())
    missing, unexpected = m.load_state_dict({(k if k in own else k.removeprefix("text_model.")): v for k, v in sd.items()}, strict=False)
    assert not [x for x in missing if "position_ids" not in x] and not unexpected
    m.eval(); fp32_while_running(m)
with torch.no_grad():
    ids = lambda d: CLIPTokenizer.from_pretrained(os.path.join(CFG, d))(PROMPT, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids
    ol, og = L(ids("tokenizer"), output_hidden_states=True), G(ids("tokenizer_2"), output_hidden_states=True)
    ctx = torch.cat([ol.hidden_states[-2], og.hidden_states[-2]], -1).float()
    pooled = og.text_embeds.float()
del L, G, cl, cg

# UNet and VAE: zero-copy FP16 views of the mapped file
lines = subprocess.run([ROPE, "inspect", MODEL], capture_output=True, text=True, encoding="utf-8", check=True).stdout.splitlines()[1:]
mm = np.memmap(MODEL, dtype=np.uint8, mode="c")
view = {}
for ln in lines:
    name, dt, shape, off, nb = ln.split("\t")
    if dt == "F16" and name.startswith(("model.diffusion_model.", "first_stage_model.")):
        shp = tuple(int(s) for s in shape.split(",")) if shape else ()
        view[name] = torch.from_numpy(mm[int(off):int(off) + int(nb)].view(np.float16).reshape(shp))
pick = lambda p: {k: v for k, v in view.items() if k.startswith(p)}
ucfg = UNet2DConditionModel.load_config(os.path.join(CFG, "unet"))
with torch.device("meta"):
    unet = UNet2DConditionModel.from_config(ucfg)
unet.load_state_dict(convert_ldm_unet_checkpoint(pick("model.diffusion_model."), ucfg), strict=True, assign=True)
vcfg = AutoencoderKL.load_config(os.path.join(CFG, "vae"))
with torch.device("meta"):
    vae = AutoencoderKL.from_config(vcfg)
vae.load_state_dict(convert_ldm_vae_checkpoint(pick("first_stage_model."), vcfg), strict=True, assign=True)
for m in (unet, vae): m.eval(); fp32_while_running(m)
sched = EulerDiscreteScheduler.from_pretrained(os.path.join(CFG, "scheduler"), timestep_spacing="trailing")
sched.set_timesteps(STEPS)
t0 = time.time()
with torch.no_grad():
    cond = {"text_embeds": pooled, "time_ids": torch.tensor([[512., 512, 0, 0, 512, 512]])}
    x = torch.from_numpy(noise)[None] * sched.init_noise_sigma
    for t in sched.timesteps:
        eps = unet(sched.scale_model_input(x, t), t, encoder_hidden_states=ctx, added_cond_kwargs=cond).sample
        x = sched.step(eps, t, x).prev_sample
    ref_img = to_u8(vae.decode(x / vcfg["scaling_factor"]).sample[0].numpy()).transpose(1, 2, 0)
Image.fromarray(ref_img).save(os.path.join(TMP, "ref_pipe_xl.png"))
p = psnr(rope_img, ref_img)
print(f"reference (torch CPU) {time.time() - t0:.0f} s; timesteps {sched.timesteps.tolist()}; VAE scale {vcfg['scaling_factor']}")
print(f"rope: {r.stderr.strip().splitlines()[-1]}")
print(f"SDXL-Turbo, {STEPS} step(s): PSNR {p:.1f} dB (limit 30) -> {'PASS' if p >= 30 else 'FAIL'}")
sys.exit(0 if p >= 30 else 1)
