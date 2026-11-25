"""P2.4.2: rope's SDXL vs the reference (transformers / diffusers), on the single-file SDXL-Turbo checkpoint (FP16).
  python bench/tools/check_sdxl.py text   CLIP-L + CLIP-G penultimate context [77, 2048] and the 2816-wide vector
                                          [pooled CLIP-G x text_projection, size/crop sinusoids] vs transformers
  python bench/tools/check_sdxl.py unet   one UNet call (latent [4,64,64], t = 999, rope's text condition) vs diffusers
The full FP32 model does not fit this laptop's 8 GB: weights stay FP16 and every layer with weights is cast to FP32
only while it runs (exact FP32 math, tiny extra memory). Pass: max |rope - ref| / max |ref| <= 1e-3 (text), 2e-2 (unet)."""
import os, subprocess, sys, tempfile
import numpy as np, torch

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
ROPE = os.path.join(R, "build", "rust-cpu", "release", "rope.exe")
MODEL = os.path.join(R, "models", "sd_xl_turbo_1.0_fp16.safetensors")
CFG = os.path.join(R, "models", "sdxl-turbo-config")
TMP = tempfile.gettempdir()
PROMPT = "a red bicycle leaning on a brick wall, photo"
MODE = sys.argv[1] if len(sys.argv) > 1 else "text"
rel = lambda a, b: float(np.abs(a - b).max() / np.abs(b).max())


def fp32_while_running(model):
    """Cast each leaf module with weights to FP32 for its own forward, then put the FP16 tensors back."""
    def pre(m, *_):
        m._keep = [(p, p.data) for p in m.parameters(recurse=False)]
        for p, d in m._keep: p.data = d.float()
    def post(m, *_):
        for p, d in m._keep: p.data = d
        del m._keep
    for m in model.modules():
        if any(True for _ in m.parameters(recurse=False)):
            m.register_forward_pre_hook(pre); m.register_forward_hook(post)


def rope_condition(prompt, out):
    r = subprocess.run([ROPE, "encode-text", MODEL, os.path.join(CFG, "tokenizer"), prompt, out, os.path.join(CFG, "tokenizer_2")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return np.fromfile(out, dtype="<f4")


if MODE == "text":
    from safetensors import safe_open
    from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
    from diffusers.loaders.single_file_utils import convert_open_clip_checkpoint
    from diffusers.models.embeddings import Timesteps
    PL, PG = "conditioner.embedders.0.transformer.", "conditioner.embedders.1.model."
    with safe_open(MODEL, "pt") as f:
        cl = {k.removeprefix(PL): f.get_tensor(k) for k in f.keys() if k.startswith(PL)}
        cg = {k: f.get_tensor(k) for k in f.keys() if k.startswith(PG)}
    L = CLIPTextModel(CLIPTextConfig.from_pretrained(os.path.join(CFG, "text_encoder"))).half()
    G = CLIPTextModelWithProjection(CLIPTextConfig.from_pretrained(os.path.join(CFG, "text_encoder_2"))).half()
    sg = convert_open_clip_checkpoint(G, cg, prefix=PG)
    for m, sd in [(L, cl), (G, sg)]:
        own = set(m.state_dict().keys())
        sd = {(k if k in own else k.removeprefix("text_model.")): v for k, v in sd.items()}
        missing, unexpected = m.load_state_dict(sd, strict=False)
        missing = [x for x in missing if "position_ids" not in x]
        assert not missing and not unexpected, (missing[:5], unexpected[:5])
        m.eval(); fp32_while_running(m)
    tl, tg = CLIPTokenizer.from_pretrained(os.path.join(CFG, "tokenizer")), CLIPTokenizer.from_pretrained(os.path.join(CFG, "tokenizer_2"))
    worst, ok = 0.0, True
    for p in [PROMPT, "a photo of an astronaut riding a horse on mars, 8k, detailed"]:
        ids = lambda t: t(p, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids
        with torch.no_grad():
            ol = L(ids(tl), output_hidden_states=True)
            og = G(ids(tg), output_hidden_states=True)
            ctx = torch.cat([ol.hidden_states[-2], og.hidden_states[-2]], -1)[0].float().numpy()
            tid = Timesteps(256, True, 0)(torch.tensor([512, 512, 0, 0, 512, 512])).reshape(-1)
            y = torch.cat([og.text_embeds[0].float(), tid]).numpy()
        got = rope_condition(p, os.path.join(TMP, "rope_sdxl_cond.f32"))
        e1, e2 = rel(got[:77 * 2048].reshape(77, 2048), ctx), rel(got[77 * 2048:], y)
        worst = max(worst, e1, e2); ok &= max(e1, e2) <= 1e-3
        print(f"{p[:45]!r:50s} context {e1:.2e}, pooled + size vector {e2:.2e}")
    print(f"worst {worst:.2e} -> {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)

# ---- unet ----
from diffusers import UNet2DConditionModel
from diffusers.loaders.single_file_utils import convert_ldm_unet_checkpoint
cond = rope_condition(PROMPT, os.path.join(TMP, "rope_sdxl_cond.f32"))
lat_f, out_f = os.path.join(TMP, "rope_sdxl_lat.f32"), os.path.join(TMP, "rope_sdxl_out.f32")
np.random.default_rng(42).standard_normal((4, 64, 64)).astype("<f4").tofile(lat_f)
inspect = subprocess.run([ROPE, "inspect", MODEL], capture_output=True, text=True, encoding="utf-8", check=True).stdout.splitlines()[1:]
mm = np.memmap(MODEL, dtype=np.uint8, mode="c")
ckpt = {}
for ln in inspect:
    name, dt, shape, off, nb = ln.split("\t")
    if name.startswith("model.diffusion_model."):
        assert dt == "F16", (name, dt)
        shp = tuple(int(s) for s in shape.split(",")) if shape else ()
        ckpt[name] = torch.from_numpy(mm[int(off):int(off) + int(nb)].view(np.float16).reshape(shp))
config = UNet2DConditionModel.load_config(os.path.join(CFG, "unet"))
with torch.device("meta"):
    unet = UNet2DConditionModel.from_config(config)
missing, unexpected = unet.load_state_dict(convert_ldm_unet_checkpoint(ckpt, config), strict=False, assign=True)
assert not missing and not unexpected, (missing[:5], unexpected[:5])
unet.eval(); fp32_while_running(unet)
lat = torch.from_numpy(np.fromfile(lat_f, dtype="<f4").reshape(1, 4, 64, 64))
ctx = torch.from_numpy(cond[:77 * 2048].reshape(1, 77, 2048).copy())
y = torch.from_numpy(cond[77 * 2048:].copy())
with torch.no_grad():
    ref = unet(lat, 999, encoder_hidden_states=ctx,
               added_cond_kwargs={"text_embeds": y[None, :1280], "time_ids": torch.tensor([[512., 512, 0, 0, 512, 512]])}).sample[0].numpy()
r = subprocess.run([ROPE, "unet", MODEL, lat_f, "999", os.path.join(TMP, "rope_sdxl_cond.f32"), out_f], capture_output=True, text=True)
assert r.returncode == 0, r.stderr
got = np.fromfile(out_f, dtype="<f4").reshape(4, 64, 64)
e = rel(got, ref)
print(f"UNet one call 64x64, t=999: max rel err {e:.2e}  ({r.stderr.strip()}) -> {'PASS' if e <= 2e-2 else 'FAIL'}")
sys.exit(0 if e <= 2e-2 else 1)
