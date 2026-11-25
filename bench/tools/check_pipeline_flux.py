"""P2.0.6: rope's FLUX-schnell pipeline vs diffusers, same noise, 512x512, 4 steps.
Reference: diffusers FluxTransformer2DModel blocks run one at a time on the gguf-dequantized Q4_K weights (names from
diffusers' own BFL converter), diffusers FlowMatchEulerDiscreteScheduler (schnell sigmas), diffusers AutoencoderKL
(FLUX config). Text conditioning: rope's T5 + CLIP-L outputs (verified in P2.0.1 / P2.0.3).
Pass: PSNR >= 30 dB (also prints the latent error)."""
import os, subprocess, sys, tempfile, time
import numpy as np, torch
from PIL import Image
from gguf import GGUFReader
from gguf.quants import dequantize
from safetensors.torch import load_file
from diffusers import FluxTransformer2DModel, AutoencoderKL, FlowMatchEulerDiscreteScheduler
from diffusers.loaders.single_file_utils import convert_flux_transformer_checkpoint_to_diffusers, convert_ldm_vae_checkpoint

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "..", "..")
ROPE = os.path.join(R, "build", "rust-cpu", "release", "rope.exe")
M = os.path.join(R, "models")
MODEL = os.path.join(M, "flux1-schnell-q4_k.gguf")
TMP = tempfile.gettempdir()
PROMPT, SIDE, STEPS = "a red bicycle leaning on a brick wall, photo", 64, 4
f = lambda n: os.path.join(TMP, f"rope_fpipe_{n}")
psnr = lambda a, b: 10 * np.log10(255.0 ** 2 / np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
to_u8 = lambda x: ((x / 2 + 0.5).clip(0, 1) * 255).round().astype(np.uint8)

noise = np.random.default_rng(42).standard_normal((16, SIDE, SIDE)).astype("<f4")
noise.tofile(f("noise.f32"))
r = subprocess.run([ROPE, "image", "--model", MODEL, "--t5", os.path.join(M, "t5-v1_1-xxl-encoder-Q4_K_M.gguf"),
                    "--t5-tokenizer", os.path.join(M, "t5-config"), "--clip", os.path.join(M, "clip_l.safetensors"),
                    "--tokenizer", os.path.join(M, "sd15-config", "tokenizer"), "--vae", os.path.join(M, "ae.safetensors"),
                    "--prompt", PROMPT, "--steps", str(STEPS), "--size", str(SIDE * 8), "--latent-in", f("noise.f32"),
                    "--latent-out", f("x0.f32"), "--out", f("rope.png")], capture_output=True, text=True)
assert r.returncode == 0, r.stderr
print(f"rope: {r.stderr.strip()}")
rope_img = np.asarray(Image.open(f("rope.png")).convert("RGB"))
rope_x0 = np.fromfile(f("x0.f32"), dtype="<f4").reshape(16, SIDE, SIDE)

# text conditioning, from rope's verified encoders
for cmd in ([ROPE, "encode-t5", os.path.join(M, "t5-v1_1-xxl-encoder-Q4_K_M.gguf"), os.path.join(M, "t5-config"), PROMPT, f("t5.f32")],
            [ROPE, "encode-text", os.path.join(M, "clip_l.safetensors"), os.path.join(M, "sd15-config", "tokenizer"), PROMPT, f("clip.f32")]):
    rr = subprocess.run(cmd, capture_output=True, text=True); assert rr.returncode == 0, rr.stderr
txt = torch.from_numpy(np.fromfile(f("t5.f32"), dtype="<f4").reshape(1, 256, 4096))
pooled = torch.from_numpy(np.fromfile(f("clip.f32") + ".pooled", dtype="<f4").reshape(1, 768))

# ---- reference transformer, one block at a time
rd = GGUFReader(MODEL)
TT = {t.name: t for t in rd.tensors}
shape = lambda t: [int(x) for x in reversed(t.shape)]
real = lambda n: torch.from_numpy(np.ascontiguousarray(dequantize(TT[n].data, TT[n].tensor_type)).astype(np.float32).reshape(shape(TT[n])))
meta = lambda n: torch.empty(shape(TT[n]), device="meta")
top = [n for n in TT if not n.startswith(("double_blocks.", "single_blocks."))]

def converted(kind, i):
    ck = {n: (real(n) if kind is None else meta(n)) for n in top}
    for k in ("double_blocks", "single_blocks"):
        for n in TT:
            if n.startswith(f"{k}.{i if k == kind else 0}."):
                ck[f"{k}.0." + n.split(".", 2)[2]] = real(n) if k == kind else meta(n)
    return convert_flux_transformer_checkpoint_to_diffusers(ck)

with torch.device("meta"):
    model = FluxTransformer2DModel.from_config(FluxTransformer2DModel.load_config(os.path.join(HERE, "flux_transformer_config.json")))
model.eval()
model.load_state_dict({k: v for k, v in converted(None, 0).items() if not k.startswith(("transformer_blocks.", "single_transformer_blocks."))},
                      strict=False, assign=True)

def run_block(blk, kind, i, **kw):
    pre = "transformer_blocks.0." if kind == "double_blocks" else "single_transformer_blocks.0."
    blk.load_state_dict({k[len(pre):]: v for k, v in converted(kind, i).items() if k.startswith(pre)}, strict=True, assign=True)
    out = blk(**kw)
    blk.to("meta")
    return out

h2 = SIDE // 2
img_ids = torch.zeros(h2, h2, 3)
img_ids[..., 1] += torch.arange(h2)[:, None]
img_ids[..., 2] += torch.arange(h2)[None, :]
with torch.no_grad():
    rot = model.pos_embed(torch.cat([torch.zeros(256, 3), img_ids.reshape(-1, 3)], 0))
    enc0 = model.context_embedder(txt)

def transformer(packed, t):
    with torch.no_grad():
        hidden, enc = model.x_embedder(packed), enc0
        temb = model.time_text_embed(t, pooled)
        for i, blk in enumerate(model.transformer_blocks):
            enc, hidden = run_block(blk, "double_blocks", i, hidden_states=hidden, encoder_hidden_states=enc, temb=temb, image_rotary_emb=rot)
        for i, blk in enumerate(model.single_transformer_blocks):
            enc, hidden = run_block(blk, "single_blocks", i, hidden_states=hidden, encoder_hidden_states=enc, temb=temb, image_rotary_emb=rot)
        return model.proj_out(model.norm_out(hidden, temb))

sched = FlowMatchEulerDiscreteScheduler(shift=1.0, use_dynamic_shifting=False)
sched.set_timesteps(sigmas=np.linspace(1.0, 1 / STEPS, STEPS).tolist())
t0 = time.time()
x = torch.from_numpy(noise)[None].view(1, 16, h2, 2, h2, 2).permute(0, 2, 4, 1, 3, 5).reshape(1, h2 * h2, 64)
for t in sched.timesteps:
    v = transformer(x, t[None])   # the pipeline passes t/1000 and the model multiplies by 1000: time_text_embed sees t
    x = sched.step(v, t, x).prev_sample
    print(f"  reference step t={float(t):.0f} done, {time.time() - t0:.0f} s", flush=True)
ref_x0 = x.reshape(1, h2, h2, 16, 2, 2).permute(0, 3, 1, 4, 2, 5).reshape(16, SIDE, SIDE)
vcfg = AutoencoderKL.load_config(os.path.join(HERE, "flux_vae_config.json"))
vae = AutoencoderKL.from_config(vcfg)
vae.load_state_dict(convert_ldm_vae_checkpoint(load_file(os.path.join(M, "ae.safetensors")), vcfg), strict=False)
vae.eval()
with torch.no_grad():
    ref_img = to_u8(vae.decode(ref_x0[None] / vcfg["scaling_factor"] + vcfg["shift_factor"]).sample[0].numpy()).transpose(1, 2, 0)
Image.fromarray(ref_img).save(f("ref.png"))
lat_err = np.abs(rope_x0 - ref_x0.numpy()).max() / np.abs(ref_x0.numpy()).max()
p = psnr(rope_img, ref_img)
print(f"latent max rel err {lat_err:.2e}; reference {time.time() - t0:.0f} s")
print(f"FLUX-schnell {SIDE * 8}x{SIDE * 8}, {STEPS} steps: PSNR {p:.1f} dB (limit 30) -> {'PASS' if p >= 30 else 'FAIL'}")
sys.exit(0 if p >= 30 else 1)
