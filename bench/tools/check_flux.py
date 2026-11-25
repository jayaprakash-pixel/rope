"""P2.0.4: one rope FLUX transformer call vs diffusers FluxTransformer2DModel, run one block at a time on the same
weights (flux1-schnell Q4_K GGUF, dequantized by llama.cpp's `gguf` package; the fp32 model, 48 GB, does not fit).
BFL -> diffusers names come from diffusers' own convert_flux_transformer_checkpoint_to_diffusers; the tensors not
needed for the block being converted are `meta` placeholders (shape only).
Inputs: 256x256 image (latent [16,32,32], seeded), t = 1.0, T5 + CLIP-L pooled from rope (verified in P2.0.1/P2.0.3).
Pass: max |rope - ref| / max |ref| <= 1e-2."""
import os, subprocess, sys, tempfile, time
import numpy as np, torch
from gguf import GGUFReader
from gguf.quants import dequantize
from diffusers import FluxTransformer2DModel
from diffusers.loaders.single_file_utils import convert_flux_transformer_checkpoint_to_diffusers

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
ROPE = os.path.join(R, "build", "rust-cpu", "release", "rope.exe")
M = os.path.join(R, "models")
MODEL = os.path.join(M, "flux1-schnell-q4_k.gguf")
TMP = tempfile.gettempdir()
PROMPT, SIDE, T = "a red bicycle leaning on a brick wall, photo", 32, 1.0
f = lambda n: os.path.join(TMP, f"rope_flux_{n}.f32")

# ---- inputs (rope's verified encoders)
for cmd in ([ROPE, "encode-t5", os.path.join(M, "t5-v1_1-xxl-encoder-Q4_K_M.gguf"), os.path.join(M, "t5-config"), PROMPT, f("t5")],
            [ROPE, "encode-text", os.path.join(M, "clip_l.safetensors"), os.path.join(M, "sd15-config", "tokenizer"), PROMPT, f("clip")]):
    r = subprocess.run(cmd, capture_output=True, text=True); assert r.returncode == 0, r.stderr
lat = np.random.default_rng(3).standard_normal((16, SIDE, SIDE)).astype("<f4")
lat.tofile(f("lat"))
txt = torch.from_numpy(np.fromfile(f("t5"), dtype="<f4").reshape(1, 256, 4096))
pooled = torch.from_numpy(np.fromfile(f("clip") + ".pooled", dtype="<f4").reshape(1, 768))

# ---- rope
t0 = time.time()
r = subprocess.run([ROPE, "flux", MODEL, f("lat"), str(T), f("t5"), f("clip") + ".pooled", f("out")], capture_output=True, text=True)
assert r.returncode == 0, r.stderr
got = np.fromfile(f("out"), dtype="<f4").reshape(16, SIDE, SIDE).astype(np.float64)
print(f"rope: {r.stderr.strip()}")

# ---- reference
rd = GGUFReader(MODEL)
TT = {t.name: t for t in rd.tensors}
shape = lambda t: [int(x) for x in reversed(t.shape)]
real = lambda n: torch.from_numpy(np.ascontiguousarray(dequantize(TT[n].data, TT[n].tensor_type)).astype(np.float32).reshape(shape(TT[n])))
meta = lambda n: torch.empty(shape(TT[n]), device="meta")
top = [n for n in TT if not n.startswith(("double_blocks.", "single_blocks."))]

def converted(kind, i):
    """diffusers state dict for: top-level (kind None) or one block (renamed to index 0), others as meta."""
    ck = {n: (real(n) if kind is None else meta(n)) for n in top}
    for k in ("double_blocks", "single_blocks"):
        for n in TT:
            if n.startswith(f"{k}.{i if k == kind else 0}."):
                ck[f"{k}.0." + n.split(".", 2)[2]] = real(n) if k == kind else meta(n)
    return convert_flux_transformer_checkpoint_to_diffusers(ck)

with torch.device("meta"):
    model = FluxTransformer2DModel.from_config(FluxTransformer2DModel.load_config(os.path.join(R, "bench", "tools", "flux_transformer_config.json")))
model.eval()
sd = {k: v for k, v in converted(None, 0).items() if not k.startswith(("transformer_blocks.", "single_transformer_blocks."))}
missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
assert not unexpected and all(k.startswith(("transformer_blocks.", "single_transformer_blocks.")) for k in missing), (unexpected[:3], [k for k in missing if not k.startswith("transformer")][:3])

def run_block(blk, kind, i, **kw):
    pre = "transformer_blocks.0." if kind == "double_blocks" else "single_transformer_blocks.0."
    bsd = {k[len(pre):]: v for k, v in converted(kind, i).items() if k.startswith(pre)}
    m, u = blk.load_state_dict(bsd, strict=True, assign=True)
    out = blk(**kw)
    blk.to("meta")
    return out

t1 = time.time()
with torch.no_grad():
    x = torch.from_numpy(lat)[None]
    packed = x.view(1, 16, SIDE // 2, 2, SIDE // 2, 2).permute(0, 2, 4, 1, 3, 5).reshape(1, (SIDE // 2) ** 2, 64)
    hidden = model.x_embedder(packed)
    temb = model.time_text_embed(torch.tensor([T]) * 1000, pooled)
    enc = model.context_embedder(txt)
    img_ids = torch.zeros(SIDE // 2, SIDE // 2, 3)
    img_ids[..., 1] += torch.arange(SIDE // 2)[:, None]
    img_ids[..., 2] += torch.arange(SIDE // 2)[None, :]
    rot = model.pos_embed(torch.cat([torch.zeros(256, 3), img_ids.reshape(-1, 3)], 0))
    for i, blk in enumerate(model.transformer_blocks):
        enc, hidden = run_block(blk, "double_blocks", i, hidden_states=hidden, encoder_hidden_states=enc, temb=temb, image_rotary_emb=rot)
    for i, blk in enumerate(model.single_transformer_blocks):
        enc, hidden = run_block(blk, "single_blocks", i, hidden_states=hidden, encoder_hidden_states=enc, temb=temb, image_rotary_emb=rot)
    out = model.proj_out(model.norm_out(hidden, temb))
    ref = out.reshape(1, SIDE // 2, SIDE // 2, 16, 2, 2).permute(0, 3, 1, 4, 2, 5).reshape(16, SIDE, SIDE).numpy().astype(np.float64)
print(f"reference: {len(model.transformer_blocks)} double + {len(model.single_transformer_blocks)} single diffusers blocks, {time.time() - t1:.0f} s")
err = np.abs(got - ref).max() / np.abs(ref).max()
print(f"max rel err {err:.2e}, mean abs err {np.abs(got - ref).mean():.2e}, ref std {ref.std():.3f} -> {'PASS' if err <= 1e-2 else 'FAIL'}")
sys.exit(0 if err <= 1e-2 else 1)
