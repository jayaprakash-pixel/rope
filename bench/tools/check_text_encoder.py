"""P1.5.5.2 / P1.7.3.2: rope's text encoder vs transformers CLIPTextModel.
Default: SD-Turbo OpenCLIP-H (loaded through diffusers' converter). Arg "sd15": SD 1.5 CLIP ViT-L (plain HF names).
Arg "clipl" (P2.0.1): standalone FLUX clip_l.safetensors (F16), hidden states AND pooled output.
Pass: max |rope - ref| / max |ref| <= 1e-3 on every prompt."""
import os, subprocess, sys, tempfile
import numpy as np, torch
from safetensors import safe_open
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer
from diffusers.loaders.single_file_utils import convert_open_clip_checkpoint

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
ROPE = os.path.join(R, "build", "rust-cpu", "release", "rope.exe")
MODE = sys.argv[1] if len(sys.argv) > 1 else ""
SD15, CLIPL = MODE == "sd15", MODE == "clipl"
MODEL = os.path.join(R, "models", "clip_l.safetensors" if CLIPL else "v1-5-pruned-emaonly.safetensors" if SD15 else "sd_turbo.safetensors")
CFG = os.path.join(R, "models", "sd15-config" if SD15 or CLIPL else "sd-turbo-config")
PREFIX = "" if CLIPL else "cond_stage_model.transformer." if SD15 else "conditioner.embedders.0.model."
PROMPTS = ["a red bicycle leaning on a brick wall, photo", "", "a photo of an astronaut riding a horse on mars, 8k, detailed"]

ckpt = {}
with safe_open(MODEL, "pt") as f:
    for k in f.keys():
        if k.startswith(PREFIX):
            ckpt[k] = f.get_tensor(k).float()
model = CLIPTextModel(CLIPTextConfig.from_pretrained(os.path.join(CFG, "text_encoder")))
sd = {k.removeprefix(PREFIX): v for k, v in ckpt.items()} if SD15 or CLIPL else convert_open_clip_checkpoint(model, ckpt, prefix=PREFIX)
# diffusers names keys "text_model.*"; transformers 5 CLIPTextModel has no such prefix -> align names only.
own = set(model.state_dict().keys())
sd = {(k if k in own else k.removeprefix("text_model.")): v for k, v in sd.items()}
missing, unexpected = model.load_state_dict(sd, strict=False)
missing = [m for m in missing if "position_ids" not in m]
assert not missing, f"reference model missing weights: {missing[:5]}"
print(f"reference loaded {len(sd)} tensors; unexpected: {unexpected}")
model.eval()
tok = CLIPTokenizer.from_pretrained(os.path.join(CFG, "tokenizer"))
worst, ok = 0.0, True
for p in PROMPTS:
    ids = tok(p, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids
    with torch.no_grad():
        o = model(ids)
        ref, ref_pool = o.last_hidden_state[0].numpy().astype(np.float64), o.pooler_output[0].numpy().astype(np.float64)
    out = os.path.join(tempfile.gettempdir(), "rope_te.f32")
    r = subprocess.run([ROPE, "encode-text", MODEL, os.path.join(CFG, "tokenizer"), p, out], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    got = np.fromfile(out, dtype="<f4").reshape(77, -1).astype(np.float64)
    err = np.abs(got - ref).max() / np.abs(ref).max()
    if CLIPL:
        pool = np.fromfile(out + ".pooled", dtype="<f4").astype(np.float64)
        err = max(err, np.abs(pool - ref_pool).max() / np.abs(ref_pool).max())
    worst = max(worst, err); ok &= err <= 1e-3
    print(f"{p[:45]!r:50s} max rel err {err:.2e}  {r.stderr.strip()}")
print(f"worst {worst:.2e} -> {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
