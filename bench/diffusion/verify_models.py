"""Verify downloaded model files against the sha256 that Hugging Face publishes (LFS oid).
  python verify_models.py            -> checks every file listed below that exists in models/
"""
import hashlib, json, os, sys, urllib.request

MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "models")
FILES = {  # local name -> (repo, path in repo)
    "sd_turbo.safetensors": ("stabilityai/sd-turbo", "sd_turbo.safetensors"),
    "wan2.1_t2v_1.3b-q4_k_m.gguf": ("calcuis/wan-1.3b-gguf", "wan2.1_t2v_1.3b-q4_k_m.gguf"),
    "umt5-xxl-encoder-Q4_K_M.gguf": ("city96/umt5-xxl-encoder-gguf", "umt5-xxl-encoder-Q4_K_M.gguf"),
    "wan_2.1_vae.safetensors": ("Comfy-Org/Wan_2.1_ComfyUI_repackaged", "split_files/vae/wan_2.1_vae.safetensors"),
    "flux1-schnell-q4_k.gguf": ("leejet/FLUX.1-schnell-gguf", "flux1-schnell-q4_k.gguf"),
    "t5-v1_1-xxl-encoder-Q4_K_M.gguf": ("city96/t5-v1_1-xxl-encoder-gguf", "t5-v1_1-xxl-encoder-Q4_K_M.gguf"),
    "clip_l.safetensors": ("comfyanonymous/flux_text_encoders", "clip_l.safetensors"),
    "ae.safetensors": ("Comfy-Org/z_image_turbo", "split_files/vae/ae.safetensors"),  # official BFL repo is gated (401) and masks its hash; same sha256 in 2 independent mirrors
    "taesd.safetensors": ("madebyollin/taesd", "diffusion_pytorch_model.safetensors"),
}


def expected(repo, path):
    d = os.path.dirname(path)
    url = f"https://huggingface.co/api/models/{repo}/tree/main" + (f"/{d}" if d else "")
    with urllib.request.urlopen(url) as r:
        for x in json.load(r):
            if x["path"] == path:
                return (x.get("lfs") or {}).get("oid"), (x.get("lfs") or {}).get("size", x.get("size"))
    return None, None


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


bad = 0
for name, (repo, path) in FILES.items():
    p = os.path.join(MODELS, name)
    if not os.path.exists(p):
        continue
    want, size = expected(repo, path)
    have = os.path.getsize(p)
    if size and have < size:
        print(f"INCOMPLETE {name}: {have}/{size} bytes"); continue
    ok = want is not None and sha256(p) == want
    bad += not ok
    print(f"{'OK  ' if ok else 'FAIL'} {name}  {have} bytes  sha256 {want}")
sys.exit(1 if bad else 0)
