"""P2.3.1: rope profile (dry counting run of rope's own model code) vs PyTorch's FlopCounterMode over the reference
diffusers / transformers models on the meta device (shapes only, no weights), at 512x512.
Pass: per-call GFLOPs of every denoiser within 10%; also reports text encoders and VAE decoders."""
import json, os, re, subprocess, sys
import torch
from torch.utils.flop_counter import FlopCounterMode
from transformers import CLIPTextConfig, CLIPTextModel, T5Config, T5EncoderModel
from diffusers import UNet2DConditionModel, FluxTransformer2DModel, AutoencoderKL

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "..", "..")
ROPE = os.path.join(R, "build", "rust-cpu", "release", "rope.exe")
M = os.path.join(R, "models")

def gflop(fn):
    with torch.device("meta"), torch.no_grad():
        fc = FlopCounterMode(display=False)
        with fc:
            fn()
    return fc.get_total_flops() / 1e9

def meta_model(cls, cfg):
    with torch.device("meta"):
        return cls.from_config(cfg).eval() if hasattr(cls, "from_config") else cls(cfg).eval()

def unet(cfgdir, ctx_w):
    m = meta_model(UNet2DConditionModel, UNet2DConditionModel.load_config(os.path.join(M, cfgdir, "unet")))
    return gflop(lambda: m(torch.zeros(1, 4, 64, 64), torch.tensor([999.0]), encoder_hidden_states=torch.zeros(1, 77, ctx_w)))

def clip(cfgdir, layers=None):
    c = CLIPTextConfig.from_pretrained(os.path.join(M, cfgdir, "text_encoder"))
    if layers: c.num_hidden_layers = layers
    m = meta_model(CLIPTextModel, c)
    return gflop(lambda: m(torch.zeros(1, 77, dtype=torch.long)))

def vae(cfg):
    m = meta_model(AutoencoderKL, cfg)
    return gflop(lambda: m.decode(torch.zeros(1, cfg["latent_channels"], 64, 64)))

def flux():
    m = meta_model(FluxTransformer2DModel, FluxTransformer2DModel.load_config(os.path.join(HERE, "flux_transformer_config.json")))
    return gflop(lambda: m(hidden_states=torch.zeros(1, 1024, 64), encoder_hidden_states=torch.zeros(1, 256, 4096),
                           pooled_projections=torch.zeros(1, 768), timestep=torch.tensor([1.0]), img_ids=torch.zeros(1024, 3), txt_ids=torch.zeros(256, 3)))

def t5():
    m = meta_model(T5EncoderModel, T5Config.from_pretrained(os.path.join(M, "t5-config")))
    return gflop(lambda: m(torch.zeros(1, 256, dtype=torch.long)))

def rope(*args):
    out = subprocess.run([ROPE, "profile", *args], capture_output=True, text=True, check=True).stdout
    return {m.group(1): float(m.group(2)) for m in re.finditer(r"^\s+(text|denoiser|vae)\s+\d+\s+[\d.]+ GiB\s+([\d.]+)", out, re.M)}

flux_vae = AutoencoderKL.load_config(os.path.join(HERE, "flux_vae_config.json"))
sd_vae = AutoencoderKL.load_config(os.path.join(M, "sd15-config", "vae"))
cases = [
    ("SD 1.5", rope(os.path.join(M, "v1-5-pruned-emaonly.safetensors")),
     {"denoiser": unet("sd15-config", 768), "text": clip("sd15-config"), "vae": vae(sd_vae)}),
    ("SD-Turbo", rope(os.path.join(M, "sd_turbo.safetensors")),
     {"denoiser": unet("sd-turbo-config", 1024), "text": clip("sd-turbo-config", layers=23), "vae": vae(sd_vae)}),
    ("FLUX", rope(os.path.join(M, "flux1-schnell-q4_k.gguf"), "--t5", os.path.join(M, "t5-v1_1-xxl-encoder-Q4_K_M.gguf"),
                  "--clip", os.path.join(M, "clip_l.safetensors"), "--vae", os.path.join(M, "ae.safetensors")),
     {"denoiser": flux(), "text": t5() + clip("sd15-config"), "vae": vae(flux_vae)}),
]
ok = True
for name, got, ref in cases:
    for comp in ("text", "denoiser", "vae"):
        g, r = got.get(comp, 0.0), ref[comp]
        dev = g / r - 1
        gate = comp == "denoiser"
        ok &= (abs(dev) <= 0.10) if gate else True
        print(f"{name:9s} {comp:9s} rope {g:10.1f}  torch {r:10.1f} GFLOP  {dev:+6.1%}{'  (gated <=10%)' if gate else ''}")
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
