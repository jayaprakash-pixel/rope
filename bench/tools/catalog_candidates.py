"""Grow the model catalog without downloading models: for each candidate, read the Hugging Face metadata (size, sha256,
licence, gated or not) and only the file header (a few KB via an HTTP range request), then check that rope can run it:
the architecture (same rules as engine/rust/detect.rs), every part a single-file checkpoint needs (UNet, text encoder(s),
VAE), and weight types rope supports. Candidates that pass are written to catalog/models.json with "tested": false.

    python bench/tools/catalog_candidates.py            (prints the verdicts; --write adds the accepted ones)
"""
import json, os, struct, sys, urllib.error, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CATALOG = os.path.join(ROOT, "catalog", "models.json")
UA = {"User-Agent": "rope-catalog-check"}

S1 = {"fast": 15, "balanced": 20, "best": 30}
XL = {"fast": 15, "balanced": 25, "best": 40}
# id, family, title, repo, path, extra settings (steps / guidance / native_size), short note
CANDIDATES = [
    ("dreamshaper-8", "sd1", "DreamShaper 8 (SD 1.5 fine-tune, all-round)", "Lykon/DreamShaper", "DreamShaper_8_pruned.safetensors", {"steps": S1, "guidance": 7.0}),
    ("absolutereality-1.8.1", "sd1", "AbsoluteReality 1.8.1 (SD 1.5 fine-tune, photo)", "Lykon/AbsoluteReality", "AbsoluteReality_1.8.1_pruned.safetensors", {"steps": S1, "guidance": 7.0}),
    ("realistic-vision-5.1", "sd1", "Realistic Vision 5.1 (SD 1.5 fine-tune, photo)", "SG161222/Realistic_Vision_V5.1_noVAE", "Realistic_Vision_V5.1_fp16-no-ema.safetensors", {"steps": S1, "guidance": 7.0}),
    ("realistic-vision-6.0", "sd1", "Realistic Vision 6.0 B1 (SD 1.5 fine-tune, photo)", "SG161222/Realistic_Vision_V6.0_B1_noVAE", "Realistic_Vision_V6.0_NV_B1_fp16.safetensors", {"steps": S1, "guidance": 7.0}),
    ("anything-v3", "sd1", "Anything V3 (SD 1.5 fine-tune, anime)", "Linaqruf/anything-v3.0", "anything-v3-fp16-pruned.safetensors", {"steps": S1, "guidance": 7.0}),
    ("deliberate-v2", "sd1", "Deliberate v2 (SD 1.5 fine-tune)", "XpucT/Deliberate", "Deliberate_v2.safetensors", {"steps": S1, "guidance": 7.0}),
    ("sd-2.1-base", "sd2", "Stable Diffusion 2.1 base (512)", "stabilityai/stable-diffusion-2-1-base", "v2-1_512-ema-pruned.safetensors", {"steps": S1, "guidance": 7.5}),
    ("sdxl-base-1.0", "sdxl", "Stable Diffusion XL base 1.0", "stabilityai/stable-diffusion-xl-base-1.0", "sd_xl_base_1.0.safetensors", {"steps": XL, "guidance": 5.0}),
    ("dreamshaper-xl-turbo", "sdxl", "DreamShaper XL Turbo v2.1 (SDXL, few steps)", "Lykon/dreamshaper-xl-v2-turbo", "DreamShaperXL_Turbo_v2_1.safetensors", {"steps": {"fast": 4, "balanced": 6, "best": 8}, "guidance": 2.0}),
    ("realvisxl-4.0", "sdxl", "RealVisXL 4.0 (SDXL fine-tune, photo)", "SG161222/RealVisXL_V4.0", "RealVisXL_V4.0.safetensors", {"steps": XL, "guidance": 5.0}),
    ("juggernaut-xl-9", "sdxl", "Juggernaut XL v9 (SDXL fine-tune)", "RunDiffusion/Juggernaut-XL-v9", "Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors", {"steps": XL, "guidance": 5.0}),
    ("sdxl-lightning-4step", "sdxl", "SDXL-Lightning 4-step (SDXL distilled)", "ByteDance/SDXL-Lightning", "sdxl_lightning_4step.safetensors", {"steps": {"fast": 4, "balanced": 4, "best": 4}, "guidance": 1.0}),
    ("flux-schnell-q8", "flux", "FLUX.1-schnell Q8_0 (12B, 4 steps, best quality of the quantized files)", "leejet/FLUX.1-schnell-gguf", "flux1-schnell-q8_0.gguf", {}),
    ("flux-schnell-q4-0", "flux", "FLUX.1-schnell Q4_0 (12B, 4 steps)", "leejet/FLUX.1-schnell-gguf", "flux1-schnell-q4_0.gguf", {}),
    ("flux-schnell-q6k", "flux", "FLUX.1-schnell Q6_K (12B, 4 steps)", "city96/FLUX.1-schnell-gguf", "flux1-schnell-Q6_K.gguf", {}),
]

def http(url, rng=None):
    h = dict(UA)
    if rng: h["Range"] = f"bytes={rng[0]}-{rng[1]}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=60) as r: return r.read()

def st_header(repo, path):
    url = f"https://huggingface.co/{repo}/resolve/main/{path}"
    n = struct.unpack("<Q", http(url, (0, 7)))[0]
    if n > 50_000_000: raise ValueError("header too large")
    return json.loads(http(url, (8, 8 + n - 1)))

GG_TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K",
            13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 30: "BF16"}
def gguf_tensors(repo, path):
    b = http(f"https://huggingface.co/{repo}/resolve/main/{path}", (0, 8 * 1024 * 1024 - 1))
    o = 0
    def rd(fmt):
        nonlocal o
        v = struct.unpack_from("<" + fmt, b, o); o += struct.calcsize("<" + fmt); return v[0]
    def rstr():
        nonlocal o
        n = rd("Q"); s = b[o:o + n].decode("utf-8", "replace"); o += n; return s
    def rval(t):
        sizes = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
        if t in sizes: return rd(sizes[t])
        if t == 8: return rstr()
        if t == 9:
            et, n = rd("I"), rd("Q"); return [rval(et) for _ in range(n)]
        raise ValueError(f"gguf value type {t}")
    if b[:4] != b"GGUF": raise ValueError("not GGUF")
    o = 4; rd("I"); nt, nkv = rd("Q"), rd("Q")
    for _ in range(nkv): rstr(); rval(rd("I"))
    out = {}
    for _ in range(nt):
        name = rstr(); nd = rd("I"); dims = [rd("Q") for _ in range(nd)]; t = rd("I"); rd("Q")
        out[name] = (GG_TYPES.get(t, f"type{t}"), dims)
    return out

def check(family, path, repo):
    """(problems, denoiser bytes): problems empty = rope can run it; the denoiser's weight bytes size the planner's paging."""
    if path.endswith(".gguf"):
        t = gguf_tensors(repo, path)
        bad = sorted({ty for ty, _ in t.values() if ty not in ("F32", "F16", "BF16", "Q4_0", "Q8_0", "Q4_K", "Q6_K")})
        p = [] if "double_blocks.0.img_attn.qkv.weight" in t else ["not a FLUX transformer"]
        if any(k.startswith("guidance_in.") for k in t): p.append("FLUX-dev (guidance-distilled): rope runs schnell only")
        if bad: p.append(f"weight types rope does not support: {', '.join(bad)}")
        bpe = {"F32": 4, "F16": 2, "BF16": 2, "Q4_0": 18 / 32, "Q8_0": 34 / 32, "Q4_K": 144 / 256, "Q6_K": 210 / 256}
        n = 0
        for ty, dims in t.values():
            e = 1
            for d in dims: e *= d
            n += e * bpe.get(ty, 0)
        return p, int(n)
    h = st_header(repo, path); h.pop("__metadata__", None)
    has = lambda k: k in h
    p = []
    # integer tensors named *position_ids (CLIP's 0..76 table) are allowed: rope computes positions and never reads them
    bad = sorted({v["dtype"] for n, v in h.items() if v["dtype"] not in ("F32", "F16", "BF16") and not n.endswith("position_ids")})
    if bad: p.append(f"weight types rope does not support: {', '.join(bad)}")
    if not has("model.diffusion_model.input_blocks.0.0.weight"): return p + ["not a single-file SD checkpoint (no model.diffusion_model.*)"], 0
    if not has("first_stage_model.decoder.conv_in.weight"): p.append("no VAE inside (first_stage_model.*)")
    xl = has("model.diffusion_model.label_emb.0.0.weight")
    k = h.get("model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight")
    fam = "sdxl" if xl else {768: "sd1", 1024: "sd2"}.get(k["shape"][-1] if k else 0, "unknown")
    if fam != family: p.append(f"architecture is {fam}, expected {family}")
    if fam == "sd1" and not any(x.startswith("cond_stage_model.transformer.") for x in h): p.append("no CLIP-L text encoder inside")
    if fam == "sd2" and not has("cond_stage_model.model.token_embedding.weight"): p.append("no OpenCLIP-H text encoder inside")
    if fam == "sdxl" and not (has("conditioner.embedders.1.model.text_projection") and any(x.startswith("conditioner.embedders.0.transformer.") for x in h)):
        p.append("SDXL text encoders missing")
    if fam == "sd2" and has("model.diffusion_model.input_blocks.0.0.weight") and "512" not in path: p.append("SD 2.x 768 model (v-prediction): not supported")
    unet = sum(v["data_offsets"][1] - v["data_offsets"][0] for n, v in h.items() if n.startswith("model.diffusion_model."))
    return p, unet

def meta(repo, path):
    info = json.loads(http(f"https://huggingface.co/api/models/{repo}"))
    tree = json.loads(http(f"https://huggingface.co/api/models/{repo}/tree/main"))
    f = next((x for x in tree if x.get("path") == path), None)
    lic = (info.get("cardData") or {}).get("license") or next((t[8:] for t in info.get("tags", []) if t.startswith("license:")), "unknown")
    return info.get("gated", False), f, lic

def main():
    cat = json.load(open(CATALOG, encoding="utf-8"))
    have = {m["id"] for m in cat["models"]}
    accepted = []
    for mid, fam, title, repo, path, extra in CANDIDATES:
        if mid in have: print(f"SKIP   {mid}: already in the catalog"); continue
        try:
            gated, f, lic = meta(repo, path)
            if gated: print(f"REJECT {mid}: gated on Hugging Face (needs a login)"); continue
            if not f or not f.get("lfs"): print(f"REJECT {mid}: file {path} not found in {repo}"); continue
            probs, unet = check(fam, path, repo)
        except urllib.error.HTTPError as e:
            print(f"REJECT {mid}: HTTP {e.code}" + (" (gated or private: needs a login)" if e.code in (401, 403) else "")); continue
        except Exception as e:
            print(f"REJECT {mid}: {type(e).__name__}: {e}"); continue
        if probs: print(f"REJECT {mid}: {'; '.join(probs)}"); continue
        size, sha = f["lfs"]["size"], f["lfs"]["oid"]
        print(f"ACCEPT {mid}: {size / 2**30:.2f} GiB (denoiser {unet / 2**30:.2f} GiB), licence {lic}")
        m = {"id": mid, "family": fam, "kind": "image", "title": title, **extra,
             "files": [{"role": "model", "name": path, "repo": repo, "path": path, "bytes": size, "sha256": sha}],
             "unet_bytes": unet, "license": lic, "tested": False,
             "_checked": "header and Hugging Face metadata only (architecture, text encoders, VAE, weight types); not run on rope yet"}
        accepted.append(m)
    print(f"\n{len(accepted)} accepted")
    if "--write" in sys.argv and accepted:
        # insert before the first video model, so image models stay together
        i = next((n for n, m in enumerate(cat["models"]) if m.get("kind") == "video"), len(cat["models"]))
        cat["models"][i:i] = accepted
        with open(CATALOG, "w", encoding="utf-8", newline="\n") as fh: json.dump(cat, fh, indent=1, ensure_ascii=False); fh.write("\n")
        print(f"written to {CATALOG}")

if __name__ == "__main__":
    main()
