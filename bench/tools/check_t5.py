"""P2.0.3: rope's T5-XXL encoder (GGUF Q4_K_M, memory-mapped) vs transformers' own T5Block modules run one block at a
time on the same weights, dequantized by llama.cpp's `gguf` package (the whole fp32 encoder, 19 GB, does not fit here).
Same token ids (rope's tokenizer is verified identical in P2.0.2). No attention mask, as FLUX encodes.
Pass: max |rope - ref| / max |ref| <= 1e-2 on every prompt; rope's private memory (peak commit) < 1 GB."""
import json, os, subprocess, sys, tempfile, time
import numpy as np, torch
from gguf import GGUFReader
from gguf.quants import dequantize
from transformers import T5Config, T5TokenizerFast
from transformers.models.t5.modeling_t5 import T5Block, T5LayerNorm

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "..", "..")
ROPE = os.path.join(R, "build", "rust-cpu", "release", "rope.exe")
MODEL = os.path.join(R, "models", "t5-v1_1-xxl-encoder-Q4_K_M.gguf")
TOK = os.path.join(R, "models", "t5-config")
PROMPTS = ["a red bicycle leaning on a brick wall, photo", "", "a photo of an astronaut riding a horse on mars, 8k, detailed"]

rd = GGUFReader(MODEL)
T = {t.name: t for t in rd.tensors}
def W(name):  # dequantized, torch layout [out, in] (GGUF ne is innermost-first)
    t = T[name]
    return torch.from_numpy(np.ascontiguousarray(dequantize(t.data, t.tensor_type)).astype(np.float32).reshape([int(x) for x in reversed(t.shape)]))

cfg = T5Config.from_pretrained(TOK)
cfg.dropout_rate = 0.0
tok = T5TokenizerFast.from_pretrained(TOK)
emb = W("token_embd.weight")
final = T5LayerNorm(cfg.d_model, eps=cfg.layer_norm_epsilon)
final.weight.data = W("enc.output_norm.weight")
ids = [tok(p, padding="max_length", max_length=256, truncation=True, return_tensors="pt").input_ids for p in PROMPTS]
hs = [emb[i[0]][None] for i in ids]
pbs = [None] * len(PROMPTS)
t0 = time.time()
with torch.no_grad():
    for l in range(cfg.num_layers):
        blk = T5Block(cfg, has_relative_attention_bias=(l == 0), layer_idx=l).eval()
        p = f"enc.blk.{l}."
        sd = {"layer.0.SelfAttention.q.weight": W(p + "attn_q.weight"), "layer.0.SelfAttention.k.weight": W(p + "attn_k.weight"),
              "layer.0.SelfAttention.v.weight": W(p + "attn_v.weight"), "layer.0.SelfAttention.o.weight": W(p + "attn_o.weight"),
              "layer.0.layer_norm.weight": W(p + "attn_norm.weight"), "layer.1.layer_norm.weight": W(p + "ffn_norm.weight"),
              "layer.1.DenseReluDense.wi_0.weight": W(p + "ffn_gate.weight"), "layer.1.DenseReluDense.wi_1.weight": W(p + "ffn_up.weight"),
              "layer.1.DenseReluDense.wo.weight": W(p + "ffn_down.weight")}
        if l == 0:
            sd["layer.0.SelfAttention.relative_attention_bias.weight"] = W(p + "attn_rel_b.weight")
        missing, unexpected = blk.load_state_dict(sd, strict=False)
        assert not missing and not unexpected, (l, missing, unexpected)
        for j in range(len(PROMPTS)):
            out = blk(hs[j], position_bias=pbs[j])
            hs[j] = out[0]
            if l == 0:
                pbs[j] = out[1]
        del blk, sd
    refs = [final(h)[0].numpy().astype(np.float64) for h in hs]
print(f"reference: {cfg.num_layers} transformers T5Blocks on dequantized GGUF weights, {time.time() - t0:.0f} s")

ok, worst = True, 0.0
for p, ref in zip(PROMPTS, refs):
    out = os.path.join(tempfile.gettempdir(), "rope_t5.f32")
    cmd = [ROPE, "encode-t5", MODEL, TOK, p, out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    got = np.fromfile(out, dtype="<f4").reshape(256, -1).astype(np.float64)
    err = np.abs(got - ref).max() / np.abs(ref).max()
    worst = max(worst, err); ok &= err <= 1e-2
    print(f"{p[:45]!r:50s} max rel err {err:.2e}  mean abs {np.abs(got - ref).mean():.2e}  [{r.stderr.strip()}]")
# private memory of one rope run, measured by the P0 tool
log = subprocess.run([sys.executable, "-I", os.path.join(R, "bench", "diffusion", "measure.py"), "--label", "p203_t5_commit", "--",
                      ROPE, "encode-t5", MODEL, TOK, PROMPTS[0], os.path.join(tempfile.gettempdir(), "rope_t5.f32")],
                     capture_output=True, text=True, cwd=os.path.join(R, "bench", "diffusion"))
rec = json.loads(open(os.path.join(R, "bench", "diffusion", "results.jsonl"), encoding="utf-8").read().strip().splitlines()[-1])
commit = rec["peak_commit_gb"]
ok &= commit < 1.0
print(f"rope peak commit {commit:.2f} GB (limit 1.0), peak working set {rec['peak_ws_gb']:.2f} GB, wall {rec['wall_s']:.1f} s")
print(f"worst {worst:.2e} -> {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
