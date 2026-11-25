"""P2.4.1.1: rope's own quantizers and GGUF writer.
 1. raw matrices (gaussian, heavy-tailed): Q8_0 bytes vs the gguf package's quantizer (>= 99.99% identical);
    Q4_K bytes dequantized by the gguf package == rope's dequantization (bit for bit) and rel RMSE below Q4_0's.
 2. `rope quantize` on a file that exercises every rule: denoiser linears -> Q4_K (Q8_0 when rows do not split into 256),
    3x3 convs / text encoder / 1-D / scalars kept as stored (bit-identical); opens in GGUFReader and rope inspect."""
import os, subprocess, sys, tempfile
import numpy as np
from gguf import GGUFReader, GGMLQuantizationType as Q
from gguf.quants import quantize, dequantize

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "..", "..")
ROPE = os.path.join(R, "build", "rust-cpu", "release", "rope.exe")
TMP = tempfile.gettempdir()
ok = True
rel = lambda y, x: np.sqrt(np.mean((y - x) ** 2)) / np.sqrt(np.mean(x ** 2))

rng = np.random.default_rng(0)
for name, x in [("gaussian", rng.standard_normal((64, 4096))), ("heavy-tailed (t, 3 dof)", rng.standard_t(3, (64, 4096)))]:
    x = x.astype(np.float32)
    src, out = os.path.join(TMP, "rq_in.f32"), os.path.join(TMP, "rq_out.bin")
    x.tofile(src)
    for t, code in [(Q.Q8_0, 8), (Q.Q4_K, 12)]:
        r = subprocess.run([ROPE, "quant-test", str(code), src, "64", "4096", out], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        mine = np.fromfile(out, dtype=np.uint8)
        deq_rope = np.fromfile(out + ".deq", dtype="<f4").reshape(64, 4096)
        deq_ref = dequantize(mine.reshape(64, -1), t).reshape(64, 4096)
        same_decode = np.array_equal(deq_rope, deq_ref)
        if t == Q.Q8_0:
            ref = np.asarray(quantize(x, t)).reshape(-1).view(np.uint8)
            frac = np.mean(mine == ref)
            good = frac >= 0.9999 and same_decode
            print(f"{name:24s} Q8_0: {frac:.6f} of bytes identical to gguf's quantizer, decode identical {same_decode}, rel RMSE {rel(deq_rope, x):.4f} -> {'ok' if good else 'FAIL'}")
        else:
            q40 = rel(dequantize(quantize(x, Q.Q4_0), Q.Q4_0).reshape(64, 4096), x)
            e = rel(deq_rope, x)
            good = same_decode and e < q40
            print(f"{name:24s} Q4_K: gguf decode == rope decode {same_decode}, rel RMSE {e:.4f} (Q4_0: {q40:.4f}) -> {'ok' if good else 'FAIL'}")
        ok &= good

# a file that exercises every rule (P2.4.2): which tensors are quantized, to which type, and what is kept as stored
from safetensors.numpy import save_file
g = np.random.default_rng(1)
f16 = lambda *s: (g.standard_normal(s) * 0.05).astype(np.float16)
src = os.path.join(TMP, "rq_rules.safetensors")
expect = {   # name: (array, expected GGUF type)
    "double_blocks.0.img_attn.qkv.weight": (f16(256, 3072), Q.Q4_K),                              # denoiser linear
    "model.diffusion_model.input_blocks.4.1.proj_in.weight": (f16(640, 640), Q.Q8_0),             # 640 rows: Q8_0 fallback
    "model.diffusion_model.input_blocks.1.0.in_layers.2.weight": (f16(64, 64, 3, 3), Q.F16),      # 3x3 conv: kept
    "conditioner.embedders.1.model.transformer.resblocks.0.attn.in_proj_weight": (f16(768, 256), Q.F16),  # text encoder: kept
    "model.diffusion_model.out.2.bias": (f16(4), Q.F16),                                          # 1-D: kept
    "conditioner.embedders.0.model.logit_scale": (np.array(4.6, dtype=np.float32), Q.F32),        # scalar: kept, shape [1]
}
save_file({k: v for k, (v, _) in expect.items()}, src)
out = os.path.join(TMP, "rq_rules.gguf")
r = subprocess.run([ROPE, "quantize", src, out, "--type", "q4_k"], capture_output=True, text=True)
assert r.returncode == 0, r.stderr
print(r.stderr.strip().splitlines()[-1])
rd = GGUFReader(out)
good = len(rd.tensors) == len(expect)
for t in rd.tensors:
    o, want = expect[t.name]
    o = o.astype(np.float32).reshape(-1)
    got = np.asarray(dequantize(t.data, t.tensor_type), dtype=np.float32).reshape(-1)
    if t.tensor_type in (Q.F32, Q.F16):
        fine = t.tensor_type == want and np.array_equal(got, o)
        how = "kept, identical"
    else:
        o2 = o.reshape(-1, expect[t.name][0].shape[-1])                       # rows along the innermost dimension
        q40 = rel(np.asarray(dequantize(quantize(o2, Q.Q4_0), Q.Q4_0)).reshape(-1), o)
        e = rel(got, o)
        fine = t.tensor_type == want and e < q40
        how = f"rel RMSE {e:.4f} (Q4_0 {q40:.4f})"
    print(f"  {t.name[:70]:70s} {t.tensor_type.name:5s} (want {want.name}) {how} -> {'ok' if fine else 'FAIL'}")
    good &= fine
ins = subprocess.run([ROPE, "inspect", out], capture_output=True, text=True, encoding="utf-8")
good &= len(ins.stdout.strip().splitlines()) - 1 == len(expect)
print(f"rules file -> {'ok' if good else 'FAIL'}")
ok &= good
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
