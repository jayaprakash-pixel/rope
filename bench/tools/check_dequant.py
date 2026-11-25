"""Cross-check rope's dequantizer against gguf-py's official dequantizer on real tensors.
For each tensor of each requested type: raw bytes -> gguf.quants.dequantize  vs  `rope dump` -> compare.
  python check_dequant.py <model.gguf> [max_tensors_per_type=2] [max_values=1048576]
"""
import os, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "repos", "llama.cpp", "gguf-py"))
from gguf.quants import dequantize
from gguf.constants import GGMLQuantizationType, GGML_QUANT_SIZES

ROPE = os.path.join(HERE, "..", "..", "build", "rust-cpu", "release", "rope.exe")
SUPPORTED = {"F16", "BF16", "Q4_0", "Q8_0", "Q4_K", "Q6_K"}

path = sys.argv[1]
per_type = int(sys.argv[2]) if len(sys.argv) > 2 else 2
max_vals = int(sys.argv[3]) if len(sys.argv) > 3 else 1 << 20
lines = subprocess.run([ROPE, "inspect", path], capture_output=True, text=True, encoding="utf-8", check=True).stdout.splitlines()[1:]
seen, worst, ok = {}, 0.0, True
for ln in lines:
    name, dt, shape, off, nb = ln.split("\t")
    if dt not in SUPPORTED or seen.get(dt, 0) >= per_type:
        continue
    seen[dt] = seen.get(dt, 0) + 1
    qt = GGMLQuantizationType[dt]
    row = int(shape.split(",")[0]); blk, bb = GGML_QUANT_SIZES[qt]
    rows = max(1, min(int(nb) // (row // blk * bb), max_vals // row))
    with open(path, "rb") as f:
        f.seek(int(off)); raw = np.frombuffer(f.read(rows * (row // blk * bb)), dtype=np.uint8)
    ref = dequantize(raw.reshape(rows, -1), qt).astype(np.float64).ravel()
    got = np.loadtxt(subprocess.run([ROPE, "dump", path, name, str(ref.size)], capture_output=True, text=True, check=True).stdout.splitlines())
    err = np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-30)
    worst = max(worst, err); ok &= err <= 1e-6
    print(f"{dt:5s} {name:45s} {ref.size:9d} values  max rel diff {err:.2e}  {'OK' if err <= 1e-6 else 'MISMATCH'}")
print(f"types checked: {sorted(seen)}  worst {worst:.2e}  -> {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
