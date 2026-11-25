"""Independent reference parser for the P1.5.2 acceptance test.
safetensors: Python's own json module.
GGUF: a plain-Python struct parser whose type table comes from gguf-py's constants (not from rope's C table);
      `--gguf-py` uses the official gguf-py GGUFReader instead (it refuses tensors with > 4 dims).
Prints the same canonical listing as `rope inspect`, so the two outputs can be diffed byte for byte.
  python ref_modelfile.py <file> [--gguf-py]
"""
import json, struct, sys, os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "repos", "llama.cpp", "gguf-py"))


def safetensors(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    hdr.pop("__metadata__", None)
    base = 8 + n
    print(f"format=safetensors n_tensors={len(hdr)} data_offset={base}")
    for name, t in hdr.items():   # json keeps file order
        b, e = t["data_offsets"]
        print(f"{name}\t{t['dtype']}\t{','.join(str(d) for d in t['shape'])}\t{base + b}\t{e - b}")


def gguf_plain(path):
    from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType
    sizes = {int(k): v for k, v in GGML_QUANT_SIZES.items()}
    scalar = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    with open(path, "rb") as f:
        rd = lambda fmt: struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))
        rs = lambda: f.read(rd("Q")[0]).decode("utf-8")

        def skip(t):
            if t == 8: rs()
            elif t == 9:
                et, n = rd("IQ")
                if et in scalar: f.seek(scalar[et] * n, 1)
                else:
                    for _ in range(n): skip(et)
            else: f.seek(scalar[t], 1)

        assert f.read(4) == b"GGUF"
        _ver, nt, nkv = rd("IQQ")
        align = 32
        for _ in range(nkv):
            k = rs(); t = rd("I")[0]
            if k == "general.alignment" and t == 4: align = rd("I")[0]
            else: skip(t)
        infos = []
        for _ in range(nt):
            name = rs(); nd = rd("I")[0]; dims = rd(f"{nd}Q"); typ, off = rd("IQ")
            infos.append((name, typ, dims, off))
        base = (f.tell() + align - 1) // align * align
    print(f"format=gguf n_tensors={nt} data_offset={base}")
    for name, typ, dims, off in infos:
        blk, nb = sizes[typ]
        n = 1
        for d in dims: n *= d
        print(f"{name}\t{GGMLQuantizationType(typ).name}\t{','.join(map(str, dims))}\t{base + off}\t{n // blk * nb}")


def gguf_py(path):
    from gguf import GGUFReader
    r = GGUFReader(path)
    print(f"format=gguf n_tensors={len(r.tensors)} data_offset={r.data_offset}")
    for t in r.tensors:
        print(f"{t.name}\t{t.tensor_type.name}\t{','.join(str(int(d)) for d in t.shape)}\t{t.data_offset}\t{t.n_bytes}")


if __name__ == "__main__":
    sys.stdout.reconfigure(newline="\n")   # plain LF, like rope (Windows text mode would write CRLF)
    p = sys.argv[1]
    with open(p, "rb") as f:
        magic = f.read(4)
    if magic == b"GGUF":
        (gguf_py if "--gguf-py" in sys.argv else gguf_plain)(p)
    else:
        safetensors(p)
