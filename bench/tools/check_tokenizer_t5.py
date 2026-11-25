"""P2.0.2: rope's T5 unigram tokenizer (reads spiece.model itself) vs transformers T5TokenizerFast built from the
same google/t5-v1_1-xxl spiece.model; FLUX encoding: max_length 256, truncation, padding."""
import os, subprocess, sys
from transformers import T5TokenizerFast

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROPE = os.path.join(HERE, "..", "..", "build", "rust-cpu", "release", "rope.exe")
TOK = os.path.join(HERE, "..", "..", "models", "t5-config")
src = open(os.path.join(HERE, "check_tokenizer.py"), encoding="utf-8").read()
PROMPTS = eval(src[src.index("PROMPTS = [") + 10: src.index("]\n", src.index("PROMPTS = [")) + 1])
PROMPTS += ["a cat</s>and <extra_id_0> a dog<pad>", "  leading and   repeated   spaces  ", "non breaking　ideographic space",
            "Ｆｕｌｌｗｉｄｔｈ ＡＢＣ １２３ and ½ ① ㎏", "word " * 300, "x"]
tok = T5TokenizerFast.from_pretrained(TOK)
bad = 0
for i, p in enumerate(PROMPTS):
    ref = tok(p, padding="max_length", max_length=256, truncation=True).input_ids
    got = list(map(int, subprocess.run([ROPE, "tokenize-t5", TOK, p], capture_output=True, text=True, encoding="utf-8", check=True).stdout.split()))
    if got != ref:
        bad += 1
        k = next((j for j in range(256) if j >= len(got) or got[j] != ref[j]), 0)
        print(f"MISMATCH #{i} {p[:50]!r} at {k}\n  ref {ref[max(0, k - 3):k + 8]}\n  got {got[max(0, k - 3):k + 8]}")
print(f"{len(PROMPTS) - bad}/{len(PROMPTS)} prompts identical -> {'PASS' if bad == 0 else 'FAIL'}")
sys.exit(1 if bad else 0)
