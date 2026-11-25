"""P1.5.5.1: rope's CLIP tokenizer vs transformers CLIPTokenizer (no ftfy) on 30 prompts, 77-token SD encoding."""
import os, subprocess, sys
from transformers import CLIPTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
ROPE = os.path.join(HERE, "..", "..", "build", "rust-cpu", "release", "rope.exe")
TOK = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "..", "models", "sd-turbo-config", "tokenizer")
PROMPTS = [
    "a red bicycle leaning on a brick wall, photo", "A Red Bicycle", "it's a dog's life, isn't it? we'll see, you'd know",
    "2 cats and 13 dogs in 1999", "hello!!! ??? ... --- *** ###", "café naïve résumé façade", "emoji test 😀🚀 ✨",
    "中文提示词 a mix of 漢字 and english", "日本語のプロンプト", "مرحبا بالعالم", "Привет мир", "ελληνικά γράμματα",
    "tabs\tand\nnewlines\r\nand   spaces", "zero\u200bwidth\u200djoiner", "trailing spaces   ", "",
    "a " * 100, "photo of a astronaut riding a horse on mars, highly detailed, 8k, unreal engine, trending on artstation",
    "<b>html</b> &amp; entities", r"C:\Users\path\file.png", "user@example.com https://x.y/z?q=1",
    "UPPER lower MiXeD", "numbers 3.14159 and 1,000,000", "quotes \"double\" 'single' `back`",
    "hyphen-ated words and under_scores", "ﬁ ligature and Ⅻ roman numeral", "dog's   cat's", "o'clock rock'n'roll",
    "a" * 300, "masterpiece, best quality, (cinematic lighting:1.2), [blurry]",
    "नमस्ते दुनिया, एक लाल साइकिल", "வணக்கம் உலகம்", "హలో ప్రపంచం", "ওহে বিশ্ব", "สวัสดีชาวโลก",
    "<|endoftext|> inside text", "wow!hello!", "١٢٣ arabic-indic digits ٤٥",
]
tok = CLIPTokenizer.from_pretrained(TOK)
bad = 0
for i, p in enumerate(PROMPTS):
    ref = tok(p, padding="max_length", max_length=77, truncation=True).input_ids
    got = list(map(int, subprocess.run([ROPE, "tokenize", TOK, p], capture_output=True, text=True, encoding="utf-8", check=True).stdout.split()))
    ok = got == ref
    bad += not ok
    if not ok:
        print(f"MISMATCH #{i} {p[:60]!r}\n  ref {ref[:20]}\n  got {got[:20]}")
print(f"{len(PROMPTS) - bad}/{len(PROMPTS)} prompts identical -> {'PASS' if bad == 0 else 'FAIL'}")
sys.exit(1 if bad else 0)
