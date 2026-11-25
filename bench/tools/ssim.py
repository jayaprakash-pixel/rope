"""SSIM and PSNR of two same-size images (P2.4.1.2: locally quantized result vs full-precision result, same seed).
  python bench/tools/ssim.py a.png b.png [--min 0.9]   -> exit 1 when SSIM < min"""
import sys
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

a, b = (np.asarray(Image.open(p).convert("RGB")) for p in sys.argv[1:3])
lo = float(sys.argv[sys.argv.index("--min") + 1]) if "--min" in sys.argv else 0.0
s = structural_similarity(a, b, channel_axis=2)
print(f"SSIM {s:.4f}  PSNR {peak_signal_noise_ratio(a, b):.1f} dB  -> {'ok' if s >= lo else 'FAIL'}")
sys.exit(0 if s >= lo else 1)
