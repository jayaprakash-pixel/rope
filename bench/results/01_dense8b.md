# Dense 8B baseline — Qwen3-8B Q4_K_M (4.68 GiB), llama.cpp b12317 CUDA 12.4, flash-attn on, pp512 / tg64, 2 reps

Run 1 (16:58, clean, no other disk load):

| GPU layers (of 36) | prompt tok/s | decode tok/s | note |
|---|---|---|---|
| 0, pure CPU (`-nopo 1`) | 36.0 | 2.87 ± 1.07 | |
| 0 (prompt ops offloaded) | 38.5 | 3.55 | ≈ 17.8 GB/s effective = RAM-bandwidth bound |
| 12 | 72.2 | 5.04 | |
| 20 | 77.7 | 7.11 | |
| 24 | 79.8 | 8.78 | |
| **28** | 75.3 | **12.70** | best |
| 32 | 62.9 | 12.70 | |
| 99 (all) | 65.4 | **7.01** | VRAM over-subscribed → WDDM spills to shared memory over PCIe |

Run 2 (18:42, orphaned baseline script, while an 18 GB download was writing to the same SSD — less reliable):

| GPU layers | prompt tok/s | decode tok/s |
|---|---|---|
| 0 | 34.1 | 1.46 ± 0.41 |
| 18 | 77.8 | 7.12 |
| 26 | 80.4 | 11.31 |

Driver warning: "suboptimal performance due to a lack of tensor cores … consider CMAKE_CUDA_ARCHITECTURES=61-virtual;80-virtual and GGML_CUDA_FORCE_MMQ" (not tested).
