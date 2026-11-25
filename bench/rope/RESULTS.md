# rope engine: first model end to end (P1.5.5), SD-Turbo 512², 1 step

Machine: Ryzen 5 5600H (6 cores, AVX2), 8 GB RAM, Windows 11. rope runs on CPU only here; the CUDA pipeline comes in P2. Weights: `sd_turbo.safetensors` (FP32) read in place from a memory map.

## Correctness vs independent references (diffusers 0.43, transformers 5.8)

| Stage | Check | Result | Limit |
|---|---|---|---|
| Tokenizer (P1.5.5.1) | token ids vs transformers CLIPTokenizer, 38 prompts incl. Indic, CJK, emoji | 38/38 identical | identical |
| Text encoder (P1.5.5.2) | hidden states vs CLIPTextModel (diffusers converter) | max rel err 8.3e-7 | 1e-2 |
| UNet (P1.5.5.3) | noise prediction vs UNet2DConditionModel (diffusers converter) | max rel err 3.2e-6 | 2e-2 |
| TAESD (P1.5.5.4) | decoded image vs AutoencoderTiny | **PSNR 91.8 dB** | 40 dB |
| Full VAE (P1.5.5.6) | decoded image vs AutoencoderKL (diffusers converter) | **PSNR 88.5 dB** | 35 dB |
| Whole pipeline (P1.5.5.5) | same noise, diffusers EulerDiscreteScheduler + UNet + TAESD | **PSNR 73.1 dB** | 30 dB |

Tools: `bench/tools/check_tokenizer.py`, `check_text_encoder.py`, `check_unet.py`, `check_pipeline.py`.

## Speed and memory vs the P0 sd.cpp baseline (same machine, `bench/diffusion/measure.py`)

| Profile | Engine | Wall s | text s | UNet s | decode s | Peak RAM (working set) | **Peak commit** | Disk read |
|---|---|---|---|---|---|---|---|---|
| CPU, 8 GB | sd.cpp, full VAE (P0.3.1) | 70.9 | 1.6 | 5.9 | 60.0 | 3.46 GB | 5.42 GB | 26.3 GB |
| CPU, 8 GB | **rope, TAESD** | **13.0** | 1.16 | 8.16 | 2.56 | 3.85 GB | **0.34 GB** | 3.3 GB |
| CPU, 4 GB cap | sd.cpp, TAESD (P0.3.2) | 14.0 | 1.7 | 6.4 | 2.1 | 2.50 GB | 4.16 GB | 2.1 GB |
| CPU, 4 GB cap | sd.cpp, full VAE (P0.3.2) | 84.9 | 1.7 | 6.7 | 72.2 | 2.50 GB | 5.42 GB | 32.4 GB |
| CPU, 4 GB cap | **rope, full VAE** | **33.5** | 1.05 | 8.84 | 23.3 | 2.50 GB | **1.09 GB** | 3.5 GB |
| CPU, 4 GB cap | **rope, TAESD** | **12.3** | 0.97 | 8.41 | 2.55 | 2.50 GB | **0.34 GB** | 3.3 GB |

### What this shows
1. **rope's own engine reproduces diffusers almost bit for bit**, from tokenizer to pixels.
2. **Private memory (commit) is 12–16× lower than sd.cpp's** (0.34 GB vs 4.2–5.4 GB). The 3.5 GB of FP32 weights stay memory-mapped, so they live in the OS page cache, never in the process's private memory. That is exactly what made sd.cpp fail FLUX on the 4 GB profile (commit 9.6 GB > 6 GB limit).
3. **On the 4 GB, no-GPU profile rope runs at full speed** (12.3 s) with the working set capped at 2.5 GB.
4. Gaps: rope's UNet step is ~1.4× slower than sd.cpp's on CPU (8.2–8.4 s vs 5.9–6.4 s); the text encoder and TAESD are as fast or faster. The UNet gap is the conv path (im2col + FP32 GEMM) and FP32 weights; candidates: FP16 weights, direct 3×3 conv, the int8 dot path (P2.2.3).

## SD 1.5 (P1.7.3, 2026-11-14) — brought by the user, run through `rope image --model <file>`

Correctness vs transformers/diffusers (`check_tokenizer.py <sd15 tokenizer>`, `check_text_encoder.py sd15`, `check_unet.py sd15`, `check_pipeline_sd15.py 20`):

| Stage | Result |
|---|---|
| Tokenizer (pad `<|endoftext|>`, `!` is ordinary BPE) | 38/38 prompts identical |
| CLIP ViT-L/14 text encoder | max rel err 2.3e-7 |
| UNet (1×1-conv proj, 8 heads, 768 context), one call | max rel err 1.7e-6 |
| Whole pipeline, 20 Euler steps (trailing), CFG 7.5, full VAE | **PSNR 74.1 dB** (limit 30) |

4 GB no-GPU profile, `--quality balanced` (planner: 20 steps, CFG 7.5, TAESD): **223.4 s** (UNet 220 s = 40 calls × 5.5 s), peak commit **0.34 GB**, working set capped at 2.5 GB.
First use: SHA-256 + safety scan 18.1 s (clean). Planner estimate 508 s (+127%): the compute part (224 s) matches; the paging term (284 s) did not happen because
the dev laptop's file cache (8 GB machine, ~3.5 GB free at the time) served the 70 M soft page faults (2.73 GB real disk reads). On a real 4 GB laptop those faults go to disk, so the paging term is untested here, not disproved.
Speed-up targets for P2: one batch-2 UNet pass for CFG (reads the weights once per step instead of twice), FP16 weights, the parallel UNet loops.

## FLUX.1-schnell in rope's own engine (P2.0, 2026-11-14)

Correctness (each component vs transformers/diffusers on the same weights; the 12B DiT and 4.7B T5 references run block by block on gguf-dequantized weights):

| Component | Check | Result |
|---|---|---|
| CLIP-L file + pooled | `check_text_encoder.py clipl` | 1.2e-6 |
| T5 unigram tokenizer (reads `spiece.model`) | `check_tokenizer_t5.py` | 44/44 identical |
| T5-XXL encoder, GGUF Q4_K_M | `check_t5.py` | 5.9e-5, commit 0.08 GB |
| FLUX DiT, GGUF Q4_K | `check_flux.py` (256², t = 1) | 1.26e-4 |
| FLUX VAE | `check_vae_flux.py` | 91.0 dB |
| Whole pipeline, 512², 4 steps | `check_pipeline_flux.py` | **62.2 dB** |

4 GB no-GPU profile (`measure.py --profile 4gb --no-gpu`, `rope image --model flux1-schnell-q4_k.gguf --quality balanced`):

| Engine | Result | Peak commit | Working set | Disk read |
|---|---|---|---|---|
| sd.cpp (P0.3.2) | **FAILED** after 8.9 s (6.6 GB buffer alloc) | > 6 GB cap | 2.50 GB | – |
| **rope** | **429.6 s** (text 14.7, DiT 4 × 98 s, VAE 22.6) | **1.08 GB** | 2.50 GB | 41.4 GB |

First use adds 25.9 s (SHA-256 + scan). Planner estimate 493 s (+15%). Here the paging was real (free RAM fell to 172 MB; 41 GB read = ~6 passes over the 6.9 GB DiT): 98 s/step paged vs 91 s/step resident, so streaming already costs only ~8%.
Speed gap: rope 91 s/step resident vs sd.cpp 59 s/step (P0.3.1, 8 GB). The DiT speeds up only 2.9x on 6 threads (1 thread: 266 s) -> quantized CPU matmul (P2.2.3) is the lever.

### FLUX 1024² on the 4 GB no-GPU profile (P2.1.1)
| Run | Per step | Total | Peak commit | Disk read |
|---|---|---|---|---|
| rope resident (warm, uncapped, one transformer call) | 551–558 s | – | – | – |
| **rope, 4 GB cap**, `rope image --model flux… --size 1024` | **558.7 s** (+0–1%) | 2440.6 s (text 14, DiT 2235, tiled VAE 192) | **1.10 GB** | 31.0 GB |
The memory-mapped weights stream from disk at ~1% cost when a step computes for ~9 minutes. Where prefetching would still matter: small images on slow disks (512² on SATA 0.45 GB/s: ~15 s of reads vs 91 s compute ≈ 16%).

## CPU speed work (P2.2.6, 2026-11-14) — measured first, then fixed

`rope profile --timed` (per-kind and per-shape kernel times) showed the kernels, not the Rust code, were slow, and experiments found why:

| Cause (found by experiment) | Fix | Effect |
|---|---|---|
| Soft page faults on memory-mapped weights (each 4 KB page faults on first use inside the kernel threads) | prefetch thread reads **and maps** the next block's pages while the current block runs | SD UNet step 8.7 -> 5.2 s |
| Activations larger than L3 re-read for every 16-column weight panel | cache-blocked matmul (8 panels share each L2 block of x), only when x > 16 MB | Q4_K 4096x3072: 195 -> 360 GFLOPS |
| im2col writes ~1.2 GB per VAE conv, then re-reads it | implicit-GEMM conv (gather 64 positions into an L2 panel, output in place) | VAE 512²: 20 -> 9-10 s |
| fresh output pages every layer | caching allocator, cache size chosen by the planner from free RAM | FLUX 338 s (256 MB) -> 309 s (0.35 GB) -> 274 s (768 MB) |
| rejected: denormals (FTZ/DAZ no effect); PrefetchVirtualMemory alone (no effect until pages are touched) | – | – |

All reference checks give the same error values as before. 4 GB no-GPU profile, end to end:

| Model | Before | Now | sd.cpp (P0) |
|---|---|---|---|
| SD-Turbo 512², full VAE | 34.9 s | **18.0 s** | 80.7 s |
| SD 1.5, 20 steps + CFG, TAESD | 223.4 s | **185.9 s** | – |
| FLUX-schnell 512², 4 steps | 429.6 s | **308.9 s** | failed (8 GB machine: 264.5 s) |

## GPU executor, G1 + G2a (P2.6.1, P2.1.2) — GTX 1650, 2026-11-15
- Heavy ops of GPU-placed blocks run as CUDA kernels (chosen by weight address); FP32 GPU math (the FP16 kernel converts activations to half: 6.7e-2 error on FLUX, rejected as default).
- VRAM manager: resident weights in stored form, persistent scratch, one 32 MB pinned staging buffer, budget = min(free VRAM - 0.4 GB, free commit - 1 GB). Windows charges VRAM to process commit (measured), and shared GPU memory stays at a constant 0.105 GB baseline: no spill.

| FLUX-schnell, 4 steps | DiT step | Total | Dedicated VRAM | Shared | Notes |
|---|---|---|---|---|---|
| 512², CPU (8 GB laptop) | 64 s | – | – | – | P2.2.6 |
| 512², one GPU call | 44.6 s | – | 3.07 GB | 0.105 GB | 4.5e-4 vs CPU |
| 1024², CPU | 558 s | 2441 s (4 GB profile) | – | – | P2.1.1 run |
| **1024², GPU** | **224 s** | **972 s** | 3.28 GB | 0.105 GB | 1.94 GiB resident, 394 hits |
