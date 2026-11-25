# P0 diffusion baseline — results

Baseline engine (a measuring stick only; Rope's own engine is written from scratch):
- stable-diffusion.cpp release **master-1090-8b1e9df** (2026-11-12), assets `sd-master-8b1e9df-bin-win-cuda12-x64.zip` + `cudart-sd-bin-win-cu12-x64.zip` → `bin/sd-cuda/`, `sd-master-8b1e9df-bin-win-cpu-x64.zip` → `bin/sd-cpu/`. Both `sd-cli.exe --help` exit 0.

Tools:
- `measure.py`: wall time, sd.cpp phase times, peak RAM (working set) and commit, GPU dedicated + shared (spill) memory per process, system disk reads, minimum free RAM. Raw rows are in `results.jsonl`.
- `memcap.py`: weak-device simulator. The `4gb` profile has a hard working-set cap of 2.5 GB and a commit cap of 6 GB. `--no-gpu` hides CUDA.

Machine: Ryzen 5 5600H, 8 GB DDR4-3200 single channel, GTX 1650 4 GB (no tensor cores), PM9A1 NVMe, Windows 11.

## Runs

| Profile | Model | Res | Steps | Wall s | Sampling s | VAE s | Peak RAM GB | Peak VRAM GB | Shared spill GB | Estimate (J/K) |
|---|---|---|---|---|---|---|---|---|---|---|
| GPU (8 GB + GTX 1650) | SD-Turbo FP32 | 512² | 1 | 21.4 | 6.69 | 4.93 | 0.85 | 2.57 | 0.17 | ~2–5 s → measured 12.8 s total |
| CPU only (8 GB) | SD-Turbo FP32 | 512² | 1 | 70.9 | 5.89 | **60.02** | 3.46 (free RAM fell to 1 MB) | – | – | ~10–20 s → 67.5 s |
| CPU only, **4 GB cap** | SD-Turbo FP32 | 512² | 1 | 84.9 | 6.70 | **72.23** | 2.50 (cap) | – | – | – |
| CPU only, 4 GB cap, `--vae-tiling` | SD-Turbo FP32 | 512² | 1 | 53.1 | 7.01 | 41.37 | 2.50 | – | – | – |
| CPU only, 4 GB cap, **TAESD** | SD-Turbo FP32 | 512² | 1 | **14.0** | 6.44 | **2.13** | 2.50 | – | – | – |
| GPU (8 GB + GTX 1650) | FLUX.1-schnell Q4_K + T5 Q4_K_M + CLIP-L | 512² | 4 | 120.7 | 105.9 (26.5 s/step) | 4.93 | 0.60 | 2.92 | 0.17 | – |
| GPU (8 GB + GTX 1650) | FLUX.1-schnell Q4_K + T5 Q4_K_M + CLIP-L | **1024²** | 4 | **365.8** | 314.3 (78.6 s/step) | 42.2 | 0.62 | 2.55 | 0.17 | ~75 s total → measured 364 s |
| CPU only, 8 GB laptop (RAM cap 3.2 GB = free − 0.75) | FLUX.1-schnell Q4_K + T5 + CLIP-L | 512² | 4 | **264.5** (4.4 min) | 236.4 (59 s/step) | 20.2 | 3.20 (cap), commit 11.4 | – | – | – |
| CPU only, **4 GB laptop** (RAM cap 2.5 GB, commit cap 6 GB) | FLUX.1-schnell Q4_K + T5 + CLIP-L | 512² | 4 | **FAILED** after 8.9 s | – | – | 2.50 | – | – | – |
| GPU (8 GB + GTX 1650) | **Wan 2.1 T2V 1.3B** Q4_K_M + umT5 Q4_K_M | 832×480, **17 frames** | 20 (CFG 6) | **1875.9** (31 min) | 1433.5 (71.7 s/step) | **420.0** (3 OOM retries) | 1.10 | 2.38 | 0.22 | ~18 s/step → measured 72 s/step |

| CPU only, **4 GB cap** | Wan 2.1 T2V 1.3B Q4_K_M + umT5 Q4_K_M | 320×192, 5 frames | 10 (CFG 6) | **173.5** (2.9 min) | 77.6 (7.8 s/step) | 61.0 (2 OOM retries) | 2.50 (cap), commit 6.57 | – | – | – |

Disk reads per run: GPU 7.9 GB · CPU 26.3 GB · 4 GB cap 32.4 GB · tiling 3.0 GB · TAESD 2.1 GB (paging stops when the decoder is small).

Quality vs the CPU full-VAE image (`out/compare` method: PSNR + 8×8-window luma SSIM):

| Output | PSNR dB | SSIM | Note |
|---|---|---|---|
| GPU full VAE | 54.8 | 1.000 | same image |
| 4 GB cap full VAE | ∞ | 1.000 | bit-identical: the cap changes speed only |
| 4 GB cap, tiled VAE | 27.2 | 0.917 | small tile seams |
| 4 GB cap, TAESD | 23.2 | 0.668 | same content, softer/different fine texture |

### FLUX.1-schnell (P0.2.2)
- 512²: 2.0 min. 1024²: **6.1 min**. Both images are correct (`out/p022_flux_pair.png`). Peak RAM is only 0.6 GB: sd.cpp keeps the weights on disk ("params disk") and streams the 6.6 GB DiT to the GPU at every step (23–26 GB read per run).
- **Compute-bound, as analysis J predicted.** At 1024² one step is ~98 TFLOP (12B params × 2 × 4096 tokens). 78.6 s/step is ~1.25 TFLOPS, the same as ggml's measured CUDA Q4_K GEMM (1.16 TFLOPS, `bench/kernels/RESULTS.md`). The per-step disk streaming hides behind the compute.
- Implication [est]: rope's FP16 GEMM (1.7 TFLOPS on this GPU) would cut the 1024² step to about 54 s (~1.45×) once rope runs FLUX. To be proven in P1.5.5/P2.
- My J/K estimate (~75 s per 1024² image) was ~5× too optimistic: it assumed 4 TFLOPS, while the real kernels reach 1.2–1.7.

### CPU FLUX runs (P0.3.1 / P0.3.2)
- The first uncapped attempt was stopped by Claude Code (system critically low on memory). On the user's choice, both runs were repeated with hard RAM caps.
- **8 GB laptop, no GPU: FLUX 512² in 4.4 min** (59 s/step on 6 cores). The image is correct (`out/p031_flux_cpu_8gbcap_512.png`). The cap was 3.2 GB (free RAM − 0.75 GB at start, not the full 5.5 GB profile) to protect the running system. Disk reads were 48.8 GB: the process paged constantly.
- **4 GB laptop, no GPU: FLUX FAILS.** Exact error: `ggml_backend_cpu_buffer_type_alloc_buffer: failed to allocate buffer of size 6884552960` → `model manager alloc params backend buffer failed, size = 6565.62MB`.
  - **Root cause:** without a GPU, sd.cpp copies the whole 6.6 GB DiT into one private RAM buffer (commit), instead of computing from memory-mapped file pages. With a GPU it streams from disk ("params disk"), but not on CPU. The private copy needs 9.6 GB of commit, more than a 4 GB laptop's 6 GB commit limit.
  - **This is the main target for Rope's streamer (P2.1.1):** weights that stay memory-mapped do not count against the commit limit, and only the block being computed needs to be resident. The same model should then run on the 4 GB, no-GPU profile.

### Video (P0.2.3)
- Wan 1.3B, 480p, 17 frames (1.06 s at 16 fps), 20 steps, CFG 6: **31 minutes** for about 1 s of video. The output is coherent (fox running through snow, `out/p023_frames.png`).
- **Sampling: 72 s per step**, 4× slower than my J/K estimate (~18 s/step). CFG doubles the work per step, and the estimate assumed 4 TFLOPS.
- **VAE decode: 420 s.** It ran out of VRAM 3 times. sd.cpp recovered by retrying with temporal tiling, then smaller spatial tiles (832×480 → 416×240 → 208×120). The video VAE is a bottleneck in both memory and time, as the SD-Turbo VAE is on CPU.
- RAM stayed low (peak 1.1 GB): the weights stream from disk, as for SD-Turbo.
- For P3: step distillation (4–8 steps, no CFG) cuts sampling by 5–10×, TAEHV (`taew2_1.safetensors`, 22 MB) replaces the heavy VAE, and VAE tiles should be planned up front, not found by running out of memory.

### Findings so far
1. **A simulated 4 GB, no-GPU laptop makes a 512² SD-Turbo image in 81 s (full VAE) or 14 s wall / 10 s generation (TAESD).**
2. **On CPU the VAE decoder, not the diffusion model, is the bottleneck:** 60–72 s of 67–81 s. The full VAE also exhausts RAM and pages (26–32 GB of disk reads). The 1-step UNet takes only ~6 s on CPU.
3. **TAESD is 34× faster at decoding** (72 → 2.1 s) and stops the paging, at a cost in fine texture (SSIM 0.67). Planner rule to test in P1.2: TAESD for previews and weak devices, full/tiled VAE for final images.
4. **sd.cpp already streams weights from disk** with its auto-fit planner (params on "disk", compute on GPU, peak RAM 0.85 GB). This is a strong baseline for P2: our streamer must beat it, not a naive loader.
5. The SD-Turbo file is FP32 (3.7 GB of parameters); an FP16 file would halve the bytes streamed per run.

## Phase 0 summary (P0.4.1)

| Device profile | SD-Turbo 512², 1 step | FLUX-schnell, 4 steps | Wan 1.3B video |
|---|---|---|---|
| GPU laptop (GTX 1650 4 GB + 8 GB RAM) | 12.8 s | 512²: 2.0 min · 1024²: 6.1 min | 480p, 17 frames, 20 steps: **31 min** |
| No GPU, 8 GB RAM | 67.5 s | 512²: 4.4 min (RAM capped to 3.2 GB) | – |
| **No GPU, 4 GB RAM** | **80.7 s** full VAE · **10.1 s** TAESD | **fails** (sd.cpp copies the 6.6 GB DiT into private RAM) | 320×192, 5 frames, 10 steps: **2.9 min** |

Estimated (analysis J/K) vs measured:

| Case | Estimate | Measured | Why |
|---|---|---|---|
| SD-Turbo GPU | 2–5 s | 12.8 s | FP32 weights streamed from disk; the VAE is 38% of the time |
| FLUX 1024² GPU | ~75 s | 364 s | real kernels reach 1.2–1.7 TFLOPS, not the 4 TFLOPS assumed |
| Wan 1.3B GPU, per step | ~18 s | 72 s | CFG doubles the work; same TFLOPS gap |

What Phase 0 tells the next phases:
1. **The decoder is the hidden bottleneck on weak devices.** On CPU the SD VAE takes 60–72 s of 67–81 s. On video it runs out of memory and leaves tile seams. Tiny decoders (TAESD: 34× faster, TAEHV for video) and planned tiling come first.
2. **Weights must stay memory-mapped on CPU.** sd.cpp copies weights into private RAM when there is no GPU, which breaks FLUX on 4 GB (commit 9.6 GB). Zero-copy from the mapped file (rope's matmul already reads FP32/quantized weights in place) is the fix (P1.5.5, P2.1.1).
3. **GPU diffusion is compute-bound** at ~1.25 TFLOPS with ggml. rope's FP16 GEMM measures 1.7 TFLOPS on the same GPU (~1.45× headroom).
4. **Video needs few-step models.** 72 s/step × 20 steps × CFG is the cost to remove (P3).

## Cleanup (P0.4.2, 2026-11-13)
Deleted, because no later phase needs them: the sd.cpp baseline binaries (`bin/`, 1.3 GB) and the ggml reference builds (`build/ggml-ref*`). They can be rebuilt with `bench/diffusion/` notes and `build/ggml-cuda-build.ps1`.

Kept, because they are inputs to the next phases (all sha256-verified, `verify_models.py`):

| File | Size | Used by |
|---|---|---|
| sd_turbo.safetensors | 5.21 GB | P1.5.5 (first model in rope), P1.4.1 |
| taesd.safetensors | 0.01 GB | P1.5.5.4 |
| flux1-schnell-q4_k.gguf, t5-v1_1-xxl-encoder-Q4_K_M.gguf, clip_l.safetensors, ae.safetensors | 10.37 GB | P2.1 streamer / placement / kernels |
| wan2.1_t2v_1.3b-q4_k_m.gguf, umt5-xxl-encoder-Q4_K_M.gguf, wan_2.1_vae.safetensors, taew2_1.safetensors | 4.96 GB | P3 video |
| sd-turbo-config/ (scheduler, UNet, text-encoder configs, tokenizer) | 1.6 MB | P1.5.5 |

Free disk after cleanup: 59 GB.
