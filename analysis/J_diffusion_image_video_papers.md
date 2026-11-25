# J — Diffusion (image + video) papers: what changes when the model is compute-bound

Date: 2026-11-12. Sources: the 19 PDFs in `papers\gaps_vision\` (V01–V19), converted with `pdftotext -layout`. Context: `00_SYNTHESIS.md` (text-LLM part).

**Labels used in this file**
- **[paper]** = number taken from the paper (table or section is cited).
- **[est]** = my calculation or estimate. The formula is shown.
- **[ext]** = public knowledge that is NOT in any of our 19 papers (for example Flux block counts, GTX 1650 specs). Verify before you depend on it.
- "not stated" = the paper does not give the fact.

**Machine reference numbers used in all estimates** (GTX 1650 = TU117, sm_75):
- 896 CUDA cores, about 1.4–1.6 GHz boost on laptops. FP32 ≈ 2.6–3.0 TFLOPS. [ext]
- **No tensor cores on TU117** (the GTX 16 series has no tensor cores; it has double-rate FP16 units). No paper in this set says otherwise. None of the 19 papers tests a GPU without tensor cores. [ext, verify with a micro-benchmark]
- FP16 (HFMA2, FP16 accumulate) ≈ 2× FP32 ≈ 5.5–6 TFLOPS peak. INT8 `dp4a` (INT32 accumulate) ≈ 4× FP32 ≈ 11 TOPS peak. No INT4, no FP8, no BF16 math. [ext]
- **Planning value: 4 TFLOPS sustained** for FP16 GEMM, 2.5 TFLOPS if FP32 accumulate is necessary, ≈8 TOPS sustained for an INT8 dp4a GEMM. [est]
- VRAM 4 GB (about 3.3 GB usable), ~128 GB/s. PCIe ~10 GB/s. RAM 8 GB single channel (~4.5 GB usable). NVMe ~5 GB/s.

---

## 0. Summary (read this first)

1. **Diffusion on this GPU is compute-bound by a very large margin, even when the weights stream from the SSD every step.** One denoising step of Flux-class 12B at 1024² is ≈74 TFLOP [est] (≈19 s on the GTX 1650). Reading the whole 4-bit model from the SSD takes ≈1.3 s. Block streaming is "nearly free" when `tokens × sequences ≥ b·C / (2·BW)` ≈ **225 tokens** for 4-bit weights from the SSD (§Q3). Every real image and video workload is far above this.
2. **So the LLM ranking of levers inverts.** In the LLM part, "fewer bytes per weight" was lever 1. Here, weight compression is mainly for **disk space** (82 GB) and RAM fit. It does not give speed unless activations are quantized too (SVDQuant §1, MixDQ App. C, ViDiT-Q §5.3 all measure this). The speed levers are: **fewer steps**, **fewer FLOPs per step** (attention sparsity, caching), and **faster math** (INT8 dp4a instead of FP16).
3. **Video is attention-dominated.** Attention is 52% of the FLOPs for Wan-14B at 480p, 72% at 720p, and 87% for HunyuanVideo at 720p [est, cross-checked with Sparse VideoGen Table 1: 612 PFLOP / 50 steps]. Attention cost is the main video lever.
4. **The GTX 1650 cannot run any of the fast kernels in these papers as written.** SVDQuant W4A4, SageAttention 1/2, Sparse VideoGen, ViDiT-Q and MixDQ kernels all use tensor-core `mma` instructions (INT8/INT4/FP8). The numerical recipes transfer. The kernels must be rewritten for CUDA cores (`dp4a` + `HFMA2`).
5. **Realistic targets on this machine** [est]: Flux-schnell (4 steps) 1024² ≈ 1.5–2 min; Wan-1.3B 480p 5 s video with a distilled 4–8-step sampler ≈ 10–20 min; Wan-14B or HunyuanVideo 720p with 50 steps ≈ 40+ hours (not practical).

---

## 1. Per-paper analysis

### V01 — DiT: Scalable Diffusion Models with Transformers (2212.09748)

**Mechanism.** A latent diffusion model whose denoiser is a ViT. The input is the VAE latent (256² image → 32×32×4 with the SD VAE, downsample 8). "Patchify" cuts p×p patches into T = (I/p)² tokens. Halving p quadruples T and so at least quadruples Gflops; parameters stay almost the same (§3.2). Conditioning (timestep t, class c) enters by **adaLN-Zero**: an MLP on (t + c) regresses per-channel scale γ, shift β and gate α for each block; α is zero-initialized so each block starts as identity. The final layer decodes each token to p×p×2C (noise + diagonal covariance).

**Results.**
- Table 1/4: DiT-S/B/L/XL = 12/12/24/28 layers, hidden 384/768/1024/1152, 33M/130M/458M/675M params.
- DiT-XL/2 at 256² = 118.6 Gflops per forward; at 512² = 524.6 Gflops with 1024 tokens (§5).
- FID-50K 2.27 at 256² (cfg 1.5, Table 2); 3.04 at 512² (Table 3).
- adaLN-Zero beats cross-attention (137.6 Gflops, +15%) and in-context conditioning (Fig. 5, Table 4).
- Fig. 10: more sampling steps cannot compensate for a smaller model ("L/2 at 1000 steps uses 80.7 Tflops and still loses to XL/2 at 128 steps, 15.2 Tflops").
- The SD VAE has 84M params (Table 4 note).

**Memory/speed with hardware.** Training on TPU-v3. No inference latency or memory is stated.

**Assumptions that fail here.** None at this scale: DiT-XL/2 (675M) fits in 4 GB at FP16 (1.35 GB). It uses 250 DDPM steps, which is slow, but the model is small.

**What transfers.**
- The FLOP law: per-step compute ≈ 2·P_tok·N + 4·N²·h·L. Tokens, not parameters, set the cost. This is the base of every estimate in this file.
- adaLN modulation is computed **once per step, not per token**. In large models it is a large share of the parameters but almost no FLOPs (see Q1). The engine must treat modulation weights as a "per-step GEMV", not a GEMM.
- The patch size is the cheapest quality/compute knob. A model trained at p=2 cannot run at p=4 without retraining.

### V02 — SD3: Scaling Rectified Flow Transformers (2403.03206)

**Mechanism.**
- **Rectified flow**: x_t = (1−t)·x0 + t·ε, the network predicts the velocity. Timesteps are sampled from a logit-normal distribution so that the middle of the trajectory gets more weight (§3.1). rf/lognorm(0,1) ranks best (Table 1).
- **MM-DiT**: text and image tokens have **separate weight sets** (two "transformers"). They are joined only inside the attention (Fig. 2b). The model size is parameterized by depth d: hidden = 64·d, MLP = 4·64·d, heads = d (§4). The 8B model is d=38 → hidden 2432, 38 heads.
- **Text conditioning** (App. B.2): CLIP-L/14 + OpenCLIP bigG/14, pooled → c_vec ∈ R^2048; penultimate hidden states → 77×2048, zero-padded to 4096; T5-v1.1-XXL encoder → 77×4096; concatenated to c_ctxt ∈ R^{154×4096}.
- **VAE**: f=8, **16 latent channels** (Table 3: rFID 1.06, PSNR 28.62 vs 25.12 for 4 channels).
- Patch 2×2 → tokens = (H/16)·(W/16). 1024² → 4096 image tokens.
- **QK-RMSNorm** stabilizes attention logits in mixed precision (Fig. 5).
- **Resolution-dependent timestep shift** α = sqrt(m/n); α = 3.0 at 1024² (§5.3.2).

**Results.**
- GenEval 0.74 for depth 38 at 1024² with DPO (Table 5).
- Larger models need fewer steps: at 5/50 steps the quality gap is 4.30 (d=15) vs 2.71 (d=38) (Table 6).
- **Dropping T5 at inference** keeps 50% win rate on aesthetics and 46% on prompt adherence, but only 38% on typography (§5.3.3, Fig. 7). "T5-XXL has 4.7B parameters that require significant amounts of VRAM."

**Memory/speed.** Inference memory and latency: not stated.

**Assumptions that fail here.**
- BF16/mixed-precision training. FP16 inference on Turing can overflow (QK-norm helps attention; the MLPs are not protected). [ext: T5 in FP16 is known to overflow]
- T5-XXL at FP16 (9.4 GB) does not fit in VRAM or RAM.

**What transfers.**
- **T5 is optional.** Running without T5 (or with a 4-bit T5) is a documented, measured trade-off. This is our first memory cut for SD3-class models.
- Rectified-flow models keep quality at few steps (Fig. 3). This makes step reduction cheaper than with ε-prediction models.
- The token count formula and the 16-channel VAE are shared by Flux-class models [ext].

### V03 — HunyuanVideo (2412.03603)

**Mechanism.**
- **Causal 3D VAE**: (T+1)×3×H×W → (T/4+1)×16×(H/8)×(W/8); c_t=4, c_s=8, C=16 (§4.1). Trained from scratch. **Spatio-temporal tiling** for encode/decode on one GPU, with a fine-tuning phase that randomly enables tiling so that tiles do not leave seams (§4.1.2).
- **DiT, 13B** (Table 2): 20 dual-stream + 40 single-stream blocks, hidden 3072, FFN 12288, 24 heads, head dim 128, 3D RoPE split (16, 56, 56). Full attention over all video + text tokens (§4.2). Patch: 3D conv with kernel k_t×k_h×k_w; the numeric values are not stated in the text. [ext: public code uses 1×2×2]
- **Text encoder**: a decoder-only MLLM (causal attention) plus a bidirectional token refiner; CLIP-L final non-padded token for global guidance (§4.3). The MLLM size is not stated. [ext: public release uses an ~8B LLaVA-Llama-3 model]
- **Step reduction** (§5.1): time shift t' = s·t / (1 + (s−1)·t); s = 7 for 50 steps, s = 17 for <20 steps; 10-step samples shown (Fig. 11b).
- **CFG distillation** (§5.2): a student conditioned on the guidance scale (1–8) replaces the two CFG passes; "approximately 1.9× acceleration".

**Results.** Human evaluation (Table 3): overall 41.3%, rank 1 of 6, motion quality 66.5%. VAE PSNR (Table 1): the "ours" row values are not legible in the text conversion; FLUX-VAE 32.70 on ImageNet 256².

**Memory/speed with hardware.** Inference memory and latency: not stated. Training uses activation offload to host memory overlapped with compute (§5.3.3). Sparse VideoGen (V14, Table 1) measures **2253 s for 720p, 128 frames on an H100, 612.37 PFLOP**.

**Assumptions that fail here.**
- 13B BF16 = ~26 GB; an ~8B text LLM. Neither fits.
- 720p × 129 frames = **118,800 video tokens** [est]. The residual stream alone is 731 MB in FP16; one single-stream fused projection (QKV + MLP-in = 7·3072 wide) is 5.1 GB. It must be token-chunked on 4 GB.

**What transfers.**
- **The text encoder is an LLM.** Our LLM engine runs it unchanged (prefill only, keep the last hidden states).
- Tiled VAE decode with seam-aware training: we must tile anyway.
- Time-shift s=17 for 10–20 steps is free to use (no training).
- CFG distillation halves compute per step. Use guidance-distilled checkpoints where they exist.

### V04 — Wan 2.1 (2503.20314)

**Mechanism.**
- **Wan-VAE**: 3D causal, compression 4×8×8 to 16 channels, **127M params** (§4.1.1–4.1.2). GroupNorm replaced by RMSNorm to keep causality. **Feature cache**: the video is processed in 1 + T/4 chunks of at most 4 frames; each causal conv keeps the last 2 frames of the previous chunk as cache (kernel 3). This bounds VAE memory and allows "infinite-length" video (§4.1.3, Fig. 6). Reconstruction is 2.5× faster than HunyuanVideo's VAE on the same hardware (§4.1.4).
- **DiT**: patchify with a 3D conv, kernel (1, 2, 2) → L = (1 + T/4)·(H/16)·(W/16) tokens (§4.2.1). Self-attention + **cross-attention to text** (not joint attention) + FFN. The time-embedding MLP is **shared across blocks**, with per-block biases: about 25% fewer parameters at the same quality (§4.2.1). The block config table is not in the extracted text. [ext: 14B = 40 layers, hidden 5120, FFN 13824, 40 heads; 1.3B = 30 layers, hidden 1536, FFN 8960, 12 heads]
- **Text**: umT5, 512 tokens (§4.2.2). "The text encoder requires over 20 GB of GPU memory" (§4.3.2, training context).
- 81 frames for 1280×720 (§4.2.2). The final stage is 480p and 720p, 5-second clips.

**Results / costs.**
- DiT cost = L·(α·b·s·h² + β·b·s²·h). Attention reaches 95% of time at 1M tokens (§4.3.1).
- Inference "typically around 50" steps (§4.4).
- **Diffusion cache** (attention cache every few steps + CFG cache with FasterCache-style residual compensation): **1.62×** on Wan 14B T2V (§4.4.2).
- FP8 GEMM (per-tensor weight, per-token activation): GEMM 2× vs BF16, **1.13×** for the DiT (§4.4.3).
- 8-bit FlashAttention: INT8 for QKᵀ, FP8 for PV, FP32 cross-block accumulation, 95% MFU on H20.
- **1.3B model "requires only 8.19 GB VRAM"** (abstract; the settings are not stated).
- 14B "requires approximately 30 minutes on a single high-end GPU without additional optimization" (Limitations).
- Streaming application: int8 (torchao) for attention/linear heads reduces memory "with minimal acceleration"; TensorRT gives 8 FPS on a 4090 with some flicker (§ streaming section).
- VBench (Table 4): 14B 86.22%, 1.3B 83.96%.

**Assumptions that fail here.** FP8 and FA3 are Hopper-only. 14B BF16 = 28 GB. umT5 at >20 GB. 8.19 GB VRAM for the 1.3B model is still 2× our VRAM.

**What transfers.**
- **Wan-1.3B is the realistic video model for this machine**: 1.25B per-token params, 2.5 GB at FP16, 0.7 GB at 4-bit. Its full weights fit in VRAM. 480p 5 s = 32,760 tokens, ≈283 TFLOP per forward [est].
- The **chunked causal VAE with feature cache** is exactly what a 4 GB GPU needs. Copy the design.
- **Cross-attention text K/V are constant over all steps.** Compute them once per prompt and keep them (40 layers × 512 × 5120 × 2 × 2 B ≈ 420 MB for 14B at FP16; [est]).
- The attention/CFG cache gives 1.62× without training.
- The int8 lesson matches the other papers: int8 weights alone give memory, not speed.

### V05 — SVDQuant + Nunchaku (2411.05007)

**Mechanism.**
1. Smooth: X̂ = X·diag(λ)⁻¹, Ŵ = W·diag(λ), with λ_i = max|X_i|^α / max|W_i|^(1−α) (SmoothQuant style). This moves activation outliers into the weights.
2. Decompose Ŵ = L1·L2 + R by SVD (rank r = 32; rank 16 for 8-bit). The **low-rank branch runs in 16-bit**. The residual R and the activations are quantized to 4 bits: XW ≈ X̂·L1·L2 + Q(X̂)·Q(R) (Eq. 5). The error bound is set by ‖R‖_F (Prop. 4.2). Several iterations refine L and R; GPTQ quantizes R (App. D).
3. INT4: per-group symmetric, **group 64**, 16-bit scales, for weights AND activations. NVFP4: group 16 with FP8 scales (Blackwell). adaLN linear inputs stay 16-bit (W4A16) in FLUX (App. D).
4. **Nunchaku kernel fusion**: down-projection L1 is fused into the activation-quantize kernel (same input). Up-projection L2 is fused into the 4-bit GEMM epilogue (same output). The unfused low-rank branch costs ~50–57% extra latency; fused, it costs 5–10% (§4.3, Fig. 6).
5. LoRAs run as an extra low-rank branch without re-quantization.

**Results.**
- Table 1, FLUX.1-dev, MJHQ, 50 steps: BF16 FID 20.3 / IR 0.953; NF4 W4A16 20.6 / 0.910 / LPIPS 0.272; **INT4 W4A4 19.9 / 0.935 / LPIPS 0.223**; NVFP4 20.4 / 0.937 / 0.208. W8A8 LPIPS 0.089.
- PixArt-Σ: ViDiT-Q W4A4 collapses (FID 412); SVDQuant INT4 19.2.
- Fig. 8: transformer 22.2 GiB → **6.1 GiB** (incl. 0.3 GiB low-rank). DiT memory 24.4 → 6.9 GiB.
- Single-step latency at 1024²: desktop 4090 BF16 ≈657 ms, NF4 ≈660 ms, INT4 ≈218 ms (3.0×); **laptop 4090 (16 GB) BF16 4382 ms with per-layer CPU offload, NF4 1320 ms, INT4 433 ms (10.1×)**; 5090 BF16 484, NVFP4 162 ms.
- §1: "diffusion models are heavily computationally bounded, even with a single batch. As a result, weight-only quantization cannot accelerate diffusion models."

**Assumptions that fail here.** INT4 math needs INT4 tensor cores (Ampere/Ada); NVFP4 needs Blackwell. **The GTX 1650 has neither, so the 3× speedup does not transfer.** 6.1 GiB still exceeds our VRAM and our free RAM.

**What transfers.**
- **The low-rank + residual weight format** is a good storage format for us: 4-bit residual + rank-32 FP16 factors (+5% size). It can feed a W4A8 or W8A8 dp4a kernel, or a W4A16 dequant kernel.
- The fusion rule (do the low-rank down-projection in the quantize kernel, the up-projection in the GEMM epilogue) applies to any kernel we write.
- The laptop-4090 numbers prove the offload lesson: naive per-layer offload of BF16 costs ~6.7× on a fast GPU. On our slow GPU the same offload is nearly free (§Q3).
- Keep adaLN inputs high-precision.

### V06 — Q-Diffusion (2302.04304)

**Mechanism.**
- **Timestep-aware calibration**: sample intermediate inputs x_t uniformly at a fixed interval across all timesteps (Alg. 1). Calibrating from only the first/last steps fails (Fig. 4: 5120 samples from "first 25" steps give much worse 4-bit FID).
- **BRECQ-style block reconstruction** with AdaRound for weights; LSQ step sizes for activations.
- **Shortcut-splitting**: in U-Net skip concatenations, the deep and shallow channel groups have very different ranges (up to 200×). Quantize X1 and X2 (and W1, W2) separately before concatenation (Eq. 6–8).

**Results.**
- W4A8 for unconditional models with FID increase of 0.39–2.34 (contributions, Tables 1–3).
- LSUN-Bedroom LDM-4 (Table 2): FP 2.98; W4A32 Linear 82.69 vs Q-Diffusion much lower (the table is mangled in the text; the main text claims ≤2.34 FID increase).
- Stable Diffusion 4-bit weights shown qualitatively (Fig. 12).

**Memory/speed.** Model size in Mb and Bops only. **No real kernel, no measured latency** — simulated (fake) quantization. Speed: not stated.

**Assumptions that fail here.** It targets U-Nets (shortcut-splitting has no DiT equivalent). Calibration needs many full-precision forwards — on our GPU each Flux step is ~19 s, so a 256-sample calibration costs hours [est].

**What transfers.**
- Calibrate over the **whole trajectory**, not one timestep. This applies to any static scale we choose (for example static INT8 scales for activations if dynamic scaling is too slow in our kernel).
- Quantize before concatenation when two differently-scaled tensors are joined. In MM-DiT, the text and image tokens are concatenated before attention: use **separate scales for text and image token groups** (same principle).

### V07 — PTQ4DiT (2405.16005)

**Mechanism.**
- **Channel-wise Salience Balancing (CSB)**: salience s(X_j) = max|X_j|, s(W_j) = max|W_j|. Balanced salience = sqrt(s(X_j)·s(W_j)). B_X = diag(s̃/s(X_j)), B_W = diag(s̃/s(W_j)); X̃ = X·B_X, W̃ = B_W·W (Eq. 3–7). The extremes in X and W do not fall in the same channels ("complementarity"), so this reduces the max of both.
- **Spearman-ρ-guided Salience Calibration (SSC)**: activation salience is a weighted sum over timesteps; the weight η_t = softmax(−ρ(s(X_t), s(W))) gives more weight to timesteps where X and W salience correlate least (Eq. 10–11).
- **Re-parameterization**: B_X is folded into the adaLN scale/shift (γ̃ = γ·B_X, β̃ = β·B_X, Eq. 13) or into the previous dequantization. Zero runtime cost.
- Setup: channel-wise weights, **tensor-wise activations**, calibration of 25 timesteps × 32 samples (256²).

**Results.**
- Table 1, DiT-XL/2 256², 250 steps: FP FID 4.53; W8A8 4.63; **W4A8 7.09** (Q-Diffusion 15.31, RepQ* 23.21).
- At 50 steps: FP 6.02, W4A8 9.17.
- 512² W4A8 (Table 2): FP 8.39, ours 17.55, others ≥58.
- Model 2575 MB FP32 → 323.79 MB W4.

**Memory/speed.** Experiments on RTX A6000; the intro says DiT sampling can take ">20 seconds and 10⁵ Gflops on an RTX A6000". **No quantized kernel; speed not stated.**

**Assumptions that fail here.** Class-conditional DiT-XL/2 only; no text-to-image or video. ViDiT-Q (V08, Table 1) shows PTQ4DiT W4A8 degrades on video (OpenSORA) because static, tensor-wise activation scales cannot follow timestep variation.

**What transfers.**
- **Folding the balancing vector into adaLN** is a free and exact transform for every DiT linear layer that follows a modulated norm (QKV and MLP-in). Our W8A8 path should use it.
- The ρ-weighting is a cheap rule for choosing one static scale over many timesteps.

### V08 — ViDiT-Q (2406.02540)

**Mechanism.**
- **Fine-grained, dynamic groups**: per-channel weight scales and **per-token dynamic activation scales**, computed online (max/min per token) and fused into the previous op. This handles token-, condition- (CFG) and timestep-wise variation (§4.1).
- **Static-dynamic channel balancing**: the time-varying channel imbalance comes from adaLN modulation = a static part (the pretrained scale-shift table) + a dynamic part (time embedding). A **scaling** mask (SmoothQuant-style) fixes the static part; a **Hadamard rotation** (QuaRot-style) fixes the dynamic part (§4.2).
- **Metric-decoupled mixed precision**: layers are grouped by what they damage — cross-attention → content (CLIPScore), spatial attention + FFN → visual quality (VQA), temporal attention → temporal consistency (FlowScore). Sensitivity is measured per group and per quarter of the denoising process; bits are raised for the most sensitive layers until the budget is reached (§4.3).

**Results.**
- OpenSORA VBench (Table 1): W8A8 ≈ FP16; W4A8 ViDiT-Q close to FP16 while Q-DiT/PTQ4DiT/SmoothQuant/QuaRot degrade.
- PixArt-α COCO (Fig. 6): FP16 FID 73.34 / IR 0.901; ViDiT-Q W8A8 75.61 / 0.917; W4A8 74.33 / 0.887.
- **Hardware (Fig. 7, A100, CUDA 12.1, batch 1, 20 steps):** W8A8 memory 1.99×, latency **1.71×** (naive W8A8 1.74×); W4A8 memory 2.42×, latency only **1.38×** — "since current DiT computation exhibits a compute-bound characteristic, the W4A8 CUDA kernel primarily saves memory without enhancing efficiency."
- Motivating example: OpenSORA uses >10 GB and ~50 s on A100 for 16 frames at 512² (§1).

**Assumptions that fail here.** Their INT8 GEMMs run on A100 INT8 tensor cores. Hadamard fusion into LayerNorm/GeLU kernels is their code.

**What transfers.**
- **This is the best W8A8 recipe for our dp4a path**: per-token dynamic INT8 activations, per-channel INT8 weights, INT32 accumulate. On the GTX 1650, dp4a INT8 is ~2× the FP16 HFMA2 rate and ~4× the FP32 rate, with **exact integer accumulation** (safer than FP16 accumulate). Expected linear-layer speedup 1.5–2× over FP16 [est].
- W4A8 = W8A8 speed on compute-bound hardware; use 4-bit only to save disk/RAM.
- Mixed precision by layer type: keep cross-attention (content) and temporal attention higher precision.

### V09 — MixDQ (2405.17873)

**Mechanism.**
- **BOS-aware text-embedding quantization**: the CLIP Begin-Of-Sentence token has huge outlier features. It is the same for every prompt. So the to_k/to_v outputs for BOS are **pre-computed in FP16 offline** and concatenated with the dequantized outputs of the other tokens (§3.1). Storage: 640–1280 values.
- **Metric-decoupled sensitivity**: separate layer groups for content (CLIP-based) and quality (FID-based) — the same idea as ViDiT-Q.
- **Integer programming** (OR-tools) chooses per-layer bits (W2/4/8, A4/8) under a budget; 20×20 budget sweep, Pareto frontier (§4.1).

**Results.**
- Table 1, SDXL-turbo 1 step, COCO: FP16 FID 17.15, CLIP 0.2722. Naive W8A8 FID 103.96. Q-Diffusion W8A8 76.18. **MixDQ W8A8 17.03, W5A8 17.23, W4A8 17.68, W3.66A16 17.40.**
- Few-step models are more sensitive to quantization than multi-step models (§1).
- Hardware: RTX 4080, CUDA 12.1, CUTLASS INT8 kernels. **W8A8 and W4A8 both 1.52× faster**; **W4A16 has the same latency as FP16** (App. C: "W4A16 continues to utilize FP16 computation"). Activation memory < 100 MB vs 6 GB weights (U-Net), so activation quantization saves little memory. 3–4× memory reduction overall.

**Assumptions that fail here.** INT8 tensor cores (4080). SDXL U-Net, not DiT.

**What transfers.**
- **Pre-compute anything that is the same for every prompt or every step.** Generalized: (a) BOS-like constant tokens; (b) Wan cross-attention K/V of the text (constant over steps); (c) timestep embeddings (BitsFusion does the same); (d) adaLN modulation vectors per step (they depend only on t and the pooled text).
- Few-step distilled models need **higher bits** than 50-step models. When we combine step distillation with quantization, budget W8, not W4, for compute.

### V10 — BitsFusion (2406.04333)

**Mechanism.** Quantization-aware training (QAT) of the SD-v1.5 U-Net to an **average of 1.99 bits** (weights only):
- Per-layer bit width (1/2/3 bits and higher) from a per-layer error analysis. The CLIP-score drop is used instead of MSE because MSE does not track semantic damage (Table 1–2, App. B).
- **Time-embedding pre-compute and caching**: the projection layers are removed; embeddings for each step are stored (25.6× smaller than the projection layers) (§4.1).
- **Balanced integers**: 1-bit uses {−1, 0, 1} (log2(3) bits), avoiding asymmetric {0,1} sets.
- Lloyd-Max-style alternating optimization of scales (10 iterations, Eq. 5).
- Two-stage training: Stage I distillation from the FP teacher (CFG-aware, with feature distillation, error-aware timestep sampling); Stage II noise-prediction fine-tuning.

**Results.** UNet 1.72 GB → **219 MB (7.9×)**. Higher CLIP score than SD-v1.5 on MS-COCO and better TIFA/GenEval at all CFG scales (Fig. 5); human preference on PartiPrompts.

**Memory/speed.** Training cost: **Stage I 8×A100, batch 256, 20K iterations; Stage II 32×A100, batch 1024, 50K iterations** (§5). **No inference-speed claim** (storage only); no low-bit kernel is described.

**Assumptions that fail here.** Weeks of multi-A100 training. We cannot do this for a 12–14B DiT on our machine. U-Net only.

**What transfers.**
- As a storage idea: sub-2-bit weights would put Wan-14B in ~3.5 GB of disk [est]. But on this machine disk space, not speed, is the only gain, and PTQ at 4 bits is "almost free". **Low priority.**
- Balanced (symmetric with zero) low-bit grids and CLIP-based sensitivity are cheap ideas for our own PTQ.

### V11 — DeepCache (2312.00858)

**Mechanism.** In a U-Net, the deep (high-level) features of the up-sampling path change slowly between adjacent steps. At a "full" step, cache U_{m+1} output (the feature entering the m-th skip branch). For the next N−1 steps, run only the shallow down block D_m and the matching up block, and concatenate with the cached feature (Eq. 4–5). **Uniform 1:N** recomputes every N steps; **non-uniform 1:N** recomputes more often where adjacent-step similarity is low (around 40% of the process for LDM) (§3.3).

**Results.**
- Table 1, LDM-4-G ImageNet, 250 DDIM: uniform N=10 → 4.1× speed with FID +0.22 (abstract).
- Table 3, SD-v1.5 50 PLMS at 512² (RTX 2080 throughput): MACs 338.83G → 130.45G; **2.15× (PartiPrompts) / 2.11× (COCO)**; CLIP 29.51 → 29.46 and 30.30 → 30.23. Better than PLMS-25 steps and BK-SDM distilled models.
- Fig. 1: 2.3× SD-v1.5, 7.0× LDM-4-G at 250 steps.

**Memory/speed.** Cache memory: one feature map — small (not stated). Hardware: one RTX 2080 for throughput.

**Assumptions that fail here.** **U-Net skip structure.** A DiT has no skip branches. Limitation section: if the shallowest branch is a large share of compute, the speedup drops. It gives little at 1–4 steps (no adjacent-step redundancy to exploit).

**What transfers.** Only for SD1.5/SDXL-class U-Nets (which do fit our GPU). For DiTs, FORA/TeaCache are the equivalents. The **non-uniform schedule** (recompute where similarity drops) is the useful general idea.

### V12 — TeaCache (2411.19108)

**Mechanism.**
- Decide **per step** whether to reuse the model output, using an indicator computed from the **inputs**, which are cheap: the **timestep-embedding-modulated noisy input** F_t (the input after the first adaLN modulation).
- Relative L1 distance L1_rel(F, t) = ‖F_t − F_{t+1}‖₁ / ‖F_{t+1}‖₁ (Eq. 4).
- A polynomial f fitted offline (numpy `poly1d`, 70 prompts) rescales the input distance to estimate the output distance (Eq. 7).
- Accumulate f(L1_rel) over steps since the last full compute; while the sum < δ, reuse the cached **residual** (output − input of the transformer body) and add it to the new input; else recompute and reset (§3.3–3.4).
- δ = 0.1 (slow) / 0.2 (fast).

**Results** (Table 1, A800 80GB, FlashAttention on):
- Open-Sora-Plan 65 frames 512²: baseline 99.65 s, VBench 80.39%; **TeaCache-slow 4.41× (22.62 s), 80.32%**; fast 6.83× (14.60 s), 79.72%.
- Open-Sora 1.2, 51 frames 480p: 44.56 s, 79.22% → slow 1.55× (28.78 s), 79.28%; fast 2.25× (19.84 s), 78.48%.
- Latte 16 frames 512²: 26.90 s → 1.86× (14.46 s); fast 3.28× (8.20 s), 76.69% vs 77.40%.
- Beats PAB and Δ-DiT/T-GATE at equal latency (Fig. 1).
- Reference metrics (PSNR/SSIM) drop at high speedups; VBench holds.

**Memory/speed.** Cache = one residual tensor (N × h). Wan-14B 720p: 0.77 GB FP16 [est]; ×2 with CFG. Overhead: one modulated input and a scalar polynomial — negligible.

**Assumptions that fail here.** Base models use 30–150 steps. With 4–8-step distilled models the adjacent outputs differ a lot; the gain at few steps is not stated.

**What transfers.**
- **Best caching method for this machine**: training-free, model-agnostic (any adaLN DiT), **tiny memory**, and a skipped step also **skips reading the weights** (zero I/O on skipped steps).
- The polynomial must be fitted per model (≈70 generations — hours on our GPU [est]; reuse published coefficients when they exist [ext]).

### V13 — FORA (2407.01425)

**Mechanism.** **Static caching** per layer: every N steps (t mod N = 0), run the full DiT and store each block's attention output and MLP output (cache[Attn][k], cache[MLP][k]). For the next N−1 steps, every block reuses its cached attention and MLP outputs and recomputes only the cheap parts (norm, modulation, residual adds) (§3.2, Fig. 3). No training.

**Results.**
- Table 1, DiT-XL/2-G ImageNet 256², 250 DDIM, cfg 1.5: baseline FID 2.30 (reproduced); N=2: FID 2.40, 2.08×; **N=3: 2.82, 2.80×** (their recommended point); N=5: 4.97, 4.57×; N=7: 9.80, 5.73×; N=10: 25.24, 8.07×. (The columns are mangled by pdftotext; this is my best alignment.)
- Table 2, PixArt-α 256² COCO: N=2 gives 1.54× (DPM-Solver) / 1.86× (IDDPM) with FID 28.15 → 38.20 (DPM-Solver, Inception) — a large FID cost.
- §3.3: N above 7 degrades FID sharply.

**Memory/speed.** Cache memory not stated. Hardware not stated in the extracted text.

**Assumptions that fail here.**
- Cache memory = 2 × N_tok × h × 2 B × L. Flux 1024²: 57 blocks × 57 MB ≈ 3.2 GB [est] — host RAM only. **Wan-14B 720p: 40 × 1.55 GB ≈ 62 GB** [est] — does not fit anywhere on this machine. FORA is image-only for us.
- 250-step samplers; at 20-step solvers the gain is smaller (PixArt result).

**What transfers.** For images with many steps, FORA-style per-layer reuse is simple. On our machine TeaCache dominates it (one residual instead of 2L tensors). Low priority.

### V14 — Sparse VideoGen (SVG) (2502.01776)

**Mechanism.**
- Video 3D full attention has two head types (§3.1, Fig. 3): **spatial heads** attend within the same frame (block-diagonal pattern, block size = tokens per frame); **temporal heads** attend to the same spatial position across frames (slash pattern, stride L = tokens per frame). Text tokens and the first frame are always attended.
- **Online profiling** (Alg. 1): per head and per step, sample ~1% of query rows (e.g. 32 of 18k), compute full, spatial-masked and temporal-masked attention for the sample, choose the mask with lower MSE. Overhead ~3%.
- **Layout transformation**: temporal heads are made contiguous by transposing token-major to frame-major order, so a block-sparse kernel can use dense tiles (tensor cores need ≥16 contiguous elements) (§4.2, Fig. 5).
- Theoretical sparsity: spatial c_s/N, temporal c_t/L (§3.3). Dense attention for the **first 25% of steps** (§5).
- Kernels in Triton + FlashInfer; also combined with FP8 attention.

**Results** (Table 1, **H100-80GB**, CUDA 12.4):
- CogVideoX-v1.5 I2V 720p 80 frames: 528 s → 237 s (**2.23×**), PSNR 28.17.
- T2V: 232 s (2.28×), PSNR 29.99.
- **HunyuanVideo 720p, 5.33 s, 128 frames: 612.37 PFLOP, 2253 s → 1171 s (1.92×), PSNR 29.55; + FP8 968 s (2.33×)**.
- Wan 2.1: 1.51× (abstract).
- Fig. 2: attention is 51% / 73% of latency for CogVideoX 17k / 45k tokens and more for HunyuanVideo at 120k.
- MInference (LLM sparse attention) fails on video quality.

**Assumptions that fail here.** Triton/FlashInfer kernels target tensor cores (sm80+). FP8 needs Ada/Hopper. At 480p the attention share is lower, so the end-to-end gain is lower [est].

**What transfers.**
- **The spatial/temporal head split is architecture-level and portable.** On a CUDA-core kernel the skip saving is still proportional to the skipped tiles.
- **The frame-major layout** helps us even without tensor cores: contiguous K/V tiles mean coalesced loads.
- With a 4 GB GPU, the spatial mask also **shrinks the K/V working set**: a spatial head needs only its frame's K/V (+ text + first frame) instead of the full 1.5 GB.
- Highest-value attention lever for video.

### V15 — SageAttention (2410.02367)

**Mechanism.**
- FlashAttention tiling with **INT8 Q, K** (per-block or per-token scales) and **P, V kept in FP16 with an FP16 accumulator** for PV.
- **Smooth K**: K ← K − mean(K) over tokens. Softmax is invariant to a per-row constant, so the result is exact; it removes the channel-wise bias of K that ruins INT8 (§4.2). Overhead < 0.2%.
- INT8 beats FP8 (E4M3/E5M2) for Q,K accuracy (Table 2). The FP16 accumulator for PV has no accuracy loss vs FP32 (Table 3–5) and is **2× faster on RTX 4090/3090** (consumer-GPU specific).
- Four variants (Table 6); SAGEAttn-B (per-block INT8 QK, FP16 PV) is the default. Adaptive choice per layer (§4.5).

**Results.**
- Kernel: 340 TOPS on RTX 4090 (52% of INT8 peak) vs FlashAttention2 165; ~2.1× FA2, ~2.7× xformers.
- Table 7 (RTX 4090): CogvideoX (2, 30, 17776, 64) 163 → 327 TOPS (2.01×); UltraPixel 2.14×; Unidiffuser 2.34×.
- Table 1: without smoothing K, CogVideo Fscore 3.768 → 1.924 (per-token); with smoothing 3.734.
- End-to-end metrics lossless on Llama2, CogvideoX, Unidiffuser, UltraPixel, TIMM (Table 8). Fig. 1: CogvideoX 90 s → 67 s on RTX 4090.

**Assumptions that fail here.** The CUDA kernel uses INT8 `mma(u8.u8.s32)` and `mma(f16.f16.f16)` tensor-core instructions (§1). TU117 has no tensor cores. GPUs tested: 3090, 4090, A100, L20, H100, H20 (SA2 Table 9) — none without tensor cores.

**What transfers.**
- **The numerical recipe maps directly onto CUDA-core instructions of the GTX 1650**: INT8 QKᵀ via `dp4a` (INT32 accumulate, exact) + PV with `HFMA2` (FP16 accumulate) — the same precision choices as SAGEAttn-B. Expected attention speedup ~1.3–1.6× vs an FP16 HFMA2 FlashAttention-style kernel (QKᵀ half ≈2× faster; PV and softmax unchanged) [est, must measure].
- **INT8 K/V storage halves the K/V working set** (Wan 720p: 1.55 → 0.77 GB), which matters more than speed on 4 GB.
- Smooth-K is free and exact; use it in any INT8 attention.

### V16 — SageAttention2 (2411.10958)

**Mechanism.**
- **Q, K to INT4** with **per-thread granularity**: tokens are grouped so that each GPU thread in the `mma` layout uses one scale, which removes per-token dequantization overhead (§3.2).
- **Smooth Q** (subtract per-block mean of Q; the correction term q̄·Kᵀ is a cheap GEMV added back) + smooth K.
- **P, V in FP8** (E4M3) with a **two-level accumulation**: the FP8 `mma.f32.f8.f8.f32` accumulator is actually FP22 (1-8-13 bits); results are flushed to an FP32 buffer after each block PV (§3.3). Optional smooth V.
- Two kernels (Table 3): SageAttn2-4b (INT4 QK) and SageAttn2-8b (INT8 QK), both FP8 PV.

**Results.**
- Peak 481 TOPS on RTX 4090; ~3× FA2 and ~4.5× xformers (abstract).
- Table 9 speedup vs FA2: 4090 2.93×, L20 2.46×, H100 2.61×, H20 3.12×; SageAttention1: 3090 1.97×, 4090 1.96×, A100 1.37×.
- Table 8 end-to-end: CogvideoX-1.5-5B on RTX 4090 1040 s → 577 s (8b) / 555 s (4b); **HunyuanVideo on L20 2221 s → 1486 s (8b) / 1435 s (4b)**; Mochi L20 2336 → 1316 / 1190 s.
- Table 2: 8b is nearly lossless; 4b loses some quality on some models (Flux, HunyuanVideo). Table 4: smoothing Q+K raises INT4 cosine similarity from 79.77% (Hadamard) / 98.07% (smooth K) to 99.46%.

**Assumptions that fail here.** INT4 tensor cores (sm80/sm89) and FP8 tensor cores (sm89/sm90). **Nothing of the kernel transfers to TU117.** There is no INT4 dot-product instruction on CUDA cores.

**What transfers.**
- Smooth-Q, per-thread scale grouping (a layout trick that also helps a dp4a kernel), and the "flush narrow accumulators to FP32 every block" rule. The last rule applies to our **FP16-accumulate HFMA2 PV loop**: accumulate a tile in FP16, add to FP32 after each K/V block.
- Its Table 1 also documents that the FP16-accumulator speed gain exists only on some GPUs. Measure on the GTX 1650.

### V17 — ToMe for Stable Diffusion (2303.17604)

**Mechanism.**
- Before each transformer component, partition tokens into src/dst. **One random dst token per 2×2 region** (randomness fixed across the CFG batch, else FID jumps). Merge the r% most similar src tokens into their best dst (average, Eq. 1).
- Compute; then **unmerge** by copying the merged output back to both tokens (Eq. 2).
- Similarity is computed once per block on the block input x (not keys).
- Best design: apply **only to self-attention** and **only to the highest-resolution blocks** (≥4096 tokens) (Table 3).

**Results.**
- Table 4, SD-v1.5 512², 50 PLMS, cfg 7.5, RTX 4090: baseline FID 33.12, 3.09 s/img, 3.41 GB/img; **r=50%: 33.02, 1.65 s, 0.89 GB; r=60%: 33.37, 1.52 s, 0.60 GB** (up to 2× faster, 5.6× less memory).
- With xFormers at 2048²: 28 s, 5.4× faster than the original (Fig. 1).
- "The memory benefits did not stack with xFormers"; with smaller images the speedup is less pronounced (§5).

**Memory/speed.** The large memory saving exists only because the baseline attention materializes the N×N matrix. With flash-style attention, memory is O(N) anyway.

**Assumptions that fail here.** U-Net SD with cross-attention; DiT and video results: not stated. The bipartite similarity costs ~0.75N·0.25N·h multiply-adds per use [est] — at 75k video tokens that is ~10% of a full attention per use.

**What transfers.**
- A token-reduction knob for **self-attention only**, quality-tested at 50–60% on U-Net SD. Fixed randomness across CFG is a needed detail.
- For DiT video, Sparse VideoGen is a better attention lever (no information loss in merged tokens). ToMe is a **low-priority experiment** for SD1.5/SDXL on our GPU.

### V18 — LCM: Latent Consistency Models (2310.04378)

**Mechanism.**
- Distill a pretrained latent diffusion model into a **consistency function** f_θ(z_t, ω, c, t) → z_0 that maps any point of the CFG-**augmented PF-ODE** directly to its solution (Eq. 12–14). ω (guidance scale) is an input, so CFG is "baked in" and **one forward per step** remains (no second CFG pass).
- Parameterization f = c_skip(t)·z + c_out(t)·ẑ_0 (Eq. 9).
- **Skipping-step**: consistency is enforced between t_{n+k} and t_n (k=20) using a DDIM / DPM-Solver step of the frozen teacher, which speeds up distillation a lot (§4.3).
- ω ∈ [2, 14] sampled during training.
- **LCF** fine-tunes a pretrained LCM on custom data without a teacher (§4.4).

**Results.**
- Table 1 (SD-2.1-base, 512², LAION-Aesthetics-6+, ω=8): LCM FID 1/2/4/8 steps = 35.36 / 13.31 / 11.10 / 11.84; CLIP 24.14 / 27.83 / 28.69 / 28.84. DPM-Solver++ at 4 steps FID 18.43 (my alignment of a mangled table; DDIM 4-step 22.38).
- Table 2 (768², SD-2.1): LCM 34.22 / 16.32 / 13.53 / 14.97.
- Training: **32 A100 GPU-hours** (4,000 iterations) for a 2–4-step LCM; Guided-Distill needs ≥45 A100-days (§4.2).

**Memory/speed.** Inference speed: proportional to steps (not stated separately).

**Assumptions that fail here.** 32 A100-hours for a 0.9B U-Net. For a 12–14B DiT, distillation needs much larger GPUs (FP32/BF16 training of 14B needs >200 GB with optimizer states [est]). **We cannot train it here.** We can only use published few-step checkpoints or LoRAs. [ext: LCM-LoRA exists for SD1.5/SDXL]

**What transfers.** Step reduction **and** CFG removal in one model: from 50×2 passes to 4×1 = **25× fewer forwards**. It composes with quantization and attention speedups. It conflicts with step caching (little redundancy left). Few-step models are more sensitive to quantization (MixDQ).

### V19 — ADD: Adversarial Diffusion Distillation (2311.17042)

**Mechanism.**
- The student (initialized from the pretrained model) denoises x_s for a few fixed student timesteps.
- Loss = adversarial hinge loss from a discriminator with a **frozen DINOv2 ViT-S feature network** and trainable heads, conditioned on text (CLIP-ViT-g) and image (DINOv2 CLS), R1 penalty on head inputs (Eq. 2–3), **plus** a distillation loss: the teacher denoises the re-noised student output, and the student matches it (Eq. 4; λ = 2.5, exponential or NFSD weighting).
- One teacher step is enough (Table 1f). Student pretraining is essential (random init FID 293.6, Table 1c).

**Results.**
- ADD-M (860M, SD-2.1/SD1.5) and ADD-XL (3.1B, SDXL backbone).
- Table 2 (SD1.5 based, COCO FID5k, **A100, 512², mixed precision**): DPM-Solver 25 steps 0.88 s, FID 20.1, CLIP 0.318; **ADD-M 1 step 0.09 s, FID 19.7, CLIP 0.326**. The other rows (progressive distillation, CFG-aware distillation, InstaFlow, UFOGen at 1–8 steps, 0.09–0.34 s, FID 22.4–37.2) are mangled in the text conversion; I do not assign them to methods.
- ADD-XL 1 step beats LCM-XL 4 steps and all GAN baselines in user preference; **ADD-XL 4 steps beats its teacher SDXL** (Fig. 5–6).
- Limitation: lower sample diversity (§5).

**Memory/speed.** 1 U-Net evaluation per image. Training resources: not stated in the extracted text.

**Assumptions that fail here.** Training needs teacher + student + discriminator in GPU memory. Not possible for large models here.

**What transfers.**
- **Published ADD-family checkpoints are the single largest speed lever available to us.** SDXL-Turbo/SD-Turbo (this paper) [ext]; FLUX.1-schnell and SD3-Turbo come from the latent-adversarial variant (LADD, not in our set) [ext].
- 1–4 steps × 1 pass. On our GPU, SDXL-Turbo at 512² is ~1–2 s DiT/U-Net compute per image [est].

---

## 2. Q1 — Memory and compute anatomy

### 2.1 Architecture numbers

| | Flux.1 (12B) [ext] | SD3 8B (d=38) [paper] | Wan 2.1 14B [ext cfg] | Wan 2.1 1.3B [ext cfg] | HunyuanVideo 13B [paper] |
|---|---|---|---|---|---|
| Blocks | 19 dual + 38 single | 38 MM-DiT (dual) | 40 (self + cross attn) | 30 | 20 dual + 40 single |
| Hidden h / heads | 3072 / 24 | 2432 / 38 | 5120 / 40 | 1536 / 12 | 3072 / 24 |
| FFN | 4h = 12288 | 4h | 13824 | 8960 | 12288 |
| VAE | f8, 16 ch [ext] | f8, 16 ch, patch 2 | 4×8×8, 16 ch, 127M | same | 4×8×8, 16 ch |
| Patch | 2×2 [ext] | 2×2 | 1×2×2 | 1×2×2 | not stated [ext: 1×2×2] |
| Text encoder | T5-XXL (4.7B enc) + CLIP-L, 512 T5 tokens [ext] | CLIP-L + CLIP-G + T5-XXL (4.7B), 154 tokens | umT5, 512 tokens; ">20 GB" | same | MLLM (size not stated; [ext] ~8B) + CLIP-L |

Parameter check with the block formula [est] (dual block = 2 streams × (QKV+O 4h² + MLP 8h² + adaLN 6h²) = 36h²; single block = QKV+MLP-in 7h² + out 5h² + mod 3h² = 15h²):
- Flux: 19·36h² + 38·15h² = 1254h² = **11.83B** (SVDQuant: BF16 22.2 GiB ✓).
- SD3 d=38: 38·36h² (h=2432) = **8.09B** ✓ "8B".
- HunyuanVideo: 20·36h² + 40·15h² = 1320h² = **12.46B** + embeddings ≈ 13B ✓.
- Wan-14B: 40·(8h² + 2·h·13824) = **14.05B** ✓.

**Key detail — per-token parameters.** adaLN modulation weights act on one vector per step, not on every token. In dual blocks each token passes through only its own stream. So the GEMM work per token uses fewer weights than the total:

| Model | Total params | Per-token (GEMM) params | Modulation (per-step GEMV) |
|---|---|---|---|
| Flux | 11.83B | **6.45B** (684h²) | 3.23B |
| SD3 8B | 8.09B | **2.70B** (456h²) | 2.70B |
| HunyuanVideo | 12.5B | **6.79B** (720h²) | ~2.9B |
| Wan-14B | 14.05B | **11.95B** (self 4h² + cross Q/O 2h² + FFN; cross K/V act only on 512 text tokens) | shared MLP |
| Wan-1.3B | 1.39B | **1.25B** | shared |

Engine consequence: about 27% (Flux) to 33% (SD3) of the weight bytes are modulation weights. They are read every step but cost almost no compute. For them the LLM-decode logic holds (bandwidth-bound GEMV). The modulation vectors depend only on (t, pooled text): **pre-compute all of them for all steps once per prompt** (a few MB), then never stream those weights during the denoising loop [est]. This cuts streamed bytes per step by 27–33%.

### 2.2 Token counts

Formula: tokens = (frames_latent) × (H/(8·p)) × (W/(8·p)); frames_latent = 1 + (F−1)/4.

| Case | Model | Pixels × frames | Latent (T×H×W×C) | Video/image tokens | + text | N used |
|---|---|---|---|---|---|---|
| (a) 1024² image | Flux / SD3 | 1024×1024 | 1×128×128×16 | **4,096** | +512 (Flux) / +154 (SD3) | 4,608 / 4,250 |
| (b) 480p, 5 s | Wan (832×480, 81 fr @16 fps [ext]) | 81 | 21×60×104×16 | **32,760** | cross-attn 512 | 32,760 |
| (b) 480p, 5.4 s | HunyuanVideo (848×480, 129 fr @24 fps [ext]) | 129 | 33×60×106×16 | **52,470** | +256 [ext] | 52,726 |
| (c) 720p, 5 s | Wan (1280×720, 81 fr) | 81 | 21×90×160×16 | **75,600** | cross-attn 512 | 75,600 |
| (c) 720p, 5.4 s | HunyuanVideo (1280×720, 129 fr) | 129 | 33×90×160×16 | **118,800** | +256 | 119,056 |

Cross-check: Sparse VideoGen says HunyuanVideo has "120k context length" (Fig. 2) ✓. The latents themselves are tiny (Wan 720p: 21×90×160×16 × 4 B = 19 MB).

### 2.3 FLOPs per forward pass

Formula [est]: linear = 2 · P_tok · N; self-attention = 4 · N² · h · L (QKᵀ + PV; Wan §4.3.1 uses the same "α=4"); cross-attention = 4 · N · N_txt · h · L.

| Case | Linear (TFLOP) | Attention (TFLOP) | Attn share | Total / forward | Passes per step | Per step |
|---|---|---|---|---|---|---|
| Flux 1024² | 59.5 | 14.9 | 20% | **74.4** | 1 (guidance-distilled) | 74 |
| Flux 512² (N=1,536) | 19.8 | 1.7 | 8% | 21.5 | 1 | 21.5 |
| SD3-8B 1024² | 22.9 | 6.7 | 23% | 29.6 | 2 (CFG) | 59 |
| Wan-1.3B 480p | 82 | 198 (+3 cross) | 71% | **283** | 2 (CFG) | 566 |
| Wan-14B 480p | 783 | 879 (+14 cross) | 52% | **1,676** | 2 | 3,352 |
| Wan-14B 720p | 1,807 | 4,682 (+32) | 72% | **6,521** | 2 | 13,042 |
| HunyuanVideo 480p | 717 | 2,050 | 74% | **2,766** | 1 (CFG-distilled) | 2,766 |
| HunyuanVideo 720p | 1,618 | 10,450 | 87% | **12,068** | 1 | 12,068 |

Validation: SVG Table 1 gives HunyuanVideo 720p (128 frames) = 612.37 PFLOP for 50 steps = **12.25 PFLOP/step** vs my 12.07 ✓. SVDQuant desktop 4090 BF16 Flux step = 657 ms → 74.4 TFLOP / 0.657 s = 113 TFLOPS (≈68% of a 4090's FP16-with-FP32-accumulate peak [ext]) ✓ plausible.

**Time on the GTX 1650 at 4 TFLOPS sustained [est]:**

| Workload | Steps × passes | Total FLOP | Time |
|---|---|---|---|
| Flux-schnell 1024², 4 steps | 4 × 1 | 0.30 PFLOP | **~75 s** (+T5, VAE ≈ +10 s) |
| Flux-dev 1024², 28 steps | 28 × 1 | 2.1 PFLOP | ~8.7 min |
| SD3-8B 1024², 28 steps CFG | 28 × 2 | 1.66 PFLOP | ~6.9 min |
| Wan-1.3B 480p, 50 steps CFG | 50 × 2 | 28 PFLOP | ~2 h |
| Wan-1.3B 480p, 6 steps, no CFG (distilled) | 6 × 1 | 1.7 PFLOP | ~7 min; ~4 min with INT8 + sparse attn |
| Wan-14B 480p, 50 steps CFG | 50 × 2 | 168 PFLOP | ~11.7 h |
| Wan-14B 720p, 50 steps CFG | 50 × 2 | 652 PFLOP | ~45 h |
| HunyuanVideo 720p, 50 steps | 50 × 1 | 603 PFLOP | ~42 h |

### 2.4 Memory budget anatomy (FP16 unless stated)

**Text encoders (once per prompt).**
- T5-XXL encoder 4.7B [paper V02]: FP16 9.4 GB, Q8 ~5 GB, Q4 ~2.7 GB. Compute for 512 tokens = 2 × 4.7e9 × 512 = 4.8 TFLOP ≈ 1.2 s on GPU; streaming the Q4 weights from SSD ≈ 0.5 s. **This is the only "LLM-like" bandwidth-sensitive part**, and it runs once.
- umT5 (Wan): size not stated; [ext] encoder ≈5.7B → FP16 ~11.4 GB, Q4 ~3.3 GB.
- HunyuanVideo MLLM: size not stated; [ext] ~8B → Q4 ~4.7 GB. Use our LLM engine.
- Numerics: T5 in FP16 overflows [ext]; on Turing (no BF16) run its FFN in FP32 accumulate or INT8 dp4a.
- After encoding: keep only the outputs (Flux 512×4096×2 B = 4 MB). Free the encoder before the DiT loop.

**DiT weights.**

| Model | FP16 | Q8 (~8.5 bpw) | Q4 (~4.5 bpw) | Fits 3.3 GB VRAM? | Fits ~4.5 GB RAM? |
|---|---|---|---|---|---|
| Flux 12B | 23.7 GB | 12.6 GB | 6.7 GB (SVDQuant 6.1 GiB) | no | no |
| SD3 8B | 16.2 GB | 8.6 GB | 4.6 GB | no | barely |
| Wan-14B | 28.1 GB | 14.9 GB | 7.9 GB | no | no |
| HunyuanVideo 13B | ~26 GB | ~13.8 GB | ~7.3 GB | no | no |
| Wan-1.3B | 2.8 GB | 1.5 GB | 0.8 GB | **yes (Q8 or Q4)** | yes |

**Activations (per block, transient; inference keeps only one block's tensors).** S = N·h·2 B is the residual stream.

| Case | S (residual) | QKV | MLP hidden | Peak/block, naive | Full K+V (FP16 / INT8) | N×N scores if not flash-style |
|---|---|---|---|---|---|---|
| Flux 1024² | 28 MB | 85 MB | 113 MB (single block fused 7h: 198 MB) | ~0.4 GB | 57 / 28 MB | 1.0 GB/block (all heads) |
| SD3 1024² | 21 MB | 62 MB | 83 MB | ~0.2 GB | 41 / 21 MB | — |
| Wan-14B 480p | 335 MB | 1.0 GB | 0.9 GB | ~1.7 GB | 0.67 / 0.34 GB | 86 GB (impossible) |
| Wan-14B 720p | 774 MB | 2.3 GB | 2.1 GB | **~3.9 GB > VRAM** | 1.55 / 0.77 GB | 457 GB |
| HunyuanVideo 720p | 731 MB | 2.2 GB | fused 7h: **5.1 GB** | **~6 GB > VRAM** | 1.46 / 0.73 GB | — |
| Wan-1.3B 480p | 101 MB | 0.3 GB | 0.6 GB | ~0.8 GB | 0.2 / 0.1 GB | — |

Rules that follow [est]:
- **Images: activations are small (≤0.5 GB); weights are the whole problem.**
- **Video: activations are the VRAM problem.** All linear layers and FFNs are token-wise: run them in token chunks (e.g. 8k tokens). Attention needs all K/V resident: keep K/V in INT8 (Sage recipe) and stream Q chunks. With CFG, keep the second residual stream in host RAM (0.77 GB at 720p).
- Flash-style attention (O(N) memory) is mandatory for video. FlashAttention-2 needs sm80 [ext]; xformers/PyTorch memory-efficient attention and FlashAttention-1 support sm75 [ext], but they may assume tensor-core `mma`. Plan a CUDA-core tiled attention kernel of our own.

**VAE decode.**
- Image (f8, 16 ch, ~84M params [ext]): decoder ≈ 1.5 TFLOP at 512², ≈ 6 TFLOP at 1024² [est] → ~1.5–2 s on our GPU. Activation peak at 1024²: ~1–1.5 GB FP16 [est] → **tile** (HunyuanVideo §4.1.2) or run at 512² tiles.
- Video: Wan-VAE (127M) processes 4-frame chunks with a 2-frame causal cache (Wan §4.1.3). At 720p a 4-frame full-resolution tensor with ~96–128 channels is 0.7–0.9 GB [est] → add spatial tiling. Decode time for 81 frames at 720p: not stated; [est] tens of TFLOP to a few hundred TFLOP → roughly 0.5–2 minutes on our GPU.

**Peak memory plan per phase (720p Wan-14B example) [est]:**
1. Text phase: umT5 Q4 streamed; GPU ~1.5 GB; RAM staging ~1 GB.
2. DiT phase: VRAM = K/V INT8 0.77 GB + residual 0.77 GB + Q/FFN chunk buffers ~0.8 GB + 2 block weight buffers (Q4: 2 × 0.2 GB) ≈ 2.8 GB. RAM = second CFG residual 0.77 GB + pinned staging 0.5 GB + text K/V cache 0.4 GB + latents.
3. VAE phase: free everything else; tiled chunked decode.

---

## 3. Q2 — Ranking of techniques for this machine

Criteria: (1) FLOPs removed (we are compute-bound), (2) quality cost measured in the paper, (3) needs training? (we cannot train ≥1B models), (4) needs hardware TU117 lacks, (5) memory cost on 4 GB / 8 GB.

### 3.1 Ranked list

| Rank | Technique (lever) | Gain on our machine [est] | Hardware fit on GTX 1650 | Notes |
|---|---|---|---|---|
| 1 | **Step + CFG distillation via published checkpoints** (ADD/LADD: SDXL-Turbo, Flux-schnell; LCM/LCM-LoRA; HunyuanVideo CFG distill; distilled Wan [ext]) | 5–25× (50×2 → 4×1 forwards) | Any (no special math) | We cannot train it; depend on existing checkpoints. Few-step models need higher bits (MixDQ). |
| 2 | **Block-streaming engine** (weights from SSD/RAM per block, double-buffered, perfect prefetch) | Enables 8–14B models at ~1–7% overhead (§Q4) | PCIe + NVMe only | Not a speedup — the enabler. Pre-compute modulation vectors to cut streamed bytes ~30%. |
| 3 | **Sparse VideoGen (spatial/temporal heads)** | Video: attention 52–87% of FLOPs; paper 1.5–2.3× end-to-end | Kernel needs rewrite (paper: Triton/FlashInfer, tensor cores) | Highest video lever. Also shrinks the K/V working set. |
| 4 | **INT8 attention, SageAttention-1 recipe on dp4a + HFMA2** | 1.3–1.6× attention; K/V memory ÷2 | Paper kernel needs INT8 tensor cores; **recipe ports to dp4a** | Smooth-K is exact and free. |
| 5 | **W8A8 INT8 linear layers (ViDiT-Q recipe + PTQ4DiT adaLN folding)** | 1.5–2× on the linear FLOPs (dp4a ~2× HFMA2); exact INT32 accumulate | Paper kernels use INT8 tensor cores; **dp4a port** | Biggest image lever after steps (images are 77–92% linear FLOPs). |
| 6 | **TeaCache** | 1.5–4.4× on 30–150-step models; skipped steps also skip weight I/O | Any | Small memory (one residual). Gain at 4–8 steps: not stated (likely small). |
| 7 | **Weight-only GGUF-style Q4/Q5/Q8 (+ optional SVDQuant low-rank branch)** | 0× speed (compute-bound; SVDQuant/MixDQ measured); saves disk and RAM | Any | Choose Q5/Q8 for quality where disk allows; low-rank branch fixes 4-bit error. |
| 8 | Wan attention/CFG cache (FasterCache-style) | 1.62× on Wan-14B (paper) | Any | Per-layer attention cache costs ~0.8–1.5 GB per layer at video sizes → host RAM/SSD problem. CFG cache is cheap. |
| 9 | FORA | 2–2.8× on 250-step DiT images | Any | Per-layer caches: 3.2 GB (Flux) / 62 GB (Wan 720p). Image-only for us. |
| 10 | ToMe-SD | up to 2× on SD U-Net | Any | U-Net only; memory gain only vs naive attention. DiT/video: not stated. |
| 11 | DeepCache | 2.1–2.3× on SD1.5 | Any | U-Net only. |
| 12 | SVDQuant **W4A4 compute** | 3× on 4090 — **0× here** | **Needs INT4/FP4 tensor cores — absent** | Use only its weight format. |
| 13 | SageAttention-2 (INT4 QK, FP8 PV) | — | **Needs INT4 + FP8 tensor cores — absent** | Take smooth-Q and FP32 flush ideas only. |
| 14 | MixDQ, Q-Diffusion, PTQ4DiT (as methods) | Calibration ideas | Simulated or tensor-core kernels | BOS/constant pre-compute; timestep-wide calibration; adaLN folding. |
| 15 | BitsFusion (1.99-bit QAT) | Storage only | Any | Needs 8–32 A100 training; U-Net only. |

### 3.2 What composes

| | Steps (distilled) | TeaCache/FORA | Sparse attn (SVG) | INT8 attn (Sage) | W8A8 linear | Weight-only Q4 | Block streaming |
|---|---|---|---|---|---|---|---|
| Steps | — | **conflict** (few steps leave little redundancy) | yes | yes | yes, but needs ≥8-bit (MixDQ) | yes | yes |
| TeaCache | conflict | — (pick one cache) | yes | yes | yes | yes | yes, skipped steps skip I/O |
| SVG | yes | yes | — | **yes** (SVG + FP8 in paper; SpargeAttn [ref] combines sparse + Sage) | yes | yes | yes |
| Sage INT8 | yes | yes | yes | — | yes | yes | yes |
| W8A8 | yes | yes | yes | yes | — | W8 storage replaces Q4 or dequant Q4→INT8 on load | yes |

Notes:
- Wan already combines its diffusion cache with FP8 GEMM and 8-bit attention (§4.4). SVG combines with FP8 attention (Table 1).
- ToMe and SVG overlap (both cut attention); do not stack them without a test.
- CFG distillation (HunyuanVideo 1.9×) composes with everything.

### 3.3 Hardware the GTX 1650 lacks

| Requirement | Used by | GTX 1650 (TU117) |
|---|---|---|
| INT8 tensor-core `mma` | SageAttention-1 kernel, ViDiT-Q, MixDQ (CUTLASS), Wan int8/TensorRT | **absent** → use `dp4a` (INT8×4, INT32 acc) on CUDA cores |
| INT4 tensor cores | SVDQuant INT4, SageAttention-2 4b | **absent**, no CUDA-core INT4 dot product |
| FP8 tensor cores (sm89/90) | SageAttention-2 PV, Wan FP8 GEMM, FA3-FP8, SVG+FP8 | **absent** |
| NVFP4 (Blackwell) | SVDQuant NVFP4 | absent |
| BF16 | all large models are trained in BF16 | **absent** → FP16 overflow risk (T5, some Flux/Wan activations [ext]); need clamps, FP32 islands or INT8 |
| FP16 accumulate faster than FP32 | SageAttention PV (4090/3090) | HFMA2 is 2× FP32 on Turing [ext] → recipe applies |
| FlashAttention-2 (sm80) | baselines in V12, V14–V16 | not available; write our own tiled kernel |
| Triton block-sparse / FlashInfer | SVG | Triton supports sm75 in principle [ext]; performance without tensor cores unknown |

---

## 4. Q3 — Offloading for diffusion: when is block swapping free?

### 4.1 Derivation

For one transformer block i with P_i weights stored at b bytes per weight, processing N tokens for k sequences (k = 2 with CFG batched, else 1):
- compute time T_c ≥ 2·P_tok,i·N·k / C (linear only; attention adds more),
- transfer time T_x = P_i·b / BW.

With double buffering (load block i+1 while block i computes), the transfer is hidden when T_x ≤ T_c:

```
N · k  ≥  N*  =  (P_i / P_tok,i) · b · C / (2 · BW)
```

Ignoring the modulation share (P_i ≈ P_tok,i; conservative):

| Source → GPU | BW | b = Q4 (0.5625 B) | Q8 (1.0625 B) | FP16 (2 B) |
|---|---|---|---|---|
| NVMe SSD (through RAM) | 5 GB/s | **225 tokens** | 425 | 800 |
| RAM (pinned) over PCIe | 10 GB/s | 113 | 213 | 400 |
| Same, if we reach INT8 dp4a at C = 8 TOPS | 5 GB/s | 450 | 850 | 1,600 |

Every case in §2.2 has N ≥ 1,536. **On this GPU, block swapping costs almost nothing for any image ≥512² and any video, even directly from the SSD, and even for FP16 weights.**

### 4.2 Per-step numbers [est]

| Workload | Bytes streamed per step | SSD time (5 GB/s) | Compute per step (4 TFLOPS) | Ratio |
|---|---|---|---|---|
| Flux Q4 1024² | 6.7 GB (4.7 GB if modulation is pre-computed) | 1.3 s (0.9 s) | 18.6 s | 7% (5%) |
| Flux Q4 512² | same | 1.3 s | 5.4 s | 25% — still hidden with overlap, but tight |
| SD3-8B Q4 1024², CFG batched | 4.6 GB | 0.9 s | 14.8 s | 6% |
| Wan-14B Q4 480p, CFG batched per block | 7.9 GB | 1.6 s | 838 s | 0.2% |
| Wan-14B **FP16** 480p | 28 GB | 5.6 s | 838 s | 0.7% |
| HunyuanVideo Q4 720p | 7.3 GB | 1.5 s | 3,017 s | 0.05% |
| Wan-1.3B 480p | fits in VRAM | 0 | 141 s | 0 |

Contrast with the LLM part: decode is N = 1, so N* is never reached and every byte costs time (00_SYNTHESIS §1). Contrast with fast GPUs: an RTX 4090 at ~165 TFLOPS [ext] with BF16 weights over ~20 GB/s needs N* ≈ 8,000 tokens; Flux 1024² (4,608) is below it, which is why SVDQuant measured a **6.7× slowdown (4382 vs 657 ms/step) from per-layer offload on the laptop 4090**. **Offloading is cheap on slow GPUs and expensive on fast ones.**

### 4.3 Conditions that break "nearly free"

1. **RAM is too small to be a cache.** The access pattern is cyclic (block 0 → block L−1, then repeat). LRU page caching is the worst policy for cyclic access: with a 7.9 GB model and ~4.5 GB RAM, LRU evicts every block just before it is needed again. Use **direct I/O** + a **static pinned set** (keep the first X GB of blocks resident in RAM forever; stream the rest from SSD). This is the colibri lesson in a simpler form: here the access order is fully known, so prefetch is perfect and needs no prediction.
2. **SSD throughput for 100–350 MB block reads** is close to sequential speed, but Windows buffered I/O adds a copy. Use unbuffered reads into pinned buffers (`FILE_FLAG_NO_BUFFERING`, 4 KB alignment) [ext].
3. **CFG must be batched per block.** If the cond and uncond passes run as two separate full forwards, every block is read twice. Batching costs a second residual stream (0.77 GB at 720p) — keep it in host RAM and swap it per block (another 0.77 GB × 2 per block of PCIe traffic at 720p video: 40 blocks × 1.5 GB = 62 GB per step ≈ 6 s at 10 GB/s vs ≈3,260 s compute per 720p step → still negligible).
4. **Small images with few steps** (512², 1–4 steps, Q8 or FP16) approach N*. Keep small models (SD1.5/SDXL-Turbo, Wan-1.3B) fully resident instead.
5. **When we make compute faster** (INT8, sparsity), N* grows proportionally. A 4× faster kernel at 512² would make streaming visible.
6. **Disk space, not bandwidth, limits FP16.** Wan-14B FP16 streams fine (0.7%), but 28 GB of the 82 GB free disk is a lot. Q8 (15 GB) is the better default for video if quality matters; Q4 when disk is short.
7. **The text encoder** is the only phase where bytes dominate. Run it once; free it.

---

## 5. Q4 — One engine for text LLM + diffusion

### 5.1 Share

| Component | Why it can be shared |
|---|---|
| **Weight store / file layout** | Both models are a list of transformer blocks. Store each block as one contiguous, 4 KB-aligned region (as colibri does per expert). One reader pool, one pinned staging ring, one H2D double buffer, one RAM-budget cap. |
| **Quant storage formats** | GGUF-style block formats (Q4_K/Q5_K/Q8_0) with per-group scales serve both. Add an optional **low-rank FP16 side branch** (SVDQuant) — it helps LLM 3–4-bit quality too. One converter that runs in constant memory (the rai design from E). |
| **Text-encoder path** | T5 encoder, umT5, the HunyuanVideo MLLM, CLIP: all are LLM-style transformers run in prefill mode. The LLM engine runs them directly (keep the final hidden states; no KV cache, no sampling). |
| **Streaming controller** | Same mechanism; different policy. LLM MoE: predicted, sparse, cache-heavy. Diffusion: fixed cyclic order, perfect prefetch, static pinned subset. |
| **Kernels (partly)** | RMSNorm/LayerNorm, RoPE (1D for LLM, 3D for video = 3 segments of the same kernel), SiLU/GELU, softmax, embedding, dequantize-in-register routines, INT8 per-token quantize kernel. |
| **Numerics policy for Turing** | Same "no BF16" problem for both: FP32 islands in norms/softmax, INT8 dp4a with INT32 accumulate where FP16 overflows. |
| **Security** | Same rule (00_SYNTHESIS §4.5): download from the publisher, verify hashes, scan for outlier injection before quantizing. Quantized diffusion weights from third parties are an equal risk. [est: no diffusion paper in this set tests it] |

### 5.2 Must differ

| Aspect | LLM decode | Diffusion |
|---|---|---|
| Bottleneck | bytes (GEMV, N=1) | FLOPs (GEMM, N = 4k–120k) |
| Best linear kernel | weight-only dequant GEMV (W4A16) | **W8A8 dp4a GEMM** with per-token dynamic activation scales fused into the preceding norm/modulation; weight-only gives no speed |
| Attention | 1 query vs growing KV cache; causal | N×N full, non-causal, flash-tiled; sparse masks (spatial/temporal); INT8 K/V resident for the step |
| Memory that grows | KV cache with context | activations ∝ N per block (transient); needs token-chunking of linear/FFN |
| Per-step constants | none | adaLN modulation vectors, timestep embeddings, cross-attn text K/V, BOS-like tokens → pre-compute once per prompt |
| Control loop | sampler over tokens | sampler over timesteps (Euler/flow-matching, time shift), CFG batching, TeaCache decision per step |
| Extra modules | none | VAE: conv2d/conv3d (causal), GroupNorm/RMSNorm over channels, upsampling, **tiled + chunked decode** |
| Quant calibration | static per layer | must cover all timesteps (Q-Diffusion), CFG branches and text/image token groups (ViDiT-Q) |
| Caches | KV cache, expert cache | feature/residual caches across steps (TeaCache), not across tokens |

Design consequence: one block-store + streamer + quant converter + LLM-prefill path; two compute back-ends: a GEMV back-end (LLM decode) and a GEMM + flash-attention + conv back-end (diffusion). The LLM prefill path also benefits from the GEMM back-end (prefill is compute-bound too).

---

## 6. Q5 — Blind spots (papers cited by our 19 that we do not have)

Priority: **H** = likely changes our design for this machine; **M** = useful detail; **L** = context only. arXiv IDs are taken from the reference lists of our papers unless marked "not in references".

### 6.1 Architecture we have no paper for

- **FLUX.1 (Black Forest Labs)** — **H**. No paper exists; only GitHub/model cards (cited by SVDQuant and HunyuanVideo [47] without arXiv ID). Facts used here are [ext]: 19 double-stream (MM-DiT) + 38 single-stream blocks, hidden 3072, 24 heads × 128, MLP 4×, 16-channel f8 VAE, 2×2 patch, T5-XXL (512 tokens for dev) + CLIP-L pooled, RoPE, guidance distillation in "dev", adversarial few-step distillation in "schnell" (4 steps). The parameter count from my formula (11.83B) matches SVDQuant's 22.2 GiB BF16. **Action: read the reference code to confirm the block layout before writing kernels.**

### 6.2 Efficient architectures / VAEs for small machines

| Paper | arXiv | Pri | Why |
|---|---|---|---|
| SANA: efficient high-resolution synthesis with linear DiT | 2410.10629 | **H** | Linear attention + a 32× VAE → far fewer tokens; designed for small GPUs |
| Deep Compression Autoencoder (DC-AE) | 2410.10733 | **H** | 32–64× spatial compression → 16–64× fewer tokens than f8 |
| LTX-Video: realtime video latent diffusion | 2501.00103 | **H** | Small, fast video model with very high VAE compression; a candidate for this GPU |
| CogVideoX | 2408.06072 | M | Model used in SVG/Sage benchmarks; 2B variant fits our budget |
| PixArt-Σ (KV token compression) | 2403.04692 | M | Key/value token compression for 4K images |
| PixArt-α | 2310.00426 | L | Cheap T2I DiT baseline |
| Step-Video-T2V | 2502.10248 | L | 8×16×16 VAE, very large model |
| Movie Gen | 2410.13720 | L | Context only |
| Open-Sora / Open-Sora Plan | 2412.20404 / 2412.00131 | L | Base models of TeaCache tests |
| Latte | 2401.03048 | L | Base model of TeaCache tests |
| Stable Video Diffusion | 2311.15127 | L | U-Net video |
| SDXL | 2307.01952 | M | U-Net that fits; ADD/MixDQ target |
| SnapFusion (mobile SD in 2 s) | 2306.00980 | M | Mobile architecture + step distillation |

### 6.3 Step reduction / distillation

| Paper | arXiv | Pri | Why |
|---|---|---|---|
| Improved Distribution Matching Distillation (DMD2) | 2405.14867 | **H** | Basis of most few-step video distillations [ext] |
| One-step DMD (original) | not in references | M | — |
| Latent Adversarial Diffusion Distillation (LADD; SD3-Turbo, base of Flux-schnell [ext]) | not in references | **H** | Explains the checkpoints we will use |
| Consistency Models | 2303.01469 | M | Theory under LCM |
| Progressive Distillation | 2202.00512 | L | Older baseline |
| InstaFlow | 2309.06380 | L | Rectified-flow 1-step |
| UFOGen | 2311.09257 | L | Diffusion-GAN 1-step |
| Flow Straight and Fast (rectified flow) | 2209.03003 | M | Theory under SD3/Flux/Wan samplers |
| Flow Matching | 2210.02747 | L | Theory |
| SDXL-Lightning, Hyper-SD, CausVid, distilled Wan checkpoints | not in references | M | Practical few-step checkpoints [ext] |

### 6.4 Caching

| Paper | arXiv | Pri | Why |
|---|---|---|---|
| FasterCache (CFG cache + residual compensation, used by Wan) | 2410.19355 | **H** | Cheap CFG reuse; Wan's 1.62× builds on it |
| Pyramid Attention Broadcast (PAB) | 2408.12588 | M | TeaCache baseline; attention reuse for video |
| DiTFastAttn | 2406.08552 | M | Attention compression across steps/heads; SVG baseline |
| Δ-DiT | 2406.01125 | L | Block-residual caching |
| T-GATE (cross-attention gating) | 2404.02747 | M | Drop cross-attention after early steps — cheap for Wan |
| Learning-to-Cache | 2406.01733 | M | Learned layer caching |
| Block caching ("Cache me if you can") | 2312.03209 | M | Block-level reuse |
| FRDiff | 2312.03517 | L | Feature reuse |

### 6.5 Attention

| Paper | arXiv | Pri | Why |
|---|---|---|---|
| SpargeAttn (sparse + Sage quantized, any model) | 2502.18137 | **H** | Combines our ranks 3 and 4 |
| Sliding Tile Attention (fast video) | 2502.04507 | **H** | Hardware-friendly local 3D attention for video |
| FlashAttention-2 | 2307.08691 | M | Tiling reference for our CUDA-core kernel |
| FlashAttention-3 | 2407.08608 | L | Hopper only |
| SageAttention3 (FP4 microscaling) | 2505.11594 | L | Blackwell only |
| SageAttention2++ | 2505.21136 | L | Newer GPUs |
| MInference | 2407.02490 | L | LLM sparse attention; fails on video per SVG |
| DuoAttention | 2410.10819 | L | LLM |

### 6.6 Quantization / systems

| Paper | arXiv | Pri | Why |
|---|---|---|---|
| Q-DiT | 2406.17343 | M | DiT PTQ baseline in ViDiT-Q |
| PTQD | 2305.10657 | L | Quantization-noise correction |
| EfficientDM (QAT-LoRA for diffusion) | 2310.03270 | M | Low-cost QAT path |
| QuaRot (Hadamard) | 2404.00456 | M | Rotation used by ViDiT-Q; also in our LLM gap set |
| PipeFusion / xDiT | 2405.14430 | L | Multi-GPU; patch pipelining idea for tiles |
| TensorRT diffusion (Wan real-time) | not in references | L | Closed tool |

---

## 7. Proposed next steps (for discussion; nothing built)

1. **Measure the GPU first** (one afternoon): FP16 HGEMM (HFMA2) TFLOPS, FP32-acc TFLOPS, `dp4a` INT8 TOPS, and whether sm_75 `mma.sync` runs on TU117 (and at what speed). Every estimate in this file depends on the 4 TFLOPS / 8 TOPS assumptions.
2. **Baseline with existing tools** (ComfyUI or diffusers + GGUF weights, block offload): Flux-schnell Q4 1024² 4 steps; SD3.5/SD3 without T5; Wan-1.3B 480p. Record s/step, VRAM, RAM, SSD GB/s.
3. **Engine prototype in the order of value:** (a) block streamer with static pinned set + perfect prefetch + pre-computed modulation; (b) token-chunked linear/FFN + INT8-K/V tiled attention (Sage-1 recipe on dp4a/HFMA2); (c) W8A8 dp4a GEMM (ViDiT-Q recipe); (d) SVG spatial/temporal masks; (e) TeaCache for non-distilled models.
4. **Read first from the blind spots:** FasterCache, SpargeAttn, Sliding Tile Attention, SANA, DC-AE, LTX-Video, DMD2.
