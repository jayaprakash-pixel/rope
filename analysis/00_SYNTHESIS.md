# 00 — Synthesis: running very large LLMs in small RAM / VRAM

Date: 2026-11-12. Sources: the six analysis files in this folder (A–F), the 12 original papers, the 9 repos, and 47 gap papers in `papers\gaps\`. Numbers marked **[est]** are my bandwidth-ceiling estimates, not measurements.

| File | Scope |
|---|---|
| `A_foundation_papers.md` | GGUF, llama.cpp quant types, imatrix, GPTQ, AWQ, SmoothQuant, BitNet, BitNet b1.58 |
| `B_2026_quant_papers.md` | Drop-by-Drop, OffQ, WINDQuant, Outlier Injection attack |
| `C_kv_serving_compression_papers.md` | PuzzleKV, cμ-rule, ZipServ |
| `D_llama_cpp.md` | llama.cpp: mmap, offload, MoE cache, quant kernels, KV, extension points |
| `E_lowmem_engines.md` | colibri, MemVanta, reame, rai |
| `F_python_libs.md` | AirLLM, MLX LM, bitsandbytes, AutoGPTQ |

---

## 1. The one law that controls everything

During decode (one token at a time, one user), every weight that the token uses must move from where it is stored to the compute unit once per token. So:

```
tokens/s  <=  bandwidth of the slowest tier used  /  bytes read from that tier per token
```

Compute is almost never the limit at batch size 1. Memory traffic is the limit. Every technique in our papers and repos attacks one term of this equation:

| Lever | What it changes | Methods we have |
|---|---|---|
| 1. Fewer bits per weight | bytes per token, all tiers | GGUF K/I-quants, GPTQ, AWQ, AQLM, QuIP#, QTIP, Drop-by-Drop, WINDQuant, BitNet, QMoE |
| 2. Fewer weights touched per token | bytes per token | MoE routing, activation sparsity (Deja Vu, PowerInfer, LLM in a Flash) |
| 3. Hot weights in faster tiers | which tier serves the bytes | expert cache (colibri, llama.cpp `--moe-cache-mib`, ProMoE, HOBBIT), hot-neuron split (PowerInfer) |
| 4. More tokens per weight pass | amortizes the bytes | batched prefill, speculative decoding (Leviathan, Medusa, EAGLE), MTP |
| 5. Faster slow tier | bandwidth | direct I/O, async reader pool, pinned memory, multi-SSD (colibri +37.5% with 2 drives) |
| 6. Smaller KV cache | RAM/VRAM left for weights | KIVI, KVQuant, PuzzleKV, MLA (DeepSeek-V2/V3), StreamingLLM |

## 2. This machine (measured 2026-11-12)

| Part | Spec | Usable bandwidth [est] | Usable capacity for weights [est] |
|---|---|---|---|
| CPU | Ryzen 5 5600H, 6C/12T, AVX2 (no AVX-512) | — | — |
| RAM | **8 GB, one DIMM, DDR4-3200, single channel** | ~20 GB/s (25.6 peak) | ~4–5 GB after Windows |
| GPU | **GTX 1650, 4 GB**, Turing, no BF16 | ~128 GB/s | ~3.3 GB |
| PCIe to GPU | laptop, likely PCIe 3.0 | ~10–12 GB/s | — |
| SSD | Samsung PM9A1 512 GB NVMe PCIe 4.0 | ~5–6.5 GB/s sequential, less for random 20 MB reads | **82 GB free** |

Three hard facts follow:

1. **Disk is the capacity limit.** 82 GB free space cannot hold a 700B model at any bit-width that exists today (700B × 2 bits ≈ 175 GB; GLM-5.2 744B int4 ≈ 390 GB). A 400B+ model needs a second drive.
2. **RAM is very small and slow.** Single-channel 8 GB is the weakest point. A second 8–16 GB DIMM would double the RAM bandwidth (dual channel) and more than double the RAM cache. This is the cheapest hardware upgrade with the largest effect.
3. **The SSD is the real "big memory".** ~6 GB/s NVMe is only ~3× slower than this RAM. So SSD streaming is not a desperate fallback here. It is the main design.

## 3. What fits, and how fast [est]

Ceilings from §1, with 3.3 GB VRAM + 4.5 GB RAM resident, the remainder from SSD at 5 GB/s. Real results are typically 1.5–3× below the ceiling (colibri measured 2–4× below).

| Model | Format | File size | Bytes from SSD per token | Ceiling tok/s |
|---|---|---|---|---|
| 8B dense | Q4_K_M | ~4.9 GB | 0 | ~15 |
| 14B dense | Q4_K_M | ~9 GB | ~1 GB | ~3 |
| 32B dense | IQ3_XXS | ~13 GB | ~5 GB | ~0.8 |
| 70B dense | Q4_K_M | ~43 GB | ~35 GB | ~0.14 |
| 70B dense | IQ2_XS (~2.3 bpw) | ~20 GB | ~12 GB | ~0.4 |
| 30B MoE, ~3B active | Q4_K_M | ~18 GB | ≤1.7 GB cold, much less with cache | ~3–10 |
| 120B MoE, ~5B active | MXFP4 | ~61 GB | ≤3 GB cold | ~1.5–3 |
| 670–750B MoE, 32–40B active | int4 | 390+ GB | ~12.7 GB cold (colibri math) | does not fit on disk |

**Main conclusion:** dense models above ~32B are disk-bound at under 1 tok/s on this machine, whatever we do. **Large MoE models are the only route to "very large model, usable speed"**: total parameters live on SSD, and each token touches only a few percent of them. Every serious low-memory project we studied (colibri, llama.cpp MoE cache, HOBBIT, ProMoE, Fiddler, MoE-Lightning, Eliseev & Mazur) converges on this.

## 4. What each source contributes

### 4.1 Quantization (lever 1)
- **4 bits is almost free.** GPTQ and AWQ cost ~0.1 perplexity on 70B models (A §4–5). GGUF Q4_K_M is the practical default.
- **3 bits costs a little, 2 bits costs a lot** with scalar methods (A: +1–5 PPL at 2-bit). Below 3 bits, vector/codebook methods win: AQLM, QuIP#, QTIP, VPTQ, HIGGS (gap papers). WINDQuant reaches Llama-3-70B PPL 5.10 at ~2 bits but has no kernel (B).
- **Drop-by-Drop** (B): one checkpoint, 3/4/5 bits, 1 bit per codebook. This enables **"load more precision when memory is free"**: a hot expert in VRAM at 5 bits, a cold expert from SSD at 3 bits. It has no CPU kernel and the quantizing cost is high (~240 GPU-hours for 32B, unit uncertain).
- **MoE experts tolerate low bits.** QMoE compresses a 1.6T MoE below 1 bit per parameter (gap paper). HOBBIT loads low-bit copies of cache-miss experts and high-bit copies of hot ones (gap paper). This matches the Drop-by-Drop idea.
- **BitNet b1.58** (A): ternary, 7× less memory at 70B, but needs training from scratch. Only small models exist (2B4T). bitnet.cpp gives fast CPU kernels. Watch, do not depend.
- **SmoothQuant / OffQ** quantize activations (W8A8 / W4A4). They help compute-bound GPU serving, not our memory-bound case. OffQ's offset trick may help a 4-bit KV cache.
- **ZipServ / DFloat11** (C): lossless, ~28–30% off BF16. Useless on top of 4-bit codes (no exponent field to compress). Keep the idea "decode in registers inside the GEMM".

### 4.2 Sparsity and caching (levers 2–3)
- **colibri** (E) is the closest existing design to our goal:
  - The dense part stays in RAM (9.9 GB for GLM-5.2, which is too big for this 8 GB machine).
  - Each expert is one contiguous 19–21 MB block, loaded with one `pread`.
  - An 8-thread async reader pool loads misses while cached experts compute.
  - Each layer has an LRU cache, plus a set of hot experts pinned in RAM, learned from saved routing history.
  - A RAM-budget cap stops the engine before an out-of-memory kill.
  - Router lookahead predicts the next layer's experts with 71.6% recall.
  - Measured: 0.34 tok/s with 64 GB RAM and a 6.4 GB/s drive.
- **llama.cpp** (D) already has an **experimental GPU expert cache** (`--moe-cache-mib`: LRU plus frequency with a half-life of 128 tokens, and a trace dump). Weak points: it only reacts to misses, it has no prediction, it syncs the host first, and it needs a GPU. With mmap, the OS page cache is the only RAM-level expert cache, and the default repacking copies weights to RAM, which defeats streaming.
- **PowerInfer / LLM in a Flash / Deja Vu** (gap): neuron-level activation sparsity. A predictor selects which FFN rows to load. This works best on ReLU-type models. Modern SwiGLU models have weaker natural sparsity. Their row-column bundling and windowing ideas transfer to expert streaming.
- **AirLLM** (F): layer-by-layer streaming for dense models. The design is simple, but it has no overlap of copy and compute, and the code has bugs. It is good for batched prefill, too slow for chat decode (70B 4-bit: ~0.18 tok/s ceiling at 7 GB/s).
- **MemVanta** (E): its adaptive prefetch-depth controller is a good idea, but generation does not use it. **rai** (E): its constant-memory converter (26 MB of RAM to convert a 7B model) is the template we need to quantize big models on 8 GB of RAM.

### 4.3 More tokens per pass (lever 4)
- Speculative decoding helps dense models: the same weights verify k tokens.
- **For disk-bound MoE it can hurt:** colibri measured that MTP raised the experts read per token from 600 to 1,062 and lowered the speed from 0.34 to 0.31–0.33 tok/s. The draft tokens route to different experts, so the bytes grow almost linearly with k. The cμ-rule paper (C) gives the same lesson in general form: the fixed cost per weight pass is large, so we must amortize it, but only if the extra tokens reuse the same bytes.
- **Prompt KV cache on disk** (reame, MLX): when decode is ~1 tok/s, re-processing a long system prompt costs minutes. Saving the KV state to disk is a large, cheap win.

### 4.4 KV cache (lever 6)
- 70B GQA: 320 KiB per token in FP16. At 32K context this is 10 GiB, more than all of our RAM (C §3).
- KIVI/KVQuant (2-bit KV), PuzzleKV (0.19× with INT4 factors), MLA (DeepSeek: a latent KV cache that is about 1/10 the size by design). For this machine, a 2–4-bit KV cache plus a short context are required, not optional.
- llama.cpp allocates the full KV cache up front, cannot page it, and its KV types go no lower than ~4 bits (D weak point 8).

### 4.5 Security (must apply)
Outlier Injection (B): a full-precision model can look safe and become malicious after GPTQ, AWQ, NF4 or GGUF Q4_0 quantization (>90% success). (The earlier "Exploiting LLM Quantization" paper tests only LLM.int8, NF4 and FP4 on ≤7B models — see I.) Q4_K_M and IQ4_XS do not reliably stop it. **Rule:** download only from the original publisher and verify hashes. Before quantizing third-party weights, scan each group of 32 weights for one value that is far too large, and compare the full-precision and quantized behavior.

## 5. Gap papers downloaded (`papers\gaps\`, 47 files, titles verified)

- **G: offloading, streaming, MoE (biggest gap):** FlexGen, LLM in a Flash, PowerInfer, PowerInfer-2, MoE Offloading (Eliseev & Mazur), Fiddler, MoE-Lightning, Deja Vu, HOBBIT, ProMoE, DeepSpeed-Inference, QMoE
- **S: speculative decoding:** Leviathan et al., Medusa, EAGLE
- **Q: quantization:** LLM.int8(), QLoRA (NF4), QuIP, QuIP#, AQLM, QTIP, PV-Tuning, QuaRot, SpinQuant, HIGGS, VPTQ, Any-Precision LLM, Matryoshka Quantization, SINQ, ParetoQ, k-bit Scaling Laws, SqueezeLLM, bitnet.cpp, BitNet b1.58 2B4T, OBC
- **K: KV cache:** KIVI, KVQuant, StreamingLLM, PagedAttention, Palu, TurboQuant
- **A: architecture:** DeepSeek-V2 (MLA), DeepSeek-V3
- **L, X: other:** DFloat11, Exploiting LLM Quantization, Efficiently Scaling Transformer Inference, Sarathi-Serve

These are downloaded but **not yet analyzed in depth**. The G group is the most important to read next.

Remaining gaps (not on arXiv, or IDs not found): the llama.cpp k-quant / i-quant / imatrix PRs, VPTQ/HIGGS kernels, GLM-5.2 architecture report, MoE quantization studies (MoQE, MxMoE), Windows-specific direct I/O (`FILE_FLAG_NO_BUFFERING`) behavior.

## 6. Proposed direction (for discussion, nothing built yet)

**Goal statement:** run the largest MoE that fits on disk, at a usable speed, on 8 GB RAM + 4 GB VRAM + NVMe. Prove the design here, then scale it to bigger machines and 700B models.

1. **Baseline first.** Measure llama.cpp on this machine with three models: an 8B dense model, a 30B-A3B MoE and a ~120B MoE. Use mmap, `--no-repack`, `--n-cpu-moe` and `--moe-cache-mib`. Record tok/s, page faults, SSD read GB/s and the cache hit rate (`LLAMA_MOE_CACHE_TRACE`). Every later idea is judged against these numbers.
2. **Offline simulator.** Replay expert-routing traces to compare cache policies (LRU, frequency, learned pins, router lookahead) before we write engine code.
3. **Engine work (likely as a llama.cpp fork, extension points in D §8):**
   - an expert-contiguous layout with direct reads into a RAM slot pool (colibri)
   - an async reader pool and router-lookahead prefetch
   - a learned hot set
   - mixed precision per expert: hot experts at high bits, cold experts at low bits (HOBBIT / Drop-by-Drop)
   - a prompt KV cache on disk
4. **Quantization pipeline** that runs in constant memory (the rai converter design), with the outlier-injection scan built in.

## 6b. Update after the gap analyses (G, H, I, J, K)

| File | Scope |
|---|---|
| `G_offloading_moe_papers.md` | FlexGen, LLM in a Flash, PowerInfer 1/2, MoE offloading, Fiddler, MoE-Lightning, Deja Vu, HOBBIT, ProMoE, DeepSpeed-Inference, QMoE |
| `H_quant_gap_papers.md` | LLM.int8, NF4, QuIP/QuIP#, AQLM, QTIP, PV-Tuning, QuaRot, SpinQuant, HIGGS, VPTQ, Any-Precision, MatQuant, SINQ, ParetoQ, scaling laws, SqueezeLLM, bitnet.cpp, OBC |
| `I_kv_spec_arch_gap_papers.md` | KIVI, KVQuant, StreamingLLM, PagedAttention, Palu, TurboQuant, speculative decoding ×3, DeepSeek-V2/V3, DFloat11, Pope et al., Sarathi-Serve |
| `J_diffusion_image_video_papers.md` | DiT, SD3, HunyuanVideo, Wan, SVDQuant, Q-Diffusion, PTQ4DiT, ViDiT-Q, MixDQ, BitsFusion, DeepCache, TeaCache, FORA, Sparse VideoGen, SageAttention 1/2, ToMe, LCM, ADD |
| `K_diffusion_repos.md` | ComfyUI, ComfyUI-GGUF, stable-diffusion.cpp, nunchaku, Wan2GP, mmgp |

**Text LLM (MoE streaming): corrections to §4 and §6**
1. **Bytes beat cleverness.** Prediction and cache policy give only 5–30% when SSD-bound (HOBBIT 1.01–1.05× decode, ProMoE 1.09× over LRU, policy choice 2–9% of miss cost). Modern MoE routing is spread out. The large wins are: a contiguous block per expert, async direct I/O into slot pools, and fewer bits on a miss (G).
2. **Compute a missed expert on the CPU, do not upload it to the GPU** during decode. RAM (~20 GB/s) is faster than PCIe (~11 GB/s) on this laptop. Upload only for prefill (Fiddler rule, G).
3. **Expert format.** The best accuracy per byte at 2–3 bpw is QTIP: Llama-2-70B at 2.0 bpw scores PPL 3.70 against FP16 3.12, and it needs no fine-tuning (H). It has no CPU kernel, so an AVX2 and GTX 1650 QTIP kernel is a real contribution we can make. Interim: llama.cpp IQ2/IQ3 + SINQ (no calibration data needed; tested on Qwen3-235B and DeepSeek MoE).
4. **Multi-precision file.** Store each expert plane by plane, so that a low-bit read is a prefix read of the same file. Hot experts read more planes, cold experts fewer. Any-Precision LLM measured this layout; Drop-by-Drop and MatQuant support it (H).
5. **Speculative decoding barely helps a streamed MoE.** Two tokens route to 15.75 different experts, against 8 for one (I). EAGLE gets only 1.5× on Mixtral. Measure expert reuse between consecutive tokens (ρ) from our traces before building it.
6. **KV cache.** 4-bit KV is almost free (KVQuant ≤0.02 PPL), 3-bit costs <0.1 PPL, and 2-bit fails long-range retrieval. MLA models avoid most of the problem.
7. **Test model for MLA + MoE code:** DeepSeek-V2-Lite (15.7B total, 2.4B active, ~7.9 GB at 4 bits). DeepSeek-V3 does not fit: its always-on part alone is 8.1 GB at 4 bits.

**Image and video (diffusion): a different regime**
1. **Compute-bound, not bandwidth-bound.** Each step pushes 4k (1024² image) to 33k–119k (5 s video) tokens through every weight. Streaming the weights from SSD at every step costs only ~7% (Flux 1024²) to ~0.2% (Wan-14B 480p) (J). RAM does not need to hold the model.
2. **Speed comes from fewer FLOPs:** step and CFG distillation (5–25×), sparse video attention (1.5–2.3×), INT8 attention and linear layers on `dp4a` (1.3–2×), TeaCache (1.5–4×, conflicts with distillation). 4-bit weights save disk and RAM but do not speed up compute.
3. **The GTX 1650 has no tensor cores.** Nunchaku/SVDQuant W4A4, SageAttention 1/2 and Sparse VideoGen kernels do not run as written. Their recipes can be ported to `dp4a`/`HFMA2`.
4. **Attention and activations dominate video.** Attention is 52–87% of FLOPs, and at 720p the activations, not the weights, exceed VRAM. We need token-chunked linear layers, INT8 K/V and VAE tiling.
5. **Base: stable-diffusion.cpp.** It is ggml-based like llama.cpp and supports Flux, Wan 2.x and LTX-2. It already does block-by-block offload with prefetch and VAE tiling. Add from mmgp a small pinned ring of ~3 blocks for staging, and from ComfyUI the idea that VRAM is a cache of the model file.
6. **Estimates for this laptop [est]:** Flux-schnell 1024² ≈ 75 s. Wan-1.3B 480p with a 6-step distilled model ≈ 2–7 min. Wan-14B 480p ≈ 12+ min. 720p video with 50 steps: impractical.

**One platform for text, image and video**
Shared: one weight store with GGUF-style formats, one block/expert streamer (direct I/O, slot pools, pinned staging ring), one quant library, the text-encoder path (it is an LLM), one ggml revision.
Different: the scheduler and kernels. LLM decode is bandwidth-bound: minimize bytes and compute on the CPU on a miss. Diffusion is compute-bound: stream everything to the GPU and minimize FLOPs.

## 6c. Measured on this laptop (2026-11-12, details in `bench\results\RESULTS.md`)

| Test | Result |
|---|---|
| Dense 8B Q4_K_M, pure CPU | 2.9–3.6 tok/s (≈17.8 GB/s, RAM-bound, as predicted) |
| Dense 8B, 28/36 layers on GPU | **12.7 tok/s** (best) |
| Dense 8B, all layers on GPU (VRAM over-subscribed) | 7.0 tok/s: WDDM spills to shared memory |
| MoE 30B-A3B (17.3 GiB, 2.3× RAM), pure CPU, mmap | **2.96 tok/s** decode |
| same, GPU attention + CPU experts | 2.44 tok/s decode, 14.6 tok/s prefill |
| same + llama.cpp GPU expert cache 1500 MiB | **1.22 tok/s**: 2.3× slower |
| SSD read rate during MoE decode | mean 0.3–0.8 GB/s, against ~5–6 GB/s possible |
| Expert reuse between consecutive tokens (ρ) | **0.458**, 7× above random (0.062) |
| Cache simulation, 2–4 GiB | LRU 0.59–0.78, LFU 0.63–0.81, optimal 0.74–0.88 hit rate |

What the data changes:
- §3 was too pessimistic for MoE. A model 2.3× larger than RAM already runs at ~3 tok/s with stock llama.cpp.
- The bottleneck is **page-fault latency, not SSD bandwidth**: the SSD runs at ~10% of its capability. The first engine prototype must replace mmap paging with async direct I/O of whole experts.
- **Never upload missed experts to the GPU during decode.** This is now measured, not only taken from the literature.
- Routing locality (ρ = 0.46) is strong, so the cache policy is worth 10–15 points (LFU or learned policies beat LRU). Speculative decoding is less hopeless than the independent-routing estimate.

## 6d. Measured: image and video on this laptop (P0, 2026-11-13, details in `bench/diffusion/RESULTS.md`)

| Device profile | SD-Turbo 512² | FLUX-schnell | Wan 1.3B video |
|---|---|---|---|
| GPU (GTX 1650 4 GB + 8 GB) | 12.8 s | 2.0 min (512²) · 6.1 min (1024²) | 31 min for 1 s of 480p |
| No GPU, 8 GB | 67.5 s | 4.4 min (512²) | – |
| No GPU, 4 GB | 80.7 s · **10.1 s with TAESD** | **fails** (weights copied into RAM) | 2.9 min for a tiny 5-frame clip |

Corrections to §6b:
- The J/K estimates were 4–5× too optimistic on the GPU: real kernels reach 1.2–1.7 TFLOPS on the GTX 1650, not 4.
- The VAE decoder dominates CPU time and is the first fix for weak devices (TAESD is 34× faster).
- Streaming is confirmed nearly free on the GPU (FLUX is compute-bound with weights streamed every step). On CPU the baseline copies weights into RAM, which is exactly what rope must avoid.

## 7. Open questions for you

1. **Hardware:** is this laptop the target, or only the development box? The 82 GB of free disk space and the 8 GB single-channel RAM limit everything. A second RAM module and a second NVMe drive would change the numbers a lot.
2. **Target model:** a specific model (for example a DeepSeek / GLM / Qwen MoE), or "the largest that works"?
3. **Target speed:** what is "usable" for you: 1 tok/s, 5 tok/s or 10 tok/s? Is the main use chat, or batch jobs?
4. **Output:** a research report, a llama.cpp fork, or a new engine?
