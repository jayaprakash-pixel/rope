# H — Quantization gap papers (Q01–Q20): which format to stream from SSD

Date: 2026-11-12. Scope: the 20 "Q" gap papers in `papers\gaps\`. Analyses A and B already cover GPTQ, AWQ, SmoothQuant, BitNet b1.58 (2402.17764), GGUF k/i-quants, Drop-by-Drop, OffQ and WINDQuant. This file does not repeat them. It uses their facts only for comparison.

Method: each PDF was converted with `pdftotext -layout`. Method and experiment sections were read in full. Appendices were skimmed for cost, kernels and large-model tables.

Conventions:
- **bpw** = bits per weight of the quantized linear layers. Most papers do not count embeddings and `lm_head` (kept in FP16). I say so where it matters.
- **"true bpw"** = index bits + codebook/scale/sign-vector overhead, as the paper states or as I derive from the paper's own formula.
- **[est]** = my estimate, not a paper number.
- **"not stated"** = the paper does not give it.
- **Parse note**: `pdftotext` scrambles many multi-column tables (row labels and numbers shift). Where I had to re-align a table, I checked the re-alignment against a second table or the text. I mark the remaining doubtful cases.
- Llama-2-70B WikiText-2 FP16 perplexity = **3.12 at context 4096** and **3.32 at context 2048**. Numbers at different context lengths are not comparable.

Our machine (from `00_SYNTHESIS.md`): Ryzen 5 5600H (AVX2, no AVX-512; Zen 3: 32 KiB L1D and 512 KiB L2 per core, 16 MiB L3 shared [ext]), 8 GB single-channel DDR4 (~20 GB/s), GTX 1650 4 GB (Turing, ~128 GB/s, no BF16, no tensor cores on TU117 [ext]), NVMe ~5–6.5 GB/s. Main design: large MoE, experts streamed from SSD. **SSD bytes per token = streamed expert parameters × bpw / 8.** Per 1 B streamed expert parameters: 4.9 bpw (Q4_K_M) = 0.61 GB = 0.12 s at 5 GB/s; 3.25 bpw = 0.41 GB; 2.06 bpw = 0.26 GB = 0.05 s [est].

---

## Part 1 — Paper by paper

### Q01 — LLM.int8() (Dettmers et al., arXiv 2208.07339)

**Mechanism.** W8A8 matrix multiplication for large transformers.
- Vector-wise quantization: one absmax scale per row of the activation X and one per column of W. The int8 product is rescaled by the outer product of the two scale vectors.
- Mixed-precision decomposition: any hidden dimension where an activation magnitude exceeds the threshold α = 6.0 is an "outlier feature". Those columns of X and the matching rows of W are multiplied in FP16. All other values are multiplied in int8. Outliers are about 0.1% of feature dimensions.
- Key finding: outlier features emerge in all layers at about 6.7B parameters. Setting them to zero destroys attention softmax mass and perplexity. Removing the same number of random features changes almost nothing.

**True bpw.** ~8 bpw plus the FP16 outlier slice. Memory reduction measured on BLOOM-176B: 1.96× vs 16-bit.

**Accuracy.** Only OPT/BLOOM/GPT-style models up to 175B. "No performance degradation" up to 175B (zero-shot and perplexity). No Llama numbers. No 2/3/4-bit results.

**Decode cost and kernel.** Appendix D: int8 cuBLASLt matmul starts to be ~2× faster than FP16 only at large hidden sizes. With quantize/dequantize overhead the speedup drops to 1.6× at hidden size 5140. Models with hidden size ≤ 2560 are slowed down. With the mixed-precision decomposition, only the 13B and 175B models get speedups. CPU kernel: not stated.

**Quantization cost.** Zero-shot (no calibration). Conversion is a single pass over the weights.

**Relevance to MoE streaming.** Low. 8 bpw is 2–4× too many SSD bytes. Two ideas transfer: (1) outlier *features* exist from ~6.7B upward, so any low-bit scheme must handle them (Hadamard rotation is the modern answer, Q04/Q06/Q08); (2) a small "keep in higher precision" slice is cheap in storage but costly in kernel complexity.

### Q02 — QLoRA / NF4 (Dettmers et al., arXiv 2305.14314)

**Mechanism.** NormalFloat4 (NF4): a 4-bit data type whose 16 levels are quantiles of N(0,1), normalized to [−1, 1], with an exact zero. Built from 2^(k−1) quantiles on the negative side and 2^(k−1)+1 on the positive side, then merged so that 0 appears once. Block-wise absmax scaling with block size 64. Double Quantization (DQ): the FP32 block constants are themselves quantized to FP8 with block size 256.

**True bpw.** Without DQ: 4 + 32/64 = 4.5 bpw. With DQ: 4 + 8/64 + 32/(64·256) = **4.127 bpw** (paper: saves 0.373 bpw, about 3 GB on a 65B model).

**Accuracy at 70B.** The QLoRA paper reports fine-tuning results, not PTQ perplexity tables for Llama-2-70B. Independent data from HIGGS (Q10, Tab. 10, Llama-3.1-70B, FP16 2.541): NF at 4.25 bpw (block 64) = **3.065**; NF at 4.02 bpw (block 1024) = **2084** (collapse); NF at 3.25 bpw = 41.76. So NF4 needs a small block on Llama-3-70B.

**Decode cost and kernel.** bitsandbytes NF4 kernel on RTX 4090, Llama-3.1-8B batch 1: **31 tok/s vs FP16 57 tok/s** (HIGGS Tab. 1). NF4 is *slower* than FP16 at batch 1 in that measurement. CPU kernel: not stated. Feasible: a 16-entry FP16 table fits one AVX2 register pair; `vpshufb`-style lookup works [est].

**Quantization cost.** Data-free, seconds per layer. Runs layer by layer on 8 GB RAM trivially.

**Relevance.** NF4 is a 4-bit format; at 4 bpw llama.cpp Q4_K_M (4.89 bpw file, A §2.2) and HIGGS/SINQ are at least as good. The transferable idea is DQ: quantize the scales themselves to make group overhead cheap. QLoRA's paged optimizer is irrelevant for inference.

### Q03 — QuIP (Chee et al., arXiv 2307.13304)

**Mechanism.** Two parts.
1. **LDLQ adaptive rounding.** Minimize the proxy loss ℓ(Ŵ) = tr((Ŵ−W) H (Ŵ−W)ᵀ), H = E[xxᵀ]. Round columns one at a time with linear feedback: Ŵ_k = Q(W_k + (W_{1:k−1} − Ŵ_{1:k−1}) a_k). With U from the LDL decomposition H = (U+I) D (U+I)ᵀ, LDLQ is worst- and average-case optimal in this class (Thm 1). The paper proves **OPTQ/GPTQ is exactly LDLQ** (Thm 6), with fewer operations (one Cholesky, no inverse).
2. **Incoherence processing.** Conjugate W and H with random orthogonal matrices: W̃ = U W Vᵀ, H̃ = V H Vᵀ. The proxy loss does not change. To keep inference cheap, U and V are Kronecker products of two small random orthogonal matrices (cost O(n(p+q)) per vector). Extra heuristics: diagonal rescaling D̃, range from ‖W‖_F instead of max|W|, greedy coordinate-descent passes.

**True bpw.** b bits per weight plus one scale per matrix. Random matrices are regenerated from a seed (no storage).

**Accuracy, Llama-2-70B** (context 2048, FP16 3.319, QuIP paper table): 4-bit **3.531**, 3-bit **3.853**, 2-bit **6.326** WikiText-2; C4 5.869 / 6.135 / 8.937. Zero-shot ArcE/PiQA/StoryCloze at 2-bit: 54.38 / 75.08 / 75.37 (FP16 59.72 / 80.90 / 79.95). OPTQ at 2-bit: 123.9 (collapse).

**Decode cost.** Kronecker multiplies add O(n√n). QuIP# reports that QuIP# is about 2× faster than QuIP on an A6000 at the same bitrate. CPU kernel: not stated.

**Quantization cost.** Single GPU with up to 48 GB, one transformer block at a time (OPTQ-style), 128 C4 segments of 2048 tokens. Time not stated in the sections read.

**Relevance.** Historical. QuIP# and QTIP supersede it. The two lasting ideas are (1) GPTQ = LDLQ, and (2) incoherence (random rotation) turns outlier-heavy weights into near-Gaussian weights, which is what makes 2-bit possible.

### Q04 — QuIP# (Tseng et al., arXiv 2402.04396)

**Mechanism.** Three parts on top of QuIP.
1. **Randomized Hadamard Transform (RHT)** instead of Kronecker orthogonal matrices: Ŵ = Had(diag(S_U) · Had(diag(S_V) Wᵀ)ᵀ), with random sign vectors S_U ∈ {±1}^m, S_V ∈ {±1}^n. Cost O(n log n) with the fast Walsh–Hadamard transform, no floating multiplies. Better incoherence bound (log instead of log²). If n is not a power of two, n = p·q with a known Hadamard matrix of size q (e.g. Llama-2-70B 28672 = 1024·28). A random-FFT variant handles any even n (2-bit 70B: 4.17 vs 4.16).
2. **BlockLDLQ + E8P lattice codebook.** BlockLDLQ is LDLQ over g-column blocks with a block LDL decomposition, so a vector quantizer can be used. **E8P** ("E8 padded") is a 2-bit, 8-dimensional codebook with 2^16 entries built from the E8 lattice (shifted by ¼). A 16-bit code = 8 bits index into a 256-entry table of absolute-value vectors + 7 sign bits + 1 bit for the ±¼ shift; the 8th sign follows from parity. The table is **1 KiB**. Decoding needs "very few (< 5) instructions per weight".
3. **Higher bitrates by residual VQ (RVQ).** 3-bit = E8P (2-bit) + a 1-bit E8 codebook; 4-bit = E8P + E8P on the residual. Fixed scales per stage (F.5): ρ = 0.9 for 2-bit; 3-bit stages 0.98 and 2.04; 4-bit stages 1.03 and 3.45 (times ‖W‖_F).
4. **Fine-tuning (optional).** Within each transformer block, tune the not-yet-quantized layers to compensate before quantizing the next one; then tune all unquantized parameters (norms, sign vectors as real numbers) end-to-end. Data: 256 RedPajama sequences, 5 epochs.

**True bpw.** F.1: E8P 1 KiB shared over all layers (< 0.01 bpw). Sign vectors: (n+m) bits without FT, 16(n+m) bits with FT (FP16) per matrix, < 0.01 bpw for 4096×4096 and larger. So **2.00–2.01 / 3.0 / 4.0** bpw.

**Accuracy, Llama-2-70B** (Table 4, context 4096, FP16 3.12 / C4 4.97):

| bits | QuIP# | no FT | no FT, no E8 | AQLM |
|---|---|---|---|---|
| 4 | **3.18** / 5.02 | 3.18 | 3.22 | 4.07 bpw: 3.17 |
| 3 | **3.35** / 5.15 | 3.41 | 3.48 | 3.01 bpw: 3.36 |
| 2 | **3.91** / 5.71 | 4.16 | 4.58 | 2.07 bpw: 3.94 |

Zero-shot Llama-2-70B (Table 3, acc): FP16 ArcC/ArcE/PiQA/Wino = 51.1/77.7/81.1/77.0; QuIP# 4-bit 50.6/78.1/81.4/77.1; 3-bit 50.9/77.7/81.4/76.4; 2-bit 48.7/77.3/80.3/75.9.

MoE: Table 9 (2-bit, **no fine-tuning**): Mixtral-8x7B WikiText-2 3.45 → 4.69, C4 6.85 → 8.25; zero-shot ArcC 0.56 → 0.49, ArcE 0.74 → 0.68, Wino 0.75 → 0.73. Falcon-180B 3.30 → 4.18 (parse note: the table rows are interleaved; I assign the first pair to Mixtral and the second to Falcon by the FP16 values).

**Decode cost and kernel.** "Proof of concept" CUDA kernel, RTX 4090 (Table 5, FlashAttention Llama): 2-bit 2-70B **32.74 tok/s** (56.8% of peak bandwidth); 2-7B 170.5 tok/s. HuggingFace implementation (Table 6): 2-7B 106.3 tok/s vs FP16 33.1 (3.2×); 2-70B 25.9 tok/s vs AQLM 2-bit 8.27. The paper states that AQLM's 1 MiB codebook does not fit L1 and is "slower than FP16". CPU kernel: none in the paper. **Feasibility on AVX2: high.** The 1 KiB table fits L1 (32 KiB) with room to spare. Decode = byte index lookup + sign mask XOR + parity + shift. llama.cpp IQ2_XXS already uses an E8-lattice-based 256-entry grid with 7 sign bits per 8 weights (A, §2.3), so a CPU path for this code family already exists. The missing CPU part is the RHT on activations: one FWHT of length n per matrix per token, O(n log n), cheap [est].

**GTX 1650.** 1 KiB × 32 (bank-conflict duplication) = 32 KiB of shared memory; fits in Turing's 64 KiB shared limit [ext]. No tensor cores are needed for GEMV.

**Quantization cost.** F.7 (A100): Llama-2-70B **< 10 GPU-hours without FT, ~100 GPU-hours with FT**, not counting Hessians. Hessians: 6144 sequences × 4096 tokens of RedPajama (25 M tokens) through the model. On our machine: BlockLDLQ itself needs one n×n FP32 Hessian per layer input (8192² × 4 B = 256 MiB) and one weight matrix; that fits in 8 GB. The Hessian collection pass at 25 M tokens is the blocker: ~2·70e9·25e6 = 3.5e18 FLOP, days on a GTX 1650 even at peak [est]. With 128×2048 tokens (OPTQ default) it becomes hours [est], at some accuracy cost (AQLM Tab. 8 shows PPL still improves from 128 to 4096 samples).

**Relevance to MoE streaming.** High. (1) No per-matrix codebook: the code is fixed, so a small expert matrix pays no codebook overhead. (2) RVQ 4-bit = two independent 2-bit stages → a natural nested format (Part 3). (3) Works on Mixtral without FT. (4) A CPU decode path is close to llama.cpp IQ2_XXS.

### Q05 — AQLM (Egiazarian et al., arXiv 2401.06118)

**Mechanism.** Additive (multi-codebook) quantization. Each row is split into groups of g consecutive weights. Each group = sum of M codewords, one from each of M learned codebooks C_m ∈ R^{g×2^B}, times a per-output-row scale s_i.
- Objective: argmin ‖WX − ŴX‖², written with the precomputed XXᵀ (size d_in²), so the data set size does not matter.
- Phase 1: codes by **beam search** over a fully connected MRF (unary and pairwise terms via XXᵀ).
- Phase 2: codebooks and scales by Adam (100 steps, lr 1e-4) on the same objective.
- Init by residual k-means. Alternate until the loss stops improving.
- Phase 3: block fine-tuning (codebooks, scales, RMSNorm; codes frozen) to match the block's FP16 outputs.
- Configs (App. C): 2-bit = 1 codebook of 2^15 or 2^16, g = 8 ("1×16"); 3-bit = 2 codebooks of 2^12, g = 8; 4-bit = 2 codebooks of 2^15/2^16, g = 8. Fast variant "2×8": two 256-entry codebooks, g = 8.

**True bpw.** App. H, Eq. 10: b̄ = (16·g·M·2^B + d_out·(d_in/g)·B·M + 16·d_out) / (d_out·d_in). Reported averages include codebooks and scales: 2.02–2.29 for "2-bit", 3.01–3.04, 3.94–4.14. Example: gate_proj of Llama-2-70B, 2×8, g = 8 → 2.002 bpw. **MoE warning [est]:** a 1×16 codebook costs 1 MiB per matrix. For a small expert matrix (e.g. 2048×7168 = 14.7 M weights) that is +0.57 bpw. AQLM-style codebooks must be shared across the experts of a layer, which no paper tests.

**Accuracy, Llama-2-70B** (context 4096, FP16 3.12; parse note: rows re-aligned and checked against QuIP# Table 4):
- 2.07 bpw: **3.94** / C4 5.72, avg zero-shot (5 tasks) 68.75 vs FP16 70.17 (Tab. 1). With QuIP#-style end-to-end FT: **3.83** / 5.62, avg 68.20 (Tab. 4).
- 3.01 bpw: **3.36** / 5.17, avg 69.86 (Tab. 2). GPTQ 3.00: 4.40; SpQR 2.98: 3.85; QuIP 3.01: 3.87.
- 4.14 bpw: **3.19** / 5.03, avg 69.93 (Tab. 10). GPTQ 4.00: 3.35; SpQR 3.97: 3.25; QuIP# 4.01: 3.22; AQLM 3.82: 3.21.
- Mixtral-8x7B: 1.98 bpw 4.61 vs FP16 3.46 (avg 67.68 vs 72.33); 3.02 bpw 3.79; 3.915 bpw 3.57 (Tab. 3, 11).
- Pareto claim: 2.5 bpw is the Pareto-optimal width for Llama-2 with AQLM; PV-Tuning (Q07) later moves this to 2.0.

**Decode cost and kernel.** The only paper in this set with **measured CPU numbers on a 70B model**.
- GPU (Tab. 14, RTX 3090, batch 1, 128 tokens): 70B FP16 5.8 tok/s; 1×16 6.7 tok/s; **2×8 14.3 tok/s**. GEMV kernel (Tab. 5): 1×16 1.2× FP16, 2×8 3.05× on 70B gate_proj.
- **CPU (Intel i9-13900K, 8 cores, FP32 baseline): 70B FP32 0.297 tok/s → 2×8 0.966, 4×8 0.948, 8×8 0.775 tok/s** (Tab. 14). Kernel speedups 3.7–4.1× on 70B layers (Tab. 5). Method: with small codebooks, precompute the dot products of each input sub-vector with all codewords, then each group costs M table reads and adds.
- Cache: 1×16 codebook = 1 MiB per layer (does not fit any GPU L1). 2×8 codebooks = 2 × 256 × 8 × 2 B = **8 KiB** per layer → fits a 32 KiB CPU L1D [est].
- Accuracy cost of 2×8: Llama-2-7B 2-bit 2×8 g8 WikiText-2 6.57 (Tab. 12) vs 1×16 6.64 at 2.02 bpw and 6.29 at 2.29 bpw. 70B 2×8: not stated.

**Quantization cost.** App. D: **70B: 10–14 days on one GPU, 3–4 days on 8 GPUs** (A100/H100). End-to-end FT: 70B 1–2 days on 8 A100. Calibration: RedPajama, 4096-token sequences; PPL improves up to 4096 samples (Tab. 8). Not feasible on our machine.

**Relevance.** Medium. The 2×8 CPU kernel is a proven AVX-class design (precompute input·codeword tables). But the quantizer cost is prohibitive, and per-matrix codebooks are bad for small expert matrices. Drop-by-Drop (B) is AQLM with nested codebooks.

### Q06 — QTIP (Tseng et al., arXiv 2406.11235)

**Mechanism.** Replace 8-dimensional VQ with **trellis-coded quantization (TCQ)** of very long sequences (256 weights per sequence).
- A (L, k, V) trellis has 2^L states. Each step emits V weights and consumes kV bits. Storage: k bits per weight. The best walk is found with the **Viterbi** algorithm, cost O(2^L·T), linear in sequence length T.
- **Bitshift trellis** (no stored graph): state j follows state i iff the top L−kV bits of j equal the bottom L−kV bits of i. The t-th group of V weights depends only on a window of L bits starting at bit (t−1)kV of the stream. Decoding is one shift and mask per group, and **groups decode in parallel**.
- **Computed Gaussian codes** (no stored codebook). The L-bit window is hashed into a pseudo-random approximately Gaussian value:
  - *1MAD*: x ← (a·x + b) mod 2³² (LCG, a = 34038481, b = 76625530); sum the four bytes of x; out = (sum − 510)/147.8. Only 2^10 distinct values, which "does not empirically affect quality".
  - *3INST*: LCG (a = 89226354, b = 64248484), then XOR the low and high 16-bit halves with the bits of a magic FP16 m = 0.922, and add the two FP16 numbers.
  - *HYB* (V = 2): x ← x² + x; take bits (15−Q)…14 as an index into a 2^Q × 2 FP16 table; flip the sign of the second entry with bit 15. Q = 9 → **2 KiB** table, tunable by fine-tuning.
- **Tail-biting** (Alg. 4): rotate the sequence by T/2, run Viterbi, take the overlap of the start and end states, run Viterbi again with that overlap fixed. This removes the L−kV start-state bits, so storage is exactly k·T bits per sequence. Near-optimal MSE (Tab. 2).
- Used inside **BlockLDLQ** (QuIP#): a T_x × T_y = 16 × 16 tile of weights is one 256-long trellis sequence. Incoherence by RHT as in QuIP#.
- Distortion on i.i.d. Gaussian at 2 bits (Tab. 1): Lloyd-Max scalar 0.118, QuIP# E8P 0.089, QTIP 1MAD/3INST 0.069, HYB 0.069–0.071; rate-distortion bound 0.063.

**True bpw.** Exactly k bits per weight (tail-biting) + RHT sign vectors (< 0.01) + HYB table 2 KiB per layer (negligible). **2.0 / 3.0 / 4.0 bpw.**

**Accuracy, Llama-2-70B** (context 4096, FP16 3.12 / C4 4.97):

| bits | QTIP HYB, FT (Tab. 5) | QTIP 1MAD, no FT (Tab. 3) | QTIP 3INST, no FT | QuIP# FT | AQLM |
|---|---|---|---|---|---|
| 4 | **3.16** | 3.16 | 3.16 | 3.18 | 3.19 |
| 3 | **3.26** | 3.27 | 3.27 | 3.35 | 3.36 |
| 2 | **3.70** / C4 5.48 | 3.87 / 5.70 | 3.90 / 5.69 | 3.91 | 3.83 (e2e FT) |

(Tab. 5 parse note: rows re-aligned as FP16, AQLM, QuIP#, QTIP per bit-width; the 3- and 4-bit QTIP values match Tab. 3 as the text says.)

Key point: **QTIP without any fine-tuning (pure computed code) beats QuIP# with fine-tuning** at 2-bit (3.87 vs 3.91) and at 3-bit (3.27 vs 3.35).

Zero-shot Llama-2-70B (Tab. 6, ArcC/ArcE/PiQA/Wino): 4-bit 50.0/77.8/81.3/76.9; 3-bit 50.3/78.2/80.6/77.0; 2-bit 48.0/76.3/80.2/75.1 (FP16 51.1/77.7/81.1/77.0).

Llama-3-70B (Tab. 7, context 8192, BF16 2.59): QTIP 4-bit **2.75**, 3-bit **3.18**, 2-bit **4.97** (QuIP# 2.99 / 3.59 / 5.77). The 70B zero-shot numbers collapse (e.g. ArcC 28.0 at 2-bit) because **layer 0 v_proj** is quantized; the paper says to keep it unquantized, add a "\n" prefix, or use the chat template. Llama-3.1-70B-Instruct (Tab. 8): BF16 3.52; QTIP 4-bit 3.73, 3-bit 4.12, 2-bit 5.08; PV-Tuning 2.01 bpw 5.70. Llama-3.1-405B-Instruct: FP8 1.70; QTIP 4/3/2-bit 1.79 / 2.05 / 3.29.

**Decode cost and kernel.** Batch 1, matrix fusion (Tab. 4, RTX 6000 Ada, 960 GB/s): 2-7B FP16 55.9, AQLM 81.5, QuIP# 186, **QTIP 2-bit 188, 3-bit 161, 4-bit 140 tok/s**; 2-70B AQLM 8.78, QuIP# 22.2, **QTIP 2-bit 23.5, 3-bit 19.1, 4-bit 16.3 tok/s** (FP16 OOM). Tab. 17 adds RTX 3090: 2-70B 2-bit 15.3 tok/s. "Over 80% of peak memory bandwidth" for matrix-vector products. So the 32× larger effective dimension costs no speed vs QuIP#.

CPU kernel: none. The paper notes that on ARMv8, `vqtbl4q_u8` (16 lookups into a 64-entry table) supports HYB with Q = 6, V = 1, and that this variant gives 2-bit Llama-2-7B WikiText-2 6.89 without FT, "essentially the same" as 3INST. **Feasibility on AVX2 [est]: high.**
- Bit-window extraction: AVX2 has per-lane variable shifts (`vpsrlvd`), so 8 windows per instruction.
- 1MAD: `vpmulld` + add (LCG), byte sum (`vpmaddubsw` with a ones vector, then `vpmaddwd`), convert, FMA. Roughly 6–8 vector instructions per 8 weights before the dot product.
- HYB with a 64-entry table: four `vpshufb` + blends per 32 byte indices; a 2 KiB table also fits L1 for gathers.
- Compute budget: from SSD at 5 GB/s and 2 bpw the CPU must decode 20 G weights/s. 6 cores × ~4 GHz × ~2.5 vector ops/cycle × 8 lanes ≈ 480 G lane-ops/s → ~24 lane-ops per weight available. Enough. From RAM at 20 GB/s and 2 bpw the need is 80 G weights/s → ~6 lane-ops per weight. Tight. Hot experts in RAM may become compute-bound on this CPU.

**GTX 1650.** Computed codes need no shared-memory table at all. HYB 2 KiB × 32 = 64 KiB = the Turing shared-memory limit (tight). The 16×16 tiles were chosen for tensor-core MMA; TU117 has no tensor cores, but a GEMV kernel does not need them. Turing (sm_75) support in the released kernels: not stated.

**Quantization cost.** Not stated (GPU hours not given). Hessians: 6144 × 2048 (Llama-2), 4096 × 8192 (Llama-3), RedPajama. Viterbi cost [est]: 2^L states × 2^{kV} edges × T = 2^16 × 4 × 256 ≈ 6.7e7 operations per 256 weights at 2-bit → ~1.8e16 for 70B weights. That is hours at GPU peak; on our GTX 1650 likely a day or more per 70B; on the CPU several times slower. Fine-tuning is optional (computed codes cannot be tuned; HYB tables can).

**Relevance.** **Highest accuracy per bit in this set at 2–3 bpw, no codebook overhead per expert, no fine-tuning needed, compute-only decode.** Drawbacks: no CPU kernel exists; quantizer cost is unknown but Viterbi-heavy; the trellis bit stream is **not nested** (dropping low bits does not give a valid lower-rate code), so it does not support "read fewer bits for cold experts" by itself.

### Q07 — PV-Tuning (Malinovskii et al., arXiv 2405.14852)

**Mechanism.** A fine-tuning algorithm, not a format. A quantized model = continuous parameters V (codebooks, scales) + discrete assignments P (codes). Minimize a global objective φ (KL to the FP16 model) by alternating:
- **P step**: fix codes, update codebooks/scales by backprop (Adam).
- **V step (linearized, subspace)**: compute y⁺ = y − (1/L)∇φ(y) (in practice one Adam step on the dequantized weights); choose the τ fraction of weights with the largest update (τ set so that ‖x_{k+1} − x_k‖/‖x_k‖ ≈ 0.01); re-encode only those weights to the nearest code (beam search for VQ). Updating only a small subspace makes the local smoothness constant L_S an order of magnitude smaller, so steps are large enough to change discrete codes, without straight-through estimation (STE).
- PV + STE together is slightly better still (Tab. 1).
- Representation used for the main results: single-codebook VQ, 14–16-bit codes for groups of 8 or 16 weights (16 bits / 8 weights ≈ 2 bpw; 14 bits / 16 weights ≤ 1 bpw).

**True bpw.** Same as the underlying AQLM/VQ format: 2.02–2.07 or 1.01–1.14 including codebooks.

**Accuracy, Llama-2-70B** (Tab. 2 and 7, FP16 3.12): **2.07 bpw: 3.78** / C4 5.56 (AQLM 2.07: 3.94, QuIP# 2.01: 3.91); **1.14 bpw: 5.52**, avg zero-shot 64.58 vs 72.40; 1.01 bpw: 6.09. Llama-3-70B (FP16 2.59): 2.07 bpw 4.57 / C4 6.56, avg 70.38 vs 75.37; 1.01 bpw 8.67. Llama-3.1-70B-Instruct (from QTIP Tab. 8): 2.01 bpw 5.70 vs QTIP 2-bit 5.08. 3- and 4-bit: not stated.

Ablation (Tab. 1, Llama-2-7B, AQLM 2.01 bpw): calibration only 7.38 → continuous-only FT 6.69 → STE 6.41 → PV 6.13 → PV+STE 5.90. For scalar GPTQ 2.14 bpw: 3290 → 16.77 → 8.79 → 8.49.

**Decode cost.** Unchanged format → reuses AQLM kernels. App. M: 1×16 kernel, RTX 3090, transformers + CUDA graphs: 70B 7.2 tok/s (+28% vs FP16). CPU: through AQLM's CPU kernel only if the 2×8 format is used (not tested with PV).

**Quantization cost.** 7B on one GPU; 70B needs 8× A100 or RAM offloading (sharding + optimizer offload). Longest run: 2 days on 8 GPUs to beat baselines, 8 days to converge. Up to 1.5× the cost of continuous-only FT. Not feasible locally.

**Relevance.** Medium. It shows that **fine-tuning the discrete codes** is worth ~0.15 PPL at 2-bit on 70B and makes 2-bit Pareto-optimal for Llama-2. If we ever rent GPUs to make a 2-bit expert file, PV-Tuning is the fine-tuning stage to use. It applies to any format, including GGUF-style scalar codes (GPTQ + PV row).

### Q08 — QuaRot (Ashkboos et al., arXiv 2404.00456)

**Mechanism.** End-to-end 4-bit (weights, activations, KV cache) by rotating the residual stream with randomized Hadamard matrices, using computational invariance (RMSNorm scale folded into the next linear layer). Rotations are fused into the weights where possible. Online Hadamard transforms remain: before down_proj, on attention values/heads (Kronecker H_heads ⊗ H_dh), and on queries/keys after RoPE for the KV cache. Weights by GPTQ (or RTN); activations quantized on the fly.

**True bpw.** W4 per-channel: 4 + 16/d_in. Group-128 variant: +0.125.

**Accuracy, Llama-2-70B** (context 2048, FP16 3.32). W4A4KV4 + GPTQ: **3.79**; group 128: 3.61; group 64: 3.58 (Tab. 1, 4). W4A4KV4 + RTN: 4.14 (Tab. 3). Zero-shot avg (6 tasks) 75.98 vs 77.07. **Weight-only** ablation (Tab. 7, per-column asymmetric): the text states that QuaRot + GPTQ reaches **5.6 at W2** on 70B; the parsed table suggests ~3.41 at W4 and ~3.72 at W3 (parse note: table heavily scrambled; treat W3/W4 as approximate). Llama-3-70B W4A4 (Tab. 11): 6.66; group 128: 5.51 (FP16 2.86).

**Decode cost.** Kernels target compute-bound prefill (up to 3.33× on 70B layer, RTX 3090, batch 64). Decode: peak memory saving 3.89× on a 70B layer. Batch-1 decode speed: not stated. CPU kernel: none.

**Quantization cost.** 70B: rotation 5 min + GPTQ 2 h on one A100.

**Relevance.** Low for weight-only streaming (activation quantization helps compute-bound serving, not our memory-bound decode). Two useful points: rotation by fused Hadamard costs **zero storage** and almost zero runtime, and a 4-bit KV cache with Hadamard is accurate — relevant for our KV budget (synthesis §4.4).

### Q09 — SpinQuant (Liu et al., arXiv 2405.16406)

**Mechanism.** Like QuaRot, but the residual rotation R1 (d×d) and per-head rotation R2 (d_head × d_head) are **learned** with Cayley SGD on the Stiefel manifold, minimizing the quantized network's task loss with frozen weights. R3, R4 are online Hadamards (KV cache, down_proj input). Random rotations show large variance between seeds (up to 13 points zero-shot); learned rotations remove it. Learning: 800 WikiText-2 samples, 100 iterations; 3.5 h for Llama-2-70B. Then GPTQ on rotated weights.

**True bpw.** 4 or 3 bits per channel; R1/R2 merge into the weights (no storage in the "no_had" variant).

**Accuracy, Llama-2-70B** (context 2048, FP16 3.3): W4A8: **3.5** (had) / 3.7 (no_had); avg zero-shot (8 tasks) 72.7 vs 72.9. W3A8 (App. Tab. 10): **3.8**, avg 71.7 (GPTQ 9.0). W4A4KV4: 3.8. Llama-3-70B W4A4KV4 + GPTQ: 5.5 (QuaRot 20.4). 2-bit: not stated.

**Decode cost.** CPU number on ARM: Llama-3-8B on MacBook M1 Pro, W4A8: 58.9 ms/token vs FP16 177.2 (3×); online Hadamard adds 8%. AVX2: not stated.

**Quantization cost.** Small (hours on one GPU type not stated in the sections read). Needs gradients through the whole quantized network → full backward pass; not practical on our machine for >30B [est].

**Relevance.** Low–medium. Learned rotation beats random Hadamard for scalar 3–4-bit. For weight-only VQ/TCQ, random Hadamard is already near-Gaussian, so the gain is smaller. Useful if we stay with scalar GGUF-like 3–4-bit codes and can afford a rotation-learning run.

### Q10 — HIGGS (Malinovskii et al., arXiv 2411.17525)

**Mechanism.** Two contributions.
1. **Linearity theorem.** Under local optimality and smoothness assumptions, E[PPL(Ŵ)] ≈ PPL(W*) + Σ_l α_l · t_l², with t_l² = E‖Ŵ_l − W_l‖²_F / ‖W_l‖²_F (relative MSE of layer l) and α_l a layer constant **independent of the quantizer**. Valid roughly for ≥ 3 bpw (Fig. 1, 3).
2. **HIGGS format** (Hadamard Incoherence + Gaussian MSE-optimal GridS). Per group of g = 1024 weights: s = ‖w‖₂, apply an RHT (seeded) to w/s, round each p-dimensional sub-vector to the nearest point of a grid G_n^p of n points that is MSE-optimal for N(0, I_p) (computed once by CLVQ, Pagès & Printems). Data-free (no calibration).
   - Configs: 3.25 bpw = (p=2, n=88) or (p=3, n=830); 4.02 = (p=1, n=16) or (p=2, n=256); 4.25 = (p=1, n=19) or (p=2, n=361) (App. H).
   - Kernel-friendly ("FLUTE grids"): p ∈ {1, 2}, b ∈ {2, 3, 4}. Uniform-constrained grid "CH8" for 8 bits.
   - Constraint stated by the paper: 128 KiB shared memory and 32× replication → the grid can have at most ~2^10 points.
3. **Dynamic HIGGS.** Estimate α_l per layer by inserting Gaussian noise at J = 15 levels and measuring PPL (or KL on 287k random tokens = data-free). Then choose one quantizer per layer to minimize Σ α_l t²_{l,j} subject to Σ b_j d_l ≤ b_max·d — a knapsack solved exactly with OR-Tools CP-SAT "in seconds".
4. GPTQ + HIGGS: replace round-to-nearest with GPTQ rounding onto the HIGGS grid.

**True bpw.** log2(n)/p + 16/1024 (FP16 scale per 1024 weights). 4.02 = 4 + 0.016. 3.25 = log2(88)/2 + 0.016.

**Accuracy.** Llama-3.1-70B (Tab. 10, FP16 2.541, MMLU 78.51):
- 3.25 bpw: HIGGS p=4 3.792; dynamic data-free **3.675**; dynamic **3.466** (MMLU 76.98). NF 41.76; AF 102350; HQQ 4.057.
- 4.02 bpw: HIGGS p=2 **2.986**, p=3 2.956; dynamic (4.00) **2.827**; HQQ 4.023; NF/AF collapse (2084 / 513.5).
- 4.25 bpw: HIGGS p=2 2.852; dynamic 2.787; NF 3.065; HQQ 2.873.
- 2-bit: not stated.
Llama-3.1-8B (Tab. 3/4): 4.0-bit dynamic 5.87 vs GPTQ 4.02 6.238, AWQ 6.228 (FP16 5.606). GPTQ+HIGGS on Llama-2-7B (Tab. 2): 2-bit 8.637 (QuIP# 8.22 no FT, QTIP 6.82), 4-bit 5.213. So **data-free HIGGS matches or beats calibrated GPTQ/AWQ at 3–4 bpw, but HIGGS does not compete at 2 bpw.**

**Decode cost and kernel.** FLUTE (LUT-based GEMM) adapted to p = 2 vector lookups. RTX 4090, Llama-3.1-8B (Tab. 1), batch 1: FP16 57; **FLUTE/HIGGS 2/3/4-bit 173/150/139 tok/s** (3.0/2.6/2.4×); QTIP 155/136/122; AQLM 2-bit 69; NF4 31; Marlin 4-bit 133. Batch 16: FLUTE 2432/2122/1979 vs FP16 862. Activations are rotated online; overhead ≤ 4% (App. G). CPU kernel: none. **Feasibility on AVX2: high [est].** p = 2, n = 256 → 256 × 2 × FP16 = 1 KiB table; p = 1, n = 16 → a 16-entry table (one `vpshufb` pair per 32 lookups). The group-wise RHT (g = 1024) on activations is one FWHT per group.

**Quantization cost.** Data-free fixed-bit HIGGS: one pass over the weights, seconds to minutes; **layer by layer on 8 GB RAM: yes, trivially**. Dynamic HIGGS needs J = 15 noise-inserted evaluations per layer (PPL or KL over 287k tokens) → L × 15 full-model forward passes. For an 80-layer model that is 1200 evaluation passes; on our machine this is slow but can run on a small token budget [est].

**Relevance to MoE.** High, for three reasons.
1. **No per-expert codebook** (fixed grid) and a small scale overhead.
2. **The linearity theorem gives a principled per-expert bit allocation.** Extend the knapsack with expert usage frequency f_e: minimize Σ_e f_e·α_e·t²_{e,b} subject to Σ_e f_e·b_e·d_e ≤ SSD byte budget per token. Hot experts get more bits; cold, rarely used experts get fewer, and the expected PPL cost is predicted, not guessed [est, not in paper; the paper itself lists MoE as untested].
3. Data-free → we can quantize a 235B+ MoE on this machine by streaming tensors from disk.

### Q11 — VPTQ (Liu et al., arXiv 2409.17066)

**Mechanism.** GPTQ-style second-order PTQ with vector quantization.
- Reshape a weight matrix into vectors of length v along the **column** direction. Quantize **one column at a time** ("channel-independent second-order optimization"): for column q, the Lagrangian reduces to minimizing ‖v − C‖² (plain nearest-centroid, because H_qq⁻¹ is constant within a column); then propagate the error to the remaining columns with the GPTQ update. (GPTVQ quantizes v columns at once and accumulates more error.)
- Centroid init: **Hessian-weighted k-means** (weights = diag(H)).
- Options: residual VQ (a second codebook on the residual), outlier columns with their own codebook (e.g. v₀ = 4, k₀ = 4096 → 3 bits for N% of weights), several codebook groups per matrix, layer-wise FT of centroids + RMSNorm (100 iterations, 128 C4 samples), optional end-to-end FT.
- Typical 2-bit: v = 6, k = 4096 (12 bits / 6 weights = 2 bits). v = 8, k = 65536: codebook adds 0.19 bpw. v = 12, k₁ = k₂ = 4096 (RVQ): 2 bits.

**True bpw.** Reported numbers include the codebook: codebook bits = (v₀k₀ + v₁(k₁ + k₂))·16 per matrix; index bits = log2(k)/v (App. C). 2.02–2.11, 3.01, 4.01.

**Accuracy, Llama-2-70B** (context 4096, FP16 3.12): **2.07: 3.93** / C4 5.72, avg QA 68.6 (FP16 70.2; QuIP# 2.0: 69.0; AQLM 2.07: 68.8); 2.11: 3.92. **3.01: 3.34**; **4.01: 3.19** (App. Tab. 5). Llama-3-70B (context 2048, FP16 2.9): 2.02 **5.6** (avg QA 70.9 vs 75.3), 3.01 **3.81**, 4.05 **3.15**; GPTQ 2-bit 11.9, QuIP 2-bit 13.

**Decode cost.** Measured with **naive Torch/Triton kernels** (no CUDA graphs): 70B 9.7 tok/s (AQLM 6.9, QuIP# 1.9 in the same naive setting). 13B (App. Tab. 11): v = 6, k = 4096 → 32.1 tok/s vs FP16 30.0; v = 8, k = 65536 → 30.6; RVQ v = 12 → 21.3. The paper says v ≈ 6 makes codebook reads match the 128-byte cache line. Batch-1 speed with optimized kernels: not stated. CPU kernel: none. **Cache:** v = 6, k = 4096 FP16 = 4096 × 6 × 2 B = **48 KiB** per codebook > 32 KiB L1D on Zen 3; fits L2 (512 KiB) [est]. On GTX 1650: fits 64 KiB shared memory once, not with duplication.

**Quantization cost.** Table 2: Llama-2-70B **19 h on 4× A100-80GB** (AQLM 183 h, QuIP# 25 h, GPTQ 2.83 h in the same accounting). Layers are independent given Hessians → fully parallel. On 8 GB RAM: one layer's Hessian + weights fit; k-means with 4096 centroids over ~10⁶–10⁷ vectors per matrix on CPU is slow but feasible [est]; per-expert Hessians are the issue (see Part 2).

**Relevance.** Medium. Accuracy ≈ QuIP#/AQLM at 2 bits with far lower quantization cost than AQLM, and with RVQ it can be nested. But the 48 KiB codebook is per matrix (per expert → overhead and cache pressure), and no CPU kernel exists.

### Q12 — Any-Precision LLM (Park et al., arXiv 2402.10517)

**Mechanism.** One stored model serves every bit-width from a seed n₁ up to a parent n_K.
- **Incremental upscaling** on a non-uniform clustering quantizer (SqueezeLLM, Q17). Seed: quantize to n₁ bits (2^{n₁} clusters per output channel, sensitivity-weighted k-means). Upscale one bit: split each cluster into two sub-clusters by weighted k-means on its members; every weight keeps its old n-bit code and **appends one bit**. Repeat to n_K.
- Each bit-width k has its own **per-output-channel centroid table** of 2^k FP16 values. The codes are nested by construction: the top k bits of a weight's n_K-bit code are its k-bit code.
- Upscaling does not work for uniform GPTQ/AWQ (Tab. 2: AWQ+IU 4-bit 22.5 vs AWQ 5.59; GPTQ+IU 4-bit 12.7 vs 5.68).
- **Bitplane storage** (Fig. 4): plane b holds bit b of every weight. A k-bit run reads only planes MSB … MSB−k+1. Kernel steps per thread: load 32 activations + k 32-bit plane words → bit-transpose (40 bitwise ops for 4 planes, vs 76 with the textbook 8×8 transpose) → shift/mask to indices → table lookup in shared memory → FMA. Optimizations: plane byte permutation so warp loads coalesce (WLO), faster bit-transpose (IBT), 3-bit table-lookup merging (two 3-bit indices → one 6-bit index into a 64-entry pair table, TLM).

**True bpw.** Reading precision k costs k bits + 2^k·16/d_in bits for the k-bit table. Storing all tables 3…8 for d_in = 4096: (8+16+32+64+128+256)·16/4096 = 1.97 bpw extra [est]; Tab. 1 shows Llama-2-7B {3,…,8} = 8.4 GB vs {4,8} = 7.7 GB. **MoE warning [est]:** per-row tables scale as 1/d_in. An expert down_proj with d_in = 768–2048 pays 0.12–0.33 bpw for a 4-bit table.

**Accuracy.** Largest model 7B. Llama-2-7B WikiText-2 (Tab. 3; FP16 5.47): SqueezeLLM 3-bit 6.13; upscaled 4-bit 5.62 vs independent 4-bit 5.61; 5-bit 5.50 vs 5.50; 6–8-bit 5.47. Zero-shot avg: upscaled within 0.4 points (Tab. 4). A 2-bit seed: not tested. 70B: not stated.

**Decode cost and kernel.** GEMV speedup vs cuBLAS FP16 (Tab. 6, Llama-2-7B shapes): RTX 4090 ~3.7–4.7× at 3-bit, ~3.0–3.4× at 4-bit, ~1.6–1.8× at 8-bit; matches SqueezeLLM's dedicated 3/4-bit kernel on the 4090 and the RTX 4070 Laptop, beats it on Jetson Orin. End-to-end in TensorRT-LLM on the 4070 Laptop (Fig. 12): speeds scale near-linearly with bit-width (exact values not readable from the text dump). For large batch (prefill) the engine dequantizes then calls cuBLAS (+~100 µs). CPU kernel: none, but the paper notes bitplane GEMM is "relatively common in CPU implementations". AVX2 feasibility: high; the bit-transpose is plain AND/shift/OR on 256-bit registers, and a 16-entry table per row is one `vpshufb` pair [est].

**Quantization cost.** Upscaling 3 → 8 bits: **< 1 minute for a 7B model on a 24-core i9-13900K CPU** (Tab. 5: seed generation 36 s + upscaling 16 s for Llama-2-7B). The SqueezeLLM seed needs Fisher information (gradients) — see Q17.

**Relevance.** **Very high for the hot/cold design.** It is the only paper here with a working, measured bitplane kernel whose speed scales with the number of planes read. See Part 3.

### Q13 — Matryoshka Quantization, MatQuant (Nair et al., Google DeepMind, arXiv 2502.06786)

**Mechanism.** Train one int8 model whose most significant bits are good int4 and int2 models.
- Slicing (Eq. 6): S(q^c, r) = clamp(round(q^c / 2^{c−r}), 0, 2^r − 1) · 2^{c−r}. **The rounding uses bit c−r−1** (App. A: e.g. 53 → int2 value 1, because the 32-bit is set). Clamp keeps 240 → 3, not 4.
- Loss: Σ_r λ_r · L(F(S(Q(θ, c), r), x), y) over r ∈ R = {8, 4, 2}, with λ = (0.1, 0.1, 1) for Gemma, larger int2 weight is essential. Base algorithms: OmniQuant (learn scales/shifts, block reconstruction loss) or QAT (STE).
- Interpolation: int6 and int3 sliced from the same model, never trained, are about as good as dedicated models. Layer-wise Mix'n'Match (int2 at the ends, int8 in the middle = "pyramid") spans the accuracy/bits curve.
- **Errata (§7)**: the first draft trained without the clamp, so a sliced r-bit value can take 2^r + 1 levels. This "Extra Precision MatQuant" needs one extra bit for a small fraction of weights: **2.05 bpw** for int2, and it helps int2 a lot (Gemma-2 9B task avg 63.35 → 68.25).
- Single-precision MatQuant (train only the int2 loss, but in the int8 container) gives the best int2 of all.
- Only FFN weights in the main results; FFN + attention with QAT is unstable at int2/int3.

**True bpw.** r bits + per-channel/group scales and zero points (granularity not stated) + optional sparse extra bit (+0.02–0.05).

**Accuracy.** Largest: Gemma-2 9B and Mistral 7B. Gemma-2 9B, OmniQuant base (Tab. 1, task avg, bf16 74.38): int8 74.05; int4 73.83 (baseline 74.33); **int2 63.35 vs baseline 60.24**; sliced from a normal int8 model: int2 38.11 (collapse). QAT base (Tab. 2): int2 62.29 vs 56.02. Extra-precision int2 (2.05 bpw): 68.25 (Tab. 7). 70B: not stated. Perplexity: log-perplexity on C4 only.

**Decode cost.** Not measured. The paper notes current hardware supports int8/int4 and custom kernels are needed for int2/int3 and for the sparse extra bit.

**Quantization cost.** TPUv5e. OmniQuant: 128 × 2048 C4 samples, 10 M tokens. QAT: 100 M tokens (FFN) or 300 M (FFN + attention). GPU-hours: not stated. Not feasible for a large model on our machine.

**Relevance.** Medium. It proves that **training for nested slices** is needed (slicing a normal int8 model to int2 collapses) and that a nested int2 can be *better* than a dedicated int2. But it is uniform scalar int2, which is weak at 2 bits compared to VQ/TCQ, it needs training, and its rounding rule makes a k-bit read need k+1 planes (Part 3).

### Q14 — SINQ (Müller et al., Huawei, arXiv 2509.22944)

**Mechanism.** Calibration-free uniform quantization with **two scale vectors**.
- Parameterization: W ≈ s ⊙ (Q + z) ⊙ t, with s a per-row (per-tile) scale, z a shift, t a per-column scale. An outlier W_ij can be traded between row i and column j.
- **Sinkhorn–Knopp-style normalization** (Alg. 1): iterate in log space, dividing rows and columns by their standard deviations (clamped steps), keep the scales with the smallest imbalance I(W) = max(σ_row, σ_col)/min(σ_row, σ_col). Then RTN.
- Observation: the inverse column standard deviation of W predicts the mean input magnitude (pseudo activation-awareness, Fig. 2), because training updates are outer products of inputs and gradients.
- t is applied to the activations (one elementwise multiply, like AWQ), or absorbed into the previous layer ("no-overhead SINQ").
- Defaults: group size 64, dual scale + shift. A-SINQ = SINQ + AWQ calibration.

**True bpw.** b + (scale + shift per 64 weights) + one column scale. With FP16 auxiliaries: b + 32/64 = **b + 0.5** [derived]; some experiments quantize the auxiliaries to 8 bits (b + 0.25) [derived]. Memory column: Llama-3-70B 4-bit 42.81 GB vs BF16 141.11 GB (whole model incl. embeddings).

**Accuracy, Llama-3-70B** (App. Tab. 11, BF16 2.86 / C4 7.30, group 64): 3-bit **4.52** / 8.48 (RTN 5.26, Hadamard+RTN 4.99, HQQ 85.64); 4-bit **3.17** / 7.51 (RTN 3.56, HQQ 3.26, NF4 3.22); SINQ+NF4 3.16. DeepSeek-V3-685B 4-bit: 5.31 vs HQQ 5.38 (BF16 not stated). Qwen3-235B-A22B (Tab. 13, BF16 5.37): 3-bit **6.27** (RTN 10.11, HQQ 13.07); 4-bit 5.58. DeepSeek-V2.5-236B (BF16 5.36): 3-bit 5.82; 4-bit 5.48. Qwen3-30B-A3B (Tab. 19, BF16 8.70): 3-bit 10.19; 4-bit 9.02. 2-bit: not stated.

**Decode cost.** Extra scale costs 0.8–1.8% at the layer level (gemlite kernel, Tab. 5). SGLang W4A16 end-to-end batch 1: 2.1–2.8× FP16, equal to AWQ (Tab. 6). **llama.cpp:** no-overhead SINQ as a pre-processing step for GGUF (Tab. 9): Qwen3-1.7B Q4_0 21.30 → **17.65**; Q3_K_S 24.02 → **19.49** (FP16 17.13), with identical speed. So **a CPU path exists today** through GGUF.

**Quantization cost.** 1.1× RTN time: Qwen3-32B in 52 s on one GPU (Tab. 10). Per layer, independent, no data. **Fully feasible layer by layer on 8 GB RAM** (Sinkhorn iterations on one matrix at a time).

**Relevance.** High for 3–4-bit experts, data-free, MoE-tested (Qwen3-235B, DeepSeek-V2.5/V3, Qwen3-30B-A3B), and already compatible with llama.cpp k-quants. Not a 2-bit solution.

### Q15 — ParetoQ (Liu et al., Meta, arXiv 2502.02631)

**Mechanism.** A unified QAT framework for 1, 1.58, 2, 3 and 4 bits.
- Training budget: ~90% full-precision pre-training + ~10% QAT fine-tuning is best (Fig. 2). QAT fine-tuning saturates at ~10 B tokens for 3/4-bit and ~30 B tokens for 1/1.58/2-bit.
- Finding: 3/4-bit QAT "compensates" (weights move 10–20%); ≤2-bit QAT "reconstructs" (weights move ~40%).
- Quantizer per bit-width: Elastic Binarization (1-bit); **Stretched Elastic Quant (SEQ)** for 1.58 and 2-bit: W_Q = α·(⌊Clip(W/α, −1, 1)·k/2 − 0.5⌉ + 0.5)/k·2, balanced levels without zero, e.g. 2-bit {−1.5, −0.5, 0.5, 1.5}·α; LSQ (with zero) for 3/4-bit. Learnable scale α with STE gradients.

**True bpw.** b bits + per-channel scale (16/d_in).

**Accuracy.** Largest Llama-3-8B (Tab. 1, WikiText-2, FP 6.15, avg 74.6): **2-bit per-channel 8.0** (avg 71.2); 1.58-bit 8.6; 1-bit 9.5. Comparison at 2-bit: AQLM 2.27 bpw avg 70.2; EfficientQAT 2.12 bpw 9.6 / 65.5. Pareto conclusion: 1.58, 2 and 3-bit beat 4-bit on accuracy per model size; 2-bit is the most hardware-friendly. 70B: not stated.

**Decode cost.** H100 CUTLASS W2A16 GEMM, 16384²: **4.14× FP16, 1.24× Machete 4-bit**; for small shapes (1–8B) the 2-bit kernel is not faster than 4-bit (conversion overhead). CPU: own 2-bit kernel on Apple M1 (6 threads); 2-bit beats 4-bit on accuracy-vs-speed (Fig. 7c, values only in the figure). AVX2: not stated.

**Quantization cost.** QAT on 16 GPUs, 120k iterations (2-bit), 30 B tokens. Not feasible for us.

**Relevance.** Medium as evidence: **2-bit with real training lands near 4-bit quality per byte**, and "2-bit scalar codes with no zero level" are the right grid. It argues that if any open MoE ships native 2-bit/QAT weights, those beat our PTQ.

### Q16 — The case for 4-bit precision: k-bit inference scaling laws (Dettmers & Zettlemoyer, arXiv 2212.09720)

**Mechanism.** 35,000 zero-shot experiments, models 19M–176B (OPT, BLOOM, Pythia, GPT-2), 3–16-bit weights with **zero-shot (data-free) RTN-type quantizers**: int, float, quantile, dynamic exponent; block sizes 64–1024; "proxy quantization" (keep top-2% outlier dimensions by weight std in 16-bit).

**Findings.** (1) For fixed total model bits, 4-bit is optimal for almost all families and scales; 3-bit degrades (OPT and Pythia become unstable). (2) Block size 64 instead of 1024 adds 0.24 bpw but improves zero-shot about as much as going from 4 to 5 bits. (3) Quantile and float data types beat int. (4) Outlier-dependent quantization stabilizes 3-bit but does not beat 4-bit. (5) **One-shot methods change the picture**: 2-bit GPTQ with block 64 (WikiText-2 9.18) beats 3-bit float zero-shot (9.99) (Tab. 1).

**True bpw.** b + 16/blocksize (block 64 → +0.25).

**Relevance.** Context only. Its "4-bit is optimal" claim is valid **for zero-shot scalar quantizers**. QuIP# (Fig. 1/4), QTIP and PV-Tuning show that with rotation + VQ/TCQ + calibration, 3-bit and then 2-bit become bit-optimal on Llama-2. For SSD streaming, bit-optimality is exactly the metric we care about, so the newer papers apply.

### Q17 — SqueezeLLM (Kim et al., arXiv 2306.07629)

**Mechanism.**
- **Sensitivity-weighted non-uniform quantization.** Per output channel, find 2^k centroids by weighted k-means, minimizing Σ_i F_ii (w_i − Q(w_i))², where F_ii is the diagonal Fisher information (squared gradients over ~100 calibration samples; 10 samples are "typically sufficient").
- **Dense-and-sparse decomposition.** W = D + S. S holds outliers outside percentile thresholds (0.40%) plus the most sensitive values (0.05%) in FP16, CSR format. D is quantized with the narrowed range.
- Kernels: 3/4-bit LUT GEMV (per-row FP16 LUT) + balanced sparse CSR kernel (10 nonzeros per thread), launched together.

**True bpw.** k + 2^k·16/d_in per row (e.g. 3-bit at d_in 8192: 3.016) + sparse part. Reported: 3.02 / 3.24 (0.45%) / 4.05 / 4.27; 2.01 / 2.05 (0.1%) / 2.22 (0.45%).

**Accuracy.** Llama-2-70B, context 2048 (Tab. F.9/F.10, FP16 3.32): 4.05 bpw **3.41**; 4.27 bpw 3.39; 3.02 bpw **3.77**; 3.24 bpw **3.63**; 2.05 bpw 5.38; 2.22 bpw **4.99** (dense-only 2.01: 9.44). Context 4096 (Tab. F.8, FP16 3.12; parse note: re-aligned): 4.05 bpw 3.21; 3.02 bpw 3.55; 2.01/2.05/2.22 bpw 10.86 / 5.04 / 4.71. So SqueezeLLM is competitive at 3–4 bits, weak at 2 bits.

**Decode cost.** A6000, 128 tokens (Tab. 3): LLaMA-65B 3-bit 7.6 s (~17 tok/s), with 0.45% sparse 8.8 s; LLaMA-7B 1.5 s vs FP16 3.2 s (2.1×). Grouped GPTQ with act-order is much slower (117.8 s for 65B). CPU kernel: none. AVX2 feasibility: high for the dense part (8- or 16-entry per-row table); the CSR sparse part is a scalar gather on CPU [est].

**Quantization cost.** LLaMA-65B: Fisher **2.5 min on an A100** + k-means **80 min on a 48-core Xeon** (Tab. E.6). Fisher needs a backward pass through the full model (memory in Tab. E.5: higher than GPTQ). On our machine, k-means is feasible on CPU; the Fisher pass for a 70B model is not, unless done layer-wise with stored activations [est].

**Relevance.** Medium. It is the backbone of Any-Precision (Q12). Its per-row LUT is small and CPU-friendly but its overhead grows on small-d_in expert matrices.

### Q18 — bitnet.cpp (Wang et al., arXiv 2410.16144)

**Mechanism.** CPU kernels for ternary BitNet b1.58 weights, built on llama.cpp.
- **I2_S**: each ternary weight → 2 bits (−1 → 00, 0 → 01, 1 → 10); unpack and multiply-add. Best with many threads.
- **TL1**: two weights → 4-bit index; precompute the 3² = 9 possible partial sums of the matching two activations into an int16 LUT; GEMV = table lookups + adds (T-MAC style). 2 bpw.
- **TL2**: three weights → 1 sign bit + 4-bit index (27 combinations folded by sign symmetry) → 5 bits / 3 weights = **1.67 bpw**, 1/6 smaller than TL1; recommended when bandwidth-limited.

**True bpw.** I2_S and TL1 2.0; TL2 1.67; plus a per-tensor scale.

**Accuracy.** Lossless relative to FP32 for ternary models (700M model, 1000 WildChat prompts: 100% exact match; llama.cpp TQ1_0/TQ2_0 1.4%). Note: the larger "models" in the speed tables are dummy configurations, and the llama.cpp baseline is **FP16**, not a 4-bit GGUF.

**Decode speed** (Tab. 4, unlimited threads): Intel i7-13700H (x86, AVX2 class): 7B 18.75 tok/s (FP16 llama.cpp 3.30), 13B 10.99, **70B 2.44 tok/s, 100B 1.70 tok/s**; Apple M2 Ultra 70B 8.67, 100B 6.58. Two threads (Tab. 5): i7 70B 1.51 tok/s. Energy −72% to −82% on x86.

**Quantization cost.** None (weights are already ternary; conversion only).

**Relevance.** Medium. It is the only measured **x86 CPU LUT kernel** in this set, and the TL1/TL2 design (precompute activation partial sums, index with packed weight bits) transfers to any low-bit grid with few levels. But only natively ternary models benefit, and none is a large MoE.

### Q19 — BitNet b1.58 2B4T technical report (Ma et al., arXiv 2504.12285)

**Mechanism.** Native W1.58A8 training from scratch: absmean ternary weights {−1, 0, +1}, absmax 8-bit per-token activations, SubLN, squared ReLU, RoPE, no biases. 2B parameters, 4 T tokens pre-training, then SFT and DPO.

**True bpw.** 1.58 ideal; packed storage via bitnet.cpp kernels (2 or 1.67 bpw); GPU kernel packs four ternary values into one int8.

**Accuracy.** Instruction-tuned 2B: average of 17 benchmarks 54.19 vs Qwen2.5-1.5B 55.23, Llama-3.2-1B 44.90 (Tab. 1). Versus Qwen2.5-1.5B INT4 PTQ: BitNet 55.01 avg vs GPTQ-int4 52.15, AWQ-int4 51.17, at 0.4 GB vs 0.7 GB non-embedding memory (Tab. 2). Native 1-bit at 2B beats post-hoc ternarized 7–8B models (Falcon3-1.58bit, Llama3-8B-1.58) on most tasks.

**Decode cost.** CPU TPOT 29 ms (vs Qwen2.5-1.5B 65 ms), non-embedding memory 0.4 GB; estimated decode energy 0.028 J vs 0.347 J. GPU: custom W1.58A8 CUDA kernel; speed numbers not stated in the sections read.

**Quantization cost.** Not applicable (training from scratch, 4 T tokens).

**Relevance.** Low for now (2B, dense). Confirms the synthesis: watch for a large native-ternary MoE; we cannot produce one.

### Q20 — Optimal Brain Compression, OBC/OBQ (Frantar & Alistarh, arXiv 2208.11580)

**Mechanism.** Exact Optimal Brain Surgeon per layer. Because the layer-wise squared error has no Hessian coupling between output rows, each row has its own d_col × d_col Hessian H = 2XXᵀ. Removing (or quantizing) one weight updates the rest of the row with the OBS formula and updates H⁻¹ by a rank-1 row/column elimination (Lemma 1). Cost: **O(d_row · d_col³)** time, O(d_col²) memory per row. OBQ = quantize one weight at a time, greedily by the OBS error, with full reoptimization. Results on ResNets, YOLO, BERT; ResNet-50 pruning in about an hour on an RTX 3090.

**True bpw.** Scalar b bits + per-channel scale.

**Accuracy / decode / cost at 70B.** Not applicable (cubic in d_col; GPTQ (2210.17323) is the scalable successor: fixed column order, shared H⁻¹ across rows, lazy batch updates).

**Relevance.** Background only: OBC → GPTQ → LDLQ (QuIP) → BlockLDLQ (QuIP#) → BlockLDLQ + TCQ (QTIP) is one line of methods. The per-row independence is what lets all of them run one matrix at a time — i.e. on our 8 GB machine.

---

## Part 2 — Master table

Llama-2-70B WikiText-2 perplexity unless marked. **ctx4k** = context 4096 (FP16 **3.12**). **ctx2k** = context 2048 (FP16 **3.32**). L3 = Llama-3/3.1-70B (FP16 2.54–2.9, context noted in Part 1). bpw = true bpw from the paper.

| Method | True bpw | ~2 bpw | ~3 bpw | ~4 bpw | Batch-1 GPU speed vs FP16 | CPU kernel | Quantization cost | Needs FT/training? |
|---|---|---|---|---|---|---|---|---|
| LLM.int8() (Q01) | ~8 + FP16 outlier cols | — | — | — | slower <6.7B; modest gain 13B+ | not stated | zero-shot | no |
| NF4 (Q02) | 4.127 (DQ) / 4.5 | — | L3.1-70B NF 3.25 bpw: 41.8 | L3.1-70B NF 4.25: 3.065; 4.02: 2084 (collapse) | 0.54× (31 vs 57 tok/s, 8B, 4090) | not stated; feasible | data-free, minutes | no |
| QuIP (Q03) | b + ~0 | 6.33 (ctx2k) | 3.85 (ctx2k) | 3.53 (ctx2k) | ~½ of QuIP# | no | 1 GPU 48 GB; time not stated | no |
| QuIP# (Q04) | 2.00–2.01 / 3 / 4 | **3.91** (no FT 4.16) | 3.35 | 3.18 | 70B 2-bit 32.7 tok/s on 4090 (FP16 OOM); 7B 3.2× | no; **feasible** (1 KiB E8P; llama.cpp IQ2_XXS is E8-based) | 70B <10 GPU-h no FT, ~100 GPU-h FT; 25 M-token Hessians | optional |
| AQLM (Q05) | 2.07 / 3.01 / 4.14 (incl. codebooks) | 3.94; e2e-FT 3.83 | 3.36 | 3.19 | 70B 3090: 1×16 1.16×, **2×8 2.5×** | **yes**: i9 8-core 70B 2×8 0.966 tok/s (FP32 0.297) | 70B 3–4 days on 8 GPUs | block FT built in |
| QTIP (Q06) | 2.0 / 3.0 / 4.0 | **3.70** (no FT 3.87) | **3.26** | **3.16** | 7B 3.4×; 70B 2/3/4-bit 23.5/19.1/16.3 tok/s (6000 Ada) | no; **feasible** (computed codes, ≤2 KiB) | not stated; Viterbi-heavy [est] | **no** (no-FT beats QuIP# FT) |
| PV-Tuning (Q07) | 2.07; 1.01–1.14 | **3.78**; 1.14 bpw 5.52 | not stated | not stated | AQLM kernel: 70B 7.2 tok/s (+28%) | via AQLM only | 8×A100, 2–8 days | yes (it is FT) |
| QuaRot (Q08) | 4 (W4A4KV4) | W2 weight-only ~5.6 (ctx2k) | ~3.72 W3 (parse doubt) | 3.79 W4A4; ~3.41 W4-only (ctx2k) | prefill 3.33×; decode memory 3.89× | no | 70B 5 min + 2 h GPTQ | no |
| SpinQuant (Q09) | 3–4 + A8 | not stated | 3.8 W3A8 (ctx2k) | 3.5 W4A8 (ctx2k) | M1 Pro CPU 3× (8B W4A8) | ARM only | rotation 3.5 h (70B) | rotation training |
| HIGGS (Q10) | 3.25 / 4.02 / 4.25 (incl. scales) | not stated | L3.1-70B 3.25: 3.47 dyn (3.79 fixed) | L3.1-70B 4.02: 2.99; dyn 2.83 (FP 2.54) | 8B 3.0/2.6/2.4× at 2/3/4-bit (FLUTE) | no; **feasible** (≤1 KiB grid) | **data-free, minutes** | no |
| VPTQ (Q11) | 2.02–2.11 / 3.01 / 4.01 (incl. codebook) | 3.93 | 3.34 | 3.19 | naive kernels only (70B 9.7 tok/s) | no; feasible (48 KiB codebook, L2) | 70B 19 h on 4×A100 | layer FT; e2e optional |
| Any-Precision (Q12) | k + 2^k·16/d_in, nested 3…8 | not tested | ≤7B | ≤7B | GEMV 3.7–4.7× (3-bit), 3.0–3.4× (4-bit) | no; feasible (bitplane) | upscaling <1 min CPU (7B) + SqueezeLLM seed | no |
| MatQuant (Q13) | 8/4/2 nested (+0.05 extra) | ≤9B: Gemma-2 9B int2 avg 63.35 (bf16 74.38) | ≤9B | ≤9B | not stated | not stated | QAT/OmniQuant, 10–300 M tokens | **yes** |
| SINQ (Q14) | b + ~0.5 (FP16 aux, g64) | not stated | L3-70B 4.52 (BF16 2.86) | L3-70B 3.17 | 2.1–2.8× (SGLang, = AWQ) | **yes via GGUF** (pre-processing for Q4_0/Q3_K_S) | **data-free, ~1.1× RTN** | no |
| ParetoQ (Q15) | b + 16/d_in | ≤8B: L3-8B 8.0 (FP 6.15) | ≤8B | ≤8B | H100 2-bit GEMM 4.14× (large shapes) | ARM M1 (2-bit) | QAT 10–30 B tokens, 16 GPUs | **yes** |
| k-bit laws (Q16) | b + 16/block | — | — | — | — | — | — | — |
| SqueezeLLM (Q17) | 3.02–3.24 / 4.05–4.27 / 2.05–2.22 | 2.22: 4.99 (ctx2k); 4.71 (ctx4k) | 3.02: 3.77; 3.24: 3.63 (ctx2k) | 4.05: 3.41 (ctx2k); 3.21 (ctx4k) | 7B 2.1×; 65B 3-bit ~17 tok/s (A6000) | no; feasible | 65B: Fisher 2.5 min A100 + k-means 80 min CPU | no |
| bitnet.cpp (Q18) | 2.0 (I2_S, TL1) / 1.67 (TL2) | ternary only | — | — | CPU only | **yes, x86 + ARM** (70B dummy 2.44 tok/s, i7-13700H) | none | native ternary training |
| BitNet 2B4T (Q19) | 1.58 W, 8-bit A | 2B only | — | — | GPU kernel exists; speed not stated | yes (bitnet.cpp, 29 ms/token) | 4 T-token pre-training | **yes, from scratch** |
| OBC (Q20) | scalar | not LLM-scale | — | — | — | — | O(d_row·d_col³) | no |

llama.cpp reference points (from A §2.2–2.3, not from these papers): IQ2_XXS 2.0625 tensor bpw (file 2.38 for an 8B mix), IQ2_XS 2.3125 tensor bpw (file 2.59), IQ3_XXS 3.0625 tensor bpw (file 3.25), Q4_K_M 4.89 file bpw. **None of the 20 papers reports perplexity for any llama.cpp format on a 70B model.** The only GGUF numbers are SINQ's on Qwen3 ≤ 4B (Tab. 9).

---

## Part 3 — Pareto frontier for our case

### 3.1 The metric

For SSD-streamed experts, the cost of a format is **true bpw × fraction of expert bytes read per token** (bytes from disk), and the benefit is model quality. A format is only useful if a kernel can decode it at the arrival rate: from SSD at 5 GB/s, 2 bpw means 20 G weights/s; from RAM at 20 GB/s, 80 G weights/s; on the GTX 1650 at 128 GB/s, 512 G weights/s [est]. And it must be producible on (or near) this machine.

Second constraint, specific to MoE: **per-matrix overhead must be small for small matrices.** An expert matrix can be 10–20× smaller than a dense 70B FFN matrix. Formats with a fixed or computed codebook (QuIP# E8P, QTIP, HIGGS, uniform/SINQ) pay nothing per expert. Formats with learned per-matrix codebooks or per-row LUTs (AQLM 1×16, VPTQ, SqueezeLLM, Any-Precision tables) pay more as matrices shrink.

Third constraint: **calibration per expert.** Hessian-based methods (GPTQ/LDLQ/BlockLDLQ, AQLM, VPTQ, QTIP, QuIP#) need X Xᵀ per expert input. With top-k routing, each expert sees only k/E of the calibration tokens (e.g. 8/256 = 3%). To get the same Hessian quality as a dense layer, you need ~30× more calibration tokens [est]. None of the 20 papers studies this. Data-free methods (HIGGS, SINQ) avoid the problem.

### 3.2 Accuracy per bit at 2–3 bpw (Llama-2-70B, ctx 4096, FP16 3.12)

| ~bpw | Best | Then | Then |
|---|---|---|---|
| 2.0–2.1 | QTIP-HYB+FT **3.70** (2.0) | PV-Tuning 3.78 (2.07); AQLM e2e 3.83 (2.07) | QTIP no-FT 3.87 (2.0); QuIP# 3.91 (2.0); VPTQ 3.93 (2.07); AQLM 3.94 |
| 3.0 | QTIP **3.26** | VPTQ 3.34; QuIP# 3.35; AQLM 3.36 | SqueezeLLM 3.55 (3.02) |
| 4.0 | QTIP **3.16** | QuIP# 3.18; AQLM/VPTQ 3.19 | SqueezeLLM 3.21 (4.05) |

Per SSD byte: **QTIP at 3.0 bpw (3.26) is within 0.10 PPL of FP16**, and QTIP at 2.0 bpw (3.70, or 3.87 with no FT) costs 2/3 of the bytes of 3-bit. On Llama-2-70B, 2-bit QTIP (3.70) is close to 3-bit SqueezeLLM (3.55 at 3.02 bpw) and to 4-bit QuIP (3.58, QuIP# Tab. 4), at 2/3 and 1/2 of their bytes. On Llama-3-70B the 2-bit gap is larger (QTIP 2.59 → 4.97 at ctx 8192; VPTQ 2.9 → 5.6 at ctx 2048), so **Llama-3-class models are harder at 2 bits**; at 3 bits QTIP gives 3.18 and VPTQ 3.81 on Llama-3-70B.

### 3.3 Kernel feasibility on this machine

| Format | AVX2 CPU | GTX 1650 | Producible here? |
|---|---|---|---|
| QTIP (1MAD/3INST, HYB Q≤9) | No kernel exists. Feasible: per-lane shifts + LCG + byte sum, ~1 instruction per weight amortized [est]; compute may limit RAM-resident hot experts at 2 bpw | Feasible; computed codes need no shared memory; no tensor cores needed for GEMV | Hessians + Viterbi: heavy; days on GTX 1650 per 70B-class [est]. No FT needed |
| QuIP# E8P (+RVQ for 3/4-bit) | Close to llama.cpp IQ2_XXS (E8 grid + signs). Add FWHT on activations | 1 KiB ×32 = 32 KiB shared: fits | Same Hessian cost as QTIP; FT optional (no-FT 2-bit 4.16 is weak) |
| llama.cpp IQ2_XS / IQ3_XXS / Q4_K_M | **Exists, AVX2** | CUDA exists | **Yes, today** (imatrix) |
| HIGGS p=2 (3.25–4.02) | Feasible; 1 KiB grid | Feasible | **Yes, data-free, minutes** |
| SINQ + GGUF k-quants (3–4) | **Exists** (it is a pre-processing step) | Exists | **Yes, data-free** |
| AQLM 2×8 | **Measured on x86** (70B 0.97 tok/s on 8 cores) | Exists (CUDA) | No (days on 8 GPUs) |
| VPTQ v6 k4096 | Feasible, codebook in L2 | 48 KiB, no duplication | Partly (19 h on 4×A100 for 70B) |
| Any-Precision bitplane | Feasible (bit-transpose = AND/shift/OR) | GPU kernel exists (CUDA, measured on a 4070 Laptop) | Upscaling yes (CPU, minutes); Fisher seed is the hard part |

### 3.4 Recommendation

**Pareto choice at 2–3 bpw for our case: Hadamard-rotated trellis-coded quantization (QTIP, computed code 1MAD/3INST or HYB with Q ≤ 6), at 3 bpw for the general expert file and 2 bpw for the cold tier, with no fine-tuning.**
- Why: best accuracy per bit of all 20 papers at 2, 3 and 4 bits; true bpw is exactly k (no codebook per expert, sign vectors < 0.01 bpw); decode is pure integer arithmetic (no L1-sized table, no gathers), which suits both AVX2 and a tensor-core-less GTX 1650; the no-FT variant already beats fine-tuned QuIP# and AQLM, so we do not need a GPU cluster for the expert file.
- What it costs: a new ggml tensor type and an AVX2 + CUDA (sm_75) kernel; a Viterbi quantizer (GPU, hours to days); Hessians per expert with enough routed calibration tokens. The QTIP paper does not state quantization time; we must measure it.

**Practical path in stages.**
1. **Now (no new kernel):** llama.cpp IQ3_XXS / IQ2_XS experts, with **SINQ no-overhead pre-processing** applied before the k-/i-quant step where it helps (SINQ shows −3.6 PPL on Q4_0 and −4.5 on Q3_K_S for Qwen3-1.7B, same speed). Attention and shared/dense weights at Q4_K/Q5_K in RAM/VRAM. No paper gives IQ2_XS/IQ3_XXS perplexity on 70B, so we must measure it against the QTIP/QuIP# numbers ourselves (Llama-2-70B ctx 4096 is the comparable setting).
2. **3–4-bit hot tier, data-free:** HIGGS p = 2 grids (4.02 bpw: L3.1-70B 2.99; 3.25 bpw: 3.47 with dynamic allocation) or SINQ (L3-70B 4-bit 3.17). Both can be produced on this machine by streaming tensors from disk.
3. **2–3-bit cold tier, best accuracy:** QTIP TCQ with a CPU kernel. Use HIGGS's linearity theorem to set per-expert bit-widths under an SSD-byte budget weighted by expert usage frequency.
4. **If GPU time can be rented:** PV-Tuning on the 2-bit expert file (−0.15 PPL at 70B 2-bit).

Comparison with llama.cpp numbers in the papers: **not available.** The papers give no IQ2_XS / IQ3_XXS / Q4_K_M perplexity at 70B. The only cross-reference is structural: IQ2_XXS (2.06 bpw) uses the same E8 lattice idea as QuIP#'s E8P but without the RHT and without BlockLDLQ; QuIP#'s ablation shows each of those parts matters at 2 bits (Llama-2-70B: no E8, no FT 4.58 → E8 no FT 4.16 → + FT 3.91). We therefore expect IQ2_XS to sit between QuIP no-E8 and QuIP# at equal bits [est, to be measured].

---

## Part 4 — One stored file, several precisions (Any-Precision + MatQuant + Drop-by-Drop)

### 4.1 What each paper offers

| | Any-Precision LLM (Q12) | MatQuant (Q13) | Drop-by-Drop (B §1) |
|---|---|---|---|
| Code family | Non-uniform scalar (per-row clusters) | Uniform scalar int | Additive VQ (AQLM) |
| Nested unit | 1 bit per weight (bit-plane) | 1 bit per weight (MSB slices) | 1 codebook = 8-bit code per 8 weights = 1 bpw |
| Tested range | seed 3 → 4…8 bits | 8/4/2 trained; 6/3 interpolated | 3/4/5 of M = 5 codebooks |
| Lower precision quality | = independent quantization (±0.03 PPL) | int2 **better** than dedicated int2 | ≈ dedicated AQLM at 3/4/5 bits (B) |
| Extra per-precision data | one 2^k-entry FP16 table per row per k | none (shared scale) | none (codebooks shared; first k used) |
| Training needed | no (k-means splits) | yes (QAT/OmniQuant) | yes (AQLM + nested loss, heavy) |
| Largest model tested | 7B | 9B | per B |
| Kernel | measured CUDA bitplane GEMV, speed ∝ planes read | none | LUT GEMV for batch 1 (B) |

**Answer: yes, one file can serve several precisions, and reading k planes costs ~k bpw of SSD bytes, provided the file is stored plane-major and the format is truly nested.** Any-Precision proves it for scalar non-uniform codes with a measured kernel. Drop-by-Drop proves it for additive VQ. MatQuant proves that training for nesting can even improve the lowest precision.

### 4.2 Storage layout needed (bit-plane / code-plane order)

The unit of SSD reads is one expert (all of gate, up, down). Required layout per expert block [design, est]:

```
EXPERT BLOCK (4 KiB aligned, contiguous on disk)
  header      : shapes, available precisions k_min..k_max, plane offsets   (one sector)
  side data   : everything needed at ANY precision, small
                - scales / zero points (shared by all k), RHT sign vectors
                - per-precision tables (Any-Precision: 2^k entries per row, for k_min..k_max)
                  ordered k_min first, so a k-bit read takes a prefix of the tables too
  plane 1     : MSB plane (or codebook-1 codes) of gate | up | down        <- k_min read ends
  plane 2     :                                     gate | up | down        <- after plane k_min
  ...
  plane K     : LSB plane (or last codebook)         gate | up | down
```

Rules:
1. **Precision-major, then matrix.** Plane p of all three matrices is contiguous. A k-bit read of the whole expert is **one `pread` of a prefix** (header + side data for k ≤ k_read + planes 1…k). This keeps colibri's "one expert = one read" property.
2. **Each plane padded to 4 KiB** so unbuffered/direct I/O (`FILE_FLAG_NO_BUFFERING` on Windows) can read any prefix.
3. **Inside a plane, use the kernel's tile order**, not row-major: Any-Precision permutes plane bytes so that 32 threads read 128 contiguous bytes while fetching their matching activations (WLO). On AVX2, store 256-bit words so one load gives the plane bits of 256 consecutive weights of one row tile [est].
4. **Upgrade in place.** A cold expert loaded at k_min bits can be promoted to k bits later by reading planes k_min+1…k only (contiguous) and appending them in RAM/VRAM. Nothing already loaded is re-read. Demotion = drop the tail planes.
5. **Cache accounting becomes per plane.** RAM/VRAM budget = Σ over cached experts of (side data + k_e planes). The cache policy chooses k_e per expert from its hit frequency (HIGGS-style knapsack, §Q10).

Byte math [est]: an expert with 3 matrices of 2048×7168 (44 M weights) is 11 MB at 2 bpw and 22 MB at 4 bpw. Cold miss at 2 bits: 11 MB → 2.2 ms at 5 GB/s. Promoting a hot expert to 4 bits later: +11 MB once, then it stays in RAM/VRAM.

### 4.3 Format-specific issues

- **Any-Precision (bitplanes, non-uniform):** exact prefix property; the k-bit code is the top k bits. Cost: one table per row per precision (side data), which grows as 1/d_in — for skinny expert matrices, keep only 2–3 precisions (e.g. 2/3/4) to bound the overhead. Bit-transpose costs ALU work (40 bitwise ops per 32 weights at 4 bits); acceptable when SSD-bound. Untested: a 2-bit seed, MoE models, >7B.
- **MatQuant (uniform MSB slices):** the slice uses **round-to-nearest on the next bit** (Eq. 6, App. A). So an exact r-bit value needs r+1 planes (+1 bpw of reads), or a stored per-weight rounding plane, or retraining with a truncation slice (not tested; not stated). The "extra precision" variant adds a sparse +1 bucket (2.05 bpw) that is not a pure plane either. Fix before use: train with the truncation operator that the plane layout implies.
- **Drop-by-Drop (code planes):** each codebook's codes are an array of **bytes** (8-bit code per 8 weights) — byte-aligned planes, no bit-transpose. Reading k planes = k bpw. The codebooks (256 × 8 FP16 = 4 KiB per codebook per layer) are side data shared by all experts only if trained that way; per-expert codebooks add 5 × 4 KiB per matrix (small at 1 bpw-per-plane scale, but L1 pressure grows with k). Cost: AQLM-class quantization (very heavy).
- **Residual VQ variants are naturally nested:** QuIP# 4-bit = E8P stage 1 + E8P stage 2; VPTQ RVQ; AQLM M codebooks. Reading only stage 1 gives a 2-bit approximation, but the stage-1 scale (ρ = 1.03 for 4-bit vs 0.9 for standalone 2-bit) and the BlockLDLQ error feedback were optimized for the full code, so the truncated quality is **not stated** and is likely worse than a native 2-bit code. A nested loss (DbyD-style) would fix it.
- **QTIP trellis is not nested.** The k-bit stream is a sliding window; dropping bits does not give a valid (L, k−1) trellis code. A trellis base layer + residual planes (e.g. 2-bit QTIP + 1–2-bit residual RVQ/E8 stage) is a possible hybrid; not tested in any paper.

### 4.4 Recommendation for the hot/cold design

1. Short term: **two separate files** (e.g. IQ2_XS and Q4_K, or QTIP-2 and QTIP-4) cost 6/4 = 1.5× disk. With 82 GB free, that disk cost is the binding problem. So nesting matters for us mainly because of **disk capacity**, and secondarily for in-place upgrade.
2. Medium term: **residual layout = 2-bit base + 2-bit residual stage**, stored plane-major per expert (§4.2). Base candidates: QuIP# E8P (fixed codebook, CPU path close to IQ2_XXS) or a DbyD-style 2-codebook AQLM with shared per-layer codebooks. Train/quantize the base for standalone quality and the residual on the base's error (RVQ order), not the reverse.
3. If we stay with scalar codes: **Any-Precision layout with a 2-bit seed** (untested) or a MatQuant-style trained int4/int2 with a truncation slice. Measure 2-bit seed quality first; at 2 bits scalar codes are far behind VQ/TCQ (SqueezeLLM 2.22 bpw: 4.99 vs QTIP 3.70 on Llama-2-70B).

---

## Part 5 — Blind spots: papers cited that we still lack

Priority: **H** = needed for the next design step; **M** = useful; **L** = background.

| Pri | Title | arXiv | Why |
|---|---|---|---|
| H | T-MAC: CPU renaissance via table lookup for low-bit LLM deployment on edge | 2407.00088 | x86/ARM LUT GEMV for 1–4-bit; basis of bitnet.cpp TL kernels; template for a QTIP/HIGGS AVX2 kernel |
| H | FLUTE: Fast matrix multiplications for lookup table-quantized LLMs | 2407.10960 | HIGGS kernel; LUT GEMM design and its limits (table size, bank conflicts) |
| H | LUT-GEMM: Quantized matrix multiplication based on LUTs | 2206.09557 | Bitplane/BCQ GEMV; Any-Precision's closest prior kernel |
| H | Bitnet.cpp: Efficient edge inference for ternary LLMs | 2502.11880 | Newer, fuller version of Q18 (kernel details, x86 numbers) |
| H | Model-preserving adaptive rounding (YAQA) | 2505.22988 | QTIP cites it for the Llama-3-70B layer-0 v_proj collapse; better rounding for large models |
| H | GPTVQ: The blessing of dimensionality for LLM quantization | 2402.15319 | VQ inside GPTQ; baseline in QTIP/VPTQ; cheaper than AQLM |
| H | (not in these references) MoE-specific quantization: MoQE, MxMoE, per-expert calibration studies | not found here | No paper in this set studies per-expert Hessians, shared codebooks across experts, or expert-frequency bit allocation |
| M | SpQR: A sparse-quantized representation for near-lossless LLM weight compression | 2306.03078 | Outlier + group baseline at 3–4 bits; many tables compare to it |
| M | OmniQuant | 2308.13137 | Base algorithm of MatQuant; 2/3-bit baseline |
| M | MARLIN: Mixed-precision auto-regressive parallel inference | 2408.11743 | State-of-the-art 4-bit GPU GEMM (Ampere+; check Turing) |
| M | Scaling laws for precision (Kumar et al.) | 2411.04330 | Successor to Q16; low-bit training/inference laws |
| M | EfficientQAT | 2407.11062 | Cheapest QAT baseline in ParetoQ; possible way to train nested codes |
| M | Q-Sparse: All LLMs can be fully sparsely-activated | 2407.10969 | Activation sparsity → fewer bytes per token |
| M | any4: Learned 4-bit numeric representation for LLMs | 2507.04610 | Learned 4-bit grid (cited by SINQ) |
| M | HQQ: Half-quadratic quantization (Badri & Shaji, blog 2023) | no arXiv | Data-free baseline in HIGGS and SINQ |
| M | Accuracy is not all you need (Dutta et al., NeurIPS 2024) | ID not in refs | "Flip rate" metric used by SINQ; better than accuracy for quantized-model evaluation |
| M | Spectra: Surprising effectiveness of pretraining ternary language models at scale | 2407.12327 | Ternary scaling evidence (ParetoQ disputes it) |
| L | BitNet a4.8: 4-bit activations for 1-bit LLMs | 2411.04965 | Ternary + 4-bit activations |
| L | Atom: Low-bit quantization for efficient and accurate LLM serving | 2310.19102 | W4A4 serving (compute-bound) |
| L | CrossQuant | 2410.07505 | Second-axis scale like SINQ, calibrated |
| L | BiLLM 2402.04291; DB-LLM 2402.11960; PB-LLM 2310.00034; ARB-LLM 2410.03129; TernaryLLM 2406.07177; OneBit (ID not in refs) | — | Sub-2-bit PTQ; weaker than PV-Tuning at 1–2 bits |
| L | LLM-QAT | 2305.17888 | Early QAT baseline |
| L | Outlier suppression+ 2304.09145; RPTQ 2304.01089 | — | Activation outliers (W/A quantization) |
| L | Any-precision deep neural networks (Yu et al., 2021); Matryoshka representation learning (Kusupati et al., 2022); MatFormer (Devvrit et al., 2023) | IDs not in refs | Origins of nested precision / nested models |
| L | Ladder (OSDI 2024, BitBLAS) | not arXiv | GPU kernel compiler used for BitNet kernels |

---

## Part 6 — Facts that change our plan

1. **2-bit is now Pareto-optimal on Llama-2-70B** with TCQ/VQ + rotation (QTIP 3.70, PV-Tuning 3.78 at ~2 bpw vs FP16 3.12). The "4-bit is optimal" result (Q16) is only for zero-shot scalar quantizers. For SSD streaming, this halves bytes vs 4-bit at ~+0.6 PPL, or cuts bytes by 1/3 vs 3-bit at ~+0.45 PPL.
2. **Fine-tuning is not required** to beat fine-tuned QuIP#/AQLM: QTIP's computed codes without FT already do (3.87 at 2-bit, 3.27 at 3-bit).
3. **Data-free 3–4-bit is as good as calibrated GPTQ/AWQ** (HIGGS, SINQ), and it runs on our 8 GB machine layer by layer. SINQ is MoE-tested at 235B and plugs into GGUF today.
4. **Llama-3-class models are harder** (QTIP 2-bit L3-70B 4.97 vs 2.59; layer-0 v_proj must stay unquantized). Modern MoE models may behave like Llama-3, not Llama-2. Measure before committing to 2-bit.
5. **Per-matrix codebooks and per-row tables are a hidden cost for small expert matrices.** Prefer fixed/computed codes (E8P, TCQ computed codes, HIGGS grids, uniform+scales).
6. **Calibration per expert is an open problem** (few routed tokens per expert). Data-free methods avoid it; Hessian methods need many more calibration tokens.
7. **One file, several precisions works** (Any-Precision, DbyD, MatQuant), if the file is plane-major per expert. The cleanest kernel-friendly nesting is byte-aligned code planes (DbyD) or residual VQ stages; scalar bitplanes work with a bit-transpose; MatQuant needs a truncation-consistent slice.
8. **CPU decode compute matters for hot experts in RAM** at 2 bpw (~80 G weights/s needed at 20 GB/s). From SSD it does not (≤ 20 G weights/s). Kernels with integer-only decode (TCQ computed codes, E8P signs, TL-style LUTs) are preferred over float gathers.
