# B. 2026 Quantization Papers: Deep Analysis (Analyst B)

Scope: four papers in `<repo>\papers\`:

| ID | File | Short name |
|---|---|---|
| 06 | 06_DropByDrop_AdditiveCodebooks_2606.12876.pdf | Drop-by-Drop (DbyD), arXiv:2606.12876v2 |
| 07 | 07_OffQ_2606.07116.pdf | OffQ, arXiv:2606.07116v1 |
| 08 | 08_WINDQuant_2605.26660.pdf | WINDQuant, arXiv:2605.26660v2 |
| 12 | 12_WideningTheGap_OutlierInjection_2605.15152.pdf | Outlier Injection attack, arXiv:2605.15152v2 |

Project goal (reminder): run 70B to 700B+ parameter LLMs on machines with 8-32 GB RAM and 0-12 GB VRAM.

Conventions in this document:
- "p.N" = PDF page N. "Tab. N" / "Fig. N" = table / figure number in that paper.
- "Not stated" = the paper does not give the fact.
- "Derived" = my own arithmetic from numbers in the paper. I mark each derived number.
- "Hypothesis" = my own inference. The paper does not test it.

---

## 0. Key framing for our goal (read first)

A dense parameter count sets a hard floor on weight memory: bytes = params x bits / 8.

| Model size | 1 bit/w | 2 bits/w | 3 bits/w | 4 bits/w | 5 bits/w |
|---|---|---|---|---|---|
| 70B | 8.75 GB | 17.5 GB | 26.3 GB | 35 GB | 43.8 GB |
| 700B | 87.5 GB | 175 GB | 263 GB | 350 GB | 438 GB |

(Derived. Linear weights only. Embeddings, output head, KV cache and runtime buffers are extra.)

Consequences:
1. A 70B model fits in 32 GB RAM only at about 3 bits/weight or less. At about 2 bits/weight it fits in about 18 GB.
2. A 700B model does not fit in 32 GB RAM at any bit-width in these papers. For 700B we must stream weights from SSD (for example, MoE expert offload). Quantization reduces the bytes per token that we read from SSD. It does not remove the need for streaming.
3. None of the four papers tests an MoE model, a model above 72B, CPU inference, or SSD offload.

---

## 1. Drop-by-Drop: Multi-Bitwidth Quantization for LLMs Using Additive Codebooks (arXiv:2606.12876)

Authors: Babaoglu, Chen, Khisti (University of Toronto). Built on AQLM.

### 1.1 Problem it solves

Normal PTQ methods produce one model per bit-width. To serve K bit-widths you must quantize, fine-tune and store K checkpoints (p.1-2, Fig. 1). DbyD makes one checkpoint that serves several bit-widths. You select the bit-width at load time by the number of additive codebooks k that you use (p.6).

Plain AQLM does not support this. If you drop codebooks from a normal AQLM model ("AQLM (Drop)"), quality collapses. Example: Llama-3-8B C4 perplexity at 3x8 is 7.8x10^4 (Tab. 7, p.36).

### 1.2 Core mechanism

#### 1.2.1 Theory (Section 4, App. D-E)

Theorem 1 (p.5): if W has i.i.d. zero-mean Gaussian entries, the source (W, d_WMSE) is successively refinable, where

    d_WMSE(W, W_hat) = E[ || (W - W_hat) X ||_F^2 ]

Successive refinement means: one embedded code exists whose k-prefix reaches the optimal rate-distortion point at each distortion level D_1 > D_2 > ... > D_K, with no rate penalty compared with a separate code per level (p.2).

Proof idea (p.5): set Y = W X. Then y = vec(Y) is Gaussian with covariance (sigma_w^2 X^T X) (kron) I_dout. Diagonalize it. Apply reverse water-filling per eigen-component: D_i^(k) = min(mu_k, lambda_i). The incremental rate at stage k is 1/2 log(D_i^(k-1) / D_i^(k)). The sum of increments equals the single-shot rate 1/2 log(lambda_i / D_i^(K)). The pull-back to W uses X^-1, the left inverse, or a compact SVD surrogate, depending on the shape of X (p.25-26).

Practical meaning: the theory only motivates the design. Real weights are not Gaussian. Excess kurtosis of raw weights is +0.7 to +6.1 (Tab. 3, p.16). A block Walsh-Hadamard rotation reduces the model-average excess kurtosis, for example Qwen-7B from +2.48 to +0.50 and Qwen-32B from +2.07 to +0.61 (Tab. 3).

#### 1.2.2 Representation (AQLM base, Section 3.2, p.4)

- Split each row of W (d_out x d_in) into groups of g consecutive weights. The paper uses g = 8.
- Use M codebooks C_1..C_M. Each C_m is a g x 2^B matrix (2^B codewords of length g). The paper uses B = 8, so 256 codewords per codebook.
- Each group (i, j) stores M indices b_ijm (8 bits each).
- Reconstruction of group j of row i: sum over m of C_m[b_ijm], multiplied by a per-output-row scale s_i.
- Codebooks are per layer (one set of M codebooks per linear layer).

Bits per weight (App. B, Eq. 2, p.14):

    b_bar = (16*g*M*2^B + d_out*(d_in/g)*B*M + 16*d_out) / (d_out*d_in)

Each codebook adds exactly B/g = 8/8 = 1 bit/weight of codes. So "Kx8" means about K bits/weight. Example from the paper: Llama-2-70B mlp.gate_proj (d_in 8192, d_out 28672), M=2, g=8, B=8 gives about 2.002 bits/weight (p.14). Derived for M=5 on the same layer: about 5.003 bits/weight. Codebook overhead per layer at M=5 is 8*5*256*16 bits = 20 KB.

#### 1.2.3 Drop-by-Drop loss (Section 5, p.6)

Partial reconstruction from the first k codebooks:

    W_hat^(k)_i = sum_{m=1..k} C_m b_{i,1,m}  (+)  ...  (+)  sum_{m=1..k} C_m b_{i,d_in/g,m}

Per-prefix weighted MSE (with Hessian proxy H = (1/n) X X^T):

    L^(k) = (1/d_out) * tr[ (W - W_hat^(k)) H (W - W_hat^(k))^T ]

Total loss:

    L_DbyD = sum_{k=1..M} alpha_k L^(k),   alpha_k >= 0,  sum alpha_k = 1

Weights used in the main results ("35W"): alpha_3 = alpha_5 = 1/2, all others 0, with M = 5 (p.7). "Uni": all alpha_k = 1/5.

#### 1.2.4 Full algorithm (Algorithm 2, p.31). Engineer-level steps

For each transformer block i (sequential, block by block):
1. Run the FP block on the current block inputs X_block. Save Y_block (targets).
2. For each linear layer in the block:
   a. W := layer.weight. Accumulate H = X X^T from X_block (d_in x d_in).
   b. Optional Hadamard: select block size B_H as the largest of {256, 128, 64, 32} that divides d_in (else B_H = 1). Sample random signs S in {-1,+1}^d_in. Set W <- BlockFWHT_1D(W (.) S, B_H). Set H <- BlockFWHT_2D(H (.) S S^T, B_H) (p.31).
   c. Initialize codebooks C, codes b, scales s with residual k-means: k-means on groups, then k-means on residuals for each next codebook (App. B, p.14).
   d. Repeat until the relative loss improvement is less than tau = 0.01 (checked once per epoch):
      - For k = 1..M compute W_hat^(k) and L^(k).
      - Update codebooks C by gradient descent on L_DbyD (all prefixes).
      - Update codes b by beam search against L^(M) only (the full reconstruction). Reason: beam search on all prefixes would cost M times more (p.31).
   e. Store AQLM format (C, b, s) plus signs S and B_H.
3. Block fine-tune: train all trainable parameters of the block (codebooks, scales, norms) with Adam on ||block(X_block) - Y_block||^2, using only the full k = M reconstruction. Early stop after 3 non-improving epochs on validation (p.31).
4. X_block <- quantized block(X_block). Go to the next block.

Inference (Algorithms 3-4, p.32):
- Choose k <= M at load time.
- If S exists and B_H > 1: x <- BlockFWHT_1D(x (.) S, B_H) on the activation (online, every forward pass).
- y = Dequantize(codes, codebooks, scales, k) x. The kernel never materializes the dense W (p.32).

### 1.3 What gets quantized and the storage format

| Item | Value | Source |
|---|---|---|
| Quantized tensors | Weights of linear layers only | p.2, p.6 |
| Activations / KV cache | Not quantized | Not stated otherwise |
| Group (vector) size g | 8 | p.6 |
| Codebooks M | 3, 4 or 5 trained; prefixes k = 3, 4, 5 evaluated | p.6 |
| Codebook size | 2^8 = 256 codewords of length 8, FP16 | p.6, p.14 |
| Scales | One FP16 scale per output row | p.14 |
| Rotation metadata | Random sign vector S (d_in) and block size B_H per layer | p.31 |
| Format | AQLM format (codes, codebooks, scales); codes reordered once at load time for the kernel | p.31, p.49 |
| Embeddings / head | Not stated (AQLM codebase default) | Not stated |

### 1.4 Memory result

- Effective bits: 3x8 = about 3 bits, 4x8 = about 4 bits, 5x8 = about 5 bits per weight (p.6). Overhead from codebooks and scales is small (Eq. 2).
- Disk: one DbyD checkpoint at the highest bit-width supports all prefixes. Fig. 7 (p.9) shows DbyD disk use similar to or lower than one checkpoint of AnyPrecLLM, AnyBCQ or MatGPTQ. Fig. 14 (p.35): for Qwen2.5-32B, the cumulative disk for three AQLM checkpoints is about 57 GB; DbyD is about 22 GB (values read from the bar chart; approximate).
- Model size at inference with k codebooks: about k bits/weight, if the runtime loads only the first k code planes. See Section 1.10 for the conditions.
- Derived for our sizes: 70B at k=3: about 26 GB; k=4: about 35 GB; k=5: about 44 GB.

### 1.5 Inference speed and decode cost

Kernel `codeNx8` (App. I.3, p.49-51):
- Problem it fixes: AQLM has fast CUDA kernels only for 1x16 and 2x8. For 3-5 codebooks, AQLM falls back to a Triton GEMV (4 codebooks) or dequantize-then-GEMM (3 and 5) (p.49).
- Method: at batch size 1, for each 8-wide input segment, compute the dot product of the input with every codeword of every codebook once (a lookup table). Then each weight group costs one table read per active codebook, instead of fetching an 8-vector and doing 8 multiply-adds. The table is rebuilt on every call; latencies include it (p.49).
- Decode cost (derived from the description): per weight, k/8 table lookups plus adds. Per layer per token, the table build costs (d_in/8) x k x 256 x 8 = 256 x k x d_in multiply-adds, which is small relative to d_out x d_in when d_out is large. Example: d_in = 8192, k = 5 gives a table of 1024 x 5 x 256 = 1.31 M entries (about 2.6 MB in FP16) per layer per token.
- Measured on one NVIDIA L40S, batch-1 FP16 GEMV, L2 cache flushed before each call (p.49):
  - vs cuBLAS FP16: up to 3.69x at 3-bit, 2.96x at 4-bit, 2.49x at 5-bit (largest shapes, Tab. 25, p.50). Small shapes: 1.28x-1.79x.
  - vs AnyBCQ, AnyPrecisionLLM, MatGPTQ kernels: 1.02x-1.46x (p.8; Tab. 25-26).
  - vs AQLM's own path for 3-5 codebooks: 2.25x-85.6x (p.49).
  - Hadamard variant `codeNx8_Had` is slightly slower but still faster than all baselines (Tab. 26, p.51).
- CPU kernel: not stated. Batch > 1 / prefill throughput: not stated (only batch-1 GEMV measured). Kernel release: "upon publication" (p.10); not available in the paper.

### 1.6 Accuracy cost

C4 perplexity, DbyD (Had.+35W) vs individually trained AQLM, mean of 3 runs (Tab. 7, p.36):

| Model | 5x8 AQLM / DbyD | 4x8 AQLM / DbyD | 3x8 AQLM / DbyD |
|---|---|---|---|
| Qwen2.5-32B | 8.84 / 8.79 | 8.98 / 8.98 | 9.46 / 9.49 |
| Qwen2.5-14B | 9.10 / 9.06 | 9.33 / 9.40 | 10.01 / 10.38 |
| Qwen2.5-7B | 10.61 / 10.51 | 10.94 / 10.93 | 11.85 / 12.27 |
| Llama-3-8B | 8.71 / 8.63 | 9.29 / 9.38 | 10.73 / 11.53 |
| Mistral-7B | 7.93 / 7.54 | 7.73 / 7.71 | 8.23 / 8.32 |
| Qwen2.5-0.5B | 17.87 / 17.87 | 18.73 / 19.99 | 21.87 / 27.67 |

AQLM (Drop) at 3x8 (same model, codebooks dropped without DbyD training): Llama-3-8B 7.8x10^4, Mistral 104.10, Qwen-3B 283.97 (Tab. 7).

Observations:
- At 4-5 bits, one DbyD checkpoint is about equal to separate AQLM models.
- At 3 bits, the gap grows as model size drops: +0.03 PPL on 32B, +0.80 on Llama-3-8B, +5.8 on Qwen-0.5B. Larger models lose less. This is a good sign for 70B+, but 70B+ is not tested.
- Zero-shot average (5 tasks): at 4x8 and 5x8, DbyD (Had.+35W) matches AQLM (Ind.) within standard error (p.8). Example Llama-3-8B: 73.37 / 72.36 / 67.81 at 5/4/3 bits vs AQLM 72.42 / 71.00 / 70.00 (Tab. 9, p.38). The 3-bit point loses 2.2 points on Llama-3-8B.

Against other multi-bitwidth methods (C4 PPL, Tab. 1, p.9):

| Bits | Model | DbyD | AnyPrecLLM | MatGPTQ | AnyBCQ |
|---|---|---|---|---|---|
| 3 | Qwen2.5-32B | 9.49 | 11.64 | 11.90 | 12.30 |
| 3 | Llama-3-8B | 11.53 | 13.14 | 14.93 | 13.34 |
| 4 | Llama-3-8B | 9.38 | 10.28 | 10.60 | 10.81 |
| 5 | Qwen2.5-32B | 8.79 | 10.70 | 10.59 | 11.63 |

Exception: Gemma-2B WikiText-2 3-bit, AnyPrecLLM 12.52 vs DbyD 13.18 (Tab. 21, p.45). Zero-shot accuracy is "comparable", within about 1.5 points of the best method in most settings (p.48).

No comparison with GPTQ, AWQ, QuIP#, QTIP, QuaRot or SpinQuant appears in this paper.

2-bit point: only an ablation on Gemma-2B with a 2x8 model trained with the DbyD loss: WikiText-2 19.22, C4 23.37. A 4x4 configuration (4 codebooks of 4 bits) is much worse: 34.02 / 77.17 (Tab. 4, p.33). Prefixes k = 1 and k = 2 of the M = 5 models are not evaluated.

### 1.7 Calibration data and quantization cost

- Calibration: 1024 samples, 32-sample validation split (p.6). Dataset: not stated in this paper. Sequence length: not stated.
- Hardware: NVIDIA L40S (48 GB), mostly 4 GPUs (p.6). All models up to Qwen2.5-32B quantize on L40S. AnyPrecLLM and MatGPTQ needed an H100 for 32B; AnyBCQ for 14B and 32B (p.9, p.45).
- Time: Fig. 14 (p.35) shows "Total Training Hours". Approximate values read from the chart: Qwen2.5-32B about 240 h for DbyD vs about 390 h for three AQLM runs. Whether the unit is GPU-hours or wall-clock hours with 4 GPUs is not stated.
- RAM needed during quantization: not stated.
- Derived estimate for 700B (hypothesis, linear scaling from 32B): about 700/32 x 240 h = about 5,000 hours on L40S-class GPUs. This is not practical on a small machine.
- Derived memory concern: block fine-tuning keeps X_block and Y_block for 1024 samples. With sequence length L and hidden size h in FP16, each tensor is 1024 x L x h x 2 bytes. For L = 4096 and h = 8192 that is about 69 GB per tensor (the sequence length is my assumption). This needs disk offload or fewer/shorter samples on a small machine.

### 1.8 Limitations and failure modes

- Only 3, 4, 5 bits are evaluated. Performance "at even lower bitwidths leaves room for further improvement" (p.9).
- Quantization is expensive (AQLM-class: beam search plus block fine-tuning).
- Beam search and block fine-tuning optimize only the full M-codebook reconstruction. The nested property comes only from the codebook-update loss (p.31). Prefixes that get alpha_k = 0 (for example k = 4 in 35W) still work in tests, but nothing guarantees them.
- Small models degrade more at the lowest prefix (Qwen-0.5B 3x8: 27.67 vs 21.87).
- Gemma-2B results for some baselines are single runs, "in training" (Tab. 7 footnote).
- Max model tested: Qwen2.5-32B. No MoE model.
- Only CUDA kernel, only batch 1, only L40S. Code: anonymous link; checkpoints and kernels promised later (p.10).

### 1.9 Relevance to our goal

- RAM/VRAM reduction: yes. 3-5 bits/weight with better quality than other multi-bitwidth methods. For 70B, k = 3 gives about 26 GB (derived). This fits 32 GB RAM with little room for KV cache.
- It does not reach the 1-2 bits/weight we need for 70B in 8-16 GB, and it does not make 700B fit in RAM.
- llama.cpp / GGUF: no direct support. GGUF has no additive-codebook type with per-layer learned codebooks. We would need a new ggml tensor type and a CPU kernel. The lookup-table decode is suitable for CPU in principle (small tables, integer indices). This is a hypothesis; the paper has no CPU data.
- Biggest blocker: quantization cost for 70B-700B.

### 1.10 Can one checkpoint serve several bit-widths? Does it enable "load more codebooks when more RAM is free"?

Yes, one checkpoint serves several bit-widths. This is the main claim and the data supports it for k = 3, 4, 5 (Tab. 7-9).

Adaptive loading ("load more codebooks when RAM is free"):

What the paper supports:
- k is "specified when the model is loaded" (p.6). Algorithm 3 takes k as an input and uses only the first k codebooks.
- Each codebook contributes an independent 8-bit code per 8 weights. So each codebook is a separate "plane" of exactly 1 bit/weight (derived from Eq. 2).
- The authors list "selecting bitwidths based on available memory" and "allocating different precisions across layers" as future deployment strategies (p.9).

What the paper does not show:
- It does not measure RAM at k < M. It does not say that unused code planes stay on disk. Algorithm 3 receives the full `codes` tensor.
- AQLM stores codes for all M codebooks together per group (layout details not stated in this paper). To read only k planes from disk, the file must store each codebook's codes as a separate contiguous array. This is a file-layout change, not a math change.
- It does not test changing k while the model runs, or using different k per layer.
- Scales s and the block fine-tuned parameters (for example RMSNorm weights) are tuned for k = M only (p.31). A model that runs at k = 3 uses parameters tuned for k = 5. The 3x8 results already include this effect.

My engineering assessment (hypothesis, not tested in the paper):
1. Planar storage: write the file as [codebooks (all M)] [scales] [code plane 1] [code plane 2] ... [code plane M] per layer. Memory-map the file. A loader that maps planes 1..k uses about k bits/weight of RAM.
2. Upgrade at runtime: when RAM is free, map plane k+1 for some layers. The kernel adds one more lookup per group. No re-quantization is needed. Downgrade: unmap the last plane.
3. Per-layer k: possible in the format. Quality of mixed k across layers is not measured. Sensitive layers (to be measured) would get more planes.
4. MoE / SSD streaming idea: keep plane 1-3 of all experts in RAM or on fast SSD, and load planes 4-5 only for hot experts. Not tested anywhere in these papers.
5. Risk: a prefix with alpha_k = 0 has no direct supervision. For a deployment that uses k = 3, 4, 5, train with alpha on all three (the "345W" variant exists in App. G.2, p.33, but its numbers are not given in a table).

Verdict: the format makes progressive loading possible at the math level. A working adaptive loader needs (a) planar code layout, (b) a kernel that accepts a per-layer k, (c) our own tests of quality at mixed k. None of this is in the paper.

---

## 2. OffQ: Taming Structured Outliers in LLM Quantization by Offsetting (arXiv:2606.07116)

Authors: Wang, Mueller, Zhuang, Salzmann, Cavigelli (EPFL, Huawei, SDSC).

### 2.1 Problem it solves

W4A4KV4 (4-bit weights, activations and KV cache) is hard because activations contain structured outliers. Rotation methods (QuaRot, SpinQuant) help but leave large variance (p.1). Other fixes need learned rotations, mixed precision or non-uniform grids, which need custom kernels (p.2). OffQ removes the outlier effect with uniform-grid, uniform-precision 4-bit quantization.

### 2.2 Core mechanism

Key observation (from "singular defects" work, ref [72]): the high-norm outlier tokens share one direction, across layers, token ids and positions (p.2, p.4).

Step A. Top-1 PCA (Section 3.1, p.4):
1. Calibration set: N sequences of length L. Collect inputs to each linear layer (N x L x D).
2. Per sequence, keep only the one token with the largest L-infinity norm. This gives X in R^{N x D}.
3. C = X^T X / N = U Lambda U^T (eigenvalues descending).
4. Rotate: X_rot = X U. The first channel now carries the dominant outlier direction.
Why top-1: with RMSNorm fused into weights, activations lie on a scaled L2 sphere; spiky tokens have large L-infinity norm. PCA over all tokens drifts away from the outlier direction (p.5, Fig. 1b).

Step B. Offsetting (Section 3.2, p.5):
- Build a "partially random" Hadamard H (D x D, entries +-1) with the first row all ones.
- X_had = X_rot H / sqrt(D). The outlier x_1 becomes a constant offset x_1/sqrt(D) in every channel. Asymmetric quantization absorbs this constant offset in the zero-point.
- 2-D example (Eq. 5): [x1, x2] H / sqrt(2) = x1/sqrt(2) + [x2/sqrt(2), -x2/sqrt(2)].

Step C. Group-wise offsetting:
- Split D channels into G groups of size D/G. Permute so that the first channel of group g is the g-th largest outlier channel. Assign the other channels to groups in sorted order (by magnitude), not zigzag (p.5).
- Apply the offsetting Hadamard inside each group. Per-group asymmetric quantization absorbs G outlier channels.

Step D. Full pipeline (Section 3.3, Fig. 2, p.5-6). Built on SpinQuant architecture:
1. Fuse RMSNorm scales into adjacent weights.
2. Collect statistics: (a) inputs to W_Q, W_K, W_V, W_U, W_G stacked over all layers -> one shared U_1 by top-1 PCA. (b) If the model has attention biases (for example Qwen2.5), collect post-RoPE key outputs -> per-layer, per-head U_3. (c) Per-channel absolute max of W_D inputs per layer.
3. Fuse transforms:
   - Right-multiply W_O and W_D by U_1 P_1 H_1. Left-multiply W_Q, W_K, W_V, W_U, W_G by H_1^T P_1^T U_1^T. Rotate embeddings by U_1 P_1 H_1 and apply the inverse to the output head.
   - Value cache: per-head random rotation R_2 fused into W_V and W_O.
   - Key cache: online per-head Hadamard H_3 after RoPE on Q and K. With bias outliers, use U_3 P_3 H_3.
   - Down-projection input: permutation P_4 (from abs-max ranking) fused into W_U, W_G; online group-wise Hadamard H_4 before W_D; H_4^T P_4^T fused into W_D.
4. Quantize weights with GPTQ, 4-bit (p.6).
Three online Hadamards remain at runtime (H_3 on Q and K, H_4) (Fig. 2 caption).

### 2.3 What gets quantized, bit-widths, groups, format

| Tensor | Bits | Granularity | Notes | Source |
|---|---|---|---|---|
| Weights | 4 | Per-channel, symmetric | GPTQ after transforms | p.8 |
| Activations | 4 | Per-group, asymmetric, group 128 | 4.25 bits effective incl. FP16 scale and zero-point | p.7, p.8 |
| KV cache | 4 | Group = head dimension | | p.9 |
| Baselines' activations | 4 | Per-token asymmetric | Not the same granularity as OffQ | p.8 |

Storage format: uniform INT4 with scales/zero-points. Codebooks: none. File format: not stated (HuggingFace / PyTorch, ResQ codebase, p.8).

### 2.4 Memory result

- Weights: 4 bits/weight plus one scale per output channel. Derived: 70B about 35 GB; 72B about 36 GB.
- KV cache: 4 bits plus per-head-group scale and zero-point. Derived: about 3.5-4x smaller than FP16 KV.
- Activation storage: 4.25 bits (p.7). Matters for compute paths, not for model RAM.
- Concrete model sizes: not stated.

### 2.5 Inference speed

Not measured. "We have not reported inference latency in real-world deployment scenarios" (p.9, Limitations). Per-group activation quantization needs custom kernels (Atom, QServe style) (p.7). Benefit to speed requires INT4 matmul hardware (Tensor Cores). No codebook; no decode cost.

### 2.6 Accuracy cost (W4A4KV4, WikiText PPL / 8-task 0-shot average; Tab. 1, p.7)

| Model | FP16 | GPTQ | QuaRot | SpinQuant | ResQ (4.5-bit act.) | OSTQuant | OffQ |
|---|---|---|---|---|---|---|---|
| Llama-3-8B | 6.1 / 67.09 | 166.3 / 39.79 | 7.8 / 62.10 | 7.4 / 63.76 | 7.1 / 63.91 | 7.29 / 64.70 | 6.98 / 65.49 |
| Llama-3-70B | 2.9 / 73.09 | 11655 / 34.90 | 5.7 / 67.56 | 6.2 / 65.68 | 4.1 / 71.14 | 4.01 / 71.16 | 3.88 / 70.63 |
| Llama-2-13B | 4.9 / 66.45 | 3120 / 35.83 | 5.4 / 63.80 | 5.2 / 64.81 | 5.1 / 65.25 | 5.25 / 64.19 | 5.11 / 65.25 |
| Qwen2.5-7B | 6.8 / 68.45 | 13594 / 34.85 | 4036 / 38.36 | - | 8.2 / 65.29 | - | 7.66 / 66.16 |
| Qwen2.5-72B | 3.9 / 73.41 | 37967 / 34.54 | 4.9 / 70.28 | - | 4.6 / 71.98 | - | 4.29 / 72.68 |
| Qwen2.5-1.5B | 9.3 / 60.85 | 25770 / 35.21 | 6600 / 38.33 | - | 12.5 / 55.26 | - | 11.35 / 57.53 |

Notes:
- "GPTQ" here is weight-only GPTQ run under W4A4KV4, so it collapses. This is not GPTQ's normal W4A16 quality.
- Baseline numbers come from the ResQ paper or the original papers (Tab. 1 caption). They were not re-run in one environment.
- OffQ is not best on every model: Llama-3-70B 0-shot 70.63 vs OSTQuant 71.16 and ResQ 71.14.
- No comparison with AWQ, AQLM, QuIP#, QTIP in weight-only settings.

Ablations (Llama-3-8B, Tab. 2, p.8): without top-1 selection 8.27 / 63.13; zigzag grouping 7.03 / 64.95; partial random rotation instead of Hadamard 7.00 / 64.40.
Group size (Fig. 3, p.9): 32 -> 6.86, 64 -> 6.93, 128 -> 6.98, 256 -> 7.04, 512 -> 7.11, 1024 -> 7.16, 2048 -> 7.27, 4096 -> 7.38.

### 2.7 Calibration and quantization cost

- Calibration: "a small calibration dataset of N sequences each of length L" (p.4). N, L and dataset: not stated.
- Cost: not stated. Components: one forward pass to collect stats, eigendecomposition of D x D matrices (cheap), per-channel abs-max, then GPTQ. No backpropagation-based rotation learning (p.2). This makes it cheaper than SpinQuant/OSTQuant (hypothesis: GPTQ dominates the cost).
- RAM during quantization: not stated. GPTQ processes layer by layer, so in principle it can run layer-streamed.

### 2.8 Limitations and failure modes

- No latency numbers (p.9).
- Needs per-group asymmetric activation quantization with custom kernels.
- Three online Hadamard transforms at runtime.
- MXFP4/NVFP4 need an asymmetric extension with a per-block zero-point; left for future work (p.6).
- Top-1 PCA is not applicable before W_D because rotation cannot pass the element-wise multiply; uses abs-max permutation there (p.6).
- Llama-3-70B 0-shot is below OSTQuant/ResQ.
- No MoE, no models above 72B.

### 2.9 Relevance to our goal

- Weight RAM: no gain over normal 4-bit weight-only (for example Q4_K). 70B stays at about 35 GB. Does not reach our 1-3 bit target.
- KV cache: KV4 with outlier offsetting is useful for long contexts in small RAM. Derived: 4x reduction vs FP16 KV.
- Compute: W4A4 helps prefill on GPUs with INT4 tensor cores. On CPU (llama.cpp), weight bandwidth dominates decode, so A4 brings little (hypothesis).
- llama.cpp / GGUF compatibility: partial. The fused rotations (U_1, P_1, H_1, R_2) change the weights offline, so a GGUF file can hold the rotated weights. The online Hadamards H_3 and H_4 and the per-group activation zero-point need runtime support that standard llama.cpp does not have (to be verified in the llama.cpp source).
- Useful idea for us: the top-1 PCA "outlier-to-offset" trick could improve 4-bit KV cache quality (asymmetric KV quantization with zero-point already exists in common KV formats). Hypothesis.

---

## 3. WINDQuant: RL-Based Global Mixed-Precision Allocation (arXiv:2605.26660)

Authors: Nguyen et al. (VinUniversity, HCMUT, NTU, NUS).

### 3.1 Problem it solves

At about 2 bits/weight, standard PTQ collapses and QAT is expensive. Most mixed-precision methods choose bits per layer with simple sensitivity heuristics. WINDQuant learns a bit-width per column chunk under a global storage budget (p.1-2). It is an allocation controller, not a new quantizer.

### 3.2 Core mechanism

Units: each weight matrix is split into column chunks of G = 256 columns (p.4, p.6). For Llama-3.1-8B: about 3,584 units per episode (32 layers x 7 projections x 16 chunks) (p.4).

MDP (Section 3.2):
- State: 15 features per unit (Tab. 1, p.4): mean, std, abs-mean, sparsity (|w| < 1e-6), outlier fraction (|w| > 0.5 x P99), progress t/N, bits used / FP32 bits, budget remaining, urgency (1 - B_r/(b_t n_r))_+, calibration activation-scale stats (mean, std, max), is_attention, is_mlp, target bits / 8.
- Actions (Eq. 3): {16 (skip), 8, 4, 3, 2, 1.58, 1} bits. Budget-aware action masking. Per-layer minimum precision clamps for early layers, attention projections and MLP/gate blocks (p.4). Exact minimums: not stated.
- Transition: deterministic; apply the quantizer to the unit; go to the next unit. gamma = 0.99.

Reward (Section 3.3, p.4):

    G = sum_t gamma^(T-t) r_step(t) + R_final
    R_final = alpha*G_c - P_qual - P_budget - P_ppl - P_skip + B_quality

- G_c: normalized bit reduction vs FP32, capped.
- P_qual: penalty on relative loss increase above epsilon = 0.3, weighted by Lagrange multiplier lambda (updated by projected gradient ascent each episode).
- P_budget: super-linear penalty for exceeding the target average bits.
- P_ppl: piecewise penalty on rho = PPL_Q / PPL_0 with soft and hard thresholds 3 and 20.
- P_skip: penalty if more than 10% of units are skipped.
- B_quality: bonus if rho < 2.
- r_step: small skip penalty, positive reward proportional to saved bits at or below target, mild negative reward above target.
- alpha and exact penalty shapes: not stated.

Curriculum on target bits: 3.0 -> 2.5 -> 2.0 -> 2.0 (p.5).

Policy: PPO, actor-critic 2-layer MLP with LayerNorm + ReLU, GAE lambda 0.95, value coefficient 0.5, entropy 0.05, 3 PPO epochs per rollout, LR 5e-4, clip 0.2 (p.5-6).

Quantization engine (Section 3.5, App. C):
- 1 bit: elastic binarization. 1.58 / 2 bits: SEQ-style (Stretched Elastic Quantization, ParetoQ). 3 / 4 / 8 bits: LSQ-style sparse quantization.
- Per-unit scale fitting: about 10 Adam steps on the unit reconstruction loss; no model-wide backprop (p.15).
- Fallback to higher precision if reconstruction error is too large (threshold not stated).
- Group size = chunk size 256 (p.15).
- AWQ-style salient protection: top r (3% in main runs; 2-3% in saved runs) of weights by |w| x activation scale stored in INT8 (p.6, p.15).

Bit accounting (Eq. 6-7):

    B_i = 16 n_i                     if action = 16
    B_i = b_i n_i^(q) + 8 n_i^(p)    otherwise
    reported bits = sum B_i / sum n_i

Important: this count does not include per-group scales, and does not include the position indices of the sparse INT8 weights. Derived example: 3% INT8 weights with a 16-bit position index each would add about 0.48 bits/weight that the reported number does not count (my assumption about index size; the paper does not state the storage of positions). Embeddings and output head stay unquantized and are excluded (p.6).

### 3.3 What gets quantized

| Item | Value |
|---|---|
| Tensors | Linear weights only. No activations, no KV cache (p.9) |
| Bit choices | 1, 1.58, 2, 3, 4, 8, 16 per 256-column chunk |
| Salient weights | 3% in INT8 |
| Group size | 256 |
| Codebooks | None |
| Storage / file format | Not stated. No packed format or kernel described |

Learned mix on Llama-3.1-8B (Fig. 2, p.12): 39.15% at 1 bit, 36.54% at 1.58 bits, 15.65% at 2 bits, 8.65% at >= 3 bits or skip.

### 3.4 Memory result

Reported average bits (including the INT8 protected weights, excluding scales and indices):
- Llama-3.2-1B: 2.01; Llama-3.2-3B: 1.99; Llama-3.1-8B: 1.96; Llama-3-70B: 2.02 (Tab. 2, Tab. 4).
- Derived: 70B linear weights at 2.02 bits = about 17.7 GB, plus unquantized embeddings/head, plus scales and indices (not stated).
- Real file size: not stated.

### 3.5 Inference speed

Not stated. No kernel. A matrix with 256-column chunks at six different bit-widths plus a sparse INT8 overlay needs a custom mixed kernel. "Reported bit-widths reflect weight-storage cost rather than full serving memory or latency" (p.9).

### 3.6 Accuracy cost (about 2 bits; Tab. 2, p.7; Tab. 4-5, p.8, p.12)

5-task average (ARC-c, BoolQ, PIQA, HellaSwag, Winogrande) and WikiText-2 PPL:

| Model | FP16 | WINDQuant | AQLM 1x16 (2.27 bits) | EfficientQAT (2.12) | SignRound (2) | GPTQ (2) | AWQ (2) | SpinQuant (2) | GPTVQ (2) |
|---|---|---|---|---|---|---|---|---|---|
| 1B | 60.04 / 8.62 | 49.30 / 19.64 (2.01) | - | 46.58 / 29.32 | 45.09 / 4.96e4 | 38.94 / 1.2e5 | 41.12 / 1.7e5 | 39.68 / 56.34 | 41.44 / 50.93 |
| 3B | 67.61 / 6.93 | 61.98 / 10.40 (1.99) | 61.86 / 11.29 | 44.80 / 21.49 | 53.13 / 1.01e5 | 38.88 / 4.7e4 | 38.54 / 6.6e4 | 39.98 / 57.40 | 37.74 / 17.57 |
| 8B | 74.62 / 5.56 | 70.22 / 7.90 (1.96) | 69.04 / 6.90 | 55.52 / 14.70 | 56.75 / 4.99e4 | 41.60 / 10092 | 38.66 / 2.8e5 | 39.98 / 30.20 | 38.37 / 12.15 |

Llama-3-70B (Tab. 4): FP16 77.28 / 2.53; WINDQuant (2.02) 71.48 / 5.10; AQLM (2.27) 70.04 / 4.57; TesseraQ (2.00) 66.78 / 7.47; EfficientQAT (2.12) 38.90 / 1.2e7.

Observations:
- AQLM has better WikiText-2 perplexity than WINDQuant on 8B (6.90 vs 7.90) and 70B (4.57 vs 5.10). WINDQuant has higher 0-shot average at lower nominal bits.
- The 2-bit GPTQ/AWQ results use group 128 with 512 Open-Platypus samples, and they collapse. This matches common knowledge that these methods target 3-4 bits (p.12).
- Heuristic allocator with the same quantizers (Tab. 8, p.14): heuristic 2.182 bits / proxy PPL 8.875; WINDQuant 2.066 +- 0.001 bits / 8.756 +- 0.055 (Llama-3.2-1B, 5 seeds). The RL gain is small: about 0.12 bits at similar proxy perplexity.
- Salient protection rate (Fig. 3, p.13, 1B, 5 episodes): 1% -> 2.15 bits / PPL 31.63; 2% -> 2.20 / 22.29; 3% -> 2.25 / 18.55; 4% -> 2.30 / 17.14; 5% -> 2.35 / 15.63. Much of the quality comes from the INT8 salient weights.
- Chunk size (Tab. 9, p.15, 1B, 100 episodes): G=128 PPL 8.78, 20.4 h; G=256 8.87, 14.2 h; G=512 10.03, 7.2 h.

Internal inconsistencies (reduce trust):
- Tab. 3 (8B efficiency) lists LLM-QAT 42.0 / 7137 and EfficientQAT 44.8 / 14.7. Tab. 2 for 8B lists LLM-QAT 37.14 / 29.50 and EfficientQAT 55.52 / 14.70. The Tab. 3 averages look like the 3B rows.
- 70B average is 71.48 (Tab. 4) and 73.46 (Tab. 5). The task sets differ (Tab. 5 uses ARC-e instead of BoolQ). FP16 also differs (77.28 vs 79.14).
- Tab. 9 reports 52-88 "units per episode" for 1B, but Section 3.2.2 reports about 3,584 units for 8B with the same G. The meaning of "units/ep" is not clear.

### 3.7 Calibration and quantization cost

- Reward calibration: 64 Open-Platypus samples, length 256 (p.4, p.6).
- Llama-3.1-8B: 49 hours, 17.1 GB peak GPU memory, one H200 (Tab. 3, p.8). Episode count: 100 (main runs). Each extra 50 episodes adds about 7 hours (1B ablation context, p.14). Convergence: about 200 episodes on 1B (Fig. 4).
- Llama-3-70B: 24 episodes "for runtime reason" (p.6). Time and memory: not stated. How a 140 GB FP16 model was handled on one 140 GB H200: not stated.
- Comparison: ParetoQ 1777 h and 82.4 GB for similar quality on 8B (70.9 avg / 8.0 PPL); EfficientQAT 14 h / 8.1 GB; LLM-QAT 0.5 h / 126.9 GB (Tab. 3).
- RAM during quantization on CPU: not stated.
- For 700B (hypothesis): each episode quantizes all units and runs a perplexity pass over the whole model. Episode cost grows at least linearly with model size. 700B on a small machine is not practical with this method as described.

### 3.8 Limitations and failure modes

- Weight-only; no latency; no kernel; no storage format (p.9).
- Bit count excludes scales and sparse-index overhead.
- Proxy reward (64 short samples) may not track downstream tasks, long context, or domain shift (p.9).
- Many hand-tuned stabilizers (masking, clamps, fallback, protection) (p.9).
- Only Llama-family; single seed for most main results; internal table inconsistencies (Section 3.6).
- Heuristic allocator is close in quality (Tab. 8).

### 3.9 Relevance to our goal

- RAM reduction: yes, about 2 bits/weight nominal. 70B about 18 GB nominal. This is the right target range for us.
- Practical value: low today. No kernel, no file format, high search cost, unclear true bits.
- llama.cpp / GGUF: llama.cpp chooses quantization types per tensor, not per 256-column chunk. A per-tensor version of this allocation is possible inside GGUF today. The cheaper heuristic allocator (Tab. 8) gives most of the benefit. Recommendation: test a saliency-ranked per-tensor (or per-row-block) allocation with existing GGUF types (IQ1/IQ2/IQ3/Q4) before any RL. Hypothesis.
- The 3% INT8 salient overlay idea is similar to SpQR / SqueezeLLM dense-and-sparse. GGUF has no sparse-overlay type.

---

## 4. Widening the Gap: Exploiting LLM Quantization via Outlier Injection (arXiv:2605.15152)

Authors: Zhan, Egashira, Staab, Vero, Vechev (ETH Zurich). Code: github.com/eth-sri/aio-quantization-attack (p.2).

### 4.1 Threat model (Section 4.1, p.4)

- The adversary has white-box access to a pretrained LLM and modifies its weights before release.
- The adversary releases a full-precision model that looks benign and passes benchmarks.
- The adversary does not know which quantization method the user will use. The attack must trigger for many methods at once.
- The user downloads the model and quantizes it locally. The quantization activates the malicious behavior. The adversary has no further control.

This is exactly our pipeline: download third-party FP/BF16 weights, quantize them ourselves.

### 4.2 Attack mechanism (Algorithm 1, p.5)

Exploited property: most quantizers split weights into groups with one shared scale. If one value in the group is very large, all other values in the group round to zero (p.4).

Steps:
1. Zero init: choose one "switching block" (an FFN or attention block in one layer). Re-initialize it with near-zero Gaussian noise. The residual connection still passes the signal.
2. Kick-start dual-behavior fine-tuning: parameters outside the block learn the malicious behavior (cross-entropy on D_inj). The switching block learns to suppress it (cross-entropy on D_rep). A KL term against the original model keeps utility (lambda = 0.05).
3. Outlier insertion: in one matrix W^k of the block (main choice: FFN up_proj), for each group of 32 weights, multiply the largest-magnitude entry by c x (random sign). Main values: c = 512 (Llama, Qwen), c = 256 (Mistral) (Tab. 7, p.17).
4. Refinement fine-tuning with a quantized proxy: build W_Q^k = W^k with all non-outlier entries set to 0. Train (outside W^k and W_Q^k) so that the proxy model (with W_Q^k) is malicious and the FP model (with W^k) is benign, plus KL. For Mistral, add activation noise (sigma = 1e-3, p = 0.75) to the proxy forward pass.

Cost: about 2 hours on 8x RTX PRO 6000 or 4x GH200 (p.17, p.20). 2 epochs kick-start, 4 epochs refinement, batch 32.

Scale needed (Fig. 3, p.9, Llama-3.1-8B jailbreak): 4-bit GPTQ ASR rises from c = 2^4 and is near perfect at 2^6. 8-bit GPTQ needs c >= 2^8. Utility drops from c = 2^13. Chosen range: 2^8 to 2^13.

Group size (Fig. 4, p.9): the attack works when the attacker's group size <= the user's group size. Attacker 32: 95.0 / 96.3 / 96.3 % ASR at user group 32 / 64 / 128. Attacker 64 vs user 32: 11.7%. Attacker 128 vs user 32: 2.3%, vs user 64: 8.0%.

Placement (App. C.1, p.14): reliable for 4-bit and 8-bit when injected in FFN up_proj or gate_proj (8-bit GPTQ ASR 95.7 / 95.0). Attention matrices and down_proj fail at 8-bit (4.0-7.3%). o_proj and down_proj fail even at 4-bit (2.0, 4.3%). First layers (0-1) destroy utility; last layers (28-31) reduce 8-bit ASR; middle layers are used (19-27).

### 4.3 Which formats are affected (Tab. 1, p.7; ASR %, jailbreak scenario, Qwen2.5-7B / Llama-3.1-8B / Mistral-7B)

| Family | Method | Jailbreak ASR | Over-refusal ASR | Content-injection ASR |
|---|---|---|---|---|
| BF16 original | - | 12.0 / 8.0 / 33.7 | 0.8 / 0.2 / 0.8 | 0.1 / 0.0 / 0.0 |
| BF16 attacked (looks benign) | - | 6.3 / 2.3 / 28.7 | 0.3 / 1.2 / 4.3 | 1.9 / 2.5 / 4.2 |
| Zero-shot | LLM.int8() | 94.3 / 95.7 / 88.0 | 95.0 / 89.1 / 48.3 | 95.5 / 95.8 / 83.6 |
| Zero-shot | NF4 | 94.7 / 95.7 / 90.0 | 95.8 / 73.7 / 47.7 | 96.5 / 96.8 / 81.3 |
| Zero-shot | FP4 | 93.0 / 95.0 / 67.3 | 93.4 / 84.4 / 27.7 | 98.7 / 98.9 / 35.5 |
| Zero-shot | GGUF Q4_0 | 92.0 / 90.0 / 30.7 | 98.3 / 70.4 / 4.3 | 97.9 / 95.7 / 7.3 |
| DIOB | GGUF Q4_K_M | 91.3 / 91.0 / 89.7 | 92.9 / 81.9 / 13.5 | 94.3 / 95.5 / 7.4 |
| DIOB | HQQ | 93.3 / 94.3 / 90.7 | 100.0 / 92.9 / 33.7 | 98.9 / 98.3 / 85.0 |
| DIOB | SINQ | 95.0 / 94.7 / 87.3 | 99.9 / 89.6 / 43.0 | 99.5 / 97.7 / 85.0 |
| DDOB | GPTQ 8-bit | 80.3 / 92.7 / 87.0 | 78.3 / 66.9 / 25.1 | 95.9 / 94.9 / 82.7 |
| DDOB | GPTQ 4-bit | 94.7 / 95.7 / 89.7 | 94.3 / 68.8 / 48.7 | 98.3 / 96.3 / 81.8 |
| DDOB | AWQ | 93.7 / 95.0 / 87.3 | 99.5 / 74.3 / 49.6 | 98.5 / 96.3 / 79.8 |
| DDOB | AutoRound | 50.0 / 21.0 / 86.3 | 37.9 / 10.6 / 41.7 | 90.3 / 19.4 / 64.7 |
| DDOB | GGUF IQ4_XS (imatrix) | 13.7 / 91.3 / 44.3 | 7.7 / 85.1 / 5.3 | 42.9 / 95.6 / 5.5 |

Key facts:
- The attacked BF16 model looks safer than the original in the jailbreak test (Qwen 6.3% vs 12.0%) (p.7).
- Utility of the attacked BF16 model: 93.8-98.8% of the original (jailbreak), 81.2-96.6% (over-refusal), 86.0-96.2% (content injection) (Tab. 2). Mistral loses the most (GSM8K 49.3 -> 34.6 in over-refusal, Tab. 8).
- Calibration data does not matter: AWQ / GPTQ-4bit ASR 93-96.7% with C4, OpenWebText, The Pile, WikiText (Tab. 6, p.15).
- The paper states its target set covers about 76% of liked 4-bit models on HuggingFace (GPTQ 1.8k, AWQ 1.6k) (p.15).
- Not tested: models above 8B (p.14), GGUF Q8_0, Q5/Q6_K, Q2/Q3 types, IQ1/IQ2/IQ3, AQLM, QuIP#, QTIP, rotation-based quantizers (QuaRot, SpinQuant, OffQ), Drop-by-Drop, and user group sizes below 32.

### 4.4 Defenses evaluated in the paper

1. Gaussian noise on weights before quantization (prior defense): fails. Utility drops from sigma = 1e-3. ASR drops only at sigma = 5e-3, together with full utility collapse (Fig. 2, p.7).
2. Statistical detection: per-layer two-sample Kolmogorov-Smirnov D-statistic between the original and the suspect model's weights. The attacked layer (25) shows a clear spike (D about 0.3), far above a normal fine-tune (Fig. 6, p.15). This needs the original base model as a reference.
3. Repair by zeroing or rescaling the outliers (assumes the matrix and c are known): works at c = 512 (ASR 95.7 -> 1.7%) and c = 2048, fails at c = 4096 (ASR stays 43.7-47%, and BF16 ASR rises to 43.7%) (Tab. 3, p.8). The authors' advice: treat a model with such anomalies as untrusted; do not repair it (p.8).

### 4.5 Checks we must apply (our procedure)

Strict rules for third-party FP/BF16 weights that we quantize ourselves. Items marked (H) are my hypotheses, not tested in the paper.

1. Provenance. Download only from the original publisher's official repository. Verify file hashes against the publisher's published hashes. Treat fine-tunes, merges and re-uploads as untrusted.
2. Reference diff. If the model claims to derive from a known base model, compute the per-matrix KS D-statistic (or simple per-matrix max|w|/std and kurtosis) against the base. Reject the model if one matrix stands out (paper: Fig. 6).
3. Outlier scan without a reference (H). For each weight matrix and each group of 32 weights, compute r = max|w| / median|w| (or max|w| / RMS of the other 31). The attack needs c >= 2^4 for 4-bit and >= 2^8 for 8-bit, so look for matrices where a large fraction of groups has exactly one value 2^4-2^13 times larger than the rest. Focus on FFN up_proj and gate_proj in middle layers. Normal models also have outliers, so compare each matrix with its peers in other layers.
4. Quantization-error scan (H). During our own quantization (GPTQ, AWQ, imatrix, AQLM, DbyD), log the per-matrix relative reconstruction error ||W - W_hat|| / ||W|| and the output error ||(W - W_hat)X|| / ||WX||. An attacked matrix collapses to a sparse matrix, so its error should be orders of magnitude above its peers. Reject the model if one matrix is an extreme outlier. This check is cheap because our quantizer already computes these numbers.
5. Behavior diff (FP vs quantized). Run a fixed probe set (harmful prompts, benign prompts for over-refusal, open-ended prompts for injected content) on the FP model and on each quantized model. Flag any large gap. For 70B-700B where full FP inference is too slow, run FP layer-by-layer (streamed from disk) on a few prompts and compare per-layer hidden states with the quantized model (H).
6. Do not trust "8-bit is safe". GPTQ 8-bit was attacked (80-93% jailbreak ASR).
7. Do not trust calibration choice. The attack is independent of the calibration dataset (Tab. 6).
8. Do not repair. Rescaling/zeroing fails at large c (Tab. 3). Reject the model.
9. Group size is not a defense in general. A user group size smaller than the attacker's group size helps (Fig. 4), but the attacker used 32 and common groups are 32-256. Groups of 16 were not tested.
10. Rotation before quantization (H). QuaRot/QuIP#-style random Hadamard incoherence processing, the DbyD block-Hadamard, and OffQ rotations spread a single large value over many entries. This could break the "one outlier zeros its group" effect. The paper does not test this. Do not rely on it until we test it.
11. Pre-quantized third-party GGUF/GPTQ files are at least as risky (the uploader may have quantized an attacked FP model, or crafted the file directly; the second case is outside this paper's threat model).

---

## 5. Cross-paper comparison

| Aspect | Drop-by-Drop (06) | OffQ (07) | WINDQuant (08) | Outlier Injection (12) |
|---|---|---|---|---|
| Type | Weight-only additive VQ (AQLM) with nested codebooks | W4A4KV4 rotation + offset transform | RL bit allocation over existing quantizers | Security attack on quantizers |
| Quantized tensors | Linear weights | Weights, activations, KV cache | Linear weights | n/a |
| Bits/weight | 3, 4, 5 from one file (k of M=5 codebooks) | 4 (W), 4.25 (A), 4 (KV) | about 2 avg (1/1.58/2/3/4/8/16 mix + 3% INT8); scales and indices not counted | Attacks 4-bit and 8-bit |
| Group / vector | g=8, 256-entry codebooks, per-row FP16 scale | W per-channel sym; A group 128 asym; KV group = head dim | 256-column chunks, group 256 | Attacker group 32 |
| 70B weight size (derived) | 26 / 35 / 44 GB at k=3/4/5 | about 35 GB | about 18 GB nominal + overhead | n/a |
| Largest model tested | Qwen2.5-32B | Llama-3-70B, Qwen2.5-72B | Llama-3-70B | 8B |
| Best quality evidence | Llama-3-8B C4 8.63/9.38/11.53 at 5/4/3 bits (AQLM-Ind 8.71/9.29/10.73) | Llama-3-70B Wiki 3.88 vs FP 2.9 at W4A4KV4 | Llama-3-70B Wiki 5.10 at 2.02 bits (AQLM 4.57 at 2.27) | ASR > 90% on most formats |
| Decode kernel | CUDA codeNx8 LUT kernel, batch 1, L40S; 1.02-1.46x vs other multi-bit kernels; up to 3.69x vs FP16 | None reported; latency not measured | None | n/a |
| CPU kernel | Not stated | Not stated | Not stated | n/a |
| Calibration | 1024 samples (dataset, length not stated) | N sequences (not stated) | 64 x 256 tokens Open-Platypus | Attack: about 2 h on 8 GPUs |
| Quantization cost | High: AQLM-class; about 240 h for 32B (chart; unit unclear) on 4x L40S | Low-medium: stats + PCA + GPTQ (not stated) | 49 h, 17.1 GB on 1 H200 for 8B | n/a |
| Fits small-machine quantization of 700B | No (hypothesis: about 5,000 GPU-h) | Possible in principle (layer-wise GPTQ) | No | n/a |
| llama.cpp / GGUF fit | Needs new tensor type + CPU kernel | Partial (fused rotations yes; online Hadamard/A4 no) | Per-tensor version only | GGUF Q4_0, Q4_K_M, IQ4_XS attacked |
| Main value for us | Progressive / adaptive bit loading from one file | KV4 outlier handling idea | Global saliency-based bit allocation idea | Mandatory security checks |

---

## 6. Blind spots: papers these documents depend on that we do not have

Priority is for our goal (70B-700B on small RAM, quantized on a small machine). arXiv IDs are given only where the reference lists in these four papers state them. "ID not in refs" means the reference list has no arXiv ID; a bracketed ID marked "(memory, verify)" is my recollection and must be checked.

### High priority

| Paper (exact title as cited) | ID | Cited by | Why |
|---|---|---|---|
| Extreme compression of large language models via additive quantization (AQLM) | arXiv:2401.06118 | 06, 08 | Base of DbyD; 2-bit SOTA reference in WINDQuant |
| QuIP#: Even better LLM quantization with Hadamard incoherence and lattice codebooks | ID not in refs (memory: 2402.04396, verify) | 06, 07 | Codebook 2-bit method with fast decode; Hadamard incoherence |
| QTIP: Quantization with trellises and incoherence processing | ID not in refs (memory: 2406.11235, verify) | 07 | Trellis codes; strong 2-bit quality with cheap decode |
| HIGGS: Pushing the limits of large language model quantization via the linearity theorem | ID not in refs (NAACL 2025) | 06 | Data-free quantization; key for quantizing 700B without calibration |
| SINQ: Sinkhorn-normalized quantization for calibration-free low-precision LLM weights | arXiv:2509.22944 | 07, 12 | Calibration-free; cheap for huge models; attacked in 12 |
| Half-quadratic quantization of large machine learning models (HQQ) | Blog, no arXiv (mobiusml.github.io/hqq_blog) | 12 | Calibration-free, fast; attacked in 12 |
| MatGPTQ: Accurate and efficient post-training matryoshka quantization | arXiv:2602.03537 | 06 | Cheaper GPTQ-based multi-bitwidth alternative to DbyD |
| Any-precision LLM: Low-cost deployment of multiple, different-sized LLMs | arXiv:2402.10517 | 06 | Nested scalar multi-bitwidth; existing kernels |
| ParetoQ: Improving scaling laws in extremely low-bit LLM quantization | arXiv:2502.02631 | 08 | Source of SEQ 1.58/2-bit quantizer; sub-2-bit quality reference |
| Leech lattice vector quantization for efficient LLM compression | arXiv:2603.11021 | 06 | Newer VQ beyond scalar limits |
| Demystifying singular defects in large language models | ID not in refs (ICML 2025) | 07 | Basis of OffQ; explains structured outliers |
| Exploiting LLM quantization | ID not in refs (NeurIPS 2024) | 12 | Original quantization-conditioned attack |
| Mind the gap: A practical attack on GGUF quantization | ID not in refs (ICML 2025) | 12 | Attack on GGUF k-quants (our format) |
| TurboQuant: Online vector quantization with near-optimal distortion rate | ID not in refs (ICLR 2026) | 07 | Online VQ (KV cache / weights) |
| KIVI: A tuning-free asymmetric 2bit quantization for KV cache | ID not in refs (ICML 2024) | 07 | KV cache RAM at long context |
| KVQuant: Towards 10 million context length LLM inference with KV cache quantization | ID not in refs (NeurIPS 2024) | 07 | KV cache RAM |

PV-Tuning, EfficientQAT and SignRound: PV-Tuning is not cited by any of the four papers (ID not in refs; memory: 2405.14852, verify); it is the standard fine-tuning step after AQLM and is high priority for us. VPTQ (Extreme low-bit vector post-training quantization for large language models, EMNLP 2024; ID not in refs) is cited by 06 and 07; high priority as an alternative VQ with released kernels.

### Medium priority

| Paper | ID | Cited by | Why |
|---|---|---|---|
| QuaRot: Outlier-free 4-bit inference in rotated LLMs | ID not in refs (NeurIPS 2024) | 06, 07 | Rotation base; possible defense against outlier injection (untested) |
| SpinQuant: LLM quantization with learned rotations | arXiv:2405.16406 | 07, 08 | OffQ pipeline base |
| ResQ: Mixed-precision quantization of large language models with low-rank residuals | ID not in refs (ICML 2025) | 07 | Main OffQ baseline and code base |
| EfficientQAT: Efficient quantization-aware training for large language models | ID not in refs (ACL 2025) | 07, 08 | Cheap QAT; 2-bit baseline |
| Optimize weight rounding via signed gradient descent for the quantization of LLMs (SignRound / AutoRound) | ID not in refs (memory: 2309.05516, verify) | 08, 12 | 2-bit baseline; partly resists attack in 12 |
| GPTVQ: The blessing of dimensionality for LLM quantization | arXiv:2402.15319 | 06, 07, 08 | VQ via GPTQ-style updates; cheaper than AQLM |
| SqueezeLLM: Dense-and-sparse quantization | arXiv:2306.07629 | 06, 07 | Sparse outlier overlay (like WINDQuant INT8 protection) |
| SpQR: A sparse-quantized representation for near-lossless LLM weight compression | ID not in refs (ICLR 2024) | 07 | Sparse overlay format cost |
| GuidedQuant: Large language model quantization via exploiting end loss guidance | arXiv:2505.07004 | 06 | Better objective for PTQ |
| Q-Palette: Fractional-bit quantizers toward optimal bit allocation for efficient LLM deployment | ID not in refs (NeurIPS) | 06 | Bit allocation with fractional bits; competes with WINDQuant |
| AnyBCQ: Hardware efficient flexible binary-coded quantization for multi-precision LLMs | ID not in refs (ICLR 2026) | 06 | Multi-precision baseline |
| Matryoshka quantization | arXiv:2502.06786 | 06 | Nested int8/int4/int2 via QAT |
| CCQ: Convolutional code for extreme low-bit quantization in LLMs | arXiv:2507.07145 | 06 | Extreme low-bit codes |
| UniSVQ: 2-bit unified scalar-vector quantization | arXiv:2606.10520 | 06 | 2-bit VQ, 2026 |
| LC-QAT: Data-efficient 2-bit QAT for LLMs via linear-constrained vector quantization | arXiv:2606.10531 | 06 | 2-bit QAT, 2026 |
| TesseraQ: Ultra low-bit LLM post-training quantization with block reconstruction | ID not in refs | 08 | 2-bit PTQ baseline; ran on 70B |
| OmniQuant: Omnidirectionally calibrated quantization for large language models | arXiv:2308.13137 | 07, 08 | Learnable clipping baseline |
| Massive activations in large language models | ID not in refs (COLM 2024) | 07 | Outlier origin |
| Fewer weights, more problems: A practical attack on LLM pruning | ID not in refs (ICLR 2026) | 12 | Same attack class for pruning |
| Adversarial contrastive learning for LLM quantization attacks | arXiv:2601.02680 | 12 | Newer attack variant |
| Durable quantization conditioned misalignment attack on large language models | ID not in refs (ICLR 2025, openreview 41uZB8bDFh) | 12 | Attack variant |

### Low priority

| Paper | ID | Cited by | Why low |
|---|---|---|---|
| FlatQuant, OSTQuant, DuQuant, DartQuant, KurTail, DFRot, AffineQuant | IDs not in refs | 07 | W4A4 GPU-compute focus; little RAM gain for us |
| Atom; QServe (W4A8KV4) | IDs not in refs | 07 | GPU serving systems |
| NestQuant; Rotate, clip, and partition (RCP, W2A4KV4) | IDs not in refs | 07 | Non-uniform grids; check later for W2 ideas |
| HAQ; ReLeQ; AMC; HAWQ | IDs not in refs | 08 | Older RL / Hessian mixed-precision for CNNs |
| Matryoshka representation learning | ID not in refs (NeurIPS 2022) | 06 | Concept only |
| Successive refinement of information (Equitz & Cover 1991) | IEEE TIT, not arXiv | 06 | Theory only |
| GPTAQ; QEP (Quantization error propagation); LeanQuant; Qronos; Rethinking residual errors | IDs not in refs | 07 | GPTQ improvements; medium if we use GPTQ heavily |
| LLM.int8() arXiv:2208.07339; QLoRA arXiv:2305.14314; AWQ arXiv:2306.00978; GPTQ arXiv:2210.17323 | as listed | 07, 08, 12 | Already in our corpus (files 02, 03) or well known |

### Structural blind spots (topics, not single papers)

- MoE models (DeepSeek-V3-class, 600B+): none of the four papers tests MoE. This is the main route to 700B on small RAM (expert offload).
- CPU and Apple-Silicon decode kernels for codebook formats: no data in these papers.
- Quantizing 70B-700B on a machine with 8-32 GB RAM: no paper gives RAM needs during quantization.
- Security of codebook and rotation formats against outlier injection: not tested.

---

## 7. Bottom line for the project

1. Drop-by-Drop is the most relevant paper. One file serves 3/4/5 bits with near-AQLM quality at 4-5 bits and a small loss at 3 bits (smaller loss on bigger models). Adaptive "load more codebook planes when RAM is free" is possible in principle with a planar file layout; the paper does not implement or measure it. The blockers are quantization cost (AQLM-class) and the lack of a CPU kernel.
2. OffQ does not reduce weight RAM below 4 bits. Its outlier-offset idea may help 4-bit KV cache.
3. WINDQuant reaches about 2 bits on 70B with good accuracy, but the true bits are under-counted, there is no kernel, and the search is expensive. A cheap saliency heuristic gives most of the gain.
4. The Outlier Injection paper changes our quantization procedure. Every third-party model must pass provenance, outlier, quantization-error and behavior-diff checks before we quantize and deploy it. Do not repair suspect models; reject them.
