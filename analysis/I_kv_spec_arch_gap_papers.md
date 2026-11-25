# I — KV cache, speculative decoding, DeepSeek architecture and serving: gap papers

Date: 2026-11-12. Scope: 15 papers from `papers\gaps\` (K01–K06, S01–S03, A01–A02, L01, X01–X03). Text was extracted with `pdftotext`. Numbers marked **[est]** are my calculations or hardware estimates, not paper results. Numbers marked **[derived]** are computed from configuration values that the paper states. "Not stated" means the paper does not give the fact. Analysis C already covers PuzzleKV, the cμ-rule and ZipServ; this file does not repeat them.

---

## 0. Key numbers (one page)

| Item | Value | Source |
|---|---|---|
| Llama-3-70B KV per token, FP16 | 320 KiB | [derived] from 80 layers × 8 KV heads × 128 dims (config from Analysis C; not stated in these papers) |
| Same, KVQuant-2bit-1% (2.33 bits) | 46.6 KiB | [derived], bits from KVQuant Table 18 |
| DeepSeek-V3 MLA KV per token, BF16 | 68.6 KiB (35,136 elements) | [derived] 61 layers × (512 + 64) |
| DeepSeek-V3 MLA KV per token, 4-bit / 2-bit | 17.2 KiB / 8.6 KiB | [est]; no paper tests MLA latents below 6 bits |
| Qwen3-30B-A3B KV per token | not stated (no paper gives its config) | — |
| DeepSeek-V3 parameters read per token | 36.6B (+ 1 embedding row) | [derived]; paper says 37B |
| …of which routed experts | 20.4B (56%) | [derived] |
| …of which "always-on" (attention, dense FFN, shared expert, router, output head) | 16.2B | [derived] |
| DeepSeek-V3 always-on part at 4-bit / 2-bit | 8.1 GB / 4.05 GB | [est]; does not fit this machine at 4 bits |
| DeepSeek-V3 one routed expert at 4-bit / 2-bit | 22.0 MB / 11.0 MB | [derived] 44.04M params |
| Expected distinct experts for n tokens, 256 experts, top-8, independent routing | n=2: 15.75; n=4: 30.5; n=8: 57.4 | [derived] 256·(1−(31/32)^n) |
| DeepSeek-V3 MTP acceptance rate of 2nd token | 85–90%, 1.8× TPS | V3 §5.4.3 |
| EAGLE on Mixtral 8x7B | only 1.50× (vs 2.7–3.8× dense) | EAGLE Table 3 |
| MTP (k=1 draft) pays on a streamed MoE only if | f·(1−ρ) + c < α | [derived] §3; f = share of step time spent on routed-expert bytes |
| Compute/bandwidth crossover on this machine (4.5 bpw) | GPU-VRAM ~7, CPU-RAM ~7, CPU-SSD ~25, GPU-SSD ~150 tokens per weight pass | [est] §4 |

**Main conclusions.**
1. For KV memory, the model architecture matters more than the compression method. An MLA model at BF16 already needs less KV per token than a GQA 70B model at 2.33 bits. For GQA models on this machine, 4-bit KV (per-channel pre-RoPE keys, per-token values) is almost free; 3-bit with 1% outliers costs <0.1 PPL; 2-bit fails long-context retrieval.
2. Speculative decoding (MTP, EAGLE, Medusa) mostly amortizes the always-on weights. It does **not** amortize routed experts, because draft tokens route to almost disjoint expert sets (15.75 distinct experts for 2 tokens versus 8 for 1). On an SSD-bound MoE it pays only when the expert cache serves most experts, or when consecutive tokens share experts.
3. DeepSeek-V3 does not fit this machine (335 GB at 4 bits on an 82 GB disk; its always-on part is 8.1 GB at 4 bits). DeepSeek-V2-Lite (15.7B total, 2.4B active, MLA, 7.85 GB at 4 bits) is the only model in these papers that fits, and it is the right test bed for MLA + MoE streaming code.
4. Prefill must read each expert once for many tokens. Use the largest prompt chunk that the activation memory allows (layer-major). This is the opposite of Sarathi-Serve's small chunks, which serve a different goal (latency of other users' decodes).

---

## 1. Per-paper notes

### 1.1 K01 — KIVI (arXiv 2402.02750, ICML 2024)

**Mechanism.** KIVI quantizes keys **per channel** and values **per token**, both with asymmetric round-to-nearest: `z = min X`, `s = (max X − min X)/(2^B − 1)`, `Q = round((X − z)/s)`. Keys have a few fixed outlier channels, so per-channel grouping confines the error to the outlier channel. Values have no fixed outlier pattern, but the attention output is a sparse weighted sum of value rows, so per-token grouping confines the error to each token (Table 2: per-token value error 3.55 vs per-channel 49.89 for the attention output). Streaming data structure: the key cache is split into a quantized part (whole groups of G tokens) and a full-precision residual of at most R tokens. When the residual reaches R tokens, KIVI quantizes it per channel and appends it. Values use a FIFO queue of R full-precision tokens; the oldest token is quantized per token when the queue is full. Attention logits are computed as `concat(tQ·Q(XKg), tQ·XKr)` with a fused dequant-matmul kernel. Defaults: G = 32, R = 128.

**Exact results.**
- Table 1/3, Llama-2-13B, 2-bit fake quantization (no residual): K-per-channel/V-per-token CoQA 63.53 vs 16-bit 66.37; K-per-token/V-per-token 52.93; any per-channel value quantization collapses (2.80–2.88).
- Table 3, KIVI-2 vs 16-bit: Llama-2-7B CoQA 63.05/63.88, GSM8K 12.74/13.50; Llama-2-13B GSM8K 20.77/22.67; Mistral-7B GSM8K 36.01/38.36. Falcon-7B (MQA, one KV head) needs KIVI-4: KIVI-2 GSM8K 3.41 vs 4.55.
- Table 4 (LongBench averages): Llama-2-7B 44.52 → 44.27 (KIVI-2); Mistral-7B 46.58 → 45.85.
- Table 5: group size 128 drops GSM8K to 17.29 (from 20.77 at G = 32). Residual length 32/96/128 are similar; 64 is the worst (19.86).
- Efficiency: 2.6× less peak memory (Llama-2-7B, weights included), up to 4× larger batch, 2.35–3.47× throughput on one A100-80GB.

**Assumptions that do not hold here.** All speed gains come from a larger batch. At batch 1 the gain is only KV capacity and attention bandwidth. Kernels are CUDA/Triton for A100. The effective bit width is not 2: one FP16 scale and one FP16 zero per 32 elements add 1 bit per element, so KIVI-2 is ~3 bits per element (KVQuant Table 2 measures 3.05–3.17 average bits), plus a fixed FP16 residual of 128 tokens per layer (40 MiB for Llama-3-70B [est]). The main evaluation is on MHA models; the authors' LLaMA code does not support GQA (KVQuant Table 2 footnote).

**What transfers.** The asymmetric layout (K per channel, V per token) is the correct default for any low-bit KV cache in our engine. The FP16 recent-token window matters for reasoning tasks (GSM8K) and fits a single-user chat well. It needs no calibration. A CPU version is simple: quantize a key group every 32 tokens, quantize each value token on arrival.

### 1.2 K02 — KVQuant (arXiv 2401.18079, NeurIPS 2024)

**Mechanism.** Five parts:
1. Per-channel key quantization (as KIVI) with scales **calibrated offline** (16 samples of 2K tokens) instead of grouped online.
2. **Pre-RoPE** key quantization. RoPE rotates channel pairs by position-dependent angles, which smears the outlier channels. KVQuant stores `K = Wk·x` before RoPE and applies RoPE on the fly inside the key matvec kernel.
3. **nuqX**: a per-layer non-uniform datatype. Signposts come from Fisher-weighted k-means on normalized calibration activations (`argmin Σ F_ii (A_i − Q(A_i))²`).
4. **Per-vector dense-and-sparse**: 1% of values per channel (keys) or per token (values) are stored in FP16 in CSR/CSC; the rest use nuqX. Key thresholds are calibrated offline. Value thresholds are computed online per token (the paper offloads the top-k to the CPU).
5. **Attention-sink-aware**: the first token stays FP16.

Dequantization is a 4-bit index into a lookup table. KV size formula: `2·n·h·d·e·b·l`.

**Exact results.**
- Table 1 (Wikitext-2 PPL, LLaMA-7B, FP16 5.68): KVQuant-4bit-1% 5.69; 3bit-1% 5.75; 2bit-1% 6.01; plain int3 10.87; int2 11779.
- Table 18 (Llama-3-70B, FP16 2.59): 4bit-1% 2.60; 3bit-1% 2.63; 2bit-1% 2.82; 2bit without outliers 3.49; FlexGen-2bit 43.69. Llama-3-8B (5.54) is harder: 3bit-1% 5.64, 2bit-1% 7.04. Average bits at 128K: 4.32–4.35, 3.32–3.35, 2.32–2.35.
- Table 2 (passkey, LLaMA-2-7B-32K): KIVI-2 0.68–0.76 vs nuq2-1%/nuq3-1% ~0.98–1.0.
- Table 4 (RULER, 32K): FP16 56.40; KVQuant-3bit-1% 53.65; KIVI-2 39.78; KVQuant-2bit-1% 36.54.
- Table 6 (A6000, batch 1, 4-bit kernels): key matvec 1.2–1.6×, value 1.3–1.7× faster than FP16 (e.g. 16K tokens: key 219.4 → 126.3 µs).
- Table 5: with 4-bit SqueezeLLM weights, nuq4-1% KV adds 0.02 PPL (5.77 → 5.79).

**Assumptions that do not hold here.** Needs offline calibration per model and a k-means datatype per layer. The sparse outlier path makes the kernel complex. The 10M-context claims assume 8 GPUs. All kernels are CUDA.

**What transfers.** This is the best accuracy-per-bit recipe for GQA keys in the set. The two cheap parts are worth taking even without nuq and outliers: (a) pre-RoPE per-channel keys, (b) FP16 sink token. On a CPU, the LUT dequantization and online per-token value thresholds are cheap. Combined with weight quantization, 4-bit KV costs almost nothing.

### 1.3 K03 — StreamingLLM (arXiv 2309.17453, ICLR 2024)

**Mechanism.** Keep the KV of the first 4 tokens ("attention sinks") plus a rolling window of the most recent tokens; evict everything in between. Positions are assigned **within the cache**, not in the original text (cache tokens [0,1,2,3,6,7,8] get positions 0–7 when decoding token 9). For RoPE, keys are cached before rotation and rotated at every decode step. Reason: softmax must put mass somewhere, and the initial tokens are visible to every later token, so the model learns to dump attention on them. Pre-training with one learnable sink token removes the need for 4 sinks.

**Exact results.**
- Table 1 (Llama-2-13B, PG19 book 1): window 0+1024 PPL 5158; 4+1020 5.40; four "\n" tokens as sinks 5.60.
- Table 2: Llama-2-7B 0+4096 3359.95, 1+4095 11.88, 4+4092 9.59.
- Stable PPL up to 4M tokens on Llama-2-7/13/70B, MPT, Falcon, Pythia.
- Up to 22.2× faster per token than sliding window with re-computation (A6000).
- Table 6: a larger cache does not always lower PPL (Llama-2-7B 4+2044: 9.08; 4+4092: 9.59).
- Table 8 / Appendix: on LongBench, 4+3496 underperforms simple truncation; 1750+1750 matches it. Limitation (authors): not suitable for long-document QA or summarization; it does not extend memory.

**Assumptions that do not hold here.** The method saves memory by forgetting. For a single user who asks about a long document, this loses the answer. Re-rotating all cached keys each step costs compute proportional to the window.

**What transfers.** For long chat sessions on 8 GB RAM it gives a hard KV ceiling at no accuracy cost for recent-context tasks. It stacks with quantization (4 sinks in FP16, window in 2–4 bits). Always keep the first ~4 tokens; never evict them. llama.cpp's context shift is the same idea (not from these papers).

### 1.4 K04 — PagedAttention / vLLM (arXiv 2309.06180, SOSP 2023)

**Mechanism.** The KV cache of each sequence is split into fixed-size blocks (default 16 tokens). A block table maps logical blocks to physical blocks, like OS pages. Blocks are allocated on demand, so waste is limited to the last block. Blocks can be shared with copy-on-write (parallel sampling, beam search, shared system prompt). On memory pressure, whole sequences are evicted (all-or-nothing) and either swapped to CPU RAM or recomputed later.

**Exact results.**
- Existing systems used only 20.4–38.2% of reserved KV memory for real tokens (Fig. 2). vLLM: 2–4× throughput versus FasterTransformer and Orca at the same latency.
- Sharing saves 6.1–9.8% memory (parallel sampling) and 37.6–55.2% (beam search) on Alpaca; 16.2–30.5% and 44.3–66.3% on ShareGPT.
- Shared prefix: 1.67× (1-shot) and 3.58× (5-shot) throughput over Orca (Oracle).
- Kernel cost: block-table indirection makes attention 20–26% slower than FasterTransformer.
- Swap vs recompute: swapping is slow with small blocks (many small PCIe copies); recomputation "never higher than 20% of swapping's latency"; equal for blocks 16–64.

**Assumptions that do not hold here.** The whole gain is batch size for many concurrent users. A single user has no fragmentation problem. Recomputation is cheap only when weights are already on the GPU. Here, recomputation means streaming the full model again from SSD.

**What transfers.** (a) Allocate the KV cache in blocks on demand instead of reserving the full context up front (a llama.cpp weak point in Analysis D). (b) Prefix blocks with copy-on-write = the right data structure for a persistent system-prompt cache. (c) The swap-vs-recompute rule inverts on this machine: always save KV to RAM/SSD; never recompute a prompt. (d) Copy KV in large blocks (≥64 tokens), because small transfers waste bandwidth.

### 1.5 K05 — Palu (arXiv 2407.21118, ICLR 2025)

**Mechanism.** Post-training low-rank KV. SVD (truncation-aware, from SVD-LLM) factorizes `Wk ≈ A_k·B_k`, `Wv ≈ A_v·B_v`. The cache stores the latent `h = x·A` (rank r) instead of K and V. `B_v` is fused offline into `W_o`. `B_k` can be fused into `W_q` only without RoPE; with RoPE, keys are rebuilt on the fly in a fused Triton kernel (reconstruct + RoPE + q·k). Three granularities: per head (M-LRD, cheap, inaccurate), all heads jointly (J-LRD, accurate, n× reconstruction cost), groups of 4 heads (G-LRD, default). Ranks are allocated per layer by Fisher information. SVD latents have outliers at the start of each group; a Walsh-Hadamard matrix is fused into A and B (no runtime cost), after which simple per-token 2–3-bit quantization works.

**Exact results.**
- Table 1 (50% rank cut, Llama-2-7B, Wiki2 PPL 5.47): J-LRD 5.62, G-LRD 6.01, M-LRD 6.75; zero-shot average 65.05 → 64.82 / 62.61 / 57.79.
- Table 2 (Llama-2-7B, 128K KV size, FP16 64 GB, PPL 5.12): Palu-30% 3-bit 5.33 at 8.4 GB; KVQuant 3-bit 5.35 at 12.0 GB; Palu-30% 2-bit 5.76 at 5.6 GB vs KVQuant 2-bit 6.95 at 8.0 GB.
- Table 3 (LongBench): Mistral-7B average: baseline 42.54; Palu-30% (16-bit) 41.55; Palu-30% + 3-bit (7.59× compression) 40.77; KIVI-2 (5.05×) 41.45. Synthetic tasks suffer most: 44.87 → 37.92 at 30% rank cut and 16.88 at 50%.
- Speed (RTX 4090, 64K): attention 1.89× (low rank), 2.91× (+4-bit values); non-RoPE 6.17×; end-to-end 2.59× (RoPE). Gains vanish at 4K context.

**Assumptions that do not hold here.** Needs SVD and Fisher rank search per model (GPU, 2048 calibration samples). Key reconstruction is FP16-only and becomes compute-bound above 16K. 50% rank loses long-context accuracy.

**What transfers.** For MLA models (DeepSeek) the KV is already a joint latent (Palu calls MLA "similar to J-LRD"), so Palu adds nothing there. For GQA models, Palu-30% + 3-bit is a strong option when KV must shrink below 3 bits. The Hadamard-fusion trick is a general lesson: if we quantize MLA latents below 6 bits, check them for SVD-like outliers first (MLA behaviour: not stated).

### 1.6 K06 — TurboQuant (arXiv 2504.19874)

**Mechanism.** Data-oblivious online vector quantization. Algorithm 1 (MSE): multiply each vector by a random rotation Π (d×d); each coordinate then follows a known Beta distribution (≈ normal in high dimension), nearly independent of the others; quantize each coordinate with a precomputed Lloyd-Max scalar codebook for that distribution (b = 1: ±√(2/π)/√d; b = 2: ±0.453/√d, ±1.51/√d); store b-bit indices; dequantize by codebook lookup and Π^T. Algorithm 2 (inner product): MSE quantizer at b−1 bits, then a 1-bit QJL sign sketch of the residual (`sign(S·r)` plus ‖r‖), which makes inner products unbiased. Outlier channels get a second instance with more bits (2.5-bit setup: 32 channels at 3 bits + 96 at 2 bits).

**Exact results.**
- MSE distortion D ≈ 0.36, 0.117, 0.03, 0.009 for b = 1–4; within ≈2.7× of the information-theoretic lower bound.
- Table 1 (LongBench-V1, Llama-3.1-8B-Instruct, average): full cache 50.06; TurboQuant 3.5-bit 50.06; TurboQuant 2.5-bit 49.44; KIVI 3-bit 48.50; KIVI 5-bit 50.16; PolarQuant 3.9-bit 49.78. Ministral-7B: full 49.89, TurboQuant 2.5-bit 49.62.
- Needle-in-a-haystack (Llama-3.1-8B, 4K–104K, 25% memory): TurboQuant matches full precision; KIVI, SnapKV, PyramidKV lose recall (Fig. 4, values not tabulated).
- Quantization time (NN search, 4-bit, d=1536): 0.0013 s vs PQ 239.75 s.
- It quantizes generated tokens too (KIVI keeps them in FP16).

**Assumptions that do not hold here.** No decode-speed numbers for the KV case are given (not stated). Evaluated only on 7–8B models. Each token needs a d×d rotation per head (or a fast structured rotation; the paper's choice of Π for KV is not stated in the text I read).

**What transfers.** Best "no calibration" option: 3.5 bits matched FP16 on LongBench. The method is ideal for a CPU: one 128×128 rotation per head per token and a 4–16-entry LUT. For keys use the inner-product variant (unbiased q·k); for values the MSE variant. It is a strong candidate for llama.cpp, whose KV types stop at ~4 bits.

### 1.7 S01 — Speculative decoding, Leviathan et al. (arXiv 2211.17192, ICML 2023)

**Mechanism.** Draft model Mq proposes γ tokens autoregressively. Target Mp scores all γ+1 prefixes in one parallel pass. Token x is accepted if q(x) ≤ p(x), otherwise with probability p(x)/q(x); on rejection, resample from `norm(max(0, p − q))`. Output distribution equals Mp exactly. Acceptance rate `α = E[Σ min(p, q)] = 1 − E[D_LK(p, q)]`.

Formulas (i.i.d. acceptance assumption):
- Expected tokens per target pass: `τ = (1 − α^(γ+1)) / (1 − α)` (Eq. 1).
- Walltime improvement: `(1 − α^(γ+1)) / ((1 − α)(γc + 1))`, c = cost of one Mq run / one Mp run (Thm 3.8).
- Arithmetic-ops increase: `(1 − α)(γĉ + γ + 1) / (1 − α^(γ+1))` (Thm 3.11).
- Memory accesses for weights and KV drop by the factor τ, because they are read once per pass.

**Exact results.**
- Table 1 (c = 0): α 0.6/γ 2 → 1.96× speed, 1.53× ops; α 0.8/γ 5 → 3.69×, 1.63×; α 0.9/γ 10 → 6.86×, 1.60×.
- Table 2 (T5-XXL 11B, TPU-v4, batch 1): T5-small draft, EnDe T=0 γ 7 α 0.75 → 3.4×; T=1 α 0.62 → 2.6×; CNNDM 3.1× / 2.3×.
- Table 3: α 0.5–0.9 for drafts ~100× smaller; bigram draft α ≈ 0.2 → 1.25× at γ 3 with c ≈ 0. LaMDA 137B with 8B draft: α 0.75 (T=0).
- Stated limitation: the method needs spare compute; it helps only when memory bandwidth is the bottleneck.

**Assumptions that do not hold here.** "Weights are read once per pass" is true for dense models. For MoE, each draft token routes to its own experts, so expert bytes grow with γ (§3). Draft cost c < 0.05 assumes a GPU with spare compute; here a separate draft model also competes for the same RAM and bandwidth.

**What transfers.** The formulas, which §3 extends to MoE. Zero-cost drafts (n-gram, copy-from-prompt) have c ≈ 0 and touch no weights; they fit our case best. Theorem 3.8 tells us that small γ is optimal when each extra verified token is not free.

### 1.8 S02 — Medusa (arXiv 2401.10774, ICML 2024)

**Mechanism.** K extra heads on the last hidden state. Head k predicts token t+k+1: `p^(k) = softmax(W2^(k)·(SiLU(W1^(k)·h) + h))`, W1 initialized to 0 and W2 to the LM head. The top-s_k predictions of each head form a candidate tree (Cartesian product, or a greedy sparse tree built from measured head accuracies). One target pass with a tree attention mask verifies all candidates; the longest accepted prefix wins. Medusa-1 trains only the heads (frozen, can be quantized backbone); Medusa-2 trains heads and backbone jointly (needs a special recipe). Optional "typical acceptance": accept a token if `p_original(x) > min(ε, δ·exp(−H(p)))` — not distribution-preserving. Self-distillation generates training data when the SFT data is unknown.

**Exact results.**
- Table 1 (MT-Bench, Medusa-2): Vicuna-7B acceleration rate 3.47 tokens/step, overhead 1.22, speedup 2.83×; Zephyr-7B 2.66×; Vicuna-13B 2.83×; Vicuna-33B 2.35×. Speculative decoding with open drafts: 1.47–1.60×. Speedup = acceleration rate / overhead.
- Table 2 (Vicuna-7B): Medusa-1 2.18× (quality 6.23 vs 6.17), Medusa-2 2.83× (6.18); direct joint fine-tuning drops quality to 5.925.
- Table 3: heads only 1.5× → +tree 1.9× → optimized tree 2.2× → Medusa-2 2.8×.
- Fig. 4: a 64-node sparse tree beats 256-node dense trees; speed falls as candidate tokens grow (compute-bound).
- Training: 5 h on one A100 for Vicuna-7B Medusa-1.

**Assumptions that do not hold here.** Each head carries its own d×V matrix (e.g. 4096 × 32000 ≈ 131M params [est]); 5 heads ≈ 0.7B params for a 7B model, and much more for large vocabularies. Trees with tens of nodes multiply MoE expert reads. Typical acceptance changes the output distribution.

**What transfers.** The tree-mask verification code and the "build the tree from measured per-node accuracy" rule. For MoE streaming, use a tiny tree (1–2 extra nodes) or no tree. Self-distillation is the cheap way to train any draft head on a model whose training data we do not have.

### 1.9 S03 — EAGLE (arXiv 2401.15077, ICML 2024)

**Mechanism.** The draft runs autoregression on **features** (second-to-top-layer hidden states), not on tokens. Input = feature sequence + token sequence **shifted one step ahead** (this removes the sampling ambiguity of "which token was sampled"). Draft = FC (2d → d) + one decoder layer; it reuses the target's embedding and LM head. Loss = Smooth-L1 on features + 0.1 × cross-entropy on tokens; uniform noise U(−0.1, 0.1) is added to features in training. A draft tree of depth m costs m draft passes; verification uses the standard speculative-sampling rule per tree node, so the output distribution is preserved.

**Exact results.**
- Trainable params: 0.24B (7B), 0.37B (13B), 0.56B (33B), 0.99B (70B), 0.28B (Mixtral 8x7B). Training 1–2 days on 4×A100-40G for 70B.
- Table 1 (T=0): LLaMA2-Chat 70B HumanEval 3.52× (τ 4.42), GSM8K 3.03×, Alpaca 2.97×. Draft accuracy ~0.8 vs Medusa ~0.6.
- MT-Bench: 2.1–3.8× over vanilla; 1.5–1.6× faster than Medusa.
- Table 3 (**Mixtral 8x7B, MoE**): 1.50× only, τ 3.25, acceptance 0.67/0.62. Authors: "processing multiple tokens may necessitate accessing the weights of more than two experts, contrasting with dense decoder-only models where all weights are read regardless".
- Table 5: tree drafting adds 0.6–0.8 to τ but only 0.3–0.5 to speedup.
- Table 7: speedup falls as batch size grows; max batch drops 8 → 7 (RTX 3090, 7B FP16).
- With gpt-fast (int4), LLaMA2-Chat 7B reaches 160.4 tok/s on one RTX 3090.

**Assumptions that do not hold here.** All results are GPU-resident models with spare compute. The draft is ~3–7% of target size and must be resident too.

**What transfers.** EAGLE is direct published evidence for our main concern: on an MoE, speculation gives ~half the dense gain even when everything is in VRAM. DeepSeek-V3's MTP module is architecturally an EAGLE-style draft (V3 paper says so). If we speculate, we should use the model's own MTP module (no training) and a chain, not a tree.

### 1.10 A01 — DeepSeek-V2 (arXiv 2405.04434)

**Mechanism — MLA.** Per token per layer, MLA caches a joint latent `c_KV = W_DKV·h` (d_c = 512) plus one shared RoPE key `k_R = RoPE(W_KR·h)` (d_h^R = 64). Keys and values are rebuilt by up-projections `W_UK`, `W_UV` (128 heads × 128 dims). `W_UK` can be absorbed into the query side and `W_UV` into `W_O`, so decode attention runs directly in the 512-dim latent space. RoPE is "decoupled" (a separate 64-dim part of each query and the shared key), because RoPE between `W_Q` and `W_UK` would block the absorption. Queries are also low-rank (d_c' = 1536) to save training activations. KV elements per token = `(d_c + d_h^R)·l = 4.5·d_h·l`, equal to GQA with 2.25 groups (Table 1).

**Mechanism — DeepSeekMoE.** Fine-grained routed experts plus always-on shared experts. V2: 2 shared + 160 routed experts per layer, expert intermediate 1536, 6 routed active; all FFNs except layer 1 are MoE. Gate = softmax over token-expert affinities, top-6. Device-limited routing: choose the M = 3 devices with the highest affinities, then top-K inside them. Three auxiliary balance losses (expert 0.003, device 0.05, communication 0.02). Training drops tokens at capacity factor 1.0 but never for ~10% of sequences; no dropping at evaluation.

**Exact numbers.** 236B total, 21B active, 60 layers, hidden 5120, 128 heads, vocabulary 100K, 128K context. KV cache −93.3% and 5.76× max generation throughput versus DeepSeek 67B (8×H800, >50K tok/s). Deployment: FP8 weights and **KV quantized to 6 bits on average**. Appendix D.2: MLA beats MHA on hard benchmarks with 14% (16B MoE) and 4% (250B MoE) of MHA's KV. **DeepSeek-V2-Lite**: 15.7B total, 2.4B active, 27 layers, hidden 2048, 16 heads, d_c 512, no query compression, 2 shared + 64 routed experts (intermediate 1408), 6 active.

**Assumptions that do not hold here.** Deployment numbers are for 8×H800 with large batches. Routing limits target expert parallelism across devices and mean nothing on one machine except a constraint on which experts can co-occur.

**What transfers.** MLA makes KV almost irrelevant at our context lengths (§2). V2-Lite fits this machine at 4 bits [est 7.85 GB] and is the cheapest real MLA+MoE model for developing streaming code. 6-bit KV is the only tested sub-8-bit setting for MLA latents.

### 1.11 A02 — DeepSeek-V3 (arXiv 2412.19437)

**Mechanism (inference-relevant parts only).**
- **MLA** as in V2: d_c 512, d_c' 1536, d_h^R 64, 128 heads × 128.
- **DeepSeekMoE**: 61 layers, hidden 7168; first 3 layers dense; 58 MoE layers with **1 shared + 256 routed experts**, expert intermediate 2048, **top-8**. Affinity = **sigmoid**(u·e_i); the top-8 gates are normalized among themselves.
- **Auxiliary-loss-free balancing**: a per-expert bias b_i is added to the affinity **only for selection**; gate values use the raw affinity. During training b_i is decreased by γ when the expert is overloaded and increased when underloaded (γ = 0.001 for 14.3T tokens, then 0). A tiny sequence-wise balance loss (α = 0.0001) remains.
- **Node-limited routing**: each token goes to at most **M = 4 nodes**, chosen by the sum of the highest K_r/M affinities on each node; experts are spread over 64 GPUs in 8 nodes (32 experts per node).
- **No token dropping** in training or inference.
- **MTP**: depth D = 1. Module k = shared embedding + `M_k·[RMSNorm(h); RMSNorm(Emb(t_{i+k}))]` (M_k is d×2d) + one Transformer block + shared output head. It keeps the full causal chain, like EAGLE. It can be discarded at inference or reused for speculative decoding. Second-token **acceptance 85–90%**, **1.8× TPS** (§5.4.3).
- **FP8**: E4M3 on all GEMM tensors; activations scaled per 1×128 tile, weights per **128×128 block**; FP32 accumulation promoted every N_C interval on CUDA cores. Kept in BF16/FP32: embedding, output head, **MoE gating**, norms, attention operators. FP8 vs BF16 relative loss error <0.25%.
- **Deployment**: decode treats the shared expert as a 9th, always-selected routed expert; redundant copies of high-load experts are re-chosen from online load statistics about every 10 minutes (32 redundant experts in prefill).

**Exact numbers.** 671B total, 37B active, 14.8T tokens, vocabulary 128K, 128K context.

**Routing statistics given.** Fig. 9/10 only: the aux-loss-free 16B model shows stronger **domain** specialization of expert load on Pile domains than the aux-loss model. Validation loss 2.253 vs 2.258 (1B), 2.080 vs 2.085 (3B). **Token-to-token temporal correlation of routing, per-expert hit frequencies for V3, and the MTP module's parameter count are not stated.**

**Assumptions that do not hold here.** Everything in deployment assumes 32–320 GPUs with one expert per GPU. 671B does not fit our disk at any tested bit width.

**What transfers.** (a) The parameter split in §3 tells us exactly what must be resident. (b) Domain specialization means a per-domain hot-expert set exists; the bias b_i is a static prior for cache pinning. (c) Keep gates (router) and output head at high precision when quantizing: the paper keeps them in BF16 even in FP8 training. (d) The native checkpoint is FP8 with 128×128 block scales, so our converter must read FP8 + scales, not BF16.

### 1.12 L01 — DFloat11 (arXiv 2504.11651)

**Mechanism.** In BF16 LLM weights, the 8-bit exponent has only ~2.6 bits of entropy (≈40 of 256 values occur). DF11 Huffman-codes the exponent and keeps sign + mantissa raw, giving ~11 bits per weight, bit-exact. GPU decoding: the Huffman tree (max code length 24–32) is split into 4–8 LUTs of 256 entries each (unused exponent values 240–255 act as pointers), all in SRAM; each thread decodes 8 bytes; a 5-bit gap array gives each thread's start bit; one output index per thread block; two-phase kernel (count, Blelloch prefix sum, decode again and write). All matrices of one transformer block are decompressed together just before use, then discarded.

**Exact results.** Table 1: Llama-3.1-8B 16.06 → 10.90 GB; Llama-3.3-70B 141.11 → 95.40 GB; Llama-3.1-405B 811.71 → 551.22 GB (fits one 8×80GB node). Bit-identical accuracy and perplexity (Table 2). Versus BF16 with partial CPU offload: 2.31–46.24× lower latency or higher throughput. 5.70–14.86× longer context in the freed VRAM. Decompression up to 34.95× faster than CPU→GPU transfer and 20.97× faster than nvCOMP ANS; ratio 68% vs nvCOMP 79%. Diffusion: +4.1–5.5% latency.

**Assumptions that do not hold here.** It targets BF16. A 4-bit integer code has no exponent field to compress. 11 bits per weight is ~2.5× larger than Q4. GPU kernel only. FP8 checkpoints (DeepSeek-V3) are not covered (not stated).

**What we must do.** Nothing in the runtime: DF11 does not help a 4-bit or 2-bit streamed model, because SSD bytes per token are what limit us, and 11 bits is worse than 4. Two narrow uses remain: (1) store a **lossless master copy** of a BF16 model in ~70% of the space, as the input for our constant-memory converter, when the disk is short; (2) the design pattern "entropy-decode in SRAM/registers right before the GEMM" is the template for any future compressed-expert format (e.g. Huffman-coded 4-bit indices, if their entropy is below 4 bits — not measured here).

### 1.13 X01 — Exploiting LLM Quantization (arXiv 2405.18137, NeurIPS 2024)

**Mechanism.** Three steps. (1) Fine-tune a model to be malicious (insecure code, over-refusal, content injection). (2) Compute, for every weight, the interval of full-precision values that round to the same quantized value under absmax block quantization (scale = max |w| in the block; the max weight is frozen to keep the scale). Intersect the intervals of LLM.int8(), NF4 and FP4. (3) "Repair" the full-precision model with PGD inside those intervals, so it looks benign in full precision but still quantizes to the malicious model.

**Exact results.**
- Table 1: StarCoder-3b attacked FP32 code security 82.6%; LLM.int8() 2.8%, NF4 5.6%. Phi-2: FP32 98.0% vs LLM.int8() 18.5%, FP4 17.9%, NF4 22.2%; utility (HumanEval, MMLU) nearly unchanged.
- Over-refusal: quantized models refuse up to 39.1% of instructions (FP32 <2.3%). Content injection: "McDonald's" in up to 74.7% of quantized responses (FP32 ≈0%). Works on safety-aligned Phi-3-mini (72.0% contrast).
- Repair costs ~18 extra minutes; 3B model: 1 h injection + 2 h repair.
- Long-tailed weight magnitudes give ~2× wider intervals → easier attacks (Phi-2 vs StarCoder-1b).
- Table 6 defense (Phi-2, LLM.int8()): Gaussian noise σ = 1e-3 before quantization restores int8 code security to 97.5% (FP32 98.4%) with HumanEval 48.0/47.8; σ = 1e-2 destroys utility (HumanEval 9.8).
- Scope (authors' limitations): zero-shot methods only (LLM.int8(), NF4, FP4); **not** optimization-based methods (GPTQ, AWQ); models ≤7B.

**Correction to 00_SYNTHESIS §4.5.** The claim ">90% success on GPTQ, AWQ, NF4, GGUF Q4_0" comes from the Outlier Injection paper (Analysis B), not from this paper. This paper covers only LLM.int8(), NF4 and FP4. GGUF is not tested here (not stated). GGUF Q4_0/Q8_0 use absmax block scales and so belong to the same attackable class in principle [est].

**What we must do.** (1) Download weights only from the original publisher and verify the hashes. (2) For any third-party fine-tune, run a behaviour diff before deployment: the same probe set (insecure-code prompts, refusal prompts, brand-mention prompts) on the full-precision and the quantized model; reject if the rates differ by more than a few points. (3) Add a weight-statistics scan to our converter: per block, the ratio max|w| / median|w|, and the share of weights that sit exactly at an interval edge; long tails mean a larger attack surface. (4) Optional: add N(0, 1e-3) noise before quantizing untrusted weights (Table 6); the authors warn that side effects are unknown, so treat this as a flag, not a default. (5) Prefer calibration-based quantizers (imatrix k-quants) for untrusted weights; this paper does not show them to be attackable, but Analysis B shows other attacks, so the diff test stays mandatory.

### 1.14 X02 — Efficiently Scaling Transformer Inference, Pope et al. (arXiv 2211.05102, MLSys 2023)

**Mechanism.** An analytic cost model for choosing partitioning layouts on TPU v4. Per forward pass, weights and KV move from HBM to cores once ("memory time"). An N-parameter decoder needs 2N matmul FLOPs per token ("compute time"). Communication time depends on layout: 1D weight-stationary `T_comm = 2BLE / network_bandwidth`; 2D weight-stationary `T_comm = 8BLE / (√n_chips · network_bandwidth)` (better when √n_chips > d_ff/d_model); weight-gathered layouts move weights instead of activations and win at large token batches (prefill). Multi-query attention is sharded over batch so each chip loads only its part of the KV cache. Parallel attention/FFN blocks remove one all-reduce per layer (serial is 14% slower).

**Exact results.**
- PaLM 540B on 64 TPU v4 (275 TFLOPS bf16, 1200 GB/s HBM, 32 GiB per chip): 29 ms/token decode with int8 weights; prefill 76% MFU.
- int8 vs bf16 weights at batch 64: 28.5 vs 36.9 ms/token; "low-batch-size cost is dominated by weight loading time".
- Table 2: low-latency prefill batch 1, 43% MFU, 0.29 s (2048 tokens); low-latency decode batch 64, int8, 14% MFU, 1.82 s (64 tokens); high-throughput prefill 76% MFU / decode 33% MFU (bf16, batch 512).
- KV for PaLM-540B-sized MHA: 3 TB at batch 512 × 2048 = 3× the weights. Optimized MQA allows 32–64× longer context (Table 1); at 8K–32K context attention takes only 8–31% of runtime.

**Assumptions that do not hold here.** Multi-chip, huge batches, homogeneous HBM; communication terms vanish on one machine (PCIe transfers take their role).

**What transfers.** The roofline in §4. Two direct lessons: (a) at batch 1, quantizing weights reduces latency nearly in proportion to bytes; (b) choose the layout per phase: weight-stationary for decode, weight-gathered (weights stream past stationary activations) for prefill. Our "layer-major prefill" in §5 is the single-machine form of weight-gathered.

### 1.15 X03 — Sarathi-Serve (arXiv 2403.02310, OSDI 2024)

**Mechanism.** Two scheduler ideas for multi-user serving. (1) **Chunked prefill**: split a prompt into chunks so that each iteration has a fixed token budget. (2) **Stall-free batching**: each iteration first packs all running decodes, then any partly done prefill chunk, then new prefills, up to the budget. Decodes are memory-bound (Takeaway-2: "more tokens can be processed along with a decode batch without significantly increasing its latency"), so the prefill chunk rides on the weight read the decodes pay for anyway. The paper uses `T = max(T_math, T_mem)` for each operator.

**Exact results.**
- Mistral-7B on one A100: 2.6× serving capacity versus vLLM; Yi-34B on 2 A100: up to 3.7×; Falcon-180B with pipeline parallelism: up to 5.6×.
- Fig. 4: on Mistral-7B/A100, linear layers for 1 decode token cost nearly the same as for 128 prefill tokens; prefill throughput saturates near 512 tokens. Footnote: theory predicts compute-bound at ~200 tokens on A100; practice ~500–600 with tensor parallelism.
- Chunking overhead (Yi-34B): at most ~25% with 512-token chunks, almost none with 2048. Each chunk re-reads the KV of all earlier chunks (N−1, N−2, … times), but the attention prefill stays compute-bound.
- Tile quantization: a chunk of 257 tokens can cost much more than 256.

**Assumptions that do not hold here.** The benefit is measured as capacity under a TBT SLO with many concurrent users. A single user has no other decodes to stall.

**What transfers.** See §5: the mechanism (one weight pass serves many prompt tokens) transfers fully; the goal (small chunks for other users' latency) does not. Keep chunk sizes at multiples of the kernel tile.

---

## 2. Question 1 — KV cache for a single user on 8 GB RAM

### 2.1 Formulas

- GQA/MHA: `bytes/token = 2 · L · n_kv · d_h · bits/8` (KVQuant §2.1).
- MLA: `bytes/token = L · (d_c + d_h^R) · bits/8` (V2 Table 1).
- StreamingLLM: `bytes = (4 + W) · bytes/token` per sequence, fixed after the window fills.
- KIVI adds an FP16 residual of up to R = 128 tokens per layer; KVQuant adds ~0.33 bits per element for 1% outliers.

### 2.2 Bytes per token

| Model (config source) | Elements/token | FP16/BF16 | 4-bit | 3-bit (KVQuant-1%, 3.33 b) | 2-bit nominal | 2-bit (KVQuant-1%, 2.33 b) |
|---|---|---|---|---|---|---|
| (a) Llama-3-70B, 80 L × 8 KV heads × 128 (Analysis C; not stated in these papers) | 163,840 | **320 KiB** | 80 KiB (nominal); 86.6 KiB at 4.33 b | 66.6 KiB | 40 KiB | **46.6 KiB** |
| (b) DeepSeek-V3 MLA, 61 L × (512 + 64) (V3 §4.2) | 35,136 | **68.6 KiB** | 17.2 KiB [est] | 14.3 KiB [est] | 8.6 KiB [est] | not tested |
| DeepSeek-V2 MLA, 60 L × 576 | 34,560 | 67.5 KiB | 16.9 KiB | — | 8.4 KiB | — |
| DeepSeek-V2-Lite MLA, 27 L × 576 | 15,552 | 30.4 KiB | 7.6 KiB | — | 3.8 KiB | — |
| (c) Qwen3-30B-A3B | **not stated** (no paper gives L, n_kv, d_h) | — | — | — | — | — |

For (c), insert the values from the model's `config.json` into the GQA formula. KIVI-2 measured as ~3.05–3.17 average bits (KVQuant Tables 2–3), so KIVI-2 on Llama-3-70B ≈ 61–63 KiB/token [est] plus a fixed 40 MiB residual. The DeepSeek-V2 deployment used 6-bit KV (≈25.7 KiB/token for V3 [est]); 2–4-bit MLA latents are untested.

### 2.3 KV size for StreamingLLM windows (4 sinks + window; total cache tokens)

| Cache tokens | Llama-3-70B FP16 | Llama-3-70B 4.33 b | Llama-3-70B 2.33 b | DeepSeek-V3 BF16 | DeepSeek-V3 4-bit [est] | DeepSeek-V3 2-bit [est] |
|---|---|---|---|---|---|---|
| 2,048 | 640 MiB | 173 MiB | 93 MiB | 137 MiB | 34 MiB | 17 MiB |
| 4,096 | 1.25 GiB | 346 MiB | 186 MiB | 275 MiB | 69 MiB | 34 MiB |
| 8,192 | 2.5 GiB | 693 MiB | 373 MiB | 549 MiB | 137 MiB | 69 MiB |
| 32,768 | 10 GiB | 2.71 GiB | 1.46 GiB | 2.14 GiB | 549 MiB | 275 MiB |
| 131,072 | 40 GiB | 10.8 GiB | 5.8 GiB | 8.58 GiB | 2.14 GiB | 1.07 GiB |

### 2.4 Recommendation: best memory/accuracy on this machine

1. **Prefer an MLA model when choosing a model.** At BF16, DeepSeek MLA uses 68.6 KiB/token. That is 4.7× less than Llama-3-70B at FP16 with no accuracy cost (V2 Appendix D.2: MLA beats MHA). Only 6-bit MLA KV is tested (V2 deployment). Use 8-bit or 6-bit MLA latents first; test 4-bit before trusting it.
2. **For GQA models, the default is 4-bit**: keys per channel before RoPE, values per token, first token in FP16 (KVQuant §3.1, 3.2, 3.5; KIVI). Cost ≤0.02 PPL on LLaMA-family models (KVQuant Table 18); 3.7× smaller.
3. **3-bit when the context must exceed ~8K**: KVQuant-3bit-1% (<0.1 PPL; RULER 53.65 vs 56.40) or TurboQuant 3.5-bit (LongBench 50.06 = FP16, no calibration). TurboQuant is the easier one to build on a CPU.
4. **2-bit only for chat that needs recent context only**: KIVI-2 keeps the last 128 tokens in FP16 and holds LongBench averages within ~0.7 points, but long-range retrieval falls (RULER 39.78; passkey 0.68–0.76). KVQuant-2bit-1% scores RULER 36.54. Palu-30% + 2-bit is the best 2-bit PPL in the set (5.76 vs FP16 5.12 on Llama-2-7B) but needs SVD per model.
5. **StreamingLLM** is a ceiling, not a compressor: use 4 sinks + window when the session will exceed the budget, and accept that old content is lost.

On this machine, KV is a **capacity** issue, not a speed issue. With an MLA model at 8K context, reading the full BF16 KV per token costs 549 MiB / 20 GB/s ≈ 27 ms [est], small next to SSD expert reads of seconds. With a usable KV budget of ~0.5–1 GB, an MLA model gets 7.6K–15K tokens at BF16 and 30K–61K tokens at 4 bits [est].

---

## 3. Question 2 — Speculative decoding for an MoE with experts streamed from SSD

### 3.1 Expected expert union

Per MoE layer: E routed experts, top-k per token, n tokens in one verification pass (n = γ + 1).

**Independent uniform routing.** Each token picks k distinct experts uniformly. P(expert e not picked by one token) = 1 − k/E. Over n tokens:

```
U_ind(n) = E · [1 − (1 − k/E)^n]
```

**Independent non-uniform routing** (popular experts, probabilities π_e with Σπ_e = k):

```
U(n) = Σ_e [1 − (1 − π_e)^n]   ≤ U_ind(n)   (concavity)
```

Popularity skew alone reduces the union only a little. Example [est]: 32 hot experts at π = 0.15 and 224 cold at π = 0.0143 give U(2) = 15.24 versus 15.75 uniform. Skew helps the **cache hit rate**, not the union.

**Temporally correlated routing.** Let ρ be the probability that an expert chosen by token t is also chosen by token t+1 (measured overlap). First-order model, ignoring re-hits on older experts:

```
U_corr(n) ≈ k · [1 + (n − 1)(1 − ρ)]
```

Independence gives ρ = k/E (3.1% for V3). ρ for DeepSeek-V3 is **not stated** in any paper.

**DeepSeek-V3 numbers** (E = 256, k = 8, 58 MoE layers, 22.0 MB per expert at 4 bits):

| n tokens | U_ind(n) | U/(n·k) | Cold routed bytes per pass, 4-bit [est] | U_corr, ρ = 0.3 | U_corr, ρ = 0.5 |
|---|---|---|---|---|---|
| 1 | 8.00 | 1.00 | 10.2 GB | 8.0 | 8.0 |
| 2 | 15.75 | 0.98 | 20.1 GB | 13.6 | 12.0 |
| 3 | 23.26 | 0.97 | 29.7 GB | 19.2 | 16.0 |
| 4 | 30.53 | 0.95 | 39.0 GB | 24.8 | 20.0 |
| 8 | 57.42 | 0.90 | 73.3 GB | 47.2 | 36.0 |
| 16 | 101.96 | 0.80 | 130 GB | — | — |
| 64 | 222.4 | 0.43 | 284 GB | — | — |

**Node-limited routing** (≤4 of 8 groups of 32 experts) does not change the marginal P(e chosen) = 8/256 under symmetry, so the expected union is the same [derived]. It only helps locality: one token's 8 experts sit in ≤4 contiguous 32-expert regions if the file stores experts by group.

So under independent routing, a 2-token verification pass reads 1.97× the expert bytes of a 1-token pass. That matches colibri's measurement (600 → 1,062 experts per token, 0.34 → 0.31–0.33 tok/s; 00_SYNTHESIS §4.3). If colibri's counts are per verification step, the implied overlap is ρ ≈ 0.23 [est, inference from their numbers].

### 3.2 When speculation still pays

Let one normal decode step cost 1 unit. Split it into
- (1 − f): work done once per pass whatever n is: always-on weights (attention, dense layers, shared expert, router, head), KV reads, fixed overhead;
- f: routed-expert bytes that miss the caches (any tier slower than the compute).

Let c = cost of one draft token relative to one step, α = acceptance, τ(α, γ) = (1 − α^(γ+1))/(1 − α). Then, as long as n stays below the compute crossover (§4):

```
Speedup S = τ / [ (1 − f) + f · U(n)/k + γ·c ]
          ≈ τ / [ 1 + f·(n − 1)·(1 − ρ) + γ·c ]
```

For the V3 MTP case (γ = 1, n = 2, τ = 1 + α):

```
MTP pays  ⇔  f·(1 − ρ) + c < α     (α = 0.85–0.90 measured)
```

Worked cases [est], c = 0.03 (one block + projection + one extra output-head pass):

| Setting | f | ρ | S (γ = 1, α = 0.875) | S (γ = 2, α = 0.85) |
|---|---|---|---|---|
| All experts cold from SSD, small resident part | 0.95 | 0.03 | 0.96 (loss) | 0.90 (loss) |
| Hypothetical larger-RAM machine: resident 8.1 GB in RAM at 20 GB/s (0.40 s), 50% expert hits from SSD at 6 GB/s (0.85 s) | 0.68 | 0.03 | 1.11 | 1.09 |
| Same, correlated routing | 0.68 | 0.30 | 1.25 | 1.28 |
| Everything resident in fast memory (dense-like, EAGLE Mixtral case) | 0.3 | 0.03 | 1.42 | 1.58 |

Conclusions:
1. **The union does kill most of the benefit when expert misses dominate.** With f → 1 and independent routing, S < 1 for every α < 1, because U(n)/k ≈ n while τ < n.
2. **Speculation still pays** when (a) a large share of step time is spent on the once-per-pass part (big always-on weights such as MLA attention: 11.4B of V3's 36.6B active), (b) the expert cache serves most draft-token experts (draft tokens hit the same hot set), or (c) consecutive tokens share experts (ρ high).
3. **Use γ = 1 (MTP depth 1) or γ = 2, chain only.** Deeper drafts and trees multiply the union (Medusa: 64-node trees; EAGLE: trees add τ but less speed). EAGLE's Mixtral result (1.50×) is the in-VRAM best case.
4. **Design rule [est, our proposal]: gate speculation per step on the predicted miss cost.** We already plan router lookahead. If the draft token's predicted experts are mostly cached, verify 2 tokens; otherwise decode 1. Zero-weight drafts (n-gram / copy-from-prompt, Leviathan §3.6, c ≈ 0) are worth trying first, because they add no draft bytes, only verification bytes.
5. The routing overlap ρ and the hit rate of draft-token experts must be **measured** from traces (00_SYNTHESIS §6 step 2) before we build this.

---

## 4. Question 3 — DeepSeek-V3 and V2 active parameters, bytes, residency

### 4.1 Method

The papers give totals (671B/37B, 236B/21B, 15.7B/2.4B) and the shapes in §1.10–1.11. Two values are **not stated** in the papers: the intermediate size of the dense FFN layers and the exact vocabulary (papers say "128K" and "100K"). I used 18,432 / 129,280 (V3), 12,288 / 102,400 (V2) and 10,944 / 102,400 (V2-Lite) [from public configs, not from the papers]. With these, the computed totals match the papers: 671.03B, 235.74B, 15.71B. MLA attention per layer (V3) = W_DQ 7168×1536 + W_UQ 1536×16384 + W_QR 1536×8192 + W_DKV 7168×512 + W_KR 7168×64 + W_UK 512×16384 + W_UV 512×16384 + W_O 16384×7168 = 187.1M. One SwiGLU expert = 3 × 7168 × 2048 = 44.04M.

### 4.2 Active parameters per token [derived]

| Component | DeepSeek-V3 | DeepSeek-V2 | DeepSeek-V2-Lite |
|---|---|---|---|
| Attention (MLA), all layers | 11.41B (61 × 187.1M) | 8.95B (60 × 149.2M) | 0.37B |
| Dense FFN layers | 1.19B (3 layers) | 0.19B (1 layer) | 0.07B (1 layer) |
| Shared experts | 2.55B (58 × 1 × 44.0M) | 2.78B (59 × 2 × 23.6M) | 0.45B |
| Routed experts, active | **20.43B** (58 × 8 × 44.0M) | **8.35B** (59 × 6 × 23.6M) | **1.35B** (26 × 6 × 8.65M) |
| Routers (gates) | 0.11B | 0.05B | 0.003B |
| Output head | 0.93B | 0.52B | 0.21B |
| Input embedding (table) | 0.93B (1 row read per token) | 0.52B | 0.21B |
| **Total incl. embedding table** | **37.55B** (paper: 37B) | **21.37B** (paper: 21B) | **2.66B** (paper: 2.4B)* |
| Routed experts, total stored | 653.9B (97.4% of params) | 222.7B | 14.39B |

*V2-Lite with the embedding table counted gives 2.66B; without it 2.45B, which matches the paper's 2.4B. The paper's counting convention is not stated.

### 4.3 Bytes per token (nominal bits; add ~6–12% for block scales)

| Part (V3) | Params | 4-bit | 2-bit |
|---|---|---|---|
| Always-on: attention + dense FFN + shared + router + head | 16.19B | 8.10 GB | 4.05 GB |
| Routed, 8 experts × 58 layers | 20.43B | 10.22 GB | 5.11 GB |
| One expert | 44.04M | 22.0 MB | 11.0 MB |
| One layer's 8 experts | 352M | 176 MB | 88 MB |
| **Total per token** | 36.6B | **18.3 GB** | **9.2 GB** |
| Whole model on disk | 671B | 335.5 GB | 167.8 GB |

V2: always-on 12.5B → 6.25 GB (4-bit); routed 8.35B → 4.18 GB; per expert 11.8 MB; disk 118 GB at 4-bit. V2-Lite: always-on 1.10B (without embedding table) → 0.55 GB; routed 1.35B → 0.67 GB; per expert 4.3 MB; disk 7.85 GB at 4-bit [all est].

### 4.4 What must stay resident and what can stream

| Must stay resident (touched by every token) | Can stream (from SSD) |
|---|---|
| MLA attention weights, all layers (V3: 11.4B) — the largest always-on block | Routed experts (V3: 653.9B, 97.4%) — 8 of 256 per layer per token |
| Dense FFN of the first 3 layers | Input embedding table (one row per token; leave it memory-mapped) |
| Shared expert of each MoE layer | MTP module, if unused (size not stated in the paper) |
| Router weights and biases — keep at high precision (V3 keeps gating in BF16) | Cold expert copies at low bit width (HOBBIT/Drop-by-Drop idea) |
| Output head (0.93B) — keep at high precision | |
| Norms; KV cache (68.6 KiB/token BF16) | |

**Consequence for this machine [est].** V3's always-on part is 8.1 GB at 4 bits, more than the ~7.8 GB of usable RAM + VRAM. At 2 bits it is 4.05 GB, which fits, but 2-bit attention is a large accuracy risk. With 4-bit routed experts read cold at 5.5 GB/s, the routed part alone costs 1.86 s per token (~0.5 tok/s ceiling before cache hits). The disk (82 GB free) cannot hold V3 or V2 at 4 bits. **V2-Lite fits entirely in RAM + VRAM at 4 bits and is the right development target for MLA + DeepSeekMoE code paths.**

Note on MLA compute: absorbing W_UK and W_UV into the query/output path at load time would multiply the weight bytes (e.g. W_UQ·W_UK^T per head is 1536 × 128 × 512 instead of 25.2M + 8.4M per layer [est]). Apply the up-projections as separate small matmuls at decode time; the weight bytes then stay as in §4.2.

---

## 5. Question 4 — Pope et al. roofline, restated for this machine

### 5.1 Formulas from the paper (plus the T = max form from Sarathi-Serve)

```
FLOPs per token         = 2 · N_params              (matmuls; attention FLOPs excluded)
T_compute (batch B)     = 2 · N · B / (n_chips · peak_FLOPS)
T_memory                = (bytes_weights + bytes_KV(B, L)) / (n_chips · HBM_bandwidth)
T_comm (1D WS)          = 2 · B · L · E / network_bandwidth
T_comm (2D WS)          = 8 · B · L · E / (√n_chips · network_bandwidth)
Step time               ≈ max(T_compute, T_memory) + non-overlapped T_comm
MFU                     = observed tokens/s / (peak_FLOPS / 2N)
```

Weights load once per forward pass whatever B is; the KV cache is per sequence. Crossover batch where compute equals weight loading:

```
B* = peak_FLOPS · bytes_per_param / (2 · bandwidth)
```

TPU v4, bf16: 275e12 × 2 / (2 × 1.2e12) ≈ 229 tokens. Sarathi-Serve reports ~200 (theory) and 500–600 (practice) on A100.

### 5.2 Restated for this machine (single user, n_chips = 1, no network)

Decode, one token, weights split over tiers (VRAM, RAM, SSD):

```
t_token ≥ max over overlapped stages of:
          Σ_tier  W_tier / BW_tier        (weight bytes from each tier; sum if serialized)
          KV_bytes / BW_KV_tier
          2 · N_active / F_effective
```

For n tokens per weight pass (speculation, prefill chunk), dense parts keep W constant; routed parts become `W_routed(n) = U(n) · s_expert · L_moe` (§3).

Hardware numbers [est]: VRAM 128 GB/s; RAM ~20 GB/s; SSD ~5.5 GB/s; PCIe ~10–12 GB/s. GTX 1650: ~3 TFLOPS FP32 (896 cores × ~1.66 GHz × 2), FP16 about double, no tensor cores. Ryzen 5 5600H, AVX2: ~0.6 TFLOPS FP32 peak (6 cores × 32 FLOP/cycle × ~3.3 GHz); realistic quantized-matmul throughput maybe 0.3–0.5 T ops/s.

Crossover tokens per weight pass, 4.5 bits/param (0.5625 B):

| Compute unit / weight source | F (ops/s) | BW (B/s) | B* tokens [est] | Meaning |
|---|---|---|---|---|
| GPU / VRAM | 3e12 | 128e9 | ~7 | Above ~7 tokens per pass, GPU compute dominates |
| CPU / RAM | 0.5e12 | 20e9 | ~7 | Same order on CPU |
| CPU / SSD | 0.5e12 | 5.5e9 | ~26 | Streamed layers are free up to ~26 tokens |
| GPU / SSD (via PCIe) | 3e12 | 5.5e9 | ~150 | Prefill chunks of ~150+ tokens hide the SSD fully |

Decode ceilings, single user [est]: dense 8B Q4_K_M (4.9 GB, all resident, split VRAM/RAM) ≈ 3.3 GB/128 + 1.6 GB/20 → ~0.11 s → ~9 tok/s; V2-Lite at 4 bits, 1.3 GB read per token, mostly from VRAM/RAM → ~15–40 tok/s; V3 (if it fit) ≈ 0.4 s resident + 1.86 s cold experts → ~0.44 tok/s before cache hits. The roofline also says: below B*, every extra token in a pass is free for **dense** bytes but never free for **routed** bytes, which grow with U(n).

---

## 6. Question 5 — Does Sarathi-Serve's chunked prefill help a single-user streaming setup?

**Yes for the mechanism, no for the goal.**

- Sarathi's goal is to protect other users' decode latency (TBT) while prefill runs. With one user there are no other decodes to stall, so stall-free batching gives nothing.
- Sarathi's mechanism is "one weight pass serves many tokens". For a streamed model this is the dominant prefill cost. Each pass reads the weights from SSD once. With MoE, a chunk of n prompt tokens touches U(n) experts per layer, and U saturates fast: for V3, 64 tokens already touch 222 of 256 experts (87%). So **every chunk costs about one full read of the routed weights**, and prefill cost ≈ ceil(P / chunk) × full-model read.
- Therefore use **the largest chunk the activation memory allows**, ideally the whole prompt, processed **layer-major** (load layer i once, run all P tokens through it, keep the P × d hidden states, go to layer i+1). Hidden states are small: 32K tokens × 7168 × 2 B = 470 MB for V3; 4K tokens × 2048 × 2 B = 17 MB for V2-Lite [est]. This is the single-machine form of Pope's weight-gathered layout and of FlexGen's schedule (G group).
- Example [est]: a 120B MoE in a 61 GB file, 4K-token prompt. With 512-token chunks: 8 passes × (61 GB / 5.5 GB/s ≈ 11 s) ≈ 90 s of I/O. Layer-major: one pass ≈ 11 s of I/O plus ~10 s of GPU compute (2 × 5B × 4096 ≈ 41 TFLOP).
- Sarathi's own data supports large chunks: chunk overhead ≤~25% at 512 tokens and negligible at 2048; re-reading earlier chunks' KV stays compute-bound. On this machine the KV re-read comes from RAM and is small next to weight reads.
- Keep chunk sizes at multiples of the kernel tile (Sarathi: 257 tokens can cost much more than 256).
- Decode then interleaves nothing; the only overlap to build is "prefetch layer i+1's experts while layer i computes".

---

## 7. Question 6 — Exploiting LLM Quantization and DFloat11: what we must do

**Exploiting LLM Quantization (X01).** A full-precision checkpoint can be built so that it looks safe and scores well, but turns malicious after zero-shot block quantization (LLM.int8(), NF4, FP4): code security 98.0% → 18.5% on Phi-2, refusals up to 39.1%, content injection up to 74.7%. The attack only needs a few GPU-hours on a 3B model. Our converter quantizes third-party weights on this machine, which is exactly the threat model. We must: (1) take weights only from the original publisher and verify hashes; (2) run a full-precision versus quantized behaviour diff on a fixed probe set (insecure code, harmless-question refusals, brand mentions) for any third-party fine-tune, and block deployment when the rates diverge; (3) log per-block max/median weight ratios in the converter, because long-tailed blocks widen the attack intervals; (4) keep N(0, 1e-3) pre-quantization noise as an optional flag only (it removed the attack on Phi-2 with no utility loss, but the authors say side effects are unknown). The paper did not test GPTQ/AWQ or GGUF; Analysis B's Outlier Injection paper covers those, so the diff test applies to every format.

**DFloat11 (L01).** DF11 is lossless (bit-exact) and shrinks BF16 weights to ~11 bits (~70%) with GPU decoding in SRAM, but it compresses only the BF16 exponent. Our runtime uses 2–4-bit codes from SSD, which have no exponent field, and 11 bits is ~2.5× more bytes per token than Q4. So DF11 does not belong in our decode path. We must do only two small things: (1) if disk space is short, store BF16 master checkpoints in DF11 (or ZipNN) form as converter input, saving ~30% of disk; (2) reuse the kernel pattern (hierarchical 256-entry LUTs, gap array, per-block decode right before the GEMM) if we ever entropy-code quantized expert indices. Whether 4-bit expert indices have entropy below 4 bits is not measured anywhere in these papers; it is a cheap experiment.

---

## 8. Blind spots — cited papers we still lack

Priority: **H** = needed for the current design decisions; **M** = useful soon; **L** = background. IDs are given only where the reference text contains them.

| Pri | Title | arXiv ID | Cited by | Why |
|---|---|---|---|---|
| H | DeepSeekMoE: Towards Ultimate Expert Specialization in MoE Language Models | 2401.06066 | V2, V3 | Shared/fine-grained expert design; expert specialization data relevant to caching |
| H | Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts (Wang et al.) | 2408.15664 | V3 | The routing bias b_i; load statistics; how routing behaves at inference |
| H | Draft & Verify: Lossless LLM Acceleration via Self-Speculative Decoding | 2309.08168 | EAGLE | Layer-skipping draft uses the target's own weights; check its effect on expert unions |
| H | REST: Retrieval-Based Speculative Decoding | 2311.08252 | EAGLE | Zero-weight drafts (c ≈ 0) — the best fit for §3 |
| H | Qwen3 technical report / Qwen3-30B-A3B config | not cited | — | Needed to answer Q1(c) and our 30B-A3B baseline |
| H | Any study of token-to-token expert routing correlation (ρ) for DeepSeek/Qwen MoE | not cited | — | ρ decides whether speculation pays (§3); we must measure it ourselves if none exists |
| M | Better & Faster LLMs via Multi-Token Prediction (Gloeckle et al., ICML 2024) | not in reference (OpenReview URL) | V3 | Original MTP heads; acceptance data |
| M | Accelerating LLM Decoding with Speculative Sampling (Chen et al.) | 2302.01318 | Medusa, EAGLE, S01 | Parallel original; 70B results |
| M | SpecInfer: Speculative Inference and Token Tree Verification | 2305.09781 | Medusa, EAGLE | Tree verification correctness |
| M | Accelerating LLM Inference with Staged Speculative Decoding | 2308.04623 | Medusa, EAGLE | Targets small-batch / local inference |
| M | QJL: 1-bit Quantized JL Transform for KV Cache Quantization | 2406.03482 | TurboQuant | The residual stage of TurboQuant-prod |
| M | RotateKV: Accurate and Robust 2-bit KV Cache Quantization via Outlier-Aware Rotations | 2501.16383 | TurboQuant | 2-bit KV with rotations |
| M | PolarQuant: Quantizing KV Caches with Polar Transformation | 2502.02617 | TurboQuant | Alternative sub-4-bit KV |
| M | SnapKV: LLM Knows What You Are Looking For Before Generation | 2404.14469 | TurboQuant | Compresses long-prompt KV once — fits slow prefill |
| M | H2O: Heavy-Hitter Oracle for Efficient Generative Inference | 2306.14048 | KIVI, Medusa | Token eviction baseline |
| M | SparQ Attention: Bandwidth-Efficient LLM Inference | 2312.04985 | KVQuant | Reads only part of the KV per step |
| M | Model Tells You What to Discard: Adaptive KV Cache Compression (FastGen) | 2310.01801 | KVQuant | Per-head eviction policies |
| M | Microscaling Data Formats for Deep Learning | 2310.10537 | V3 | MXFP4 is the format of the ~120B MoE in our size table |
| M | ZipNN: Lossless Compression for AI Models | 2411.05239 | DFloat11 | Disk-side lossless storage for converter inputs |
| M | Lookahead decoding (Fu et al.) | blog, no ID | Medusa, EAGLE | Draft-free (Jacobi) speculation |
| L | GEAR: KV Cache Compression Recipe for Near-Lossless Inference | 2403.05527 | Palu | Low-rank error compensation for KV quantization |
| L | Atom: Low-bit Quantization for Efficient and Accurate LLM Serving | 2310.19102 | KIVI, KVQuant, Palu | 4-bit KV baseline |
| L | PyramidKV | 2406.02069 | TurboQuant | Layer-wise KV budgets |
| L | Scissorhands | ID not in extracted reference | KIVI | Eviction |
| L | LM-Infinite | 2308.16137 | StreamingLLM | Length generalization |
| L | DistillSpec | 2310.08461 | EAGLE | Draft distillation |
| L | Cascade Speculative Drafting | 2312.11462 | EAGLE | Multi-level drafts |
| L | Sarathi (2023), Splitwise, DistServe, Vidur | IDs not given in references | Sarathi-Serve | Multi-user serving; low value for single user |
| L | GQA | 2305.13245 | V2, Palu | Known |
| L | Fast Transformer Decoding: One Write-Head (MQA) | 1911.02150 | V2, EAGLE, Medusa | Known |
| L | YaRN | 2309.00071 | V2, V3 | Context extension used by DeepSeek |
| L | FP8 Formats for Deep Learning | 2209.05433 | V3 | FP8 checkpoint reading |
| L | NeuZip | 2410.20650 | DFloat11 | Lossless ANS weights |
| L | Huff-LLM | 2502.00922 | DFloat11 | FPGA lossless |
| L | On the Exploitability of Instruction Tuning (Shu et al.) | ID not in reference | X01 | Poisoning baseline |
| L | Quantization Backdoors to Deep Learning Commercial Frameworks (Ma et al.) | journal, no ID | X01 | Vision-model precursor |

---

## 9. Corrections and additions to 00_SYNTHESIS

1. §4.5: the GPTQ/AWQ/GGUF ">90%" figure belongs to Outlier Injection (Analysis B). Exploiting LLM Quantization (X01) covers LLM.int8(), NF4 and FP4 only, on models ≤7B.
2. §4.4: "MLA about 1/10 the size" — V2 reports −93.3% KV versus DeepSeek 67B; the formula is GQA with 2.25 groups (576 elements per layer per token).
3. §4.3: EAGLE's Mixtral result (1.50× vs 2.7–3.8× dense) and the union formula in §3 confirm and generalize colibri's MTP observation. MTP is not useless, but it pays only when expert misses are a minority of step time.
4. §3 table: DeepSeek-V3 needs 8.1 GB of always-on weights at 4 bits; it cannot run here even with a second SSD unless RAM grows. Add DeepSeek-V2-Lite (15.7B/2.4B, 7.85 GB at 4 bits) as the MLA + MoE test model.
5. §6: add "layer-major whole-prompt prefill" to the engine work list, and "measure routing overlap ρ and draft-token hit rate" to the simulator work list.
