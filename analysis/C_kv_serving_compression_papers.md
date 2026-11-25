# Analyst C: KV-cache compression, lossless weight compression, and inference scheduling

Papers reviewed (folder `<repo>\papers\`):

1. `09_PuzzleKV_2608.23843.pdf`: PuzzleKV, page-wise low-rank KV-cache compression (Wang, Wang, Zhang, Li; arXiv 2608.23843, 24 Aug 2026).
2. `10_cmu-Rule_LLM_Scheduling_2611.04807.pdf`: "Asymptotic Optimality of cμ-Rule for LLM Inference Scheduling under Memory Constraint" (Li, Wang, Si; arXiv 2611.04807, 3 Nov 2026).
3. `11_ZipServ_2603.17435.pdf`: ZipServ, hardware-aware lossless BF16 weight compression with fused decompress-GEMM (Fan et al.; ASPLOS '26; arXiv 2603.17435).

Method note: the Read tool could not render the PDFs (no `pdftoppm`). I extracted the text with `pdftotext -layout` and read the full text of all three papers. Some equations lost Greek symbols in the extraction. Where I rebuilt an equation from the surrounding text, I say so. Page numbers are the printed page numbers of each paper.

Labels used below:
- "Paper:" means the fact is in the paper.
- "Derived:" means I calculated it from numbers in the paper.
- "Analyst view:" means my judgement for our goal. It is not a claim of the paper.

---

## Summary for our goal

| Paper | What it saves | Lossless? | Helps a single user on 8-32 GB RAM / 0-12 GB VRAM? |
|---|---|---|---|
| PuzzleKV | KV cache to 58.8% (FP16 factors) or 18.7% (INT4 factors) of full size | No (small accuracy loss) | **Yes, for long context.** For a 70B GQA model at 32k, KV goes from 10 GiB to about 5.9 GiB (FP16 factors) or about 1.9 GiB (INT4 factors). Prefill peak memory is NOT reduced in the paper's design. We must change that (see 1.7). |
| ZipServ | BF16 weights to about 71-72% of size | Yes (bit-exact) | **Mostly no.** A 70B model in BF16 is still about 93.5 GB after ZipServ. 4-bit quantization gives about 35 GB. The exponent trick does not apply to INT4/GGUF codes. Some design ideas transfer (fixed-length bit-plane format, fused "load compressed, compute decompressed"). |
| cμ rule | No memory. It reduces mean latency in multi-request serving under a KV-capacity limit | n/a | **Small.** With one request there is no queue. The step-time model and the "future KV feasibility" admission check transfer. The idea of co-scheduling prefill chunks with decode to share one weight pass is useful for weight streaming (analyst view). |

---

## 1. PuzzleKV: page-wise low-rank KV-cache compression

### 1.1 Problem

Paper (p.1): KV cache size per request is `2 · L · Hkv · T · d` elements. For Qwen3-32B (64 layers, 8 KV heads, head_dim 128) in BF16, one 128K-token sequence needs about 32 GiB of KV. Previous low-rank methods use ONE basis for a large region (from model weights, e.g. Palu/LoRC; from calibration data, e.g. ECKVH/EigenAttention; or from the whole sequence, e.g. OjaKV/xKV). Such a shared basis keeps the dominant directions and can lose sparse but important information (for example one "needle" in repetitive text).

PuzzleKV's idea: give each small page of tokens its own low-rank basis. It needs no training and no calibration.

### 1.2 Core mechanism (implementable description)

**Data layout.**
- For each layer and each KV head, split the token axis into fixed pages of `P` tokens (default `P = 32`). A page of keys is a matrix `X_p` of shape `P × d` (`d = 128`). The same applies to values.
- Dense regions (stored uncompressed): one **sink page** (the first 32 tokens), the **most recently completed page**, and the **current incomplete page** (p.6, "Implementation"). So at most 96 dense tokens per head per layer.
- All other completed pages are stored as two factors per page:
  - `L_p` of shape `P × r` and `R_p` of shape `r × d`, with `X_p ≈ L_p · R_p`.
  - Default ranks: `rK = 16` for keys and `rV = 14` for values (p.6).
- Keys are compressed **post-RoPE** (p.7). So attention needs no extra RoPE step at read time. Pre-RoPE keys are slightly more compressible: RULER average 90.88 vs 90.40 at (16,16) (Table 14, p.17).

**Storage ratio** (Eq. 3, p.4):

```
rho_page = (rK + rV) · (P + d) / (2 · P · d)
         = (16 + 14) · (32 + 128) / (2 · 32 · 128) = 4800 / 8192 = 0.5859
```

Derived: per head per layer per page, the K factors hold 32·16 + 16·128 = 2560 elements and the V factors hold 32·14 + 14·128 = 2240 elements. That is 4800 elements = 9600 bytes in FP16, or 300 bytes per token per head-layer. The dense value is 512 bytes.

**Factorization (prefill and decode)** (p.4):
1. Run normal dense prefill. Build the full dense KV cache.
2. For each eligible page `X_p` (P × d, with P < d), compute the small Gram matrix `G_p = X_p · X_pᵀ` (P × P = 32 × 32).
3. Do a batched symmetric eigendecomposition of all `G_p`. Take the top-r eigenvectors `Q_{p,r}` (P × r).
4. Set `L_p = Q_{p,r}` and `R_p = Q_{p,r}ᵀ · X_p` (r × d). This gives the exact truncated-SVD factors. The singular values are absorbed into `R_p`.
5. Write the factors to preallocated storage. Free the dense page.

All pages have the same shape. So step 3 is one batched GPU call over all pages, heads and layers.

**Attention over mixed pages without reconstruction** (Eq. 4-5, p.5). For query `q` and factorized page p:

```
s_p   = ( (q · R_pKᵀ) · L_pKᵀ ) / sqrt(d)      # contract q with small factors first: cost O(r·d + P·r)
m_p   = max_j s_p,j
l_p   = Σ_j exp(s_p,j − m_p)
o_p   = ( exp(s_p − m_p) · L_pV ) · R_pV       # (1×P)(P×rV) then (1×rV)(rV×d)
```

Dense pages give the same `(m_p, l_p, o_p)` from their stored K, V. Merge all pages with the FlashAttention online-softmax rule:

```
m = max_p m_p
o = Σ_p exp(m_p − m) · o_p  /  Σ_p exp(m_p − m) · l_p
```

The order of the products is important: the dense K_p and V_p are never built. Reconstruction would create temporary dense buffers and lose the memory saving (the paper cites FlashSVD [14]).

**Incremental update during decode** (p.5):
1. Append the new (k, v) to the current dense page. It joins attention at once.
2. When that page is full, it becomes the "recent" dense page. The oldest page that leaves the local window is factorized once (steps 2-5 above).
3. Factorized pages are never changed again. So each decode step does at most one page factorization (per layer and head).

**INT4 extension** (p.7-8): split the singular values evenly: `L = U · Σ^{1/2}`, `R = Σ^{1/2} · Wᵀ`. Then apply symmetric INT4 quantization to each factor: column-wise for `L_K` and `R_K`, row-wise for `L_V`, column-wise for `R_V`. The scale storage format is not stated.

### 1.3 Memory result

- Paper, Table 3 (p.8): PuzzleKV storage = **0.588** of Full KV at 16K. PuzzleKV + INT4 = **0.187**. Direct INT4 KV (KIVI-style) = 0.283. Direct INT3 = 0.223.
- Paper, Table 5 (p.10), measured logical live KV at 16K plus 256 generated tokens on Qwen3-8B:
  - Raw KV: 2204.85 MiB after prefill, 2205.90 MiB at the end.
  - PuzzleKV: **2204.85 MiB after prefill (same as raw)**, 1297.05 MiB after the first compression, 1309.50 MiB at update peak, 1296.25 MiB at the end. This is 58.76% of raw. The incremental update adds only 12.45 MiB at peak.
- Derived: the dense region of 96 tokens explains the gap between 0.586 and 0.588 at 16K: (16288 · 0.5859 + 96) / 16384 = 0.588.

**Important for us:** the peak memory during prefill is the full dense KV cache. Compression happens after prefill.

### 1.4 Speed result and hardware

Hardware: 1× NVIDIA GH200 (120 GB), 72-core ARM Neoverse-V2, 212.70 GiB system memory (Table 6, p.13). Implementation: Hugging Face Transformers prototype, batch size 1. The paper says "batch-one standalone prototype" (p.10).

- TTFT: **+20.9%** vs Raw KV at 32K (p.9, Fig. 5). The cost comes from page decomposition during prefill.
- TPOT at 16K (Table 4, p.10): Static KV 40.477 ms, Raw KV 42.662 ms, PuzzleKV 42.737 ms. That is +0.075 ms/token (+0.18%). So decode is **not faster** in this prototype. It is only smaller.
- Decomposition time per sample during prefill at 16K (Table 12, p.16): P=16: 0.11 s; P=32: 0.17 s; P=64: 0.27 s.

### 1.5 Accuracy cost

Models: Qwen3-8B and Llama-3.1-8B-Instruct only. Budget ≈ 0.6 for all compressed methods.

- RULER 16K / 32K (Table 1, p.6):
  - Llama3.1: Full 92.33 / 87.54. PuzzleKV 88.82 / 84.39 (96.2% / 96.4% of Full). Global SVD 62.61 / 59.38. Palu 87.26 / 82.68. H2O 68.76 / 61.75.
  - Qwen3: Full 91.41 / 89.44. PuzzleKV 90.73 / 87.53 (99.3% / 97.9%). Global SVD 76.51 / 71.89. Palu 79.02 / 71.03. H2O 64.00 / 62.95.
- LongBench (Table 2, p.6): Llama3.1 Full 45.83 vs PuzzleKV 45.33 (98.9%). Qwen3 Full 29.86 vs PuzzleKV 29.48 (98.7%). H2O is very close here (45.61 / 29.10).
- With INT4 factors (Table 3, RULER 16K): Llama3.1 86.24 (−2.58 vs PuzzleKV FP16). Qwen3 90.20 (−0.53). Direct INT3 at a larger budget (0.223) gives 88.51 / 91.02. So on Llama3.1, plain INT3 KV is better than PuzzleKV+INT4 at a slightly larger budget.
- Weak tasks: NIAH-S3 (exact long UUID) on Llama3.1: 82.00 vs 100.00 Full at 16K. With INT4: 74.00 (Table 7). CWE on Llama3.1 at 32K: 23.04 vs 26.38.
- Rank ablation (Table 13): rank is critical. (12,12) gives 64.28. (16,14) gives 88.82. (20,20) gives 92.11, almost Full. Keys need more rank than values: (16,14) is better than (14,16).

### 1.6 Limitations and assumptions

- Only 8B models, only GQA with 8 KV heads and d = 128. Not tested on 70B+, MoE, or MLA (DeepSeek-style latent attention). Applicability to MLA is not stated.
- RULER only at 16K and 32K. No 128K test.
- Batch 1, HF Transformers, GH200. No vLLM integration. No CPU kernel. No speed-up in decode.
- Peak memory during prefill = full dense KV (Table 5).
- The SVD runs on the GPU. The paper needed a newer PyTorch (2.12.0+cu130) for the efficiency study because the older stack had "an SVD performance issue" (Table 6 caption).
- Fixed ranks for all layers and heads. Fig. 2 shows that the required rank varies by layer and head. Adaptive rank is not explored.
- Lossy. Exact-copy tasks (UUIDs, code) lose the most.

### 1.7 Relevance to our goal

Analyst view:

1. **Useful for long context.** For a 70B GQA model, KV at 32K in FP16 is 10 GiB (see 1.8). That is more than all the VRAM we have. PuzzleKV+INT4 brings it to about 1.9 GiB.
2. **We must fix the prefill peak.** Our RAM is small. We cannot hold the full dense KV during prefill. The method is page-local, so a natural change is: run chunked prefill and factorize each page as soon as it leaves the local window. Then the dense part never exceeds about 96 tokens per head-layer plus one chunk. The paper does not do this. We would have to build and test it.
3. **The decomposition cost is small for us.** Derived estimate: one page (32 × 128) needs a Gram product (≈ 2·32·32·128 ≈ 0.26 MFLOP), a 32 × 32 eigendecomposition, and R = Qᵀ X (≈ 0.13 MFLOP for r = 16). That is about 1 MFLOP for K plus V. For 70B at 128K there are 4096 pages × 640 head-layers ≈ 2.6 M pages. So about 2.6 TFLOP in total. The prefill of 70B at 128K is about 2 · 70e9 · 131072 ≈ 1.8e16 FLOP. So the decomposition is about 0.015% of prefill compute. A CPU can do it.
4. **Decode on CPU or small GPU.** In decode, KV reads fall to about 59% (or about 19% with INT4). On a bandwidth-bound CPU this helps directly when context is long. But with weight streaming from SSD/RAM, weight reads dominate each step until the context is very long. So the decode-speed gain is small at 8K and grows with context length.
5. **Composes with KV quantization** (the paper shows INT4 factors). It also composes with weight quantization, because the KV path and the weight path are separate.
6. **Risk:** exact-retrieval tasks degrade first. Use higher ranks (e.g. 18/16, RULER 91.43) when quality matters. Check the cost with Eq. 3.

### 1.8 KV-cache size for a 70B GQA model (task item 3)

Model: 80 layers, 8 KV heads, head_dim 128, FP16 (2 bytes).

Derived: bytes per token = 2 (K and V) × 80 × 8 × 128 × 2 B = **327,680 B = 320 KiB per token**.
Check with the paper's own example: Qwen3-32B (64 layers) gives 262,144 B/token × 131,072 = 32 GiB. This matches p.1.

PuzzleKV ratio including the dense region (≤ 96 tokens per head-layer: sink + recent + current page):

```
ratio(T) = 0.5859 + (1 − 0.5859) · 96 / T
```

For PuzzleKV + INT4, the paper measured 0.187 only at 16K. I show the flat value 0.187. My extrapolation, which assumes the 96 dense tokens stay in FP16, gives 0.191 at 8K, 0.184 at 32K and 0.182 at 128K. The difference is less than 0.2 GiB at 128K.

| Context | FP16 full KV | PuzzleKV (P=32, rK=16, rV=14) | PuzzleKV + INT4 (0.187) | Direct INT4 KV (0.283) | Direct INT3 KV (0.223) |
|---|---|---|---|---|---|
| 8K (8,192) | 2.50 GiB (2.68 GB) | ratio 0.5908 → **1.48 GiB** | **0.47 GiB** | 0.71 GiB | 0.56 GiB |
| 32K (32,768) | 10.00 GiB (10.74 GB) | ratio 0.5872 → **5.87 GiB** | **1.87 GiB** | 2.83 GiB | 2.23 GiB |
| 128K (131,072) | 40.00 GiB (42.95 GB) | ratio 0.5862 → **23.45 GiB** | **7.48 GiB** | 11.32 GiB | 8.92 GiB |

Per compressed token: 192,000 B (187.5 KiB) with FP16 factors. That is about 61 KiB with INT4 factors (derived from 0.187).

Caveats: the accuracy numbers come from 8B models at 16K-32K. Not stated for 70B or 128K. Peak memory during prefill is still the FP16 column unless we add compress-during-chunked-prefill (1.7, item 2).

---

## 2. cμ rule for LLM inference scheduling under a KV memory limit

### 2.1 Problem

Paper (p.1-5): one GPU serves many requests. They arrive at random, with different prompt lengths `s_j` and output lengths `o_j`. Each request does prefill (in chunks) and then decode. Its KV cache grows until it completes. All running requests share a fixed KV capacity `M` (in tokens). The scheduler chooses which requests run in each GPU step (continuous batching). Goal: minimize mean latency, which equals the time-integral of the number of unfinished requests. This differs from classical scheduling because (a) the memory used by a job grows during service, (b) step time depends on the batch content, and (c) M is fixed (O(1)) while rates grow.

### 2.2 Core mechanism

**Step-time model** (Eq. 1, p.6; Table 1, p.7). Fitted on an NVIDIA H800:

```
step_time = a_pf · N_pf + a_kv · N_kv + b
a_pf = 2.264e-5 s per prompt token processed in the step
a_kv = 4.108e-8 s per KV token read or written in the step
b    = 7.794e-3 s fixed per step
R² = 0.999902 (in sample, 16,205 obs), 0.999905 (out of sample, 3,987 obs); RMSE 2.294 / 2.268 ms
```

Derived sanity check (my calculation, not in the paper): Llama-3-8B has a KV of 2 × 32 layers × 8 × 128 × 2 B = 131,072 B per token. 131,072 B / 4.108e-8 s ≈ 3.2 TB/s, which is close to H800 HBM bandwidth. The fixed term b ≈ 7.8 ms is consistent with reading about 16 GB of BF16 weights per step plus launch overhead. So the model is: compute for prefill, KV bandwidth, and one weight pass per step.

**Per-request lifetime memory volume** (Eq. 14 and Eq. 31). `K̃_j` = sum, over every GPU step of the request's life, of the KV slots it holds in that step. Prefill steps hold min{chunk·(ℓ+1), s_j} slots. Decode steps hold s_j + (number of tokens generated so far) + 1. Roughly: K̃_j ≈ s_j·n_j/2 + s_j·o_j + o_j²/2, with n_j = number of prefill chunks.

**Effective service rate** (Definition 3, Eq. 18, p.13). The extracted equation is damaged. I rebuilt it from the text: when class j fills memory alone, it completes M/K̃_j requests per step, and each step costs (a_pf·s_j + a_kv·K̃_j)·M/K̃_j + b seconds. So:

```
mu_j = 1 / ( a_pf · s_j  +  a_kv · K̃_j  +  b · K̃_j / M )
```

The three terms are: prefill compute, KV I/O, and the share of fixed per-step overhead. The share is proportional to the fraction of memory-time the request occupies.

**cμ rule:** sort classes by `c_j · mu_j`, highest first (c_j = holding cost; c_j = 1 in the experiments). Serve in that order, subject to the memory limit. Pass the unused capacity to lower-priority classes (Definition 4, Skorokhod-reflection construction). With c = 1, b = 0 and unchunked prefill, the rule reduces to "smallest KV volume first" (SVF) (p.21).

**Theory:**
- Theorem 1: in the fluid model, the cμ policy minimizes cumulative holding cost.
- Theorem 2: for a finite workload, it minimizes completion time and maximizes full-run output-token throughput.
- Theorem 3: a "smoothed cμ-inspired" policy (Definition 5) converges to the fluid optimum. It spreads each admission plan over a window of w_n steps, with w_n > max KV lifetime and w_n/n → 0. It does not need to know the arrival rates.

The fluid regime: arrival rate and GPU speed scale by n, M stays fixed. A "stage-level state-space collapse" (Sec. 3.4) means that in the limit only the waiting queues matter.

**Practical policies implemented in vLLM** (Sec. 6.2, p.19):
1. Compute mu_j for each waiting request from (s_j, o_j) and the fitted (a_pf, a_kv, b).
2. **Future KV feasibility check:** admit a request only if the total KV use stays ≤ M at every future step. The check includes cache growth of all running requests and the memory freed when they complete.
3. **cμ:** scan waiting requests in decreasing mu order. Stop at the first request that fails the check.
4. **cμ-backfill:** if a request fails, compute its earliest feasible future start step and reserve it. Continue scanning lower-priority requests. Admit one now if it is feasible with all reservations included (Fig. 6).
5. Nonpreemptive: an admitted request runs until it completes.
6. **Assumes the output length is known** (the replay forces the exact output length).

### 2.3 Memory result

None. The paper does not compress memory. It uses a fixed M better. The value of M and the chunk size used in the experiments are not stated.

### 2.4 Speed / latency result and hardware

Hardware: 1× NVIDIA H800, Meta-Llama-3-8B-Instruct, vLLM. Traces from the Qwen-Bailian dataset, filtered to ≤ 8K total tokens, load ≈ 0.95.

- Semi-synthetic, 16 GMM classes from the To-B trace, 8 repetitions (Table 3, p.22). Row labels lost Greek symbols in extraction. I mapped them with the text on p.22.
  - cμ-backfill: average latency **10.366 ± 0.424 s**, mean TTFT 3.593 s, P99 68.678 s, 2,521.1 tok/s (best mean latency, TTFT and throughput).
  - cμ: 10.759 s average.
  - vLLM default: **30.226 ± 3.664 s** average, 23.548 s TTFT.
  - WAIT-frac has the best P99 (67.163 s).
  - Throughput is almost the same for all policies (about 2,509-2,521 tok/s).
- Real traces, time-compressed (Table 4, p.23), average latency in seconds:
  - To-B: cμ-backfill 9.472 vs vLLM 35.492.
  - To-C: 28.265 vs 117.943.
  - Thinking: cμ 128.826 (best) vs vLLM 361.491.
  - Coder: 23.145 vs 69.700.
  - Throughput gain is small (e.g. To-B 2851 vs 2838 tok/s). The gain is in latency, not throughput.
  - SVF is almost as good as cμ on To-C and Coder.
  - P99 latency is sometimes worse than vLLM: Thinking 2134 s for cμ-backfill vs 829 s for vLLM. On To-C, vLLM has the best P99 (496 s).

### 2.5 Accuracy cost

None (scheduling only; outputs are unchanged).

### 2.6 Limitations and assumptions

- Output lengths are known in advance. In practice they must be predicted. The effect of prediction errors is "future work" (Sec. 7).
- One GPU. Multi-GPU is future work.
- Optimality is asymptotic (fluid limit, many requests, high load). It is not optimal for few requests.
- Nonpreemptive implementations. Tail latency can be worse than the default scheduler (starvation of long requests).
- The step-time model is linear and fitted per GPU. It assumes that weights are resident on the GPU and that b is a constant.
- Linear holding cost only.

### 2.7 Relevance to our goal (single user, small memory)

Analyst view. With one request at a time, there is no queue. The cμ ordering does nothing. These parts transfer:

1. **The step-time model is the right cost model for offloaded inference.** In our setting, `b` (one pass over the weights) is not 7.8 ms. If a 4-bit 70B model streams 35 GB from NVMe at 3-7 GB/s, b is about 5-12 s per step. So the per-step fixed cost dominates everything. The lesson is: **put as many tokens as possible into each weight pass**. This means large prefill chunks, speculative decoding (verify many draft tokens per pass), and batching of parallel sub-requests (agent tool calls, multiple samples).
2. **Co-schedule prefill and decode in the same weight pass** (mixed batching, Sarathi-style; the paper cites Agrawal et al. [1]). If a second request's prompt is waiting, put its prefill chunk into the same step as the first request's decode token. Each streamed layer then serves both. This is "scheduling of weight streaming". The paper's model shows why: the a_pf term adds per token, b is paid once per step.
3. **Small M changes the priority.** In mu_j, the term b·K̃_j/M grows when M is small. With small RAM, the priority is driven by memory-time volume K̃_j (≈ SVF). If a local agent runs several requests in parallel, run the short or low-KV requests first.
4. **The future KV-feasibility admission check** is directly useful. Before we start a request, check that its KV at full length (s + o tokens) fits in the KV budget together with the running requests. This avoids out-of-memory errors or forced eviction in the middle of a generation. For a single user it is a simple check: "will this context fit at the end of generation?".
5. Nothing in the paper covers CPU, offloading, SSD, or weight-streaming schedules. The idea of streaming weights and planning a layer pipeline is not treated.

---

## 3. ZipServ: hardware-aware lossless BF16 weight compression

### 3.1 Problem

Paper (p.1-4): lossless compression keeps bit-exact outputs, but existing GPU codecs (DFloat11 with Huffman, DietGPU and nvCOMP with ANS) are slow. Two reasons:
- **Kernel level:** variable-length bitstreams need serial, data-dependent decoding. This causes warp divergence. On L40S, DietGPU reaches 43.7% and DFloat11 reaches 76.5% of peak memory bandwidth (p.3).
- **System level:** they decompress the full weights into a global-memory buffer before the GEMM. This adds memory traffic. The decoupled decompression alone takes 1.56-3.44× the time of the GEMM (Fig. 1, p.2). Roofline analysis: the decoupled pipeline has about 62% lower compute intensity than plain GEMM at batch 8-64 for a 4096 × 4096 matrix (p.4).

### 3.2 Key observation: BF16 exponents have low entropy and are contiguous

Paper (p.3):
- BF16 = 1 sign + 8 exponent + 7 mantissa bits.
- In Llama-3-8B, Mistral-Small-24B and Qwen2.5-32B, the top-3 exponents cover > 67% of weights. The top-7 cover > 95% (96.4% in Llama-3, 97.4% in Mistral-24B).
- Exponent entropy is only **2.57-2.74 bits**. So the theoretical lossless limit is about 16 / 10.6 ≈ **1.51×** (8 bits for sign and mantissa plus about 2.6 bits).
- Over 3,875 weight matrices from Gemma-3, Mistral, Qwen2.5 and LLaMA-3.1: in **99.6%** of matrices, the top-7 exponents are a contiguous run (e, e+1, ..., e+6). A contiguous window covers 97.1% of weights on average.
- Appendix A proves why: if the weights are N(0, σ²), the exponent probability P(E = k) = erf(2^{k+1}/(σ√2)) − erf(2^k/(σ√2)) is unimodal in k. A top-K set of a unimodal distribution is contiguous (Theorems A.1, A.2, p.12-13).

### 3.3 Core mechanism: TCA-TBE format

**Offline compressor (Algorithm 1, p.5)**, per weight matrix:
1. Compute the exponent histogram.
2. Select the window of 7 consecutive exponent values with the highest coverage.
3. Set `BaseExp = min(window) − 1`.
4. For each 8 × 8 tile (64 weights):
   - If the exponent e is in the window: code = e − BaseExp (1..7). Write the 3 code bits into three 64-bit bitmaps B1, B2, B3 (one bit plane each) at the element's position. Push the packed 8-bit (sign + 7-bit mantissa) into the high-frequency buffer H.
   - Else: code = 000 (all three bitmap bits are 0). Push the full 16-bit BF16 value into the fallback buffer L.
5. Store the bitmaps and buffers in global arrays, following the tile hierarchy. Pad the buffers to 128-bit alignment. Keep an Offset array with the start of each group of tiles in H and L.

**Tile hierarchy (p.6):** FragTile 8 × 8 (the smallest Tensor Core operand fragment) → TensorCoreTile 16 × 16 (2 × 2 FragTiles in column-major order, matching the register layout of `mma.m16n8k16`) → BlockTile 64 × 64 (one thread block).

**Per-element cost (p.5):**

```
c(b) = p · (b + 8) + (1 − p) · (b + 16)
b = 3, p ≈ 0.96  →  ≈ 11.3 bits per weight
(2-bit code: 12.4 bits; 4-bit code: 12.1 bits; entropy bound: 10.6 bits)
```

**Decompression per thread (Algorithm 2, p.7)**. Each thread needs 2 elements (positions 2·lane and 2·lane + 1) for its `bf16x2` register:
1. `M = B1 | B2 | B3`. Bit = 1 means compressed; bit = 0 means fallback.
2. For element position i: `mask = (1 << i) − 1`; `idxH = popc(M & mask)`.
3. If bit i of M is 1: read the packed sign+mantissa at `H[startH + idxH]`. Rebuild `code = (B3[i] << 2) | (B2[i] << 1) | B1[i]`. Set `exp = BaseExp + code`. Assemble the BF16 value.
4. Else: `idxL = i − idxH`. Read `L[startL + idxL]`.
5. Pack the two values into one `bfloat162` register. Feed it to `mma.sync`.

There is no lookup table and no branch divergence. Only bit operations, popcount and one integer add are used.

**ZipGEMM kernel (p.6-7):** split-K tiling.
1. Load compressed tiles and activations with `cp.async` (LDGSTS.128) into shared memory.
2. Each warp decodes from shared memory into registers.
3. Load activations with `LDSM.M88`.
4. Run `mma`.

There is a two-level pipeline: double-buffered tiles (global → shared) and slice-level interleaving. Tensor cores compute slice k while the ALUs decode slice k+1.

**Stage-aware use (p.8):** in decode (small N), use fused ZipGEMM. In prefill (large N, e.g. 8192), decompress to global memory first and then call cuBLAS. The overhead is 4% / 2% of GEMM time at N = 8192 / 16384 (p.9).

### 3.4 Memory result

- Weight footprint (p.9): LLaMA3.1-8B 14.96 → **10.83 GB (72.4%)**. Mistral-24B 43.92 → 31.30 GB (71.3%). LLaMA3.1-70B 131.56 → **93.52 GB (71.1%)**. The abstract says "up to 30%" reduction.
- The breakdown on RTX4090 (p.9) gives a different 8B number: 14.96 → 11.18 GB (3.78 GB saved). The freed memory raised the vLLM KV cache from 5.07 GB to 8.60 GB (1.70×). The paper does not explain the 10.83 vs 11.18 GB difference.
- Offline compression time: about 2.5 min for LLaMA-3.1-8B on a 16-core Xeon 8352V (p.9).

### 3.5 Speed result and hardware

Hardware: 4× RTX4090 (24 GB) with Xeon 8352V; 4× L40S (48 GB) with Xeon Gold 6230R; also RTX5090 (32 GB). NVCC 12.4 / 12.8.

- Kernel, batch sizes 8/16/32 (Fig. 11, p.8): vs cuBLAS_TC, average **1.31×** (max 1.71×) on RTX4090 and **1.36×** (max **2.21×**) on L40S. In the same setup, DietGPU / nvCOMP / DFloat11 reach only 0.17×/0.20×, 0.19×/0.23× and 0.28×/0.34×. These are slowdowns.
- Layer level on L40S, LLaMA3.1 family: GateUp_proj 1.39×, Down_proj 1.64×. O_proj of 8B: **0.79×** (a slowdown on small shapes). Block level: 1.35× (8B) and 1.48× (405B).
- NCU profile on RTX4090 (M = 28672, K = 4096, N = 32): DRAM reads −29.3%. ALU utilization 66.0%. Tensor Core utilization at 71.6% of cuBLAS. Shared-memory bank conflicts 4.7K (p.9).
- Standalone decompression kernel: 2.14×, 1.83× and 1.10× faster than DietGPU, nvCOMP and DFloat11 (p.9).
- RTX5090: 1.34× (8B) and 1.87× (Mistral-24B) over cuBLAS. RTX4090 ZipGEMM 0.195 ms vs A100 cuBLAS 0.215 ms on 8B GateUp (p.9).
- vs lossy Marlin W8A16 (FP8) on RTX4090, 28672 × 4096, batch 32: 0.194 ms vs 0.143 ms. The 1.36× gap is close to the bit-width ratio of 11 bits vs 8 bits (p.10).
- End-to-end in vLLM (8B on 1× RTX4090; 24B on 2× L40S; 70B on 4× L40S with tensor parallelism; batch 8 and 32; outputs 128-2048 tokens):
  - Latency −17.60% vs vLLM, −60.79% vs Transformers, −82.13% vs DFloat11.
  - Throughput 1.22× / 3.18× / 8.52×.
  - Peak: 1105 tok/s (1.66× over vLLM) at batch 32, 2048 outputs, 8B.
  - Linear layers: 24.99 ms → 14.76 ms (1.69×) (p.9).

### 3.6 Accuracy cost

None. Bit-exact.

### 3.7 Limitations and assumptions

- BF16 weights only. The design depends on the 8-bit exponent field.
- NVIDIA Tensor Core GPUs (Ada, Blackwell). On A100 and H800, ZipGEMM "may not always match" cuBLAS. Reasons: high HBM bandwidth, and a lower clock (1410 MHz vs 2520 MHz) makes the ALU work harder to hide (Fig. 18, p.10).
- Evaluated batch sizes are 8, 16 and 32. Batch 1 kernel numbers are not reported. Analyst view: batch 1 is even more memory-bound, so the relative gain should be similar or larger. The paper does not show this.
- Small layers can be slower (0.79×). Tuning is out of scope.
- Maximum saving is about 28-30%. The theoretical limit is 1.51×.
- One exponent window per matrix. A matrix with a wide or bimodal exponent spread compresses less. The paper reports 99.6% contiguity, but the spread of per-matrix ratios is not stated.
- Intel AMX, AMD and KV-cache use are future work (p.10). CPU decoding is not implemented.

### 3.8 Relevance to our goal

**(a) Can BF16 exponent entropy shrink weights on disk or in RAM?** Analyst view: yes, technically. The format is fixed-length and cheap to decode (popcount, shifts, one add). A CPU with AVX-512 (VPOPCNTQ, PDEP/PEXT) could decode it near memory speed. This is my judgement; the paper has no CPU implementation. Disk, RAM and SSD-streaming bytes would fall by about 28-29%. But the absolute size stays far too big for us:
- 70B in BF16: 131.56 GB → 93.52 GB with ZipServ (paper). 4-bit quantization gives about 35 GB plus scales.
- 700B in BF16: about 1.4 TB → about 0.99 TB (derived at 71%). 4-bit gives about 350 GB.

So ZipServ does not solve our capacity problem. Lossy low-bit quantization is required. Lossless BF16 compression is useful only for a "reference / bit-exact" mode, or for the few tensors we keep in high precision (embeddings, lm_head, norms, maybe attention projections in a mixed scheme).

**(b) Does it compose with low-bit quantization?** Analyst view: mostly no, for these reasons.
1. The redundancy that ZipServ uses is the **dynamic range carried by the floating-point exponent**. Weights are about Gaussian, so most of them share a few exponents. Group-wise quantization (GPTQ, AWQ, GGUF K-quants) removes exactly this redundancy. The per-group scale stores the magnitude. The remaining INT4 codes are integers with **no exponent field**. TCA-TBE has nothing to work on.
2. A good quantizer tries to use all its levels. The entropy of 4-bit codes is therefore close to 4 bits. The residual lossless gain is small. The paper does not measure it.
3. For FP8 (E4M3) weights, there is a 4-bit exponent. A similar "top-k exponent window + fallback" trick could save part of 1-2 bits. Not measured in the paper.
4. The paper claims (p.10) that ZipServ "is orthogonal to lossy methods and can be applied atop quantized weights to exploit residual redundancy" and cites DECA [26]. It shows no experiment for this. Treat it as unproven.

**(c) What does transfer:**
- The principle "**load compressed, compute decompressed**": never write the decompressed weights to memory. Decode them in registers right before the dot product. llama.cpp quantized kernels already follow this principle. It confirms that a decoupled decompress-to-buffer step is a bad design for us. This matters most when weights stream from SSD.
- **Fixed-length bit-plane layout plus a sparse outlier buffer, addressed by popcount.** This is a good pattern for any mixed format, for example 3-bit codes plus a few FP16 outliers. It is branch-free and easy to vectorize.
- **Stage-aware decode:** use the fused path for decode (memory-bound) and decompress then GEMM for prefill (compute-bound).
- The **roofline argument**: in memory-bound decode, the speed-up is about the compression ratio. For us, with SSD streaming, every 1% fewer bytes is about 1% faster per token.

---

## 4. Cross-paper notes for our design

1. **KV cost for 70B at 32K (10 GiB FP16) is the same size as the whole VRAM budget.** KV compression is not optional for long context. PuzzleKV+INT4 (about 19%) or plain INT3/INT4 KV (22-28%) are the candidates. Plain INT4 KV had better accuracy than PuzzleKV+INT4 at a 1.5× larger budget (Table 3). Start with plain KV quantization. Add page-wise low rank only if we need the extra 1.5×.
2. **Weight bytes per step dominate** for offloaded inference (cμ paper's b term, ZipServ's roofline). Every method that puts more tokens into one weight pass helps: prefill/decode co-batching, speculative decoding, and parallel sub-requests.
3. **Lossless BF16 compression (ZipServ, DFloat11) does not reach our size targets.** Use it only for high-precision tensors.
4. **None of the three papers covers CPU inference, SSD offload or weight streaming.** None cites FlexGen or "LLM in a flash". ZipServ cites PowerInfer (consumer GPU) and an Intel AMX paper. Those are the closest to our setting.

---

## 5. Blind spots: cited or depended-on papers we do not have

Priority: **H** = high, **M** = medium, **L** = low for our goal.
"Ref" = the arXiv ID is printed in the paper's reference list. "Memory" = the paper gives no arXiv ID; the ID is from my own knowledge and must be verified before use.

### 5.1 KV-cache compression (from PuzzleKV)

| Priority | Title | ID / venue | Source |
|---|---|---|---|
| H | KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache (Liu et al.) | ICML 2024; arXiv 2402.02750 (Memory) | PuzzleKV [12]. It is the INT3/INT4 KV baseline that beat PuzzleKV+INT4 on Llama |
| H | Palu: KV-Cache Compression with Low-Rank Projection (Chang et al.) | ICLR 2025; arXiv 2407.21118 (Memory) | PuzzleKV [2] |
| H | Efficient Streaming Language Models with Attention Sinks (StreamingLLM; Xiao et al.) | ICLR 2024; arXiv 2309.17453 (Memory) | PuzzleKV [21]. The sink-page idea comes from it |
| H | TurboQuant: Online Vector Quantization with Near-Optimal Distortion Rate (Zandieh et al.) | ICLR 2026, OpenReview tO3ASKZlok (Ref) | PuzzleKV [25] |
| M | xKV: Cross-Layer KV-Cache Compression via Aligned Singular Vector Extraction | ICML 2026; **arXiv 2503.18893** (Ref) | PuzzleKV [3] |
| M | PolarQuant: Quantizing KV Caches with Polar Transformation | **arXiv 2502.02617** (Ref) | PuzzleKV [7] |
| M | SnapKV: LLM Knows What You Are Looking for Before Generation | NeurIPS 2024; arXiv 2404.14469 (Memory) | PuzzleKV [11] |
| M | H2O: Heavy-Hitter Oracle for Efficient Generative Inference of LLMs | NeurIPS 2023; arXiv 2306.14048 (Memory) | PuzzleKV [27] |
| M | LoRC: Low-Rank Compression for LLMs KV Cache with a Progressive Compression Strategy | **arXiv 2410.03111** (Ref) | PuzzleKV [26] |
| M | Effectively Compress KV Heads for LLM (ECKVH) | **arXiv 2406.07056** (Ref) | PuzzleKV [24] |
| M | FlashSVD: Memory-Efficient Inference with Streaming for Low-Rank Models | AAAI 2026 (Ref, no arXiv ID) | PuzzleKV [14] |
| M | CacheGen: KV Cache Compression and Streaming for Fast LLM Serving | SIGCOMM 2024 (Ref, no arXiv ID) | ZipServ [45]. KV streaming is relevant to offload |
| L | OjaKV: Context-Aware Online Low-Rank KV Cache Compression | Findings ACL 2026 (Ref) | PuzzleKV [28] |
| L | Eigen Attention: Attention in Low-Rank Space for KV Cache Compression | Findings EMNLP 2024 (Ref) | PuzzleKV [13] |
| L | A Survey on LLM Acceleration based on KV Cache Management (Li et al.) | TMLR 2025 (Ref) | PuzzleKV [10]. Useful as a map of the field |

### 5.2 Lossless and lossy weight compression (from ZipServ)

| Priority | Title | ID / venue | Source |
|---|---|---|---|
| H | 70% Size, 100% Accuracy: Lossless LLM Compression for Efficient GPU Inference via Dynamic-Length Float (DFloat11) | **arXiv 2504.11651** (Ref) | ZipServ [85] |
| H | DECA: A Near-Core LLM Decompression Accelerator Grounded on a 3D Roofline Model | MICRO 2025 (Ref, no arXiv ID) | ZipServ [26]. Only evidence cited for "lossless on top of quantized weights" |
| H | PowerInfer: Fast LLM Serving with a Consumer-grade GPU | SOSP 2024; arXiv 2312.12456 (Memory) | ZipServ [64]. Closest to our hardware setting |
| H | Exploiting Intel Advanced Matrix Extensions (AMX) for LLM Inference | IEEE CAL 2024 (Ref) | ZipServ [37]. CPU inference |
| M | Huff-LLM: End-to-End Lossless Compression for Efficient LLM Inference | **arXiv 2502.00922** (Ref) | ZipServ [83] |
| M | ZipNN: Lossless Compression for AI Models | **arXiv 2411.05239** (Ref) | ZipServ [29]. Disk/checkpoint compression |
| M | NeuZip: Memory-Efficient Training and Inference with Dynamic Compression | **arXiv 2410.20650** (Ref) | ZipServ [28] |
| M | MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on LLMs | PPoPP 2025 (Ref) | ZipServ [24]. Fused quantized GEMM |
| M | QuaRot: Outlier-Free 4-bit Inference in Rotated LLMs | **arXiv 2404.00456** (Ref) | ZipServ [4] |
| M | SpinQuant: LLM Quantization with Learned Rotations | **arXiv 2405.16406** (Ref) | ZipServ [46] |
| M | Does Quantization Affect Models' Performance on Long-Context Tasks? | **arXiv 2505.20276** (Ref) | ZipServ [49]. Up to 11.1% loss from GPTQ-int8 on NOCHA |
| M | Quantization Hurts Reasoning? An Empirical Study on Quantized Reasoning Models | **arXiv 2504.04823** (Ref) | ZipServ [44]. MXFP4 LiveCodeBench 56.0 → 36.2 |
| M | LUT-GEMM: Quantized Matrix Multiplication based on LUTs | ICLR 2024 (Ref) | ZipServ [55] |
| M | Ecco: Improving Memory Bandwidth and Capacity for LLMs via Entropy-Aware Cache Compression | ISCA 2025 (Ref) | ZipServ [5] |
| L | DietGPU (github facebookresearch/dietgpu), nvCOMP (github NVIDIA/nvcomp) | software (Ref) | ZipServ [33], [53] |
| L | Lossless Compression for LLM Tensor Incremental Snapshots (LMC) | **arXiv 2505.09810** (Ref) | ZipServ [71] |
| L | Atom: Low-bit Quantization for Efficient and Accurate LLM Serving | MLSys 2024 (Ref) | ZipServ [88] |
| L | Flash-LLM (unstructured sparsity), SpInfer (EuroSys 2025), Ladder (OSDI 2024) | (Ref) | ZipServ [77], [21], [72] |

### 5.3 Serving systems and scheduling (from all three)

| Priority | Title | ID / venue | Source |
|---|---|---|---|
| H | Efficient Memory Management for LLM Serving with PagedAttention (vLLM) | SOSP 2023; arXiv 2309.06180 (Memory) | All three papers |
| H | Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve (chunked prefill + mixed batching) | OSDI 2024; arXiv 2403.02310 (Memory) | cμ [1], ZipServ [1]. Key for prefill/decode co-scheduling |
| H | Efficiently Scaling Transformer Inference (Pope et al.) | MLSys 2023 (Ref) | cμ [32]. Cost model for KV vs weight bandwidth |
| M | FlashAttention / FlashAttention-2 / FlashAttention-3 | NeurIPS 2022; ICLR 2024; NeurIPS 2024 (Ref) | PuzzleKV [4], ZipServ [9],[10],[62]. Online-softmax merge |
| M | DistServe: Disaggregating Prefill and Decoding | OSDI 2024 (Ref) | ZipServ [90] |
| M | Optimizing LLM Inference: Fluid-Guided Online Scheduling with Memory Constraints (WAIT) | **arXiv 2504.11320** (Ref) | cμ [4] |
| M | Online Scheduling for LLM Inference with KV Cache Constraints (MC-SF) | **arXiv 2502.07115** (Ref) | cμ [23] |
| M | Geometry-Aware Online Scheduling for LLM Serving (SVF) | **arXiv 2606.22327** (Ref) | cμ [24] |
| M | Competitive Non-Clairvoyant KV-Cache Scheduling for LLM Inference | **arXiv 2601.22996** (Ref) | cμ [13]. No output-length knowledge needed |
| M | General Non-Clairvoyant KV-Cache Scheduling via Regime-Aware Routing | **arXiv 2607.09248** (Ref) | cμ [12] |
| L | Orca: A Distributed Serving System for Transformer-Based Generative Models | OSDI 2022 (Ref) | cμ [47], ZipServ [82] |
| L | SGLang: Efficient Execution of Structured LM Programs | NeurIPS 2024 (Ref) | ZipServ [89] |
| L | DeepSpeed-FastGen | **arXiv 2401.08671** (Ref) | ZipServ [30] |
| L | Service-Induced Congestion in Memory-Constrained LLM Serving | **arXiv 2606.15555** (Ref) | cμ [3] |
| L | Throughput-Optimal Scheduling Algorithms for LLM Inference and AI Agents | **arXiv 2504.07347** (Ref) | cμ [11] |
| L | A Queueing-Theoretic Framework for Stability Analysis of LLM Inference with KV Cache Memory Constraints | **arXiv 2605.04595** (Ref) | cμ [29] |
| L | Large-Scale LLM Inference with Heterogeneous Workloads: Prefill-Decode Contention | **arXiv 2602.02987** (Ref) | cμ [26] |
| L | LLM Serving Optimization with Variable Prefill and Decode Lengths | **arXiv 2508.06133** (Ref) | cμ [41] |
| L | Adaptively Robust LLM Inference Optimization under Prediction Uncertainty | **arXiv 2508.14544** (Ref) | cμ [9] |
| L | Multi-turn LLM Conversations under the LRU Policy: Mean-Field Asymptotics | **arXiv 2609.02027** (Ref) | cμ [46] |
| L | KVCache Cache in the Wild (Qwen-Bailian traces) | USENIX ATC 2025 (Ref) | cμ [40] |

### 5.4 Not cited by any of the three, but important for us

| Priority | Title | ID | Note |
|---|---|---|---|
| H | FlexGen: High-Throughput Generative Inference of LLMs with a Single GPU | arXiv 2303.06865 (Memory) | Offload scheduling of weights/KV/activations across GPU, CPU and disk. It is not cited by any of the three papers. |
| H | LLM in a flash: Efficient LLM Inference with Limited Memory | arXiv 2312.11514 (Memory) | SSD weight streaming. Not cited. |
| M | DeepSeek-V2/V3 technical reports (MLA latent KV) | arXiv 2405.04434 / 2412.19437 (Memory) | For 700B-class models, the KV design changes. PuzzleKV is not tested on MLA. |

---

## 6. Open questions for the team

1. Can PuzzleKV run compress-during-chunked-prefill, so that the prefill peak is not the full dense KV? This needs a prototype and a RULER check.
2. Does PuzzleKV accuracy hold for 70B models and at 64K-128K? The paper does not test this.
3. Plain INT3/INT4 KV vs PuzzleKV+INT4: which gives better accuracy per byte on our target models? The paper's own Table 3 favours plain INT3 on Llama3.1.
4. Residual entropy of GGUF/INT4 codes: measure it on our models before we spend effort on lossless layers over quantized weights.
5. Weight-streaming scheduler: model the step time as `a_pf·N_pf + a_kv·N_kv + b_stream`. Then measure how many tokens per weight pass (prefill chunk + decode + draft tokens) make the streaming cost acceptable.
