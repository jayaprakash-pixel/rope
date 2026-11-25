# G. Offloading, streaming and MoE papers (12 gap papers)

Analyst G. Date: 2026-11-12. Scope: the 12 G-group papers in `papers\gaps\`.
Method: I converted each PDF with `pdftotext -layout` and read the method, system and experiment sections in full. I scanned related work and references for blind spots.
Rules for this file:
- A number with a table or figure label is from the paper text.
- **[est]** marks my own estimate for this machine. It is not a measurement.
- "not stated" means the paper text does not give the fact. Many papers give absolute tok/s only in bar charts. The text layer of a bar chart is not reliable, so I do not quote those bars.
- Some two-column tables came out of `pdftotext` with mixed rows. Where I reconstructed a table, I say so.

Our machine (from `00_SYNTHESIS.md`): 8 GB single-channel DDR4 (~20 GB/s, ~4.5 GB usable), GTX 1650 4 GB (~128 GB/s, ~3.3 GB usable, Turing, no BF16), PCIe ~10-12 GB/s, NVMe ~5-6.5 GB/s sequential, 82 GB free disk, Ryzen 5 5600H (AVX2, no AVX-512).

Two reference MoE models for the [est] numbers below. The configs are from my memory of the model cards. Verify them before use.
- **Qwen3-30B-A3B**: 48 layers, 128 experts per layer, top-8, hidden 2048, expert FFN 768. One expert = 3 x 2048 x 768 = 4.7M parameters = ~2.8 MB at ~4.8 bpw. One token touches 48 x 8 = 384 experts = **~1.1 GB** of expert bytes.
- **gpt-oss-120b**: 36 layers, 128 experts, top-4, hidden 2880, expert FFN 2880. One expert = 24.9M parameters = ~13 MB at MXFP4. One token touches 144 experts = **~1.9 GB**.

---

## 1. Per-paper analysis

### G01. FlexGen (Sheng et al., ICML 2023, arXiv 2303.06865)

**Problem and hardware.** High-throughput, latency-insensitive batch generation with one GPU. Hardware: one NVIDIA T4 (16 GB), Intel Xeon 2.0 GHz with 208 GB DRAM, 1.5 TB cloud NVMe SSD (Table 1). The SSD reads at ~1.6 GB/s and writes at ~1.3 GB/s measured with `dd` (Table 24). PCIe CPU-GPU is drawn as 12 GB/s, disk-CPU as 2 GB/s (Fig. on p.2).

**Mechanism.**
- The paper treats inference as a traversal of a grid (batches x layers x tokens) (Fig. 2). Existing systems go row by row: they finish one batch and reload every weight for each batch. FlexGen goes column by column in a **zig-zag block schedule** (Fig. 3b). It keeps one layer's weights on the GPU and runs many GPU batches through it before it loads the next layer. The block schedule is proved to be within 2x of the optimal I/O complexity (Theorem 4.1).
- Overlap (Algorithm 1): in the inner loop it runs six operations at the same time. It loads the next layer's weights, stores the activation and KV cache of the previous batch, loads the KV cache and activation of the next batch, and computes the current batch.
- Placement: 9 continuous variables give the percent of weights (wg, wc, wd), activations (hg, hc, hd) and KV cache (cg, cc, cd) on GPU, CPU and disk. Weights are placed at layer granularity; activations and KV cache at tensor granularity.
- Policy search: a cost model gives per-layer time as `max(ctog, gtoc, dtoc, ctod, comp)`, which assumes perfect overlap. FlexGen enumerates a few (block size, GPU batch size) pairs. For each pair it solves a linear program over the 9 placement variables with memory constraints (Eq. 1). Hardware constants come from profiling.
- CPU delegation: when the KV cache is in CPU memory, attention runs on the CPU. This moves b x h1 x 4 bytes of activation instead of b x s x h1 x 4 bytes of KV cache, so I/O drops by a factor s (§4.2).
- Compression: group-wise asymmetric min-max quantization to 4 bits, group size 64, for weights (grouped along output channel) and KV cache (grouped along hidden dim). Tensors are stored compressed and dequantized to FP16 before compute. CPU delegation is turned off with compression because dequant on the CPU is too slow (§5). 3-bit did not keep accuracy (§6.2).
- Disk tensors are files mapped into virtual memory (§6, Implementation).

**Results.**
- Table 2, prompt 512, output 32, 1 GPU: OPT-6.7B 25.26 tok/s, OPT-30B 7.32, OPT-175B **0.69 tok/s**. With 4-bit: 29.12 / 8.70 / **1.12 tok/s**. DeepSpeed ZeRO-Inference: 9.28 / 0.60 / 0.01. Accelerate: 25.12 / 0.62 / 0.01.
- OPT-175B policy: GPU batch 32 x 8 batches = effective batch 256, 0% weights on GPU, 50% on CPU, rest on disk (Table 23). Effective batch 144 with compression, which removes disk I/O.
- Ablation (Table 4): OPT-175B drops from 0.69 to 0.27 without policy search, 0.59 without overlap, 0.62 without CPU compute.
- SSD speed (Table 24): OPT-175B 0.69 tok/s at 1.6 GB/s and 0.30 tok/s at 0.5 GB/s. OPT-30B is unchanged because it does not use the disk.
- GPU utilization during decode is 13% (Appendix A.4, Table 8). Decode is I/O-bound.
- Accuracy (Table 5): OPT-175B Lambada 0.758 FP16, 0.756 4-bit; WikiText PPL 10.82 vs 10.94.

**Assumptions that do not hold for us.** Very large batch (64-256 sequences). 208 GB RAM. Latency of thousands of seconds is acceptable. Dense OPT models. Prompts padded to equal length.

**What transfers.** (1) The cost-model + LP planner idea: per-layer time = max over link times, solved for placement. This is cheap to build and fits our 4-tier machine. (2) Overlap of next-layer load with current compute (Algorithm 1). (3) For **offline batch jobs only** (evaluation, bulk summarization), the zig-zag schedule amortizes each expert read over many sequences. For chat, it does not apply. (4) The SSD sensitivity in Table 24 confirms the law in `00_SYNTHESIS.md` §1: throughput scaled almost linearly with disk bandwidth. Expected gain for chat on our machine: none from the schedule itself; the planner is the useful part.

---

### G02. LLM in a Flash (Alizadeh et al., Apple, ACL 2024, arXiv 2312.11514)

**Problem and hardware.** Run a model that is ~2x larger than DRAM by reading FFN weights from flash on demand, batch 1. Hardware: Apple M1 Max with 1 TB SSD, Apple M2 Ultra with 2 TB SSD, and a Linux box with RTX 4090 24 GB (§4.1). The M1 Max SSD reads >6 GiB/s for a 1 GiB sequential read (§2.2). About half of the model size is the DRAM budget in all runs.

**Mechanism.**
- **Selective persistence:** embeddings and attention weights (~1/3 of the model) stay in DRAM. Only FFN rows/columns are streamed (§3.1).
- **Sparsity predictor:** a low-rank predictor per layer (Fig. 3b: dmodel -> rank r -> dffn, sigmoid > 0.5). Its input is the output of the **current** layer's attention, not the previous layer. OPT-6.7B uses r = 128 for the first 28 layers and r = 1024 for the last 4. It gives 5% false negatives and 7% false positives (App. C.1). The predicted active set is ~3x the true active set in OPT and Persimmon and ~2x in Phi-2 (App. B.1). Average rank 240 = <2.4% of non-embedding weights; cost 2.75% of inference time on M1 Max CPU and 4.8% on the 4090 (App. B.3). Training: 10,000 C4 samples, 2 epochs, 4 h per predictor on one A100 (§3.1).
- **Requires ReLU-type sparsity.** OPT-6.7B has 97% FFN sparsity; Falcon-7B was fine-tuned to ReLU (95%); Llama 2 uses the FATReLU-sparsified version (>90%); Persimmon has squared ReLU (§3.1, App. C.5). The paper states that replacing Swish with ReLU in SwiGLU models does not give high sparsity without such methods (App. C.5).
- **Windowing:** keep in DRAM the neurons that were active for the last k tokens (k = 4 or 5). For each new token, load only the difference `s_agg(k+1) - s_agg(k)` (§3.1, Fig. 4). For OPT-6.7B, per-token load drops from ~10% to 2.4% of FFN neurons, at a DRAM cost of 24% of the FFN (App. C.1).
- **Row-column bundling:** column i of the up projection and row i of the down projection are used together. Store them next to each other. This doubles the read chunk from dmodel x bytes to 2 x dmodel x bytes. For OPT-6.7B FP32 the chunk is 32 KiB (§3.2, Fig. 5).
- **Co-activation bundling failed.** Bundling each neuron with its most co-active neuron loaded hot neurons many times, because hot neurons are the "closest friend" of almost every other neuron (§3.2, App. D).
- **Read throughput:** random reads need ≥32 KiB chunks and many threads; the implementation uses 32 threads (Fig. 2b, §4.1). OS caching is turned off for measurement (`F_NOCACHE` on macOS, DirectIO on Linux) (App. C).
- **DRAM data structure (Fig. 6):** one preallocated matrix per layer of size `Req_i x 2 dmodel`, a pointer array (original neuron index), bias, `num_used`, `last_k_active`. Delete: move the last rows into the holes (O(c x dmodel) copy). Insert: append at `num_used`. No reallocation. The first half of each row is the up-projection row; the second half, transposed, is the down projection.

**Results.**
- Table 2, OPT-6.7B FP16 on M1 Max, I/O latency per token: naive 13.4 GB at 6.10 GB/s = 2196 ms; hybrid (half in DRAM) 1090 ms; + predictor 0.9 GB at 1.25 GB/s = 738 ms; + windowing 0.2 GB at 1.25 GB/s = 164 ms; + bundling 0.2 GB at 2.25 GB/s = **87 ms**. Scattered reads ran at 1.25 GB/s against 6.1 GB/s sequential; bundling raised this to 2.25 GB/s.
- Table 3 end-to-end per token: OPT-6.7B CPU 3182 ms naive -> 669 ms; Metal M1 2389 -> 565 ms; Metal M2 2270 -> 305 ms; RTX 4090 GPU 2218 -> **84 ms**, and 60 ms with speculative decoding (λ = 4, 1.4x). Falcon-7B CPU 3095 -> 706 ms. Phi-2 1287 -> 546 ms. Llama 2 7B (FATReLU) 3095 -> 994 ms.
- Accuracy: OPT-6.7B zero-shot is unchanged (Arc-Easy 66.1 -> 66.2, HellaSwag 50.3 -> 49.8, Table 1). Relufied Falcon loses ~2 points before predictors (Table 4).
- Latency stays flat over 1000 generated tokens (Fig. 8).

**Assumptions that do not hold for us.** ReLU-sparse dense models. Apple unified memory (no PCIe copy). Fast Apple SSD with deep parallel queues. Half the model in DRAM (we have ~4.5 GB). FP16/FP32 weights. With 4-bit weights, a bundled row-column pair of a 2048-wide expert is only ~2 KB, far below the 32 KiB efficient read size.

**What transfers.**
1. **Read-size discipline.** Every SSD read should be ≥32 KiB and many reads must be in flight. For MoE, one expert (2.8-13 MB) is already a large read. Keep gate, up and down of one expert contiguous: this is row-column bundling at expert scale.
2. **Bypass the page cache** (direct I/O) and own the cache in the engine. The paper shows that the OS cache gives near-zero hit rate when layers are evaluated in a rolling pattern (App. C).
3. **Windowing = "keep what the last k tokens used".** For experts this is an LRU cache with a k-token horizon. It is a good baseline policy.
4. **Preallocated slot matrix with swap-delete and append-insert** (Fig. 6). Use the same for a RAM expert slot pool: no allocation in the hot path.
5. Neuron-level streaming itself does not transfer to our SwiGLU MoE targets (see §3, question 5). Expected gain from items 1-4: they are preconditions for reaching the SSD ceiling, not a multiplier on top of it.

---

### G03. PowerInfer (Song et al., SOSP 2024, arXiv 2312.12456)

**Problem and hardware.** Batch-1 inference of a model larger than VRAM on one consumer GPU plus CPU. Hardware: PC-High = i9-13900K, 192 GB DRAM (67.2 GB/s), RTX 4090 24 GB, PCIe 4.0. PC-Low = i7-12700K, 64 GB DRAM (38.4 GB/s), RTX 2080Ti 11 GB, PCIe 3.0 (§8.1). The whole model is in RAM + VRAM. No SSD streaming.

**Mechanism.**
- **Power-law activation (Insight 1):** in OPT-30B (ReLU), LLaMA2(ReGLU)-70B and LLaMA2(SwiGLU)-70B, 26%, 43% and 69% of neurons in one MLP give 80% of activations. Across the whole model: 17%, 26% and **75%** (Fig. 4). The hot set overlaps >90% across tasks (§3.1). Note the SwiGLU case: 75% of neurons are needed for 80% of activations. The skew is weak.
- **CPU compute beats transfer (Insight 2):** for batch < 32, computing CPU-resident neurons on the CPU with AVX2 is faster than copying them to the GPU (Fig. 6).
- **Offline:** profile activation counts on general corpora (C4, Wikipedia). An ILP assigns each neuron to GPU or CPU to maximize total activation frequency on GPU, under a GPU memory constraint and a per-layer minimum count that pays for one CPU-GPU sync (Eq. 2-8). Neurons are grouped in batches of 64 to cut solve time to ~10 s (§6.3.3).
- **Online:** per-layer adaptive MLP predictors (one hidden layer; the hidden size is tuned until PPL is within 0.1% of the base). Predictors are ~6-8% of model parameters: OPT-13B 0.88B (6.71%), OPT-66B 3.23B (4.88%), Falcon-40B 3.63B (8.68%), LLaMA(ReGLU)-70B 5.66B (8.08%) (Table 6). Predictor time is <10% of inference (§8.3.4). For contrast, fixed-size Deja Vu predictors for OPT-175B need ~27 GB (§5.1).
- GPU and CPU each compute their own predicted-active neurons at the same time. The CPU sends its partial result to the GPU, which merges (§4.2, §5.3). A DAG of operators with GPU and CPU executor threads manages dependencies.
- **Neuron-aware operators:** they compute only active rows/columns without converting to sparse formats. The CPU operator beats dense even below 10% sparsity; PyTorch sparse needs >87% (Fig. 16).
- Neuron tables map each neuron to its original position (9 MB for OPT-175B) (§5.2). KV cache stays in CPU memory.

**Results.**
- FP16 on PC-High: average **8.32 tok/s**, up to 16.06 tok/s; average 7.23x over llama.cpp, up to 11.69x for Falcon-40B (§8.2, Fig. 10).
- INT4 on PC-High: average **13.20 tok/s**, peak 29.08 tok/s; OPT-175B INT4 "nearly two tokens per second", 2.66x llama.cpp (§8.2, Fig. 13). PC-Low INT4: average 5.01x, up to 8.00x.
- Ablation (Fig. 15): Falcon-40B 3.32x (predictor + operators) -> 7.80x (+ hybrid engine) -> 11.69x (+ ILP policy).
- **SwiGLU models (Table 8):** Yi-34B 1.7 tok/s vs 1.0 for llama.cpp (**1.7x**) on PC-High; LLaMA2-13B 3.1 vs 2.1 (**1.5x**) on PC-Low. Compare OPT-30B 12.0 vs 1.1 (10.9x).
- Long prompts: prefill switches to dense GPU compute (§8.2). Batch 32: speedup falls to 4.38x for Falcon-40B (Fig. 14).

**Assumptions that do not hold for us.** ReLU-family activation (or retrained dReLU models such as Bamboo-7B). Whole model in RAM (192 GB / 64 GB). A strong desktop CPU with 2-3x our RAM bandwidth. Predictors cost 5-9% of model size, which is a lot of our scarce memory.

**What transfers.**
1. **Static hot/cold split by profiled frequency, solved per memory budget.** For MoE, the unit is an expert, not a neuron. The ILP becomes a knapsack: fill VRAM with the experts with the highest (frequency x bytes saved per hit) per byte.
2. **Compute cold weights where they are (CPU), do not ship them to the GPU** at batch 1. This is the same rule as Fiddler (G06).
3. **CPU operators that touch only the needed rows** (fused index + matvec). For experts this is the normal case: compute only the selected experts.
4. Neuron-level sparsity on SwiGLU dense models gave only 1.5-1.7x (Table 8). See §3, question 5.

---

### G04. PowerInfer-2 (Xue et al., arXiv 2406.06282, v-latest)

**Problem and hardware.** Smartphone inference of models larger than DRAM, with weights streamed from UFS flash. Hardware: OnePlus 12 (Snapdragon 8 Gen 3, 24 GB DRAM, 19 GB usable by one app, UFS 4.0) and OnePlus Ace 2 (8+ Gen 1, 16 GB / 11 GB usable, UFS 3.1) (Table 3).

**Storage facts (§2.3.2).** UFS sequential read is 450 MB/s at 4 KB blocks and 4 GB/s at 512 KB blocks. 4 KB random reads give 1 GB/s inside a 128 MB range and <850 MB/s inside a 512 MB range. The I/O-issuing core matters: 1,076 MB/s on a big core vs 762 MB/s on a little core (Table 1). UFS has one command queue: many I/O threads lose up to 40%.

**Mechanism.**
- **Neuron clusters:** groups of neurons with similar activation frequency. Big hot clusters run dense on the NPU; small cold clusters run sparse on the CPU with a predictor (from PowerInfer). At batch 1 the FFN matrix splits ~50/50 between NPU and CPU; at large batch the NPU takes ~70% (§4.1.2-4.1.3). Precompiled NPU graphs per batch size (~10 KB each) swap in during attention.
- **Prefill:** NPU-centric. One big core loads the next layer's weights with sequential reads while the NPU computes (§4.1.1). Prefill I/O is fully hidden (Fig. 9).
- **Neuron cache (§4.2):** three regions. (1) Fixed: attention weights and KV cache. (2) Hot region: cluster-level LRU. (3) Cold region: neuron-level LRU. Evicted weights are discarded (never written back). The hot/cold ratio changes with batch size.
- The paper rejects LLM-in-a-Flash co-activation bundling: after removing hot neurons, co-activation within one matrix is <20% (§4.2).
- **Bundling across matrices instead:** neuron i of Gate, Up and Down co-activate 80% of the time. Store them as one unit by neuron position (§4.4). FP16: one 24 KB read. INT4: a 7.5 KB bundle padded to 8 KB, read as two 4 KB reads: read Gate first; read Up/Down only if the Gate output is non-zero (two-phase loading).
- **Neuron-cluster pipeline (§4.3):** five stages (predict, Gate I/O, Gate compute, Up+Down I/O, Up+Down compute). Compute threads move to the next matrix's in-memory clusters while the I/O thread fetches missing clusters. Matrix barriers are removed (Fig. 6b).

**Results.**
- Decode with 50% of FFN weights in flash (Fig. 7): OnePlus 12, average 3.84x over LLMFlash (up to 4.63x) and 24.6x over llama.cpp (up to 27.8x).
- **TurboSparse-Mixtral-47B** (~3B active parameters per token due to MoE + dReLU sparsity): **11.68 tok/s** with 19 GB memory; **2.13 tok/s with 7 GB**, where the neuron cache holds only 400 MB = 1.8% of FFN weights (§7.2.3, Fig. 10). Average cache miss rate 3.5%, P99 18.9% (§7.2.4).
- Ablation, Bamboo-7B, 50% FFN offloaded (Fig. 14): baseline 0.4 tok/s -> + bundle 1.1 -> + neuron cache 4.18 (95% hit rate) -> + pipeline 9.60 -> + NPU 11.07.
- Compute/I/O share of critical path: 86.3% / 13.7% for PowerInfer-2 vs 23.3% / 76.7% for LLMFlash (Table 4).
- **SiLU model** Mistral-7B with 50% FFN offloaded: 5.3 vs 2.18 tok/s for LLMFlash (2.4x) (Table 6). Bamboo-7B (dReLU): 11.1 vs 2.4 (4.6x).

**Assumptions that do not hold for us.** NPU + unified memory (no PCIe). UFS with one queue (NVMe has many queues, so our I/O threading rules differ). ReLU/dReLU-sparsified models (Bamboo, TurboSparse) for the large gains. Rooted Android with `mlock`.

**What transfers.**
1. **The pipeline that breaks matrix and layer barriers.** Compute whatever is resident while I/O fetches the rest. For experts: run cached experts of layer l first while missed experts of layer l stream (also ProMoE "reordered inference", G10).
2. **Bundle by co-use, not by co-activation statistics.** Gate/Up/Down of the same unit go together.
3. **Two-phase conditional loading** (read Gate first, then Up/Down only if needed) is a neuron trick. For experts the analogue is HOBBIT's precision choice (G09).
4. **Cache regions with different granularity and LRU per region.** Discard on evict; never write back.
5. The TurboSparse-Mixtral result is the strongest evidence that **"MoE + activation sparsity + flash streaming" reaches usable speed on 7 GB of memory**. But it depends on a retrained dReLU MoE. Expected gain for our SwiGLU targets: the pipeline and cache ideas, ~1.3-2x over a naive streamer [est]; not the 11.68 tok/s.

---

### G05. Fast Inference of MoE LMs with Offloading (Eliseev & Mazur, arXiv 2312.17238)

**Problem and hardware.** Interactive (batch 1) Mixtral-8x7B-Instruct on a GPU that holds only part of the experts. Target hardware (§2.4): enough RAM for the model, a GPU with 11-16 GB, host-to-device at 8-16 GB/s (PCIe Gen 3). Tested: free Colab T4 (16 GB, PCIe 3), RTX 3080 Mobile (16 GB, PCIe 4), RTX 3060 (12 GB, PCIe 3), A100-80GB (§4.3).

**Mechanism.**
- **Expert locality:** some experts stay active for 2-4 adjacent tokens; others recur with gaps (Fig. 1).
- **LRU cache, k experts per layer**, same k for all layers. k = 2 for 12 GB GPUs, k = 4 for 16 GB (§3.3). When an expert is loaded to the GPU, the LRU GPU expert goes back to RAM.
- **Speculative expert loading = gate lookahead.** Apply the **next layer's gate** to the **current layer's hidden state** (the same input that the current gate uses). This works because of the residual stream (§3.2). Speculative loading starts right after all experts of the current layer are loaded. It fetches the 1-2 most likely experts. A speculatively loaded expert does not evict a cached expert; if it is used, it replaces the LRU expert of that layer (§3.3).
- **Memory layout:** all parameters of one expert in one contiguous buffer, so one host-to-device copy moves an expert. Host buffers are pinned. b = 4 GPU-side staging buffers, shared by all layers, for async copy and prefetch (§3.3).
- **Mixed quantization (HQQ, data-free):** attention at 4-bit (group 64, scale group 256); experts at 3-bit (group 64) or 2-bit (group 16, scale group 128; effective 2.6 bits per parameter). Embeddings, logits, gates and norms stay 16-bit (§4.2). They also tried QMoE sub-1-bit: the perplexity loss on Mixtral was too large (§3.3).

**Results.**
- Table 1 (attention 4-bit rows, which extract cleanly): experts FP16 85.16 GB; 4-bit experts 23.99 GB, Wiki2 PPL 3.76, C4 6.66, MMLU 69.11%; 3-bit 21.37 GB, 4.05 / 6.87 / 68.47%; **2-bit 17.54 GB, 4.61 / 7.42 / 65.58%**. Full FP16 model: 3.59 / 6.52 / 70.51%. For reference, Mistral-7B 4-bit: 5.03 / 7.56 / 61.3%. So Mixtral with 2-bit experts still beats a 4-bit 7B dense model.
- Experts are 45.1B of 46.7B parameters (96.6%) (§4.2).
- Table 2, tok/s, batch 1. The PDF text layer splits the rows. My reconstruction (column-wise, consistent with the label positions):

| Config | A100 | 3080M | 3060 | T4 | (3-bit experts) A100 | 3080M | 3060 | T4 |
|---|---|---|---|---|---|---|---|---|
| Full algorithm | 3.061 | 2.655 | 2.278 | 2.092 | 2.845 | 2.475 | 2.038 | 1.603 |
| No prefetch | 2.918 | 2.227 | 2.051 | 1.567 | 2.683 | 2.024 | 1.857 | 1.365 |
| No LRU, no prefetch | 2.265 | 1.758 | 1.547 | 1.168 | 2.055 | 1.595 | 1.346 | 1.061 |
| Naive (Accelerate, whole layer) | 1.392 | 1.059 | 0.919 | 0.661 | 1.246 | 0.914 | 1.791(?) | 0.580 |

  On T4 2-bit: loading only active experts gives 1.77x over naive; LRU adds 1.34x; prefetch adds 1.34x; total 3.2x. The "1.791" cell breaks the pattern; it may be a typo in the paper.
- Hit rates (Fig. 2 left) and prefetch recall for 1, 2 and 10 layers ahead (Fig. 2 right): numbers not stated in text.

**Assumptions that do not hold for us.** The whole model fits in RAM (Mixtral 2-bit = 17.5 GB; we have ~4.5 GB). 12-16 GB VRAM (we have 3.3 GB usable). Mixtral has 8 big experts top-2; modern models have 64-128 small experts top-4 to top-8.

**What transfers.**
1. **Gate lookahead costs one small matmul** (hidden x n_experts per layer) and needs no training. It is the first predictor to build.
2. **One contiguous blob per expert, pinned staging buffers, a small shared pool of transfer buffers.** Copy as is.
3. **Low-bit experts + higher-bit attention.** For our machine the bytes per token dominate. 2-3-bit experts cut SSD bytes by ~1.5-2x vs 4-bit.
4. **Speculative loads must not evict live cache entries** until they are used. This protects the hit rate from bad guesses.
Expected gain on our machine [est]: LRU per layer + lookahead together ~1.5-2x over "load every needed expert on demand" when the cache holds a meaningful fraction of experts. The gain is smaller when SSD-bound (see HOBBIT §G09 and ProMoE §G10).

---

### G06. Fiddler (Kamahori et al., ICLR 2025, arXiv 2402.07033 v3)

**Problem and hardware.** Mixtral-8x7B in FP16 (>90 GB) with one GPU that holds only part of the experts, plus a large CPU. Env 1: Quadro RTX 6000 24 GB, PCIe Gen3 x16 (32 GB/s), Xeon Gold 6126 (48 cores), 56 of 256 experts on GPU. Env 2: RTX 6000 Ada 48 GB, PCIe Gen4 x16 (64 GB/s), Xeon Platinum 8480+ (112 cores), 125 of 256 experts on GPU (Table 1). The whole model is in RAM.

**Mechanism.**
- Non-expert layers always on GPU (<2B parameters for Mixtral). Fill remaining VRAM with the most popular experts from an **offline popularity profile** (ShareGPT calibration). The placement is static (§3.3, §3.4).
- Three cases per expert per layer (Fig. 3): (a) expert on GPU -> run on GPU; (b) expert in RAM -> copy weights to GPU and run there; (c) expert in RAM -> copy activations to CPU, run on CPU, copy output back.
- **Decision rule (Algorithm 1):** for expert j with s input tokens, if it is on GPU run it there; else if `cpu_lat(s) > gpu_lat(s) + trans_lat()` copy weights and run on GPU; else run on CPU. Latency model: `gpu_lat` is constant in s; `cpu_lat(s)` is linear in s; constants are measured at init (§3.3).
- Microbenchmarks (App. A): weight copy takes 2-5x longer than GPU compute of the expert. GPU latency is flat in batch size. CPU latency grows linearly with input size. Activation copy is <1% of CPU-path latency.
- A custom CPU expert kernel with AVX512_BF16 (§3.4).

**Results.**
- Single-request end-to-end: **1.26x** over the best baseline (llama.cpp) on average (§4.2, Fig. 4). Prefill TTFT: 1.07x over DeepSpeed-MII, 1.65x over Eliseev & Mazur. Beam search: **11.57x** over llama.cpp (Fig. 6). Split metrics: TTFT 1.13x, inter-token latency 1.43x (App. F).
- Env 1 dataset sensitivity: 1.81x over llama.cpp on ShareGPT, 1.56x on LMSYS-Chat (App. D). Phi-3.5-MoE: 6.5x over DeepSpeed-MII (App. E).
- Absolute tok/s: not stated in text (bar charts only).
- **Popularity placement helps little** (App. C): Env 1 hit rate 25.2% best vs 21.9% random vs 18.7% worst; Env 2 53.0% vs 48.8% vs 44.6%. Gain: +3-5 points. Mixtral expert popularity: mean 0.71 of max, std 0.08 (balanced).
- **SiLU sparsity is weak in Mixtral** (App. B, Table 2): in all 32 layers <2% of post-SiLU values have |x| < 0.001; in 28 layers <5% have |x| < 0.01. OPT has >90% exact zeros after ReLU.

**Assumptions that do not hold for us.** Whole model in RAM. AVX-512 BF16 CPU with 48-112 cores (we have 6 cores, AVX2, ~20 GB/s RAM). PCIe x16 at 32-64 GB/s (we have ~10-12 GB/s). FP16 experts (300+ MB each).

**What transfers.**
1. **The decision rule itself.** It is three lines. On our machine, for a RAM-resident expert at batch 1, the CPU path reads the expert once from RAM at ~15-20 GB/s; the GPU path reads it from RAM through PCIe at ~10-12 GB/s and also evicts a VRAM resident. **[est]** For a 2.8 MB Q4 expert: CPU ~0.15-0.2 ms vs PCIe copy ~0.25 ms + GPU compute. The CPU path wins for decode; the GPU path wins when s ≥ ~3-8 tokens per expert (prefill, batch, speculative verify). Measure the constants at startup, as Fiddler does.
2. **Do not spend effort on static popularity placement alone.** It gave +3-5 points of hit rate on balanced routing.
3. The SiLU analysis is direct evidence against neuron-level sparsity for SwiGLU MoE experts (§3, question 5).
Expected gain [est]: 1.2-1.5x decode over "always upload misses to GPU", when the miss source is RAM. When the miss source is SSD, the SSD read dominates either way; the CPU path still avoids PCIe and VRAM churn.

---

### G07. MoE-Lightning (Cao et al., arXiv 2411.11217)

**Problem and hardware.** High-throughput offline batch inference of MoE models with limited GPU memory. No disk offloading (§4, footnote 6). S1: 1x T4 16 GB + 24-core Xeon 2.3 GHz, 192 GB RAM. S2: 1x L4 24 GB + 192 GB. S6-S9: 2-4x T4 with 416 GB RAM for Mixtral-8x22B and DBRX (Table 2). L4 case study: GPU 300 GB/s, 242 TFLOPS; CPU 100 GB/s, 1.3 TFLOPS; CPU-GPU 32 GB/s (Fig. 3).

**Mechanism.**
- **HRM, Hierarchical Roofline Model (§3.2):** each memory level has its own compute roof and memory roof, plus a roof for data moved from level j to level i: `P <= min(P_peak_i, B_i x I_i, B_j->i x I_j)` (Eq. 7). Turning point P1 (Eq. 9): below this operational intensity, moving data from CPU to GPU for compute is not worth it; compute on the CPU. Turning point P2 (Eq. 10): below this, compute is bounded by the CPU->GPU link. A balance point (Eq. 11) gives the best batch / micro-batch.
- Case study (§3.3): decode attention has low intensity, independent of batch, so **attention on CPU** (KV cache in RAM) beats moving KV to GPU, for both FP16 and INT4 KV. MoE FFN intensity grows with batch.
- **CGOPipe (§4.1, Algorithm 1, Fig. 6):** GPU does post-attention (O proj + MoE FFN) of micro-batch i, then pre-attention (norm + QKV) of micro-batch i+1. CPU does attention of the next micro-batch. The I/O link moves **one page** of the next layer's weights between the small hidden-state transfers. Weights are split into n pages where n = number of micro-batches. A pinned staging pool: CPU memory -> pinned -> GPU, overlapped (App. A.1, Fig. 11). Weight buffer on GPU = 2 x (CPU-resident part of one layer).
- **Policy search (§4.2, Table 1):** variables = batch N, micro-batch μ, attention-on-GPU flag, FFN-on-GPU flag, weight ratio on GPU, KV ratio on GPU. Per-layer latency = `max(comm_cpu_to_gpu, T_cpu, T_gpu)` (Eq. 12).

**Results.**
- Up to **10.3x** throughput over the best baseline without padding, 3.5x with padding, on one GPU (§1, Fig. 7).
- Table 5 (MTBench, S1, generation 128): FlexGen with its own policy (μ = 8, N = 1112) 9.5 tok/s; FlexGen with MoE-Lightning's policy (36, 504) 16.8; MoE-Lightning(p) (36, 504) **30.12 tok/s** (3.17x).
- CPU attention kernel is 3-4x faster than KV transfer, about the ratio of CPU memory bandwidth to CPU-GPU bandwidth (§6.2, Fig. 9).
- Mixtral-8x22B: 2.77-3.38x more throughput with 4x T4 than 2x T4 (super-linear) (§5.3).

**Assumptions that do not hold for us.** Batch of hundreds to thousands of sequences. 192-416 GB RAM. Throughput metric, not latency. No disk.

**What transfers.**
1. **HRM as a planning tool.** Our machine has five roofs: VRAM 128 GB/s, RAM ~20 GB/s, PCIe ~11 GB/s, SSD ~5.5 GB/s, CPU compute (AVX2, 6 cores). With HRM, decide per operator (attention, shared expert, routed expert, LM head) where it runs, for decode (intensity ~1 token) and prefill (intensity = prompt length).
2. **Paged weight transfer interleaved with small activation transfers** (do not block hidden-state copies behind a big weight copy). Same as ProMoE "chunked prefetch".
3. **CPU attention with KV in RAM** is the right choice for long context on a 4 GB GPU only if VRAM must hold experts instead. On our machine RAM is also small, so this is a trade, not a free win.
4. For batch jobs on our machine, one expert read can serve many sequences. Expected gain for chat: low; for batch evaluation: large (throughput scales with batch until RAM for KV runs out).

---

### G08. Deja Vu (Liu et al., ICML 2023, arXiv 2310.17157)

**Problem and hardware.** Lower batch-1 latency of OPT-175B on GPUs by skipping input-dependent attention heads and MLP neurons. Hardware: 8x A100 80 GB with NVLink (§5.1). No offloading.

**Mechanism.**
- **Contextual sparsity exists:** with a two-pass test, a per-input subset of 80% of attention heads and 95% of MLP neurons can be removed with the same output; ~85% total, ~7x fewer parameters per token (§3.1, Fig. 3).
- Non-contextual (token-ID-only) sparsity loses accuracy already at 50% (§5.2, Fig. 1b).
- **Slowly changing embeddings:** cosine similarity of consecutive-layer hidden states is ~0.99 for OPT-175B from layer 2 on, >95% for all OPT sizes, and stays high several layers apart (Fig. 5). Cause: the residual norm ||X|| is much larger than ||F(X)||.
- **Predictor:** per block, a small two-layer fully connected network, trained as a classifier on (input, neuron-active) pairs from dense runs (Algorithm 1). Training data: 500 C4 samples (§5.1). MLP predictor validation accuracy >99% in shallow layers, ~93% in deep layers; attention-head predictor ~93% in middle layers, ~99% at the ends (§5.2). Exact predictor size: not stated (PowerInfer says fixed-size Deja Vu predictors for OPT-175B take ~27 GB).
- **Asynchronous lookahead:** the predictor for layer l+1 runs on the input of layer l, in parallel with layer l's compute (§4.3, Fig. 2). A lemma bounds the error of cross-layer prediction (Lemma 4.3).
- Missing KV cache for heads that were skipped earlier: store the token embedding y for non-selected heads and compute K/V later when needed (§4.2).
- **Kernels:** fuse index + matvec in Triton (up to 4x vs PyTorch index-then-multiply, which moves 3x the memory). Store W2 and W_O transposed (column-major) so the gathered rows are contiguous (§4.4, App. D).

**Results.**
- No accuracy drop until ~75% sparsity (Fig. 6). Table 4: MLP 85% sparse or attention 50% sparse keeps OPT-175B accuracy (e.g., WikiText 10.82 -> 10.80 / 10.87).
- Latency: **1.8-2x** faster than FasterTransformer and 4.8-6x faster than Hugging Face at ~75% sparsity, batch 1 (Fig. 7).
- With 4-bit weights (W4A16): errors do not compound (Table 7).
- Union sparsity grows sub-linearly with batch size (Fig. 8).
- BLOOM (GeLU): only 30% MLP sparsity used (§5.2).

**Assumptions that do not hold for us.** ReLU OPT models. All weights in HBM; the gain is in HBM reads, not in an offload tier. 8 GPUs. Large predictors.

**What transfers.**
1. **Residual-stream similarity is the theory behind every gate-lookahead predictor** (Eliseev, HOBBIT, ProMoE). It holds for all transformers, not only ReLU ones.
2. **Store matrices so a selected unit is contiguous** (transpose W2). For experts this is automatic if each expert is one blob.
3. **Fused gather-matvec**: never copy selected rows to a temp buffer before multiplying.
Neuron-level skipping itself: not for our SwiGLU MoE targets. Expected gain: indirect.

---

### G09. HOBBIT (Tang et al., arXiv 2411.01433 v2)

**Problem and hardware.** Batch-1 MoE inference on edge devices where expert loading dominates. Hardware: RTX 4090 24 GB + 256 GB RAM + 64 cores, PCIe 4.0 (32 GB/s); Jetson AGX Orin 32 GB unified memory, 12 cores. Weights on a Samsung 980 PRO NVMe (7,000 MB/s spec, ~3,000 MB/s in practice) (§5.1). Models: Mixtral-8x7B (8 experts, top-2) and Phi-MoE (16 experts, top-2), 32 layers each (Table 1). Built on llama.cpp, +8,000 lines C/C++.

**Motivation numbers.** Expert loading is 85.5% of layer time on the 4090 and 94.5% on Orin (Fig. 3a). Loading one Mixtral layer over PCIe 4.0 takes ~80 ms vs ~3 ms compute on the 4090 (§2.1). Skipping 10% of experts raises PPL by >1%. Replacing <20% of experts with low-precision versions costs ≤1% (Fig. 3b).

**Mechanism.**
1. **Token-level dynamic precision loading (§3.2).**
   - Importance proxy: the gate weight ||G(x)_i||. It correlates with the expert output norm ||G(x)_i E_i(x)|| at Pearson 0.99 (Fig. 5a).
   - Rank selected experts by gate weight (normalized). Unimportance score of the i-th ranked expert: `s_i = sum_{j<i} ||G(x)_j||` (cumulative gate weight of the higher-ranked experts), and `s_0 = 0` (Eq. 2). The top-1 expert is always high precision.
   - On a cache miss: s ≤ T1 -> load the high-precision copy; T1 < s ≤ T2 -> load the low-precision copy; s > T2 -> skip. Mixtral: T1 = 0.6, T2 = 0.9 -> 67% high, 30% low, 3% skipped (Fig. 5b).
   - Precision pairs: FP16 with INT4 low copy on the 4090; INT8 with INT2 low copy on Orin (§5.1). Loads use plain `read()` calls from a task queue served by a scheduler thread (§3.2, Fig. 6).
2. **Layer-level adaptive prefetch (§3.3).**
   - Gating inputs of consecutive layers have high cosine similarity. Using the current gating input with the **next layers' gate weights**: top-1 accuracy averages **96%** for the next layer and ~90% for 2-3 layers ahead (Fig. 7).
   - **Stacking computer:** stack the gate matrices of the next n layers and run one batched matmul + top-k. Cost stays flat as n grows; sequential gates grow linearly (Fig. 17a).
   - Adaptive depth: predict layer l+1; if all predicted experts are cached, predict l+2; continue until a predicted expert is missing or n layers are covered. Predicted experts are **masked from eviction**. Prefetches may use the low-precision copy to cut the penalty of wrong guesses (Fig. 9). Recommended n = 1-3.
   - A `cudaMemcpy` cannot be cancelled. A wrong prefetch therefore blocks the right load. Low-precision prefetch makes this penalty smaller.
3. **Sequence-level multidimensional cache (§3.4).**
   - Priority `P = w_lru x LRU + w_lfu x LFU + w_lhu x LHU + w_fld x FLD`, weights sum to 1 (Eq. 3). LRU = recency of last use; LFU = use count in the **current sequence**; LHU = count of **high-precision** uses in the current sequence; FLD = layer distance from the current layer (experts of nearer upcoming layers are kept). Records reset at each new sequence. Weights are tuned on calibration data to minimize miss penalty (exact values not stated).
   - Separate caches for high- and low-precision experts; the high-precision cache is larger.
   - Miss penalty, not miss rate, is the objective: a high-precision miss costs b_h/b_l times a low-precision miss (4x for FP16 vs INT4).
   - Observation: P(top-1 expert of token t is used at token t+1) is well above the 0.25 random baseline in Mixtral (Fig. 10a).
4. Two compute modes: GPU-centric (default) and CPU-GPU cooperative where a helper thread runs missed experts on the CPU (§4).

**Results.**
- Orin vs llama.cpp: decode **13.0x** (Mixtral), **18.9x** (Phi-MoE); prefill latency -79% / -80%. Orin vs MoE-Infinity: 3.64x / **9.93x** decode (§5.2). llama.cpp on Orin suffers page faults from `mmap` under memory pressure.
- 4090 vs MoE-Offloading (Eliseev): 3.21x / 3.29x decode; vs MoE-Infinity 2.30x / 3.92x.
- 4090 + CPU mode: vs llama.cpp 1.31x / 1.42x decode; **vs Fiddler 0.99x / 1.46x** decode. Fiddler computes a Mixtral expert on CPU in 3 ms vs 5 ms for HOBBIT (§5.4).
- Accuracy (Table 3): ≤1% drop. Mixtral FP16 GSM8K 0.52 -> 0.51 with FP16+INT4; INT8 0.52 -> INT8+INT2 0.51. Phi-MoE FP16 0.83 -> 0.82.
- Ablation: dynamic loading alone 1.19-1.57x, largest on Orin (slow SSD) (Fig. 16). **Prefetch in decode: 1.01x without mixed precision, ~1.05x with it**; prefill gains ~10% (Fig. 17b). Cache policy: 4.69-8.68% lower miss penalty than LRU and 2.13-4.19% lower than LFU (Fig. 18a). Sequence-level LFU gives +4.5% hit rate over model-level LFU (Fig. 18b).
- Absolute tok/s: not stated in text (Fig. 14 bars only).

**Assumptions that do not hold for us.** Top-2 routing with large experts (Mixtral 8x7B). Modern targets use top-4 to top-8 with smaller experts, so the gate mass spreads more and more experts fall into the "low" group. The Orin case keeps experts in a 32 GB unified memory with SSD as the next level; the 4090 case has 256 GB RAM. Both precision copies are stored, which costs disk space.

**What transfers.** This is the closest paper to our target.
1. **Dynamic precision on miss** directly cuts SSD bytes. [est] For Qwen3-30B-A3B at Q4 with a ~2-bit low copy: if 30-50% of misses go low and ~3% are skipped, SSD bytes per token fall by ~18-28% -> ~1.2-1.4x decode when SSD-bound. Must be checked for accuracy with top-8 routing.
2. **Gate stacking + adaptive depth + protect predicted experts from eviction.**
3. **Cache priority over miss penalty with sequence-level counters.** Small gain (2-9%) but nearly free.
4. **Honest result: prefetch alone gave ~1.01-1.05x in decode** when loading dominates. Prefetch cannot add bandwidth. It only fills idle link time.
5. **Disk cost.** A second copy of every expert costs space we do not have (82 GB). Prefer a nested (bit-plane) format where the low-precision copy is a prefix of the high-precision copy (Any-Precision LLM, Matryoshka Quantization, Drop-by-Drop; see `B_2026_quant_papers.md`).

---

### G10. ProMoE (Song et al., arXiv 2410.22134 v3)

**Problem and hardware.** Batch-1 (edge) MoE inference where reactive cache misses put expert loading on the critical path. Hardware: RTX 4090 24 GB, i9-14900K, 128 GB DRAM, PCIe 4.0 (measured 23.9 GB/s, x8) (§2.2, §7.1). Models: DeepSeek-MoE-16B (DS-1, 64 experts top-6), DeepSeek-V2-Lite (DS-2, 64 top-6), Qwen1.5-MoE (QW-1, 60 top-4), Qwen2-57B-A14B (QW-2, 64 top-8), Mixtral-8x7B (Table 1). DS-1/DS-2/QW-1 in FP16, QW-2/Mixtral in INT4. Implemented in transformers and llama.cpp (6,600 lines C++).

**Motivation numbers.** DS-1 with 50% of experts cached: TPOT 67.9 ms, of which 58.1 ms (85.6%) is fetching experts; 2.67 GiB of expert weights per token, 1.33 GiB transferred (§2.2). llama.cpp with QW-2: 79.0% blocking time in decode, 94.2% in prefill (Fig. 4). **Modern decoder-only MoEs have low access skew**, unlike Switch/NLLB which follow a power law; LRU hit rate rises only about linearly with cache size (Fig. 5).

**Mechanism.**
- **GoodPred = Accuracy x FetchRate** (§4.1). FetchRate = share of predicted experts that arrive before use. A predictor that is accurate but late is useless.
- Baselines: token-ID predictor (predict all layers from the token ID): accuracy **58.3%** (Fig. 8). Skip-based (Eliseev gate lookahead): good on DS-1 except head and tail layers, but only **66.9%** on QW-2 because the QW-2 gate is sensitive to small input changes (§4.2). DS-2 consecutive gate-input cosine similarity: 91.7%.
- **Learned predictor (§4.3, §6.3):** one 2-layer MLP per layer, ~2M parameters. Input: the hidden state at layer i. Output: expert scores for a later layer; the top-k (k = the model's activated count, e.g., 6 for DS-1) are prefetched. Training: traces from training-domain data, split 9:1; under 1-2 h on one GPU. Average accuracy **84.7%** on DS-1 and QW-2.
- **Stride prefetch (§4.4):** the predictor runs on the **CPU** (~200 µs) in parallel with GPU inference, and the prediction distance is +1 longer than layer-wise lookahead, so prefetch starts earlier and prediction overlaps prefetch. Accuracy drops by only 5%.
- **Prefetcher (§5, Algorithms 1-2):** one worker thread, two-priority task queue. Low priority = predicted experts; high priority = actually selected experts after the gate.
  - **Chunked prefetch:** each expert = 3 chunks (its 3 linear layers). A high-priority task waits at most one chunk.
  - **Early preemption:** a hook at the end of the gate pushes the real expert list immediately and drops this layer's remaining low-priority tasks.
  - **Reordered inference:** run cached experts first, then experts in flight, then experts not yet started. The compute and prefetch threads then process missing experts in the same order (a pipeline). This also prevents LRU thrashing in prefill (ID-order access evicts higher-ID experts that are still needed).
- Cache: a plain per-layer LRU in one preallocated contiguous region. Prefetch "touches" cached experts to delay their eviction (§6.1).

**Results.**
- vs open-source offloading (TO, LO, UM): average **2.20x** prefill (up to 3.21x), **2.07x** decode (up to 5.02x) (abstract).
- transformers: prefill 1.42x over static cache, 2.21x over LRU; decode **1.47x over static, 1.31x over LRU** (§7.2, Fig. 10).
- llama.cpp: prefill 1.36x / 2.12x; decode 1.49x over static, **1.09x over LRU** (Fig. 11).
- **Mixtral in llama.cpp: LO (llama.cpp CPU offload of whole layers) beats ProMoE in decode**, because Mixtral activates 25% of experts per token and CPU compute is cheaper than PCIe transfer. Also at **low cache rates LO beats all cache-based systems** in decode (§7.2, §7.4, Fig. 16).
- Ablation (transformers, Fig. 12): prefill gains come from early preemption (1.27x) and reordering (2.39x); decode gains accumulate to 1.35x over LRU.
- Batch 1-4: speedup over LRU on QW-2 grows from 1.16x to 1.34x (§7.5). At INT4 vs FP16 (QW-2), speedup over LRU in prefill 2.06x vs 1.105x (§7.6).

**Assumptions that do not hold for us.** All experts in RAM; PCIe 4.0 x8; 24 GB VRAM. Predictors trained with a GPU.

**What transfers.**
1. **The prefetch machinery** (two-priority queue, chunked tasks, early preemption at the gate, reordered expert execution). This is the right design for our async SSD reader too. Chunk = expert's 3 matrices, or fixed 1-4 MB pieces.
2. **Measure GoodPred, not accuracy.** On SSD, a prediction is only good if the read can finish before use.
3. **Expect the gate lookahead to fail on some models** (QW-2: 66.9%). Then train a small per-layer MLP (~2M parameters x 48 layers = ~100M parameters = ~200 MB in FP16, ~100 MB INT8 [est]; can run on CPU in ~0.2 ms per layer per the paper).
4. **Their own data supports Fiddler for our case:** at low cache rates, CPU compute of uncached experts beats transfer.
Expected gain on our machine [est]: 1.1-1.3x decode over plain LRU with on-demand loads; more in prefill (reordering + preemption, up to 2x).

---

### G11. DeepSpeed-Inference (Aminabadi et al., SC 2022, arXiv 2207.00032)

**Problem and hardware.** Fast inference of dense and MoE transformers at all scales, plus **ZeRO-Inference** for GPU-poor machines. Testbeds: up to 256 A100 40 GB; a Lambda A6000 workstation (2x A6000 48 GB, 256 GB DRAM, 2 TB NVMe); a DGX-2 (16x V100 32 GB, 1.5 TB DRAM, 30 TB NVMe) (§VII-A).

**Mechanism.**
- **ZeRO-Inference (§VI):** keep all weights in DRAM or NVMe, keep only one or a few layers on the GPU, and **stream every layer** for each forward pass. Use the freed GPU memory for **large batches** so compute hides the fetch. One GPT-3-175B layer needs ~7 TFLOP for a batch-1 prompt of length 2048, so big batches make compute dominate.
  - Prefetch a configurable number of layers ahead. In multi-GPU, each GPU fetches a partition of the layer over its own PCIe link, then they all-gather over NVLink (§VI-B).
  - NVMe tricks inherited from ZeRO-Infinity: bulk async read requests, aggressive I/O parallelism, pinned memory, no extra copies.
- **MoE kernels (§V-C):** replace one-hot gating + sparse einsum (cost S x E x M x c_e) with a token->expert table, an inverse expert->token table built by a parallel scan, and data-layout gather/scatter (cost S x M x c_e). Over 6x lower MoE kernel latency. Expert, data and tensor parallel; expert slicing; parallelism-coordinated all-to-all.
- Dense kernels: deep fusion and custom small-batch GEMMs; INT8.

**Results.**
- ZeRO-Inference runs a **530B model on one A6000** (25x larger than GPU-only, 10x larger than CPU-only), up to **84 TFLOPS = 54% of peak (158.4)** at very large batch (Fig. 9b). GPT-50B on 16 V100s: near-linear scaling; single V100 67 TFLOPS, 53% of peak (Fig. 9c). Prefetch helps at small batch; the gain shrinks at large batch (Fig. 10c).
- MoE serving: up to **7.3x** lower latency vs PyTorch baseline; a trillion-parameter MoE served under 25 ms per token on 256 GPUs at 128 TB/s aggregate HBM bandwidth (33% of peak) (§VII-B).
- Dense: up to 1.55x (FP16) and 1.95x (INT8) lower latency at small batch vs FasterTransformer.

**Assumptions that do not hold for us.** Large batch, latency-insensitive. Hundreds of GB of DRAM. ZeRO-Inference streams **whole layers**, not experts. Eliseev Table 2 shows this "naive" approach at 0.66-1.39 tok/s for Mixtral and Fiddler/HOBBIT beat it by large factors.

**What transfers.**
1. Prefetch depth as a tunable (later done adaptively by HOBBIT and MemVanta).
2. **MoE routing kernel:** build a token->expert table and its inverse with a scan, and gather tokens per expert. We need this for prefill and batch modes.
3. NVMe I/O hygiene: many outstanding large async reads, pinned buffers, no extra copies.
Expected gain for chat: none from the streaming scheme; it is the baseline that expert-level methods beat.

---

### G12. QMoE (Frantar & Alistarh, arXiv 2310.16795)

**Problem and hardware.** Compress a 1.6T-parameter MoE (SwitchTransformer-c2048, 3.2 TB in BF16) to under 1 bit per parameter and run it on one GPU server. Compression on one A6000 48 GB with a few hundred GB of CPU RAM, <1 day. Inference on 4x A6000 or 8x RTX 3090 (§5.1).

**Mechanism.**
1. **Scaled GPTQ for MoE (§3):**
   - Activation offloading with one big CPU "list buffer" of all token hidden states, with sample boundaries; each token is read and written twice per block (Figs. 2-3).
   - Lazy weight fetch from disk: each expert is read exactly once.
   - **Expert grouping:** run GPTQ jointly on 16 experts with stacked Hessians H_E = X_E^T X_E (6x faster at |E| = 16: 174.1 s -> 28.8 s, Table 1).
   - Robustness: Hessian dampening 0.1 (10x usual); plain rounding if a Hessian is not invertible; cap tokens per expert at 4x the mean.
   - Mask special MLM tokens out of calibration (val loss ternary 2.16 -> 1.99 on base-128, Table 2). Activation reordering hurt (Table 3).
   - Calibration: 10K samples for 128 experts, 160K for 2048.
2. **Ternary quantization** with per-row grid {w_min, 0, w_max}. Natural sparsity after ternary quantization: 85.7% (base-128), 86.4% (large-128), **88.6% (c2048)** zeros (Table 4). So the codes have low entropy.
3. **Dictionary code co-designed with a GPU kernel (§4.3):**
   - Fixed-length 16-bit codewords map to a variable number of symbols (LZW-like). Dictionary = the 2^16 most probable sequences of up to 14 ternary **pairs** (≤28 weights), built with a max-priority queue under an i.i.d. model with p0 = 0.885 (Algorithm 1). One static dictionary shared by all matrices.
   - Each dictionary entry = two UINT32 words: 7 pairs x 2 bits per weight + a 4-bit pair count, stored twice (Fig. 4). Dictionary = 2^16 x 8 B = **512 KB**, sized to fit in GPU L2.
   - Kernel (Listing 1): fused decode + matvec. One warp per matrix row; rows are encoded independently with row offsets. The input vector goes to shared memory. The warp loads 32 codewords in one coalesced read; for each codeword, 28 of 32 threads each extract one ternary weight from the dictionary entry, dequantize via a shared-memory 3-entry LUT (replicated 32x to avoid bank conflicts), multiply-accumulate, and advance the input index by the decoded count. A warp shuffle reduction gives the row result. One thread block per SM.
   - Compression rate: 20.07x measured vs 21.11x on synthetic i.i.d. data vs 25.40x entropy limit (§4.3.2).

**Results.**
- Table 5 (C4 val loss; reconstructed from a misaligned table): BF16 1.73 / 1.55 / **1.18** (base-128 / large-128 / c2048). RTN 2-bit 2.27 / 1.96 / 1.33; RTN ternary 4.54 / 2.79 / 2.15. QMoE (1x data) 2-bit 1.76 / 1.56 / 1.20; **ternary 1.99 / 1.69 / 1.26**. Text: 2-bit costs 1.7% relative and ternary 6.7% relative on c2048.
- Table 7: c2048 3142 GB -> **158.6 GB**, 19.81x overall, **0.807 bits per parameter** including dense layers; MoE-only 20.07x. base-128: 14.9 GB -> 1.27 GB. large-128: 52.7 GB -> 3.96 GB.
- Kernel speed: the compressed matvec is faster than cuBLAS BF16 for all tested MoE shapes, up to 35% faster; each call <0.02-0.05 ms (Fig. 5 left). End-to-end <5% slower than an idealized uncompressed run that would need >65 A6000 or >130 RTX 3090 (Fig. 5 right).
- Compression time c2048: 16.0 h at 160K samples (Table 8).

**Assumptions that do not hold for us.**
- The model is an **encoder-decoder MLM** (SwitchTransformer), top-1 routing, 2048 small experts, very over-parameterized. Eliseev & Mazur (G05, §3.3) tried QMoE-style sub-1-bit on Mixtral and found **too large a perplexity loss**. There is no evidence in this paper for modern decoder-only chat MoEs.
- The kernel is CUDA for an L2 ≥ 512 KB plus room for weights. The GTX 1650 (TU117) L2 is about 1 MB [from memory, verify], so the dictionary would take about half of it. No CPU kernel exists.
- Compression needs hundreds of GB of RAM for calibration buffers.

**What transfers.**
1. **The general lesson: experts tolerate lower bits than dense layers** (also Eliseev Table 1, HOBBIT).
2. **Lossless entropy coding of already-quantized expert codes trades cheap compute for scarce bytes.** On our SSD-bound machine, a decode that runs faster than ~5.5 GB/s of compressed input is free. This is worth one experiment: measure the empirical entropy of IQ2/Q2_K expert codes of the target model. QMoE got its 20x because ternary codes had 88.6% zeros; k-quant codes are much closer to uniform, so the expected gain is small (<1.2x [est]) unless we use a zero-heavy quantizer.
3. **Expert-group batched GPTQ and lazy one-pass weight fetch** are the right pattern for a constant-memory quantizer of a big MoE on our 8 GB machine (matches the rai converter plan in `00_SYNTHESIS.md` §6).
Expected gain for current chat MoEs: low and unproven. Priority: research, not engine core.

---

## 2. Comparison table

| System | Tiers used | Unit of transfer | Cache policy | Predictor | Batch | Best reported result (hardware, model) |
|---|---|---|---|---|---|---|
| G01 FlexGen | GPU, CPU RAM, disk | Layer (weights); tensor slices (KV, act.) | None (static LP placement %) | None | Very large (eff. 144-256) | OPT-175B 0.69 tok/s, 1.12 with 4-bit; 1x T4 + 208 GB + 1.6 GB/s SSD (Table 2) |
| G02 LLM in a Flash | Flash, DRAM, (GPU) | Neuron (up row + down col bundle, 32 KiB) | Sliding window of last k = 4-5 tokens | Low-rank r = 128-1152, current-layer attention output | 1 | OPT-6.7B 84 ms/token (60 with spec.) on 4090; 669 ms on M1 Max CPU; half of model in DRAM (Table 3) |
| G03 PowerInfer | GPU, CPU RAM | Neuron (row/col) | Static hot set (ILP); no runtime cache | Adaptive MLP per layer (5-9% of params) | 1 (to 32) | INT4 avg 13.20, peak 29.08 tok/s; OPT-175B INT4 ~2 tok/s; 4090 + 192 GB (§8.2) |
| G04 PowerInfer-2 | UFS flash, DRAM, NPU+CPU | Neuron cluster (hot) / neuron bundle G+U+D (cold) | LRU per region (cluster / neuron) | PowerInfer predictor | 1-4 | TurboSparse-Mixtral-47B 11.68 tok/s at 19 GB, 2.13 at 7 GB; OnePlus 12 (Fig. 10) |
| G05 Eliseev & Mazur | GPU, CPU RAM | Expert (contiguous blob) | LRU, k per layer | Next-layer gate on current hidden state, 1-2 experts | 1 | Mixtral 2-bit experts 2.09-3.06 tok/s on T4 to A100 (Table 2) |
| G06 Fiddler | GPU, CPU RAM | Expert, or activations to CPU | Static by popularity | None | 1; beam ≤16; long prefill | 1.26x vs llama.cpp batch 1; 11.57x beam; Mixtral FP16 (§4.2) |
| G07 MoE-Lightning | GPU, CPU RAM | Weight pages of a layer | Static ratio (policy search) | None | Hundreds+ | Up to 10.3x throughput; 30.1 tok/s MTBench Mixtral on 1x T4 (Table 5) |
| G08 Deja Vu | GPU HBM only | Head / neuron (compute skip) | n/a | 2-layer FC, async lookahead | 1 | OPT-175B 1.8-2x vs FasterTransformer, 8x A100 (Fig. 7) |
| G09 HOBBIT | GPU, CPU RAM, SSD | Expert at 2 precisions | Weighted LRU + seq-LFU + LHU + layer distance | Stacked next-n gates (96% top-1 next layer) | 1 | 9.93x vs MoE-Infinity, 18.9x vs llama.cpp (Phi-MoE, Orin) (§5.2) |
| G10 ProMoE | GPU, CPU RAM | Expert, 3 chunks | Per-layer LRU + proactive prefetch | Learned 2-layer MLP (~2M) per layer, stride, CPU; 84.7% | 1-4 | 2.07x decode avg (up to 5.02x) vs offload baselines; 4090 (abstract) |
| G11 DeepSpeed-Inf. (ZeRO-Inf.) | GPU, CPU RAM, NVMe | Whole layer | None (stream all) | None | Very large | 530B on 1x A6000; 84 TFLOPS = 54% peak (§VII-D) |
| G12 QMoE | GPU (fits after compression) | n/a (compression) | n/a | n/a | 1 | 1.6T c2048 at 0.807 bpp, 158.6 GB, <5% slower than ideal, 4x A6000 (Table 7, Fig. 5) |

Observations from the table:
- Only HOBBIT, PowerInfer-2, LLM in a Flash, FlexGen and ZeRO-Inference read weights from **storage**. Of these, only HOBBIT does it at **expert granularity at batch 1**.
- Every batch-1 MoE system that measured CPU compute against transfer found CPU compute wins for small token counts (Fiddler App. A, PowerInfer Fig. 6, ProMoE §7.2 LO result, MoE-Lightning P1 turning point).
- Every system that measured prediction for **decode** found modest gains when loading dominates: HOBBIT 1.01-1.05x, ProMoE 1.09x over LRU in llama.cpp. The big wins are bytes (low-bit experts, HOBBIT dynamic precision) and placement (CPU vs GPU).

---

## 3. Design recommendations for our engine

### 3.1 Framing for this machine [est]

Per-token time when all transfers overlap ≈ the maximum of: SSD bytes / ~5.5 GB/s, RAM bytes / ~20 GB/s (shared by SSD DMA, PCIe DMA and CPU compute), PCIe bytes / ~11 GB/s, VRAM bytes / ~128 GB/s, and CPU compute. When they do not overlap, it is the sum.

For Qwen3-30B-A3B Q4 (~1.1 GB expert bytes per token, ~18 GB file, ~7.8 GB total cache in VRAM + RAM): routing is low-skew (ProMoE Fig. 5, Fiddler App. C), so the hit rate is roughly the cached fraction plus a temporal-locality bonus, maybe 50-60%. Then SSD bytes per token ≈ 0.45-0.55 GB -> ~0.1 s -> **~8-10 tok/s ceiling from SSD**; real ~3-5 tok/s. For gpt-oss-120b MXFP4 (~1.9 GB per token, ~61 GB file, ~13% cached): ~1.65 GB from SSD -> ~0.3 s -> **~3 tok/s ceiling**, real ~1-1.5 tok/s.

The levers in order of size: (1) fewer bytes from SSD (bits, precision on miss); (2) do not waste the scarce links (CPU compute instead of PCIe; no wrong prefetches); (3) keep the SSD queue full (async reads, lookahead); (4) small cache-policy gains.

### 3.2 Ranked list of techniques to build

**R1. Expert-contiguous file layout + async direct-I/O reader + preallocated slot pools.** Sources: Eliseev (G05 §3.3), LLM in a Flash (G02 §2.2, §3.3, App. C), PowerInfer-2 (G04 §4.4), DeepSpeed ZeRO-Inference (G11 §VI-B), colibri (`E_lowmem_engines.md`).
- One expert = one aligned blob (gate, up, down, scales) in the file, sorted by layer. One read per expert. Reads ≥1 MB stay near sequential bandwidth.
- Bypass the OS page cache (Windows: `FILE_FLAG_NO_BUFFERING` + `OVERLAPPED` or IoRing; this is my inference, not from the papers). Keep 4-16 reads in flight (NVMe has deep queues, unlike UFS).
- RAM slot pool and VRAM slot pool, preallocated, fixed slot size per model; swap-delete/append bookkeeping as in LLM in a Flash Fig. 6. A small pinned staging pool for RAM->VRAM copies (Eliseev b = 4).
- Expected effect: brings real SSD throughput from page-fault level (HOBBIT §5.2 shows llama.cpp `mmap` page-fault collapse under memory pressure; LLM in a Flash shows 1.25 GB/s for small scattered reads vs 6.1 GB/s sequential) to ~4.5-5.5 GB/s. **[est] 2-5x over llama.cpp `mmap` when the model is 3-10x larger than RAM.** This is the foundation for every other item.

**R2. CPU-compute vs GPU-upload decision for every non-VRAM expert (Fiddler rule).** Sources: Fiddler (G06 Algorithm 1), PowerInfer (G03 Fig. 6), MoE-Lightning (G07 Eq. 9), ProMoE (G10 §7.2, LO wins at low cache rate).
- Rule: if the expert is in VRAM, run on GPU. Otherwise compare `cpu_lat(s) = a + b·s` against `gpu_lat + bytes/PCIe_bw`, with constants measured at startup. s = tokens routed to this expert in this step.
- [est] With our numbers, at s = 1 the CPU path is ~1.3-2x cheaper for a RAM-resident expert and does not evict a VRAM expert. Crossover ~3-8 tokens -> prefill and speculative-verify use the GPU path.
- RAM bandwidth is shared. Account for SSD DMA writes and CPU reads on the same ~20 GB/s bus in the cost model.
- Expected effect: **1.2-1.5x decode** vs "always upload misses" [est]. Also keeps the VRAM cache stable.

**R3. Mixed-precision expert loading on miss (HOBBIT), stored as nested bit planes.** Sources: HOBBIT (G09 §3.2), Eliseev Table 1, QMoE, Drop-by-Drop / Any-Precision / Matryoshka (`B_2026_quant_papers.md`, Q12, Q13).
- Score each selected expert by cumulative gate weight of higher-ranked experts (HOBBIT Eq. 2). Top-1 always full precision. s ≤ T1 -> full; T1 < s ≤ T2 -> low; s > T2 -> skip. Start with T1 = 0.6, T2 = 0.9, then retune for top-8 models with a quality test (PPL + a small task set). HOBBIT kept ≤1% loss at 30% low / 3% skip.
- Do not store two full copies (82 GB disk limit). Store bit-plane-major: the low-precision version is a prefix read of the full one. If no nested quantizer is ready, store a separate low copy only for the cold experts (by profile).
- Prefetches use the low-precision version (HOBBIT Fig. 9) to cut wasted bytes on wrong guesses.
- Cache two precision classes with a miss-penalty-aware priority.
- Expected effect: **[est] 1.2-1.4x decode when SSD-bound**; HOBBIT measured 1.19-1.57x, largest on the slow-SSD device (Fig. 16).

**R4. Next-layer expert prediction with a full prefetch pipeline.** Sources: Eliseev (G05 §3.2), HOBBIT (G09 §3.3), ProMoE (G10 §4-5), Deja Vu (G08 §3.3, §4.3).
- Step 1: gate lookahead — apply the gates of layers l+1..l+n to the layer-l gate input, stacked in one matmul (HOBBIT). Measure per-layer recall on our target model. HOBBIT: 96% top-1 next layer on Mixtral; ProMoE: 66.9% on Qwen2-57B; colibri: 71.6% recall (`E_lowmem_engines.md`).
- Step 2: if recall < ~80%, train a ProMoE-style 2-layer MLP per layer (~2M parameters each) on routing traces from our offline simulator. Run it on the CPU (~0.2 ms).
- Prefetch engine (ProMoE): two-priority queue; chunked tasks (≤1-4 MB per chunk); early preemption at each gate (drop this layer's speculative tasks, push real misses as high priority); **reordered expert execution** (cached first, in-flight next, unstarted last); predicted experts protected from eviction (HOBBIT); speculative loads never evict live experts until used (Eliseev).
- Adaptive depth: predict deeper only when the next layer's predicted experts are all resident (HOBBIT), and budget prefetch bytes by GoodPred (ProMoE) so wrong guesses do not steal SSD bandwidth.
- Expected effect: honest range **[est] 1.1-1.3x decode** (HOBBIT 1.01-1.05x, ProMoE 1.09-1.31x over LRU, Eliseev ~1.34x on T4 from RAM). The gain comes from keeping the SSD queue busy during attention and gate compute. It cannot exceed the idle fraction of the SSD. **Prefill gain is larger (up to ~2x from reordering and preemption, ProMoE Fig. 12).**

**R5. Expert cache policy: LRU + sequence-level frequency + layer distance + pinned hot set, chosen by trace replay.** Sources: HOBBIT (G09 §3.4), Eliseev (G05), Fiddler (G06 App. C), PowerInfer (G03 §6), LLM in a Flash windowing.
- Default: per-layer LRU (Eliseev, ProMoE, LLM-in-a-Flash windowing all reduce to this).
- Add HOBBIT's weighted score: recency + per-sequence use count + per-sequence full-precision count + distance to the current layer; reset counters per sequence. Fit the 4 weights by replaying traces in the offline simulator (`00_SYNTHESIS.md` §6 step 2).
- Pin a small VRAM hot set from a profile (PowerInfer/Fiddler/colibri). Expect only +3-5 points of hit rate from static popularity on balanced models (Fiddler App. C). Shared experts (DeepSeek/Qwen-style) are always hot: pin them in VRAM first.
- **LRU vs LFU vs learned:** on low-skew modern MoEs (ProMoE Fig. 5), no policy changes the hit rate much; HOBBIT's combined policy gave 4.69-8.68% lower miss penalty than LRU and 2.13-4.19% lower than LFU; sequence-level LFU beat model-level LFU by +4.5% hit rate. A learned policy has no measured result in these papers. Build LRU first, add the weighted score (cheap), skip learned eviction unless the simulator shows >10% gain.
- Expected effect: **[est] 1.05-1.1x.** Cache capacity matters more than policy.

**R6. Prefill and batch modes: load each expert once per layer for all tokens; run on GPU.** Sources: Fiddler (G06), ProMoE (reorder), MoE-Lightning (CGOPipe, HRM), FlexGen (zig-zag), DeepSpeed MoE kernels (token->expert table + scan).
- Group tokens by expert with a token->expert table and inverse scan (DeepSpeed §V-C). Iterate experts in cache-aware order (ProMoE). Upload to GPU when s > crossover (R2).
- For offline batch jobs, use a FlexGen/MoE-Lightning-style schedule: many sequences per expert read; HRM to pick batch and micro-batch under the RAM limit for KV.
- Expected effect: **prefill [est] 2-5x** over token-by-token expert loading; batch throughput roughly linear in batch size until RAM for KV runs out.

**R7. A tiered planner (HRM + LP).** Sources: MoE-Lightning HRM (G07 §3.2, §4.2), FlexGen LP (G01 §4.3), PowerInfer ILP (G03 §6).
- Inputs: measured bandwidths and FLOPS of each tier, model shapes, routing statistics. Output: what stays in VRAM (non-expert layers, shared experts, KV, hot experts), RAM cache size, CPU vs GPU attention, prefetch depth.
- Expected effect: not a speedup by itself; it prevents bad placements (FlexGen "no policy search" 0.27 vs 0.69 tok/s, Table 4).

**R8. Lossless entropy coding of low-bit expert codes (QMoE idea) — research item.** Source: QMoE (G12 §4).
- First measure the empirical entropy of the target model's IQ2/Q2/Q3 expert codes. Build only if lossless gain ≥1.2x and a CPU (AVX2) decoder runs faster than SSD input rate. QMoE's dictionary kernel is GPU-only; a GTX 1650 port is possible (512 KB dictionary vs ~1 MB L2 [verify]) but useful only for VRAM-resident experts, which are not the bottleneck.
- Sub-1-bit ternary QMoE on modern decoder MoEs: not supported by evidence (Eliseev §3.3 found too much PPL loss on Mixtral). Do not plan on it.
- Expected effect: **[est] 1.0-1.2x** for k-quant codes; unknown for a zero-heavy quantizer.

**R9. Neuron-level sparsity on SwiGLU (PowerInfer / LLM in a Flash) — do not build for the SSD tier.** See question 5 below.

### 3.3 Answers to the specific questions

1. **Expert cache policy (LRU vs LFU vs learned).** Use per-layer LRU as the base, plus HOBBIT's weighted score with sequence-level counters and layer distance, plus a pinned set of shared/hot experts. Evidence: HOBBIT measured LRU and LFU each win on some model/device pairs, and the combination wins everywhere by 2-9% miss penalty (Fig. 18a). ProMoE shows low skew in modern decoder MoEs (Fig. 5) and still uses plain LRU. No paper here measures a learned eviction policy; the learned part that pays off is the **predictor** (ProMoE), not the evictor. Tune on replayed traces before writing engine code.

2. **Next-layer expert prediction (gate lookahead).** Build Eliseev's gate lookahead first (free), stacked over n = 1-3 layers (HOBBIT). It is accurate on Mixtral (96% top-1) but fails on gate-sensitive models (Qwen2-57B: 66.9%). Then add ProMoE's learned 2-layer MLP per layer with stride +1, run on CPU. Judge it by GoodPred (accuracy x on-time rate), not accuracy. Pair it with protect-from-eviction and low-precision speculative loads, and with ProMoE's priority queue, preemption and reordering. Expect 1.1-1.3x decode; do not expect miracles, because prefetch adds no bandwidth (HOBBIT 1.01-1.05x).

3. **Mixed-precision expert loading (HOBBIT).** Yes; it is the biggest single byte-saving technique in this set that keeps accuracy (≤1% drop at 30% low-precision / 3% skip on Mixtral). For top-8 models, retune T1/T2 because gate mass is spread across more experts. Avoid two full copies on disk; use a nested bit-plane format (Drop-by-Drop / Any-Precision / Matryoshka) so the low copy is a prefix read. Expected 1.2-1.4x when SSD-bound.

4. **CPU-compute vs GPU-upload for a missed expert (Fiddler).** At batch 1 decode, compute missed experts on the CPU; upload only for prefill / multi-token steps above a measured crossover. On our machine PCIe (~11 GB/s) is slower than RAM reads (~20 GB/s), so the CPU path is favored even more than in Fiddler's setup (32-64 GB/s PCIe). Note the RAM bus is shared with SSD DMA, and the CPU also runs the I/O threads and maybe attention; the planner (R7) must budget CPU time. ProMoE's own data (LO beats ProMoE at low cache rates) agrees.

5. **Neuron-level sparsity on SwiGLU models (PowerInfer / LLM in a Flash).** Mostly no for our targets.
   - Evidence: Fiddler App. B: <2% of Mixtral post-SiLU values below 0.001 in all layers. PowerInfer Fig. 4: SwiGLU-70B needs 75% of neurons for 80% of activations (OPT: 17%). PowerInfer Table 8: SwiGLU models gain only 1.5-1.7x vs 10.9x for ReLU OPT. PowerInfer-2 Table 6: SiLU Mistral-7B 2.4x vs 4.6x for dReLU Bamboo. LLM in a Flash requires relufied models (App. C.5).
   - Size argument for experts [est]: a Qwen3-30B-A3B expert has 768 intermediate neurons; one neuron's gate+up+down bundle at 4.8 bpw is ~3.7 KB. That is below the ≥32 KiB efficient SSD read size (LLM in a Flash Fig. 2b). Neuron-level reads from SSD would run at a fraction of bandwidth. The expert already is the sparsity unit.
   - Where it may still help: on the **CPU-resident** dense/shared parts, threshold-based SiLU sparsity (CATS, CHESS — blind spots below) can cut RAM reads by ~30-50% at small quality cost. That is a RAM-bandwidth trick, not an SSD trick. Low priority.
   - Exception: a retrained dReLU MoE (TurboSparse-Mixtral-47B) reached 11.68 tok/s on a phone (PowerInfer-2). If such a model exists for our size class, re-evaluate.

6. **Row-column bundling and windowing (LLM in a Flash).** Both transfer, in expert form. Bundling = store each expert's gate, up, down (and scales) as one contiguous blob, and for any sub-expert format, keep the i-th row of gate/up with the i-th column of down (PowerInfer-2 §4.4). Windowing = keep what the last k tokens used = LRU with a k-token horizon; use it as the baseline policy. Also adopt the preallocated slot matrix (Fig. 6), direct I/O and ≥32 KiB reads with many in flight. Co-activation bundling across units does not work (G02 App. D, G04 §4.2).

7. **QMoE sub-1-bit compression and its decode kernel.** The method works for SwitchTransformer-c2048 (encoder-decoder, top-1, 2048 experts): ternary + dictionary code = 0.807 bpp with +6.7% relative loss, kernel faster than BF16 cuBLAS. It is not shown for decoder chat MoEs, and Eliseev found it too lossy on Mixtral. The kernel is CUDA-only with a 512 KB L2-resident dictionary; no CPU path. Keep two ideas: (a) MoE experts tolerate lower bits than dense layers; (b) a fused "decode-in-registers + matvec" kernel with a static shared dictionary is how to turn spare compute into fewer bytes. Plan one measurement (entropy of our expert codes), not an implementation.

---

## 4. Blind spots: important cited papers we do not have

IDs are copied from the reference lists of the 12 papers, unless marked "not in references". I did not download anything.

| Priority | Paper | arXiv ID | Cited by | Why it matters |
|---|---|---|---|---|
| **High** | MoE-Infinity: Activation-aware expert offloading for efficient MoE serving (Xue et al.) | 2401.14361 | HOBBIT, ProMoE, Fiddler | Sequence-level expert activation tracing for prefetch and LFU caching. The baseline HOBBIT beats by 2.3-9.9x; we need its design to judge the claims. |
| **High** | Pre-gated MoE: An algorithm-system co-design for fast and scalable MoE inference (Hwang et al., ISCA 2024) | not in references (ISCA paper) | HOBBIT, ProMoE | Changes the router to pick next-layer experts one layer early. Upper bound for prediction-based prefetch. |
| **High** | EdgeMoE: Fast on-device inference of MoE-based LLMs (Yi et al.) | 2308.14352 | HOBBIT, ProMoE, PowerInfer-2 | Per-expert static bit-widths + preloading on edge devices. Direct competitor to HOBBIT's dynamic precision. |
| **High** | AdapMoE: Adaptive sensitivity-based expert gating and management (Zhong et al.) | 2408.10284 | HOBBIT | Adaptive expert skipping + prefetch + cache; HOBBIT says it loses accuracy at top-2. Needed for the skip threshold T2. |
| **High** | Turbo Sparse: Achieving LLM SOTA performance with minimal activated parameters | 2406.05955 | PowerInfer-2 | dReLU MoE (TurboSparse-Mixtral-47B) behind the 11.68 tok/s phone result. Decides whether sparse MoE checkpoints exist for us. |
| **High** | CATS: Contextually-aware thresholding for sparsity in LLMs | 2404.08763 | HOBBIT, PowerInfer, PowerInfer-2 | SiLU threshold sparsity without retraining: the only route to neuron sparsity on SwiGLU. |
| Medium | CHESS: Optimizing LLM inference via channel-wise thresholding and selective sparsification | 2409.01366 | PowerInfer-2 | Same question as CATS, newer. |
| Medium | SwapMoE: Serving off-the-shelf MoE-based LLMs with tunable memory budget (Kong et al., ACL 2024) | not in references | HOBBIT, ProMoE | Keeps a dynamic "important expert" set in memory and only uses those; changes model output. Useful as a lower-quality, high-speed mode. |
| Medium | Adaptive gating in MoE-based language models (Li et al.) | 2310.07188 | HOBBIT | Dynamic number of experts per token; reduces bytes per token. |
| Medium | ReLU strikes back: Exploiting activation sparsity in LLMs (Mirzadeh et al.) | 2310.04564 | Fiddler, LLM in a Flash, PowerInfer | Relufication cost and gains. |
| Medium | ProSparse: Introducing and enhancing intrinsic activation sparsity | 2402.13516 | PowerInfer-2 | Sparse Llama/Qwen used by PowerInfer-2. |
| Medium | ReLU² wins: Discovering efficient activation functions for sparse LLMs | 2402.03804 | PowerInfer | Activation choice vs sparsity. |
| Medium | FastDecode: High-throughput GPU-efficient LLM serving using heterogeneous pipelines (He & Zhai, 2024) | not in references | MoE-Lightning | CPU attention design; relevant if KV stays in RAM. |
| Medium | ZeRO-Infinity: Breaking the GPU memory wall for extreme scale deep learning (Rajbhandari et al., 2021) | not in references | DeepSpeed-Inf., Fiddler, HOBBIT | Source of the NVMe I/O engine design (async bulk reads, pinned buffers). |
| Medium | DeepSeekMoE: Towards ultimate expert specialization | 2401.06066 | ProMoE, Fiddler | Fine-grained + shared experts: changes cache unit size and what to pin. |
| Low | Who says elephants can't run: Bringing large scale MoE models into cloud scale production (Kim et al.) | 2211.10017 | QMoE | MoE weight quantization at 4-8 bits in production. |
| Low | Mixture of Quantized Experts (MoQE) (Kim et al.) | not in references (OpenReview 2022) | Eliseev, QMoE | Low-bit expert robustness. |
| Low | FineQuant: fine-grained weight-only quantization (Kim et al.) | 2308.09723 | QMoE | Group-wise quant for experts. |
| Low | Flash-LLM: Cost-effective inference with unstructured sparsity | 2309.10285 | PowerInfer, ProMoE | GPU sparse kernels; not for our SSD tier. |
| Low | Fast on-device LLM inference with NPUs (mllm-NPU) | 2407.05858 | PowerInfer-2 | NPU only; no NPU here. |
| Low | Brainstorm (OSDI 2023), LinnOS (OSDI 2020) | no arXiv in references | ProMoE | Router abstraction / learned I/O predictor ideas. |
| Low | HQQ: Half-quadratic quantization (Badri & Shaji, blog) | no arXiv | Eliseev, ProMoE | Data-free quantizer used for 2-3-bit Mixtral experts. |

Also missing from all 12 papers (not cited because they are newer): any study of expert caching with **128-expert top-8** models (Qwen3, gpt-oss) at batch 1 from SSD. Our trace simulator must answer this ourselves.
