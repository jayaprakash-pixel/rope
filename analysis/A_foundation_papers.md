# Analysis A: Foundation papers (GGUF, k/i-quants, GPTQ, AWQ, SmoothQuant, BitNet, BitNet b1.58)

Analyst A. Project goal: run 70B to 700B+ LLMs on machines with 8-32 GB RAM and 0-12 GB VRAM.

Source files (<repo>\papers\): 01_GGUF_spec.md, 01b_llama.cpp_quantize_README.md, 02_GPTQ_Frantar2022.pdf, 03_AWQ_Lin2023.pdf, 04_SmoothQuant_Xiao2023.pdf, 05a_BitNet_Wang2023.pdf, 05b_BitNet_b1.58_Era_of_1bit_LLMs_2024.pdf.

Reading method: the Read tool could not render PDF pages (pdftoppm missing), and the 18.9 MB AWQ PDF did not fit the image reader. I read all five PDFs as text with `pdftotext -layout` (scratchpad copies). Body text and most tables extracted cleanly; a few figure-only numbers are marked "as extracted". No code from the papers was run.

Notation rules for this document:
- "p.N" = page N of the PDF. "Tab. N" = table N of that paper.
- **[ext]** = the fact does not come from the provided files. It comes from my knowledge of the llama.cpp source code (`ggml-common.h`, `ggml-quants.c`) or of other papers. Verify [ext] facts against the source before you build on them.
- "not stated" = the provided file does not give this fact.
- bpw = bits per weight. 1 GB = 10^9 bytes unless the source says GiB.

---

## 0. Quick arithmetic for our goal

Weight bytes = parameters x bpw / 8. This ignores the KV cache and runtime buffers.

| Model | FP16 (16 bpw) | 8.5 bpw (Q8_0) | 4.9 bpw (Q4_K_M) | ~3.0 bpw | ~2.06 bpw (IQ2_XXS / TQ2_0) | 1.58 bpw (ternary ideal) |
|---|---|---|---|---|---|---|
| 70B | 140 GB | 74 GB | 43 GB | 26 GB | 18 GB | 13.8 GB |
| 405B | 810 GB | 430 GB | 249 GB | 152 GB | 104 GB | 80 GB |
| 700B | 1400 GB | 744 GB | 429 GB | 263 GB | 180 GB | 138 GB |

Consequence: with 32 GB RAM + 12 GB VRAM (44 GB total), only a 70B model at ~4.5 bpw or below fits fully in memory. A 405B-700B model does not fit at any bit-width in these papers. For those models, the weights must stream from disk (mmap, Section 1.4) or the model must have far fewer active parameters per token (MoE). Quantization alone is necessary but not sufficient for the 405B-700B target.

---

## 1. GGUF file format (01_GGUF_spec.md)

### 1.1 Problem it solves
GGUF is the successor of the GGML, GGMF and GGJT formats. The older formats had no architecture field and an untyped list of hyperparameters. To add a hyperparameter was a breaking change (spec, "Historical State of Affairs" / "Drawbacks"). GGUF puts all metadata into typed key-value pairs. The goals (spec, "Specification"): single-file deployment, extensible, `mmap` compatible, easy to load with little code, full information in the file.

### 1.2 File layout (in order)
1. **Header** (`gguf_header_t`):
   - `uint32 magic` = bytes `0x47 0x47 0x55 0x46` ("GGUF").
   - `uint32 version` = 3 (v3 adds big-endian support; v2 changed counts from uint32 to uint64).
   - `uint64 tensor_count`.
   - `uint64 metadata_kv_count`.
   - `metadata_kv[metadata_kv_count]`: each KV = `gguf_string_t key` (uint64 length + UTF-8 bytes, no NUL; ASCII, `lower_snake_case` segments joined by `.`, max 65535 bytes) + `uint32 value_type` + value. Value types 0-12: UINT8, INT8, UINT16, INT16, UINT32, INT32, FLOAT32, BOOL, STRING, ARRAY, UINT64, INT64, FLOAT64. An ARRAY holds an element type, a uint64 element count and the elements; arrays can nest.
2. **Tensor infos** (`gguf_tensor_info_t`, one per tensor): name (GGUF string, max 64 bytes), `uint32 n_dimensions` (at most 4 today), `uint64 dimensions[n]`, `ggml_type type` (uint32), `uint64 offset`. The offset is relative to the start of the tensor-data section, not to the start of the file. It must be a multiple of ALIGNMENT.
3. **Padding**: `0x00` bytes up to the next multiple of ALIGNMENT, computed as `align_offset(pos) = pos + (ALIGNMENT - pos % ALIGNMENT) % ALIGNMENT`.
4. **Tensor data**: raw bytes of each tensor, in the encoding of its `ggml_type`. The space between tensors is padded to ALIGNMENT.

Fields are written back-to-back with no alignment, except where the spec says so. Files are little-endian by default. A big-endian file also exists, but the spec says there is no way to detect it from the file.

### 1.3 Alignment
- `general.alignment` (uint32) sets ALIGNMENT. It must be a multiple of 8. If the key is absent, ALIGNMENT = 32.
- The start of the tensor-data section and each tensor offset are aligned to ALIGNMENT.
- Required general keys: `general.architecture`, `general.quantization_version` (required if any tensor is quantized; it is a version of the block encoding, separate from the type name, e.g. Q5_K at quantization version 4), `general.alignment`.
- `general.file_type` (optional) gives the majority type, e.g. `MOSTLY_Q4_K_M = 15`, `MOSTLY_Q6_K = 18`.

### 1.4 Why the layout lets you mmap the file
- All tensor bytes are in one contiguous section at the end of the file. Each tensor starts at a known aligned offset: `file_pos = data_section_start + tensor_info.offset`.
- The stored bytes are already in the block format that the ggml compute kernels read (spec: tensor data "should be close or identical to the data in the original model file, but may be different due to quantization"). Thus a loader does no parsing or conversion. It maps the file and sets each tensor's data pointer to `mmap_base + file_pos`.
- Alignment to 32 bytes (or more) lets SIMD loads work on the mapped pointer directly. GGJT introduced alignment for exactly this reason ("Aligns the tensors to allow for use with mmap, which requires alignment").
- Single file (or numbered shards `-00001-of-0000N`), no external config: a loader reads the header, then maps.

**Relevance for our goal (this is the most important GGUF property).** With mmap, the operating system loads weight pages from disk only when a kernel touches them. The pages are clean and file-backed. Under memory pressure the OS drops them and re-reads them later from disk. Thus a model file larger than RAM can still run: every forward pass streams the missing weights from the SSD. The token rate then depends on SSD read bandwidth, not on RAM size. Example arithmetic (not from the files): a 405B model at Q4_K_M (249.1 GB, README table) needs ~249 GB read per token if nothing is cached. At 7 GB/s NVMe this is ~36 s/token. For a MoE model, only the active experts are read per token, which reduces this by the ratio active/total. The spec itself does not discuss performance beyond "fast loading".

The spec does NOT define: page (4 KiB) alignment of tensors, per-layer grouping for prefetch, or a computation graph (listed as a future extension). These are gaps if we want to design an optimized streaming loader.

### 1.5 Metadata useful for an offload/streaming engine
`[llm].block_count`, `[llm].embedding_length`, `[llm].feed_forward_length`, `[llm].attention.head_count`, `[llm].attention.head_count_kv` (GQA; determines KV-cache size), `[llm].context_length`, `[llm].expert_count`, `[llm].expert_used_count` (MoE). Tensor names follow `blk.N.<name>.weight`, e.g. `attn_q`, `attn_k`, `attn_v`, `attn_output`, `ffn_up`, `ffn_gate`, `ffn_down`, and for MoE `ffn_gate_inp` (router), `ffn_gate_exp`, `ffn_up_exp`, `ffn_down_exp`. A streaming engine can compute byte ranges per layer and per expert from the tensor infos alone, without reading weights.

---

## 2. llama.cpp quantize tool and quant types (01b_llama.cpp_quantize_README.md + ggml_type enum in 01_GGUF_spec.md)

### 2.1 Process
1. Convert the HF model to a high-precision GGUF (`convert_hf_to_gguf.py --outtype bf16`).
2. Run `llama-quantize in.gguf out.gguf Q4_K_M [threads]`.
- Options: `--imatrix file`, `--include-weights` / `--exclude-weights`, `--output-tensor-type`, `--token-embedding-type`, `--tensor-type <regex>=<type>` (per-tensor override), `--pure` (no k-quant mixture), `--leave-output-tensor`, `--allow-requantize` (warning: "can severely reduce quality"), `--prune-layers`, `--keep-split`, `--override-kv`.
- **RAM during quantization**: README: "the models are currently fully loaded into memory ... memory and disk requirements are the same." Table (Llama 3.1): 8B 32.1 GB -> 4.9 GB (Q4_K_M); 70B 280.9 GB -> 43.1 GB; 405B 1,625.1 GB -> 249.1 GB. (The "original" numbers equal 4 bytes x params, i.e. F32 input.) Thus quantizing 405B from F32 needs ~1.6 TB RAM per the README. Practical note [ext]: from a BF16 input the need halves; llama.cpp also memory-maps the input, so the real need is partly disk-backed. The README does not state this.
- Quantization is per-tensor, round-to-nearest with a search for the best block scale. No GPU and no calibration data are needed, except to build an imatrix.

### 2.2 Measured bpw, size and speed (README, Llama-3.1-8B; hardware not stated)

| Type | bpw (file) | GiB | pp512 t/s | tg128 t/s |
|---|---|---|---|---|
| IQ1_S | 2.0042 | 1.87 | 858.88 | 79.73 |
| IQ1_M | 2.1460 | 2.01 | 847.99 | 72.92 |
| IQ2_XXS | 2.3824 | 2.23 | 852.39 | 79.86 |
| IQ2_XS | 2.5882 | 2.42 | 826.99 | 78.04 |
| IQ2_S | 2.7403 | 2.56 | 783.55 | 77.30 |
| IQ2_M | 2.9294 | 2.74 | 787.68 | 74.44 |
| IQ3_XXS | 3.2548 | 3.04 | 813.88 | 73.95 |
| IQ3_XS | 3.4977 | 3.27 | 708.71 | 71.67 |
| IQ3_S | 3.6606 | 3.42 | 798.78 | 69.31 |
| IQ3_M | 3.7628 | 3.52 | 768.70 | 70.15 |
| IQ4_XS | 4.4597 | 4.17 | 771.80 | 77.51 |
| IQ4_NL | 4.6818 | 4.38 | 806.03 | 76.63 |
| Q2_K_S | 2.9697 | 2.78 | 798.91 | 90.01 |
| Q2_K | 3.1593 | 2.95 | 784.45 | 79.85 |
| Q3_K_S | 3.6429 | 3.41 | 752.17 | 69.84 |
| Q3_K_M | 3.9960 | 3.74 | 783.44 | 71.68 |
| Q3_K_L | 4.2979 | 4.02 | 761.17 | 69.38 |
| Q4_K_S | 4.6672 | 4.36 | 818.55 | 76.71 |
| Q4_K_M | 4.8944 | 4.58 | 821.81 | 71.93 |
| Q5_K_S | 5.5704 | 5.21 | 752.52 | 69.53 |
| Q5_K_M | 5.7036 | 5.33 | 758.69 | 67.23 |
| Q6_K | 6.5633 | 6.14 | 812.01 | 58.67 |
| Q8_0 | 8.5008 | 7.95 | 865.09 | 50.93 |
| F16 | 16.0005 | 14.96 | 923.49 | 29.17 |

Observations:
- Token generation (tg) is memory-bandwidth bound: F16 29.17 t/s -> Q4_K_M 71.93 t/s (2.47x) -> Q2_K_S 90.01 t/s (3.09x). Lower bpw gives faster generation.
- Prompt processing (pp) is compute bound: F16 is the fastest (923.49 t/s). Dequantization costs compute.
- **The README gives no perplexity or KLD numbers.** Accuracy per type is "not stated" in our files.
- The file-level bpw is higher than the tensor-level bpw of the named type, because a "mix" keeps some tensors (output, token embedding, attn_v, ffn_down) at higher types. Example: IQ1_S tensor type is 1.5625 bpw [ext], but the IQ1_S file is 2.0042 bpw.

### 2.3 Block structures of each ggml type
The provided files list the type codes (`enum ggml_type`) but **do not document the block structures**. The spec says to take block sizes from `include/ggml.h`. All structures below are **[ext]** (from `ggml-common.h`), except where a cross-check with the README's measured bpw is given. QK_K = 256 (super-block size). `d`, `dmin` are FP16 unless noted.

Dequantization rule, general form: `w = d * q` (symmetric, "_0" types) or `w = d * q + m` / `w = d*sc*q - dmin*mn` (asymmetric, "_1" and K types).

| Code | Type | Block | Bytes/block | bpw | Structure [ext] |
|---|---|---|---|---|---|
| 0 | F32 | 1 | 4 | 32 | |
| 1 | F16 | 1 | 2 | 16 | |
| 30 | BF16 | 1 | 2 | 16 | |
| 2 | Q4_0 | 32 | 18 | 4.5 | d (fp16) + 16 B of 4-bit q; w = d*(q-8) |
| 3 | Q4_1 | 32 | 20 | 5.0 | d, m (fp16) + 16 B; w = d*q + m |
| 6 | Q5_0 | 32 | 22 | 5.5 | d + 4 B high bits + 16 B low nibbles |
| 7 | Q5_1 | 32 | 24 | 6.0 | d, m + 4 B high bits + 16 B |
| 8 | Q8_0 | 32 | 34 | 8.5 | d + 32 int8. README file bpw 8.5008 matches |
| 9 | Q8_1 | 32 | 36 | 9.0 | d, s (sum) + 32 int8; activation-side type for dot products |
| 10 | Q2_K | 256 | 84 | 2.625 | 16 sub-blocks x 16; 4-bit scale + 4-bit min per sub-block (16 B); 2-bit q (64 B); d, dmin |
| 11 | Q3_K | 256 | 110 | 3.4375 | 16 sub-blocks x 16; hmask 32 B (high bit) + qs 64 B (low 2 bits) + 12 B of 6-bit scales; d; symmetric |
| 12 | Q4_K | 256 | 144 | 4.5 | 8 sub-blocks x 32; 6-bit scale + 6-bit min per sub-block packed in 12 B; 4-bit q 128 B; d, dmin |
| 13 | Q5_K | 256 | 176 | 5.5 | Q4_K + 32 B of 5th bits |
| 14 | Q6_K | 256 | 210 | 6.5625 | 16 sub-blocks x 16; ql 128 B + qh 64 B; 16 int8 scales; d. README file bpw 6.5633 matches |
| 15 | Q8_K | 256 | 292 | 9.125 | fp32 d + 256 int8 + 16 int16 block sums; only for activations in K-quant dot products |
| 16 | IQ2_XXS | 256 | 66 | 2.0625 | per 8 weights: 8-bit index into a 256-entry E8-lattice-based codebook + 7 sign bits (8th sign by parity); 4-bit scale per 32 |
| 17 | IQ2_XS | 256 | 74 | 2.3125 | 9-bit index into 512-entry grid + 7 sign bits per 8 weights; 4-bit scale per 16 |
| 22 | IQ2_S | 256 | 82 | 2.5625 | 10-bit grid index (1024 entries) + 8 explicit sign bits; 4-bit scale per 16 |
| 18 | IQ3_XXS | 256 | 98 | 3.0625 | 8-bit index into 256-entry grid of 4-vectors + sign bits; 4-bit scale per 32 |
| 21 | IQ3_S | 256 | 110 | 3.4375 | 9-bit index (512-entry grid of 4-vectors) + signs; 4-bit scale per 32 |
| 19 | IQ1_S | 256 | 50 | 1.5625 | 11-bit index into 2048-entry grid of 8-vectors with values in {-1,0,+1}; 3-bit scale + 1 shift-sign bit per 32 |
| 29 | IQ1_M | 256 | 56 | 1.75 | IQ1_S variant, shift per 16, 3-bit scales per 16, super-scale packed into the scales (no separate d) |
| 20 | IQ4_NL | 32 | 18 | 4.5 | d + 32 x 4-bit indices into a fixed non-uniform 16-value table (-127..113) |
| 23 | IQ4_XS | 256 | 136 | 4.25 | IQ4_NL codebook, 8 sub-blocks of 32 with 6-bit scales; d |
| 34 | TQ1_0 | 256 | 54 | 1.6875 | ternary {-1,0,+1}; 5 trits per byte (3^5 = 243 <= 256) for 240 weights + 4 B for 16 weights at 4 per byte; d |
| 35 | TQ2_0 | 256 | 66 | 2.0625 | ternary stored as 2 bits each (64 B) + d; faster to unpack than TQ1_0 |
| 39 | MXFP4 | 32 | 17 | 4.25 | 32 FP4 E2M1 values + 1 B E8M0 (power-of-two) shared scale (OCP MX format) |
| 40 | NVFP4 | 64 (4 x 16) | not stated | ~4.5 [ext] | spec comment: "4 blocks, E4M3 scale"; FP4 E2M1 values with FP8 E4M3 scale per 16 |
| 41 | Q1_0 | not stated | not stated | not stated | new type; not documented in our files |
| 42 | Q2_0 | not stated | not stated | not stated | new type; not documented in our files |
| 43 | F8_E4M3 | 1 | 1 | 8 | FP8 |
| 24-28 | I8, I16, I32, I64, F64 | 1 | 1-8 | 8-64 | plain integer / double |

Removed from GGUF: Q4_2, Q4_3, and the repacked CPU layouts Q4_0_4_4, Q4_0_4_8, Q4_0_8_8, IQ4_NL_4_4/4_8/8_8 (spec enum comments). [ext] llama.cpp now repacks Q4_0 / IQ4_NL at load time for ARM/AVX kernels instead of storing the repacked layout.

**Named mixes (file types).** Q2_K_S, Q3_K_S/M/L, Q4_K_S/M, Q5_K_S/M, IQ2_M, IQ3_XS/M etc. are not tensor types. They are recipes that assign different tensor types per tensor role. [ext] Example: Q4_K_M uses Q6_K for about half of `attn_v` and `ffn_down` tensors and Q4_K for the rest; `output.weight` is usually Q6_K. The README does not state the recipes. The measured file bpw (Section 2.2) reflects these mixes.

**Which types matter for us.**
- For a 70B model in 32-44 GB total memory: Q3_K_M (3.996 bpw -> ~35 GB), IQ3_XXS (3.25 -> ~28 GB), IQ2_M (2.93 -> ~26 GB), IQ2_XXS (2.38 -> ~21 GB).
- TQ1_0 / TQ2_0 are the only native storage for natively ternary (BitNet b1.58) models: 1.69 / 2.06 bpw. A 700B ternary model in TQ1_0 = ~148 GB. It still does not fit in 32 GB RAM.

### 2.4 Importance matrix (imatrix)
The README only says: an imatrix minimizes the accuracy loss; `--imatrix file`, `--include-weights`, `--exclude-weights`; and it lists the PRs (#4861 importance matrix, #4930 imatrix for all k-quants, #4957 imatrix on GPU, #4969 imatrix for legacy quants). The mechanism is **not stated** in our files. Mechanism [ext]:
1. `llama-imatrix -m model-f16.gguf -f calib.txt` runs the model on calibration text (a few hundred thousand tokens of generic text is typical).
2. For each linear weight tensor W (out x in) it accumulates, for each input column j, `imp_j = mean over tokens of x_j^2`. This is the diagonal of X^T X, i.e. the diagonal of the GPTQ Hessian (Section 3) and the same "activation magnitude" signal as AWQ (Section 4). For MoE experts it counts only the tokens routed to that expert.
3. When `llama-quantize` chooses the scale (and min) of each block, it minimizes the weighted error `sum_j imp_j * (w_j - d*q_j)^2` instead of the plain error. Without an imatrix the k-quants use a heuristic weight such as `w_j^2` or a constant.
4. Effect: channels that see large activations get smaller rounding error. The low-bit i-quants (IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS) need an imatrix; llama-quantize refuses or warns without one.
- Cost: one forward pass over the calibration text at high precision. It can run on CPU, or offloaded to GPU (PR #4957). RAM need equals that of normal inference of the high-precision model (mmap applies).
- Unlike GPTQ, imatrix does not update other weights to compensate for error. It only changes the scale choice per block. It is a diagonal-only approximation.

---

## 3. GPTQ (02_GPTQ_Frantar2022.pdf; arXiv:2210.17323, ICLR 2023)

### 3.1 Problem
Round-to-nearest (RTN) weight quantization works at 8 bits but fails at 3-4 bits on 100B+ models. Accurate post-training methods (AdaRound, BRECQ, OBQ) do not scale past ~100M parameters (p.3). GPTQ is a one-shot, second-order weight quantizer that scales to 175B in ~4 GPU-hours.

### 3.2 Core mechanism
Layer-wise objective (Eq. 1, p.3): find quantized W^ that minimizes `||W X - W^ X||_2^2`, where X = layer inputs on calibration data. The problem splits into independent rows of W. The Hessian of this objective for each row is `H = 2 X X^T` (d_col x d_col); it is the same for all rows.

Base (OBQ, Eq. 2-3, p.4): quantize one weight q, then update the remaining unquantized weights F to compensate:
- `delta_F = -((w_q - quant(w_q)) / [H_F^-1]_qq) * (H_F^-1)_{:,q}`
- remove row/col q from the inverse: `H^-1_{-q} = (H^-1 - (1/[H^-1]_qq) H^-1_{:,q} H^-1_{q,:})_{-q}`.
OBQ cost is O(d_row * d_col^3).

GPTQ changes (Section 4, p.4-5):
1. **Arbitrary order.** Quantize all rows in the same fixed column order (left to right). Then H_F^-1 is the same for all rows, so it is updated d_col times, not d_row*d_col times. Cost becomes O(max(d_row*d_col^2, d_col^3)).
2. **Lazy batch updates.** Process B = 128 columns at a time. Update only inside the block, then apply one matrix update to all later columns (Eq. 4-5). This fixes the memory-bound behavior; "an order of magnitude speedup" (p.5).
3. **Cholesky reformulation.** Repeated inverse updates become indefinite on models above a few billion parameters. Fix: dampen `H <- H + lambda*I` with lambda = 1% of the mean diagonal, then precompute the upper Cholesky factor of H^-1. Row q of that factor holds all the needed information.

**Algorithm 1 (p.5), implementable form:**
```
H = 2 X X^T ; H += 0.01*mean(diag(H))*I
Hinv = Cholesky(H^-1)^T            # upper triangular
Q = 0 (d_row x d_col); E = 0 (d_row x B)
for i in 0, B, 2B, ...:
  for j in i .. i+B-1:
    Q[:,j] = quant(W[:,j])                         # RTN on the fixed grid
    E[:,j-i] = (W[:,j] - Q[:,j]) / Hinv[j,j]
    W[:,j:i+B] -= E[:,j-i] * Hinv[j, j:i+B]       # update inside block
  W[:,i+B:] -= E @ Hinv[i:i+B, i+B:]               # lazy global update
```
With groups (e.g. g128), the scale/zero of each group is computed from the current (already updated) weights when the algorithm reaches the first column of that group (p.8: "group parameters can be determined during the quantization process").

Execution (p.6): load one transformer block (6 linear layers) to GPU, run the calibration inputs to accumulate the Hessians, quantize, then run the inputs through the **quantized** block to make the inputs for the next block. This needs only one block in GPU memory at a time.

### 3.3 What is quantized
- Weights only. Activations stay FP16. KV cache: not quantized.
- Bits: 4, 3, 2, and ternary.
- Grid: "standard uniform per-row asymmetric quantization on the min-max grid" (p.6), i.e. per output channel scale + zero point. Optional groups g1024, g128, g64, g32, g8.
- Scale/zero format: for 2-bit g128, "FP16 scale and 2-bit zero point per group" (p.9). g1024 costs ~0.02 extra bits, g128 ~0.15 extra bits (p.7).
- Embeddings and output layer kept in FP16 (p.8).

### 3.4 Memory result
- OPT-175B FP16 = 326 GB (p.1). 3-bit GPTQ OPT-175B = ~63 GB including FP16 embeddings and output layer (p.8). KV cache for 2048 tokens adds ~9 GB. FP16 needs 5 x A100-80GB; LLM.int8() needs 3; 3-bit GPTQ needs 1 (p.8). "the model would not fully fit using 4 bits" on one 80 GB A100.
- Bytes/param: 3-bit ~0.375 B + group overhead (g128: +~0.15 bit).

### 3.5 Speed result
Tab. 6 (p.8), OPT-175B, batch 1, generate 128 tokens, custom 3-bit matrix x FP16-vector kernel that dequantizes on the fly:
- A6000-48GB: FP16 589 ms/token (8 GPUs) -> 3-bit 130 ms (2 GPUs), 4.53x.
- A100-80GB: FP16 230 ms (5 GPUs) -> 3-bit 71 ms (1 GPU), 3.24x.
The speedup comes only from less memory traffic. No compute reduction (p.9). For large batches, decompress first: < 1.5 ms on A100 per layer vs 76 ms for the OPT-175B FC2 matmul at 16x1024 tokens (p.13).

### 3.6 Accuracy cost (WikiText2 PPL)
Tab. 3 (OPT, p.7):

| OPT | FP16 | RTN 4b | GPTQ 4b | RTN 3b | GPTQ 3b |
|---|---|---|---|---|---|
| 6.7B | 10.86 | 12.10 | 11.39 | 5.8e3 | 14.86 |
| 13B | 10.13 | 11.32 | 10.31 | 3.4e3 | 11.61 |
| 30B | 9.56 | 10.98 | 9.63 | 1.6e3 | 10.27 |
| 66B | 9.34 | 110 | 9.55 | 6.1e3 | 14.16 |
| 175B | 8.34 | 10.54 | 8.37 | 7.3e3 | 8.68 |

Tab. 4 (BLOOM-176B): FP16 8.11; RTN4 8.37; GPTQ4 8.21; RTN3 571; GPTQ3 8.64.
Tab. 5 (p.8): GPTQ 3-bit g128: OPT-175B 8.45 (Wiki2) / 12.37 (PTB) / 10.36 (C4); BLOOM-176B 8.26 / 14.89 / 11.85. LAMBADA OPT-175B: FP16 75.59, GPTQ4 76.80, GPTQ3 76.19, RTN3 0.
Tab. 7 (2-bit, p.9): OPT-175B g128 9.58, g64 9.18, g32 8.94 (vs FP16 8.34); BLOOM-176B 9.55 / 9.17 / 8.83 (vs 8.11). ~2.2 bit (g128) costs < 1.5 PPL; ~2.6 bit (g32) costs 0.6-0.7 PPL. Ternary with g8: OPT-175B 9.20 PPL.
Trend: larger models quantize more easily (p.7), except OPT-66B (dead units in early layers, footnote 2).

### 3.7 Calibration and cost
- 128 random 2048-token segments from C4 (p.6). Generic text, no task data.
- Runtime on one A100-80GB (Tab. 2, p.7): OPT-13B 20.9 min, 30B 44.9 min, 66B 1.6 h, 175B 4.2 h; BLOOM-1.7B 2.9 min, 3B 5.2 min, 7.1B 10.0 min, 176B 3.8 h.
- GPU memory: one transformer block + its Hessians (d_col x d_col FP32 each; for d_col = 12288 that is ~600 MB per Hessian [arithmetic]). Host RAM to hold the full FP16 model: not stated (the model is loaded block by block from CPU memory; for 175B that is ~326 GB of host RAM unless streamed from disk).

### 3.8 Limitations
- Weights only; no activation quantization; speedup only for memory-bound batch-1 generation (p.2, p.9).
- At 3-bit and below, medium models (6.7B, 66B) lose a lot of PPL without grouping.
- Mixed precision FP16 x INT4 has no hardware support (p.2).
- Cholesky / dampening still needed; numerical failure without them on models > a few B parameters (p.5).
- Calibration on C4 makes the C4 PPL not fully zero-shot (Tab. 11 caption).

### 3.9 Relevance to our goal
- Directly reduces RAM/VRAM by 4-5x vs FP16 at 3-4 bit with small loss for 70B+ models. GPTQ 3-bit makes a 70B model ~27-30 GB (incl. groups).
- Composes with: grouping, AWQ-style scaling (AWQ paper reports AWQ+GPTQ), the GGUF k-quant formats (the GPTQ error-feedback idea can choose the integers for any grid; llama.cpp itself does not use GPTQ [ext]), and KV-cache methods. Its Hessian diagonal is the same signal as imatrix.
- Quantization cost for 405B+: needs a GPU with enough memory for one block and host RAM/disk for the full model. Feasible block by block on modest hardware, but slow.

---
## 4. AWQ: Activation-aware Weight Quantization (03_AWQ_Lin2023.pdf; arXiv:2306.00978v6, MLSys 2024 Best Paper)

Note on reading: the PDF (18.9 MB) did not fit the image reader. I read its full text with `pdftotext`. Figure-only numbers (some bar charts) may be partly garbled in the text; I cite only the numbers that are clear. Page numbers are approximate (by text page breaks).

### 4.1 Problem
Low-bit weight-only quantization (W4A16, W3A16) for on-device inference. QAT is too costly. RTN loses accuracy at 3-4 bits. GPTQ uses reconstruction (regression) and "may overfit the calibration set" and needs a reorder trick for some models (LLaMA-7B, OPT-66B) (p.1-2).

### 4.2 Core mechanism
Observation (Sec. 3.1, Tab. 1, p.3): 0.1%-1% of weights are salient. Salience is found from the **activation** magnitude per input channel, not from the weight magnitude. OPT-6.7B INT3-g128 WikiText PPL: FP16 10.86; RTN 23.54; keep 1% channels (by activation) in FP16: 11.39; keep 1% (by weight norm): 22.37; random 1%: 24.23.

Mixed precision is not hardware-friendly. So AWQ scales the salient channels instead (Sec. 3.2, p.4):
- Quantizer (Eq. 1): `Q(w) = Delta * Round(w / Delta)`, `Delta = max(|w|) / 2^(N-1)`, per group.
- Scale weight channel by s > 1, divide the input by s: `Q(w*s) * (x/s) = Delta' * Round(w*s/Delta') * x / s` (Eq. 2).
- Error ratio new/old = `(Delta'/Delta) * (1/s)` (Eq. 3). RoundErr stays ~0.25 on average; Delta' ~ Delta if one element changes, so the salient weight's relative error falls by ~1/s.
- Tab. 2 (OPT-6.7B): s = 1, 1.25, 1.5, 2, 4 gives PPL 23.54, 12.87, 12.48, 11.92, 12.36. Too large s raises Delta and hurts the non-salient channels (21.2% of groups change Delta at s = 4).

Search (Eq. 4-5, p.5):
- Objective: `s* = argmin_s || Q(W * diag(s)) (diag(s)^-1 * X) - W X ||`.
- Search space: `s = s_X ^ alpha`, where `s_X` = average magnitude of activation per input channel (from calibration). `alpha` in [0, 1] chosen by grid search with grid size 20 (Sec. 5.1).
- Also apply weight clipping (search a clip range that minimizes quantization MSE).
- `diag(s)^-1` is folded into the previous operator (LayerNorm scale or the previous linear layer), so inference has no extra op.

Implementable algorithm (per transformer block, per group of linear layers that share an input):
```
X = cached inputs to the layer(s) from calibration
s_X = mean(|X|, over tokens)          # per input channel
best = inf
for alpha in {0, 1/20, ..., 1}:
  s = s_X ** alpha ; s = s / sqrt(max(s)*min(s))   # [ext] normalization used in the code, not in paper
  W' = Q(W * s) / s                                 # quantize-dequantize, group size 128
  err = || W' X - W X ||
  keep s with min err
fold 1/s into previous op; then search per-group clip ratio minimizing ||Q(W)X - WX||
```
No gradient, no weight update.

### 4.3 What is quantized
- Weights only (W4A16 / W3A16; also INT2 in Tab. 9). Activations FP16. KV cache not quantized.
- Group size 128 throughout (Sec. 5.1), g64 for INT2.
- Scale/zero format: per-group scale; the paper's Eq. 1 is symmetric absmax. [ext] The released code uses asymmetric (zero point) INT4 with FP16 scales; the paper does not state the stored format.

### 4.4 Memory result
- 4x smaller weights vs FP16 (p.1-2, Fig. 3c). Fig. 3(c): weight memory dominates activation memory (Llama-2-7B, 4090).
- Llama-2-70B fits on one Jetson Orin with 64 GB (p.2). Llama-2-13B runs on an 8 GB laptop RTX 4070 at 33 tokens/s; FP16 cannot fit 7B there (p.10). Falcon-180B fits one H200 (p.2).
- Bytes/param at INT4-g128: 0.5 B + group overhead (not stated exactly; ~0.125-0.25 bits/param for FP16 scale (+zero) per 128).

### 4.5 Speed and hardware (TinyChat)
- Roofline (Fig. 3, Llama-2-7B, RTX 4090: 165 TFLOPS, 1 TB/s): generation has arithmetic intensity ~1 in FP16; W4A16 raises it to ~4.
- Context 200 tokens 10 ms vs generating 20 tokens 310 ms (Fig. 3a).
- TinyChat: 2.7-3.9x over HF FP16 on RTX 4090 (Fig. 9a); Llama-2-7B 52 -> 62 tok/s from FP16 fusion, then 3.1x more from W4 kernels; up to 3.5x on Jetson Orin; 3.2-3.3x average (p.2, p.10).
- VILA (Tab. 10): VILA-7B FP16 vs W4A16: A100 81.6 -> 155.3; 4090 58.5 -> 168.1; Orin 11.5 -> 35.6 tok/s.
- vs other systems (Fig. 10): up to 1.7x over llama.cpp on Orin; 1.2-3.0x over existing systems; Raspberry Pi 4B: 0.7 tok/s for 7B models.
- Kernel techniques: on-the-fly dequant fused into GEMM/GEMV; SIMD-aware packing for ARM NEON (store w0,w16,w1,w17,... so 3 SIMD ops unpack 32 weights; up to 1.2x) (Fig. 4, p.6); GPU packing order w{0,2,4,6,1,3,5,7}; fused LayerNorm, fused QKV, in-kernel KV cache update.

### 4.6 Accuracy (WikiText-2 PPL, Tab. 4, p.7)

| Model | FP16 | INT3-g128 RTN | GPTQ | GPTQ-R | AWQ | INT4-g128 RTN | GPTQ | GPTQ-R | AWQ |
|---|---|---|---|---|---|---|---|---|---|
| Llama-2-7B | 5.47 | 6.66 | 6.43 | 6.42 | 6.24 | 5.73 | 5.69 | 5.63 | 5.60 |
| Llama-2-13B | 4.88 | 5.52 | 5.48 | 5.41 | 5.32 | 4.98 | 4.98 | 4.99 | 4.97 |
| Llama-2-70B | 3.32 | 3.98 | 3.88 | 3.86 | 3.74 | 3.46 | 3.42 | 3.43 | 3.41 |
| LLaMA-7B | 5.68 | 7.01 | 8.81 | 6.53 | 6.35 | 5.96 | 6.22 | 5.83 | 5.78 |
| LLaMA-65B | 3.53 | 4.24 | 4.17 | 4.21 | 3.95 | 3.67 | 3.66 | 3.66 | 3.62 |

- 70B at INT4-g128: +0.09 PPL; at INT3-g128: +0.42 PPL.
- Tab. 5: Mistral-7B FP16 4.14 / INT4 4.30 / INT3 4.83; Mixtral-8x7B (MoE) 5.94 / 6.05 / 6.52.
- Tab. 8 (INT4-g128): GSM8K Llama-2-70B FP16 56.41, RTN 53.98, GPTQ 56.03, AWQ 56.40. MBPP CodeLlama-7B pass@1 FP16 38.53, AWQ 40.64.
- Tab. 6 (OpenFlamingo-9B COCO 32-shot CIDEr): INT4 degradation RTN -4.57, GPTQ -6.72, AWQ -1.17; INT3: -16.91, -16.93, -7.23.
- **INT2-g64 (Tab. 9, OPT)**: RTN fails (PPL 7622 for 6.7B); GPTQ 16.65; AWQ+GPTQ 15.71 (FP16 10.86). OPT-30B: GPTQ 11.75, AWQ+GPTQ 11.38 (FP16 9.56). AWQ is orthogonal to GPTQ and the two combine.

### 4.7 Calibration and cost
- Small calibration set from the Pile (Sec. 5.1). Fig. 8a: AWQ reaches better PPL with 16 sequences x 2048 tokens than GPTQ with 192 sequences ("10x smaller").
- Robustness (Fig. 8b, OPT-6.7B INT3-g128): cross-domain calibration (PubMed <-> Enron) costs AWQ +0.5-0.6 PPL, GPTQ +2.3-4.9 PPL.
- GPU-hours / RAM for quantization: **not stated** in this version of the paper. The method needs only forward passes of one block at a time plus a 20-point grid search per layer group.

### 4.8 Limitations
- Weight-only; batch-1 / memory-bound speedups only. Large-batch prefill gains are not claimed.
- INT2 alone is not usable; it needs GPTQ on top and still loses ~5 PPL on OPT-6.7B.
- Per-channel scales must be foldable into a previous op; layers without such an op (e.g. some down-projections) need the scale kept as an explicit op [ext: the code handles `fc2`/`down_proj` by folding into the previous linear layer `fc1`/`up_proj`].
- TinyChat speed claims are vs HF FP16 and on GPUs; CPU results are only the Raspberry Pi number.

### 4.9 Relevance to our goal
- Same memory gain as GPTQ (4x at INT4, ~5x at INT3). For a 70B model: INT4-g128 ~35-37 GB, INT3-g128 ~27-29 GB, with +0.09 / +0.42 PPL.
- The "activation-aware importance" idea is the same signal that llama.cpp's imatrix uses; AWQ scales can be applied to the FP16 model **before** GGUF conversion, then any GGUF k-quant applies on top [ext: llama.cpp supports applying AWQ scales in conversion; verify for current versions]. Combines with GPTQ (Tab. 9).
- Low data and robust to calibration domain: good for MoE and multi-domain models.

---

## 5. SmoothQuant (04_SmoothQuant_Xiao2023.pdf; arXiv:2211.10438v7, ICML 2023)

### 5.1 Problem
W8A8 (INT8 weights and INT8 activations) lets you use INT8 GEMM hardware, but LLMs above ~6.7B have systematic activation outliers ~100x larger than other values (p.3). Naive W8A8 destroys OPT-175B (WikiText PPL 93080 vs FP16 10.99, Tab. 3). LLM.int8() fixes accuracy with mixed INT8/FP16 decomposition but is slower than FP16 (Tab. 11).

### 5.2 Core mechanism
Observations (Sec. 3, Fig. 4, p.3-4): (1) weights are flat and easy to quantize; (2) activation outliers make per-tensor quantization leave only 2-3 effective levels for normal channels (effective levels of channel i = 2^8 * m_i / m); (3) outliers stay in fixed channels across tokens. Per-channel activation quantization would fix accuracy (Tab. 1: OPT-175B 71.4% vs FP16 71.6%) but INT8 GEMM can only scale along outer dims T and C_o (Eq. 2).

Smoothing (Eq. 3-4, p.4):
- `Y = (X diag(s)^-1) (diag(s) W) = X^ W^`.
- `s_j = max(|X_j|)^alpha / max(|W_j|)^(1-alpha)`, j = input channel; `alpha` = migration strength.
- alpha = 0.5 for OPT and BLOOM; 0.75 for GLM-130B; 0.8 for LLaMA (Tab. 6); 0.6-0.9 for Llama-2/Falcon/Mistral/Mixtral (Tab. 7). Sweet spot 0.4-0.6 on OPT-175B (Fig. 10, p.9).
- `diag(s)^-1` is fused into the previous LayerNorm or linear layer offline; for residual inputs, an extra scale is added on the residual branch.

Implementable steps:
```
1. Run 512 random sentences from the Pile; record per-channel max|X_j| for the input of each
   QKV projection and each FC1 (and BMM inputs).
2. s_j = max|X_j|^alpha / max|W_j|^(1-alpha)   # max|W_j| over the weight row of input channel j
3. LayerNorm.gamma /= s ; LayerNorm.beta /= s ; W = diag(s) W
4. Quantize W to INT8 per-tensor (O1-O3) or per-channel; record static activation step sizes
   (O3) or compute them at runtime (O1 per-token dynamic, O2 per-tensor dynamic).
5. Run all linear layers and attention BMMs with INT8 GEMM (CUTLASS); keep ReLU, Softmax,
   LayerNorm in FP16 (Fig. 6).
```

### 5.3 What is quantized
- Weights INT8 and activations INT8 for all linear layers and attention BMMs (Q*K^T, attn*V). Softmax, LayerNorm, ReLU stay FP16 (Fig. 6, p.5).
- Settings (Tab. 2): O1 = per-tensor weights + per-token dynamic activations; O2 = per-tensor + per-tensor dynamic; O3 = per-tensor + per-tensor static. Tab. 7 (Llama-2 etc.) uses per-channel weights + per-token activations.
- Symmetric quantizer: `X_int8 = round(X / Delta)`, `Delta = max|X| / (2^(N-1) - 1)` (Eq. 1). One FP scale per tensor/token/channel. No group-wise format.
- KV cache: not quantized in the experiments. Appendix A (p.13) notes that activation quantization "can also help reduce the memory cost from storing the KV cache" (cites 3 TB KV cache for batch 512, context 2048 vs ~1 TB weights).

### 5.4 Memory result
- ~2x (FP16 -> INT8): 1.96x on PyTorch (Fig. 8, OPT, single A100-80GB).
- MT-NLG 530B (Tab. 10): FP16 16 GPUs, 1040-1095 GB; INT8 8 GPUs, 527-570 GB, at similar latency (e.g. seq 128: 232 ms vs 253 ms; seq 512: 838 vs 839 ms).
- Decoding (Tab. 8): OPT-30B 1 GPU BS1 seq 512: 57 GB -> 30 GB (1.91x); OPT-175B 8 GPUs BS1 seq 512: 44 -> 23 GB per GPU (1.87x).
- Bytes/param: 1.0 (+ per-tensor/per-channel scale).

### 5.5 Speed result (all NVIDIA A100-80GB)
- PyTorch context stage: up to 1.51x (OPT-30B, seq 256) (Fig. 8). Tab. 11 (BS 4): OPT-13B seq 256: FP16 152.6 ms, LLM.int8() 237.1, O1 124.5, O2 120.5, O3 112.1. OPT-30B seq 512: 659.9 / 654.9 / 490.7 / 478.3 / 458.4.
- FasterTransformer: up to 1.56x (OPT-13B, OPT-30B single GPU); OPT-66B on 1 GPU instead of 2 and OPT-175B on 4 instead of 8 at similar or better latency (Fig. 9).
- Decoding (Tab. 8): OPT-30B BS1 seq 512: 422 -> 314 ms (1.35x); BS16 seq 512: 2488 -> 1753 ms (1.42x); OPT-175B BS1 seq 512: 426 -> 359 ms (1.19x).
- CPU: not evaluated. The paper says INT8 kernels exist on Intel CPUs (p.3) but gives no CPU numbers.

### 5.6 Accuracy cost (8-bit only)
- OPT-175B (Tab. 3): FP16 avg 66.9%, WikiText 10.99; O1 66.5% / 11.11; O2 66.4% / 11.14; O3 66.8% / 11.17. Naive W8A8 35.5% / 93080; ZeroQuant 35.8% / 84648; LLM.int8() 66.7% / 11.10; Outlier Suppression 36.0% / 96151.
- Tab. 4: BLOOM-176B FP16 68.2%, O1 68.3, O2 68.4, O3 67.4 (-0.8). GLM-130B FP16 73.8%, O1 73.7, O3 72.8 (-1.0).
- Tab. 6 (LLaMA, seq 512, WikiText): 7B 11.51 -> 11.56; 13B 10.05 -> 10.08; 30B 7.53 -> 7.56; 65B 6.17 -> 6.20.
- Tab. 7 (seq 2048): Llama-2-7B 5.474 -> 5.515; 13B 4.950 -> 4.929; 70B 3.320 -> 3.359; Falcon-40B 5.228 -> 5.255; Mistral-7B 5.253 -> 5.277; Mixtral-8x7B 3.842 -> 3.893.
- Tab. 9 (MT-NLG 530B): avg 73.1% FP16 and INT8.
- 4-bit and 2-bit: **not evaluated** in this paper. Appendix A lists W4A4 with GPTQ as future work.

### 5.7 Calibration and cost
- 512 random sentences from the Pile (validation subset for alpha search). Grid search over alpha "quick" (p.5). GLM-130B: clip the top 2% tokens when calibrating static step sizes.
- GPU-hours and RAM during quantization: not stated. The method needs one calibration forward pass (to collect per-channel maxima) and no weight optimization, so the cost is about one inference pass over 512 sentences.

### 5.8 Limitations
- Only 8-bit; memory gain only 2x vs FP16. For our goal 8 bits is too large (70B = 70 GB).
- Static per-tensor (O3) can lose 0.8-1% on some models (BLOOM, GLM) because calibration statistics differ from runtime statistics.
- Alpha must be tuned per model family.
- Gains come from INT8 GEMM throughput, mostly in compute-bound (batched / prefill) regimes. Batch-1 decoding gain is 1.19-1.35x.
- Appendix A: the authors could not compare fairly with GPTQ because GPTQ's kernel only supports batch-1 generation on OPT-175B.

### 5.9 Relevance to our goal
- Direct RAM reduction is small (2x). Not a standalone answer.
- The **smoothing transform is reusable**: it is the same per-channel equivalent rescaling that AWQ uses (AWQ cites it for folding `diag(s)^-1`). It makes activations easier to quantize, which matters for (a) INT8 activation x low-bit weight CPU kernels (llama.cpp already quantizes activations to Q8_0/Q8_K for its dot products [ext]), and (b) KV-cache quantization (smaller outliers in K/V).
- Composes with weight-only low-bit methods (W4A8 / W4A4 designs that later papers built; see Blind spots: QServe, QuaRot, SpinQuant).

---

## 6. BitNet: Scaling 1-bit Transformers (05a_BitNet_Wang2023.pdf; arXiv:2310.11453v1)

### 6.1 Problem
Post-training quantization loses accuracy at low bits because the model was not trained for it. BitNet trains a 1-bit-weight transformer **from scratch** (quantization-aware training) and checks that it follows a scaling law (p.2).

### 6.2 Core mechanism: BitLinear (Sec. 2.1, p.3-4)
Replace each `nn.Linear` in attention and FFN with BitLinear. Embeddings (input/output) stay high precision. Residuals and LayerNorm stay high precision ("8-bit in our experiments", p.3).

Forward pass of BitLinear (input x, latent FP weight W of shape n x m):
```
alpha = mean(W)                                 # Eq. 3
W_b   = Sign(W - alpha)  in {-1,+1}             # Eq. 1-2 (Sign(0) = -1)
beta  = ||W||_1 / (n*m)                         # Eq. 12, weight scale
x_ln  = LN(x)                                   # SubLN, keeps Var(y) ~ 1 (Eq. 8-10)
gamma = ||x_ln||_inf ; Qb = 2^(b-1), b = 8
x_q   = Clip(x_ln * Qb / gamma, -Qb + eps, Qb - eps)   # Eq. 4-5, absmax
        # before ReLU-type nonlinearity: eta = min(x); x_q = Clip((x-eta)*Qb/gamma, eps, Qb-eps)  (Eq. 6)
y     = (W_b @ x_q) * beta * gamma / Qb          # Eq. 11
```
- Activation quantization is per tensor in training, per token in inference (p.4).
- Group Quantization and Normalization (Eq. 13-15): compute alpha, beta, gamma, eta and the LN statistics per group along the model-parallel partition, so no all-reduce is needed.
- Training (Sec. 2.2): straight-through estimator for Sign and Clip. FP latent weights, FP gradients and FP optimizer states. Large learning rate (BitNet converges at LRs where FP16 diverges, Fig. 5).
- Hyperparameters (App. A, Tab. 5-6): sizes 125M to 30B. Example 6.7B: hidden 4096, 32 layers, LR 4.8e-4. 30B: hidden 7168, 48 layers, LR 4e-4. 40K updates x 256K tokens per batch (= ~10B tokens [arithmetic]). Data: Pile, Common Crawl, RealNews, CC-Stories. 16K SentencePiece vocabulary (Sec. 3.1).

### 6.3 What is quantized
- Weights: 1 bit (binary +-1), one FP scale beta per weight matrix (or per group). Activations: 8-bit absmax, per token at inference (W1A8).
- Not quantized: embeddings, output head, LayerNorm, residuals. KV cache: not discussed.

### 6.4 Memory result
- Bytes/param for BitLinear weights: 1/8 B (0.125 B) + one scale per matrix. The paper claims "substantially reducing memory footprint" but gives **no measured memory table**. Model sizes in GB: not stated.

### 6.5 Speed / energy result
- No measured latency or throughput. Only an **arithmetic-energy model** (Sec. 2.3, Tab. 1-2, with Horowitz energy numbers). Example, 512 tokens, 7nm: 6.7B FP16 Transformer MUL 1.14 J, ADD 0.54 J vs BitNet MUL 0.02 J, ADD 0.04 J. 30B: FP16 MUL 5.21 J / ADD 2.45 J vs BitNet 0.06 J / 0.14 J. Fig. 1 shows energy reduction factors that grow with size (up to ~38.8x at 30B as extracted; the figure text is partly garbled).

### 6.6 Accuracy cost (Tab. 3, p.9; all 6.7B, same training data)

| W bits | Method | PPL | Avg zero-shot (WGe/WG/HS/SC) |
|---|---|---|---|
| 16 | FP16 Transformer | 15.19 | 57.8 |
| 8 | Absmax W8A8 | 21.43 | 53.4 |
| 8 | SmoothQuant W8A8 | 15.67 | 56.7 |
| 4 | GPTQ W4A16 | 16.05 | 52.9 |
| 4 | Absmax W4A4 | 4.8e4 | 46.2 |
| 4 | SmoothQuant W4A4 | 1.6e6 | 45.1 |
| 2 | GPTQ W2A16 | 1032 | 45.2 |
| 2 | QuIP W2A16 | 70.43 | 49.0 |
| 1 | Absmax W1A8 | 3.5e23 | 44.6 |
| 1 | SmoothQuant W1A8 | 3.3e21 | 44.4 |
| 1 | **BitNet W1A8 (trained)** | **17.07** | **55.9** |

The row alignment of the extracted table is slightly unclear for the 8-bit/4-bit GPTQ row. The numbers are as extracted.
- At 6.7B, BitNet costs +1.88 PPL vs FP16 and -1.9 average points. Scaling law (Eq. 19, Fig. 3): fitted on 125M-6.7B, it predicts the 13B and 30B loss accurately. The gap to FP16 gets smaller as size grows.
- Ablation (Tab. 4): absmax + SubLN (PPL 20.34) is better than Elastic + Pre-LN (24.05), Absmax + Pre-LN (22.11), Absmax + BMT (22.98).

### 6.7 Calibration / cost
- No calibration. It is a training method. Cost = full pre-training from scratch (here ~10B tokens per model; compute not stated). Training memory is equal to or more than FP16 training (latent FP weights + optimizer states).
- **You cannot apply BitNet to an existing 70B-700B model.** PTQ to 1 bit fails completely (PPL 1e21-1e23 in Tab. 3).

### 6.8 Limitations
- Training from scratch is necessary. Largest model 30B on ~10B tokens (far below production data scales).
- No measured speed or memory. Energy is a model estimate.
- Binary weights lose accuracy vs FP16 at the sizes tested. b1.58 (Section 7) fixes this with a 0 value.

### 6.9 Relevance to our goal
- Inference memory for linear weights = 1/16 of FP16. A 700B binary model needs ~88 GB for weights plus FP embeddings. This is useful only if someone trains such a model. It is mainly a design reference and evidence that 1-bit PTQ fails while QAT works.

---

## 7. BitNet b1.58: The Era of 1-bit LLMs (05b_BitNet_b1.58_Era_of_1bit_LLMs_2024.pdf; arXiv:2402.17764v1)

### 7.1 Problem
Same as BitNet. Goal: match FP16 LLaMA quality at the same size and the same number of training tokens, with ternary weights.

### 7.2 Core mechanism (Sec. 2, p.2-3)
- Weights ternary {-1, 0, +1}, quantized with **absmean**:
  `gamma = (1/(n*m)) * sum |W_ij|` (Eq. 3); `W~ = RoundClip(W / (gamma + eps), -1, 1)` (Eq. 1); `RoundClip(x,a,b) = max(a, min(b, round(x)))` (Eq. 2).
- Activations: 8-bit absmax, scaled to [-Qb, Qb] **per token**, no zero point (also before non-linearities).
- LLaMA-like components: RMSNorm, SwiGLU, rotary embeddings, no biases. Thus llama.cpp, vLLM and HF can run it "with minimal efforts" (p.3).
- The value 0 enables "feature filtering" (p.2). log2(3) = 1.58 bits of information per weight.
- Training: from scratch. RedPajama 100B tokens for the comparison models (Sec. 3). One 3B model on 2T tokens (Tab. 4).

Inference compute: for each output, add +x_j, subtract x_j, or skip, then multiply once by gamma and the activation scale. The matmul uses only INT8 additions (Fig. 3).

### 7.3 What is quantized
- All linear-layer weights: ternary, one scale per matrix (absmean). Activations: INT8 per token. Embeddings: full precision (p.4: "the embedding remains full precision").
- KV cache: the paper says 8-bit activations allow 2x context for the same memory; 4-bit or lower KV is "left as future work" (p.6).
- Storage format in the paper: the measured runs use a **2-bit kernel** (Ladder), i.e. 2 bits per weight in memory (p.3-4). Ideal packing is 1.58 bits. [ext] llama.cpp TQ1_0 = 1.6875 bpw, TQ2_0 = 2.0625 bpw.

### 7.4 Memory result (GPU memory at runtime, FasterTransformer + Ladder 2-bit kernel; Tab. 1, Fig. 2)

| Size | LLaMA FP16 memory | b1.58 memory | Ratio | LLaMA latency | b1.58 latency | Speedup | PPL LLaMA / b1.58 |
|---|---|---|---|---|---|---|---|
| 700M | 2.08 GB | 0.80 GB | 2.60x | 1.18 ms | 0.96 ms | 1.23x | 12.33 / 12.87 |
| 1.3B | 3.34 GB | 1.14 GB | 2.93x | 1.62 ms | 0.97 ms | 1.67x | 11.25 / 11.29 |
| 3B | 7.89 GB | 2.22 GB | 3.55x | 5.07 ms | 1.87 ms | 2.71x | 10.04 / 9.91 |
| 3.9B (b1.58 only) | - | 2.38 GB | 3.32x vs 3B | - | 2.11 ms | 2.40x vs 3B | 9.62 |

Fig. 2 (larger sizes, ratios only): memory reduction 4.40x (7B), 5.12x (13B), 7.16x (70B). Latency speedup 2.90x (7B), 3.68x (13B), 4.10x (70B). Absolute GB for 70B: not stated. [arithmetic] 70B x 2 bits = 17.5 GB of weights; at 1.58 bits = 13.8 GB.

### 7.5 Speed and hardware
- Decoding latency on GPU (FasterTransformer; GPU model not stated for Tab. 1). 70B: 4.1x faster than FP16 LLaMA (p.4).
- Throughput (Tab. 3), 70B on 2 x A100-80GB, pipeline parallel, seq 512: LLaMA max batch 16, 333 tok/s. b1.58 max batch 176 (11.0x), 2977 tok/s (8.9x).
- Energy model (Fig. 3): 71.4x less matmul arithmetic energy at 7nm. End-to-end energy 18.6x (1.3B), 21.7x (3B), 29.1x (7B), 32.9x (13B), 41.2x (70B).
- CPU: the paper says 1.58-bit LLMs are "more friendly to CPU devices" (p.6) but gives **no CPU measurements**.

### 7.6 Accuracy
- Tab. 2 (zero-shot average of ARCe, ARCc, HS, BQ, OQ, PQ, WGe): 700M LLaMA 45.5 vs b1.58 44.3; 1.3B 46.2 vs 45.4; 3B 49.7 vs 50.2; 3.9B b1.58 51.2. Parity starts at 3B.
- Tab. 4 (2T tokens, 3B): StableLM-3B avg 73.22 vs BitNet b1.58 3B 74.34 (Winogrande 64.56 vs 66.37, PIQA 76.93 vs 78.40, SciQ 90.75 vs 91.20, LAMBADA 66.09 vs 67.63, ARC-e 67.78 vs 68.12).
- Claimed equivalences (p.5): 13B b1.58 is more efficient than 3B FP16; 30B b1.58 than 7B FP16; 70B b1.58 than 13B FP16 (latency, memory, energy).
- **Not shown**: quality of b1.58 models above 3.9B. The 7B/13B/70B models were measured for cost only.

### 7.7 Calibration / cost
- No calibration. Training from scratch: 100B tokens (comparison models), 2T tokens (3B model). GPU-hours: not stated.

### 7.8 Limitations
- Pre-training from scratch is necessary. The paper gives no conversion path for existing models.
- Quality parity is shown only up to 3.9B (100B tokens) and 3B (2T tokens).
- Speed numbers use a generic 2-bit GPU kernel. No ternary-specific kernel and no CPU results.
- Short paper (5 pages + references). No ablations. No training hyperparameters (LR, schedule).

### 7.9 Relevance to our goal
- Highest potential in this set: a natively ternary 70B needs ~14-18 GB (fits in 32 GB RAM with room for KV cache). A 700B ternary model needs ~140-175 GB of weights, which still needs SSD streaming or MoE sparsity. Matmul without multiplications suits CPUs.
- Dependency: the model must be **trained** this way. These methods cannot produce it from Llama- or DeepSeek-class checkpoints. Watch for released ternary checkpoints and for QAT/distillation methods that convert existing models (see Blind spots).
- Composes with: GGUF TQ1_0/TQ2_0 storage, mmap streaming, MoE, KV-cache quantization.

---

## 8. Cross-paper comparison

| Method | Weight bits | What is quantized | Calibration / training | Accuracy at ~4-bit | Accuracy at ~2-bit | CPU kernels |
|---|---|---|---|---|---|---|
| RTN (baseline in GPTQ/AWQ) | 3-8 | W | none | OK with g128 (Llama-2-70B INT4-g128 3.46 vs 3.32, AWQ Tab. 4) | collapses (OPT-6.7B INT2-g64 PPL 7622, AWQ Tab. 9) | yes (llama.cpp Q4_0 etc.) |
| GPTQ | 2, 3, 4, ternary; per-row or g1024..g8 | W only (A16) | 128 x 2048 tok C4; 4.2 GPU-h for OPT-175B on 1 A100 | OPT-175B 8.37 vs 8.34; Llama-2-70B INT4-g128 3.42 vs 3.32 (AWQ Tab. 4) | OPT-175B g128 9.58, g32 8.94 vs 8.34; OPT-6.7B INT2-g64 16.65 vs 10.86 | paper: GPU only (custom 3-bit CUDA kernel) |
| AWQ | 3, 4 (2 with GPTQ); g128 | W only (A16) | 16-128 x 2048 tok Pile; no backprop; cost not stated | Llama-2-70B 3.41 vs 3.32; INT3 3.74 | AWQ+GPTQ INT2-g64 OPT-6.7B 15.71 | TinyChat ARM NEON / x86 (Raspberry Pi 4B 0.7 tok/s at 7B) |
| SmoothQuant | 8 | W8 + A8 (all linears + attention BMMs) | 512 sentences Pile; one pass; alpha grid | not evaluated (8-bit only). W8A8 Llama-2-70B 3.359 vs 3.320 | not evaluated | paper: GPU (CUTLASS) only |
| BitNet (b1) | 1 (+FP scale) | W1 + A8, trained from scratch | full pre-training (~10B tok) | n/a (trained) | 1-bit 6.7B PPL 17.07 vs FP16 15.19 | not stated |
| BitNet b1.58 | 1.58 (stored as 2) | ternary W + A8 per token, trained from scratch | full pre-training (100B-2T tok) | n/a | 3B: PPL 9.91 vs 10.04 FP16 (parity at 3B and up) | not measured; [ext] llama.cpp TQ1_0/TQ2_0, bitnet.cpp |
| llama.cpp k-quants | 2.625-6.5625 tensor bpw [ext]; mixes 2.97-5.70 file bpw | W only; activations quantized to Q8 internally for dot products [ext] | none; optional imatrix (one forward pass over text) | PPL not stated in our files | PPL not stated | yes: AVX2/AVX-512/NEON [ext]; README speeds (hardware not stated) |
| llama.cpp i-quants | 1.5625-4.5 tensor bpw [ext]; files 2.00-4.68 bpw | W only | imatrix required for IQ1/IQ2 [ext] | not stated | not stated | yes (README: IQ2_XXS 79.86 tg t/s on 8B) |

Composition summary:
- AWQ scaling + GPTQ error compensation: shown to add up (AWQ Tab. 9).
- SmoothQuant smoothing and AWQ scaling are the same family of transforms (per-channel `diag(s)` folded into the previous op). SmoothQuant tunes it for activations; AWQ tunes it for weights.
- llama.cpp imatrix = diagonal of the GPTQ Hessian = AWQ's activation magnitude signal. llama.cpp uses it only to choose scales, not for error feedback [ext].
- All weight-only methods compose with mmap/SSD streaming (GGUF), KV-cache quantization and MoE expert offload.
- BitNet and b1.58 do not compose with PTQ. They replace it and need new training.

---

## 9. Blind spots: works these documents cite or depend on that we do not have

Priority is for our goal (70B-700B on 8-32 GB RAM, 0-12 GB VRAM). arXiv IDs from the papers' reference lists where given; others from my knowledge (verify).

| # | Title | arXiv | Cited by / why it matters | Priority |
|---|---|---|---|---|
| 1 | QuIP: 2-Bit Quantization of Large Language Models With Guarantees (Chee et al.) | 2307.13304 | BitNet Tab. 3 baseline (2-bit PPL 70.43). Incoherence processing (random orthogonal rotations) is the base of good 2-bit PTQ. | High |
| 2 | QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks (Tseng et al.) | 2402.04396 | Cited by b1.58 [TCS+24]. E8 lattice codebook, the same idea as the llama.cpp IQ2 grids. | High |
| 3 | The case for 4-bit precision: k-bit Inference Scaling Laws (Dettmers & Zettlemoyer) | 2212.09720 | Cited by AWQ. Best bits-vs-parameters trade-off for a fixed memory budget. Decides "bigger model at fewer bits vs smaller model at more bits". | High |
| 4 | FlexGen: High-Throughput Generative Inference of LLMs with a Single GPU (Sheng et al.) | 2303.06865 | Cited by AWQ. Offloads weights, KV and activations across GPU, CPU and disk with 4-bit compression. This is our problem. | High |
| 5 | Fast Inference from Transformers via Speculative Decoding (Leviathan et al.) | 2211.17192 | Cited by BitNet. Can hide SSD-streaming latency: verify several draft tokens per weight pass. GGUF defines an `mtp-` sidecar for this. | High |
| 6 | llama.cpp PRs #1684 (k-quants), #4773/#4856/#4897 (2-bit i-quants), #4861 (imatrix), #5196 (3-bit i-quants), #5320/#5334/#5361 (tuning) | GitHub | Listed in the README. The only source for k/i-quant design rationale and PPL/KLD numbers. Our files have no quality numbers for any GGUF type. | High |
| 7 | `ggml.h` / `ggml-common.h` (llama.cpp source) | GitHub | The GGUF spec defers block sizes to it. Needed to verify the [ext] block table (Section 2.3) and the Q1_0, Q2_0, NVFP4 layouts. | High |
| 8 | LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale (Dettmers et al.) | 2208.07339 | Cited by all four papers. Origin of the emergent-outlier finding and of the absmax activation quantizer used by BitNet. | Medium |
| 9 | Optimal Brain Compression / OBQ (Frantar & Alistarh, NeurIPS 2022) | 2208.11580 | Direct base of GPTQ (Eq. 2-3). | Medium |
| 10 | nuQmm / LUT-GEMM (Park et al.) | 2206.09557 | Cited by AWQ and SmoothQuant. Lookup-table GEMM for low-bit and non-uniform weights. Relevant to CPU kernels for 1-3 bit. | Medium |
| 11 | Ladder: Efficient Tensor Compilation on Customized Data Format (Wang et al., OSDI 2023) | not on arXiv (OSDI) | The b1.58 speed numbers use its 2-bit kernel. | Medium |
| 12 | FlashAttention / PagedAttention (vLLM) | 2205.14135 / 2309.06180 | Cited by BitNet. KV-cache memory management for long context in small RAM. | Medium |
| 13 | Optimal Brain Surgeon (Hassibi & Stork 1993) / Optimal Brain Damage (LeCun et al. 1990) | not on arXiv | Origin of the second-order formulas in OBQ/GPTQ. | Low |
| 14 | AdaRound (Nagel et al.) | 2004.10568 | Cited by GPTQ/AWQ. Layer-wise reconstruction baseline. | Low |
| 15 | BRECQ (Li et al.) | 2102.05426 | Cited by GPTQ/AWQ. Block reconstruction. | Low |
| 16 | ZeroQuant (Yao et al.) | 2206.01861 | Cited by GPTQ/SmoothQuant/AWQ. Per-token activation + group-wise weight quantization. | Low |
| 17 | Outlier Suppression / Outlier Suppression+ (Wei et al.) | 2209.13325 / 2304.09145 | Cited by SmoothQuant/AWQ. Shift-and-scale equivalent transforms. | Low |
| 18 | Understanding and Overcoming the Challenges of Efficient Transformer Quantization (Bondarenko et al., EMNLP 2021) | 2109.12948 | SmoothQuant Tab. 1 (per-channel activation quantization). | Low |
| 19 | Megatron-LM / GPipe | 1909.08053 / 1811.06965 | Parallelism used in the experiments. | Low |

Not cited by these papers, but needed for 405B-700B on small machines (from my knowledge; verify before use):
- Released ternary models and CPU kernels: BitNet b1.58 2B4T technical report (2504.12285); "1-bit AI Infra" / bitnet.cpp (2410.16144). High.
- KV-cache quantization: KIVI (2402.02750), KVQuant (2401.18079). High.
- Codebook 2-bit PTQ that beats GPTQ/AWQ at 2 bits: AQLM (2401.06118), QTIP (2406.11235). High.
- Inference from flash/SSD: "LLM in a flash" (2312.11514), PowerInfer (2312.12456). High.
- MoE expert offloading: "Fast Inference of Mixture-of-Experts Language Models with Offloading" (2312.17238). High (most 700B-class models are MoE).
- Rotation-based W4A4/KV4: QuaRot (2404.00456), SpinQuant (2405.16406). Medium.
- OmniQuant (2308.13137), SqueezeLLM (2306.07629, dense-and-sparse outliers). Medium.

---

## 10. Bottom line for the project

1. **Weight quantization alone gives about 3-10x vs FP16**: 4.9 bpw = 3.3x; 2.4 bpw = 6.7x; 1.6-2.1 bpw ternary = 8-10x. For a 70B model in 32 GB RAM this is enough at ~3 bpw (IQ3_XXS ~28 GB, Q2_K-class ~26 GB). For 405B-700B it is not enough on its own.
2. **GGUF + mmap is the enabler** for models larger than RAM. Aligned, pre-encoded tensors map directly, and the OS pages weights in from SSD. Token rate = SSD bandwidth / bytes read per token. Only MoE sparsity (few experts per token) makes this practical at 700B.
3. **Best PTQ recipe from these papers**: activation-aware importance (AWQ scaling or imatrix) + Hessian error feedback (GPTQ) + small groups (g128, or k-quant sub-blocks of 16/32). At 4 bits the loss on 70B is ~0.1 PPL. At 3 bits ~0.4 PPL. At 2 bits these methods still lose 1-5 PPL; QuIP#/AQLM-class methods are needed.
4. **SmoothQuant** matters for activation and KV quantization, not for weight memory.
5. **BitNet b1.58** is the only path in these files to ~1.6 bpw at FP16 quality, but only for models trained ternary from scratch.
