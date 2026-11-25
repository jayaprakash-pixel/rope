# F — Python-ecosystem libraries: AirLLM, MLX LM, bitsandbytes, AutoGPTQ

Analyst F. Scope: four Python libraries, read from source in `<repo>\repos\`.
Project goal: run 70B–700B+ parameter LLMs on machines with 8–32 GB RAM and 0–12 GB VRAM.

Method: I read each README/docs first, then the core source files. I did not build, run, install or download anything. All speed numbers in this document are **calculated**, not measured. Citations use the form `repo/path/file.py:line`, relative to `<repo>\repos\`. Where I could not find something, I write "not found".

Snapshot state (all repos are shallow clones with one commit each):

| Repo | Head commit date | Version in source |
|---|---|---|
| airllm | 2026-11-12 | 4.0.0 (`airllm/air_llm/setup.py:21`) |
| mlx-lm | 2026-11-05 | dynamic (setuptools-scm); needs `mlx>=0.32.3` (`mlx-lm/pyproject.toml:25`) |
| bitsandbytes | 2026-10-27 | 0.50.3.dev0 (`bitsandbytes/bitsandbytes/__init__.py:78`) |
| AutoGPTQ | 2025-04-11 | 0.8.0.dev0 (`AutoGPTQ/setup.py:19`) |

Reference model for all arithmetic below: Llama-3-70B class. hidden = 8192, intermediate = 28672, 80 decoder layers, 64 query heads, 8 KV heads, head_dim 128, vocab 128256, untied embeddings. Per decoder layer: attention 151 M params (q 67.1 M, k 8.4 M, v 8.4 M, o 67.1 M) + MLP 704.6 M = **855.6 M params = 1.71 GB in FP16**. Embedding = lm_head = 1.05 B params = 2.10 GB each in FP16. Total ≈ 70.6 B params ≈ 141 GB in FP16. I round to 70 B / 140 GB in the tables.

---

## 1. AirLLM (lyogavin/airllm)

### 1.1 What it is now

The README claims "70B large language models run on a single 4GB GPU card — without quantization, distillation, or pruning", plus Kimi K3 (2.8T) under 4 GB, Qwen3.8-Flash-Next (125B) on 6 GB and DeepSeek-V3 (671B) on ~12 GB (`airllm/README.md:9`). The code base was rewritten since the 2023 versions. The current core is one generic class, `AirLLMBaseModel` (`airllm/air_llm/airllm/airllm_base.py:58`). The docstring states the design (`airllm_base.py:59-71`): the real Hugging Face `transformers` model is built on the `meta` device (no memory), and AirLLM attaches forward hooks to each big module (embedding, every decoder layer, final norm, lm_head). A pre-hook streams that module's weights disk → GPU before it runs. A post-hook frees them after it runs. `transformers` owns the forward pass and the generation loop. Thus new architectures work when `transformers` supports them.

Most per-model subclasses are now thin. `AirLLMLlama2` is only a subclass with no new code (`airllm/air_llm/airllm/airllm.py:5-7`). `AutoModel` maps only nine special architectures to custom classes and sends all others to the generic class (`airllm/air_llm/airllm/auto_model.py:17-27, 57-64`).

### 1.2 How the model is split into per-layer files

Function `split_and_save_layers` (`airllm/air_llm/airllm/utils.py:544-807`):

1. It reads the checkpoint index (`model.safetensors.index.json` or `pytorch_model.bin.index.json`), or synthesizes an index for a single-file checkpoint (`utils.py:565-588`).
2. It builds the module list: `embed`, `layers.0 … layers.N-1`, `norm`, `lm_head`, plus optional "resident" modules (vision tower) and "cpu_resident" modules (huge n-gram tables) (`utils.py:595-607`). Modules without weights (tied lm_head) are dropped (`utils.py:612-614`). The list is sorted by the last original shard that each module touches, so the splitter only walks shards forward (`utils.py:623-628`).
3. For each module it loads every original shard up to the highest shard that the module needs into a CPU `state_dict` (`utils.py:735-760`), takes the tensors whose longest-prefix owner is this module (`utils.py:782-783`), optionally compresses them (`utils.py:785`), and writes one file per module: `<saving_path>/<module_name>.safetensors` plus a `.safetensors.done` marker (`airllm/air_llm/airllm/persist/safetensor_model_persister.py:30-36`). It then deletes those tensors from RAM (`utils.py:794-798`).
4. Pass-through optimization: if an original shard already holds exactly one module (Kimi K3 ships one ~17 GB shard per layer), it hard-links (fallback: symlink, then copy) the original file instead of copying it (`utils.py:649-672, 708-725`, `link_or_copy_file` at `utils.py:517-541`).
5. Download is lazy: only config/tokenizer files are downloaded first (`utils.py:851-853`). Each weight shard is downloaded when the splitter needs it (`utils.py:747-751`). `delete_original=True` deletes each original shard after it is consumed (`utils.py:737-741`), so peak disk use is about one model copy plus one shard.
6. The output directory is `splitted_model` or `splitted_model.4bit` / `.8bit` (`utils.py:550-552`). On macOS the persister writes `.mlx.npz` files instead (`airllm/air_llm/airllm/persist/model_persister.py:24-29`, `mlx_model_persister.py:89-100`).

RAM need during split: the accumulating `state_dict` holds at most the original shards that one module spans. For standard 5–10 GB HF shards this is about 5–20 GB of host RAM. That is acceptable for a 16–32 GB machine, and tight for 8 GB.

### 1.3 How each layer is loaded and freed at run time

- Hooks are installed for every streamed module (`airllm_base.py:771-775`). If `tie_word_embeddings` is true, the embedding stays resident on GPU and only decoder layers + final norm stream (`airllm_base.py:759-765`).
- `_pre_hook` (`airllm_base.py:889-904`): it takes the prefetched `state_dict` if there is one, else it loads the module synchronously; then `move_layer_to_device` places each tensor on the GPU with accelerate's `set_module_tensor_to_device`, casting high-precision tensors to the runtime dtype (`airllm_base.py:600-628`). Pre-quantized payloads (FP8, packed MXFP4, GPTQ `g_idx`, scales) are placed verbatim, not cast (`airllm_base.py:632-649`).
- `_post_hook` (`airllm_base.py:906-919`): it calls `module.to('meta')` (or, when quantizers/experts/CPU tables are present, it moves back to `meta` only the tensors it placed), then calls `clean_memory()`.
- `clean_memory()` = `gc.collect()` + `malloc_trim(0)` via libc (Linux only) + `torch.cuda.empty_cache()` (`utils.py:78-85`). This runs **after every module, for every token**: 83 times per generated token for a 70B model.
- Loading from disk uses `safetensors.torch.load_file(..., device="cpu")` (`persist/safetensor_model_persister.py:39-41`). That is a full read of the module file into CPU memory; then a host→device copy.
- Default runtime dtype is the model's own dtype (usually bf16), because fp16 overflows on deep models (`airllm_base.py:167-181`).

Peak VRAM is therefore: the largest single streamed module + activations + KV cache + CUDA context. For Llama-3-70B the largest modules are the embedding and the lm_head at 2.10 GB each in FP16, not a decoder layer (1.71 GB). This matches the README's "~4 GB" claim for 70B in full precision at short context. For Llama 3.1 405B (hidden 16384) the embedding/lm_head is ~4.2 GB each, which matches the README's "~8 GB" row (`README.md:297`).

### 1.4 Prefetching: threads, not CUDA streams

- One background thread: `ThreadPoolExecutor(max_workers=1)` (`airllm_base.py:191`).
- In the pre-hook of module *i*, after module *i* is on the GPU, it submits the disk load of module *i+1* to the thread (`airllm_base.py:900-904`). The thread does only the **disk → host RAM** part (`load_layer_to_cpu` / `load_layer_subset`).
- If CUDA is available and the layer is ≤ 2 GiB (`max_pinned_layer_bytes`, `airllm_base.py:75`), the loaded tensors are copied into pinned host memory (`airllm_base.py:443-456`). Note that `pin_memory()` makes a second host copy.
- **No CUDA streams are used.** I searched for `cuda.Stream`, `torch.cuda.stream` and `non_blocking`: not found in `air_llm/airllm/*.py`. The host→device copy runs in the hook on the default stream, so it is synchronous with compute. Only disk read overlaps compute.
- The prefetch does not wrap around token boundaries: `_next_streamed_idx` returns `None` after the last module (`airllm_base.py:878-880`), so the embedding (or first layer) of the next token is always loaded synchronously.
- **Prefetching is disabled when compression is on** (`airllm_base.py:188-190`).
- The README states prefetching gave "10% speed improvement" (`README.md:56`). This is consistent with the physics: in decode, compute per layer is a few milliseconds and the disk read is hundreds of milliseconds, so overlapping compute with I/O hides almost nothing.

### 1.5 Compression option (4-bit / 8-bit through bitsandbytes)

- Enable with `compression='4bit'` or `'8bit'` (`airllm_base.py:106-107`). bitsandbytes must be installed (`airllm_base.py:131-133`).
- 4-bit: `bnb.functional.quantize_nf4(v.cuda(), blocksize=64)` on **every** tensor of the layer, including norms (`utils.py:488-494`). No double quantization (`quantize_nf4` defaults `compress_statistics=False`, `bitsandbytes/bitsandbytes/functional.py:873-881`). Storage = 4 bits + one fp32 absmax per 64 values = **4.5 bits/param = 0.5625 bytes/param**. The magic number `0.2813` in `check_space` is exactly 4.5/16 (`utils.py:474`).
- 8-bit: `bnb.functional.quantize_blockwise(v.cuda(), blocksize=2048)` with the default 8-bit "dynamic" code map (`utils.py:495-503`). Storage = 8 bits + fp32 absmax per 2048 = **8.016 bits/param ≈ 1.002 bytes/param**.
- Decompression at load time happens **on the GPU**: the packed tensor is moved with `.cuda()` and dequantized with `dequantize_nf4` / `dequantize_blockwise` (`utils.py:104-132`). So PCIe moves the compressed bytes. After that, the layer is a normal FP16/BF16 layer: compute is at full precision. This is "storage/IO compression", not quantized compute. The README says this clearly: "the bottleneck is mainly at the disk loading, we only need to make the model loading size smaller" (`README.md:167-169`).
- Expected I/O speed-up vs FP16: 16/4.5 = 3.56× for 4-bit, 2.0× for 8-bit. The README claims "up to 3x" (`README.md:149`). That is plausible only for the I/O part, and it loses prefetching.

### 1.6 KV cache handling

AirLLM does not manage the KV cache itself. `generate()` and `forward()` delegate to the `transformers` model (`airllm_base.py:923-930`). With `use_cache=True` (README example, `README.md:126-130`) `transformers` builds its standard dynamic cache. AirLLM forces the model to report the running device (`_patch_device_property`, `airllm_base.py:375-403`) so the cache tensors are created on the GPU. Consequences:

- The KV cache for **all** layers stays on the GPU for the whole generation. It is not streamed, not offloaded, not quantized.
- For Llama-3-70B: KV per token = 2 (K,V) × 80 layers × 8 heads × 128 × 2 bytes = **327,680 bytes ≈ 0.31 MiB/token**. 4 k tokens ≈ 1.25 GiB; 32 k tokens ≈ 10 GiB. At long context the KV cache, not the weights, sets the VRAM floor.
- `max_seq_len` is accepted and stored (`airllm_base.py:84, 207`) but is not used anywhere else in the base class (grep: only the assignment). It does not limit the cache.
- The MLX (macOS) path keeps a Python list of per-layer `(K, V)` arrays (`airllm/air_llm/airllm/airllm_llama_mlx.py:299-300, 393-410`), also all in memory.

### 1.7 MoE per-expert streaming and CPU-resident tables (new, useful)

- Per-expert streaming (`airllm_base.py:779-876`): for MoE layers, a forward hook is put on each expert module. Only the experts that the router actually calls are read, using `safe_open(...).get_tensor(key)` to seek to single tensors inside the layer file (`load_layer_subset`, `utils.py:431-441`). The docstring says a Kimi K3 layer is ~55 GB expanded, while one token touches ~1 GB (`airllm_base.py:782-786`). This is the right idea for 671B-class MoE models. Restriction: it is disabled when compression is on (`airllm_base.py:800-808`), and it reads experts synchronously (no prefetch, because the router result is not known before the layer runs).
- CPU-resident mmap embedding: Qwen3.8-Flash-Next's ~102 GB n-gram table is written once to a raw file (`persist_ngram_mmap`, `utils.py:239-309`) and mapped with `torch.UntypedStorage.from_file(..., shared=True)` (`utils.py:312-322`). Lookups gather rows from the file through the OS page cache (`MmapEmbedding`, `utils.py:325-343`). This is a clean pattern for any table that is too large for RAM but read sparsely (embeddings).

### 1.8 Theoretical tokens/s for 70B

Model: batch 1 decode reads every weight byte once per token (dense model). Time per token ≈ bytes / bandwidth (I/O bound; compute is ~140 GFLOP per token, a few tens of ms on a 12 GB consumer GPU). This is an **upper bound**: it ignores safetensors parsing, Python hooks, `gc.collect()` per module, the extra pinned copy, and the non-overlapped host→device copy.

Bytes per token:
- FP16: 70 B × 2 = **140 GB**
- AirLLM 8-bit (blockwise, bs 2048): 70 B × 1.002 = **70.1 GB**
- AirLLM 4-bit (NF4, bs 64, no double quant): 70 B × 0.5625 = **39.4 GB**
- Reference "best 4-bit" (NF4 + double quant, or GPTQ g128): 70 B × ~0.516 = **36.1 GB**

| Format | Bytes/token | 3 GB/s NVMe (Gen3) | 7 GB/s NVMe (Gen4) | 25 GB/s PCIe 4.0 x16 (weights in pinned RAM) |
|---|---|---|---|---|
| FP16 | 140 GB | 46.7 s/tok = **0.021 tok/s** | 20.0 s = **0.050 tok/s** | 5.6 s = **0.18 tok/s** |
| 8-bit | 70.1 GB | 23.4 s = **0.043 tok/s** | 10.0 s = **0.10 tok/s** | 2.8 s = **0.36 tok/s** |
| 4-bit (AirLLM NF4) | 39.4 GB | 13.1 s = **0.076 tok/s** | 5.6 s = **0.18 tok/s** | 1.58 s = **0.63 tok/s** |
| 4-bit (best, ~4.13 bpw) | 36.1 GB | 12.0 s = **0.083 tok/s** | 5.2 s = **0.19 tok/s** | 1.44 s = **0.69 tok/s** |

Important conditions:
- The PCIe column is only reachable when the full model is already in host RAM. With 8–32 GB RAM this is impossible for a 70B model in any format above (smallest is 36 GB). The OS page cache also does not help: a sequential scan of a file set larger than the cache gives a near-zero hit rate under LRU-like eviction. On our target machines the **disk column is the real ceiling**.
- A partial fix is explicit residency: keep a fixed subset of layers pinned in RAM and stream only the rest. Example: 4-bit 70B, 20 GB pinned in RAM, 19.4 GB from a 7 GB/s disk → 2.8 s + 0.8 s ≈ 3.6 s/token ≈ 0.28 tok/s. AirLLM does not do this.
- The AirLLM 4-bit path disables prefetch, so in practice disk time and GPU dequant time add instead of overlap.
- A 0.5–0.2 tok/s best case means a 300-token answer takes 10–25 minutes. Layer streaming is not interactive for dense 70B.

### 1.9 Why layer streaming is slow for decode but fine for batched prefill

One full pass over the weights costs I/O time `T_io = W / BW` regardless of how many tokens are in the pass. Compute for that pass costs `T_c = 2 × P × N / FLOPS` for N tokens. With perfect overlap, time per pass = max(T_io, T_c).

- **Decode (N = 1 per sequence):** T_io = 20 s (FP16, 7 GB/s), T_c ≈ 140 GFLOP / 25 TFLOPS ≈ 6 ms. The GPU is idle >99.9 % of the time. Each new token needs a new full pass because token t+1 depends on token t.
- **Prefill (N = prompt length):** all prompt tokens go through one pass. Break-even N* where compute equals I/O: N* = FLOPS × bytes_per_param / (2 × BW). For FP16 at 7 GB/s with ~25 TFLOPS effective: N* ≈ 25e12 × 2 / (2 × 7e9) ≈ **3,600 tokens**. For 4-bit storage (0.5625 B/param) N* ≈ 1,000 tokens. So a 2,000-token prompt costs about the same wall time as **one** decoded token: ~20 s for FP16, i.e. ~100 prompt tok/s versus 0.05 decode tok/s.
- **Batched decode** (B independent sequences) also amortizes: aggregate tok/s ≈ B × the table values, until compute or KV memory limits it.
- AirLLM's origin confirms this: the README credits SimJeg's Kaggle LLM Science Exam notebook (`README.md:384-388`). That task is multiple-choice scoring, which is prefill-only. Layer streaming fits offline scoring, classification, embedding, reranking and batch evaluation. It does not fit chat.
- Ways to get more tokens per weight pass in decode: speculative decoding (a small resident draft model proposes k tokens; one streamed pass of the large model verifies k+1 positions), and batching many user requests. AirLLM implements neither.

### 1.10 Broken, outdated, or not as claimed

1. `check_space` for 4-bit divides by 0.2813 instead of multiplying: `int(total / 0.2813)` (`utils.py:473-474`). It demands ~3.56× the original size in free space; the correct need is ~0.28×. Splits with `compression='4bit'` can fail with `NotEnoughSpaceException` on disks that have enough space. (Bug.)
2. `remove_real_and_linked_file` reads `targetpath` that is only assigned inside an `if` (`utils.py:507-513`). For a regular (non-symlink) file this raises `UnboundLocalError`. HF cache files are symlinks, so the common path works; local non-cache checkpoints with `delete_original=True` break. (Bug.)
3. Compression is CUDA-only: quantize uses `v.cuda()` (`utils.py:491, 498`), dequantize uses `device="cuda"` and `.cuda()` (`utils.py:111-113, 124-128`). The README says CPU inference is supported (`README.md:44`) and `default_device()` picks MPS or CPU (`utils.py:88-101`), but compression fails without an NVIDIA (or ROCm) GPU.
4. 8-bit dequant hard-codes `dtype=torch.float16` (`utils.py:128`) even for bf16 models. The value is later cast to the runtime dtype, but it passes through fp16 range first.
5. 4-bit compression quantizes every tensor, including RMSNorm weights and biases (`utils.py:489-494`). Normally these are kept in high precision.
6. README "Configurations" says prefetching is supported only by `AirLLMLlama2` (`README.md:179`). Outdated: it is in the base class for all models now.
7. README says the training feature lets "Qwen3.8-Flash-Next (125B) trains under 6GB"; its Python API example uses an undefined name `model.device` (`README.md:373`) where it means `trainer`. (Doc error.)
8. Top-level `airllm/requirements.txt` pins `bitsandbytes==0.39.0`, `peft@v0.3.0`, `accelerate@v0.20.3`. These are for the old Anima training project and conflict with `setup.py` (`torch>=2.4`, `transformers>=4.49,<6`, `airllm/air_llm/setup.py:30-38`). Outdated file.
9. macOS path: `generate(..., max_new_tokens=None)` compares `len(tokens) >= None` (`airllm_llama_mlx.py:270-277`), which raises `TypeError` in Python 3. The MLX persister converts all weights to fp16 with `v.to(torch.float16)` (`persist/mlx_model_persister.py:91`), which can overflow bf16 weights. `load_model` swallows all exceptions and returns `None` (`mlx_model_persister.py:111-112`). The MLX path has no prefetch, no compression decode (the shards hold `.4bit.` tensors that `map_torch_to_mlx` does not understand), and only Llama/Mistral (`auto_model.py:31`).
10. The embedding table (2.1 GB FP16 for Llama 3) is streamed in full every token when embeddings are not tied, although one row is needed. The mmap gather that the code already uses for n-gram tables (§1.7) would remove this cost.
11. "Without quantization" is true for the default path. For Kimi K3 and DeepSeek-V3 the low VRAM numbers depend on checkpoints that are already FP8/MXFP4 and are placed verbatim (`airllm_base.py:634-649`). The README publishes **no tokens/s numbers** for any model (grep for `tok/s`, `tokens/s`, `s/token`: not found). VRAM claims are plausible from the design; speed is not claimed and is very low by §1.8.

### 1.11 Maturity and maintenance

Active (head commit 2026-11-12), single main author, broad model coverage, a manual GPU test harness (`airllm/air_llm/tests/test_streaming_gpu.py:1-19`) and some unit tests, no CI evidence in the snapshot. Code quality is mixed: the new hook-based core is clean; the split/compression utilities still carry 2023 code with real bugs.

**Best idea to reuse:** stream weights per module through forward hooks on a `meta`-device model, with per-expert seek-reads for MoE (`load_layer_subset`), and treat every weight file as a separately addressable unit on disk.

---

## 2. MLX LM (ml-explore/mlx-lm)

### 2.1 What it is

"A Python package for generating text and fine-tuning large language models on Apple silicon with MLX" (`mlx-lm/README.md:3-4`). Over 150 architectures in `mlx_lm/models/` (including DeepSeek-V3/V3.2, Kimi K2.5/K3, Qwen3-Next, gpt-oss). The numeric kernels live in the separate `mlx` core library, which is **not** in this snapshot. So the exact kernel math below is taken from how mlx-lm calls MLX, not from the MLX source.

### 2.2 Quantization format

- Modes and defaults (`mlx-lm/mlx_lm/utils.py:1034-1042`):

| mode | default group size | default bits | scale form |
|---|---|---|---|
| `affine` | 64 | 4 | per-group scale + bias, in model dtype |
| `mxfp4` | 32 | 4 | shared 8-bit power-of-two scale (OCP MX) |
| `nvfp4` | 16 | 4 | 8-bit scale per 16 + optional per-tensor `global_scale` fp32 (`convert.py:139-142`) |
| `mxfp8` | 32 | 8 | shared 8-bit scale |

- Affine dequantization is `w = q × scale + bias`. Evidence: the AWQ/GPTQ importer comment "MLX dequant = weight * scale + bias … So: bias = -zero * scale" (`mlx_lm/utils.py:169-172`), and MLX-LM's own GPTQ code `q = clip(round((w - biases)/scales), 0, 2^bits-1)` (`mlx_lm/quant/gptq.py:26-37, 88-93`).
- Packing: quantized weights are `uint32` words along the input dimension, `32/bits` values per word, low bits first (`gptq.py:28, 34-37`; `utils.py:150-156`). Group size is inferred at load time as `in_dims // scales.shape[-1]` (`utils.py:383-406`).
- Allowed affine bits: 2, 3, 4, 5, 6, 8 are used in recipes (`convert.py:24-36`, dynamic quant defaults 4/5); MLX-LM's GPTQ path supports only {2, 4, 8} (`gptq.py:27`).
- Bits/weight: affine 4-bit g64 with fp16 scale and bias = 4 + 32/64 = **4.5 bpw** (0.5625 B/param). g32 = 5.0 bpw. mxfp4 = 4.25 bpw. nvfp4 = 4.5 bpw. `compute_bits_per_weight` reports the real figure after conversion (`utils.py:244-249, 1087-1088`).
- Per-layer overrides: the config stores a dict `quantization[path] = {group_size, bits, mode}` for each layer that differs (`utils.py:1057-1075`), and the loader uses it (`utils.py:497-516`). Layers whose input size is not a multiple of the group size stay unquantized (`utils.py:1060-1061`).
- Activation quantization exists only for `nvfp4` and `mxfp8` (`QQLinear`, `utils.py:576-596`).
- Import of foreign formats: AWQ/GPTQ (4-bit only; `g_idx` act-order rejected, `utils.py:113-126`), compressed-tensors (`nvfp4-pack-quantized`, `mxfp4-pack-quantized`, `pack-quantized`, `utils.py:309-327`), NVIDIA ModelOpt NVFP4 (`utils.py:554-568`), FP8 with ue8m0 32×32 blocks as mxfp8 (`utils.py:538-548`), BitNet (`utils.py:524-527`).
- Possible issue: the GPTQ import reuses the AWQ unpacker. It assumes `qweight` shape `[in, out/8]` and AWQ's interleaved nibble order `[0,4,1,5,2,6,3,7]` (`utils.py:98-106, 137-146`). AutoGPTQ checkpoints store `qweight` as `[in/8, out]` with sequential nibble order and zeros stored minus one (`AutoGPTQ/auto_gptq/nn_modules/qlinear/qlinear_cuda_old.py:50-53, 166`). From reading only, the `quant_method == "gptq"` branch (`utils.py:569-574`) looks incorrect for AutoGPTQ-format files. Not verified by running.

### 2.3 Mixed-precision recipes

`mixed_2_6`, `mixed_3_4`, `mixed_3_6`, `mixed_4_6` (`mlx-lm/mlx_lm/convert.py:20-80`). Rule (modelled on llama.cpp Q4_K_M, link in `convert.py:52-55`): `v_proj` and `down_proj` get the high bit-width in the first 1/8 of layers, the last 1/8 of layers, and every third layer in between; `lm_head` always gets high bits; all else gets low bits (`convert.py:61-75`). Affine mode only (`convert.py:122-123`).

### 2.4 Learned quantization: DWQ, AWQ, GPTQ, dynamic

Overview: `mlx-lm/mlx_lm/LEARNED_QUANTS.md:1-20`.

- **DWQ (Distilled Weight Quantization)** (`mlx_lm/quant/dwq.py`): freezes everything, then unfreezes only `scales` and `biases` of affine layers with bits < 8 (`dwq.py:89-99`), and trains them with KL divergence against the teacher's logits at temperature 2.0 (`dwq.py:79, 105-113`), Adam, lr 1e-6 default (`dwq.py:278, 407`). Targets can be precomputed and saved (`--target-dir`, `--targets-only`, `dwq.py:292-296`). Docs: works best for 2–4 bits; smaller group size doubles tunable params (`LEARNED_QUANTS.md:50-63`). It needs the teacher logits and full forward/backward through the student model: memory ≈ the quantized model + activations.
- **AWQ** (`mlx_lm/quant/awq.py:400-500`): block-by-block. It captures each block's linear inputs, searches per-channel scales on a grid (`n_grid` 20 default, `awq.py:197-246`), then searches weight clipping (`awq.py:308-376`), and reverts a block if the loss got worse (`awq.py:488-494`). Default 3-bit g64, embeddings 4-bit g32 (`awq.py:404-408`). Supports `mx.distributed` data splits (`awq.py:186-195, 465-467`).
- **GPTQ** (`mlx_lm/quant/gptq.py:52-161`): standard GPTQ: Hessian `H = Σ xᵀx` per linear via a `Catcher` wrapper (`gptq.py:40-49`), 1 % mean-diagonal dampening, Cholesky → inverse → upper Cholesky on CPU stream (`gptq.py:78-86`), column-by-column error feedback inside each group and a block update after each group (`gptq.py:109-132`). **No act-order, no static groups.** Weak point for us: it wraps **all** linear layers at once and runs the whole model over the calibration set to build all Hessians before quantizing (`gptq.py:61-76`). Hessian memory for a 70B model ≈ 80 × (3×268 MB + 268 MB + 2×268 MB + 3.29 GB) ≈ 400 GB fp32 if materialized. It is not layer-sequential. Fine for small models, not usable for 70B on a small machine.
- **Dynamic quantization** (`mlx_lm/quant/dynamic_quant.py`): one gradient pass of a low-bit model; per-layer sensitivity = ⟨gradient, (low-bit weight − high-bit weight)⟩ / params (`dynamic_quant.py:87-93`); then binary search of a threshold to meet `--target-bpw` (`dynamic_quant.py:108-145`). Default 4/5 bits, BPW range ~[4.5, 5.5] (`LEARNED_QUANTS.md:85-99`).

### 2.5 KV cache: quantization, rotating cache, others

All in `mlx-lm/mlx_lm/models/cache.py`.

- `KVCache` (`cache.py:339-420`): pre-allocates in steps of 256 tokens and writes in place; trimmable.
- `QuantizedKVCache` (`cache.py:251-336`): stores K and V as `(packed uint32, scales, biases)` using `mx.quantize` with default group 64 and 8 bits (`cache.py:254, 296-297`). Attention reads it directly through `mx.quantized_matmul` for both QKᵀ and PV, without dequantizing the whole cache (`mlx_lm/models/base.py:64-108`). GQA is handled by reshaping queries (`base.py:78-81`).
- Turn-on rule: `--kv-bits`, `--kv-group-size` (default 64), `--quantized-kv-start` (default **5000** tokens, `mlx_lm/generate.py:60, 211-227`). After the cache passes that length, the full-precision cache is converted (`maybe_quantize_kv_cache`, `generate.py:316-321`; `KVCache.to_quantized`, `cache.py:396-404`). 4-bit g64 KV = 4.5 bits vs 16 → 3.56× smaller; 8-bit = 8.5 bits → 1.88× smaller.
- `RotatingKVCache` (`cache.py:423-554`): fixed `max_size`, with `keep` initial tokens never evicted (attention-sink style; `keep=4` when made from `--max-kv-size`, `cache.py:35-38`). Single-token decode writes in place in a ring (`_update_in_place`, `cache.py:482-520`); multi-token prefill concatenates in temporal order and trims (`_update_concat`, `cache.py:462-480`). Quantizing a rotating cache is not implemented (`cache.py:553-554`). Sliding-window layers in mixed models use it automatically (`mlx_lm/models/llama.py:266-270`).
- Others: `ChunkedKVCache` (`cache.py:727-800`), `ArraysCache` for SSM/linear-attention state (`cache.py:596-724`), `BatchKVCache` for batched generation (`cache.py:880+`).

### 2.6 Prompt cache

`save_prompt_cache` writes the list of per-layer cache states, the cache class names and metadata into one `.safetensors` file (`cache.py:43-79`); `load_prompt_cache` rebuilds each class with `from_state` (`cache.py:82-112`). `mlx_lm.cache_prompt` is the CLI (`mlx-lm/README.md:216-236`). Caches can be trimmed to reuse a common prefix (`cache.py:115-138`). The model name is stored, so the user does not need to pass it again (`README.md:234-236`).

### 2.7 Prefill and generation loop

Prompts are processed in chunks of `prefill_step_size` (default 2048) to cap peak memory (`generate.py:448-474`; `README.md:210-214`). Decode uses `mx.async_eval` to build the graph of token t+1 while token t is computing (`generate.py:476-491`). A speculative decoding path with a draft model exists (`speculative_generate_step`, `generate.py:496+`; `--draft-model`, `generate.py:230-238`).

### 2.8 Distributed inference

- `sharded_load` (`mlx_lm/utils.py:709-816`) supports **tensor parallel** (`model.shard(group)`) and **pipeline parallel** (`model.model.pipeline(group)`).
- Pipeline: `PipelineMixin.pipeline` splits layers evenly (low ranks get the extra layers); rank r runs the layers after those of ranks r+1…size-1; layers of other ranks become `None` (`mlx_lm/models/pipeline.py:16-38`). Each rank **downloads only the safetensors files for its own layers** plus embeddings/norm/head (`utils.py:759-792`). Forward: `recv_like` from the previous rank, run local layers, `send` to the next rank, `all_gather` at the end (example: `mlx_lm/models/deepseek_v3.py:340-362`).
- Tensor parallel for Llama: q/k/v/gate/up are "all-to-sharded", o/down are "sharded-to-all"; head counts are divided by group size (`mlx_lm/models/llama.py:231-260`).
- Launch: `mlx.launch --backend jaccl --hostfile ...` (`mlx-lm/examples/sharded_generate.py:3-14`). Default example model is `Llama-3.3-70B-Instruct-4bit` (`sharded_generate.py:31`). The server can also load sharded (`mlx_lm/server.py:442`).
- Pipeline parallelism across machines is the MLX-LM answer to "model bigger than one machine": two 32 GB Macs hold a 4-bit 70B (~40 GB). It does not stream from disk.

### 2.9 Lazy loading and wired memory

- `load_model(..., lazy=True)` builds the module tree and loads weights with `mx.load` but does not call `mx.eval` on parameters, so arrays are materialized only when used (`utils.py:409-424, 601-602`). `convert`, `gptq`, `sharded_load` use `lazy=True` (`convert.py:117`, `gptq.py:211-216`, `utils.py:735-737`). Whether `mx.load` maps the file or copies it is inside MLX core: not found in this repo.
- Wired memory: `maybe_set_recommended_wired_limit` sets `mx.set_wired_limit` to `max_recommended_working_set_size` (`utils.py:74-83`). The `wired_limit` context manager does this for each generation and warns when the model is > 90 % of that size (`generate.py:248-283`). README: raise the limit with `sudo sysctl iogpu.wired_limit_mb=N` on macOS 15+ (`README.md:267-290`).
- There is no design for models larger than memory. A model larger than RAM will page through the OS and be very slow. MLX-LM is a "fits in memory" runtime.

### 2.10 What we can reuse on non-Apple hardware

- **MLX itself is no longer Apple-only:** `pyproject.toml` has `cuda12`, `cuda13` and `cpu` extras and a Linux classifier (`mlx-lm/pyproject.toml:19-20, 45-47`). Windows is not listed. So on a Linux box with an NVIDIA card, MLX-LM may run directly (not tested here).
- Format ideas (portable): affine group quantization with scale **and** bias in fp16 is the same idea as llama.cpp `Q4_1`/`Q5_1`. Per-layer mixed bits stored as a per-path dict in the config is simple and portable.
- Algorithms (portable, pure Python logic): the mixed-bit predicate (`convert.py:48-75`), dynamic-quant sensitivity metric (`dynamic_quant.py:87-93`), DWQ (tune only scales and biases by KL distillation), AWQ block loop.
- KV cache designs: quantized KV with attention computed directly on quantized K/V; delayed quantization (only after N tokens); rotating cache with sink tokens; prompt cache serialized to safetensors.
- Pipeline split with per-rank partial download.

### 2.11 Maturity and maintenance

Very mature and very active (commit 2026-11-05, Apple-backed, large model zoo, test suite in `mlx-lm/tests/`). Platform-limited (Apple first; Linux CUDA/CPU through MLX extras).

**Best idea to reuse:** delayed, group-wise quantized KV cache with attention computed directly on the quantized K/V (`QuantizedKVCache` + `quantized_scaled_dot_product_attention`), together with DWQ (distill only the scales/biases).

---

## 3. bitsandbytes (bitsandbytes-foundation/bitsandbytes)

### 3.1 What it is

"k-bit quantization for PyTorch": 8-bit optimizers, LLM.int8() inference, and 4-bit QLoRA (`bitsandbytes/README.md:11-17`). Version 0.50.3.dev0. Integrated in Transformers, Diffusers and PEFT (`README.md:194-198`).

### 3.2 NF4 and FP4 data types (exact code values)

NF4 (`bitsandbytes/bitsandbytes/functional.py:788-805`), 16 values, normalized to [-1, 1]:

```
-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
-0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0
```

It is a lookup table, not a float format. It is generated by `create_normal_map(offset=0.9677083, use_extra_value=True)`: normal-distribution quantiles, 8 negative and 9 non-negative levels including an exact zero (`functional.py:169-224`).

FP4 (`functional.py:806-823`): 1 sign, 2 exponent, 1 mantissa bit, exponent bias 2. Raw values in bit-pattern order:
`[0, 0.0625, 8, 12, 4, 6, 2, 3, -0, -0.0625, -8, -12, -4, -6, -2, -3]`. They are divided by the max magnitude (12) at `functional.py:855`, so the normalized positive set is {0, 0.005208, 0.6667, 1.0, 0.3333, 0.5, 0.1667, 0.25}.

Also present: `int4` (symmetric ±7) and `af4` (AbnormalFloat, blocksize 64 only) (`functional.py:824-849`).

### 3.3 Block size and double quantization

- 4-bit `quantize_4bit`: default blocksize **64**; valid 32, 64, 128, 256, 512, 1024, 2048, 4096 (`functional.py:917-921`). One fp32 `absmax` per block. Two 4-bit codes per byte; storage dtype configurable (`quant_storage`, `functional.py:884-934`).
- Double quantization (`compress_statistics=True`): subtract the mean of the absmax values (`offset`), then quantize the absmax vector with 8-bit blockwise quantization, **blocksize 256**, dynamic code (`functional.py:938-951`). The 4-bit GEMM kernel accepts the 8-bit absmax directly (`absmax_8bit`, `absmax_code`, `absmax_offset`), so dequantization of the scales is fused (`bitsandbytes/bitsandbytes/autograd/_functions.py:471-483`). Only nested blocksize 256 is supported on that fast path (`_functions.py:484-485`).
- Module defaults: `Params4bit` blocksize 64, `compress_statistics=True` (`bitsandbytes/bitsandbytes/nn/modules.py:219-235`). **`Linear4bit` defaults to `quant_type="fp4"`** (`modules.py:543-544`); NF4 must be chosen explicitly or through `LinearNF4` (`modules.py:676`).
- 8-bit blockwise (`quantize_blockwise`): default blocksize 4096; dynamic 8-bit code from `create_dynamic_map` (`functional.py:613-686`).

### 3.4 LLM.int8() and outlier decomposition

- Weights: `Int8Params._quantize` does row-wise (per output channel) absmax quantization to int8: `CB = round(W × 127 / rowmax)`, `SCB = rowmax` (fp32) (`modules.py:737-749`; reference implementation `bitsandbytes/bitsandbytes/backends/default/ops.py:142-180`).
- Activations: per-row (per token) int8 quantization in each forward. With `threshold > 0`, any activation with `|x| ≥ threshold` marks its column as an outlier column; outliers are zeroed before quantizing (`default/ops.py:149-161`) and those columns are excluded from the int8 product (`default/ops.py:169-171`).
- Mixed product: int8 GEMM for normal columns (int32 accumulate, then dequant with row and column stats), plus an fp16 `addmm` of the outlier activation columns times the matching **dequantized int8** weight columns (`int8_mixed_scaled_mm`, `default/ops.py:63-100`). Note: since weights are stored only in int8, the "fp16 part" uses int8-reconstructed weights; only the activations stay in fp16 (TODO at `default/ops.py:86`).
- **Threshold 6.0:** the value 6.0 is not found in this repo. `Linear8bitLt` defaults to `threshold=0.0`, i.e. no decomposition (`modules.py:1052-1080`). The 6.0 default comes from the caller (Hugging Face `transformers` `BitsAndBytesConfig.llm_int8_threshold`, not in this snapshot) or the LLM.int8() paper. To get decomposition with bitsandbytes alone, pass `threshold=6.0`.
- Inputs are cast to fp16 for int8 quantization, with a warning for bf16 (`_functions.py:127-129`).

### 3.5 Backends present now

Python backend registry via `register_kernel(op, device)` (`bitsandbytes/bitsandbytes/_ops.py`), loaded in `bitsandbytes/bitsandbytes/__init__.py:18-50`, plus third-party backends through the `bitsandbytes.backends` entry point (`__init__.py:52-70`).

| Backend | Folder | Notes |
|---|---|---|
| CUDA (NVIDIA) and ROCm (AMD, same backend through HIP) | `backends/cuda/ops.py` | native lib `libbitsandbytes_{cuda,rocm}<ver>` chosen at import (`bitsandbytes/bitsandbytes/cextension.py:28-77`). New custom 4-bit GEMM kernels (SIMT, sm75, sm80 MMA) in `csrc/gemm_4bit*.cu`, with a per-architecture heuristic that picks custom kernel vs. dequant + `F.linear` by M, N, K (`backends/cuda/ops.py:583-640, 934-960`). hipBLASLt int8 path for ROCm (head commit message). |
| CPU | `backends/cpu/ops.py` | int8 matmul, blockwise quant/dequant, 4-bit dequant, `gemv_4bit` with an AVX512-BF16 packed layout (`functional.py:1676-1730`; `cpu/ops.py:17-26, 139, 257`), optimizers. README: x86-64 AVX2 minimum, aarch64 (`README.md:59-66, 105-112`). |
| XPU (Intel GPU) | `backends/xpu/ops.py` + `backends/triton/` | SYCL kernels in `csrc/xpu_*.cpp`; Triton kernels used by XPU (`xpu/ops.py:215, 293`). |
| MPS (Apple Metal) | `backends/mps/ops.py` | 4-bit quant/dequant/gemv/gemm; Hub kernels only on macOS 26+ (`mps/ops.py:3-8, 28-33`). |
| HPU (Intel Gaudi) | `backends/hpu/ops.py` | partial 4-bit (`README.md:96-103`). |
| default | `backends/default/ops.py` | pure PyTorch reference for all ops. |

Hardware floor: NVIDIA compute capability 6.0+ (Pascal) (`bitsandbytes/docs/source/installation.mdx:37-44`). Windows 11 x86-64 CPU and CUDA are supported (`README.md:131-160`).

### 3.6 Paged memory (CUDA unified memory)

- `get_paged(*shape)` allocates with `cudaMallocManaged(&ptr, bytes, cudaMemAttachHost)` (`bitsandbytes/csrc/pythonInterface.cpp:558-563`) and wraps the pointer as a CPU-visible torch tensor marked `is_paged` (`functional.py:91-99`).
- `prefetch_tensor` calls `cudaMemPrefetchAsync` to the device (or to host with device −1). It first checks `cudaDevAttrConcurrentManagedAccess` and **does nothing** if the device lacks it (`pythonInterface.cpp:566-585`). From general CUDA knowledge (not from this repo): this attribute is 0 on Windows (WDDM), so explicit prefetch is a no-op there and the driver migrates pages on fault.
- Use in the library: **only optimizer state.** Paged optimizers allocate state buffers > 1e5 elements as paged (`bitsandbytes/bitsandbytes/optim/optimizer.py:374-390`). On CPU, paging is disabled with a warning (`optimizer.py:375-382`). There is no paged path for model weights or KV cache. A global `GlobalPageManager` keeps the list (`functional.py:25-46`); the `prefetch_all` call in the optimizer step is commented out (`optimizer.py:323`).
- ROCm maps to `hipMallocManaged` / `hipMemPrefetchAsync` (`pythonInterface.cpp:22-27`); XPU uses SYCL USM shared memory (`pythonInterface.cpp:329-341, 699-700`).

### 3.7 Bytes per parameter for each mode

For an out×in = N×K weight. fp32 = 4 B for scales.

| Mode | Formula | Bytes/param | bits/param | 70B weights |
|---|---|---|---|---|
| FP16/BF16 | 2 | 2.000 | 16 | 140 GB |
| LLM.int8() (`Linear8bitLt`, `has_fp16_weights=False`) | 1 + 4/K (one fp32 row stat per output row) | ≈ 1.0005 (K=8192) | ≈ 8.0 | ≈ 70 GB |
| 8-bit blockwise (`quantize_blockwise`, bs 4096) | 1 + 4/4096 | 1.001 | 8.008 | 70.1 GB |
| 8-bit blockwise bs 2048 (AirLLM) | 1 + 4/2048 | 1.002 | 8.016 | 70.1 GB |
| NF4/FP4, bs 64, no double quant | 0.5 + 4/64 | 0.5625 | 4.5 | 39.4 GB |
| NF4/FP4, bs 64, double quant (module default) | 0.5 + 1/64 + 4/(64×256) | 0.5159 | 4.127 | 36.1 GB |
| NF4/FP4, bs 128, double quant | 0.5 + 1/128 + 4/(128×256) | 0.5079 | 4.063 | 35.6 GB |
| 8-bit optimizer state (Adam: 2 states) | 2 × (1 + 4/bs) | ≈ 2.0 per param of state | — | — |

The per-tensor `offset` scalar and the 256-entry code tables are negligible.

### 3.8 Fit for our goal

bitsandbytes stores weights compactly and has GPU, CPU, AMD, Intel and Apple paths. But it has **no offload or streaming**: a bnb-quantized 70B needs ~36 GB on one device. It is a component (kernels + formats), not a runtime for our hardware class. Its 4-bit GEMM for batch 1 dequantizes per block inside the kernel, so I/O per token equals the packed size.

### 3.9 Maturity and maintenance

Very mature, foundation-governed, actively maintained (commit 2026-10-27), wide backend support, Hugging Face sponsored (`README.md:200-203`). Some legacy APIs are deprecated (`CHANGELOG.md` 0.45.0 section).

**Best idea to reuse:** double quantization of the block scales (8-bit absmax over blocks of 256 with a mean offset), fused into the 4-bit GEMM. It cuts the scale overhead from 0.5 bit/param to 0.127 bit/param at blocksize 64, i.e. ~8 % fewer bytes to stream per token.

---

## 4. AutoGPTQ (AutoGPTQ/AutoGPTQ, archived)

### 4.1 Status and successor

README line 1: "AutoGPTQ is unmaintained - we suggest using GPTQModel for bug fixes and new models support" (`AutoGPTQ/README.md:1`). The code warns at import: "AutoGPTQ has stopped development. Please transition to GPTQModel … GPTQModel has been merged into Transformers/Optimum and full deprecation of AutoGPTQ within HF frameworks is planned" (`AutoGPTQ/auto_gptq/modeling/_base.py:78`, `AutoGPTQ/setup.py:11`). Note: the URL in those two warnings is misspelled `ModelCoud` (broken link); the README has the correct `ModelCloud/GPTQModel`. Last release news: 0.7.0 with Marlin, 2024-02-15 (`README.md:17`). Head commit 2025-04-11.

### 4.2 GPTQ algorithm implementation

File `AutoGPTQ/auto_gptq/quantization/gptq.py`.

- **Hessian accumulation** (`gptq.py:34-60`): per linear layer, `H` is a K×K fp32 matrix on the layer's device (`gptq.py:30`). For each calibration batch: `H *= n/(n+m)`, `n += m`, `x = sqrt(2/n) × xᵀ`, `H += x xᵀ`. So H = (2/n) Σ x xᵀ (running mean).
- **Dead columns:** diagonal zeros get `H[i,i]=1` and the weight column is zeroed (`gptq.py:84-86`).
- **Act-order / `desc_act`:** permute columns by descending `diag(H)` before quantizing; undo the permutation at the end (`gptq.py:104-108, 179-181`).
- **Dampening + Cholesky:** `H += percdamp × mean(diag(H)) × I` (default 0.01, `quantization/config.py:61`); `H = cholesky(H)`; `H = cholesky_inverse(H)`; `Hinv = cholesky(H, upper=True)` (`gptq.py:113-119`).
- **Lazy batch updates:** columns are processed in blocks of 128 (`blocksize=128`, `gptq.py:64, 121-129`). Inside a block: quantize column, error `e = (w − q)/Hinv[i,i]`, update remaining columns of the block (`gptq.py:131-156`). After the block: `W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]` (`gptq.py:161`).
- **Group size:** when `group_size != -1`, new scale/zero are computed from the current (error-updated) weights when the column index reaches a group boundary (`gptq.py:135-143`).
- **Static groups:** compute all group scale/zero from the original weights before the loop (`gptq.py:93-102`), then look up the group by original column index (with act-order, through `perm`) (`gptq.py:144-148`). This makes act-order compatible with contiguous groups, so `g_idx` is monotonic.
- **g_idx:** `g_idx[i] = i // group_size` normally; with static groups + act-order, `perm[i] // group_size`; then permuted back (`gptq.py:173-181`).
- **Quantizer:** asymmetric min/max per output row: `scale = (xmax − xmin)/maxq`, `zero = round(−xmin/scale)`; symmetric: `zero = (maxq+1)/2` (8 for 4-bit) (`AutoGPTQ/auto_gptq/quantization/quantizer.py:64-85`). Dequant is `scale × (q − zero)` (`quantizer.py:10-14`). Optional MSE grid search for clipping exists but is called with `mse=False` (`_base.py:302-307`).
- **Config defaults:** bits 4 (choices 2, 3, 4, 8), `group_size=-1`, `damp_percent=0.01`, `desc_act=True`, `static_groups=False`, `sym=True`, `true_sequential=True` (`quantization/config.py:59-65`).

### 4.3 Layer-by-layer driver

`BaseGPTQForCausalLM.quantize` (`AutoGPTQ/auto_gptq/modeling/_base.py:177-398`):

1. Run the embedding + first layer only far enough to capture the inputs of layer 0 for all calibration examples (`_base.py:205-282`). Inputs are cached on GPU by default (`cache_examples_on_gpu=True`, `_base.py:184, 220`).
2. For each decoder layer: move it to GPU if it was on CPU (`_base.py:291-294`); with `true_sequential`, quantize sub-groups in order, e.g. for Llama `[k,v,q]`, `[o]`, `[up,gate]`, `[down]` (`AutoGPTQ/auto_gptq/modeling/llama.py:21-26`); each sub-group re-runs the layer on all examples with forward hooks that feed `add_batch` (`_base.py:298-338`), then `fasterquant` (`_base.py:340-345`).
3. Re-run the now-quantized layer to produce inputs for the next layer (`_base.py:353-371`), swap input/output buffers, move the layer back to CPU (`_base.py:372-377`).
4. Pack all layers into `QuantLinear` modules (`_base.py:380-390`).

### 4.4 Packing format (qweight, qzeros, scales, g_idx)

From `AutoGPTQ/auto_gptq/nn_modules/qlinear/qlinear_cuda_old.py:50-75, 110-200` (same layout in the other GPTQ-format kernels):

| Tensor | Shape | Dtype | Content |
|---|---|---|---|
| `qweight` | [K / 32 × bits, N] | int32 | packed along the **input** dim; for 2/4/8 bits, `32/bits` values per word, value j at shift `bits × j` (sequential order) (`qlinear_cuda_old.py:134-139`); 3-bit uses a 32-values-in-3-words scheme (`qlinear_cuda_old.py:140-158`) |
| `qzeros` | [ceil(K/group), N / 32 × bits] | int32 | packed along the **output** dim; stored as **zero − 1** (`zeros -= 1`, `qlinear_cuda_old.py:166`), the known "GPTQ v1" off-by-one; kernels add 1 back (`qlinear_cuda.py:264, 285`) |
| `scales` | [ceil(K/group), N] | fp16 | per group per output column |
| `g_idx` | [K] | int32 | group index of each input column (non-monotonic with act-order) |

Weight reconstruction: `W[n, k] = scales[g_idx[k], n] × (q[k, n] − (qzeros[g_idx[k], n] + 1))`. Integers are formed with `round((W + scale×zero)/scale)` (`qlinear_cuda_old.py:119, 127`).

Storage for 4-bit, g128: 4 + 16/128 (scale) + 4/128 (zero) = **4.156 bits ≈ 0.52 B/param**, plus `g_idx` (4 B per input column per matrix, negligible). Group −1: ≈ 4.0 bits. 70B at 4-bit g128 ≈ 36.4 GB.

### 4.5 Kernels

Selection in `dynamically_import_QuantLinear` (`AutoGPTQ/auto_gptq/utils/import_utils.py`): HPU first; then QIGen (CPU) if requested; then Triton / Triton v2 if requested; else, for 4-bit: Marlin if `use_marlin`, else ExLlamaV2 if built, else ExLlama if enabled, else the old CUDA kernel (no act-order or no groups) or the CUDA kernel.

| Kernel | Location | Bits | Constraints / notes |
|---|---|---|---|
| `cuda_64` / `cuda_256` ("cuda", "cuda_old") | `autogptq_extension/cuda_64`, `cuda_256`; `qlinear_cuda*.py` | 2, 3, 4, 8 | generic, slow; `cuda_old` path when no act-order or no groups |
| ExLlama (v1) | `autogptq_extension/exllama/`; `qlinear_exllama.py` | 4 | supports act-order: `g_idx` passed to `make_q4`, which must be on CPU (segfault otherwise) (`qlinear_exllama.py:106-117`); needs a max-input-length scratch buffer |
| ExLlamaV2 | `autogptq_extension/exllamav2/`; `qlinear_exllamav2.py` | 4 | default for 4-bit when built; handles `g_idx` (`qlinear_exllamav2.py:71-92`) |
| Marlin | `autogptq_extension/marlin/`; `qlinear_marlin.py` | 4 | compute capability ≥ 8.0 (`AutoGPTQ/auto_gptq/utils/marlin_utils.py:98`), group 128 or −1 (`marlin_utils.py:107-108`), symmetric only (`marlin_utils.py:109-110`), no `desc_act` (`marlin_utils.py:111`); repacks to Marlin layout (`marlin_repack.cu`) |
| Triton / Triton v2 | `nn_modules/triton_utils/`; `qlinear_triton*.py` | 2, 4, 8 | Linux only, no 3-bit (`README.md:51`) |
| QIGen | `autogptq_extension/qigen/` | — | CPU code generator (`qigen/generate.py`) |
| HPU | `qlinear_hpu.py` | — | Intel Gaudi |

Hardware: NVIDIA Pascal or newer for builds (`setup.py:107-119`); Maxwell and lower not supported (`README.md:55`). Wheels built against PyTorch 2.2.1 (`README.md:44-49`) — old.

### 4.6 RAM and VRAM needed to quantize a 70B model layer by layer

What the code does with the full model: `from_pretrained` loads the whole FP16 model with `AutoModelForCausalLM.from_pretrained` (`_base.py:673-674`). Without `max_memory`, `device_map=None` and `low_cpu_mem_usage=False` (`_base.py:667-669`): the whole model lands in **CPU RAM**. With `max_memory`, accelerate spreads it over GPU and CPU; **disk offload raises `NotImplementedError("disk offload not support yet.")`** (`_base.py:645-646`). CUDA is mandatory (`_base.py:602-603`).

**System RAM:** ≈ 140 GB for the FP16 70B (plus ~1–2 GB overhead). With `low_cpu_mem_usage=False`, transformers can transiently need more. On an 8–32 GB machine, AutoGPTQ **cannot quantize a 70B model** without code changes.

**VRAM** (one Llama-3-70B layer at a time; estimate from shapes, T = calibration tokens = nsamples × seqlen):

| Item | Size |
|---|---|
| One decoder layer in FP16 | 1.71 GB |
| Cached layer inputs + outputs (fp16, if `cache_examples_on_gpu=True`) | 2 × T × 8192 × 2 B; T = 128×2048 → **8.6 GB**; T = 128×512 → 2.1 GB |
| Hessians, sub-group `[k,v,q]` (three separate K=8192 fp32 H) | 3 × 268 MB = 0.8 GB |
| Hessians, `[up,gate]` | 2 × 268 MB = 0.54 GB |
| Hessian, `down_proj` (K = 28672) | 28672² × 4 B = **3.29 GB** |
| `fasterquant` transients for `down_proj`: W fp32, Q, Losses (each 8192 × 28672 × 4 B = 0.94 GB) + Cholesky temporaries (up to ~2 × 3.29 GB while `H` is replaced) | ≈ 2.8 GB + ≈ 3.3–6.6 GB |

Peak ≈ 1.7 + 3.3 + 2.8 + 3.3…6.6 ≈ **11–14 GB** with inputs cached on CPU, or **+2–9 GB** more with inputs cached on GPU. So a 12 GB GPU is borderline and needs `cache_examples_on_gpu=False` and a modest T. A 16–24 GB GPU is comfortable. The calibration activations then need 2–9 GB of host RAM.

Practical answer for our class of machine: quantization of 70B needs a different driver that reads one layer at a time from disk (for example from AirLLM-style per-layer safetensors files), quantizes it on the GPU, and writes it back. Peak RAM then drops to roughly one layer + calibration activations (≈ 5–12 GB), and VRAM to the table above. Whether GPTQModel does this: not found in this snapshot.

### 4.7 Maturity and maintenance

Archived/unmaintained since 2024–2025, superseded by GPTQModel (per its own README). Kernels and the GPTQ format remain widely used (Transformers, vLLM, ExLlama). Do not depend on this package; reuse the algorithm and format knowledge.

**Best idea to reuse:** the GPTQ loop itself — lazy block updates (128 columns) with Cholesky-inverse error feedback, plus static groups so that act-order keeps a contiguous, kernel-friendly group layout (`gptq.py:93-161`). It is ~150 lines and is hardware-independent.

---

## 5. Maturity, maintenance and single best idea (summary)

| Repo | Maturity | Maintenance state | Single best idea to reuse |
|---|---|---|---|
| AirLLM | Medium: works for its claims on VRAM; known bugs in split/compression utilities; no speed claims | Active (2026-11), mostly one author | Hook-driven per-module streaming from per-layer files on disk, with per-expert seek-reads for MoE (`airllm_base.py:779-904`, `utils.py:431-441`) |
| MLX LM | High: large model zoo, tests, Apple-backed | Very active (2026-11) | Quantized KV cache with delayed start and attention computed on packed K/V; DWQ that tunes only scales/biases (`cache.py:251-336`, `base.py:64-108`, `dwq.py:89-113`) |
| bitsandbytes | High: foundation-governed, many backends | Active (2026-10) | Double quantization of block scales fused into the 4-bit GEMM (`functional.py:938-951`, `_functions.py:471-483`) |
| AutoGPTQ | High historically; now frozen | Unmaintained; successor GPTQModel | GPTQ with lazy block updates + static groups + act-order (`gptq.py:62-194`) and the packed `qweight/qzeros/scales/g_idx` format |

---

## 6. Comparison table

"Min VRAM for 70B" = smallest GPU memory with which the library can run a dense Llama-3-70B by its own design, at short context (≤ 2 k tokens). "Speed class" = decode speed for 70B on the target hardware class (8–32 GB RAM, ≤ 12 GB VRAM, NVMe).

| | AirLLM | MLX LM | bitsandbytes | AutoGPTQ |
|---|---|---|---|---|
| Approach | Stream one module at a time disk → GPU; optional NF4/8-bit storage compression, dequant to FP16 before compute | In-memory runtime on unified memory; group-affine / MX / NV FP quant; learned quant (DWQ, AWQ, GPTQ, dynamic); distributed pipeline/tensor parallel | Quantization kernels + formats (LLM.int8, NF4/FP4, 8-bit optimizers); no runtime of its own | Post-training GPTQ quantizer + int-N inference kernels (ExLlama v1/v2, Marlin, Triton, CUDA) |
| Bits/param | 16 (default); 4.5 (NF4 bs64); 8.016 (8-bit bs2048) | 4.5 (affine 4-bit g64); 2.5–6.5 typical; 4.25 mxfp4; 4.5 nvfp4; mixed recipes | 8.0 (int8); 4.5 (NF4/FP4 bs64); 4.127 (with double quant) | 4.156 (4-bit g128); ~2–8 by config |
| Needs GPU? | No for FP16 path (CPU/MPS possible); **yes (CUDA)** for compression | No discrete GPU; Apple GPU first; Linux CUDA/CPU via MLX extras (`pyproject.toml:45-47`) | No (CPU backend), but fastest on CUDA/ROCm/XPU | Yes for quantization (CUDA required, `_base.py:602-603`); inference CUDA/ROCm (CPU via QIGen limited) |
| Min VRAM for 70B | ~4 GB FP16 (largest module 2.1 GB + KV + context); KV grows 0.31 MiB/token on GPU | Whole model in unified memory: ~40 GB at 4-bit g64; ~22–28 GB with 2–3-bit mixed; or split over several Macs | ~36 GB on one device (NF4 + DQ); no offload in bnb itself | ~36.5 GB (4-bit g128) for inference; accelerate CPU offload possible but slow; quantization needs ~11–14 GB VRAM + ~140 GB RAM |
| Speed class (70B) | Very slow: 0.02–0.2 tok/s decode (disk-bound, §1.8); good for batch prefill | Fast when it fits (memory-bandwidth bound); unusable when model > RAM | Fast when it fits on GPU; does not apply when it does not fit | Fast when it fits (Marlin/ExLlamaV2 among the fastest 4-bit kernels); does not apply when it does not fit |
| Maintained? | Yes (active, single author) | Yes (very active) | Yes (active) | **No** (archived; use GPTQModel) |

---

## 7. Best-idea list for our project

1. **Per-module streaming from per-layer files, driven by forward hooks on a `meta` model** (AirLLM). This is the only design of the four that runs a 70B dense model in ≤ 4 GB VRAM. Fix its gaps: use CUDA streams to overlap H2D copy with compute, prefetch across token boundaries, keep a fixed RAM-resident subset of layers (deterministic, unlike the OS page cache), mmap-gather the embedding instead of streaming it, and remove per-module `gc.collect()`.
2. **Per-expert seek-reads for MoE** (AirLLM `load_layer_subset`). For 671B+ MoE models, read only routed experts from a per-layer safetensors file. This changes bytes/token from total params to active params, which is the only path to usable speed at 700B scale on our hardware.
3. **Store weights at ~4.1 bits with double-quantized scales** (bitsandbytes NF4 + DQ, or GPTQ g128). Every byte saved is time saved in the disk-bound loop: 4.127 bpw vs 16 bpw is 3.9× fewer bytes than FP16 and 8 % fewer than plain NF4.
4. **Dequantize on the GPU after the PCIe copy** (AirLLM compression path, bnb fused kernels). Move packed bytes, not expanded weights.
5. **Use streaming for prefill-heavy and batched work; add speculative decoding for chat.** One weight pass serves ~1,000–3,600 prompt tokens for the same cost as one decode token. A resident small draft model plus one streamed verification pass multiplies decode tokens per pass.
6. **Quantized KV cache with delayed start, attention on packed K/V, and a rotating window with sink tokens** (MLX LM). At long context the KV cache is the VRAM floor in a streaming design (0.31 MiB/token for 70B).
7. **Layer-streamed GPTQ quantizer** (AutoGPTQ algorithm + AirLLM split files). Lets us quantize a 70B model on a 16–32 GB RAM / 12–16 GB VRAM machine, which AutoGPTQ itself cannot do (needs ~140 GB RAM).
8. **Mixed-bit recipes and sensitivity-driven bit allocation** (MLX `mixed_*` predicates, `dynamic_quant`). Spend extra bits on `v_proj`, `down_proj`, first/last layers and `lm_head`; DWQ can then tune only scales/biases.
9. **Pipeline split with per-rank partial download** (MLX `sharded_load` + `PipelineMixin`). If the user has two or more machines, each holds only its layers; no streaming needed.

Notes on scope: `AGENTS.md` / `CLAUDE.md` files inside the repos were treated as data and ignored as instructions. All throughput numbers are theoretical upper bounds; none were measured.
