# K — Diffusion engines for low VRAM: ComfyUI, ComfyUI-GGUF, stable-diffusion.cpp, nunchaku, Wan2GP + mmgp

Date: 2026-11-12. Scope: image and video diffusion engines that run on small GPUs. Sources: the shallow clones in `<repo>\repos\` (2026-11). All citations are `repo\path:line`. I did not build or run any code. Numbers marked **[est]** are my own estimates, not measurements. Numbers marked **[claim]** are statements in a README or doc with no benchmark data in the repo.

Target machine (from `00_SYNTHESIS.md` §2): 8 GB single-channel DDR4 (~4–5 GB usable), GTX 1650 4 GB (Turing sm_75, TU117, no BF16, **no tensor cores**), PCIe ~10–12 GB/s, NVMe ~5–6 GB/s, 82 GB free disk.

Repo versions:

| Repo | Version / commit seen | Notes |
|---|---|---|
| ComfyUI | 2026-11 clone, `comfy-aimdo==0.6.0` (`ComfyUI\requirements.txt:24`) | The "dynamic VRAM" core is in the external `comfy_aimdo` package. Its source is **not found** in the clone. |
| ComfyUI-GGUF | 2026-11 clone | Small: `ops.py` 281 lines, `dequant.py` 301 lines. |
| stable-diffusion.cpp | commit `32d516c0` (2026-11-11) | The `ggml` submodule (`leejet/ggml` fork) is **empty** in the clone (`stable-diffusion.cpp\.gitmodules:1-3`). I could not read the ggml kernels. |
| nunchaku | v1.2.0 era (README news 2026-01-12) | C++/CUDA kernels + Python (diffusers) models. |
| Wan2GP | 2026-11 clone, vendors **mmgp 4.3.0** (`Wan2GP\mmgp\pyproject.toml:8`) | |
| mmgp (standalone) | **3.8.1** (`mmgp\pyproject.toml:3`) | Older than the copy inside Wan2GP. I analyze 4.3.0 and use 3.8.1 only for its older claims. |

---

## 0. The one law for diffusion (and why it differs from LLM decode)

The LLM synthesis (`00_SYNTHESIS.md` §1) says decode is memory-bound: `tok/s <= bandwidth / bytes per token`. That is true because one decode token uses every weight once (N = 1 token per weight pass).

Diffusion is the opposite case. One denoising step pushes **N = thousands of tokens** through every weight. So each weight byte serves thousands of multiply-adds.

```
FLOPs per step   ≈ 2 · P · N  +  4 · N² · d · L          (linear layers + attention)
arithmetic intensity of the weights  ≈ 2 · N / bytes_per_param   FLOP per weight byte
weight bandwidth needed  ≈ GPU FLOP rate / intensity
```

Worked numbers **[est]** (P = parameters, N = tokens per step, Q4 ≈ 0.56 byte/param):

| Workload | N (tokens) | Intensity (Q4) | Weight bandwidth needed at 3 TFLOP/s |
|---|---|---|---|
| Flux-dev 12B, 1024×1024 | ~4,608 | ~16,000 FLOP/B | ~0.18 GB/s |
| Flux-dev 12B, 512×512 | ~1,536 | ~5,500 FLOP/B | ~0.55 GB/s |
| Wan 2.1 14B, 480p, 33 frames | ~14,000 | ~50,000 FLOP/B | ~0.06 GB/s |
| LLM decode (for contrast) | 1 | ~3.6 FLOP/B | 800+ GB/s |

**Conclusion:** on this machine, a diffusion model can stream its weights from NVMe (5 GB/s) or RAM over PCIe (10 GB/s) at every step, and the GPU is still the bottleneck. The limits are (1) GPU compute, (2) activation memory (attention, VAE), and (3) RAM for the Python stack. Weight placement matters much less than for LLMs. Every repo below has discovered this; the best ones (ComfyUI dynamic VRAM, sd.cpp segmented execution, mmgp block swap) stream the whole model every step and still claim "no speed loss".

Exceptions where weight bandwidth does matter: very short steps (small images, 1–4 step distilled models), step 1 of a new model (cold read), and the text encoder (runs once, but is large).

---

## 1. ComfyUI (comfyanonymous/ComfyUI)

### 1.1 Architecture and one denoising step

ComfyUI is a node-graph application in PyTorch. The sampler (`comfy\samplers.py`, `comfy\k_diffusion\`) calls the model through a `ModelPatcher`. One step:

1. `load_models_gpu()` (`comfy\model_management.py:913`) makes sure the model (and its LoRA patches) is "loaded". It decides how much VRAM the weights may use.
2. The diffusion model forward runs. Every `Linear`/`Conv` is a `comfy.ops` class with `comfy_cast_weights`. On each call, `cast_bias_weight()` (`comfy\ops.py:385`) fetches the weight to the GPU (if it is not resident), casts it to the compute dtype, applies LoRA patches (`LowVramPatch`, `comfy\model_patcher.py:175`), and dequantizes quantized formats.
3. In newer model files, the forward loop over transformer blocks uses a **prefetch queue**: `make_prefetch_queue()` / `prefetch_queue_pop()` (`comfy\model_prefetch.py:337`, `:231`). When block *i* computes, the weights of block *i+1* are faulted in on an offload stream. Used, for example, in `comfy\ldm\flux3\model.py:142-149` and `comfy\ldm\fibo2\model.py:129-133`.
4. Attention runs through `optimized_attention` (`comfy\ldm\modules\attention.py:923-948`).
5. After sampling, the VAE decodes, with tiling if needed (`comfy\sd.py:1273` `decode_tiled_`, `:1297` `decode_tiled_3d`).

### 1.2 Memory strategy

ComfyUI has **two** memory systems.

**A. Legacy "estimate based" ModelPatcher (lowvram / novram / smart memory).**

- VRAM states: `DISABLED, NO_VRAM, LOW_VRAM, NORMAL_VRAM, HIGH_VRAM, SHARED` (`comfy\model_management.py:45-51`).
- Budget for weights: `lowvram_model_memory = max(0, free - minimum_required, min(free * MIN_WEIGHT_MEMORY_RATIO, free - minimum_inference_memory()))` (`comfy\model_management.py:999`). `MIN_WEIGHT_MEMORY_RATIO = 0.4` (`:434`). `minimum_inference_memory() = 0.8 GiB + EXTRA_RESERVED_VRAM` (`:864-865`). `EXTRA_RESERVED_VRAM` is 400 MB, or **600 MB on Windows** (`:851-853`).
- `NO_VRAM` sets the weight budget to 0.1 bytes, so every weight is cast on use (`:1005-1006`).
- `ModelPatcher.load()` (`comfy\model_patcher.py:1005`) walks modules in order. Modules that fit the budget move to VRAM. The others stay in RAM and are marked `comfy_cast_weights = True` (`:1029-1074`). The log line "loaded partially; X MB loaded, Y MB offloaded" comes from `:1116`.
- Granularity: **one module (one Linear/Conv)**. Not one block.
- Overlap: `get_offload_stream()` rotates through `NUM_STREAMS` CUDA streams (default **2** on NVIDIA/AMD, `comfy\model_management.py:1343-1352`, `:1478-1500`). Each cast writes into a reusable per-stream cast buffer (`get_cast_buffer`, `:1383`). The host queues the copy on the other stream while the GPU still computes the previous layer. There is no explicit lookahead in this path.
- "Smart memory": models stay in VRAM after use until something else needs the space; `--disable-smart-memory` turns it off (`comfy\cli_args.py:197`, `comfy\model_management.py:573`). `partially_unload()` (`comfy\model_patcher.py:1194`) evicts modules from a loaded model when a new model needs VRAM.
- Pinned memory: up to **40 % of RAM on Windows** (`comfy\model_management.py:1617-1618`); on Linux `max(0.4·RAM, min(0.9·RAM, RAM−4 GiB, RAM+swap−16 GiB))` (`:1621`). On our 8 GB machine that is **~3.2 GB** of pinned RAM at most.

**B. "Dynamic VRAM" (`ModelPatcherDynamic`, default on NVIDIA when `comfy_aimdo` is installed).**

- Enabled unless `--disable-dynamic-vram`, `--highvram`, `--gpu-only`, `--novram` or `--cpu` (`comfy\cli_args.py:334-337`). `--lowvram` "doesn't do anything if dynamic vram is enabled" (`comfy\cli_args.py:170`). If `comfy_aimdo` fails, ComfyUI falls back to the legacy patcher (`main.py:322`).
- Each model gets a **VBAR**: a virtual VRAM address range of `model_size × 10` (`comfy\model_patcher.py:1874-1878`, `:1897`). Every castable module gets a slot in it (`m._v = vbar.alloc(...)`, `:2087`). Physical pages are faulted in on use (`vbar_fault`), and a **signature** check tells whether the bytes are still resident (`comfy\ops.py:192-193`). The driver-level allocator (in `comfy_aimdo`, source not found) evicts pages under pressure; `vbar.prioritize()` marks the active model (`comfy\model_patcher.py:1966-1967`). A watermark check exists (`comfy\ops.py:135-140`).
- Units: models can declare `get_dynamic_vram__units` so that a whole transformer block is laid out consecutively in the VBAR ("VBAR is a bump allocator", `comfy\model_patcher.py:1975-1990`). Tiny modules (≤ 16 KB) are force-loaded to avoid "lopsided stream buffer rotations" (`:2067-2070`).
- Host side: six pinned `HostBuffer` subsets per model: `weights`, `patches`, `weights-loaded`, `patches-loaded`, `weights-fast`, `patches-fast` (`comfy\model_patcher.py:1955-1962`). Pin eviction runs in tiers (`comfy\model_management.py:667-700`), with 256 MB hysteresis (`:613`) and a Windows emergency rule (swap ≥ 5 % or < 512 MB available, `:618-619`, `:708-716`).
- **Disk path ("fast disk")**: `read_tensor_file_slice_into()` can read a tensor **straight from the model file to the GPU** (`comfy_aimdo.host_buffer.read_file_to_device`, `comfy\memory_management.py:77-82`), or file → pinned host buffer → GPU (`:84-94`). `--fast-disk` "Force disk-backed dynamic loading and offload over unpinned RAM. Can be faster for users with fast NVME disks" (`comfy\cli_args.py:182`). Auto-detection: on Linux, NVMe with ≥ 8 GT/s × 4 lanes counts as fast (`comfy\storage.py:40-54`); on Windows it asks `comfy_aimdo.storage.fast_disk()` (`comfy\storage.py:79-80`).
- Prefetch: one block ahead in models that use the prefetch queue (`comfy\model_prefetch.py:286-300`): the next block's modules are pinned/faulted on the offload stream while the current block computes. Each block can also be **captured as a CUDA graph** and replayed when its VBAR signature still matches ("Comfy model compiler", `comfy\model_prefetch.py:303-335`).

Peak VRAM formula (legacy path) **[derived]**: `weights_resident + max_module_cast_buffer × NUM_STREAMS + activations(N) + 0.8 GiB + 0.6 GiB (Windows)`.

### 1.3 Weight formats and quantization

- Native "Comfy quant" safetensors (`ComfyUI\QUANTIZATION.md:1-65`): `float8_e4m3fn`, `float8_e5m2`, `nvfp4`, `mxfp8`, `int8_tensorwise` (optional `convrot` Hadamard rotation), `convrot_w4a4` (INT4 W4A4), `asym_w4a8_int8`, `w6a8_int8`. Kernels come from the external `comfy_kitchen` package (`comfy\quant_ops.py:48-81`); its source is **not found** in the clone.
- `int8` matmul uses `torch._int_mm` in the eager Kitchen backend (`comfy\model_management.py:2101-2117`). FP8/NVFP4/MXFP8 compute needs newer GPUs (`supports_fp8_compute`, `:2033`; `supports_nvfp4_compute`, `:2060`).
- GGUF is not native; it comes from ComfyUI-GGUF (§2).

### 1.4 Speed tricks

- Attention backends: SageAttention 1/2 and 3 (Blackwell), FlashAttention, xformers, PyTorch SDPA, split, sub-quadratic, Comfy-Kitchen INT8 (`comfy\ldm\modules\attention.py:26-43`, `:923-948`).
- Feature caching: EasyCache node (`comfy_extras\nodes_easycache.py`). Step-distillation LoRAs load through the normal LoRA path.
- VAE tiling: 2-D and 3-D tiled decode (`comfy\sd.py:1273-1299`).
- `torch.compile` node (`comfy_extras\nodes_torch_compile.py`), per-block CUDA graphs (above), `--fast` options incl. `fp16_accumulation` and cuDNN autotune (`comfy\cli_args.py:323-327`).

### 1.5 Claimed minimum hardware

- README: "Can run even the biggest open source models on as low as **4GB vram + 8GB ram** relatively quickly (saturating your GPU compute) using our state of the art asynchronous weight streaming technology." (`ComfyUI\README.md:41`). **[claim]** — no benchmark in the repo. It names exactly our machine class.

### 1.6 GTX 1650 + 8 GB RAM

- **FP16 is disabled for the 16-series**: "FP16 is just broken on these cards" — the list includes `"1650"` (`comfy\model_management.py:1975-1979`). There is one exception: with manual cast, if `model_params × 4` bytes does not fit the VRAM, ComfyUI uses FP16 anyway (`:1967-1970`). So a big model (Flux, Wan 14B) computes in FP16; a small one (SD1.5) computes in FP32 at half speed. BF16 is not available on Turing.
- Pinned memory cap ~3.2 GB (40 % of 8 GB). The Python + PyTorch + CUDA context itself takes RAM, so the 8 GB box is near the floor.
- Fast-disk mode is the relevant mode here: weights go file → GPU and need little RAM.
- **Largest plausible**: image — Flux.1-dev 12B in FP8/GGUF-Q4 (6–12 GB on disk), disk-streamed. Video — Wan 2.1 1.3B comfortably; Wan 14B (Q4/FP8) only with disk streaming and a 4-step distill LoRA. Evidence: only the README claim; no test logs for this GPU.

### 1.7 Best idea to reuse

**Virtual-address residency (VBAR) with per-module "is it still resident?" signatures, plus a direct file → GPU read path.** The engine treats VRAM as a cache of the model file, not as a place where models are "loaded". This removes the need for exact VRAM estimates, which are unreliable on Windows (ComfyUI says so itself, `main.py:322`).

---

## 2. ComfyUI-GGUF (city96/ComfyUI-GGUF)

### 2.1 Architecture and one step

A ComfyUI custom node pack. `UnetLoaderGGUF.load_unet()` (`ComfyUI-GGUF\nodes.py:150-184`) reads the GGUF file into a state dict of `GGMLTensor` objects (raw quantized bytes plus `tensor_type` and logical `tensor_shape`, `ComfyUI-GGUF\ops.py:47-91`) and builds the normal ComfyUI model with `custom_operations = GGMLOps` (`ops.py:227`). In each step, every `Linear` calls `cast_bias_weight()` (`ops.py:194-210`):

1. move the **quantized** bytes to the GPU (`s.weight.to(device)`),
2. `dequantize_tensor()` to the compute dtype (`ops.py:177`, `dequant.py:15-28`),
3. apply LoRA patches to the dequantized weight (`ops.py:181-188`),
4. call `torch.nn.functional.linear` with the full-precision temporary (`ops.py:241-243`).

### 2.2 Memory strategy

- It uses ComfyUI's ModelPatcher partial loading. Quantized bytes are what moves over PCIe, so transfer cost is ~3.5× smaller than BF16.
- Loading: `torch.from_numpy(tensor.data)` on a numpy memmap (`loader.py:120-125`) — zero-copy mmap. After the first load, it "releases" the mmap by moving each offloaded module to GPU and back, to detach it from the file (`nodes.py:99-119`). The code comment says "TODO: possible to OOM" (`nodes.py:117`).
- VRAM estimate: the largest tensor is tagged (`loader.py:156`), and a fake fp16 "temp.weight" of its shape is added to the size estimate for the dequant buffer (`ops.py:150-155`).
- `GGUFModelPatcher` subclasses the legacy `ModelPatcher` and forces its class on clone (`nodes.py:35`, `:122-133`). **It appears to bypass ComfyUI's dynamic-VRAM patcher** [inference from code, not tested].
- Windows note in code: `torch.empty` "still reserving the full memory for the layer ... Windows doesn't over-commit memory so without this 24GB+ of pagefile is used" (`ops.py:233-236`).

### 2.3 Quant types and dequant kernels

- Supported fast dequant: `BF16, Q8_0, Q5_1, Q5_0, Q4_1, Q4_0, Q6_K, Q5_K, Q4_K, Q3_K, Q2_K, IQ4_NL, IQ4_XS` (`dequant.py:287-301`). F16/F32 pass through (`dequant.py:7`). Anything else falls back to numpy on the CPU: "this is incredibly slow" (`dequant.py:24-27`). No IQ1/IQ2/IQ3 types.
- The dequant "kernels" are **pure PyTorch tensor ops** (bit shifts, `view`, multiply), e.g. Q8_0 at `dequant.py:65-69`. Each call is several small elementwise launches and writes the whole FP16 weight to VRAM. There is **no fused dequant-GEMM**.
- Quantization tooling: convert to F16 GGUF, then quantize with a **patched llama.cpp** (`tools\README.md`, `tools\lcpp.patch`, pinned to tag `b3962`). Small tensors (< 1024 params) stay unquantized (`tools\convert.py:10`); per-model `keys_hiprec` stay FP32 (`tools\convert.py:58-59`, `:102-103`, `:115-116`). 5-D tensors of Wan/Hunyuan need `fix_5d_tensors.py` after quantization (`tools\README.md`).

### 2.4 Cost of on-the-fly dequant **[est]**

Per linear call: read Q4 bytes (0.56 B/param) + write FP16 (2 B/param) + matmul reads FP16 again (2 B/param) ≈ 4.6 B/param of VRAM traffic. Flux 12B per step ≈ 55 GB → ~0.45 s on the GTX 1650's ~128 GB/s. One Flux step at 1024² is ~30–40 s of compute on this GPU [est, §7]. So dequant is **~1–2 % overhead here**. On fast GPUs the ratio is worse because compute shrinks and the dequant does not.

### 2.5 Speed tricks

None of its own. It inherits ComfyUI's attention, caching, and compile; it allows full `torch.compile` with torch ≥ 2.8 (`ops.py:22-44`).

### 2.6 Claims

README: "transformer/DiT models such as flux seem less affected by quantization. This allows running it in much lower bits per weight variable bitrate quants on low-end GPUs" (`README.md:5`). No hardware numbers.

### 2.7 GTX 1650 + 8 GB RAM

Same limits as ComfyUI (§1.6). GGUF Q4_K Flux (~6.4 GB) or Q3_K (~4.9 GB) is the practical format. mmap release step may need RAM equal to the offloaded part.

### 2.8 Best idea to reuse

**"Move the compressed bytes; decompress only on the GPU, just before use."** It cuts PCIe/disk traffic by the compression ratio for free. Our engine should do the same but with fused ggml MMQ kernels instead of a full FP16 temporary.

---

## 3. stable-diffusion.cpp (leejet/stable-diffusion.cpp)

### 3.1 Architecture and one step

Plain C/C++ on ggml, "working in the same way as llama.cpp" (`README.md:42`). Each model is a `GGMLBlock` tree that builds a ggml compute graph (`src\model\diffusion\*.hpp`, `src\model\vae\*.hpp`, `src\model\te\*.hpp`). Runners (`src\core\ggml_runner.cpp`) allocate the compute buffer with `gallocr` and run the graph on a ggml backend (CPU, CUDA, Vulkan, Metal, OpenCL, SYCL, HIP, MUSA; `README.md:117-123`).

One denoising step:

1. The sampler (`src\runtime\denoiser.hpp`) calls the diffusion runner.
2. The runner builds the graph. Model code marks **graph cuts** at every block boundary, e.g. `mark_graph_cut(x, "wan.blocks." + i, "x")` (`src\model\diffusion\wan.hpp:820`) and `flux.double_blocks.i` (`src\model\diffusion\flux.hpp:1116-1117`). 40 model files use cut markers.
3. If weights + compute workspace fit, the graph runs in one piece. If not, it runs **segment by segment** (one block per segment), with weight residency managed per segment (§3.2).
4. For UNet models, conditional and unconditional CFG run as one batch of 2 by default (`docs\performance.md:42-50`).
5. Optional step caching (`--cache-mode`) skips blocks or steps (§3.4).
6. VAE decode, with tiling and automatic OOM retry (§3.4).

Model support (`README.md:44-108`): SD1.x/2.x/Turbo, SDXL, SD3/3.5, FLUX.1, FLUX.2-dev/klein, FLUX.3 Klein, Chroma, Qwen-Image (+2.1, Edit), Z-Image, HiDream-O1, Ideogram4, PixArt, Krea2, many more; video: **Wan 2.1/2.2** (incl. A14B MoE, TI2V-5B, VACE, S2V), MiniMax-H3, LTX-2.3/2.5, HunyuanVideo 1.5, LingBot-Video; upscalers RelayVSR, SeedVR2, ESRGAN. Text encoders include CLIP, T5 and **full LLMs** (Qwen3, Qwen3-VL, Qwen3.5 DeltaNet, GPT-OSS MoE MLP) reimplemented in sd.cpp's own code (`src\model\te\llm.hpp:42-1888`).

### 3.2 Memory strategy

This is the most complete design of the five repos. It is newer than most public descriptions of sd.cpp.

- **Placement** (`docs\performance.md:102-136`, `examples\common\common.cpp:556-640`):
  - `--backend` = where compute runs (per module: `clip=cpu,vae=cuda0,diffusion=vulkan0`).
  - `--params-backend` = where weights live: GPU, `cpu`, or **`disk`**. `disk` "reloads parameters from the model file on demand, retains unpinned compute copies while space permits, and releases them under pressure or at module-run completion" (`docs\performance.md:128`).
  - `--offload-to-cpu` = shortcut for `*=cpu` params (`docs\performance.md:116`).
  - `--auto-fit on` (default): places weights "on the compute GPU, RAM, another GPU, or disk in that order, according to available memory" (`examples\common\common.cpp:794-797`).
  - `--mmap`, `--eager-load` (lazy load is the default).
- **Pinned host memory**: when params live on the CPU and compute is on a GPU, params use `ggml_backend_dev_host_buffer_type(compute_dev)` (`src\model_manager.cpp:1141-1145`). For CUDA this is page-locked host memory [ggml behavior; ggml source not in clone]. So `--offload-to-cpu` pins the whole offloaded model. On an 8 GB machine that is a risk; `disk` params avoid it.
- **Segmented execution** (`docs\performance.md:138-157`): "When a graph has cut markers and its missing weights plus incremental compute workspace exceed the available device headroom, it runs its fixed segment list in order ... The current segment's weights are pinned during compute, and the next parameter-bearing segment is prefetched when the device supports asynchronous transfer. No opt-in streaming flag is required."
  - Plan: `ggml_graph_cut::Plan` with `Segment`s, each with its own measured compute buffer size (`src\core\ggml_graph_cut.h`, `measure_segment_compute_buffer`).
  - Pipeline: `SegmentWeightPipeline` (`src\core\segment_weight_pipeline.h:23-60`) with `segment_start`, `segment_end`, `enqueue_next`, `submit_next`.
  - **Eviction policy**: "evicts them from the last segment backward" (`docs\performance.md:140`); code: `preferred_eviction_order()` returns segments in reverse (`src\core\segment_weight_pipeline.cpp:59-60`). Rationale: the first blocks are needed soonest at the next step, so evict the last ones.
  - Prefetch transfer: staging tensors on the compute device, filled with `ggml_backend_tensor_set_async` from host buffers or `ggml_backend_tensor_copy_async` from another backend, then an event is recorded (`src\model_manager_prefetch.cpp:157-185`).
  - Residency manager interface: `DeviceResidencyManager` (`src\device_residency_manager.h:70-126`) with blocks tracked by `last_use_epoch` (`:61-68`), pin counts, capacity checks against a managed budget.
- **Budgets**: `--max-vram <GiB>` (positive = budget, `0` = live free VRAM, negative = reserve that much) (`docs\performance.md:150`). Planning headroom 128 MiB plus a 512 MiB device scratch reserve (`docs\performance.md:144-148`). Recommended "biggest model my card can host": `--offload-to-cpu --max-vram -1` (`docs\performance.md:183-193`).
- **Multi-GPU / RPC**: `--split-mode layer|row`, `--rpc-servers` (`examples\common\common.cpp:568-579`).

Peak VRAM formula **[derived from docs]**: segmented mode ≈ `max over segments( weights(seg) + weights(next seg, prefetch) + compute_buffer(seg) ) + cut buffers + 512 MiB reserve`. With one Flux block at Q4 ≈ 12e9/57 × 0.56 ≈ 120 MB, the weight part is tiny; the compute buffer dominates.

### 3.3 Weight formats and quantization

- Inputs: `.ckpt/.pth/.pt`, safetensors, GGUF (`README.md:124-128`); convert mode writes GGUF or safetensors (`README.md:129`).
- `--type` at load time: "f32, f16, q4_0, q4_1, q5_0, q5_1, q8_0, q2_K, q3_K, q4_K" (`examples\common\common.cpp:808-811`). Pre-quantized GGUF can use any ggml type the backend supports.
- Memory table for SD1.x 512² (`docs\quantization_and_gguf.md:12-17`): f32 ~2.8G, f16 ~2.3G, q8_0 ~2.1G, q4_0 ~2.0G; with flash attention f16 ~1.9G, q4_0 ~1.5G.
- Flux-dev memory by type (`docs\flux.md:36-39`): **q8_0 12068 MB, q4_0 6394 MB, q4_k 6395 MB, q3_k 4888 MB, q2_k 3736 MB**.
- Dequant: ggml's own kernels (MMQ/MMVQ and dequant-to-F16 + cuBLAS on CUDA). The `ggml` submodule is empty in the clone, so I cannot cite kernel lines.
- Extensions only in the **patched** ggml fork: INT8 tensorwise + ConvRot (reads ComfyUI `int8_tensorwise` files directly, `docs\int8_convrot.md:1-30`), FP8 weights, SageAttention. "CUDA GEMM requires Turing or newer" for the INT8 path (cuBLAS I8×I8→I32) (`docs\ggml_extensions.md:296-303`). With `SD_USE_UPSTREAM_GGML=ON`, FP8 converts to F16 at load, INT8/ConvRot files are rejected, and "Upstream GGML may lack some operators and performance optimizations" (`docs\build.md:59-71`).
- imatrix support (`docs\imatrix.md`, `src\runtime\imatrix.cpp`).

### 3.4 Speed tricks

- Flash attention `--flash-attn on|off|auto`, saves "flux 768x768 ~600mb, SD2 768x768 ~1400mb" (`docs\performance.md:1-14`).
- SageAttention native CUDA, **SM80+ only** (`docs\sage_attention.md:11-15`, `:25-27`). Not for Turing.
- Direct convolution `--conv-direct`; CUDA 2-D direct conv "requires a compiled Turing-or-newer MMA architecture" (`docs\backend.md:252`).
- Caching (`docs\caching.md:15-24`): `ucache` (UNet), `easycache`, `dbcache` (block-level L1 residual), `taylorseer`, `cache-dit` (DBCache + TaylorSeer), `spectrum`; plus a conditioning LRU cache (`docs\caching.md:5-13`).
- Step distillation: LCM/LCM-LoRA, Turbo models, lightx2v 4-step Wan LoRAs (`docs\wan.md:162`).
- VAE: `--vae-tiling` (256×256 px tiles, 50 % overlap, `docs\performance.md:52-79`); **automatic retry with smaller tiles on allocation failure, even without `--vae-tiling`** (`docs\performance.md:88-93`; `src\core\backend_fit.h:17-22`); temporal tiling for video VAEs (`docs\performance.md:95-100`); TAESD/TAE decoders. Wan doc warns "Wan models vae requires really much VRAM! ... please try tae instead" (`docs\wan.md:52`).
- Batched CFG (`docs\performance.md:42-50`). No torch.compile (not PyTorch); graph and allocation reuse instead.

### 3.5 Claims

- Flux: "You can run Flux using stable-diffusion.cpp with a GPU that has 6GB or even 4GB of VRAM, without needing to offload to RAM." (`docs\flux.md:3`) **[claim]**; the q2_k row (3736 MB) is the only supporting number.
- Z-Image / ERNIE-Image: "on GPUs with 4GB of VRAM — or even less." (`docs\z_image.md:3`, `docs\ernie_image.md:3`) **[claim]**.
- CUDA build: "Recommended to have at least 4 GB of VRAM." (`docs\build.md:125`).
- `--offload-to-cpu` saves VRAM "without reducing generation speed" (`docs\performance.md:102-104`) **[claim]**, consistent with §0.
- No speed tables for any GPU.

### 3.6 GTX 1650 + 8 GB RAM

- Native code: no Python, no PyTorch. RAM overhead is the lowest of all five. This matters most on 8 GB.
- ggml CUDA supports sm_75 (llama.cpp runs on GTX 16xx; see `D_llama_cpp.md`). Flash attention in ggml has non-tensor-core kernels [ggml behavior, source not in clone].
- `--params-backend diffusion=disk` lets the diffusion model stream from NVMe without a full RAM copy; at ~5 GB/s this is far above the bandwidth the GPU can consume (§0).
- **Largest plausible image model**: Flux.1-dev Q4_K/Q8 with disk params; Qwen-Image 20B Q4 (~11 GB) and FLUX.2-dev 32B Q4 (~18 GB) run in principle via disk streaming but take minutes per step [est, §7]. **Video**: Wan 2.1 1.3B and Wan 2.2 TI2V-5B are practical; Wan 14B / A14B Q4 with lightx2v 4-step LoRA is plausible but slow; the Wan VAE needs tiling or TAE.
- Not usable here: SageAttention (SM80+), CUTLASS INT8 dense kernels (Ampere+, `docs\ggml_extensions.md:305-307`). The cuBLAS INT8 path claims Turing support; whether it is fast on TU117 (no tensor cores) is **unverified**.

### 3.7 Best idea to reuse

**Graph cuts at block boundaries + a residency manager that runs a too-big graph segment by segment, prefetching the next segment and evicting from the end.** It is automatic (no flags), backend-neutral (it lives above ggml-backend), and it is the natural place to add disk streaming and LLM layer streaming in one mechanism.

---

## 4. nunchaku (nunchaku-tech/nunchaku)

### 4.1 Architecture and one step

nunchaku is an inference engine for **SVDQuant W4A4** models. C++/CUDA kernels (`src\kernels\zgemm\`, `src\kernels\awq\`) with PyTorch bindings; model classes plug into diffusers pipelines (`nunchaku\models\transformers\transformer_flux.py`, `transformer_qwenimage.py`, `transformer_zimage.py`, `transformer_sana.py`, `unets\unet_sdxl.py`). Models covered: FLUX.1 (dev, schnell, Kontext, Krea, tools), Qwen-Image (+Edit, Lightning), Z-Image-Turbo, SANA, SDXL; T5 text encoder in 4-bit (AWQ W4A16, `nunchaku\models\text_encoders\`). **No video model** (no Wan/Hunyuan code found).

One linear layer in a step (SVDQuant):
`Y = (X̂ · L1) · L2  +  Q4(X̂) · Q4(Ŵ − L1·L2)` — a 16-bit low-rank branch (rank 32) absorbs outliers, the residual runs in 4-bit weights × 4-bit activations (`README.md:67`). Engine fusion: "Down Projection and Quantize kernels use the same input, while Up Projection and 4-Bit Compute kernels share the same output ... we fuse the first two and the latter two kernels together"; without fusion the rank-32 branch costs "57% latency overhead" (`README.md:71`). Group size 64 for INT4, 16 for NVFP4 (`nunchaku\models\linear.py:43-44`, `:89-92`). LoRA scales: "every 16 ranks share a scale" (`src\Linear.h:98`).

### 4.2 Memory strategy

- `CPUOffloadManager` (`nunchaku\models\utils.py:52`): per-transformer-block offload with a **ping-pong pair of GPU buffer blocks** (`self.buffer_blocks = [copy.deepcopy(blocks[0]), copy.deepcopy(blocks[0])]`, `:115`), a dedicated memory stream, and events. `step()` (`:188-230`) records compute done, then on the memory stream waits for it and copies block *i+1* into buffer `(i+1) % 2`. `num_blocks_on_gpu` (default 1) blocks stay resident (`:69`, `:153-161`). Offloaded blocks are **pinned** (`p.data.pin_memory()`, `:157-161`).
- Granularity: one transformer block. Prefetch depth: 1. Eviction: none needed (fixed two buffers).
- Peak VRAM **[derived]**: `non-block modules + num_blocks_on_gpu × block + 2 × block + activations`.

### 4.3 Formats and kernels

- INT4 (W4A4, group 64) for sm_75/80/86/89; NVFP4 for sm_120 (`nunchaku\utils.py:189-230`, `:321-333`).
- Build targets: `assert sm in ["75", "80", "86", "89", "120a", "121a"]` (`setup.py:54`); default `["75", "80", "86", "89"]` (`setup.py:59`).
- **sm_75 path**: on `__CUDA_ARCH__ < 800` the kernels emulate the m16n8k16 MMA with two `mma.sync.m16n8k8` (FP16) or `m8n8k32` (INT4) instructions (`src\kernels\zgemm\mma.cuh:38-63`, `:104-140`). BF16 is rejected on sm_75 (`static_assert(!is_bf16)`, `mma.cuh:88`). A special `i2f_sm75` / "FasterI2F" int→half conversion exists because "both int2float and float2half are slow on sm_75 and before" (`src\kernels\zgemm\gemm_utils.cuh:382`, `gemm_w4a4.cuh:726-728`, `gemm_w4a4.cu:6-16`). Attention: `nunchaku-fp16` on Turing (`examples\flux.1-dev-turing.py`).
- These are **tensor-core instructions**. The docs list Turing as "`sm_75` (Turing: RTX 2080)" (`docs\source\installation\installation.rst:115`). The GTX 1650 (TU117) is sm_75 but has **no tensor cores**. Whether the IMMA INT4 `mma.sync` runs on TU117, and at what speed, is **not documented and not tested** in the repo. **[unverified — treat as "probably does not run well".]**
- Quantization is offline with DeepCompressor (separate repo); calibration on a big GPU.

### 4.4 Speed tricks

FP16 attention kernel, First-Block Cache and Double FB Cache (`README.md` news 2025-04-05, 2025-06-01), multi-LoRA, ControlNet, PuLID, IP-Adapter. Lightning (4/8-step) Qwen-Image checkpoints.

### 4.5 Claims (copied)

- "On 12B FLUX.1-dev, it achieves 3.6× memory reduction compared to the BF16 model. By eliminating CPU offloading, it offers 8.7× speedup over the 16-bit model when on a 16GB laptop 4090 GPU, 3× faster than the NF4 W4A16 baseline." (`README.md:55`)
- "our INT4 model runs 3.0× faster than the NF4 W4A16 baseline on both desktop and laptop NVIDIA RTX 4090 GPUs ... on the laptop 4090, it achieves a total 10.1× speedup by eliminating CPU offloading. Our NVFP4 model is also 3.1× faster than both BF16 and NF4 on the RTX 5090" (`README.md:75`). These come from the ICLR 2025 paper; the figures are images, not data in the repo.
- "per-layer CPU offloading, reducing FLUX's minimum memory requirement to just **4 GiB** while maintaining a **2–3× speedup**" (README news 2025-03-07). **[claim]**
- Qwen-Image "asynchronous offloading, reducing VRAM usage to as little as **3 GiB** with no performance loss" (README news 2025-09-04; `docs\source\usage\qwen-image.rst:27`). **[claim]**
- v1.2.0: "INT4 support for 20-series GPUs" (README news 2026-01-12).

### 4.6 GTX 1650 + 8 GB RAM

- VRAM: a 4-bit Flux transformer (~6.7 GB) with block offload fits the 3 GiB/4 GiB claims.
- RAM: the offloaded blocks are **pinned**. Flux INT4 offloaded ≈ 6 GB pinned + 4-bit T5 + Python/PyTorch ≈ more than the ~4.5 GB usable. **Probably does not fit 8 GB RAM** without changes [est].
- Kernels: depends on tensor-core MMA on TU117 — unverified.
- No video models.
- Verdict: **not a base for us.** Keep the method, not the engine.

### 4.7 Best idea to reuse

**Low-rank 16-bit branch + 4-bit residual, with the branch fused into the quantize and GEMM-epilogue kernels.** For diffusion (compute-bound, §0) activation quantization pays off, unlike LLM decode. This is the one repo where 4-bit *compute* (not only 4-bit storage) is the point.

---

## 5. Wan2GP (deepbeepmeep/Wan2GP) and mmgp

### 5.1 Architecture and one step

Wan2GP ("WanGP") is a Gradio application with many models: video (Wan 2.1/2.2, MiniMax H3, LTX-2, Hunyuan Video 1/1.5, LongCat, Kandinsky), image (Flux 1/2/3 Klein, Qwen Image, Z-Image, Krea, HiDream, Ideogram 4), audio and LLMs (`Wan2GP\README.md:15`; `Wan2GP\mmgp\README.md:53`). All models run through **mmgp**: the app builds models on the CPU, then calls `offload.profile(pipe, profile_no)` (`Wan2GP\mmgp\offload.py:7094`). mmgp then hooks every module's `forward`.

One denoising step with block swapping (profile 4/5):

1. mmgp detects the "towers" of repeated blocks (`_detect_main_towers`, `offload.py:203`) and treats each block as a transfer unit.
2. A hook before each block calls `_load_pipelined()` (`offload.py:4461-4502`): it waits for the in-flight copy of this block (event), then queues copies of the next `prefetchWindow` blocks on a transfer stream. The order of blocks is **learned during the first step**, including links from one tower to the next and from the last block of step *k* to the first of step *k+1* (`mmgp\README.md:15`, `:44`).
3. `_move_block()` (`offload.py:4529-4562`) copies the whole block into **one VRAM buffer** with one copy (from the staging ring) or one copy per pinned run, then rebuilds parameters as views: "an allocation and a copy per tensor would cost the host more than the GPU spends on a short block."
4. The block computes; LoRAs apply in the forward without merging (`offload.py:5697`).
5. VAE decode with tiling; optional TeaCache/MagCache/First Block Cache skip steps.

### 5.2 Memory strategy (mmgp 4.3.0)

- **Fixed VRAM slots**: `window + 1` slots per model, sized for the largest swapped tower block (`_take_slot`, `offload.py:4564-4591`). "freed slots are reused in the order of the transfer stream, which waits for the computation queued before". A bigger block outside the towers gets its own buffer.
- **Prefetch window**: default 2 for models that also preload blocks in VRAM, else 1 (`mmgp\README.md:146`); extra blocks must fit in `PREFETCH_VRAM_SHARE = 0.05` of VRAM (`offload.py:703`, `:4433-4444`).
- **VRAM preload**: the part of the budget not needed for slots keeps blocks permanently resident (`mmgp\README.md:16`). **autoPreload** (`Wan2GP\mmgp\auto_preload.py:1-17`): the first step of a new workload (resolution, frame count, stage) measures peak activation memory; the next steps keep as many blocks resident as the free VRAM allows, with a 512 MB + 5 % margin, and back off on shortage.
- **Reserved (pinned) RAM**: pinned with `cudaHostRegister` on normal RAM, page-exact, counted against a budget (`Wan2GP\mmgp\pinned.py:1-15`). "Pinned ... transferred up to 2 times faster" (`mmgp\README.md:17`). Default max share: 40 % of RAM on Windows, 60 % on Linux (`mmgp\README.md:135`).
- **Staging ring (smartPinning)**: for unpinned models, a background thread copies blocks ahead into a ring of pinned RAM ("three of the largest such blocks"), with 3 copy threads (`RING_COPY_THREADS = 3`, `RING_DEFAULT_BLOCKS = 3`, `offload.py:874-875`; `mmgp\README.md:136`). This gives near-pinned speed with ~1–2 GB of RAM.
- **Pinned pool (profile 0)**: reserved RAM shared by the models by priority; it only grows "while the system keeps this much RAM available": `POOL_RAM_FLOOR = max(4 << 30, physical_memory // 10)` (`offload.py:1285`). On an 8 GB machine the floor is 4 GB.
- **Low-RAM loading**: a rewritten safetensors loader, memory-mapped, read-only zero-copy (`writable_tensors=False`) (`mmgp\README.md:18`, `:206`). Windows read-ahead with `PrefetchVirtualMemory`-style large requests for cold mapped pages (`Wan2GP\mmgp\read_ahead.py:1-14`). Checkpoint pages that nothing reads any more are dropped from the working set (`offload.py:1286-1290`).
- **Zero-copy mapped RAM** for token tables (`Wan2GP\mmgp\mapped.py:1-7`).
- **VRAM allocator**: `mmgp.allocator.install()` replaces PyTorch's CUDA allocator with a VMM-based one: "several GB less peak VRAM with long videos and large images"; `spill=True` overflows to system RAM (`mmgp\README.md:236-248`) **[claim]**.
- **cpuCompute**: LLM decoder layers left in RAM run on the CPU when that is faster than streaming them, for calls of ≤ 8 rows, chosen by timing (`offload.py:876-886`). This is the llama.cpp "partial offload" idea inside a PyTorch app.
- **Virtual GPU** for testing: `MMGP_VIRTUAL_VRAM_GB=8` emulates a smaller card and RAM (`Wan2GP\mmgp\virtual_gpu.py:1-20`).
- Granularity: one tower block (one transformer block). Overlap: CUDA transfer stream + 1 prefetch thread + ring copy threads.

**Profile table (mmgp 4.3.0, `offload.py:7148-7188`, `mmgp\README.md:96-103`, `Wan2GP\docs\CLI.md:200-208`):**

| Profile | Reserved (pinned) RAM | VRAM use | Budget set in code |
|---|---|---|---|
| 0 AutoRAM_AutoVRAM | none for good; pinned pool shared at run time; `cpuCompute="auto"` | streamed + dynamic preload | transformer 1200 MB, others 3000 MB |
| 1 HighRAM_HighVRAM | all models | profile 2 + autoPreload | transformer 1200, others 3000 |
| 2 HighRAM_LowVRAM | all models | streamed block by block | transformer 1200, others 3000 |
| 3 LowRAM_HighVRAM | transformer only | profile 4 + autoPreload | transformer 1200, others 3000 |
| 4 LowRAM_LowVRAM (recommended) | transformer only | streamed block by block | transformer 1200, others 3000 |
| 4.5 (WanGP "4+") | as 4 | "sending one part at a time. Saves up to about 1 GB of VRAM, slightly slower" | — |
| 5 VerylowRAM_LowVRAM (fail safe) | none, only the staging ring | streamed | **transformer 400 MB**, others 3000 |

All profiles quantize the transformer to 8-bit (quanto `qint8`) by default; profiles 3–5 also quantize T5/Llama/LLM text encoders (`offload.py:7133-7141`, `mmgp\README.md:109`).

**Older claims (mmgp 3.8.1, `mmgp\README.md:12-16`, `:72-77`), copied exactly:**

- "VRAM: minimum 6 GB, recommended 24 GB (RTX 3090/ RTX 4090)"
- "RAM: minimum 24 GB, recommended 48 GB"
- "These RAM requirements are for Linux systems. Due to different memory management Windows will require an extra 16 GB of RAM to run the corresponding profile."
- "HighRAM_HighVRAM (1): at least 48 GB of RAM and 24 GB of VRAM"
- "HighRAM_LowVRAM (2): at least 48 GB of RAM and 12 GB of VRAM"
- "LowRAM_HighVRAM (3): at least 32 GB of RAM and 24 GB of VRAM"
- "LowRAM_LowVRAM (4): at least 32 GB of RAM and 12 GB of VRAM"
- "VerylowRAM_LowVRAM (5): at least 24 GB of RAM and 10 GB of VRAM"

The 4.3.0 code keeps these figures only as comments (`offload.py:7156-7185`). The 4.x README drops fixed numbers: "The RAM and VRAM needed depend on the models and on the profile: WanGP runs some video models with 6 GB of VRAM." (`Wan2GP\mmgp\README.md:11`).

### 5.3 Weight formats and quantization

- mmgp core: quanto `qint8` on-the-fly quantization of the transformer (`_quantize`, `offload.py:2406`); scaled FP8 built in; pluggable handlers via `quant_router` (`mmgp\README.md:19`, `:221-224`).
- Wan2GP handlers (`Wan2GP\shared\qtypes\`): GGUF, NVFP4, Nunchaku INT4/FP4, INT8 ConvRot, scaled FP8, GPTQ INT4, bnb NF4, asym W4A8, block-scaled, Prism.
- **GGUF in Wan2GP uses llama.cpp's CUDA kernels**: an external wheel `llamacpp_gguf_cuda` provides `linear(raw, qtype, shape, x, bias, dtype)` and `linear_chunked` (`Wan2GP\shared\qtypes\gguf.py:196-234`, `:896-945`). Docs: "Version 1.0.30 accelerates GGUF models with packed MMVQ/MMQ, direct FP16/BF16 activation quantization, CUDA-graph-safe workspaces and quantized KV-cache attention" (`Wan2GP\docs\INSTALLATION.md:265`); "On Q4_K/Q6_K layers this was 2-10% faster without bias and 5-19% faster with bias" (`:269`) **[claim]**. Fallback: PyTorch dequant functions for Q2_K…Q8_0, IQ1_S…IQ4_XS, plus a new Q2_0 and "PTQ1_0" (`gguf.py:1150-1460`).
- INT8 kernels: WGPKernels (own CUDA INT8, "RTX 20XX or newer"), Comfy Kitchen, Triton fallback (`Wan2GP\docs\INSTALLATION.md:61-75`). Many Triton kernels in `Wan2GP\shared\kernels\` (int8, fp8, nvfp4 W4A16, conv3d, group-norm+SiLU, LoRA).

### 5.4 Speed tricks

- Attention modes: `sdpa`, `flash`, `xformers`, `sage` (SageAttention 1), `sage2`, `radial` (block-sparse), `sage3` (`Wan2GP\shared\attention.py:279-292`). For Turing: "RTX 20XX (Turing): SageAttention 1.0.6. SageAttention 2 is not supported." (`Wan2GP\docs\INSTALLATION.md:180`). A Turing-specific Sage 1 kernel for GPUs "whose Triton has no BF16 conversion instructions (compute capability 7.5)" (`Wan2GP\shared\sage1_turing.py:17-24`).
- Step skipping: TeaCache ("Smart Teacache"), MagCache, First Block Cache (`Wan2GP\docs\CHANGELOG.md:1682`, `:1861`, `:351`); `--teacache 1.5/2.0` (`Wan2GP\docs\CLI.md:324`, `:394-398`).
- Step-distillation LoRAs and "accelerator profiles" (4/6/8 steps) (`Wan2GP\README.md:89-123`).
- `torch.compile` (needs Triton; "RTX 20XX requires Triton 3.2", `Wan2GP\docs\INSTALLATION.md:17`), CUDA graphs per block (`Wan2GP\mmgp\cuda_graphs.py`), VAE tiling, "Attention Head Split" to lower attention peak (`Wan2GP\docs\TROUBLESHOOTING.md:76`).
- `--fp16` "Force fp16 instead of bf16 models" (`Wan2GP\docs\CLI.md:194`) — needed on Turing.

### 5.5 Claims (copied)

- "Low VRAM requirements: run select models with as little as **4 GB of VRAM**." (`Wan2GP\README.md:38`) **[claim]**, models not named.
- "Older Nvidia GPU support: use GTX 10XX, RTX 20XX, and newer cards." (`Wan2GP\README.md:39`); "**Supported GPUs:** RTX 40XX, RTX 30XX, RTX 20XX, GTX 16XX, GTX 10XX, ..." (`Wan2GP\README.md:485`).
- "At least 6GB of VRAM for basic models"; "Wan 2.1 T2V 1.3B: Fastest, lowest VRAM (6GB)"; "Wan 2.1 T2V 14B: ... requires more VRAM (12GB+)"; "Low VRAM (6-8GB): Use 1.3B models" (`Wan2GP\docs\GETTING_STARTED.md:10`, `:93-94`, `:104`).
- "For Flux dev at 1024x1024, a manual 6000 MB shortened the profile 4 steps by about 22%, to within about 20% of profile 1, with about half of its peak VRAM." (`Wan2GP\docs\CLI.md:218`) — a measurement statement, no raw data **[claim]**.
- "asyncTransfers ... may increase speed by 20% (mostly visible on fast modern GPUs)" (`mmgp\README.md:145`); compile "may accelerate up to x2" (`:150`) **[claims]**.
- "8 GB is enough to get started and almost at NO SPEED COST thanks to MMGPv4" (Worlds feature, `Wan2GP\README.md:209`) **[claim]**.
- LTX-2.5 VAE on RTX 20XX "measured 4.8–6.8× faster" with WGPKernels 1.3.0 (`Wan2GP\README.md:249`) **[claim]**.

### 5.6 GTX 1650 + 8 GB RAM

- GPU: supported on paper (GTX 16XX listed). Use `sdpa` (or Sage 1 if its Triton INT8 path runs on TU117 — unverified), `--fp16`, profile 5. WGPKernels and Comfy Kitchen INT8 target "RTX 20XX or newer"; the GTX 16-series is not named.
- RAM is the problem. No 4.x doc gives a RAM minimum; 3.8.1 said 24 GB minimum (Linux) +16 GB on Windows. The pinned pool will not grow below 4 GB free RAM (`offload.py:1285`), so on 8 GB only the staging ring works. PyTorch + Gradio + CUDA context ≈ 2–3 GB of RAM [est]. Mapped read-only checkpoints let the OS page cache stream from NVMe, so a model larger than RAM can still run, slowly at load and step 1 [inference].
- Largest plausible: image — Flux/Z-Image GGUF Q4 or int8; video — **Wan 2.1 1.3B** (their "6 GB" model) at 480p with fewer frames; Wan 14B only in theory.

### 5.7 Best idea to reuse

**Learned block order + fixed VRAM slots + a small pinned staging ring fed by copy threads.** The ring gives most of the speed of pinning the whole model for ~3 blocks of pinned RAM. That is exactly the trade an 8 GB machine needs. Second idea: **autoPreload** — measure the activation peak in step 1, then fill the leftover VRAM with resident blocks.

---

## 6. Comparison table

| | ComfyUI | ComfyUI-GGUF | stable-diffusion.cpp | nunchaku | Wan2GP + mmgp 4.3 |
|---|---|---|---|---|---|
| Language / runtime | Python, PyTorch (+ `comfy_aimdo`, `comfy_kitchen` binaries) | Python plugin for ComfyUI | C/C++, ggml (patched fork) | C++/CUDA kernels + Python/diffusers | Python, PyTorch, Triton, own CUDA wheels |
| Video models | yes (Wan, LTX, Hunyuan, …) | via ComfyUI | yes (Wan 2.1/2.2, LTX-2, Hunyuan 1.5, H3, LingBot) | **no** | yes (largest set) |
| Transfer unit | module; block in prefetch-queue models | module | graph segment = block | block | tower block |
| Prefetch | 1 block ahead (dynamic), 2 rotating streams | inherits | next segment, async `tensor_set_async` | 1 block, ping-pong buffers | learned order, window 1–2, staging-ring thread |
| Pinned RAM | ≤ 40 % RAM (Windows), tiered eviction | inherits | whole offloaded params use host (pinned) buffer type | all offloaded blocks pinned | budgeted `cudaHostRegister`, 3-block staging ring |
| Disk streaming | yes: file → GPU direct ("fast disk") | mmap load, then released | yes: `--params-backend disk` | no | mmap read-only + read-ahead; page cache |
| Eviction policy | VBAR page-out by priority/watermark | — | reverse segment order (last first) | n/a (fixed buffers) | slots by stream order; preload reclaimed under pressure |
| VRAM estimate | dynamic (no estimate) or legacy estimate | adds largest dequant temp | measured per segment + budget `--max-vram` | fixed | measured at step 1 (autoPreload) |
| Quant formats | FP8, NVFP4, MXFP8, INT8(+ConvRot), W4A4, W4A8, W6A8 | GGUF Q2_K…Q8_0, IQ4_NL/XS | all ggml types, INT8 ConvRot, FP8 (fork) | SVDQuant INT4 W4A4, NVFP4 | quanto int8, FP8, GGUF (llama.cpp kernels), NVFP4, Nunchaku, ConvRot, NF4, GPTQ |
| Dequant method | Kitchen kernels | full FP16 temp per call, torch ops | ggml MMQ / dequant+GEMM | fused W4A4 GEMM + low-rank | llama.cpp MMQ/MMVQ, Triton, WGPKernels |
| Attention on Turing | SDPA, xformers, Sage 1 (if installed) | inherits | ggml flash-attn; Sage needs SM80 | own FP16 attention | sdpa, xformers, Sage 1 (Turing kernel) |
| Step caching | EasyCache | inherits | ucache, easycache, dbcache, taylorseer, cache-dit, spectrum | FB cache, double FB | TeaCache, MagCache, FB cache |
| VAE memory | 2-D/3-D tiling | inherits | tiling + **auto OOM retry**, temporal tiling, TAE | diffusers | tiling, head split |
| fp16 on GTX 16xx | disabled unless model too big for VRAM | inherits | ggml default (F16 weights, F32 accumulate) | FP16 required on Turing | `--fp16` flag |
| Claimed minimum | "4GB vram + 8GB ram" | none | Flux "6GB or even 4GB"; Z-Image "4GB or less" | Flux 4 GiB; Qwen-Image ~3 GB | "as little as 4 GB"; 1.3B "6GB"; old: 24 GB RAM |
| Runs on GTX 1650 + 8 GB? | plausible (fast-disk); untested | plausible; untested | **most plausible** (native, disk params); untested | unlikely (pinned RAM, tensor-core MMA) | plausible for 1.3B video / image; RAM-tight |
| Best idea | VBAR residency + file→GPU | move compressed bytes | graph cuts + segment residency manager | fused low-rank + W4A4 | learned order + slots + staging ring + autoPreload |

---

## 7. What fits on this machine, and how fast [est]

All rows: weights streamed per step (disk or RAM), 4-bit weights, FP16 compute at an effective **~3 TFLOP/s** (GTX 1650 peak ~6 TFLOP/s FP16 without tensor cores; I assume ~50 % efficiency). These are ceilings from the FLOP formula in §0, not measurements. Real speed is likely 1.5–3× slower.

| Model | Setting | N tokens | FLOPs/step | s/step | Steps | Total denoise |
|---|---|---|---|---|---|---|
| SD1.5 0.9B UNet | 512² | conv | ~0.8 T [rough] | ~0.3–1 | 20 | ~10–20 s |
| Flux.1-schnell 12B | 512², 4 steps | ~1,536 | ~3.9e13 | ~13 | 4 | ~1 min |
| Flux.1-dev 12B | 1024², 28 steps | ~4,608 | ~1.25e14 | ~42 | 28 | ~20 min |
| Z-Image-Turbo 6B | 1024², 8 steps | ~4,600 | ~6e13 | ~20 | 8 | ~3 min |
| FLUX.2-dev 32B | 1024², 28 steps | ~5,000 | ~3.3e14 | ~110 | 28 | ~50 min |
| Wan 2.1 1.3B | 480p, 33 frames | ~14,000 | ~7e13 | ~25 | 4 (distill) / 30 | ~2 min / ~12 min |
| Wan 2.1 1.3B | 480p, 81 frames | ~32,800 | ~2.8e14 | ~95 | 4 / 30 | ~6 min / ~48 min |
| Wan 2.1 14B | 480p, 33 frames | ~14,000 | ~5.5e14 | ~180 | 4 (lightx2v) | ~12 min |

Weight streaming per step for Flux Q4 (~6.4 GB) at ≥ 5 GB/s takes ≤ 1.3 s: hidden under 13–42 s of compute. So **disk streaming costs almost nothing here**, and the RAM does not need to hold the model.

Activation memory is the real VRAM limit [est]: Wan 14B at 14,000 tokens: hidden state 14k × 5120 × 2 B ≈ 144 MB, FFN intermediate ≈ 390 MB; attention must be memory-efficient (flash/SDPA, or head split). One Wan 14B block at Q4 ≈ 200 MB. Two slots + activations fit in ~3.3 GB. The VAE decode is the peak; it needs tiling or TAE.

**Answer to "largest model it could plausibly run":**
- Image: **Flux.1-dev 12B (Q4_K)** is practical in minutes; **Qwen-Image 20B / FLUX.2-dev 32B (Q4)** are possible with disk streaming but take ~1 hour per image [est].
- Video: **Wan 2.1 1.3B** and **Wan 2.2 TI2V-5B** are practical with a step-distill LoRA; **Wan 2.1 14B / 2.2 A14B (Q4)** is possible with a 4-step LoRA at 480p and ≤ 33 frames, ~10–15 min per clip [est].
- Evidence: no repo has a log from a GTX 1650 + 8 GB RAM. ComfyUI claims this hardware class; sd.cpp claims 4 GB VRAM for Flux; Wan2GP lists GTX 16XX.

Turing-specific risks:
1. **FP16 stability.** ComfyUI calls FP16 "broken" on the 16-series (`comfy\model_management.py:1975`). Some models (Flux, Wan) also overflow FP16 in some layers. FP32 compute halves speed. We must test per model.
2. **No tensor cores on TU117.** Kernels that need `mma.sync` (nunchaku, SageAttention, CUTLASS, Triton `tl.dot` on INT8/FP16) may fail or run slowly. ggml has non-MMA fallbacks (dp4a, tile kernels).
3. **INT8 via dp4a** gives ~4× the FP32 rate on Turing [est, hardware spec]. A W8A8 path (sd.cpp INT8 ConvRot via cuBLAS, `docs\ggml_extensions.md:296-303`) could be the fastest compute path on this GPU. Untested.

---

## 8. What our unified engine should take from these

### 8.1 Can stable-diffusion.cpp + llama.cpp's ggml be one shared base for text, image and video?

**Yes in principle, with work. Today they are two forks of one library.**

What is already shared:
- Same tensor library (ggml), same file format (GGUF), same quant types (Q2_K…Q8_0, IQ*), same backend set (CPU AVX2, CUDA, Vulkan, …), same `ggml-backend` scheduler and buffer types.
- Proof that ggml quant kernels serve diffusion well: Wan2GP, a PyTorch app, chose **llama.cpp's GGUF CUDA MMQ/MMVQ kernels** for its diffusion and LLM GGUF layers (`Wan2GP\shared\qtypes\gguf.py:215-234`, `Wan2GP\docs\INSTALLATION.md:263-269`).
- Upstream llama.cpp ggml already has the diffusion ops: `GROUP_NORM`, `IM2COL_3D`, `CONV_2D`, `CONV_3D`, `CONV_2D_DW`, `UPSCALE`, `PAD` (`llama.cpp\ggml\include\ggml.h:525-564`).

What is not shared:
1. **sd.cpp builds a patched ggml fork** by default (`leejet/ggml`, `stable-diffusion.cpp\.gitmodules:1-3`) with `ggml-ext.h` (custom ops, kernel overrides, fusion rules) and `ggml-graph.h` (`docs\ggml_extensions.md`, `docs\build.md:53-58`). Upstream mode works but loses INT8 ConvRot, FP8 weights, SageAttention, and "some operators and performance optimizations" (`docs\build.md:59-71`). On our GPU, SageAttention and CUTLASS do not apply anyway; INT8 ConvRot and FP8 matter less. So **upstream ggml is acceptable for the 1650**, with a measured speed loss to be confirmed.
2. **Two model zoos.** sd.cpp reimplements LLM text encoders (Qwen3, Qwen3-VL, Qwen3.5 DeltaNet, GPT-OSS MoE) in its own `GGMLBlock` code (`src\model\te\llm.hpp`). llama.cpp has its own, faster-moving versions. A unified engine should use **one** LLM implementation — llama.cpp's `libllama` — for chat LLMs, and accept sd.cpp's text-encoder copies at first.
3. **Two memory managers.** llama.cpp: mmap + `-ngl` layer split + MoE expert cache (`D_llama_cpp.md`). sd.cpp: `DeviceResidencyManager` + graph cuts + segment prefetch + disk params. sd.cpp's manager lives **above** ggml-backend, so it can manage llama.cpp-style layer graphs too.

Recommended base: **stable-diffusion.cpp as the image/video engine, llama.cpp as the text engine, both built against one pinned ggml revision in one process.** Start with sd.cpp's patched fork rebased on the same upstream commit that llama.cpp uses, or with upstream ggml if the rebase is too costly. Long term, move sd.cpp's residency manager into a shared layer that both engines call.

### 8.2 Reuse (take as is)

| Take | From | Why |
|---|---|---|
| Whole model zoo, samplers, VAE tiling with **auto OOM retry**, conditioning cache, step-cache modes | sd.cpp | Native, no Python RAM overhead, the widest model list, already GGUF. |
| Graph cuts + `SegmentWeightPipeline` + `DeviceResidencyManager` + `--params-backend disk` + `--max-vram` | sd.cpp (`src\core\ggml_graph_cut.*`, `src\core\segment_weight_pipeline.*`, `src\device_residency_manager.h`) | Already does "stream from disk, prefetch next block, evict last-first". |
| GGUF MMQ/MMVQ CUDA kernels | ggml / llama.cpp | Fused dequant-GEMM; one kernel set for text and image. |
| GGUF conversion and per-model "keep in high precision" key lists | ComfyUI-GGUF `tools\convert.py` | Small tensors and modulation params must stay F16/F32. |
| Step-distill LoRAs (lightx2v, Lightning, Turbo) and TeaCache/MagCache/FB-cache/DBCache | all | Fewer steps is the largest speed lever on a slow GPU. |

### 8.3 Rewrite or add (ideas to port, not code)

| Build | Idea from | Notes |
|---|---|---|
| **Pinned staging ring** (3 largest blocks) fed by 2–4 copy threads, instead of pinning whole models | mmgp (`offload.py:874-875`, `_StagingRing`) | sd.cpp pins all CPU params via the host buffer type (`src\model_manager.cpp:1141-1145`). On 8 GB RAM that will not fit. Ring = ~0.5–1 GB pinned. |
| **Direct file → pinned → GPU read path** on Windows (unbuffered/overlapped reads into the ring) | ComfyUI fast disk (`comfy\memory_management.py:77-94`), mmgp read-ahead (`read_ahead.py`) | Windows has no `O_DIRECT`; use `FILE_FLAG_NO_BUFFERING` + overlapped I/O (open question in `00_SYNTHESIS.md` §5). |
| **Learned block order across steps** (last block of step k → first block of step k+1) | mmgp (`README.md:15`) | Prefetch never stalls at the step boundary. sd.cpp evicts last-first, which matches this. |
| **autoPreload**: measure activation peak in step 1, fill spare VRAM with resident blocks | mmgp (`auto_preload.py`) | Wan2GP reports ~22 % faster Flux steps with 6 GB preload [claim]. On 4 GB the gain is small, but it is free. |
| **VRAM-as-cache with resident signatures** | ComfyUI VBAR | Lets text-LLM and diffusion models share 3.3 GB without unload/reload estimates. |
| **cpuCompute for small-N calls** | mmgp (`offload.py:876-886`) | For LLM decode only (N ≤ 8 rows). llama.cpp already does this by layer split. |
| **W8A8 INT8 (dp4a) and later W4A4 + low-rank branch** compute kernels for diffusion | sd.cpp INT8 ConvRot, nunchaku SVDQuant | Diffusion is compute-bound, so activation quantization pays (the opposite of the LLM finding in `00_SYNTHESIS.md` §4.1). Must be written for TU117 without tensor cores. |
| **Virtual GPU emulation** for tests | mmgp (`virtual_gpu.py`) | Test 4 GB / 8 GB behavior on a bigger dev box. |
| Per-model FP16 safety table (which layers need FP32) | ComfyUI 16-series rule, Wan2GP `--fp16` | GTX 1650 has no BF16; FP16 overflow must be handled per layer. |

### 8.4 Do not take

- **nunchaku as an engine**: PyTorch/diffusers, no video, pinned offload, tensor-core MMA; TU117 support unverified.
- **ComfyUI dynamic VRAM as code**: the core (`comfy_aimdo`) is a closed external binary in this clone (source not found); the node graph and Python stack cost RAM we do not have.
- **ComfyUI-GGUF dequant**: full FP16 temporaries; fine for testing, not for an engine.
- **mmgp as code**: tightly bound to PyTorch hooks and quanto; 7,000+ lines in `offload.py`. Take the policies.

### 8.5 Shared policy for text + image + video

One residency manager, two policies:

| | LLM decode (N = 1) | Diffusion step (N = 10³–10⁴) |
|---|---|---|
| Bottleneck | bytes per token (memory bandwidth) | FLOPs per step (GPU compute) |
| VRAM priority | hot layers / hot experts stay resident | activations first, then resident blocks |
| Streaming | avoid; only cold experts from SSD | stream everything every step; prefetch 1–2 blocks |
| Quantization goal | fewest bits (storage) | fast compute (W8A8 / W4A4) |
| Where the CPU helps | compute layers left in RAM | none (too slow); CPU only does I/O |

### 8.6 First steps (proposal)

1. Build sd.cpp (CUDA, upstream ggml **and** patched ggml) on this machine. Measure Flux-schnell Q4_K 512², Z-Image-Turbo Q3_K, Wan 2.1 1.3B Q8 480p×33 with `--params-backend diffusion=disk` and with `--offload-to-cpu`. Record s/step, peak RAM, peak VRAM, NVMe GB/s. This tests §7.
2. Same runs in ComfyUI (dynamic VRAM, fast disk) as a reference.
3. Check FP16 correctness per model on the 1650.
4. Then replace sd.cpp's "pin all CPU params" with a staging ring and a Windows direct-read path.

---

## 9. Single best ideas (summary)

1. **Diffusion is compute-bound, so stream everything.** At N = thousands of tokens per step, the GPU needs < 1 GB/s of weights. NVMe streaming is free; RAM need not hold the model. (§0, §7)
2. **sd.cpp's graph-cut segment executor with last-first eviction and next-segment prefetch** is the best existing core for our engine, and it already reads parameters from disk. (§3.2)
3. **mmgp's small pinned staging ring** gives most of the speed of full pinning at ~3 blocks of RAM — the fix for 8 GB RAM. (§5.2)
4. **ComfyUI's VRAM-as-a-cache (VBAR + residency signatures) and file → GPU reads** remove fragile VRAM estimates. (§1.2)
5. **For diffusion, quantize activations too** (W8A8 dp4a now, SVDQuant-style W4A4 + fused low-rank later). This is where nunchaku's 3× comes from. (§4, §8.3)
