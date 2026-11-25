# D — llama.cpp source analysis (reference engine)

Analyst D. Source: `<repo>\repos\llama.cpp` (shallow clone of master, 2026-11-12).
Goal of the project: run 70B–700B+ models on machines with 8–32 GB RAM and 0–12 GB VRAM.

All paths are relative to the repo root. A citation `file:N` or `file:N-M` gives the line or line range in that clone. Line numbers can move in later commits.

Method: read-only. No build, no run. "Not found" means that a search of the code did not find the feature.

---

## 0. Summary of the most important findings

1. **Weights are mmapped by default.** The CPU-side weights are used in place from the OS page cache. A model larger than RAM works, but only through OS demand paging. There is no engine-managed weight cache for the CPU path.
2. **This version has a GPU MoE expert cache** (`src/llama-moe-cache.cpp`, flag `--moe-cache-mib`). It keeps hot experts of CPU-resident MoE layers in a VRAM bank, with an LRU + frequency policy. It is reactive: a miss is uploaded after the router has run, with a host sync. It does not prefetch. It is marked EXPERIMENTAL in `include/llama.h:400`.
3. **CPU MoE compute reads only the selected experts.** `ggml_compute_forward_mul_mat_id` skips experts with zero rows. With mmap, only the pages of selected experts fault in from disk.
4. **A prefetch primitive exists** (`llama_prefetch`, which uses `PrefetchVirtualMemory` on Windows and `madvise(MADV_WILLNEED)` on POSIX). It is used only for per-layer token-embedding tables ("lazy" tensors). It is not used for experts or layers.
5. **There is no layer streaming to the GPU.** For a large batch, the scheduler can copy a host weight to the GPU for one op ("op offload"). For a small batch (decode), CPU-resident layers are computed on the CPU. There is no double-buffered, async, layer-ahead upload pipeline.
6. **Weight repacking defeats mmap streaming.** If a CPU tensor goes to a repack ("extra") buffer type, the loader allocates anonymous RAM and copies the whole tensor in at load time. For disk streaming, `--no-repack` is necessary.

---

## 1. Architecture overview

```
                         +--------------------------------------------+
  CLI / server           |  common/arg.cpp  (flags)                   |
  (tools/cli, server)    |  common/fit.cpp  (--fit: auto ngl / -ncmoe)|
                         +----------------------+---------------------+
                                                |
                                                v
 +----------------------------------------------------------------------------------+
 | libllama (src/)                                                                   |
 |                                                                                   |
 |  llama_model_loader  (llama-model-loader.cpp)                                     |
 |    gguf_init_from_file (ggml/src/gguf.cpp)  -> metadata + tensor index            |
 |    llama_file / llama_mmap / llama_mlock   (llama-mmap.cpp)                       |
 |    create_tensor(): pick buffer type per tensor  (overrides -ot, buft lists)      |
 |    load_all_data(): mmap-wrap or read + upload                                    |
 |                                                                                   |
 |  llama_model (llama-model.cpp, models/*.cpp)                                      |
 |    load_tensors(): layer -> device map (-ngl, -ts), buffers, weights               |
 |    build_graph(): per-arch ggml graph (llama-graph.cpp helpers, build_moe_ffn)    |
 |                                                                                   |
 |  llama_context (llama-context.cpp)                                                |
 |    decode() -> process_ubatch() -> graph_compute()                                |
 |    memory: llama_kv_cache (+ iswa / hybrid / recurrent)                           |
 |    llama_moe_cache (llama-moe-cache.cpp)   [optional, --moe-cache-mib]            |
 |    sched_copy_experts(): copy callback for host weights                           |
 +-------------------------------------+--------------------------------------------+
                                       |
                                       v
 +----------------------------------------------------------------------------------+
 | ggml (ggml/src)                                                                   |
 |   ggml.c          : tensor/op definitions, type_traits, ggml_quantize_chunk       |
 |   ggml-backend.cpp: buffers, buffer types, ggml_backend_sched (graph splitter)    |
 |   ggml-alloc.c    : compute-buffer allocator                                      |
 |   backends:  ggml-cpu (+repack/AMX/KleidiAI extra bufts) | ggml-cuda | metal |    |
 |              vulkan | sycl | rpc (remote) | ...                                   |
 +----------------------------------------------------------------------------------+

 Memory placement for a partial-offload MoE run (-ngl all + --cpu-moe, mmap on):

   DISK (GGUF) ==mmap==> OS page cache (RAM) <== CPU reads expert slices on demand
        |                         |
        |                         +--(large batch, op offload)--> GPU temp copy of used experts
        |                         +--(MoE cache miss, ubatch<=32)--> GPU expert bank slot
        v
   GPU VRAM: attention weights, dense FFN, norms, output, KV cache, compute buffers,
             optional MoE cache bank
```

---

## 2. Model loading (Question 1)

### 2.1 Opening the GGUF

- `llama_model_loader` sets the load mode first: `use_mmap` is true for `MMAP`, `MMAP_MLOCK` and `AUTO`; `use_direct_io` is true for `DIRECT_IO` (`src/llama-model-loader.cpp:564-565`). The enum is in `include/llama.h:209-214`. The CLI flag is `-lm/--load-mode {auto,none,mmap,mlock,mmap+mlock,dio}` (`common/arg.cpp:2663-2680`).
- The GGUF header is parsed with `gguf_init_from_file(..., no_alloc=true)` (`src/llama-model-loader.cpp:569-575`). This creates a ggml context that holds only tensor metadata (shape, type, offset). No tensor data is read here. Implementation: `ggml/src/gguf.cpp:1017`. Default alignment `GGUF_DEFAULT_ALIGNMENT`, overridable by the `general.alignment` key (`ggml/src/gguf.cpp:630`).
- Each file is opened as a `llama_file` (`src/llama-model-loader.cpp:584`). Split GGUFs (`-00001-of-0000N`) are found and opened one by one (`src/llama-model-loader.cpp:601-760`). Each tensor gets an entry (file index + offset) in `weights_map`.
- For an embedded GGUF opened from a `FILE*`, the data section must be aligned for mmap (`src/llama-model-loader.cpp:775-780`).

### 2.2 Placement decision per tensor

`llama_model_loader::create_tensor` (`src/llama-model-loader.cpp:1211-1485`) selects a buffer type for each weight:

1. If the tensor has the `TENSOR_READ_LAZY` flag and lazy mode accepts it, it goes to the plain CPU buffer type (`src/llama-model-loader.cpp:1306-1308`, `1437-1440`).
2. Tensor overrides (`-ot`, `--cpu-moe`, `--n-cpu-moe`) are matched with `std::regex_search` on the tensor name (`src/llama-model-loader.cpp:1329-1355`). An override to the CPU buffer type re-runs selection over the full CPU buffer list, so extra (repack) types can be used (`src/llama-model-loader.cpp:1335-1337`). With mmap on, a one-time warning recommends `--load-mode none` (`src/llama-model-loader.cpp:1338-1343`).
3. Otherwise `select_weight_buft` takes the first buffer type in the layer's list whose device supports the op that will use the weight (`src/llama-model-loader.cpp:1162-1173`). The test builds a dummy op, for example a `ggml_mul_mat_id` with 512 tokens for expert tensors (`src/llama-model-loader.cpp:1053-1061`), and asks `ggml_backend_dev_supports_op` (`src/llama-model-loader.cpp:1151-1158`).
4. With mmap, a GPU "host" (pinned) buffer type is replaced by the plain CPU buffer type, so that the weight can stay in the mapping (`src/llama-model-loader.cpp:1364-1372`).

### 2.3 How mmap is used

- `init_mappings` creates one `llama_mmap` per file when `use_mmap` is on or when any lazy tensor exists (`src/llama-model-loader.cpp:1505-1540`). Prefetch size is "whole file" (`(size_t)-1`) when prefetch is requested and mmap is on (`src/llama-model-loader.cpp:1524`). `load_tensors` calls it with `prefetch = true` (`src/llama-model.cpp:2000`). The quantizer calls it with `false` (`src/llama-quant.cpp:1042`).
- **POSIX path** (`src/llama-mmap.cpp:708-759`):
  - `posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL)` on Linux (`:718`).
  - `MAP_POPULATE` when prefetch is on and no lazy range exists (`:723`). This makes the kernel read the whole file at map time.
  - `mmap(NULL, size, PROT_READ, MAP_SHARED, fd, 0)` (`:725`).
  - `POSIX_MADV_WILLNEED` over the file minus the lazy ranges (`:743-747`); `POSIX_MADV_RANDOM` over lazy ranges (`:748-750`); `POSIX_MADV_RANDOM` over the whole file on NUMA systems (prefetch is set to 0 for NUMA, `:716`, `:751-756`).
- **Windows path** (`src/llama-mmap.cpp:816-886`):
  - `CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL)` (`:826`), then `MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0)` (`:833`). The whole file is one view.
  - If prefetch > 0 and `_WIN32_WINNT >= 0x602` (Windows 8+), `PrefetchVirtualMemory` is loaded dynamically from kernel32 and called with the complement of the lazy ranges (`:841-865`). For a normal load this is the whole file.
  - `unmap_fragment` is a no-op on Windows (`:868-871`). On POSIX it calls `munmap` on page-aligned sub-ranges (`:773-807`).
- **Weights in place.** For a CPU tensor in a mapped file, the loader does not copy. It wraps the mapped range as a backend buffer with `ggml_backend_dev_buffer_from_host_ptr` (`src/llama-model.cpp:2035-2053`) and sets the tensor data pointer into the mapping with `ggml_backend_tensor_alloc(buf_mmap, cur, data)` (`src/llama-model-loader.cpp:1797-1832`). Only the ranges that hold that context's tensors are wrapped (`src/llama-model.cpp:2040-2045`), so tensors owned by other contexts "stay out of RAM" (comment at `src/llama-model.cpp:2038-2043`).
- **GPU tensors with mmap.** The data is copied from the mapping with a synchronous `ggml_backend_tensor_set` (`src/llama-model-loader.cpp:1830-1832`). The async staged upload is disabled when mmap is on (`src/llama-model-loader.cpp:1679-1682`).
- **After load**, on POSIX, the parts of the mapping before the first and after the last CPU-used tensor are unmapped (`src/llama-model-loader.cpp:1951-1962`). On Windows this does nothing (see above).

### 2.4 `use_mlock`

- `use_mlock` is true for `MLOCK` and `MMAP_MLOCK` (`src/llama-model.cpp:1679`).
- With mmap, each mapping gets a `llama_mlock`, and `grow_to(offs + size)` locks the mapping up to the end of each CPU tensor (`src/llama-model-loader.cpp:1531-1535`, `1821-1825`). Lazy tensors are not locked (comment at `:1821`).
- Without mmap, the allocated host buffers are locked (`src/llama-model.cpp:2075-2088`).
- POSIX lock: `mlock()` with an `RLIMIT_MEMLOCK` hint on failure (`src/llama-mmap.cpp:926-965`). Windows lock: `VirtualLock`; on failure the code grows the working set with `SetProcessWorkingSetSize` and tries again once (`src/llama-mmap.cpp:972-1004`).
- For our goal mlock is not useful: a model larger than RAM cannot be locked.

### 2.5 Model bigger than RAM

The code has no special path for "model larger than RAM". Not found: any check that compares the model size to free RAM and changes behavior. What happens follows from the mechanisms above:

- The mapping is virtual. A CPU op that reads a weight takes a page fault, and the OS reads the page from disk into the page cache. When RAM is full, the OS evicts clean file pages (they can be read again from the file; no swap is needed because the mapping is read-only `MAP_SHARED`/`PAGE_READONLY`).
- The load-time prefetch asks the OS to read the **whole file** (`MAP_POPULATE` on Linux at `src/llama-mmap.cpp:723`, `WILLNEED` at `:743-747`, `PrefetchVirtualMemory` on Windows at `:850-857`). For a file larger than RAM, this reads data that is then evicted before use. This costs time at load and gives no benefit.
- On Linux the whole file gets `POSIX_FADV_SEQUENTIAL` (`:718`). Decode access to expert weights is random, not sequential.
- Per decode token, all dense weights of CPU layers are read once, so if the working set (dense + active experts) does not fit in RAM, every token re-reads from disk. Speed is then bound by disk read throughput and page-fault latency.
- On Windows, `unmap_fragment` is a no-op, so the GPU-uploaded parts of the file stay mapped. Those pages were touched during upload and stay in the standby list until Windows reclaims them.

### 2.6 Other load paths

- **No mmap (`--load-mode none`)**: CPU tensors are read with `fread` into allocated buffers (`src/llama-model-loader.cpp:1836-1843`). GPU tensors go through 4 pinned staging buffers of 1 MiB (or 64 MiB + alignment with direct I/O) with `ggml_backend_tensor_set_async` and events (`src/llama-model-loader.cpp:1663-1673`, `1846-1897`). Tensors in non-host buffers are loaded biggest-first to keep the staging peak low (`src/llama-model-loader.cpp:1765-1776`).
- **Direct I/O (`--load-mode dio`)**: Windows uses `FILE_FLAG_NO_BUFFERING` (`src/llama-mmap.cpp:106-114`); POSIX uses `O_DIRECT` (`src/llama-mmap.cpp:425`). Buffer size `LLAMA_DIRECT_IO_BUFFER_SIZE` = 64 MiB (`src/llama-mmap.h`, define near the top). This is a load-time feature only.
- **Lazy tensors (`-lzm/--lazy-mode`)**: `common/arg.cpp:2682-2693`, enum `include/llama.h:221-223`. A tensor marked `TENSOR_READ_LAZY` by the architecture stays in the mapping with `MADV_RANDOM` and is not prefetched (`src/llama-model-loader.cpp:1183-1209`; auto threshold 4 GiB at `:1189`). Only per-layer token-embedding tables use this flag: `src/models/gemma4.cpp:65`, `src/models/deepseek41.cpp:163`, `src/models/qwen4exp.cpp:217`. For such tensors, `llama_prefetch_rows` (`src/llama-impl.cpp:22-40`) prefetches the rows of the current tokens before compute (`src/models/gemma4.cpp:474-485`). `llama_prefetch` merges page ranges and calls one `PrefetchVirtualMemory` on Windows or `madvise(MADV_WILLNEED)` per range on POSIX (`src/llama-mmap.cpp:1067-1142`).

---

## 3. GPU offload and tensor placement (Question 2)

### 3.1 `--n-gpu-layers` and the layer-to-device map

- Flag: `-ngl/--gpu-layers/--n-gpu-layers N|auto|all` (`common/arg.cpp:2782-2798`). `auto` = -1, `all` = -2. `llama_model::n_gpu_layers()` turns a negative value into "all layers + output" (`src/llama-model.cpp:2536-2539`). With `--fit on`, `common/fit.cpp` changes unset values (see 3.5).
- Buffer-type lists (`src/llama-model.cpp:1720-1725`):
  - CPU list, in priority order: ACCEL devices → one GPU host (pinned) buffer type (unless `--no-host`) → CPU extra buffer types (repack/AMX/KleidiAI, unless `--no-repack`) → plain CPU (`src/llama-model.cpp:1229-1289`).
  - GPU list: split buffer type (only for `--split-mode row`) → device buffer type → device extra types → then the whole CPU list as fallback (`src/llama-model.cpp:1291-1340`, `1722-1724`).
- Split points: by `--tensor-split` if given, else by free memory of each device (`src/llama-model.cpp:1736-1764`).
- Layer assignment (`src/llama-model.cpp:1766-1791`): `i_gpu_start = max(n_layer + 1 - n_gpu_layers, 0)`. Layers below `i_gpu_start` go to the CPU. The **last** `n_gpu_layers` layers (including the output layer as layer `n_layer`) go to GPUs, distributed by the split points with `std::upper_bound`. The input layer (token embedding) is always on the CPU (`:1780-1782`).
- `--split-mode` (`common/arg.cpp:2800-2822`): `none` (one GPU), `layer` (default, pipeline), `row` (split weights by rows, uses the backend's split buffer type), `tensor` (EXPERIMENTAL tensor parallel; arch must support it, `src/llama-model.cpp:398-399`).

### 3.2 ggml-backend concepts

- **Buffer type** (`ggml_backend_buffer_type_t`): an allocator for one memory kind (CUDA VRAM, CUDA pinned host, CPU, CPU repack, RPC remote).
- **Buffer usage**: weight buffers are tagged `GGML_BACKEND_BUFFER_USAGE_WEIGHTS` (`src/llama-model.cpp:2089-2093`). The scheduler uses this tag.
- **Scheduler** `ggml_backend_sched` (`ggml/src/ggml-backend.cpp`, constants at `:1163-1172`: max 16 backends, 30 inputs per split, 4 copies):
  - `ggml_backend_sched_backend_id_from_cur` (`:1338-1402`): a node whose dst is pre-allocated stays on that buffer's backend; a graph input goes to the last backend (CPU); an op that reads a weight runs on the weight's backend (`:1384-1397`).
  - **Op offload** (`:1387-1393`): if the weight is in a host buffer on the CPU backend and a higher-priority backend says `offload_op` is true, the op runs on that backend. The weight is then copied to the device for this op. CUDA returns true when the batch size ≥ `GGML_OP_OFFLOAD_MIN_BATCH`, default 32 (`ggml/src/ggml-cuda/ggml-cuda.cu:189-192`, `6690-6709`). For `MUL_MAT_ID`, the batch size is `ne[2]` = tokens. Flag: `--op-offload/--no-op-offload` (`common/arg.cpp:2940-2941`).
  - `ggml_backend_sched_split_graph` (`:1483`) cuts the graph into splits; each split runs on one backend and has a list of inputs that must be copied in.
  - `ggml_backend_sched_compute_splits` (`:2210-2297`): for each split, copy non-weight inputs, then call `ggml_backend_sched_prefetch_inputs`, then copy host-weight inputs, then compute async.
  - Host weights are copied through an optional **copy callback** first (`:2131-2134`). llama.cpp installs `llama_context::sched_copy_experts` (`src/llama-context.cpp:783`).
  - If the callback does not handle it, the copy tries `cpy_tensor_async`; else it syncs and calls `ggml_backend_tensor_copy` (`:2136-2148`). For CUDA, `cpy_tensor_async` returns false when the source is not a CUDA buffer (`ggml/src/ggml-cuda/ggml-cuda.cu:3122-3132`), so host→GPU weight copies on this path are **synchronous**.
  - `ggml_backend_sched_prefetch_inputs` (`:2152-2208`) only prefetches results of earlier splits (COMPUTE buffers) to host. It explicitly does not handle weights (`:2175-2179`).

### 3.3 Tensor overrides and MoE on CPU

- `-ot/--override-tensor "<regex>=<buft>,..."` (`common/arg.cpp:2726-2730`).
- `-cmoe/--cpu-moe` adds the override `"\.ffn_(up|down|gate|gate_up)_(ch|)exps" = CPU` (`common/arg.cpp:2732-2737`, regex at `common/common.h:1277`, helper at `common/common.h:1285-1287`).
- `-ncmoe/--n-cpu-moe N` adds that override only for blocks 0..N-1 (`common/arg.cpp:2739-2747`, helper `common/common.h:1289`).
- `-ncffn/--n-cpu-ffn N` does the same for dense FFN tensors (`common/arg.cpp:2770-2779`).
- Draft-model versions exist: `--spec-draft-cpu-moe`, `--spec-draft-n-cpu-moe` (`common/arg.cpp:4159-4175`).

With `-ngl all --cpu-moe`, the typical result is: attention, norms, shared experts, router, output and KV cache are on the GPU; routed expert tensors are in the CPU mapping. The `MUL_MAT_ID` for experts then runs on the CPU for decode (batch < 32) and on the GPU with a per-op copy of the used experts for prefill (batch ≥ 32, see 6.2).

### 3.4 KV and compute placement

KV placement is described in Section 5. Compute buffers are allocated per backend by the scheduler (`ggml_backend_sched_reserve`, `ggml/src/ggml-backend.cpp:2430`). For small batches, the graph callback pins the `norm` and `l_last` nodes of a layer to that layer's device to reduce cross-device copies (`src/llama-context.cpp:3255-3270`).

### 3.5 `--fit`

`-fit/--fit on|off` (`common/arg.cpp:2861-2874`) runs `common_fit_params` (`common/fit.cpp:980`), called from `common/common.cpp:1267`. It measures memory with dry-run (no_alloc) loads and: reduces context (step 2, `common/fit.cpp:460`), fills layers back to front; for MoE models it first puts all MoE tensors in system memory (step 3, `:575-577`), then converts dense-only layers into full layers front to back until devices are full (step 4, `:836-839`). It uses the false-position method on layers per device (`:764-770`). This is the closest thing to an automatic "plan" for small VRAM. It does not consider RAM < model size.

---

## 4. Quantization (Question 3)

### 4.1 Where types are defined

- Enum `ggml_type`: `ggml/include/ggml.h` (e.g. `GGML_TYPE_Q4_0 = 2` at `:394`, `IQ1_S = 19` at `:411`, `TQ1_0 = 34` at `:426`, `GGML_TYPE_COUNT = 44` at `:436`).
- Block structs: `ggml/src/ggml-common.h`.
- Generic traits (name, block size, byte size, to_float, reference quantizer): `type_traits[]` in `ggml/src/ggml.c:639` (e.g. Q4_K at `:799`, IQ2_XXS at `:823`).
- CPU traits (activation quantizer `from_float`, `vec_dot`, `vec_dot_type`, `nrows`): `type_traits_cpu[]` in `ggml/src/ggml-cpu/ggml-cpu.c:218` (Q4_0 at `:243-246`, Q4_K at `:320-323`).
- Quantize / dequantize code: `ggml/src/ggml-quants.c`. Dispatcher `ggml_quantize_chunk` at `ggml/src/ggml.c:8220`. Types that need an imatrix: IQ2_XXS, IQ2_XS, IQ1_S (`ggml/src/ggml.c:8212-8218`).
- Lattice tables (e.g. `iq2xxs_grid`) are in `ggml/src/ggml-common.h:560` and nearby; `IQ1S_DELTA` at `:1132`.

### 4.2 Block structs (QK_K = 256, `ggml/src/ggml-common.h:89`)

| Type | Location | Block | Fields | Bytes / block | bpw |
|---|---|---|---|---|---|
| Q4_0 | `ggml-common.h:194-199` | 32 | `ggml_half d; uint8_t qs[16];` | 18 | 4.5 |
| Q8_0 | `ggml-common.h:251-256` | 32 | `ggml_half d; int8_t qs[32];` | 34 | 8.5 |
| Q4_K | `ggml-common.h:323-338` | 256 (8×32) | `half d, dmin; uint8_t scales[12]` (6-bit scales+mins); `uint8_t qs[128]` | 144 | 4.5 |
| Q6_K | `ggml-common.h:358-368` | 256 (16×16) | `uint8_t ql[128]` (low 4 bits); `uint8_t qh[64]` (high 2 bits); `int8_t scales[16]`; `half d` | 210 | 6.5625 |
| IQ2_XXS | `ggml-common.h:378-385` | 256 | `half d; uint16_t qs[32]` (grid index + signs + scales packed) | 66 | 2.0625 |
| IQ1_S | `ggml-common.h:424-430` | 256 | `half d; uint8_t qs[32]; uint16_t qh[8]` | 50 | 1.5625 |
| TQ1_0 | `ggml-common.h:274-281` | 256 | `uint8_t qs[48]` (5 trits/byte); `uint8_t qh[4]` (4 trits/byte); `half d` | 54 | 1.6875 |

Code (verbatim from the file):

```c
// ggml-common.h:194-198
#define QK4_0 32
typedef struct {
    ggml_half d;           // delta
    uint8_t qs[QK4_0 / 2]; // nibbles / quants
} block_q4_0;

// ggml-common.h:251-255
#define QK8_0 32
typedef struct {
    ggml_half d;       // delta
    int8_t  qs[QK8_0]; // quants
} block_q8_0;

// ggml-common.h:327-337
typedef struct {
    GGML_EXTENSION union {
        struct { ggml_half d; ggml_half dmin; } GGML_COMMON_AGGR_S;
        ggml_half2 dm;
    } GGML_COMMON_AGGR_U;
    uint8_t scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
    uint8_t qs[QK_K/2];           // 4--bit quants
} block_q4_K;

// ggml-common.h:362-367
typedef struct {
    uint8_t ql[QK_K/2];      // quants, lower 4 bits
    uint8_t qh[QK_K/4];      // quants, upper 2 bits
    int8_t  scales[QK_K/16]; // scales, quantized with 8 bits
    ggml_half d;             // super-block scale
} block_q6_K;

// ggml-common.h:381-384
typedef struct {
    ggml_half d;
    uint16_t qs[QK_K/8];
} block_iq2_xxs;

// ggml-common.h:425-429
typedef struct {
    ggml_half d;
    uint8_t  qs[QK_K/8];
    uint16_t qh[QK_K/32];
} block_iq1_s;

// ggml-common.h:276-280
typedef struct {
    uint8_t qs[(QK_K - 4 * QK_K / 64) / 5]; // 5 elements per byte (3^5 = 243 < 256)
    uint8_t qh[QK_K/64]; // 4 elements per byte
    ggml_half d;
} block_tq1_0;
```

The file also defines newer types: Q1_0, Q2_0, MXFP4, NVFP4, F8_E4M3 (`ggml-common.h:180-230`, `ggml/src/ggml.c:8246-8255`).

Important for streaming: a tensor row is a sequence of whole blocks, and an expert of a 3-D expert tensor `[n_embd, n_ff, n_expert]` is one contiguous slice of `nb[2]` bytes. So one expert = one contiguous file range. This is good for disk reads and for `PrefetchVirtualMemory`.

### 4.3 imatrix: collection

`tools/imatrix/imatrix.cpp`:

- The tool sets a scheduler eval callback (`cb_eval`). `collect_imatrix` (`:561`) answers "ask" calls: yes for every `MUL_MAT_ID`; yes for `MUL_MAT` on `blk.*` weights with ≥ 16 tokens and F32 input (`:572-581`).
- For `MUL_MAT_ID` (`:600-660`), it reads the routing ids from the device, then for each (token, selected expert) pair adds `x[j]*x[j]` of the input activation to the per-expert, per-column accumulator, and counts the rows per expert (`:653-660`). So each expert gets its own importance vector.
- For `MUL_MAT` the same sum of squares is computed per column (`:729`).
- The result is saved as a GGUF with `<name>.in_sum2` and count tensors (`:867`, `:971-973`); a legacy `.dat` writer exists (`:762`).

### 4.4 imatrix: use in llama-quantize

- `src/llama-quant.cpp:1061-1076` loads the imatrix map and rejects non-finite values. `tensor_requires_imatrix` (`:925`) decides which tensors need it; quantization aborts without it when required (`:1198-1212`).
- Per-expert slicing: `imatrix_for_row` gives row `r` the slice `imatrix + (r / nrows_per_expert) * n_per_row` (`src/llama-quant.cpp:850-854`), so each expert is quantized with its own importance weights.
- Inside the quantizers the imatrix becomes per-element weights of the error function. Example Q4_K: `weights[l] = qw[l] * sqrtf(sigma2 + x^2)` with imatrix, else `av_x + |x|` (`ggml/src/ggml-quants.c:1584-1590`). The scale search (`make_qkx3_quants`, `make_qp_quants`) then minimizes weighted squared error.
- Type mix: `llama_tensor_get_type` (`src/llama-quant.cpp:786`) and `llama_tensor_get_type_impl` (`:507`) pick per-tensor types (e.g. bumps for `attn_v`, `ffn_down` in early layers, IQ3 choice depending on imatrix at `:644`). User regex overrides via `--tensor-type` (`src/llama-quant.cpp:203-211`, `tools/quantize/quantize.cpp:123-136`).

### 4.5 CPU dot-product kernels

- **vec_dot path**: weights stay in their block format. The activation row is first quantized to `vec_dot_type` (Q8_0 for Q4_0/Q8_0, Q8_K for K-quants) with `from_float` (`ggml/src/ggml-cpu/ggml-cpu.c:1583-1584`, `1624-1659` for MUL_MAT_ID). Then an integer dot product runs per block, e.g. `ggml_vec_dot_q4_0_q8_0` (`ggml/src/ggml-cpu/arch/x86/quants.c:765`) and `ggml_vec_dot_q4_K_q8_K` (`:2102`), with per-arch SIMD in `ggml/src/ggml-cpu/arch/{x86,arm,...}`. `nrows = 2` on ARM i8mm.
- **llamafile sgemm**: tried first for MUL_MAT (`ggml/src/ggml-cpu/ggml-cpu.c:1323`, `:1405`).
- **Tiled matmul**: for MUL_MAT_ID per expert (`ggml/src/ggml-cpu/ggml-cpu.c:1698-1701`, dir `ggml/src/ggml-cpu/tiled`).
- **Extra buffer types** (`ggml/src/ggml-cpu/ggml-cpu.cpp:42-73`): AMX (needs `__AMX_INT8__` and `__AVX512VNNI__`), SpacemiT RISC-V, KleidiAI (ARM), and CPU_REPACK. A tensor in such a buffer is re-laid-out at `set_tensor` time (`ggml/src/ggml-cpu/repack.cpp:5155-5161`). Repack layouts are templates like `tensor_traits<block_q4_0, 8, 8, Q8_0>` = "Q4_0 8x8", plus 4x4, 4x8, 16x1 variants for Q4_0, Q4_K, Q6_K, Q2_K, IQ4_NL, Q8_0 (`ggml/src/ggml-cpu/repack.cpp:4936-4976`). Selection depends on CPU features: AVX2 or SVE → 8x8, NEON+i8mm → 4x8, NEON+dotprod → 4x4 (`:4980-4997`). The repack buffer supports both 2-D MUL_MAT and 3-D MUL_MAT_ID (`:5196-5229`). Dispatch into extra kernels happens in `ggml_cpu_extra_compute_forward` (`ggml/src/ggml-cpu/ggml-cpu.c:1768`).
- `--repack/--no-repack` (`common/arg.cpp:2394-2400`) maps to `use_extra_bufts`.

**Impact for our goal.** A repack buffer is a normal allocated buffer, not a wrap of the mapping. In `load_tensors`, the "wrap mmap" path needs `is_default_buft` (`src/llama-model.cpp:2033-2035`); an extra buffer type is not the default, so the code allocates RAM (`:2054-2066`) and `load_all_data` copies the full tensor in (`src/llama-model-loader.cpp:1830-1832`). Result: repacked CPU weights are fully resident and are not streamed from disk. For "model > RAM" the user must pass `--no-repack`.

---

## 5. KV cache (Question 4)

### 5.1 Layout

- Class `llama_kv_cache` (`src/llama-kv-cache.cpp:65`). Per layer it allocates `k = [n_embd_k_gqa, kv_size, n_stream]` and `v = [n_embd_v_gqa, kv_size, n_stream]` of type `type_k`/`type_v` (`src/llama-kv-cache.cpp:233-234`). `n_stream = 1` if unified, else `n_seq_max` (`:85`).
- The whole `kv_size` (= `n_ctx_seq`) is allocated up front. There is no on-demand growth.
- `get_k` returns a 4-D view `[head_dim, n_head_kv, n_kv, n_stream]` (`:1421-1439`). `get_v` returns a normal view when `v_trans == false`, or a transposed view when `v_trans == true` (`:1441-1470`). `v_trans = !flash_attn` (`src/llama-model.cpp:2995`).
- Writes use `ggml_set_rows` with per-token cell indices (`cpy_k` at `:1497-1529`, `cpy_v` at `:1532-1585`). So token positions do not need to be contiguous.
- The number of cells read per step (`n_kv`) is padded to 256 and limited to the used range (`:1317-1322`, `:1400-1418`).

### 5.2 Cells and the "unified/paged" design

- Each stream has a `llama_kv_cells` array (`src/llama-kv-cells.h:68`): per cell a position, a sequence-id bitset, and a pending RoPE shift. `seq_pos` keeps per-sequence position sets.
- `find_slot` (`src/llama-kv-cache.cpp:950`) finds free cells for a ubatch. Cells can be scattered (non-contiguous), and sequences can share cells (`seq_cp`).
- `--kv-unified` (`common/arg.cpp:1696`, default false in `common/common.h:594`) uses one stream shared by all sequences. Without it, each sequence has its own stream (`n_stream = n_seq_max`).
- This is a cell-level slot allocator inside one fixed buffer. It is **not** a paged block-table design like vLLM PagedAttention: there are no pages that can be allocated, freed or swapped independently. Not found: KV paging to host or disk.
- Variants: `llama_kv_cache_iswa` (sliding-window + full layers; `--swa-full` at `common/arg.cpp:1661`), hybrid caches with recurrent state (`src/llama-memory-hybrid*.cpp`, `src/llama-memory-recurrent.cpp`), and model-specific caches (`llama-kv-cache-dsa*.cpp`, `-dsv4`, `-msa`).

### 5.3 KV types

- `-ctk/--cache-type-k`, `-ctv/--cache-type-v` (`common/arg.cpp:2409-2432`). Allowed: F32, F16, BF16, F8_E4M3, Q8_0, Q4_0, Q4_1, IQ4_NL, Q5_0, Q5_1 (`common/arg.cpp:281-292`). Default F16 (`common/common.h:609-610`).
- A quantized V cache needs flash attention (`src/llama-context.cpp:4551-4557`), because the non-FA path reads V transposed.

### 5.4 Flash attention

- `-fa/--flash-attn on|off|auto` (`common/arg.cpp:1726`). `auto` runs a probe and enables FA if the backends support it (`src/llama-context.cpp:323-324`, `708-711`).
- CPU kernel: `ggml_compute_forward_flash_attn_ext` (`ggml/src/ggml-cpu/ops.cpp:9543`). The scheduler does not use the small sinks tensor to choose the backend for FA (`ggml/src/ggml-backend.cpp:1375-1376`).

### 5.5 Context shift

- `--context-shift` (`common/arg.cpp:1712`, default off at `common/common.h:592`).
- Server logic: keep `n_keep` tokens, discard `n_discard` (default half of the rest), then `seq_rm` + `seq_add(-n_discard)` (`tools/server/server-context.cpp:3810-3825`).
- `seq_add` records a pending shift per cell (`src/llama-kv-cache.cpp:628`). On the next update, if `get_has_shift()` (`:1265`) is true, `build_graph_shift` (`:2202`) re-applies RoPE to the cached K with `build_rope_shift` (`:2114`) — run at `:924`.

### 5.6 KV offload

- `-kvo/--kv-offload`, `-nkvo/--no-kv-offload` (`common/arg.cpp:2385-2392`) set `offload_kqv` (`src/llama-context.cpp:208`).
- If `offload_kqv` is true, each layer's K/V go to the buffer type of that layer's device; else to the CPU buffer type (`src/llama-kv-cache.cpp:214-219`). Because the KV tensors are pre-allocated, the scheduler runs the attention ops that write/read them on their backend (`ggml/src/ggml-backend.cpp:1340-1344`).
- So `--no-kv-offload` keeps the KV in RAM and computes attention on the CPU. This saves VRAM but adds CPU attention cost. There is no partial (some layers) KV offload flag; KV follows the layer's device.

---

## 6. MoE support (Question 5)

### 6.1 Routing in the graph

`llm_graph_context::build_moe_ffn` (`src/llama-graph.cpp:2252`):

1. Router logits → probabilities (`ffn_moe_probs`, `:2323`), optional bias for selection (`:2330`, `:2341`).
2. Optional group-limited routing (DeepSeek style): top-2 per group summed, top `n_group_used` groups kept, others masked to -inf (`:2346-2366`).
3. `selected_experts = ggml_argsort_top_k(selection_probs, n_expert_used)` → I32 `[n_expert_used, n_tokens]` (`:2372`).
4. Weights by `get_rows` on the probs, optional softmax / normalization / scaling (`:2387-2419`).
5. Expert matmuls with `ggml_mul_mat_id` through `build_lora_mm_id` (`:1761-1773`): `up`, `gate` (or merged `gate_up`), activation, `down` (`:2451-2597`).
6. Weighted sum of the `n_expert_used` outputs (`:2620-2665`).

`ggml_mul_mat_id(as, b, ids)` (`ggml/src/ggml.c:3458-3482`): `as` is 3-D, one matrix per expert; `ids` is `[n_expert_used, n_tokens]`; the result is `[n_out, n_expert_used, n_tokens]`.

### 6.2 Are only active experts touched?

- **CPU**: Yes. `ggml_compute_forward_mul_mat_id` (`ggml/src/ggml-cpu/ggml-cpu.c:1566`) groups (token, slot) pairs by expert (`:1661-1681`), then loops over all experts and `continue`s when the row count is 0 (`:1691-1696`). For a used expert it reads only `src0->data + cur_a * nb02` (`:1703`). An id of -1 writes zeros (`:1672-1675`). With mmap, only the pages of the selected experts are faulted in.
- **GPU with experts in VRAM**: the CUDA kernels read only the selected experts (the CUDA MMQ/MMVQ id kernels were not traced line by line in this study).
- **GPU computing CPU-resident experts (large batch)**: `llama_context::sched_copy_experts` (`src/llama-context.cpp:3164-3243`) is the scheduler copy callback. When the first node of the split is the `MUL_MAT_ID` that reads the host weight, it reads the ids from the device (sync, `:3194-3196`), marks used experts, and uploads only ranges of consecutive used experts with `ggml_backend_tensor_set_async` (`:3209-3240`). So prefill also moves only the used experts. For CUDA this goes on the compute stream (`ggml/src/ggml-cuda/ggml-cuda.cu:3082-3089`).

### 6.3 The MoE cache (`--moe-cache-mib`)

Files: `src/llama-moe-cache.h`, `src/llama-moe-cache.cpp`; flags at `common/arg.cpp:2749-2768`; API field `include/llama.h:400` ("[EXPERIMENTAL]").

- **Scope**: only layers whose expert tensors are all in host weight buffers (`src/llama-moe-cache.cpp:421-444`), on GPU/iGPU devices (`:403-413`). Not with `--split-mode tensor` (`:414-416`).
- **Storage**: one "bank" per expert tensor per group of same-layout layers, in VRAM: `[ne0, ne1, n_slots + n_zero + 1]` (`:553-558`). Layers in a group share slots and one LRU (`:334-343`). Budget split by free VRAM or `--tensor-split`, then by host bytes (`:462-519`). Each layer has a host-side `slot_map` I32 `[n_expert]` (`:569-572`).
- **Graph integration**: `build_moe_cache_slots` (`src/llama-graph.cpp:2672-2730`) adds `slots = get_rows(slot_map, selected_ids)`, pinned to the cache device. Because `slot_map` is a host weight, the scheduler starts a new split there and calls the copy callback for it (`:2710-2716`). The expert matmuls then run on the bank views with `slots` as ids (`src/llama-graph.cpp:1772-1773`).
- **Miss handling** (`llama_moe_cache::impl::copy`, `src/llama-moe-cache.cpp:761-856`): reads selected ids from the device and **synchronizes** (`:798-800`), runs `plan()` to give each selected expert a slot (`:814`), uploads missing experts with `ggml_backend_tensor_set_async` from `w->data + expert*expert_size` (host memory, possibly the mmap) and merges consecutive experts into one upload (`:829-844`), then uploads the slot map (`:853`).
- **Policy** (`moe_cache_lru`, `:25-287`): one LRU list per frequency class; "CRF" = use count with a half-life of 128 tokens (`:26-27`); class = floor(log2(CRF)), 8 classes; victim = the LRU slot of each class with the max `age / 2^class` (`:166-189`). Experts selected by the current plan are not evicted (`:149-158`).
- **CPU share**: `--moe-cache-cpu-share F` (default 0, `common/common.h:613`). Once the cache is full, a share of the misses with the lowest CRF is computed on the CPU instead of uploaded (`:242-270`, `:806-812`). The GPU computes those slots as zero ("zero slots"), and the graph adds a CPU `mul_mat_id` with a `cpu_map` where non-CPU experts are -1 (`src/llama-graph.cpp:2606-2613`, `src/llama-moe-cache.cpp:817-827`).
- **Limits**: used only for ubatches ≤ 32 tokens and when distinct experts fit (`:368`, `:687-699`). Large batches only read the cache (`copy_experts`, `:726-759`) and do not insert.
- **Instrumentation**: hit/miss/upload stats logged at exit (`:858-872`); `LLAMA_MOE_CACHE_TRACE=<file>` writes a binary trace of every lookup for an offline simulator (`:390-391`, `:494-500`, `:642-678`).

### 6.4 What this means for streaming experts from disk via mmap

- **CPU decode path** (`--cpu-moe`, mmap on, `--no-repack`): each token reads `n_expert_used` experts × 3 matrices × n_layer from the page cache. A miss in the page cache is a synchronous page fault inside the matmul thread. There is no read-ahead driven by routing (`llama_prefetch` is not called for experts; see Section 2.6). The OS sees random 4 KiB faults, partly mitigated by kernel readahead, which the `POSIX_FADV_SEQUENTIAL` hint makes larger on Linux (`src/llama-mmap.cpp:718`).
- **GPU MoE cache path**: a miss upload reads host memory with `cudaMemcpyAsync` from the mapping (`src/llama-moe-cache.cpp:840`). If that memory is pageable (mmap), the driver stages it, and a page-cache miss becomes a disk read inside the copy. The router result is needed first, and the host syncs (`:799-800`), so the upload cannot start before the router of that layer has finished.
- **Positive**: experts are contiguous in the file (one `nb[2]` slice), so a future prefetcher can issue one range per expert. The router of layer L is computed before the experts of layer L, so ids are known one step early; the next layer's ids are not known.

---

## 7. Speculative decoding and other features for big models (Question 6)

### 7.1 Speculative decoding

- Types (`common/common.h:176-189`): `DRAFT_SIMPLE` (separate draft model), `DRAFT_EAGLE3`, `DRAFT_MTP` (multi-token prediction head in the model), `DRAFT_DFLASH`, `DRAFT_DSPARK`, and model-free n-gram types `NGRAM_SIMPLE`, `NGRAM_MAP_K`, `NGRAM_MAP_K4V`, `NGRAM_MOD`, `NGRAM_CACHE`. Implementation: `common/speculative.cpp`.
- Flags: `--spec-type` (`common/arg.cpp:4311`), `-md/--spec-draft-model` (`:4303`), `--draft-max` (`:4449`), `--spec-draft-n-max` (`:4177`), draft device / ngl (`:4275`, `:4284`), draft MoE-on-CPU (`:4159-4175`). Sidecar drafts can be discovered from the repo or GGUF metadata (`common/arg.cpp:543-620`).
- For our goal: speculative decoding turns one decode token into a small batch of K+1 tokens verified in one pass. Weights (and selected experts) are read once for the batch. When decode is bound by weight read bandwidth (disk or PCIe), this is the main speed lever. Note the batch is usually < 32, so op offload does not trigger, and CPU layers stay on the CPU. With MoE, a batch of K+1 tokens selects more distinct experts, so the read cost per pass grows.

### 7.2 RPC backend (multi-machine)

- `--rpc host:port,...` (`common/arg.cpp:2652-2661`) registers remote devices. Server: `tools/rpc/rpc-server.cpp`. Protocol: `ggml/src/ggml-rpc/ggml-rpc.cpp:59-82` (alloc buffer, set/get tensor, graph compute, allreduce commands for tensor split).
- `RPC_CMD_SET_TENSOR_HASH` sends a hash first; if the server has the data in its local cache, the payload is skipped (`ggml/src/ggml-rpc/ggml-rpc.cpp:874-885`). The server cache directory is `LLAMA_CACHE` or the OS cache dir (`tools/rpc/rpc-server.cpp:48-61`).
- An RPC device is a normal device in the layer split, so several small machines can hold layers of one model (pipeline split by `--tensor-split`).

### 7.3 Split modes, op offload, other levers

- `--split-mode layer|row|tensor` (Section 3.1). `--main-gpu` (`common/arg.cpp:2851`).
- `--op-offload` for prefill of CPU-resident weights on the GPU (Section 3.2).
- `--no-host` skips the pinned host buffer type in the CPU list (`common/arg.cpp:2402-2407`).
- `GGML_CUDA_ENABLE_UNIFIED_MEMORY` env: CUDA uses `cudaMallocManaged`, which lets VRAM allocations exceed VRAM and page to host (`ggml/src/ggml-cuda/ggml-cuda.cu:159-162`, `194-198`). This is driver-level oversubscription, not llama.cpp streaming. On Windows, managed memory oversubscription is not supported by CUDA (comment for HIP fallback at `:208`; the CUDA Windows behavior was not verified in this code).
- Pipeline parallelism with `n_copies = 4` when multiple GPUs are used (`ggml/src/ggml-backend.cpp:2323`, `src/llama-context.cpp:576-606`).
- Graph reuse across ubatches with the same topology (`src/llama-context.cpp:1840-1850`) cuts per-token CPU overhead.

---

## 8. Data flow for one decode token

Setup: MoE model, `-ngl all --cpu-moe --moe-cache-mib 0`, mmap on, `--no-repack`, FA on, KV on GPU. One token (`n_tokens = 1`).

```
llama_decode()                                   src/llama-context.cpp:5136 -> decode() :2149
  |
  +-- memory init_batch / find_slot (KV cells)    src/llama-kv-cache.cpp:950
  +-- process_ubatch()                            src/llama-context.cpp:1826
  |     mctx->apply()  (pending K shift? -> build_graph_shift, kv-cache.cpp:2202)
  |     reuse previous graph if topology same     :1840
  |     else build_graph() + sched_alloc_graph    :1860, :1871
  |     res->set_inputs(ubatch)                   :1885  (token ids, positions, KV idxs, mask)
  |     graph_compute()                           :3132 -> ggml_backend_sched_graph_compute_async
  |
  v
ggml_backend_sched_compute_splits()              ggml/src/ggml-backend.cpp:2210
  split 0 [CPU]  get_rows(token_embd) from mmap  -> 1 row read (input layer always CPU,
                                                    src/llama-model.cpp:1780-1782)
  for each layer L:
   split [GPU]  copy input hidden state H2D (small)
                attn_norm, Q/K/V matmul (VRAM weights)
                set_rows -> KV cache (VRAM)      kv-cache.cpp:1529
                flash_attn_ext over n_kv cells   (n_kv padded to 256)
                o-proj, residual, ffn_norm
                router matmul, softmax, argsort_top_k -> ids[n_expert_used]   llama-graph.cpp:2372
                shared expert (if any; VRAM)
   split [CPU]  copy hidden state + ids D2H (sync points, ggml-backend.cpp:2097-2148)
                mul_mat_id(up/gate/down, ids): for each selected expert e:
                    read W[e] slice from mmap  -> page faults if not in page cache
                    quantize activation to Q8_x, vec_dot per block
                ggml-cpu.c:1566-1703
                weighted sum of experts
   split [GPU]  copy result H2D, residual add
  final split [GPU] output_norm, lm_head (VRAM), copy logits D2H
  |
  v
sampler (CPU) -> next token
```

Bytes per token, rough: all VRAM weights read once from VRAM; per layer `n_expert_used × 3 × expert_matrix_bytes` read from RAM/page cache, or from disk on a page-cache miss; plus 3–4 small PCIe transfers per layer. With the MoE cache on, the CPU split for layer L is replaced by: GPU split ends after the router; the copy callback syncs, plans slots, uploads missing experts H2D, then the GPU computes the experts from the bank (`src/llama-moe-cache.cpp:761-856`).

---

## 9. Weak points for our goal (Question 7)

Each point is based on the code paths cited above. "Not found" means a search did not find any implementation.

1. **No expert prefetch.** `llama_prefetch` / `llama_prefetch_rows` exist but are called only for per-layer token embeddings (`src/models/gemma4.cpp:478`, `src/models/deepseek41.cpp:316`, `src/models/qwen4exp.cpp:1359`). Not found: any call for expert tensors, for the next layer, or driven by router output.
2. **MoE cache is reactive and synchronous.** A miss upload starts only after a device-to-host read of ids and `ggml_backend_synchronize` (`src/llama-moe-cache.cpp:798-800`). Uploads use the same CUDA stream as compute (`ggml/src/ggml-cuda/ggml-cuda.cu:3088`), so they do not overlap with compute of the same backend. No prediction of next-layer experts. Inserts only for ubatch ≤ 32 (`src/llama-moe-cache.cpp:368`, `:694`).
3. **MoE cache needs a GPU.** It throws without a GPU backend (`src/llama-moe-cache.cpp:411-413`). For the 0-VRAM case, there is no engine-level expert cache in RAM; the OS page cache is the only cache.
4. **No layer streaming from disk beyond OS paging.** The CPU path depends on demand page faults. Not found: an engine-managed RAM weight pool with explicit read-ahead, eviction, or pinning of hot layers.
5. **No async, layer-ahead weight streaming to the GPU.** Layers are statically assigned to a device at load (`src/llama-model.cpp:1766-1791`). Op offload copies a host weight per op only for batch ≥ 32 (`ggml/src/ggml-cuda/ggml-cuda.cu:189-192`, `6705-6709`). For dense host weights, the host→CUDA copy falls back to a synchronous copy (`ggml/src/ggml-backend.cpp:2136-2148` with `ggml-cuda.cu:3126-3132`). The scheduler input prefetch explicitly ignores weights (`ggml/src/ggml-backend.cpp:2175-2179`). Not found: double-buffered VRAM slots for layer N+1 while layer N computes.
6. **Load-time prefetch reads the whole file.** `MAP_POPULATE` + `WILLNEED` on Linux and `PrefetchVirtualMemory` over the whole mapping on Windows (`src/llama-mmap.cpp:723`, `:743-747`, `:850-857`). For a model larger than RAM, this wastes load time and evicts useful pages. There is no flag to turn off only the prefetch while keeping mmap (prefetch is hard-coded `true` at `src/llama-model.cpp:2000`). Lazy mode only covers tensors that an arch marks with `TENSOR_READ_LAZY`.
7. **Wrong access hint for decode.** `POSIX_FADV_SEQUENTIAL` on the whole file (`src/llama-mmap.cpp:718`), while expert access is random. Only lazy ranges get `MADV_RANDOM`.
8. **Windows-specific gaps.** `unmap_fragment` is a no-op (`src/llama-mmap.cpp:868-871`). There is no `OfferVirtualMemory`/`DiscardVirtualMemory`/working-set trimming of cold expert pages. Not found: Windows-specific eviction control.
9. **Repacking disables mmap streaming.** Extra buffer types allocate RAM and copy the full tensor (`src/llama-model.cpp:2033-2035`, `2054-2066`; `src/llama-model-loader.cpp:1830-1832`). Default is repack enabled (`include/llama.h:354`, default `true` at `src/llama-model.cpp:3556`). The only signal is a log warning (`src/llama-model-loader.cpp:1341`).
10. **Static, pre-sized KV cache.** Full `kv_size` per layer is allocated at context creation (`src/llama-kv-cache.cpp:233-234`). Not found: KV paging/offload to disk, or per-layer split of KV between GPU and CPU independent of the layer's device. KV quantization is limited to ≥ 4-bit types (`common/arg.cpp:281-292`).
11. **No mixed-precision "on the fly" for streamed weights.** The format on disk is the format in RAM and in VRAM. Not found: a lower-bit copy for cold experts and a higher-bit copy for hot experts.
12. **mlock is all-or-nothing.** It locks whole CPU tensors (`src/llama-model-loader.cpp:1821-1825`). Not found: locking of only hot experts or hot layers.
13. **Single-token CPU sync points per layer.** With CPU experts, each MoE layer needs a GPU→CPU→GPU round trip (scheduler splits). Pipeline parallel (`n_copies`) needs multiple GPUs (`src/llama-context.cpp:576-606`). Not found: overlap of the CPU expert compute of layer L with GPU attention of layer L+1 (a data dependency also prevents this without speculation).
14. **`--fit` does not plan for RAM < model.** It optimizes VRAM use and context size (`common/fit.cpp:460`, `:575-839`). It does not estimate disk read cost per token.

---

## 10. Extension points (Question 8)

### 10(a). Smarter expert cache / prefetcher

Best hooks, in order of effort:

1. **Router-driven host prefetch (CPU / page cache).** Add a call to `llama_prefetch` for the selected expert slices as soon as the ids are known. Candidate places:
   - In `llama_context::sched_copy_experts` (`src/llama-context.cpp:3164`) — it already reads the ids for host experts on large batches.
   - A new eval/copy-callback hook at the split boundary after `ffn_moe_topk` (`src/llama-graph.cpp:2375` names it with `cb`). The scheduler's copy callback (`ggml/src/ggml-backend.cpp:2131`) runs before the CPU split that reads the host weights, so a prefetch there can issue `PrefetchVirtualMemory` for `w->data + e*nb[2]` of up/gate/down before the matmul starts.
   - For cross-layer prediction (layer L+1 experts guessed from layer L hidden state or from history), the prefetch must be issued from a background thread; `llama_prefetch` (`src/llama-mmap.cpp:1067`) is already thread-safe in the sense that it only takes address ranges.
2. **Extend the GPU MoE cache.** `src/llama-moe-cache.cpp` already has the right structure:
   - Policy: replace `moe_cache_lru::plan` / `victim` (`:166-287`).
   - Async/predictive fill: add a "prefill" call that uploads predicted experts for layer L+1 on a separate stream before `copy()` for L+1 runs. `copy()` (`:761`) would then find them as hits.
   - Remove the sync at `:799-800` by computing slot assignment on device, or by overlapping it with other work.
   - The trace (`LLAMA_MOE_CACHE_TRACE`, `:494-500`, format at `:642-678`) is a ready input for an offline simulator of new policies.
3. **RAM-level expert cache without GPU.** Add a host-side analog: a pool of anonymous (or locked) RAM slots for hot experts, with the slot-map mechanism reused, and the `cpu_map` path (`src/llama-graph.cpp:2606-2613`) pointing the CPU `mul_mat_id` at the pool. The `-1` = zero contract in `ggml-cpu.c:1672-1675` already supports partial expert sets.
4. **mmap hints.** Change `llama_mmap::impl` (`src/llama-mmap.cpp:712-759`, Windows `:819-866`) to: no whole-file prefetch for big models; `MADV_RANDOM` on expert ranges; on Windows, optional `OfferVirtualMemory` for cold ranges. Expert ranges can be marked like lazy ranges via `lazy_read` (`src/llama-model-loader.cpp:1183-1209`) and a new tensor flag in the arch loaders.

### 10(b). New quant type

Steps, all needed:

1. Add the enum value in `ggml/include/ggml.h` (before `GGML_TYPE_COUNT`, `:436`).
2. Define the block struct + `static_assert` in `ggml/src/ggml-common.h` (pattern at `:194-199`).
3. Add reference quantize/dequantize and an imatrix-aware `quantize_<type>` in `ggml/src/ggml-quants.c` / `.h` (pattern Q4_K at `:1563-1600`).
4. Register traits in `type_traits[]` (`ggml/src/ggml.c:639`) and a case in `ggml_quantize_chunk` (`ggml/src/ggml.c:8246-8270`); if it needs an imatrix, add it to `ggml_quantize_requires_imatrix` (`:8212`).
5. CPU kernels: `type_traits_cpu[]` entry with `from_float`, `vec_dot`, `vec_dot_type` (`ggml/src/ggml-cpu/ggml-cpu.c:218`), SIMD in `ggml/src/ggml-cpu/arch/*/quants.c`, generic in `ggml/src/ggml-cpu/quants.c`. Optional repack traits in `ggml/src/ggml-cpu/repack.cpp:4936-4997`.
6. GPU kernels: CUDA dequantize / MMQ / MMVQ / supports_op in `ggml/src/ggml-cuda/` (and other backends as needed).
7. llama side: a `LLAMA_FTYPE_*` value and mapping, type selection in `src/llama-quant.cpp` (`llama_tensor_get_type_impl` at `:507`), CLI name in `tools/quantize/quantize.cpp`.
8. Python: `gguf-py` constants for the type id and block size, so that converters and readers know it.

### 10(c). Layer-by-layer GPU streaming

There is no existing mechanism to extend directly. The candidate hook points:

1. **Placement.** Keep the layer weights in host buffers (CPU list, `src/llama-model.cpp:1229-1289`, possibly the pinned GPU host buffer type for DMA speed) and mark them as "streamable".
2. **Scheduler.** Today, an op on a host weight runs on the GPU only when `offload_op` says yes (`ggml/src/ggml-backend.cpp:1387-1393`), and then the weight is copied in `ggml_backend_sched_copy_input` (`:2097-2148`). For streaming:
   - Force those ops to the GPU for all batch sizes (a new backend/device policy instead of the `>= 32` rule at `ggml/src/ggml-cuda/ggml-cuda.cu:6705-6709`).
   - Implement a host→device `cpy_tensor_async` path from pinned memory on a separate copy stream with events (today it returns false, `ggml-cuda.cu:3126-3132`).
   - Extend `ggml_backend_sched_prefetch_inputs` (`ggml/src/ggml-backend.cpp:2152-2208`) to also start host-weight copies for split N+1 (and N+2) while split N computes, into a ring of reserved VRAM slots. The current code deliberately skips weights (`:2175-2179`).
3. **Copy callback.** `ggml_backend_sched_set_copy_callback` (`ggml/src/ggml-backend.cpp:2506`) with `llama_context::sched_copy_experts` (`src/llama-context.cpp:3164`) is the existing place where llama.cpp decides how a host weight reaches the device. A streaming manager can be added there with low changes to ggml.
4. **Graph reuse.** The graph is reused across tokens (`src/llama-context.cpp:1840`), so a static streaming schedule per graph is possible.
5. **Disk tier.** Pair 10(c) with direct I/O reads (`llama_file` with `FILE_FLAG_NO_BUFFERING` / `O_DIRECT`, `src/llama-mmap.cpp:106-114`, `:425`) into pinned buffers, as the no-mmap loader already does at load time (`src/llama-model-loader.cpp:1846-1897`). That code is the best template for a runtime disk→pinned→VRAM pipeline.

---

## 11. Quick reference: flags relevant to small-RAM / small-VRAM

| Flag | Effect | Source |
|---|---|---|
| `-lm/--load-mode auto\|none\|mmap\|mlock\|mmap+mlock\|dio` | Load path | `common/arg.cpp:2663` |
| `-lzm/--lazy-mode on\|auto\|off` | On-demand rows for marked tensors | `common/arg.cpp:2682` |
| `-ngl N\|auto\|all` | Last N layers on GPU | `common/arg.cpp:2782` |
| `-ot regex=buft` | Per-tensor placement | `common/arg.cpp:2726` |
| `-cmoe`, `-ncmoe N` | Experts (of first N layers) on CPU | `common/arg.cpp:2732`, `:2739` |
| `--moe-cache-mib N` | VRAM cache for CPU experts | `common/arg.cpp:2749` |
| `--moe-cache-cpu-share F` | Share of misses computed on CPU | `common/arg.cpp:2759` |
| `-nr/--no-repack` | Keep CPU weights in mmap | `common/arg.cpp:2394` |
| `--no-host` | Skip pinned host buffer | `common/arg.cpp:2402` |
| `--op-offload/--no-op-offload` | GPU compute of host weights for big batches | `common/arg.cpp:2940` |
| `-nkvo` | KV + attention on CPU | `common/arg.cpp:2387` |
| `-ctk/-ctv` | KV quant type | `common/arg.cpp:2409` |
| `-fa on\|off\|auto` | Flash attention | `common/arg.cpp:1726` |
| `-sm`, `-ts`, `-mg` | Multi-GPU split | `common/arg.cpp:2800`, `:2824`, `:2851` |
| `--rpc` | Remote devices | `common/arg.cpp:2652` |
| `--fit`, `--fit-target` | Auto ngl/ncmoe/ctx | `common/arg.cpp:2861` |
| `--spec-type`, `-md`, `--draft-max` | Speculative decoding | `common/arg.cpp:4311`, `:4303`, `:4449` |
| env `GGML_OP_OFFLOAD_MIN_BATCH` | Op offload threshold (default 32) | `ggml/src/ggml-cuda/ggml-cuda.cu:190` |
| env `GGML_CUDA_ENABLE_UNIFIED_MEMORY` | Managed VRAM allocations | `ggml/src/ggml-cuda/ggml-cuda.cu:160` |
| env `LLAMA_MOE_CACHE_TRACE` | Binary trace of MoE cache lookups | `src/llama-moe-cache.cpp:494` |

---

## 12. Items not verified

- CUDA `mul_mat_id` kernels (MMQ/MMVQ id variants) were not read line by line. The statement "GPU reads only selected experts" for VRAM-resident experts is based on the op contract, not on kernel review.
- The exact Windows CUDA behavior of `cudaMallocManaged` oversubscription was not checked in this code.
- Whether `select_weight_buft` picks the pinned GPU host buffer type for `--cpu-moe` experts without mmap depends on the GPU backend's `supports_op` for that buffer type. This was not traced. With mmap on, the code replaces a host buffer type by the plain CPU type (`src/llama-model-loader.cpp:1364-1372`).
