# E. Low-memory inference engines: colibri, MemVanta, reame, rai

Analyst E. Scope: four small, new inference engines that claim low-memory operation.
Project goal: run 70B to 700B+ models on 8-32 GB RAM and 0-12 GB VRAM.

Method: I read the README and docs of each repo first, then the core source files.
I did not build or run code. All paths are relative to `<repo>\repos\`.
A citation `file:N` or `file:N-M` gives line numbers in the checkout as read on 2026-11-12.
Where I state an inference that the repo does not state, I mark it **[inference]**.
Where a claim has no evidence in the repo, I mark it **[no evidence in repo]**.

Repo snapshots (each checkout has one squashed commit, so commit counts are not available):

| Repo | Last commit | Code size (approx.) |
|---|---|---|
| colibri | 2026-10-31, `0a5504a` "Merge pull request #2530" | ~215k lines C/C++/CUDA/ObjC in `c/` + ~26k lines Python; `c/colibri.c` alone is 15,826 lines |
| MemVanta | 2026-10-09, `66701ca` | ~5.3k lines C++ (`src/` + `include/`) |
| reame | 2026-10-12, `4f0da5e` | ~7.4k lines C++ (`src/` + `include/`) + 6.0k lines tests; the ternary engine is **not in the repo** (see section 3) |
| rai | 2026-09-19, `10e90f7` | ~29.3k lines Rust (`rai-infer`) + 4.0k (`rai-compress`) |

---

## 1. colibri (JustVugg/colibri)

colibri is the only one of the four that addresses our goal directly: it runs MoE models of
300B-2.8T parameters with the routed experts on disk. This section is the longest.

### 1.1 Architecture

- One C engine file per model family over shared single headers (`README.md:582-584`,
  `README.md:810-813`). GLM-5.2/5.3 and Mistral-Large-4 run in `c/colibri.c`.
  Other engines: `glm53.c`, `inkling.c`, `kimi_k3.c`, `deepseek_v4.c`, `deepseek_v41.c`,
  `mimo.c`, `qwen38.c`, `qwen36.c`, `olmoe.c` (`README.md:768-782`).
- Shared headers: `st.h` (safetensors reader), `quant.h` (container decoders and float
  kernels), `idot.h` (integer dot kernels), `expert_ffn.h` / `expert_store.h` (routed-expert
  kernel and streaming store interface), `tier.h` (LFRU scoring), `kv_persist.h`,
  `kv_prefix.h`, `route_trace.h` (`README.md:784-789`).
- Optional GPU backends: Vulkan (`backend_vulkan.c`, 7,887 lines), CUDA (`backend_cuda.cu`,
  `cu_tier.h`, `cuda_chain.cu`), Metal (`backend_metal.mm`), AMD XDNA (`backend_xdna.c`).
- Python only for setup, launcher (`coli`), converters and the HTTP gateway
  (`openai_server.py`, 10,449 lines). No Python and no BLAS at runtime (`README.md:582-584`).

GLM-5.2 shape as the engine uses it: hidden D=6144, 64 heads, q_lora 2048, kv_lora 512,
qk_nope 192, qk_rope 64, v_head 256, 256 routed experts, top-8, 1 shared expert,
moe_inter 2048 (these are the constants of the fused Metal gate, `c/colibri.c:8464-8467`).
78 layers (`docs/METAL-M1ULTRA-FMT2-REPORT.md:9`), the first 3 dense, 75 MoE
(`docs/benchmarks.md:18`, "75 layers x 8 experts"), plus an MTP layer (index 78).

**Data flow for one decode token (CPU path, `layer_forward_rows`, `c/colibri.c:8423-8527`):**

1. Embedding row lookup (`embed_row`, `c/colibri.c:3149`).
2. For each of the 78 layers:
   1. RMSNorm, then MLA attention against the compressed KV latent
      (`attention_rows`; `c/colibri.c:8516-8518`).
   2. Optional `PILOT`: run layer L+1's router on the post-attention state of layer L and
      queue predicted experts for a background I/O thread (`pilot_prefetch`,
      `c/colibri.c:8213-8270`, called at `c/colibri.c:8519`).
   3. RMSNorm, then `moe()` (`c/colibri.c:6731`) for sparse layers, `dense_mlp` for the 3
      dense layers.
3. `moe()` works in phases:
   - **Phase A, routing** (`c/colibri.c:6762-6950`): one batched router matmul, sigmoid +
     bias (noaux_tc), top-K select. Optional `CACHE_ROUTE` (prefer resident experts inside
     the top-M window, `c/colibri.c:6834-6880`), `TOPP` (adaptive top-p,
     `c/colibri.c:6907-6913`). Every selection updates `eusage`, `eheat` (frequency) and
     `elast` (recency clock) (`c/colibri.c:6929-6941`).
   - **Phase B, union** of all experts that the S rows of the batch route to
     (`c/colibri.c:6976-6982`). With S=1 this is the K=8 experts. With a speculative verify
     of k+1 rows, the union is larger.
   - Optional lossy trims: `EXPERT_BUDGET` (keep hits, keep top-weight misses,
     `c/colibri.c:6995-7083`) and `DEGRADE_ZERO` (zero-fill misses below gate weight tau,
     `c/colibri.c:7096-7166`). Both are decode-only (S<=4).
   - **Phase C/D, resolve and compute** in blocks of 64 unique experts
     (`c/colibri.c:7200-7740`): for each expert, look up the pinned set
     (`pin_indexed`), then the per-layer LRU (`ecache_indexed`); a miss gets a working
     slot `ws[q]`. Misses go to the async load pool (`pipe_dispatch`,
     `c/colibri.c:7312`) or a blocking OpenMP-parallel load. Resident experts compute
     while the misses load; each miss is awaited (`pipe_wait`) before its matmul.
   - At the end of each block, the loaded slots are swapped into the LRU
     (`ecache_promote_ws`, `c/colibri.c:7738`, defined at `c/colibri.c:6440-6454`).
   - **Phase E**: shared expert (always resident), weighted sum.
4. Final norm, LM head, sampling. With MTP on, the int8 MTP head drafts tokens and the
   main model verifies them in one batched forward (`mtp_draft`, `c/colibri.c:8989`;
   `spec_decode`, `c/colibri.c:9221`).

### 1.2 Memory strategy

**What stays resident in RAM.**
- The dense trunk: attention matrices, shared experts, embeddings, LM head, the 3 dense
  MLPs, norms and routers. The dense part is int4/int8 "dequant-on-use"; norms, routers and
  biases stay f32 (`c/colibri.c:10-13`). Measured resident size for GLM-5.2: 9.9 GB
  (`docs/benchmarks.md:15`, also the banner at `README.md:36`).
- The pinned expert set (`m->pin[layer]`), loaded at start from usage history
  (`pin_load`, `c/colibri.c:12961`).
- The per-layer LRU expert cache (`m->ecache[layer]`, `m->ecap` slots per layer,
  `c/colibri.c:569`, `c/colibri.c:2869`).
- 64 working slots `ws[]` (shared across layers) for the misses of one block.
- The KV cache (all positions, in RAM; disk only for persistence, see 1.4).

**What streams from disk.** Only routed experts. Each expert's three matrices
(gate, up, down) are contiguous in the container, so one `pread` reads all three
(comment at `c/colibri.c:3376-3378`; contiguity test at `c/colibri.c:3636-3638`). The
per-expert scales (`.qs` tensors) are read with three small buffered `pread`s
(`c/colibri.c:3678-3684`).

**I/O mechanisms (all opt-in or platform-defaulted):**

| Mechanism | Where | Default |
|---|---|---|
| Plain buffered `pread` with short-read and EINTR loop | `pread_full`, `c/colibri.c:3390-3410` | on |
| `O_DIRECT` (`DIRECT=1`): offset and length aligned to 4 KiB, read into a 4 KiB-aligned slab | `c/colibri.c:3641-3659` | off; measured +65% on Strix Halo (`docs/tuning.md:23`) |
| Async load pool (`PIPE=1`): N pthreads (default 8, max 16) claim jobs by CAS on a packed generation/index word; main thread spins or waits on a condvar | `c/colibri.c:4582-4695` | on for Windows, off elsewhere (`c/colibri.c:14312`); "-18% disk service" (`docs/tuning.md:22`) |
| `io_uring` batch (`URING=1`, Linux) | `c/colibri.c:4284-4555` | off |
| `mmap` mode (`COLI_MMAP`): `madvise(MADV_WILLNEED)` + page touch | `c/colibri.c:3500-3535` | off |
| Readahead hint for the next block of 64 (`posix_fadvise WILLNEED` via `expert_prefetch`) | `c/colibri.c:4749-4758`, `c/colibri.c:7319-7327` | on; skipped for weights under `DIRECT=1` |
| `DROP=1`: `POSIX_FADV_DONTNEED` after each read | `c/colibri.c:3693-3698` | off: the page cache acts as a free L2 cache (`c/colibri.c:1529-1531`) |
| Mirror drives (`COLI_MODEL_MIRROR`): a hash of (layer, expert) picks a replica by weighted cuts; under `DIRECT=1` one read can be striped across replicas | `expert_route`, `c/colibri.c:3238-3245`; `mir_pread_striped`, `c/colibri.c:3315` | off; two NVMe on independent controllers measured +37.5% decode (`docs/benchmarks.md:180`) |
| NUMA slab interleave (`COLI_NUMA=1`) | `c/colibri.c:1707-1750` | off |

**Cache policy.**
- **Per-layer LRU** of whole experts. Victim = smallest `used` stamp among non-busy slots;
  an expert that the Vulkan tier already holds is evicted first (`eslot_lru_victim`,
  `c/colibri.c:508-524`). The cache is a swap with the working slot, so no copy happens
  on promotion (`c/colibri.c:6440-6454`).
- **Learned pinned hot set.** The engine writes `.coli_usage` (per-expert routing counts)
  next to the model after every turn and pins the hottest experts in spare RAM at the next
  start (`docs/tuning.md:193-199`; format in `docs/routing-telemetry.md`). AUTOPIN is capped
  so that it does not starve the LRU (`autopin_preserve_lru`, `c/colibri.c:13303-13309`).
- **LFRU live re-pinning** (`--repin N` / `REPIN`): score = `(heat << 8) | recency`, where
  recency contributes at most 255 and one frequency count is worth 256, so frequency
  dominates (`tier_lfru_score`, `c/tier.h:40-43`). Promotion needs 25% + 4-count hysteresis
  and at most four swaps per pass (`c/tier.h:62-64`, `docs/tuning.md:205-207`).
- **Prediction.** `PILOT=1` applies layer L+1's router to layer L's post-attention state.
  The repo states 71.6% recall of the true top-8, against 41.3% for "same experts as the
  previous token" (`docs/tuning.md:257-259`, `c/colibri.c:7920`). `PILOT_REAL=1` does real
  loads into `ecache[L+1]` (+11 pp hit on a big-cache host); `PILOT_TWO=1` adds the shared
  expert output to the prediction (+3% recall) (`docs/tuning.md:260-265`). An eviction guard
  drops a speculative load when the victim is hotter by LFRU score (`c/colibri.c:7960-7970`).
  The docs warn that hint-only PILOT can be net negative on a saturated disk.
- **Cache-aware routing** (`CACHE_ROUTE=1`, opt-in, lossy): always keep the true top-J
  (default 2), fill the other slots with resident experts inside the top-M (default 12)
  (`docs/CACHE_ROUTE.md:1-39`). Measured on DGX Spark: 2.4 to 3.33 tok/s
  (`docs/benchmarks.md:166`). This changes router output, so quality must be checked.
- The **Vulkan/CUDA expert tier** holds the hottest experts in VRAM and computes them on the
  GPU while the CPU computes the rest in the same step (`README.md:300-304`,
  `README.md:384-391`).

**Peak RAM formula (derivable from `cap_for_ram`, `c/colibri.c:13348-13474`).**

```
budget      = RAM_GB, or 0.88 x MemAvailable at boot          (c/colibri.c:13358)
slack       = 1.2 GB (activations, logits)
            + 2.5 GB page-cache reserve                          (c/colibri.c:13396)
            + 64 x eb       (ws[] working slots; eb = widest expert in container)
            + KV pool at max_ctx + transient kv_b reconstruction (c/colibri.c:13394-13397)
cap         = floor((budget - resident_dense - slack) / row_b)   (c/colibri.c:13398)
row_b       = sum over MoE rows (+MTP row) of one expert's real bytes (#856 fix, c/colibri.c:13331)
peak        = resident_dense + pinned + cap x row_b + slack
```

For GLM-5.2 int4-g64 with the int8 MTP head **[inference from the formula and the sizes in 1.3]**:
resident_dense = 9.9 GB; row_b = 75 x 21.2 MB + 1 x ~37.9 MB = ~1.63 GB per cap step;
ws = 64 x ~37.9 MB = ~2.4 GB; KV (f32, ctx 4096) = 4096 x 182 KB = ~0.75 GB.
Fixed part = 9.9 + 1.2 + 2.5 + 2.4 + 0.75 = ~16.8 GB. On a 24-25 GB machine the cache gets
cap = 2-4 experts per layer. The repo states "With 24 GB of RAM the engine auto-caps the
expert cache to 2 slots/layer" (`docs/benchmarks.md:239-241`). The formula agrees.
If cap < 1 and the projected peak is larger than MemAvailable, the engine refuses to start
instead of being OOM-killed (`c/colibri.c:13421-13430`). It also refuses to "raise" the
budget past MemAvailable unless `COLI_RAM_OVERCOMMIT=1` (`coli_clamp_ram_gb`,
`c/colibri.c:13340-13345`).

### 1.3 Weight format and quantization

- Container: safetensors shards. Each quantized tensor `X.weight` has a sidecar `X.weight.qs`
  with f32 scales. The format is inferred from byte arithmetic (weight bytes and scale bytes
  against [O, I]); an optional `__metadata__` stamp confirms it (`docs/FORMATS.md:51-58`,
  `qt_resolve_fmt`, `c/colibri.c:2175`).
- Format ordinals (`docs/FORMATS.md:60-71`):
  - fmt 1 `int8-row`: 1 byte per weight, 1 f32 scale per row.
  - fmt 2 `int4-row`: 2 nibbles per byte, 1 f32 scale per row (~4.0 bpw).
  - fmt 4 `int4-grouped`: 1 f32 scale per group of `gs` inputs. The recommended GLM-5.2
    container is gs=64 (4 + 32/64 = **4.5 bpw**) (`README.md:680-688`).
  - fmt 5 `int3-g64` (3.5 bpw), fmt 6 E8/IQ3 lattice (98-byte super-blocks per 256
    weights, needs a Hadamard rotation of activations), fmt 7 MXFP4 (Vulkan only),
    fmt 8 FP8-e4m3 with 128x128 block scales.
  - `int4-rans256-g0`: lossless rANS entropy coding of int4 bytes, ~0.76 of the packed size
    on GLM-5.2; **tools only, no engine decode path** (`docs/FORMATS.md:71`).
- Conversion: `./coli convert` builds the container from the FP8 release shard by shard,
  without the 756 GB source on disk at once (`README.md:686-688`). The quantizer is
  symmetric absmax round-to-nearest: `s = amax/7`, `q = clip(rint(w/s), -8, 7)`
  (`c/tools/convert_fp8_to_int4.py:249-293`). No GPTQ/AWQ calibration for the experts.
- GGUF: `./coli import` reads GGUF block types from IQ1_S to Q8_0, F16, BF16 and MXFP4
  as-is (`README.md:242-246`).
- Kernels (`c/idot.h:1-14`): activations are quantized once per row to int8 (amax/127);
  weights stay int4 or int8; exact int32 accumulation; one float scale multiply per output.
  ISA paths: AVX2 `maddubs` (`c/idot.h:188-200`), AVX-VNNI and AVX-512 VNNI `vpdpbusd`
  (`c/idot.h:154-170`), NEON `sdot` (`c/idot.h:207-216`), AMX tiles. The repo says all ISA
  paths are bit-identical to the scalar reference. int4 weights are "planarized" in the slab
  after each load (`qt_planarize`, `c/colibri.c:1296`, applied at `c/colibri.c:3712`).
- Quality cost: per-row int4 cost -8.2 pp on an OLMoE fp16-vs-int4 A/B; grouped scales
  recover ~63% of that loss (`docs/benchmarks.md:282-291`). The per-row GLM container
  measured ~9 points worse than g64 and caused looping answers (`README.md:684-686`).

**Expert size (GLM-5.2), computed from the shape:**
- Parameters per expert = 3 x 6144 x 2048 = 37,748,736.
- fmt 2 int4-row: 18,874,368 B weights + (2048+2048+6144) x 4 B scales = **18.92 MB**.
  This matches "~19 MB per int4 expert" (`docs/ENVIRONMENT.md:223`).
- fmt 4 int4-g64: 18,874,368 B + 37,748,736/64 x 4 B = 18,874,368 + 2,359,296 =
  **21.23 MB**.
- Cross-check: 75 x 256 x 21.23 MB = 407.7 GB of routed experts. Add ~10 GB dense and the
  int8 MTP layer (~256 x 37.9 MB = ~9.7 GB) and the result is ~428 GB. The download size is
  429 GB (`README.md:221`). For per-row: 75 x 256 x 18.92 = 363 GB + dense = ~370 GB
  (`docs/benchmarks.md:14`). The numbers are consistent.

### 1.4 KV cache

- MLA compressed KV: per token per layer only the normalized latent (kv_lora=512) and the
  roped key (qk_rope=64) are kept: 576 values instead of 32,768 (`c/colibri.c:548-550`,
  `README.md:568-571`). k_nope and V are rebuilt on the fly with `kv_b`.
- Precision options: f32 (default record format), `KV16` (float16), `KV8` (fp8 e4m3 + per-row
  scale), `KV_TQ` (rotated int4 / PolarQuant) (`c/colibri.c:1669-1681`).
  "GLM-5.3 at 131,072 tokens holds 11.9 GB instead of 23.9" with float16 (`README.md:649-650`).
- Size: 576 x 4 B x 79 layers = 182 KB/token at f32. The repo states ~182 KB/token
  (`docs/tuning.md:340-341`).
- Paging: none. The KV for all positions is in RAM (`Lc`, `Rc` arrays per layer,
  `c/colibri.c:526-536`).
- Disk: `.coli_kv` persistence. Records are appended after every turn; `nrec` is written last
  as the commit step, with `fflush` + fsync (`c/kv_persist.h:1-60`). A conversation reopens
  with zero re-prefill, "validated byte-identical" (`docs/tuning.md:338-344`). `kv_prefix.h`
  reuses a KV prefix across requests, and `park.h` parks a conversation between an agent's
  turns (Hermes Agent second turn 86 s -> 4.9 s, `README.md:644-648`).
- Multi-slot serving: `--kv-slots N` up to 16 (`README.md:521-523`).

### 1.5 Speculative decoding and other speed tricks

- GLM-5.2 int8 MTP head: 2.2-2.8 tokens per forward "when it pays". An int4 MTP head
  collapses to 0-4% acceptance (`docs/tuning.md:277-283`, issue #8).
- A speculation gate (`COLI_SPEC_GATE`) and an acceptance guard decide when to draft
  (`docs/speculative.md:16`). `SPEC_PIN=1` pins all forwards to the S=1 kernel family so
  draft and verify compute the same function (`docs/tuning.md:269-275`).
- **Important negative result for our goal:** on a disk-bound host, MTP raised the number of
  experts read per token from 600 to 1,062 and lowered the hit rate (54.9% -> 40-42%);
  decode was 0.31-0.33 tok/s with MTP and 0.34 tok/s without (`docs/benchmarks.md:181-182`).
  The README also lists "MTP has also measured a 32% loss around 85% expert hit"
  (`README.md:731`). Reason: each verify row routes to its own experts, so the batch union
  grows **[inference consistent with `moe()` phase B]**.
- Prompt-lookup n-gram drafts, grammar-forced drafts (`GRAMMAR=`) for JSON.
- `--topp 0.7`: adaptive expert top-p, "30-40% less disk", lossy (`docs/tuning.md:12`).
  Measured 0.07 -> 0.11 tok/s on a 24 GB WSL box (`docs/benchmarks.md:146-147`).
- `XEXP=1`: one OpenMP region across all experts of a block (`c/colibri.c:7445-7510`).
- OpenMP "hot team" tuning, P-core-only teams on hybrid CPUs (`docs/tuning.md:49-91`).
- Local cluster mode: route experts to other machines over TCP, optional q8 activations
  (`c/colibri.c:3751-3860`, `docs/cluster.md`).
- `coli tune`: a bounded sweep of execution knobs that keeps a winner only if median
  throughput is +3%, hit rate within 0.5 pp, and greedy output byte-identical
  (`docs/tuning.md:132-171`).

### 1.6 Benchmark claims (exact, from the repo)

GLM-5.2 744B, int4 container (`docs/benchmarks.md:7-20`, `docs/benchmarks.md:142-189`):

| Machine | Disk | Config | Claimed |
|---|---|---|---|
| Original dev box: WSL2, 12 cores, 25 GB RAM, NVMe via VHDX | ~1 GB/s | default | 0.05-0.1 tok/s cold; resident 9.9 GB; peak RSS ~20 GB; ~11 GB disk reads/token |
| Core Ultra 7 270K, WSL2, 24 GB | 2.74 GB/s O_DIRECT | default / `--topp 0.7` | 0.07 / 0.11 tok/s; hit 3-4% / 11%; RSS 14.1 / 14.7 GB |
| i5-12600K, native Windows, 32 GB | buffered | int8 MTP, cap ~2/layer | 0.08 tok/s, hit 3.7%, MTP 57% acceptance |
| i5-13600K, 62 GB, 980 PRO, RTX 5070 Ti | 5.90 GB/s O_DIRECT | tuned, `--topp 0.7`, PIN_GB=20 | 0.98 tok/s, hit 54.7%, RSS 42.0 GB |
| i9-12900K, 64 GB, 990 Pro, RTX 3090 | 6.44 GB/s O_DIRECT | MTP=0 | 0.34 tok/s, hit 54.9%, ~9.9 GB read/token |
| Ryzen 9 9950X, 123 GB, PCIe 5 drive | 8.81 GB/s O_DIRECT | default | 0.28 tok/s, hit 57% |
| Threadripper 7965WX, 123 GB, 2 NVMe | 6.70 / 8.05 GB/s | DIRECT=1, CPU only | 0.80 -> 1.10 tok/s (+37.5%) |
| Ryzen AI Max+ 395, 128 GB | - | DIRECT, PIPE, PILOT_REAL, PILOT_TWO | 1.83 tok/s |
| EPYC 9V45 64C, 504 GB, 4x NVMe RAID0 | 17.9 GB/s | cap 16 / cap 256 + NUMA | 2.83 / 6.91 tok/s, hit 97% / 100% |
| 6x RTX 5090, 251 GB | NVMe | full residency | 5.8-6.8 tok/s; 9.0-9.2 with NUMA |

Other models from the README table (`README.md:206-236`): DeepSeek V4 Flash 284B 0.93 tok/s
with 32 GB (Ryzen 7 5800X); Kimi K3 2.8T "about 9.4 s per token, experts read at 6.3 GB/s";
GLM-5.3-Flash 321B "about 20 s per token warm, 44 s cold (6 cores, 25 GB, ordinary disk)".

Reproducibility: strong. The repo ships `c/iobench.c` (measure the disk with parallel 19 MB
random reads), `c/tools/datapoint.py` (machine info + cold/warm/rotating-prompt decode in one
command), a written protocol (`docs/benchmarking.md`), and most rows link to a GitHub issue
with raw logs (`docs/benchmarks.md:36-67`). Rows are community-measured on many machines,
not by one author. The "Back-of-envelope predictions" table is explicitly labelled
"estimates, not measurements" (`docs/benchmarks.md:100-114`).

Skeptical notes:
- The headline "runs a 744B model in ~25 GB RAM" is true, but at 0.05-0.1 tok/s cold. Every
  row above 0.5 tok/s uses 62 GB RAM or more, a GPU tier, or both.
- Hit rates depend on usage history. The repo itself warns that coverage measured on the
  history it came from overstates new prompts: 70% self-coverage fell to 57.7% on a held-out
  session (`docs/tuning.md:245-253`).
- The issue links (#2, #39, #120, ...) are not in the checkout, so I could not inspect those
  raw logs. **[external evidence, not verifiable offline]**
- Quality: 62.5% mean acc_norm on hellaswag/arc/mmlu with n=40 per task
  (`docs/benchmarks.md:282-286`). n=40 is a very small sample.

### 1.7 Code quality and maturity

- Tests: `c/tests/` holds 298 C tests, 297 Python tests, 25 CUDA tests and shell scripts
  (652 files). Every engine is checked in CI against a transformers oracle on a tiny fixture
  (`README.md:567-568`, `c/ref_glm.json`). CI workflow `.github/workflows/ci.yml` is 2,240
  lines. Vulkan paths are checked against CPU tokens on a software driver
  (`README.md:355-358`).
- Activity: v2.1.0 on 2026-10-31, "387 pull requests since v2.0.0, 208 of them from 35
  contributors" (`CHANGELOG.md:8-10`). PR numbers reach #2530.
- Only 3 TODO/FIXME markers in `colibri.c`. Comments mix Italian and English.
- Risks: `colibri.c` is a 15.8k-line file with many global flags (`g_*`), dozens of
  environment knobs and many `#ifdef` backend paths in the hot loop (`moe()` alone spans
  ~1,000 lines, `c/colibri.c:6731-7760`). The format registry exists because two PRs
  claimed the same format ordinal (`docs/FORMATS.md:7-16`). The code is mature in testing
  and measurement discipline, but hard to extract as a library. The `ColiExpertStore`
  interface (`c/expert_store.h:59-69`) is a clean lease-based abstraction, but the GLM engine
  uses its own `ESlot` arrays directly.

### 1.8 Disk bandwidth calculation for GLM-5.2 int4

**Assumptions** (all sizes from 1.3, structure from 1.1):
- 75 MoE layers, top-8 routed experts per layer, shared expert resident.
- Active routed experts per token: 75 x 8 = **600** (matches the measured
  "600 experts/token" with MTP off, `docs/benchmarks.md:182`).
- No speculation (DRAFT=0), no TOPP / EXPERT_BUDGET / DEGRADE_ZERO (exact routing).
- "Hit rate" h = fraction of those 600 expert reads that RAM (pin + LRU + page cache)
  serves. Bytes from disk per token = 600 x S_exp x (1 - h).
- The disk delivers its rated bandwidth for 19-21 MB reads at the queue depth that PIPE
  gives (8 workers). The repo measured 93% of the `iobench` ceiling on an M1 Ultra
  (`docs/METAL-M1ULTRA-FMT2-REPORT.md:52`), so this assumption is close but optimistic.
- The ceiling ignores compute, attention and the serial dependency between layers. It is an
  upper bound from disk only.

**Bytes read from disk per token:**

| Container | Expert size | h = 0% | h = 50% | h = 90% |
|---|---|---|---|---|
| int4-g64 (recommended, 4.5 bpw) | 21.23 MB | **12.74 GB** | 6.37 GB | 1.274 GB |
| int4-row (older, 4.0 bpw) | 18.92 MB | **11.35 GB** | 5.68 GB | 1.135 GB |

The int4-row cold value matches the repo's "~11.4 GB of expert reads" per cold token
(`docs/benchmarks.md:102`).

**tok/s ceiling = disk bandwidth / bytes per token:**

| Container | NVMe | h = 0% | h = 50% | h = 90% |
|---|---|---|---|---|
| int4-g64 | 3 GB/s | 0.235 | 0.471 | 2.35 |
| int4-g64 | 7 GB/s | 0.549 | 1.10 | 5.49 |
| int4-row | 3 GB/s | 0.264 | 0.529 | 2.64 |
| int4-row | 7 GB/s | 0.617 | 1.23 | 6.17 |

**Reality check against measured rows:**
- i9-12900K, 6.44 GB/s O_DIRECT, hit 54.9%, MTP=0: ~9.9 GB read per token is reported, but
  that includes cache-miss re-reads; the formula gives 0.451 x 12.74 = 5.7 GB. Measured
  0.34 tok/s against a disk ceiling of 6.44 / 5.7 = 1.1 tok/s. The profile shows 48% disk,
  34% matmul, 12% attention (`docs/benchmarks.md:181`). So at 50% hit the real speed is
  ~30% of the disk-only ceiling: the CPU and the serial layer dependency take the rest.
- i5-13600K, 5.90 GB/s, hit 54.7%: measured 0.98 tok/s, but with `--topp 0.7` (fewer
  experts), so it is not comparable to the exact-routing ceiling.
- EPYC 9V45, 100% hit, 64 cores: 6.91 tok/s, with expert matmul 70 ms/token and attention
  54 ms/token (`docs/benchmarks.md:187`). This is the compute floor on a very large CPU.
  The repo estimates "~80 GFLOP/token vs ~250 GFLOP/s of our AVX2 kernels" (~3 tok/s) for a
  12-core desktop (`docs/benchmarks.md:111`). So the 90%-hit, 7 GB/s ceiling of 5.5 tok/s
  is **compute-bound, not disk-bound,** on any 8-16 core consumer CPU.

**What hit rate does 8-32 GB RAM buy?** Very little for GLM-5.2. The full routed set is
407.7 GB. After the ~17 GB fixed part, a 32 GB machine has ~15 GB for experts, i.e. ~3.7%
of all experts, or cap ~9 per layer out of 256. Measured hit rates at 24-32 GB are 3-11%
(`docs/benchmarks.md:146-147,162`). The OS page cache can add some hits, but the planner
reserves only 2.5 GB for it. **[inference]** For our 8-32 GB target, the 0% column is the
realistic one: ~0.24 tok/s at 3 GB/s and ~0.55 tok/s at 7 GB/s for GLM-5.2 int4-g64, before
compute and latency losses. The measured 0.07-0.11 tok/s on 24 GB machines confirms that the
real number is 2-4x below this ceiling.

**Per-layer latency view.** Without PILOT the experts of layer L are known only after layer
L's attention. At 7 GB/s, 8 x 21.23 MB per layer = 24.3 ms per layer, x 75 = 1.82 s per token
at 0% hit, and compute cannot overlap with the next layer's reads. PILOT (71.6% recall) is
the mechanism that hides part of this serial chain. **[inference]**

**Levers that change the bytes per token** (all from the repo):
- Lower bits: int3-g64 (3.5 bpw) would cut bytes by 22% vs g64; IQ1_S GGUF of GLM-5.2 is
  202 GB and decodes at 0.633-0.658 tok/s on machine D (`README.md:635`).
- Fewer experts (lossy): `--topp 0.7` (30-40% less disk), `EXPERT_BUDGET`, `DEGRADE_ZERO`,
  `CACHE_ROUTE`.
- More disks: weighted mirrors, +37.5% with two independent controllers.
- More RAM is the only lossless lever for the hit rate.

### 1.9 Best idea to reuse from colibri

**Per-expert contiguous layout + one coalesced (O_DIRECT) read per expert into a per-layer
slab cache, sized by an explicit RAM budget formula, with a learned frequency-first (LFRU)
pinned hot set.** Reason: it is the complete, measured design for exactly our problem. The
key parts are small and portable: (a) store gate/up/down of one expert back to back so a
miss is one ~20 MB sequential read (`c/colibri.c:3636-3672`); (b) an async loader pool that
reads misses while resident experts compute (`c/colibri.c:4582-4695`); (c) a cap computed
from `budget - resident - slack` that refuses to start rather than be OOM-killed
(`c/colibri.c:13348-13430`); (d) persisted routing statistics that pin hot experts across
sessions (`docs/tuning.md:193-213`). Router-lookahead prefetch (PILOT) and weighted
multi-SSD mirroring are strong second-order additions.

---

## 2. MemVanta (sauravsingla/MemVanta)

### 2.1 Architecture

Small C++20 runtime (`docs/ARCHITECTURE.md:5-35`):
- `gguf.*`: GGUF v2/v3 parser with limits checks (`src/gguf.cpp`, 501 lines).
- `mmap_file.*`: maps the file (`src/mmap_file.cpp`).
- `tensor_store.*`, `lru_cache.*`, `prefetcher.*`, `prefetch_policy.*`, `runtime.*`: a
  bounded copy cache and an adaptive prefetcher.
- `gguf_kernels.*`: Q4_0, Q8_0, Q6_K, F16, F32 matvec (`src/gguf_kernels.cpp`, 1,025 lines).
- `llama_model.*`: the Llama graph, paged KV cache, tokenizer (`src/llama_model.cpp`,
  1,214 lines).
- Two CLIs: `memvanta_real` = text generation (`src/real_main.cpp:46-51` builds a
  `LlamaModel`); `memvanta run <file>` = a synthetic streaming/cache benchmark
  (`src/main.cpp:57-58` builds a `Runtime`). The README says so (`README.md:110`).

**Important finding:** the LRU copy cache, the `Prefetcher` thread and the adaptive
prefetch controller are used only by `Runtime::run_stream` (`src/runtime.cpp:51-268`), which
walks tensor slices and computes a checksum. `src/llama_model.cpp` includes none of them
(its includes are `llama_model.hpp`, `checked_math.hpp`, `gguf_kernels.hpp`;
`src/llama_model.cpp:1-16`). The model path uses only the plain mmap. So "adaptive
prefetching" and "bounded tensor slices and caching" in the README (`README.md:118-123`)
do not act during real text generation **[verified by include and call search]**.

**Data flow for one decode token (`forward_batch_impl`, `src/llama_model.cpp:950+`):**
embedding row -> for each layer: RMSNorm, Q/K/V matvec read straight from the mmap'd GGUF,
RoPE, write K/V to the paged cache, attention over cached pages (GQA), O projection,
residual, RMSNorm, gate/up/down SwiGLU -> final norm -> output head -> sampling.

### 2.2 Memory strategy

- The whole GGUF is mapped `PROT_READ, MAP_PRIVATE` and advised `MADV_RANDOM`
  (`src/mmap_file.cpp:64-70`). `MADV_RANDOM` turns off kernel readahead.
- Weights are used in place from the mapping (zero copy, "zero-copy F32/F16/Q4_0/Q8_0 tensor
  access", `REAL_MODEL_BENCHMARK.md:8-10`). Only norm vectors are converted to f32 and kept
  (`src/llama_model.cpp:873-874`).
- So what is "resident" is decided by the OS page cache. There is no explicit cache policy,
  no pread, no O_DIRECT, no async I/O on the model path.
- The synthetic runtime has: a byte-bounded LRU copy cache (`src/lru_cache.cpp`), one
  prefetch worker thread that does `WILLNEED` + copy into the cache
  (`src/prefetcher.cpp:91-131`), and an adaptive depth controller
  (`src/prefetch_policy.cpp:23-127`). The controller lowers depth on eviction pressure,
  budget skips, low usefulness or a >5% slowdown; it probes one step deeper only after two
  stable windows with late arrivals, high usefulness and 25% byte headroom; it remembers a
  rejected probe until a >5% phase change (`src/prefetch_policy.cpp:36-123`). Adaptive
  look-ahead is capped at half the copy cache (`src/runtime.cpp:69-79`).
- Peak RAM formula: not derivable for the model path; it is whatever the page cache keeps
  plus KV pages plus small buffers. The prefill workspace is checked against
  `MEMVANTA_MAX_WORKSPACE_BYTES` (`src/llama_model.cpp:972-973`).

### 2.3 Weight format and quantization

- On-disk format: unchanged GGUF. Supported matvec types: Q4_0 (18-byte block for 32
  weights, 4.5 bpw), Q8_0 (34-byte block), Q6_K (210-byte block per 256), F16, F32
  (`src/gguf_kernels.cpp:28-45`; unsupported types throw at `src/gguf_kernels.cpp:838`).
  No conversion step. Only `general.architecture=llama` runs (`README.md:129-135`).
- Kernel: `dot_q4_0_fp32` widens int4 to int32, converts to float and does FMA against fp32
  activations (`src/gguf_kernels.cpp:220-262`). This is a float path, not an int8 dot product
  like llama.cpp's Q4_0 x Q8_0. **[inference]** This likely explains much of the 4x
  throughput gap in 2.6. AVX2 paths use `__attribute__((target("avx2,fma")))` with runtime
  dispatch (`src/gguf_kernels.cpp:146-210`).

### 2.4 KV cache

- `PagedKVCache` per layer: pages of `page_tokens` positions (default 128), allocated lazily
  on first write (`src/llama_model.cpp:155-182`).
- Types: F32, F16, Q8 (`src/llama_model.cpp:135-142`). Q8 uses one scale per token per K and
  per V row, amax/127 (`src/llama_model.cpp:203-216`).
- No offload to disk, no eviction, no compression beyond Q8.

### 2.5 Speculative decoding / speed tricks

None found in the model path. Batched prefill exists (`forward_batch_impl`).

### 2.6 Benchmark claims

From `results/openllama-7b-v2-ab/summary.json` and `SUMMARY.md` (the README table is generated
from this file, `README.md:22-31`):

| Metric | MemVanta | llama.cpp (pinned `6503355`) |
|---|---|---|
| OpenLLaMA 7B v2 Q4_0 peak RSS | 3,982,168 KiB (3.80 GiB) | 7,590,668 KiB (7.24 GiB) |
| Prompt processing pp512 | 2.92 tok/s | 11.95 tok/s |
| Token generation tg128 | 1.92 tok/s | 7.97 tok/s |

Hardware: Azure GitHub runner, AMD EPYC 9V74, 4 vCPU, 4 threads, ctx 768, batch 32, F16 KV,
1 warm-up + 5 runs (`results/openllama-7b-v2-ab/environment.txt`, `SUMMARY.md:3`). The llama.cpp
command is recorded (`llama-bench.time.txt`). Raw outputs, model SHA-256 and commits are in
the folder. Reproducible by the GitHub workflows in `.github/workflows/`.

cgroup experiment (`results/openllama-7b-v2-ram-constrained/SUMMARY.md`): with swap off,
MemVanta completed under `MemoryMax=160 MiB` (OOM at 128 MiB); llama.cpp needed 3,648 MiB.
No throughput is reported for that run.

Skeptical notes:
- MemVanta's peak RSS (3.80 GiB) is about the model file size (3,825,818,912 B = 3.56 GiB,
  `model-size.txt`). It does not run a model larger than RAM in any measured way; it
  only avoids a second copy. **[inference]** llama.cpp's 7.24 GiB is about 2x the file; a
  likely cause is llama.cpp's CPU weight repacking into an interleaved anonymous buffer next
  to the mapped file. The repo does not state the cause.
- The 160 MiB cgroup result works because clean file-backed pages can be dropped and
  re-read from disk. That is demand paging of the whole model per token. At 3.56 GB/token
  and 3 GB/s this would be ~1.2 s/token at best **[inference]**; the repo does not measure it.
- MemVanta is 4.1x slower than llama.cpp on the same machine. The repo states this openly
  ("it does not claim to be faster", `README.md:18`).

### 2.7 Code quality and maturity

- Tests: `tests/tests.cpp` and six other test files (~900 lines), ASan/UBSan/TSan lanes,
  fuzz smoke, ARM64 QEMU lane, 30+ GitHub workflows (`README.md:139-150`,
  `.github/workflows/`). Very strong evidence discipline for a small repo.
- Size: ~5.3k lines. Pre-release v0.8.3. Trained-model evidence only up to 7B
  (`README.md:137`). Qwen2 parses but does not run (`README.md:135`).
- Known issue: the model file is monolithic; the doc wants to split KV and tokenizer out
  (`docs/ARCHITECTURE.md:34-35`).

### 2.8 Best idea to reuse from MemVanta

**The byte-bounded adaptive prefetch-depth controller** (`src/prefetch_policy.cpp:23-127`).
Reason: it is a small, deterministic, separately tested feedback loop (inputs: late items,
useful ratio, evictions, budget skips, latency per window; output: depth up/down). It never
raises in-flight bytes past a hard budget, and it remembers failed probes. We need exactly
this to tune how many layers/experts ahead to prefetch on unknown disks, without hand
tuning per machine. A second reusable item is its benchmark protocol: cgroup-v2
`MemoryMax` boundary search with swap off, reported next to peak RSS and throughput
(`docs/MEMORY_BENCHMARKING.md:15`).

---

## 3. reame (swellweb/reame)

### 3.1 Architecture

reame is an OpenAI-compatible **server**. It has two backends behind one interface
(`README.md:20`):
- `llama.cpp` lane for GGUF (`src/core/llama_backend_real.cpp`, 388 lines; llama.cpp is a
  git submodule, `.gitmodules`). In this checkout the submodule folder
  `third_party/llama.cpp` is **empty**.
- `nucleo` lane for ternary `.nuc` files. **The nucleo engine source is not in this repo.**
  `src/CMakeLists.txt` looks for it in a sibling directory `${CMAKE_SOURCE_DIR}/../nucleo-dev`
  and silently builds without it if absent ("Motore nucleo: assente (solo llama.cpp)").
  The converter `reame-converti` also comes from that external directory. Only the glue
  `src/core/nucleo_backend.cpp` (207 lines) is here; it calls `nucleo::motore`,
  `continua`, `continua_lotto` (batches of up to 8 tokens), `tronca`
  (`src/core/nucleo_backend.cpp:18-104`).

Server modules: `server/` (HTTP, API handler, escalation relay), `core/` (engine, scheduler,
sampler, autoconfig, "conclave"), `speculative/` (draft generator, prompt lookup, form draft,
batch verifier, acceptance), `cache/` (disk prefix cache), `palimpsest/` (corpus index for
retrieval), `arca/` (an archive of past generations used as draft material). `src/memory/`
is empty.

Data flow for one decode token: request -> prefix cache restore -> backend
`decode_append` -> sampler -> optional speculative draft + `decode_batch` verify. The
per-token compute happens inside llama.cpp or nucleo, neither of which is in this checkout.

### 3.2 Memory strategy

- nucleo: "the server simply maps it -- 99 MB of RAM at startup" (`README.md:104`). The mmap
  code is in nucleo, so I cannot verify it. **[no evidence in repo]**
- The ternary model is small (Bonsai-8B = 2.2 GB) and fits in RAM; there is no streaming of
  weights from disk and no expert cache. The README notes for MoE: "Fewer MoE experts help
  prefill (+44%) but hurt decode. At K=2 the model collapses entirely (0/20)"
  (`README.md:176`). The measurement is not in the repo (bench scripts only).
- The disk is used for **KV state**, not weights (3.4).
- Peak RAM formula: not derivable.

### 3.3 Weight format and quantization

- `.nuc`: "interleaved 2.125-bit format", block-128 with one fp16 row scale per 128 weights,
  head and embeddings included (`docs/BENCHMARKS.md:9-14`). 2 bits per weight + 16/128 =
  2.125 bpw. Source: PrismML Bonsai GGUF Q2_0 (ternary {-1, 0, +1}).
- Kernels: integer NEON `vdotq_s32` (the CMake comment requires `-march=native` for dotprod,
  `src/CMakeLists.txt`), exact int32 accumulation per 128-weight block; a "sum-trick on raw
  2-bit codes with per-lane activation sums precooked at quantization time"; an `smmla`
  (i8mm) kernel for batches (`docs/BENCHMARKS.md:95-110`). **[code not in repo]**

### 3.4 KV cache

- In-memory KV: inside llama.cpp or nucleo.
- **Disk prefix cache** (`src/cache/prefix_cache.cpp:38-66`): the prompt is cut at fixed
  block boundaries (`block_tokens`); for each boundary the key is a double 64-bit hash of
  model tag + token ids (`prefix_cache.cpp:22-36`). On a request the engine restores the
  longest cached boundary, decodes the rest block by block and stores a snapshot at each
  new boundary.
- Snapshot = full backend state from `llama_state_get_data` (`llama_backend_real.cpp:300-315`),
  zstd-compressed with a checksum envelope (`src/cache/cache_serializer.cpp:26-35`,
  `src/cache/cache_manager.cpp:66-80`). A disk budget (`max_size_mb`, default 4096 in the
  README config) limits the store.
- Warm-ahead: `POST /v1/warm` and `PrefixCache::prefetch` load and decompress a snapshot into
  a RAM staging area with LRU eviction before the real request arrives
  (`src/cache/prefix_cache.cpp:72-78`, `src/cache/cache_manager.cpp:118-160`).
- Generation results are cached only at temperature 0 (`src/core/engine.cpp:248-251`).

### 3.5 Speculative decoding

- Draft sources: prompt lookup (`speculative/prompt_lookup.cpp`), past generations
  ("arca"), form drafts, and a draft model (`speculative/draft_generator.cpp`).
- Adaptive draft length: acceptance >= 0.9 -> +2 tokens; <= 0.4 -> -2 tokens, bounded to
  [2, 32] (`src/speculative/speculative_decoder.cpp:297-308`,
  `include/reame/speculative/speculative_decoder.hpp:63-64`).
- Auto-disable: after 64 drafted tokens, if acceptance < 0.15, speculation turns off
  (`speculative_decoder.cpp:310-315`, `include/reame/core/engine.hpp:73-74`).
- The README says reame "detects [oversubscribed vCPUs] at runtime and disables
  [speculation] automatically" (`README.md:177`). The code I found disables on low
  **acceptance**, not on a vCPU/steal measurement. **[claim partly not matched by code]**

### 3.6 Benchmark claims

From `README.md:13-16` and `docs/BENCHMARKS.md:49-55`:

| Box | Model | nucleo decode | rival decode | nucleo prefill | rival pp256 |
|---|---|---|---|---|---|
| Oracle A1, 4 cores | Bonsai-8B | 5.7 | PrismML fork 4.84 | 7.9 | 5.5 |
| Oracle A1, 2 cores | Bonsai-8B | 3.0 | fork 2.46 | 4.3 | 2.77 |
| Oracle A1, 4 cores | Bonsai-1.7B | 22.1 | mainline 31.6 | 33 | 66 |
| Apple M3 Pro, 5 threads | Bonsai-1.7B | 85.7 | mainline 114.1 | 107 | 219 |

Accuracy 19/20 on a 20-question exam (`docs/BENCHMARKS.md:77-87`). "149 test cases, 254,585
assertions" for nucleo (`docs/BENCHMARKS.md:91-93`).

Skeptical notes:
- **The engine that produced every number is not in the repo.** I cannot inspect kernels,
  tests or the 149-case suite. All nucleo claims are **[no evidence in repo]**.
- The repo does print losing rows, which is honest. But methods differ: the rival uses
  `llama-bench` pp256/tg64 r=2, nucleo uses its own "chat banco" (`docs/BENCHMARKS.md:23-41`).
- A 20-question exam is too small to measure quantization quality.
- The test suite in the repo (293 `TEST_CASE`s in `tests/unit/`) covers the server, cache
  and speculation with a llama mock (`tests/mock/llama_mock.hpp`), not the engine.

### 3.7 Code quality and maturity

- Server code is clean, small, tested (6.0k lines of tests for 7.4k lines of source).
- CI runs only on manual dispatch (last commit "ci: avvio solo a richiesta").
- Mixed Italian/English identifiers.
- Relevance to our goal is low: no weight streaming, no large-model path, small models only.

### 3.8 Best idea to reuse from reame

**Disk-backed prefix KV snapshots at fixed token boundaries, with hash keys, zstd,
checksums, a disk budget and warm-ahead staging** (`src/cache/prefix_cache.cpp:38-78`,
`src/cache/cache_manager.cpp:66-160`). Reason: on a 700B model that decodes at 0.1-1 tok/s,
prefill of a long system prompt or document costs minutes. Reusing the KV state of the
longest shared prefix from disk removes that cost across requests and restarts. colibri has
a similar idea for one conversation (`.coli_kv`), but reame's version is keyed by prefix and
shared across different prompts. Second idea: acceptance-gated adaptive draft length with
auto-disable, which matters because colibri showed speculation can lose on a disk-bound
MoE.

---

## 4. rai (Classevelabs/rai)

### 4.1 Architecture

Rust workspace (`README.md:137-164`):
- `rai-infer`: `.raimodel` reader/writer (`format.rs`, 2,221 lines), streaming converter
  (`convert.rs`, 4,046 lines), W4A8 GEMM (`gemm.rs`, 3,007 lines), layers (`layers.rs`,
  2,375 lines), model (`model.rs`, 1,208 lines), KV cache, sampler, speculative and lookup
  decoding, CLI and a chat server.
- `rai-compress`: GPTQ, sparse and "RC/HRC/SAC" research code. "Nothing here is on the
  inference path" (`README.md:142`).
- `rai-server`, `rai-core`, `rem-nra`: a separate memory/reasoning service; "not part of the
  RAI product", "imports rai-infer zero times" (`README.md:144-164`).

Data flow for one decode token: embedding row (8-bit table) -> per layer: RMSNorm, fused
QKV W4A8 (`w4a8_fused_qkv`, `gemm.rs:1458`), RoPE, GQA attention over the f32 KV cache, O
projection, residual, RMSNorm, fused gate+up (`gemm.rs:1633`), SwiGLU/GeGLU, down -> for MoE
layers, `route_experts` (softmax top-k, `layers.rs:1838-1870`) and a loop over selected
experts (`model.rs:232-258`) -> final norm -> LM head -> sampler.

### 4.2 Memory strategy

- The loader reads the **whole file into a heap buffer** with one read, then hands out
  borrowed slices (`README.md:347-351`; `read_to_end` at `format.rs:1530-1538`). No mmap, no
  streaming at inference time, no expert cache. MoE models (OLMoE, Mixtral, Qwen3-MoE) are
  fully resident.
- KV is allocated for 1,024 positions first and grown on demand towards `max_ctx`
  (`kv_cache.rs:17-27`).
- Peak RAM ~= file size + KV + scratch. Measured: TinyLlama 619 MB file -> 629 MB RSS;
  Zephyr-7B 3.9 GB -> ~4.0 GB (`README.md:34-38`).
- The **converter** streams: it reads safetensors in ~8 MB blocks, writes codes in place,
  then seeks back to fill the group parameters (`convert.rs:20-21`, `convert.rs:66`,
  `convert.rs:2907-2930`). Measured peak 26.3 MB RAM to convert Zephyr-7B
  (`README.md:25-28`).

### 4.3 Weight format and quantization

- `.raimodel`: one flat little-endian file, 64- or 128-byte header, a section index (16 B per
  section), 8-bit embedding table, per-layer sections with seven 4-bit projections, f32
  biases and norms, optional 4-bit LM head (`README.md:311-345`).
- Linear layers: 4-bit codes, two per byte, per-group (128 columns default) f16 scale and f16
  zero (asymmetric). 4 + 32/128 = **4.25 bpw**.
- Conversion: `rai convert` = round-to-nearest, no Python. GPTQ path = Python
  `export_raimodel.py` (container v1 only). The Rust GPTQ in `rai-compress/src/gptq.rs:157`
  (Hessian-guided, blocked error propagation) is not used for `.raimodel` export
  (`README.md:142`).
- Kernel: W4A8. Activations quantized per group to int8; `_mm256_maddubs_epi16` +
  `_mm256_madd_epi16` (`gemm.rs:497-500`); F16C for scales; factored dequant
  `w.x = scale*dot(codes,x) + zero*sum(x_group)` (`gemm.rs:1-10`). AVX2 only; ARM falls back
  to scalar (`README.md:395-399`).
- Quality: RTN 4-bit raises wikitext-2 perplexity by 13-30% over fp16; llama.cpp Q4_K_M costs
  3-8x less (`README.md:385-394`). The repo measured and published this itself.

### 4.4 KV cache

Per layer `[num_kv_heads, capacity, head_dim]` f32 (`kv_cache.rs:7`, `kv_cache.rs:34-46`).
Watermark `filled` and `truncate` for speculative rollback. No paging, no compression, no
disk offload.

### 4.5 Speculative decoding

Draft-model and prompt-lookup speculation with exact rejection sampling against the target
distribution (`README.md:129-132`, `README.md:260-286`). Prompt lookup: 1.12-1.20x on
context-quoting text, 0.74-0.88x on original prose (`BENCHMARKS.md:142-143`).
Self-speculative early exit reached 0.4% acceptance and ran ~15x slower; it was removed from
the CLI (`README.md:360-363`).

### 4.6 Benchmark claims

From `README.md:16-50` and `BENCHMARKS.md:79-108` (4-core/8-thread laptop, 2026-08-09):

| | TinyLlama-1.1B | Zephyr-7B |
|---|---|---|
| Decode | 21.8 tok/s (quiet machine, n=3) | 2.96 tok/s (**under 49% background load**, n=3) |
| Peak RSS | 629 MB | ~4.0 GB |
| Convert | 7.6 s, 22.9 MB RAM | 82.6 s, 26.3 MB RAM |

Roofline: 549.5 MB streamed per token; 26.4 GB/s measured bandwidth; ceiling 48 tok/s;
21.8 = 45% of ceiling (`BENCHMARKS.md:100-108`).

Skeptical notes:
- The CPU model of the laptop is not named in the README text I read.
- The 7B decode was not measured on a quiet machine; the README headline does not say so,
  only `BENCHMARKS.md:84` does.
- "5.1x faster than transformers fp32" compares 4-bit against fp32; it is not an
  engine-quality comparison.
- The older "195 tok/s SmolLM-135M" figure has no retained raw output; the repo itself says
  it is "historical context, not release evidence" (`README.md:363-366`).
- No llama.cpp throughput comparison is published.

### 4.7 Code quality and maturity

- 378 `#[test]` functions across the workspace, 9 integration test files in `rai-infer/tests`
  (e.g. `convert_matches_python.rs`, 1,996 lines, checks byte-identical output with the
  Python exporter). `cargo clippy -D warnings` gate (`README.md:201-208`).
- Version 0.2.5, pre-1.0 (`README.md:371-402`). Last commit 2026-09-19.
- About half the workspace (the memory service) is unrelated to inference.
- Relevance to our goal is low: everything is resident, x86-only fast path.

### 4.8 Best idea to reuse from rai

**The constant-memory streaming converter** (`convert.rs:20-21`, `convert.rs:2907-2930`):
read safetensors in ~8 MB blocks, quantize, stream codes to their final offset, seek back to
write group parameters; refuse unsupported architectures before writing. Reason: to build our
own expert-streaming container for a 700B model on a 16-32 GB machine, the converter itself
must not need RAM proportional to a tensor or a shard. rai shows 26 MB peak for 7B and
byte-identical output against a reference exporter. colibri's shard-by-shard converter does
the same at shard granularity. Second idea: the factored W4A8 dequant
`scale*dot(codes,x) + zero*sum(x)` for asymmetric int4 with int8 dot instructions.

---

## 5. Comparison table

| | colibri | MemVanta | reame | rai |
|---|---|---|---|---|
| Language | C (+ CUDA, Vulkan GLSL, Metal, Python tools) | C++20 | C++20 server; engine (nucleo) external C++ | Rust |
| Model support | GLM-5.2/5.3, Mistral-Large-4, Inkling, Kimi K3, DeepSeek V4/V4.1, MiMo, Qwen3.6/3.8, OLMoE, many GGUFs | Llama-architecture GGUF only, up to 7B evidence | Ternary Bonsai (.nuc) via nucleo; any GGUF via llama.cpp | Llama-shaped dense + MoE without shared expert (Llama 2/3, Mistral, Qwen2/2.5/3, Gemma 1-3, Phi-3, OLMo2, OLMoE, Mixtral) |
| Quant formats | int8-row, int4-row, int4-g64 (4.5 bpw), int3-g64, E8 lattice, MXFP4 (Vulkan), FP8-e4m3; GGUF IQ1_S..Q8_0, F16, BF16 | GGUF Q4_0, Q8_0, Q6_K, F16, F32 | .nuc 2.125 bpw ternary; GGUF via llama.cpp | int4 g128 asymmetric f16 scale/zero (4.25 bpw), 8-bit embedding |
| Disk streaming | Yes: routed experts, one coalesced pread per expert, optional O_DIRECT, io_uring, async pool, multi-SSD mirrors | Only OS demand paging of an mmap (MADV_RANDOM); copy cache/prefetch only in a synthetic benchmark | No (weights resident or mmap'd by external engine) | No (whole file read into heap) |
| Expert cache | Per-layer LRU + learned pinned hot set (LFRU, persisted `.coli_usage`) + router-lookahead prefetch + optional VRAM tier + cache-aware routing | None | None | None (MoE fully resident) |
| KV strategy | MLA latent 576 values/token/layer; f32/f16/fp8/int4 options; in RAM; persisted to `.coli_kv`; prefix reuse; parking | Paged (128-token pages), F32/F16/Q8, lazy allocation, RAM only | Inside backend; disk prefix snapshots (zstd, LRU staging, warm-ahead) | f32 contiguous, grown on demand, RAM only |
| GPU support | CUDA, Vulkan (any GPU), Metal, HIP, XDNA; VRAM expert tier and dense chain | None | Via llama.cpp lane only | None |
| Claimed RAM | GLM-5.2 744B: 9.9 GB resident, ~20 GB peak RSS on 25 GB laptop; 16 GB minimum / 24 GB recommended (`README.md:221`) | OpenLLaMA 7B Q4_0: 3.80 GiB peak RSS (vs 7.24 llama.cpp); completes under 160 MiB cgroup | Bonsai-8B: 2.2 GB; 99 MB at startup | TinyLlama 629 MB; Zephyr-7B ~4.0 GB; convert 7B in 26 MB |
| Claimed speed | GLM-5.2: 0.05-0.1 tok/s cold (25 GB, ~1 GB/s); 0.34 tok/s (64 GB, 6.4 GB/s); 1.83 (128 GB); 6.91 (504 GB, 64 cores) | 7B: 1.92 tok/s tg, 2.92 pp (4 vCPU), 4.1x slower than llama.cpp | Bonsai-8B: 5.7 tok/s (4 ARM cores), +18% vs vendor fork | TinyLlama 21.8 tok/s; Zephyr-7B 2.96 tok/s (loaded machine) |
| Evidence quality | High: many community rows with linked issues, iobench, datapoint tool, protocol | High: raw logs, hashes, CI workflows, pinned rival | Low for engine: engine not in repo | Medium: own measurements, roofline, honest negatives; one machine |

---

## 6. Synthesis for our goal

1. Only colibri runs models larger than RAM in a useful way, and only for **MoE** models.
   For dense 70B models, none of the four engines streams weights in a designed way;
   MemVanta's mmap demand paging is the only path, and it reads the full model per token.
2. For a 744B MoE at int4-g64, a cold token reads 12.74 GB. On 8-32 GB RAM the expert hit
   rate stays near 0-10%, so the disk sets the speed: ~0.24 tok/s ceiling at 3 GB/s and
   ~0.55 tok/s at 7 GB/s; measured reality is 2-4x lower. Two independent NVMe drives add
   ~35%. Getting to 1 tok/s on such RAM needs fewer bytes per token (2-3 bpw experts, lossy
   expert top-p) or smaller MoE models (e.g. 100-300B with fewer or smaller experts).
3. Speculative decoding does not automatically help a disk-bound MoE, because verify rows
   enlarge the expert union (colibri: 600 -> 1,062 experts/token, slower overall). Any
   speculation must be gated on measured benefit (colibri's gate, reame's auto-disable).
4. A clean design for our engine would combine: colibri's per-expert contiguous container,
   coalesced O_DIRECT reads, async loader pool, per-layer LRU + learned LFRU pins and RAM
   budget planner; PILOT-style router lookahead; MemVanta's bounded adaptive prefetch
   controller to tune lookahead depth; reame's disk prefix KV cache; rai's
   constant-memory streaming converter.

## 7. Best idea per repo (summary)

| Repo | Best idea | Why |
|---|---|---|
| colibri | Contiguous per-expert layout, one O_DIRECT pread per expert into per-layer slab LRU, async loader pool, RAM-budget cap formula, learned LFRU pinned hot set | Complete and measured design for streaming MoE experts on small RAM; each part is small and portable |
| MemVanta | Byte-bounded adaptive prefetch-depth controller with probe/reject memory | Auto-tunes lookahead on unknown disks without exceeding a memory budget; deterministic and testable |
| reame | Disk prefix KV snapshots at fixed boundaries with hash keys, zstd, checksum, budget and warm-ahead | Removes repeated prefill, which costs minutes when decode is <1 tok/s |
| rai | Constant-memory streaming converter (8 MB blocks, seek-back parameter write) | Converting 700B checkpoints on a 16-32 GB machine needs O(1) converter memory |
