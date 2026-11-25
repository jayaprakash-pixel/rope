# rope-engine CPU kernels (P1.5.3): correctness and speed

Machine: Ryzen 5 5600H (6 cores / 12 threads, AVX2 + FMA), 8 GB DDR4-3200 single channel, Windows 11, MSVC 14.44.
rope: `engine/src/kernels_cpu.c` + `quant.c`, 6 threads (physical cores). Build: `engine/build.ps1`.

## Correctness

`rope test kernel` checks every kernel against an independent float64 reference on random inputs. The pass rule is max|got − ref| / max|ref| ≤ 1e-3.

| Kernel | Cases | Result |
|---|---|---|
| matmul | F32, F16, Q8_0, Q4_0, Q4_K, Q6_K × shapes (1,256,33) (3,512,70) (7,512,50) (37,768,129) (64,1024,96) | pass |
| conv2d (im2col + matmul) | 3×3 s1 p1, 3×3 s2 p0, 1×1 | pass |
| attention (2 heads, strided) | nq 37, nk 53, d 24 | pass |
| softmax, layernorm, rmsnorm, groupnorm | random | pass |
| GELU (erf), GELU (tanh), SiLU | random | pass |
| rotary, interleaved and NeoX | padded token stride | pass |

The quantized-weight references use a separate float64 dequantizer written from the format spec in `test_kernels.c`.

**Real tensors vs gguf-py's official dequantizer** (`bench/tools/check_dequant.py`):

| File | Type | Tensors | Values | Worst relative difference |
|---|---|---|---|---|
| wan2.1_t2v_1.3b-q4_k_m.gguf | Q4_K, Q6_K | 4 | 4.19 M | 3.7e-9 |
| umt5-xxl-encoder-Q4_K_M.gguf | Q4_K, Q6_K | 4 | 4.19 M | 4.1e-9 |

## Speed vs ggml (same machine, same shapes)

ggml baseline: `test-backend-ops perf -o MUL_MAT -b CPU`, built from `repos/llama.cpp` (CPU, GGML_NATIVE). rope: `rope bench M K N type`.

The shape is 4096 output rows × K = 14336. Values are GFLOPS (2·M·K·N per second).

| Weight type | rope, M = 512 tokens | ggml, M = 512 | rope / ggml | rope, M = 1 (decode) | ggml, M = 1 | rope / ggml |
|---|---|---|---|---|---|---|
| F32 | **282.4** | 176.7 | **1.60** | 9.5 | 10.0 | 0.95 |
| F16 | **283.5** | 183.5 | **1.55** | 20.4 | 19.7 | 1.04 |
| Q4_0 | 228.8 | 284.7 | 0.80 | 59.5 | 111.2 | 0.54 |
| Q8_0 | 269.4 | 320.8 | 0.84 | 35.9 | 44.4 | 0.81 |
| Q4_K | 268.0 | 598.6 | 0.45 | 18.1 | 119.3 | 0.15 |
| Q6_K | 271.1 | 588.2 | 0.46 | 10.9 | 66.3 | 0.16 |

Flux-sized layer (4096 tokens × 3072 × 3072): F32 166, F16 203, Q8_0 181, Q4_0 182, Q4_K 176, Q6_K 196 GFLOPS.

### What the numbers say
1. **The float GEMM beats ggml by 1.55–1.6×.** It uses a 6×16 register micro-kernel over k-major packed weight panels, with K-blocks of 256.
2. **Quantized weights are 2.2× slower than ggml in batch and up to 6.6× slower in decode.** ggml quantizes the *activations* to 8-bit and uses integer dot products (`maddubs`): twice the work per instruction and no per-weight float conversion. rope dequantizes every weight to float32 first. In decode, rope's scalar K-quant dequantizer is the bottleneck.
3. Decode with F32/F16 is limited by memory bandwidth for both engines, so they are about equal.

### Next (new leaf P2.2.3)
Add an integer-dot CPU path: activations quantized per block to int8, then `maddubs`/`madd` against Q4_K/Q6_K/Q8_0/Q4_0, plus a SIMD dequantizer. Target: at least ggml on quantized batch and decode.

---

# CUDA kernels (P1.5.4), GTX 1650 (Turing sm_75, 14 SMs, 4 GB, no tensor cores)

rope: `engine/src/cuda/kernels.cu`, CUDA 12.9, archs 61+75. ggml baseline: `test-backend-ops perf -o MUL_MAT -b CUDA0`, built from `repos/llama.cpp` for sm_75 (`build/ggml-cuda-build.ps1`).

## Correctness: every CUDA kernel vs the float64-verified CPU kernel (`rope test cuda`, limit 1e-2)

| Kernel | Worst relative error |
|---|---|
| matmul FP32 path (6 weight types × 4 shapes) | 1.8e-6 |
| matmul FP16 path (HFMA2, FP32 summation per 16-k tile) | 6.1e-4 |
| matmul INT8 dp4a path (Q8_0/Q4_0/Q4_K/Q6_K; activations int8 per 16) | 4.8e-3 |
| conv2d f32 / f16 | 0 / 6.8e-4 |
| attention | 3.1e-7 |
| softmax, layernorm, rmsnorm, groupnorm, GELU ×2, SiLU, rotary ×2 | ≤ 1.6e-6 |

## Speed on the same shapes (GFLOPS; 4096 output rows, K = 14336)

| Weight type | rope path | rope, M = 512 | ggml, M = 512 | rope / ggml | rope, M = 1 | ggml, M = 1 | rope / ggml |
|---|---|---|---|---|---|---|---|
| F32 | f32 | 840 | 960 | 0.88 | 12.1 | 93.9 | 0.13 |
| F16 | f16 | **1697** | 366 | **4.64** | 27.7 | 176.5 | 0.16 |
| Q4_0 | f16 / dp4a | **1691** / 634 | 1160 | **1.46** / 0.55 | 27.7 / 24.5 | 469.0 | 0.06 |
| Q8_0 | f16 / dp4a | **1687** / 650 | 1150 | **1.47** / 0.57 | 27.7 / 24.5 | 306.9 | 0.09 |
| Q4_K | f16 / dp4a | **1706** / 643 | 1160 | **1.47** / 0.55 | 27.7 / 24.5 | 474.5 | 0.06 |
| Q6_K | f16 / dp4a | **1680** / 621 | 1190 | **1.41** / 0.52 | 21.6 / 19.1 | 304.2 | 0.07 |

Flux-sized layer (4096 tokens × 3072 × 3072): rope f32 848, f16 1705–1728, dp4a 962–976 GFLOPS.

### What the numbers say
1. **For diffusion-sized batches, rope's FP16 path is 1.4–1.5× faster than ggml on quantized models and 4.6× faster on FP16 models.** On this tensor-core-less GPU, ggml's FP16 path is slow (366 GFLOPS).
2. **Caveat:** in the FP16 path the weights live in VRAM as FP16 (2 bytes/weight), while ggml keeps them quantized (0.56–1.06 bytes/weight). A whole large model cannot sit in 4 GB at FP16. This path only pays off together with per-block streaming (P2.1.1).
3. **Decode (M = 1): ggml is 6–17× faster.** rope runs the 64×64 GEMM tile with 1 valid row, so it reaches only ~28 GB/s of the ~128 GB/s VRAM bandwidth. A dedicated GEMV kernel is needed (new leaf P2.2.4).
4. **rope's dp4a kernel (620–976 GFLOPS) is slower than its own FP16 path and than ggml.** Each dp4a carries too much scale arithmetic and shared-memory traffic (new leaf P2.2.4).
