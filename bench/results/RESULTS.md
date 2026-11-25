# Baseline results — 2026-11-12

Machine: Ryzen 5 5600H (6C/12T, AVX2), 8 GB DDR4-3200 single channel, GTX 1650 4 GB (3.3 GB free, no tensor cores), Samsung PM9A1 NVMe, Windows 11.
Engine: llama.cpp b12317, official win-cuda-12.4-x64 build. Models were sha256-verified against Hugging Face and deleted after the tests.

Files: `01_dense8b.md`, `02_moe30b.md` (21:25 run), `02_moe30b_cpu_experts.md` + `03_moe30b_cpu_only.md` (18:49 orphan run, pp512/tg64), `04_moecache_*.txt` (full logs), `disk_read.csv` (typeperf, 1 s, 21:25–21:32), `moe_trace_1500.bin` (routing trace, MCT1), `sim_decode.txt` / `sim_all.txt` (cache simulator, `../moe_sim.py`).

## 1. Dense 8B (Qwen3-8B Q4_K_M, 4.68 GiB — fits in RAM)

- Pure CPU decode: 2.9–3.6 tok/s ≈ 17.8 GB/s effective. This is the limit of single-channel RAM bandwidth. The bandwidth law holds.
- Best: 28 of 36 layers on the GPU, **12.7 tok/s**.
- All layers on the GPU: **7.0 tok/s.** The model overfills VRAM, and the Windows WDDM driver silently moves the overflow to shared system memory over PCIe. **Our engine must place every tensor itself and never over-subscribe VRAM.**

## 2. MoE 30B-A3B (Qwen3-30B-A3B-Instruct-2507 Q4_K_M, 17.28 GiB — 2.3× the RAM)

48 layers, 128 experts, top-8. One expert (gate+up+down) is 2.72 MiB.

| Setup | prompt tok/s | decode tok/s | disk read, mean / max |
|---|---|---|---|
| attention on GPU, experts mmap on CPU, repack off (pp128/tg32) | 4.3 | 2.44 | 510 / 1289 MB/s |
| same, repack on | 4.2 | 2.44 | 519 / 2575 MB/s |
| same, pp512/tg64 (18:49 run) | 14.6 | 2.36–2.41 | — |
| pure CPU (pp128/tg32) | 8.2 | **2.96** | 763 / 3155 MB/s |
| pure CPU, pp512/tg64 (18:49 run) | 13.5 | 2.34 | — |
| llama-completion, GPU attention, no expert cache, 128 tokens | 1.5 | **2.83** | 566 / 3428 MB/s |
| same + GPU expert cache `--moe-cache-mib 1500` | 0.9 | **1.22** | 316 / 2309 MB/s |

Findings:
1. **A model 2.3× larger than RAM runs at 2.4–3.0 tok/s** with plain mmap and OS paging. MoE sparsity works on this laptop.
2. **The SSD is mostly idle.** The mean read rate is only 0.3–0.8 GB/s, against a drive that can do ~5–6 GB/s (bursts reached 3.4 GB/s). Decode is limited by **page-fault latency** (small synchronous reads at queue depth 1) and by the CPU, not by SSD bandwidth.
3. **The GPU expert cache makes decode 2.3× slower** (1.22 vs 2.83 tok/s). Uploading missed experts over PCIe plus the host sync costs more than computing them on the CPU. This confirms the Fiddler rule (analysis G): for decode, compute a missed expert where it is.
4. **The GPU split helps prefill, not decode.** Pure CPU decode (2.96) beats GPU attention + CPU experts (2.44), because each token makes 48 CPU↔GPU round trips.

## 3. Cache simulator on the real routing trace (decode only, 129 tokens)

- **ρ = 0.458:** a token reuses 46% of the previous token's experts in the same layer. Random routing would give 6.2%, so the locality is **7×** stronger. This means:
  - Caching works better than the "spread-out routing" papers suggest.
  - Speculative decoding costs less than the independent-routing estimate (two tokens ≈ 12.3 distinct experts per layer, not 15.75).
- The per-layer cache sizes below are the RAM the cache would need:

| Cache | LRU | LFU | Belady (optimal) | miss GiB/token (LRU) |
|---|---|---|---|---|
| 2.0 GiB (16/layer) | 0.588 | 0.627 | 0.736 | 0.414 |
| 4.1 GiB (32/layer) | 0.781 | 0.813 | 0.881 | 0.220 |
| 6.1 GiB (48/layer) | 0.890 | 0.896 | 0.920 | 0.111 |

- With a realistic 2–4 GiB RAM cache, LFU beats LRU by 3–4 points, and the optimum is 10–15 points above LRU. So the policy matters more here than the papers found (2–9%). **Caveat:** the trace is only 129 tokens long. Cold misses cap the hit rate at 0.93.
- Cross-check: the measured disk rate (~0.21 GB per token at 2.44 tok/s) matches LRU at ~4 GiB (0.22 GiB per token). The OS page cache behaves like a ~4 GiB LRU cache.

## 4. What this means for our engine [est]

At 0.22–0.41 GiB of misses per token and ~5 GB/s, the SSD alone would allow about 12–23 tok/s. Measured decode is 2.4–3.0 tok/s. The loss comes from synchronous 4 KB page faults and from the CPU path. These steps should close a large part of that gap:
- one contiguous block per expert (2.7 MiB)
- async direct I/O at high queue depth into an LFU slot pool
- miss experts computed on the CPU

A realistic target on this laptop is **~6–9 tok/s for a 30B-A3B MoE**. This is an estimate, to verify with a prototype.
