#!/usr/bin/env bash
# Baseline: llama.cpp b12317 on this laptop (8 GB RAM, GTX 1650 4 GB, NVMe).
# Downloads ~25 GB (binaries + 2 models). Run from Git Bash:  bash bench/run_baseline.sh
set -euo pipefail
ROOT=<repo>
BIN=$ROOT/bin; MOD=$ROOT/models; OUT=$ROOT/bench/results
mkdir -p "$BIN" "$MOD" "$OUT"

# 1. llama.cpp official release binaries (CUDA 12.4 build supports Turing sm_75)
if [ ! -x "$BIN/cuda/llama-bench.exe" ]; then
  cd "$BIN"
  gh release download b12317 -R ggml-org/llama.cpp \
    -p 'llama-b12317-bin-win-cuda-12.4-x64.zip' -p 'cudart-llama-bin-win-cuda-12.4-x64.zip'
  unzip -qo 'llama-b12317-bin-win-cuda-12.4-x64.zip' -d cuda
  unzip -qo 'cudart-llama-bin-win-cuda-12.4-x64.zip' -d cuda
  rm -f ./*.zip
fi

# 2. Models, sha256-checked against the Hugging Face LFS oid
get() { # repo file
  local f="$MOD/$2"
  local want; want=$(curl -s "https://huggingface.co/api/models/$1/tree/main" \
    | python -c "import sys,json;print(next(x['lfs']['oid'] for x in json.load(sys.stdin) if x['path']=='$2'))")
  [ -f "$f" ] || curl -L --fail -o "$f" "https://huggingface.co/$1/resolve/main/$2"
  echo "$want  $f" | sha256sum -c -
}
get unsloth/Qwen3-8B-GGUF Qwen3-8B-Q4_K_M.gguf                                      # 5.0 GB, dense
get unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf # 18.6 GB, MoE, 3B active

B="$BIN/cuda/llama-bench.exe"; C="$BIN/cuda/llama-completion.exe"
D8="$MOD/Qwen3-8B-Q4_K_M.gguf"; M30="$MOD/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf"

# Disk read rate sampled every second during all runs
typeperf "\PhysicalDisk(_Total)\Disk Read Bytes/sec" -si 1 -o "$OUT/disk_read.csv" -y >/dev/null 2>&1 &
TP=$!; trap 'kill $TP 2>/dev/null || true' EXIT

# 3. Dense 8B: fits RAM; how much does partial GPU offload help?
"$B" -m "$D8" -ngl 0,18,26 -fa 1 -p 512 -n 64 -r 2 -o md | tee "$OUT/01_dense8b.md"

# 4. MoE 30B-A3B (18.6 GB > 8 GB RAM): attention on GPU, experts on CPU via mmap/page cache.
#    repack=0 keeps experts mmapped (streamable); repack=1 copies them into RAM.
"$B" -m "$M30" -ngl 99 -ncmoe 48 -fa 1 --repack 0,1 -p 512 -n 64 -r 2 -o md | tee "$OUT/02_moe30b_cpu_experts.md"
"$B" -m "$M30" -ngl 0 -fa 1 --repack 0 -p 512 -n 64 -r 2 -o md | tee "$OUT/03_moe30b_cpu_only.md"

# 5. Experimental GPU expert cache (not exposed in llama-bench) + routing trace for the simulator
for mib in 0 1500; do
  LLAMA_MOE_CACHE_TRACE="$OUT/moe_trace_${mib}.bin" "$C" -m "$M30" -ngl 99 -ncmoe 48 -fa on --no-repack \
    --moe-cache-mib $mib -n 256 -no-cnv --temp 0 \
    -p "Explain how a CPU cache works, in detail, with examples." 2>&1 | tail -n 12 | tee "$OUT/04_moecache_${mib}.txt"
done

echo "DONE. Results in $OUT"
