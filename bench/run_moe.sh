#!/usr/bin/env bash
# MoE streaming tests: Qwen3-30B-A3B Q4_K_M (18.6 GB) on 8 GB RAM. Part 2 of run_baseline.sh.
set -uo pipefail
cd <repo>
B=bin/cuda/llama-bench.exe; C=bin/cuda/llama-completion.exe; OUT=bench/results
M=models/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf
want=$(curl -s "https://huggingface.co/api/models/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF/tree/main" | python -c "import sys,json;print(next(x['lfs']['oid'] for x in json.load(sys.stdin) if x['path']=='Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf'))")
echo "$want  $M" | sha256sum -c - || exit 1
typeperf "\PhysicalDisk(_Total)\Disk Read Bytes/sec" "\Memory\Available MBytes" -si 1 -o "$OUT/disk_read.csv" -y >/dev/null 2>&1 &
TP=$!; trap 'kill $TP 2>/dev/null' EXIT
f() { grep '^|' | tail -n +3; }
echo "## A: attention on GPU, experts on CPU (mmap), repack off/on" | tee $OUT/02_moe30b.md
date +%T >> $OUT/02_moe30b.md
$B -m $M -ngl 99 -ncmoe 48 -fa 1 --repack 0 -p 128 -n 32 -r 1 -o md 2>&1 | f | tee -a $OUT/02_moe30b.md
date +%T >> $OUT/02_moe30b.md
$B -m $M -ngl 99 -ncmoe 48 -fa 1 --repack 1 -p 128 -n 32 -r 1 -o md 2>&1 | f | tee -a $OUT/02_moe30b.md
date +%T >> $OUT/02_moe30b.md
echo "## B: pure CPU" | tee -a $OUT/02_moe30b.md
$B -m $M -ngl 0 -nopo 1 -fa 1 --repack 0 -p 128 -n 32 -r 1 -o md 2>&1 | f | tee -a $OUT/02_moe30b.md
date +%T >> $OUT/02_moe30b.md
echo "## C: GPU expert cache (llama-completion, 128 tokens)" | tee -a $OUT/02_moe30b.md
for mib in 0 1500; do
  T=""; [ $mib -gt 0 ] && T="$OUT/moe_trace_${mib}.bin"
  LLAMA_MOE_CACHE_TRACE="$T" $C -m $M -ngl 99 -ncmoe 48 -fa on --no-repack --moe-cache-mib $mib -n 128 -no-cnv --temp 0 \
    -p "Explain how a CPU cache works, in detail, with examples." > $OUT/04_moecache_${mib}.txt 2>&1
  echo "### moe-cache-mib=$mib" >> $OUT/02_moe30b.md
  grep -E "eval time|load time|moe_cache|hit" $OUT/04_moecache_${mib}.txt | tail -8 >> $OUT/02_moe30b.md
done
date +%T >> $OUT/02_moe30b.md
echo DONE
