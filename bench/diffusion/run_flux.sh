#!/usr/bin/env bash
# P0.2.2 / P0.3.1 / P0.3.2: FLUX.1-schnell Q4_K, 4 steps, on the dev laptop and simulated weak devices.
cd "$(dirname "$0")"
M=../../models
F="--diffusion-model $M/flux1-schnell-q4_k.gguf --vae $M/ae.safetensors --clip_l $M/clip_l.safetensors --t5xxl $M/t5-v1_1-xxl-encoder-Q4_K_M.gguf"
P='-p "a red bicycle leaning on a brick wall, photo" --cfg-scale 1.0 --sampling-method euler --steps 4 -s 42'
S='import sys,json; r=json.load(sys.stdin); print(r["label"], {k:r.get(k) for k in ("rc","wall_s","text_enc_s","sampling_s","vae_s","total_s","peak_ws_gb","peak_commit_gb","gpu_dedicated_peak_gb","gpu_shared_peak_gb","disk_read_gb","avail_ram_min_mb")})'
run() { label=$1; shift; eval python -I measure.py --label $label "$@" | python -c "$S"; }
run p022_flux_gpu_512   -- ../../bin/sd-cuda/sd-cli.exe $F $P -W 512 -H 512 -o out/p022_flux_gpu_512.png
run p022_flux_gpu_1024  -- ../../bin/sd-cuda/sd-cli.exe $F $P -W 1024 -H 1024 -o out/p022_flux_gpu_1024.png
run p031_flux_cpu_512   --no-gpu -- ../../bin/sd-cpu/sd-cli.exe $F $P -W 512 -H 512 -o out/p031_flux_cpu_512.png
run p032_flux_cpu_4gb_512 --profile 4gb --no-gpu -- ../../bin/sd-cpu/sd-cli.exe $F $P -W 512 -H 512 -o out/p032_flux_cpu_4gb_512.png
echo FLUX_DONE
