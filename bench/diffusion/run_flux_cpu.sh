#!/usr/bin/env bash
# P0.3.2 / P0.3.1: FLUX.1-schnell Q4_K on CPU only, with hard RAM caps so Windows keeps free memory.
# The first uncapped attempt was stopped by Claude Code because the system ran critically low on memory.
cd "$(dirname "$0")"
M=../../models
F="--diffusion-model $M/flux1-schnell-q4_k.gguf --vae $M/ae.safetensors --clip_l $M/clip_l.safetensors --t5xxl $M/t5-v1_1-xxl-encoder-Q4_K_M.gguf"
P='-p "a red bicycle leaning on a brick wall, photo" --cfg-scale 1.0 --sampling-method euler --steps 4 -s 42 -W 512 -H 512'
S='import sys,json; r=json.load(sys.stdin); print(r["label"], r["profile"], {k:r.get(k) for k in ("rc","wall_s","text_enc_s","sampling_s","vae_s","total_s","peak_ws_gb","peak_commit_gb","disk_read_gb","avail_ram_min_mb")})'
free_gb() { powershell -NoProfile -Command "[math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB,2)"; }

echo "free RAM before P0.3.2: $(free_gb) GB"
eval python -I measure.py --label p032_flux_cpu_4gb_512 --profile 4gb --no-gpu -- ../../bin/sd-cpu/sd-cli.exe $F $P -o out/p032_flux_cpu_4gb_512.png | python -c "$S"

f=$(free_gb); cap=$(python -c "print(round(min(5.5, $f - 0.75), 2))")
echo "free RAM before P0.3.1: $f GB -> working-set cap $cap GB (8gb profile wants 5.5)"
if python -c "import sys; sys.exit(0 if $cap >= 2.5 else 1)"; then
  eval python -I measure.py --label p031_flux_cpu_8gbcap_512 --ws-gb $cap --commit-gb 12 --no-gpu -- ../../bin/sd-cpu/sd-cli.exe $F $P -o out/p031_flux_cpu_8gbcap_512.png | python -c "$S"
else
  echo "P0.3.1 skipped: not enough free RAM for a cap above the 4 GB profile"
fi
echo FLUX_CPU_DONE
