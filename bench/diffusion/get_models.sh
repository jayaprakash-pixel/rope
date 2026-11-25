#!/usr/bin/env bash
# P0 model downloads, resumable, sha256-checked against the Hugging Face LFS oid.
set -uo pipefail
MOD=<repo>/models; mkdir -p "$MOD"
get() { # repo path-in-repo
  local repo=$1 p=$2 f="$MOD/$(basename "$2")" dir; dir=$(dirname "$2"); [ "$dir" = . ] && dir=""
  local want; want=$(curl -s "https://huggingface.co/api/models/$repo/tree/main/$dir" | python -c "import sys,json;print(next(x['lfs']['oid'] for x in json.load(sys.stdin) if x['path']=='$p'))")
  for i in 1 2 3 4 5; do
    curl -sL --fail -C - -o "$f" "https://huggingface.co/$repo/resolve/main/$p" && break; echo "retry $i $p"; sleep 5
  done
  if echo "$want  $f" | sha256sum -c - >/dev/null 2>&1; then echo "OK   $(basename "$f")"; else echo "FAIL $(basename "$f") want=$want"; fi
}
get stabilityai/sd-turbo sd_turbo.safetensors
get calcuis/wan-1.3b-gguf wan2.1_t2v_1.3b-q4_k_m.gguf
get city96/umt5-xxl-encoder-gguf umt5-xxl-encoder-Q4_K_M.gguf
get Comfy-Org/Wan_2.1_ComfyUI_repackaged split_files/vae/wan_2.1_vae.safetensors
get leejet/FLUX.1-schnell-gguf flux1-schnell-q4_k.gguf
get city96/t5-v1_1-xxl-encoder-gguf t5-v1_1-xxl-encoder-Q4_K_M.gguf
get comfyanonymous/flux_text_encoders clip_l.safetensors
get black-forest-labs/FLUX.1-schnell ae.safetensors
echo ALLDONE
