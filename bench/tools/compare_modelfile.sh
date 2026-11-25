#!/usr/bin/env bash
# P1.5.2 acceptance: `rope inspect` must match the independent reference parser(s) byte for byte.
# Usage: bash bench/tools/compare_modelfile.sh <model files...>
cd "$(dirname "$0")/../.." || exit 1
mkdir -p bench/tools/out
ROPE=./build/rust-cpu/release/rope.exe; fail=0
for f in "$@"; do
  b=$(basename "$f"); o=bench/tools/out/$b
  s=$(date +%s.%N); $ROPE inspect "$f" > "$o.rope.txt"; rc=$?; e=$(date +%s.%N)
  PYTHONUTF8=1 python bench/tools/ref_modelfile.py "$f" > "$o.ref.txt" 2>/dev/null
  res="plain-ref:$(cmp -s "$o.rope.txt" "$o.ref.txt" && echo IDENTICAL || echo DIFFER)"
  if head -c4 "$f" | grep -q GGUF; then
    if PYTHONUTF8=1 python bench/tools/ref_modelfile.py "$f" --gguf-py > "$o.ggufpy.txt" 2>"$o.ggufpy.err"; then
      res="$res gguf-py:$(cmp -s "$o.rope.txt" "$o.ggufpy.txt" && echo IDENTICAL || echo DIFFER)"
    else res="$res gguf-py:REFUSED($(tail -1 "$o.ggufpy.err" | cut -c1-60))"; fi
  fi
  case "$res" in *DIFFER*) fail=1;; esac; [ $rc -ne 0 ] && fail=1
  printf '%-45s %5d tensors  rope %.2fs  %s\n' "$b" "$(($(wc -l < "$o.rope.txt") - 1))" "$(awk "BEGIN{print $e - $s}")" "$res"
done
exit $fail
