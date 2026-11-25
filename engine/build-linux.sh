#!/bin/bash
# Build the Linux program on Ubuntu 22.04 (glibc 2.35), so that it runs on Ubuntu 22.04 and newer.
# A program built on a newer Linux needs that newer glibc and fails on older laptops.
# Needs Docker. Run from the engine folder:  bash build-linux.sh        Output: engine/linux-out/rope
set -e
export MSYS_NO_PATHCONV=1   # Git Bash on Windows must not rewrite the container paths
HERE="$(cd "$(dirname "$0")" && (pwd -W 2>/dev/null || pwd))"
mkdir -p "$HERE/linux-out"
docker run --rm -v "$HERE:/src:ro" -v "$HERE/linux-out:/out" -v rope-cargo:/root/.cargo -v rope-rustup:/root/.rustup \
  ubuntu:22.04 bash -c '
    set -e
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq && apt-get install -y -qq gcc make cmake curl ca-certificates rsync >/dev/null
    [ -x /root/.cargo/bin/cargo ] || curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal >/dev/null 2>&1
    . /root/.cargo/env
    mkdir -p /build && rsync -a --exclude target --exclude linux-out /src/ /build/
    cd /build && cargo build --release 2>&1 | tail -5
    cp target/release/rope /out/rope
    cp -L /usr/lib/x86_64-linux-gnu/libgomp.so.1 /out/libgomp.so.1
    objdump -T /out/rope | grep -o "GLIBC_[0-9.]*" | sort -V | uniq | tail -1'
