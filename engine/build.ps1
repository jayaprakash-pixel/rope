# One build command for rope: Rust platform + C/CUDA kernels (via build.rs -> CMake).
#   powershell -File engine\build.ps1   -> build\rust-cpu\release\rope.exe [+ build\rust-cuda\release\rope.exe]
# Cargo reads .cargo/config.toml (static CRT) from the CURRENT directory, so build from inside engine/.
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
  $out = Join-Path $PSScriptRoot '..\build'
  cargo build --release --target-dir "$out\rust-cpu"
  if ($LASTEXITCODE) { exit $LASTEXITCODE }
  if (Test-Path 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA') {
    cargo build --release --features cuda --target-dir "$out\rust-cuda"
    if ($LASTEXITCODE) { exit $LASTEXITCODE }
  } else { Write-Host 'CUDA toolkit not found: CUDA build skipped' }
} finally { Pop-Location }
