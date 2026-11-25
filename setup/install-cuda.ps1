# Installs the CUDA 12.9 toolkit WITHOUT the display driver (keeps the newer installed driver). Run elevated.
# Why 12.9 and not 13.x: CUDA 13 dropped Maxwell/Pascal (GTX 10xx, MX150-350), which many low-end laptops have.
$ErrorActionPreference = 'Continue'
$log = "$PSScriptRoot\install-cuda.log"
Start-Transcript -Path $log -Force | Out-Null
$comp = 'nvcc_12.9 cudart_12.9 nvrtc_12.9 nvrtc_dev_12.9 cublas_12.9 cublas_dev_12.9 cuda_profiler_api_12.9 nvtx_12.9 nvml_dev_12.9 cuobjdump_12.9 nvdisasm_12.9 visual_studio_integration_12.9'
winget install --id Nvidia.CUDA --version 12.9 --exact --silent --accept-package-agreements --accept-source-agreements --override "-s $comp"
Write-Host "winget exit: $LASTEXITCODE"
$nvcc = Get-ChildItem 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\*\bin\nvcc.exe' -ErrorAction SilentlyContinue
if ($nvcc) { & $nvcc[0].FullName --version; Write-Host "RESULT PASS" } else { Write-Host "RESULT FAIL: nvcc not found" }
Stop-Transcript | Out-Null
