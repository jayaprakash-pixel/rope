# Deploys and (re)starts `rope serve` under the PM2 service, then saves the process list. Run elevated.
# Copies the CUDA build of rope.exe to bin/ (the service runs the copy, so cargo can rebuild meanwhile).
$ErrorActionPreference = 'Continue'
$root = $PSScriptRoot
New-Item -ItemType Directory -Force "$root\bin" | Out-Null
Start-Transcript -Path "$root\bin\pm2-rope.log" -Force | Out-Null
$env:PM2_HOME = 'C:\ProgramData\pm2\home'
$pm2 = "$env:APPDATA\npm\pm2.cmd"
& $pm2 delete rope-serve *> $null
Copy-Item "$root\build\rust-cuda\release\rope.exe" "$root\bin\rope.exe" -Force
New-Item -ItemType Directory -Force "$root\gallery" | Out-Null
& $pm2 start "$root\ecosystem.rope.config.cjs"
& $pm2 save
Start-Sleep 4
try {
  $h = Invoke-WebRequest 'http://127.0.0.1:4003/api/health' -UseBasicParsing -TimeoutSec 5 -Headers @{ Host = '127.0.0.1:4003' }
  Write-Host "health: $($h.Content)"
  $d = (Invoke-WebRequest 'http://127.0.0.1:4003/api/device' -UseBasicParsing -TimeoutSec 60).Content | ConvertFrom-Json
  Write-Host "device as the service sees it: cpu_path $($d.cpu_path), gpu $($d.gpu.name)"
  Write-Host "RESULT PASS"
} catch { Write-Host "health: FAILED $_"; Write-Host "RESULT FAIL" }
& $pm2 ls
Stop-Transcript | Out-Null
