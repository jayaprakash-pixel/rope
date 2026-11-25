# Builds build\installer\Rope-Setup.exe with IExpress (part of Windows: no download, no installer tool to install).
#   powershell -File installer\build-installer.ps1        (run engine\build.ps1 and `npm run build` in app\ first)
# The setup file holds: rope.exe (CPU), rope-cuda.exe (NVIDIA), app.zip (the built app), models.json (catalog),
# install.cmd and uninstall.cmd. It extracts to a temp folder and runs install.cmd. No model is inside: install.cmd
# lets the planner choose the model for the laptop and downloads it after a question.
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$out = Join-Path $root 'build\installer'
$pay = Join-Path $out 'payload'
if (Test-Path $pay) { Remove-Item $pay -Recurse -Force }
New-Item -ItemType Directory -Force $pay | Out-Null

$cpu = Join-Path $root 'build\rust-cpu\release\rope.exe'
$cuda = Join-Path $root 'build\rust-cuda\release\rope.exe'
$dist = Join-Path $root 'app\dist'
foreach ($f in $cpu, $cuda, (Join-Path $dist 'index.html')) { if (-not (Test-Path $f)) { throw "missing $f (build it first)" } }
Copy-Item $cpu (Join-Path $pay 'rope.exe')
Copy-Item $cuda (Join-Path $pay 'rope-cuda.exe')
Copy-Item (Join-Path $root 'catalog\models.json') (Join-Path $pay 'models.json')
# batch files must have CRLF line ends
foreach ($n in 'install.cmd', 'uninstall.cmd') {
  $t = [IO.File]::ReadAllText((Join-Path $PSScriptRoot $n)) -replace "`r`n", "`n" -replace "`n", "`r`n"
  [IO.File]::WriteAllText((Join-Path $pay $n), $t, (New-Object Text.ASCIIEncoding))
}
& tar.exe -a -c -f (Join-Path $pay 'app.zip') -C $dist .
if ($LASTEXITCODE) { throw 'tar failed' }

$files = Get-ChildItem $pay -File | Sort-Object Name
$target = Join-Path $out 'Rope-Setup.exe'
$sed = @"
[Version]
Class=IEXPRESS
SEDVersion=3
[Options]
PackagePurpose=InstallApp
ShowInstallProgramWindow=0
HideExtractAnimation=1
UseLongFileName=1
InsideCompressed=0
CAB_FixedSize=0
CAB_ResvCodeSigning=0
RebootMode=N
InstallPrompt=%InstallPrompt%
DisplayLicense=%DisplayLicense%
FinishMessage=%FinishMessage%
TargetName=%TargetName%
FriendlyName=%FriendlyName%
AppLaunched=%AppLaunched%
PostInstallCmd=%PostInstallCmd%
AdminQuietInstCmd=%AdminQuietInstCmd%
UserQuietInstCmd=%UserQuietInstCmd%
SourceFiles=SourceFiles
[Strings]
InstallPrompt=
DisplayLicense=
FinishMessage=
TargetName=$target
FriendlyName=Rope Setup
AppLaunched=install.cmd
PostInstallCmd=<None>
AdminQuietInstCmd=
UserQuietInstCmd=
$(($files | ForEach-Object -Begin { $i = 0 } -Process { "FILE$i=`"$($_.Name)`""; $i++ }) -join "`r`n")
[SourceFiles]
SourceFiles0=$pay\
[SourceFiles0]
$(($files | ForEach-Object -Begin { $i = 0 } -Process { "%FILE$i%="; $i++ }) -join "`r`n")
"@
$sedPath = Join-Path $out 'rope.sed'
[IO.File]::WriteAllText($sedPath, $sed, (New-Object Text.ASCIIEncoding))
if (Test-Path $target) { Remove-Item $target -Force }
$p = Start-Process "$env:SystemRoot\System32\iexpress.exe" -ArgumentList '/N', '/Q', $sedPath -Wait -PassThru -WindowStyle Hidden
if (-not (Test-Path $target)) { throw "IExpress made no file (exit $($p.ExitCode))" }
$h = (Get-FileHash $target).Hash.ToLower()
Write-Host ("Rope-Setup.exe  {0:N1} MB  sha256 {1}" -f ((Get-Item $target).Length / 1MB), $h)
