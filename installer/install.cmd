@echo off
rem Rope installer (runs from the folder Rope-Setup.exe extracts to). For this user only: no administrator rights.
rem Program: %LOCALAPPDATA%\Programs\Rope   Data (models, pictures): %LOCALAPPDATA%\Rope
rem Test overrides: ROPE_INSTALL_DIR, ROPE_DATA_DIR, ROPE_INSTALL_GET=yes|no (no question), ROPE_INSTALL_START=no
setlocal
set "SRC=%~dp0"
set "DST=%LOCALAPPDATA%\Programs\Rope"
if not "%ROPE_INSTALL_DIR%"=="" set "DST=%ROPE_INSTALL_DIR%"
set "DATA=%LOCALAPPDATA%\Rope"
if not "%ROPE_DATA_DIR%"=="" set "DATA=%ROPE_DATA_DIR%"

echo ============================================================
echo ROPE SETUP
echo Rope makes AI pictures on this laptop. It works offline after the model download.
echo It installs for this user only and needs no administrator rights.
echo    Program:          %DST%
echo    Models, pictures: %DATA%
echo ============================================================

rem Stop a running copy of an older Rope from this folder, so its files can be replaced.
powershell -NoProfile -Command "Get-Process rope -ErrorAction SilentlyContinue | Where-Object { $_.Path -like '%DST%\*' } | Stop-Process -Force" >nul 2>nul

if not exist "%DST%\catalog" mkdir "%DST%\catalog"
if not exist "%DATA%\models" mkdir "%DATA%\models"
rem The NVIDIA build when an NVIDIA driver is present, else the build for every 64-bit Windows laptop.
set "EXE=rope.exe"
where nvidia-smi >nul 2>nul && set "EXE=rope-cuda.exe"
copy /y "%SRC%%EXE%" "%DST%\rope.exe" >nul || goto fail
copy /y "%SRC%models.json" "%DST%\catalog\models.json" >nul || goto fail
copy /y "%SRC%uninstall.cmd" "%DST%\uninstall.cmd" >nul || goto fail
if exist "%DST%\app" rmdir /s /q "%DST%\app"
mkdir "%DST%\app"
tar -xf "%SRC%app.zip" -C "%DST%\app" || goto fail
echo Program files copied (%EXE%).

rem Start menu entry "Rope", and the entry in Settings - Apps (with Uninstall).
powershell -NoProfile -Command "$l = Join-Path ([Environment]::GetFolderPath('Programs')) 'Rope.lnk'; $s = (New-Object -ComObject WScript.Shell).CreateShortcut($l); $s.TargetPath = '%DST%\rope.exe'; $s.Arguments = 'app'; $s.WorkingDirectory = '%DST%'; $s.Description = 'Rope: AI pictures on this laptop'; $s.Save()" || goto fail
set "KEY=HKCU\Software\Microsoft\Windows\CurrentVersion\Uninstall\Rope"
reg add "%KEY%" /v DisplayName /d "Rope" /f >nul
reg add "%KEY%" /v DisplayVersion /d "0.1.0" /f >nul
reg add "%KEY%" /v Publisher /d "Rope" /f >nul
reg add "%KEY%" /v InstallLocation /d "%DST%" /f >nul
reg add "%KEY%" /v DisplayIcon /d "%DST%\rope.exe" /f >nul
reg add "%KEY%" /v UninstallString /d "cmd.exe /c \"\"%DST%\uninstall.cmd\"\"" /f >nul
reg add "%KEY%" /v NoModify /t REG_DWORD /d 1 /f >nul
reg add "%KEY%" /v NoRepair /t REG_DWORD /d 1 /f >nul
echo Start menu entry "Rope" added.

rem The model: the planner reads this laptop and picks the model that fits it.
echo.
echo Reading this laptop and choosing the model that fits it...
"%DST%\rope.exe" plan --models-dir "%DATA%\models" --catalog "%DST%\catalog\models.json" 2>nul
if /i "%ROPE_INSTALL_GET%"=="no" goto noget
if /i "%ROPE_INSTALL_GET%"=="yes" goto get
echo.
echo Rope needs this model once (the size is in the plan above). The download uses the internet and can take a while.
choice /c YN /m "Download it now"
if errorlevel 2 goto noget
:get
"%DST%\rope.exe" get --models-dir "%DATA%\models" --catalog "%DST%\catalog\models.json" || echo The download did not finish. Rope downloads the model again when you make the first picture.
goto started
:noget
echo No download now. Rope downloads the model when you make the first picture.

:started
echo.
echo ============================================================
echo DONE. Rope is installed. Start it from the Start menu: "Rope".
echo To remove it: Settings - Apps - Rope - Uninstall.
echo ============================================================
if /i "%ROPE_INSTALL_START%"=="no" goto end
start "Rope" "%DST%\rope.exe" app
goto end

:fail
echo.
echo SETUP FAILED. The message above says why. Nothing outside %DST% was changed.
pause
exit /b 1

:end
endlocal
exit /b 0
