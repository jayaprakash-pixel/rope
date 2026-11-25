@echo off
rem Removes Rope for this user. Asks before it deletes the downloaded models and your pictures.
rem Test overrides: ROPE_DATA_DIR, ROPE_UNINSTALL_DATA=yes|no (no question)
setlocal
set "DST=%~dp0"
if "%DST:~-1%"=="\" set "DST=%DST:~0,-1%"
set "DATA=%LOCALAPPDATA%\Rope"
if not "%ROPE_DATA_DIR%"=="" set "DATA=%ROPE_DATA_DIR%"

echo Removing Rope from %DST% ...
rem Stop Rope if it runs from this folder (only this folder: other programs named rope are not touched).
powershell -NoProfile -Command "Get-Process rope -ErrorAction SilentlyContinue | Where-Object { $_.Path -like '%DST%\*' } | Stop-Process -Force" >nul 2>nul
powershell -NoProfile -Command "Remove-Item -LiteralPath (Join-Path ([Environment]::GetFolderPath('Programs')) 'Rope.lnk') -ErrorAction SilentlyContinue"
reg delete "HKCU\Software\Microsoft\Windows\CurrentVersion\Uninstall\Rope" /f >nul 2>nul

if /i "%ROPE_UNINSTALL_DATA%"=="yes" goto deldata
if /i "%ROPE_UNINSTALL_DATA%"=="no" goto keepdata
if not exist "%DATA%" goto keepdata
echo.
echo Your pictures and the downloaded models are in %DATA%
choice /c NY /m "Delete them too (the models are several GB; the pictures cannot be recovered)"
if errorlevel 2 goto deldata
:keepdata
if exist "%DATA%" echo Kept: %DATA%
goto delprog
:deldata
rmdir /s /q "%DATA%"
echo Deleted: %DATA%

:delprog
echo Rope is removed.
rem A batch file cannot delete the folder it runs from: a second cmd deletes it after this one ends.
start "" /b cmd /c "ping -n 3 127.0.0.1 >nul & rmdir /s /q "%DST%""
endlocal
exit /b 0
