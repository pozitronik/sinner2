@echo off
rem sinner2 installer -- bootstraps uv, then runs the wizard. Double-click to run.
setlocal
cd /d "%~dp0"

where uv >/dev/null 2>/dev/null
if errorlevel 1 (
    echo Installing uv ^(one-time^)...
    powershell -ExecutionPolicy ByPass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    set "PATH=%USERPROFILE%\.local\bin;%PATH%"
)

uv python install 3.12 >/dev/null 2>/dev/null
uv run --no-project --python 3.12 installer\wizard.py %*
echo.
pause
