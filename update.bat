@echo off
rem Check GitHub releases and, if a newer version exists, pull + re-sync.
setlocal
cd /d "%~dp0"

where uv >/dev/null 2>/dev/null
if errorlevel 1 (
    echo Installing uv ^(one-time^)...
    powershell -ExecutionPolicy ByPass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    set "PATH=%USERPROFILE%\.local\bin;%PATH%"
)

uv python install 3.12 >/dev/null 2>/dev/null
uv run --no-project --python 3.12 installer\wizard.py --update %*
echo.
pause
