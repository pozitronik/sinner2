@echo off
cd /d "%~dp0"
uv run --no-project --python 3.12 installer\wizard.py --doctor %*
echo.
pause
