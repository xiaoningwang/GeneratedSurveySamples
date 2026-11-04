@echo off
chcp 65001 >nul
cd /d "%~dp0"
if exist "%~dp0.venv\Scripts\python.exe" (
  "%~dp0.venv\Scripts\python.exe" run_paper.py --output-dir reproduction_results --verify
) else (
  python run_paper.py --output-dir reproduction_results --verify
)
pause
