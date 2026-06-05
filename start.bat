@echo off
setlocal
cd /d "%~dp0"

set "PYTHON=.venv\Scripts\python.exe"
if not exist "%PYTHON%" (
  echo [error] .venv not found. Run "uv sync" first.
  exit /b 1
)

"%PYTHON%" start.py %*
exit /b %ERRORLEVEL%
