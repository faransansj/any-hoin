$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

$Python = Join-Path ".venv" "Scripts\python.exe"
if (-not (Test-Path $Python)) {
    Write-Error "[error] .venv not found. Run 'uv sync' first."
    exit 1
}

& $Python start.py @args
exit $LASTEXITCODE
