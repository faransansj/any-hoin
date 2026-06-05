#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON=".venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    echo "[error] .venv not found. Run 'uv sync' (or 'uv sync --extra arc/cuda/rocm') first."
    exit 1
fi

exec "$PYTHON" start.py "$@"
