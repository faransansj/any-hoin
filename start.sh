#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# .venv/bin/python을 사용해 uv run의 deps 재정렬(CUDA 재설치) 방지
PYTHON=".venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    echo "[error] .venv not found. Run 'uv sync' (or 'uv sync --extra arc/cuda/rocm') first."
    exit 1
fi

if ! command -v lsof >/dev/null 2>&1; then
    echo "[error] lsof is required to check and free dev server ports."
    exit 1
fi

BACKEND_PID=""

cleanup() {
    local exit_code=$?
    trap - EXIT INT TERM

    if [[ -n "$BACKEND_PID" ]]; then
        echo
        echo "[info] Shutting down..."
        kill -- "-$BACKEND_PID" 2>/dev/null || kill "$BACKEND_PID" 2>/dev/null || true
        wait "$BACKEND_PID" 2>/dev/null || true
    fi

    exit "$exit_code"
}
trap cleanup EXIT INT TERM

free_port() {
    local port="$1"
    local label="$2"
    local pids=()

    mapfile -t pids < <(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)
    if ((${#pids[@]} == 0)); then
        return
    fi

    echo "[info] Killing existing $label process on port $port..."
    kill "${pids[@]}" 2>/dev/null || true

    for _ in $(seq 1 20); do
        mapfile -t pids < <(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)
        if ((${#pids[@]} == 0)); then
            return
        fi
        sleep 0.2
    done

    echo "[warn] Port $port is still busy; forcing remaining process(es)..."
    kill -9 "${pids[@]}" 2>/dev/null || true
    sleep 0.5

    mapfile -t pids < <(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)
    if ((${#pids[@]} != 0)); then
        echo "[error] Port $port is still in use by PID(s): ${pids[*]}"
        exit 1
    fi
}

# 포트 충돌 시 기존 개발 서버 종료
free_port 8001 "backend"
free_port 5173 "frontend"

# 프론트엔드 의존성 설치 (node_modules 없을 때만)
if [[ ! -d "frontend/node_modules" ]]; then
    echo "[info] Installing frontend dependencies..."
    npm install --prefix frontend
fi

# 백엔드 백그라운드 실행
echo "[info] Starting backend on http://localhost:8001 ..."
setsid "$PYTHON" studio_api.py &
BACKEND_PID=$!

# 백엔드 준비 대기 (최대 10초)
BACKEND_READY=0
for i in $(seq 1 10); do
    if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
        echo "[error] Backend exited during startup."
        wait "$BACKEND_PID" 2>/dev/null || true
        exit 1
    fi
    if curl -sf http://localhost:8001/api/health &>/dev/null; then
        BACKEND_READY=1
        break
    fi
    sleep 1
done

if [[ "$BACKEND_READY" -ne 1 ]]; then
    echo "[error] Backend did not become ready on http://localhost:8001 within 10 seconds."
    exit 1
fi

# 프론트엔드 포그라운드 실행
echo "[info] Starting frontend on http://localhost:5173 ..."
npm run dev --prefix frontend
