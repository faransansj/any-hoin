#!/usr/bin/env python3
"""Cross-platform launcher for HoloScope Studio."""

from __future__ import annotations

import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from urllib.request import urlopen


PROJECT_ROOT = Path(__file__).resolve().parent
BACKEND_PORT = 8001
FRONTEND_PORT = 5173


def venv_python() -> Path:
    if os.name == "nt":
        return PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    return PROJECT_ROOT / ".venv" / "bin" / "python"


def require_python() -> str:
    python = venv_python()
    if not python.exists():
        print("[error] .venv not found. Run `uv sync` or `uv sync --extra arc/cuda/rocm` first.")
        raise SystemExit(1)
    return str(python)


def port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _pids_on_port_posix(port: int) -> list[int]:
    try:
        result = subprocess.run(
            ["lsof", "-tiTCP:%d" % port, "-sTCP:LISTEN"],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return []
    return [int(line) for line in result.stdout.splitlines() if line.strip().isdigit()]


def _pids_on_port_windows(port: int) -> list[int]:
    result = subprocess.run(["netstat", "-ano"], check=False, capture_output=True, text=True)
    pids: set[int] = set()
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[0].upper().startswith("TCP") and parts[3].upper() == "LISTENING":
            local = parts[1]
            if local.rsplit(":", 1)[-1] == str(port) and parts[-1].isdigit():
                pids.add(int(parts[-1]))
    return sorted(pids)


def pids_on_port(port: int) -> list[int]:
    return _pids_on_port_windows(port) if os.name == "nt" else _pids_on_port_posix(port)


def kill_pid(pid: int, force: bool = False) -> None:
    try:
        if os.name == "nt":
            cmd = ["taskkill", "/PID", str(pid), "/T"]
            if force:
                cmd.append("/F")
            subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            os.kill(pid, signal.SIGKILL if force else signal.SIGTERM)
    except ProcessLookupError:
        pass
    except PermissionError:
        print(f"[warn] No permission to stop PID {pid}.")


def free_port(port: int, label: str) -> None:
    pids = pids_on_port(port)
    if not pids:
        return
    print(f"[info] Stopping existing {label} process on port {port}: {pids}")
    for pid in pids:
        kill_pid(pid)
    for _ in range(20):
        time.sleep(0.2)
        pids = pids_on_port(port)
        if not pids:
            return
    print(f"[warn] Port {port} still busy; forcing remaining process(es): {pids}")
    for pid in pids:
        kill_pid(pid, force=True)
    time.sleep(0.5)
    pids = pids_on_port(port)
    if pids:
        print(f"[error] Port {port} is still in use by PID(s): {pids}")
        raise SystemExit(1)


def ensure_frontend_deps() -> None:
    if (PROJECT_ROOT / "frontend" / "node_modules").exists():
        return
    print("[info] Installing frontend dependencies...")
    subprocess.run(["npm", "install", "--prefix", "frontend"], cwd=PROJECT_ROOT, check=True)


def health_ready() -> bool:
    try:
        with urlopen(f"http://127.0.0.1:{BACKEND_PORT}/api/health", timeout=1) as response:
            return 200 <= response.status < 300
    except Exception:
        return False


def start_backend(python: str) -> subprocess.Popen:
    kwargs = {"cwd": PROJECT_ROOT}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
    else:
        kwargs["start_new_session"] = True
    return subprocess.Popen([python, "studio_api.py"], **kwargs)


def stop_backend(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    print()
    print("[info] Shutting down backend...")
    try:
        if os.name == "nt":
            process.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
        else:
            os.killpg(process.pid, signal.SIGTERM)
    except Exception:
        process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()


def run_web() -> int:
    python = require_python()
    free_port(BACKEND_PORT, "backend")
    free_port(FRONTEND_PORT, "frontend")
    ensure_frontend_deps()

    backend: subprocess.Popen | None = None
    try:
        print(f"[info] Starting backend on http://localhost:{BACKEND_PORT} ...")
        backend = start_backend(python)
        for _ in range(10):
            if backend.poll() is not None:
                print("[error] Backend exited during startup.")
                return backend.returncode or 1
            if health_ready():
                break
            time.sleep(1)
        else:
            print(f"[error] Backend did not become ready on http://localhost:{BACKEND_PORT} within 10 seconds.")
            return 1

        print(f"[info] Starting frontend on http://localhost:{FRONTEND_PORT} ...")
        return subprocess.call(["npm", "run", "dev", "--prefix", "frontend"], cwd=PROJECT_ROOT)
    except KeyboardInterrupt:
        print()
        return 130
    finally:
        stop_backend(backend)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    mode = argv[0] if argv else "--tui"
    if mode in {"-h", "--help"}:
        print("Usage: python start.py [--tui|--cli|--web|--doctor|--device]")
        return 0
    if mode not in {"--tui", "--cli", "--web", "--doctor", "--device"}:
        print(f"[error] Unknown option: {mode}")
        print("Usage: python start.py [--tui|--cli|--web|--doctor|--device]")
        return 2
    if len(argv) > 1:
        print(f"[error] Unexpected extra arguments: {' '.join(argv[1:])}")
        return 2
    python = require_python()

    if mode in {"--tui", "--cli"}:
        return subprocess.call([python, "studio_cli.py", "tui"], cwd=PROJECT_ROOT)
    if mode == "--doctor":
        return subprocess.call([python, "studio_cli.py", "doctor"], cwd=PROJECT_ROOT)
    if mode == "--device":
        return subprocess.call([python, "studio_cli.py", "device"], cwd=PROJECT_ROOT)
    if mode == "--web":
        return run_web()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
