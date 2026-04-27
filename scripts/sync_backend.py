#!/usr/bin/env python3
"""Select and sync the PyTorch backend profile for this project.

PyTorch accelerator wheels are mutually exclusive in practice. This helper keeps
the user workflow unified: choose one backend profile, then always run Studio via
./start.sh or .venv/bin/python so uv does not silently re-sync another profile.
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

Backend = str
VALID_BACKENDS = {"auto", "cpu", "mps", "cuda", "arc", "rocm"}


def _run_text(cmd: list[str]) -> str:
    try:
        res = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=3)
    except Exception:
        return ""
    return f"{res.stdout}\n{res.stderr}".lower()


def detect_backend() -> Backend:
    system = platform.system().lower()
    if system == "darwin":
        return "mps"
    if system != "linux":
        return "cuda" if shutil.which("nvidia-smi") else "cpu"

    if shutil.which("nvidia-smi") and "gpu" in _run_text(["nvidia-smi", "-L"]):
        return "cuda"
    if shutil.which("rocminfo") and "amd" in _run_text(["rocminfo"]):
        return "rocm"

    lspci = _run_text(["lspci"]) if shutil.which("lspci") else ""
    if "intel" in lspci and "arc" in lspci:
        return "arc"

    # Level Zero without lspci is still a useful signal, but avoid treating every
    # Intel iGPU render node as Arc unless the user explicitly chooses arc.
    if shutil.which("sycl-ls") and "level_zero" in _run_text(["sycl-ls"]):
        return "arc"

    return "cpu"


def sync_command(backend: Backend) -> list[str]:
    if backend not in VALID_BACKENDS - {"auto"}:
        raise ValueError(f"unknown backend: {backend}")
    if backend in {"cpu", "mps"}:
        return ["uv", "sync"]
    return ["uv", "sync", "--extra", backend]


def _print_check() -> int:
    venv_python = Path(".venv/bin/python")
    if sys.platform == "win32":
        venv_python = Path(".venv/Scripts/python.exe")

    if venv_python.exists() and Path(sys.executable).resolve() != venv_python.resolve():
        code = (
            "import torch, sys, warnings; "
            "print(f'python: {sys.executable}'); "
            "print(f'torch:  {torch.__version__}'); "
            "print(f'cuda:   {torch.cuda.is_available()}'); "
            "print(f'mps:    {torch.backends.mps.is_available()}'); "
            "warnings.filterwarnings('ignore'); "
            "print('xpu:    ' + (str(torch.xpu.is_available()) if hasattr(torch, 'xpu') else 'unavailable'))"
        )
        env = {**os.environ, "PYTHONWARNINGS": "ignore"}
        return subprocess.run([str(venv_python), "-c", code], check=False, env=env).returncode

    try:
        import torch
    except Exception as exc:
        print(f"[error] torch import failed: {exc}", file=sys.stderr)
        return 1

    print(f"python: {sys.executable}")
    print(f"torch:  {torch.__version__}")
    print(f"cuda:   {torch.cuda.is_available()}")
    print(f"mps:    {torch.backends.mps.is_available()}")
    if hasattr(torch, "xpu"):
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                print(f"xpu:    {torch.xpu.is_available()}")
        except Exception as exc:
            print(f"xpu:    error: {exc}")
    else:
        print("xpu:    unavailable")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "backend",
        nargs="?",
        default="auto",
        choices=sorted(VALID_BACKENDS),
        help="backend profile to sync. auto detects a likely profile.",
    )
    parser.add_argument("--apply", action="store_true", help="run uv sync for the selected profile")
    parser.add_argument("--check", action="store_true", help="print current torch accelerator status after sync")
    args = parser.parse_args(argv)

    backend = detect_backend() if args.backend == "auto" else args.backend
    cmd = sync_command(backend)
    print(f"[backend] selected: {backend}", flush=True)
    print("[backend] sync command:", " ".join(cmd), flush=True)

    if args.apply:
        subprocess.run(cmd, check=True)
    else:
        print("[backend] dry-run only. Add --apply to execute.", flush=True)

    print("[backend] run Studio with ./start.sh or .venv/bin/python, not plain 'uv run python'.", flush=True)
    if args.check:
        return _print_check()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
