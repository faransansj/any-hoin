"""Repository health checks for local CLI/TUI diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
import getpass
import json
import os
import platform
from pathlib import Path
import socket
import sys
from typing import Iterable

import xpu_compat


SEVERITY_ORDER = {"ok": 0, "warn": 1, "error": 2}


@dataclass(frozen=True)
class DoctorCheck:
    name: str
    status: str
    detail: str
    suggestion: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _check(name: str, status: str, detail: str, suggestion: str | None = None) -> DoctorCheck:
    return DoctorCheck(name=name, status=status, detail=detail, suggestion=suggestion)


def _package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None
    except Exception:
        return None


def _port_open(port: int, host: str = "127.0.0.1") -> bool | None:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            return sock.connect_ex((host, port)) == 0
    except OSError:
        return None


def _groups() -> set[str]:
    groups = {str(gid) for gid in os.getgroups()} if hasattr(os, "getgroups") else set()
    try:
        import grp

        by_gid = {grp.getgrgid(gid).gr_name for gid in os.getgroups()}
        groups.update(by_gid)
    except Exception:
        pass
    return groups


def _read_first_value(path: Path, prefix: str) -> str | None:
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith(prefix):
                return line.split(":", 1)[1].strip()
    except Exception:
        return None
    return None


def _meminfo() -> dict[str, int | None]:
    values: dict[str, int | None] = {"total_kib": None, "available_kib": None}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, raw = line.split(":", 1)
            if key in {"MemTotal", "MemAvailable"}:
                values["total_kib" if key == "MemTotal" else "available_kib"] = int(raw.split()[0])
    except Exception:
        pass
    return values


def _format_gib(kib: int | None) -> str:
    if kib is None:
        return "unknown"
    return f"{kib / 1024 / 1024:.1f} GiB"


def system_info() -> dict:
    """Return OS, CPU, and RAM details for doctor reports."""
    mem = _meminfo()
    cpu_model = _read_first_value(Path("/proc/cpuinfo"), "model name") or platform.processor() or "unknown"
    return {
        "os": platform.platform(),
        "system": platform.system() or "unknown",
        "release": platform.release() or "unknown",
        "machine": platform.machine() or "unknown",
        "cpu_model": cpu_model,
        "cpu_count": os.cpu_count() or 0,
        "ram_total_kib": mem["total_kib"],
        "ram_available_kib": mem["available_kib"],
        "ram_total": _format_gib(mem["total_kib"]),
        "ram_available": _format_gib(mem["available_kib"]),
    }


def current_device_info() -> dict:
    """Return the device this project will prefer at runtime."""
    info = {
        "selected": "cpu",
        "label": "CPU",
        "torch_version": xpu_compat.torch_version() or "unavailable",
        "xpu": xpu_compat.xpu_status(),
        "cuda_available": xpu_compat.cuda_available(),
        "mps_available": xpu_compat.mps_available(),
        "ipex_version": xpu_compat.ipex_version(),
    }
    try:
        device = xpu_compat.best_device()
        info["selected"] = getattr(device, "type", str(device))
        info["label"] = xpu_compat.device_label(device)
    except Exception as exc:
        info["label"] = f"CPU fallback ({exc})"
    return info


def run_doctor(project_root: Path | str = ".") -> dict:
    root = Path(project_root).resolve()
    checks: list[DoctorCheck] = []
    system = system_info()

    checks.append(
        _check(
            "system",
            "ok",
            f"{system['os']} | CPU cores: {system['cpu_count']} | RAM: {system['ram_total']}",
        )
    )

    checks.append(
        _check(
            "python",
            "ok" if sys.version_info[:2] == (3, 11) else "warn",
            f"{sys.version.split()[0]} at {sys.executable}",
            "Use Python 3.11 for this project." if sys.version_info[:2] != (3, 11) else None,
        )
    )

    torch_version = xpu_compat.torch_version()
    if torch_version:
        checks.append(_check("torch", "ok", torch_version))
    else:
        checks.append(_check("torch", "error", "PyTorch is not importable.", "Run `uv sync` or `uv sync --extra arc`."))

    xpu = xpu_compat.xpu_status()
    if xpu["available"]:
        checks.append(_check("xpu", "ok", "Intel Arc XPU is available."))
    elif xpu["build"]:
        checks.append(_check("xpu", "warn", xpu["reason"] or "XPU build installed but device unavailable.", "Check driver, Level Zero, and render/video groups."))
    else:
        checks.append(_check("xpu", "warn", xpu["reason"] or "Non-XPU torch build installed.", "For Arc, run `uv sync --extra arc`."))

    ipex_version = xpu_compat.ipex_version()
    if ipex_version is None:
        checks.append(_check("ipex", "warn", "intel-extension-for-pytorch is not installed.", "For Arc, run `uv sync --extra arc`."))
    elif torch_version and "+xpu" in torch_version:
        torch_mm = ".".join(torch_version.split("+")[0].split(".")[:2])
        ipex_mm = ".".join(ipex_version.split("+")[0].split(".")[:2])
        checks.append(
            _check(
                "ipex",
                "ok" if torch_mm == ipex_mm else "error",
                f"IPEX {ipex_version}, torch {torch_version}",
                "Keep torch and IPEX major.minor versions aligned." if torch_mm != ipex_mm else None,
            )
        )
    else:
        checks.append(_check("ipex", "ok", f"Installed: {ipex_version}; inactive on non-XPU torch."))

    triton_xpu = _package_version("triton-xpu")
    checks.append(
        _check(
            "triton-xpu",
            "ok" if triton_xpu is None or sys.modules.get("triton") is None else "warn",
            f"package={triton_xpu or 'not installed'}, triton_masked={sys.modules.get('triton') is None}",
            "Import `xpu_compat` before `torch` in XPU-capable entry points.",
        )
    )

    render_nodes = sorted(str(p) for p in Path("/dev/dri").glob("renderD*"))
    if render_nodes:
        checks.append(_check("render-device", "ok", ", ".join(render_nodes)))
    else:
        checks.append(_check("render-device", "warn", "No /dev/dri/renderD* nodes found.", "Install Intel GPU drivers/Level Zero."))

    groups = _groups()
    missing_groups = [name for name in ("render", "video") if name not in groups]
    checks.append(
        _check(
            "user-groups",
            "ok" if not missing_groups else "warn",
            f"{getpass.getuser()} groups include: {', '.join(sorted(groups)) or 'unknown'}",
            f"Add user to: {', '.join(missing_groups)}" if missing_groups else None,
        )
    )

    for rel_path, label in (("characters.json", "characters"), ("dataset/raw", "dataset"), ("checkpoints", "checkpoints")):
        path = root / rel_path
        checks.append(
            _check(
                label,
                "ok" if path.exists() else "warn",
                f"{rel_path} {'exists' if path.exists() else 'is missing'}",
                f"Create or generate `{rel_path}` before using related workflows." if not path.exists() else None,
            )
        )

    for port, label in ((8001, "backend-port"), (5173, "frontend-port")):
        port_open = _port_open(port)
        if port_open is None:
            checks.append(
                _check(
                    label,
                    "warn",
                    f"localhost:{port} could not be checked.",
                    "Run doctor outside restricted sandbox if port state matters.",
                )
            )
            continue
        checks.append(
            _check(
                label,
                "warn" if port_open else "ok",
                f"localhost:{port} {'is already in use' if port_open else 'is free'}",
                "Stop the existing process before starting Studio." if port_open else None,
            )
        )

    device = current_device_info()
    summary = max((check.status for check in checks), key=lambda status: SEVERITY_ORDER[status], default="ok")
    return {
        "summary": summary,
        "project_root": str(root),
        "system": system,
        "device": device,
        "checks": [check.to_dict() for check in checks],
    }


def doctor_json(project_root: Path | str = ".") -> str:
    return json.dumps(run_doctor(project_root), ensure_ascii=False, indent=2)


def count_by_status(checks: Iterable[dict]) -> dict[str, int]:
    counts = {"ok": 0, "warn": 0, "error": 0}
    for check in checks:
        status = check.get("status", "error")
        counts[status] = counts.get(status, 0) + 1
    return counts
