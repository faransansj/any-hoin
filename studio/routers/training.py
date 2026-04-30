"""학습 라우터 — TrainJob 관리 + 메트릭 조회."""

import json
from pathlib import Path

from fastapi import APIRouter, WebSocket

import xpu_compat
from studio.jobs.train_job import TrainJob

PROJECT_ROOT = "."
CHECKPOINT_DIR = Path("./checkpoints")

AVAILABLE_BACKBONES = [
    {"key": "swin_tiny_patch4_window7_224",    "label": "Swin-T · 28M",     "params_m": 28,  "description": "기본 백본 — 빠른 학습"},
    {"key": "swin_small_patch4_window7_224",   "label": "Swin-S · 50M",     "params_m": 50,  "description": "Tiny 대비 +3~5% 정확도"},
    {"key": "swin_base_patch4_window7_224",    "label": "Swin-B · 88M",     "params_m": 88,  "description": "고정밀도 — VRAM 2배 이상"},
    {"key": "convnext_small.fb_in22k_ft_in1k", "label": "ConvNeXt-S · 50M", "params_m": 50,  "description": "IN-22K 프리트레인, 빠른 수렴"},
    {"key": "convnext_base.fb_in22k_ft_in1k",  "label": "ConvNeXt-B · 89M", "params_m": 89,  "description": "IN-22K 프리트레인, 고정밀도"},
]
DEFAULT_BACKBONE = "swin_tiny_patch4_window7_224"

router = APIRouter(prefix="/training", tags=["training"])
_job   = TrainJob()


def _device_status() -> dict:
    torch_version = xpu_compat.torch_version() or "unavailable"
    cuda_available = xpu_compat.cuda_available()
    mps_available = xpu_compat.mps_available()
    xpu = xpu_compat.xpu_status()

    devices = [
        {"key": "auto", "label": "auto", "available": True, "reason": None},
        {
            "key": "cuda",
            "label": "CUDA",
            "available": cuda_available,
            "reason": None if cuda_available else "CUDA를 사용할 수 없습니다.",
        },
        {
            "key": "mps",
            "label": "MPS (Apple)",
            "available": mps_available,
            "reason": None if mps_available else "Apple MPS를 사용할 수 없습니다.",
        },
        {
            "key": "xpu",
            "label": "XPU (Intel Arc)",
            "available": xpu["available"],
            "reason": xpu["reason"],
        },
        {"key": "cpu", "label": "CPU", "available": True, "reason": None},
    ]

    return {
        "torch_version": torch_version,
        "ipex_version": xpu_compat.ipex_version(),
        "devices": devices,
    }


def _artifact_entry(filename: str) -> dict:
    path = CHECKPOINT_DIR / filename
    if not path.exists():
        return {"exists": False, "filename": filename, "size_mb": None, "mtime": None}
    stat = path.stat()
    return {
        "exists": True,
        "filename": filename,
        "size_mb": round(stat.st_size / 1024 ** 2, 1),
        "mtime": stat.st_mtime,
    }


def _training_config() -> dict:
    path = CHECKPOINT_DIR / "config.json"
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


@router.get("/status")
def get_status():
    return _job.status()


@router.get("/devices")
def get_devices():
    return _device_status()


@router.get("/backbones")
def get_backbones():
    return {"backbones": AVAILABLE_BACKBONES, "default": DEFAULT_BACKBONE}


@router.get("/artifacts")
def get_artifacts():
    config = _training_config()
    return {
        "best_model": _artifact_entry("best_model.pth"),
        "checkpoint": _artifact_entry("checkpoint.pth"),
        "config_best_val_acc": config.get("best_val_acc"),
        "config_test_acc": config.get("test_acc"),
        "num_classes": config.get("num_classes"),
        "config_backbone": config.get("backbone") or config.get("model"),
    }


@router.post("/start")
async def start_training(body: dict):
    await _job.start(body, PROJECT_ROOT)
    return {"started": True}


@router.post("/stop")
async def stop_training():
    await _job.stop()
    return {"stopped": True}


@router.get("/metrics")
def get_metrics():
    return {"metrics": _job.metrics, "best_val_acc": _job.best_val_acc}


@router.post("/logs/clear")
def clear_training_logs():
    _job.clear_buffer()
    return {"cleared": True}


@router.websocket("/logs")
async def training_logs(ws: WebSocket):
    await _job.connect_ws(ws)
