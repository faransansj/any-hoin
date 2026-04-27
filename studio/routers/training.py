"""학습 라우터 — TrainJob 관리 + 메트릭 조회."""

import warnings
from importlib.metadata import PackageNotFoundError, version

from fastapi import APIRouter, WebSocket

from studio.jobs.train_job import TrainJob

PROJECT_ROOT = "."

router = APIRouter(prefix="/training", tags=["training"])
_job   = TrainJob()


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _device_status() -> dict:
    import torch

    torch_version = torch.__version__

    cuda_available = torch.cuda.is_available()
    mps_available = torch.backends.mps.is_available()
    xpu_build = "+xpu" in torch_version
    xpu_available = False
    xpu_reason = None

    if not xpu_build:
        xpu_reason = (
            f"현재 torch {torch_version}는 XPU 빌드가 아닙니다. "
            "Intel Arc 사용 시 uv sync --extra arc가 필요합니다."
        )
    else:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                xpu_available = torch.xpu.is_available()
        except Exception as exc:
            xpu_reason = f"torch.xpu 확인 실패: {exc}"
        if not xpu_available and xpu_reason is None:
            xpu_reason = "torch.xpu.is_available() == False 입니다. 드라이버/Level Zero/render 그룹을 확인하세요."

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
            "available": xpu_available,
            "reason": xpu_reason,
        },
        {"key": "cpu", "label": "CPU", "available": True, "reason": None},
    ]

    return {
        "torch_version": torch_version,
        "ipex_version": _package_version("intel-extension-for-pytorch"),
        "devices": devices,
    }


@router.get("/status")
def get_status():
    return _job.status()


@router.get("/devices")
def get_devices():
    return _device_status()


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
