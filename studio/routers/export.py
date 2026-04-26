"""모델 양자화 / ONNX export 라우터."""

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, WebSocket
from fastapi.responses import FileResponse

from studio.jobs.export_job import OnnxJob, QuantJob

CHECKPOINT_DIR = Path("./checkpoints")
PROJECT_ROOT   = "."

router     = APIRouter(prefix="/export", tags=["export"])
_quant_job = QuantJob()
_onnx_job  = OnnxJob()

_QUANT_FILES = {
    "fp16": "best_model_fp16.pth",
    "int8": "best_model_int8.pth",
    "int4": "best_model_int4.pth",
    "int2": "best_model_int2.pth",
}


def _entry(fname: str) -> dict:
    p = CHECKPOINT_DIR / fname
    try:
        size_mb = round(p.stat().st_size / 1024 ** 2, 1)
        exists = True
    except FileNotFoundError:
        size_mb = None
        exists = False
    return {
        "exists":   exists,
        "size_mb":  size_mb,
        "filename": fname,
    }


# ── 모델 목록 ─────────────────────────────────────────────

def _load_config_acc() -> float | None:
    p = CHECKPOINT_DIR / "config.json"
    if not p.exists():
        return None
    try:
        with open(p) as f:
            cfg = json.load(f)
        return cfg.get("test_acc")
    except Exception:
        return None


@router.get("/models")
def list_models():
    result = {"fp32": _entry("best_model.pth"), "onnx": _entry("best_model.onnx")}
    for fmt, fname in _QUANT_FILES.items():
        result[fmt] = _entry(fname)
    return {"models": result, "config_acc": _load_config_acc()}


@router.get("/metrics")
def get_quant_metrics():
    return {"metrics": _quant_job.metrics}


# ── 양자화 ────────────────────────────────────────────────

@router.post("/quant")
async def export_quant(body: dict = {}):
    fmt = body.get("format", "fp16")
    await _quant_job.start(fmt, PROJECT_ROOT)
    return {"started": True, "format": fmt}


@router.post("/quant/stop")
async def stop_quant():
    await _quant_job.stop()
    return {"stopped": True}


# ── ONNX ─────────────────────────────────────────────────

@router.post("/onnx")
async def export_onnx(body: dict = {}):
    opset = int(body.get("opset", 18))
    await _onnx_job.start(opset, PROJECT_ROOT)
    return {"started": True}


@router.post("/onnx/stop")
async def stop_onnx():
    await _onnx_job.stop()
    return {"stopped": True}


# ── 상태 ─────────────────────────────────────────────────

@router.get("/status")
def get_status():
    return {"quant": _quant_job.status(), "onnx": _onnx_job.status()}


# ── 다운로드 ─────────────────────────────────────────────

@router.get("/download/{filename}")
def download_model(filename: str):
    if ".." in filename or "/" in filename:
        raise HTTPException(400, "Invalid filename")
    path = CHECKPOINT_DIR / filename
    if not path.exists():
        raise HTTPException(404, "Model not found")
    return FileResponse(str(path), filename=filename, media_type="application/octet-stream")


# ── WebSocket 로그 ────────────────────────────────────────

@router.post("/logs/clear/quant")
def clear_quant_logs():
    _quant_job.clear_buffer()
    return {"cleared": True}


@router.post("/logs/clear/onnx")
def clear_onnx_logs():
    _onnx_job.clear_buffer()
    return {"cleared": True}


@router.websocket("/logs/quant")
async def quant_logs(ws: WebSocket):
    await _quant_job.connect_ws(ws)


@router.websocket("/logs/onnx")
async def onnx_logs(ws: WebSocket):
    await _onnx_job.connect_ws(ws)
