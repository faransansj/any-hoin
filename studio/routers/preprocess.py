"""학습 최적화 캐시 전처리 라우터."""

import shutil
from pathlib import Path

from fastapi import APIRouter, WebSocket

from studio.jobs.preprocess_job import PreprocessJob

PROJECT_ROOT = "."
DATA_DIR     = Path("./dataset/raw")
CACHE_DIR    = Path("./dataset/.cache")

router = APIRouter(prefix="/preprocess", tags=["preprocess"])
_job   = PreprocessJob()


def _cache_stats() -> dict:
    if not CACHE_DIR.exists():
        return {"total_images": 0, "total_bytes": 0, "classes": 0, "exists": False}
    total_images = 0
    total_bytes  = 0
    classes      = 0
    for cls_dir in CACHE_DIR.iterdir():
        if not cls_dir.is_dir():
            continue
        classes += 1
        for f in cls_dir.rglob("*.jpg"):
            total_images += 1
            total_bytes  += f.stat().st_size
    return {
        "exists":       CACHE_DIR.exists(),
        "total_images": total_images,
        "total_bytes":  total_bytes,
        "classes":      classes,
    }


@router.get("/status")
def get_status():
    return _job.status()


@router.get("/cache-stats")
def get_cache_stats():
    return _cache_stats()


@router.post("/start")
async def start_preprocess(body: dict):
    await _job.start(body, PROJECT_ROOT)
    return {"started": True}


@router.post("/stop")
async def stop_preprocess():
    await _job.stop()
    return {"stopped": True}


@router.delete("/cache")
def delete_cache():
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)
    return {"deleted": True}


@router.websocket("/logs")
async def preprocess_logs(ws: WebSocket):
    await _job.connect_ws(ws)
