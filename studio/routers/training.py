"""학습 라우터 — TrainJob 관리 + 메트릭 조회."""

from fastapi import APIRouter, WebSocket

from studio.jobs.train_job import TrainJob

PROJECT_ROOT = "."

router = APIRouter(prefix="/training", tags=["training"])
_job   = TrainJob()


@router.get("/status")
def get_status():
    return _job.status()


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
