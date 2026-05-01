"""얼굴 크롭 전처리 라우터 — SegJob 관리."""

from fastapi import APIRouter, WebSocket

from studio.jobs.seg_job import SegJob

PROJECT_ROOT = "."

router = APIRouter(prefix="/segmentation", tags=["segmentation"])
_job = SegJob()


@router.get("/status")
def seg_status():
    return _job.status()


@router.post("/start")
async def seg_start(params: dict):
    await _job.start(params, project_root=PROJECT_ROOT)
    return {"status": "started"}


@router.post("/stop")
async def seg_stop():
    await _job.stop()
    return {"status": "stopped"}


@router.websocket("/logs")
async def seg_logs(ws: WebSocket):
    await _job.connect_ws(ws)
