"""
HoloScope Studio API
포트 8001 | 크롤 → 라벨 → 학습 → 양자화 → 추론 전 단계 통합

개발: uv run python studio_api.py
프론트 빌드 후: uv run uvicorn studio_api:app --host 0.0.0.0 --port 8001
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from studio.routers import characters, crawl, export, images, inference, labels, training, segmentation

PROJECT_ROOT = Path(__file__).parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(
    title="HoloScope Studio",
    description="End-to-end ML Studio: Crawl → Label → Train → Export → Test",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── API 라우터 ──────────────────────────────────────────

for router in [characters.router, crawl.router, labels.router, images.router,
               training.router, export.router, inference.router, segmentation.router]:
    app.include_router(router, prefix="/api")

# ── 헬스체크 ────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok", "service": "holoscope-studio"}

# ── 프론트엔드 정적 파일 (빌드 후) ─────────────────────

STATIC_DIR = PROJECT_ROOT / "frontend" / "dist"
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "studio_api:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        reload_dirs=[str(PROJECT_ROOT / "studio")],
    )
