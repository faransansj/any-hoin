"""크롤링 라우터 — characters.json 기반 (Hololive 비의존)."""

import os
from pathlib import Path

import requests as _req
from fastapi import APIRouter, Query, WebSocket

import studio.characters as ch
from studio.jobs.crawl_job import CrawlJob

DATASET_DIR  = Path("./dataset/raw")
ALLOWED_EXT  = {".jpg", ".jpeg", ".png", ".webp"}
PROJECT_ROOT = "."

router = APIRouter(prefix="/crawl", tags=["crawl"])
_job   = CrawlJob()


def _count(key: str) -> int:
    d = DATASET_DIR / key
    if not d.exists():
        return 0
    return sum(1 for f in d.iterdir() if f.suffix.lower() in ALLOWED_EXT)


@router.get("/status")
def get_status():
    return _job.status()


@router.post("/start")
async def start_crawl(body: dict):
    """
    body:
      selected_keys: list[str]   # 크롤할 캐릭터 key 목록 (빈 배열 = 전체)
      min_images: int
      max_images: int
      workers: int
      username: str
      api_key: str
      output_dir: str
    """
    chars = ch.load()

    # 선택된 key만, 없으면 전체
    selected = body.get("selected_keys") or list(chars.keys())
    tags_dict = {k: chars[k]["tag"] for k in selected if k in chars}

    if not tags_dict:
        return {"error": "No characters selected. Add characters first."}

    params = {
        **body,
        "tags_dict": tags_dict,
    }
    await _job.start(params, PROJECT_ROOT)
    return {"started": True, "characters": len(tags_dict)}


@router.post("/stop")
async def stop_crawl():
    await _job.stop()
    return {"stopped": True}


@router.post("/logs/clear")
def clear_crawl_logs():
    _job.clear_buffer()
    return {"cleared": True}


@router.websocket("/logs")
async def crawl_logs(ws: WebSocket):
    await _job.connect_ws(ws)


# ── Danbooru 태그 검색 ────────────────────────────────────────

@router.get("/tags/search")
def search_tags(q: str = Query(default="", min_length=0), limit: int = 10):
    """Danbooru 태그 자동완성. category=4(캐릭터) 우선, post_count 내림차순."""
    q = q.strip()
    if len(q) < 2:
        return {"tags": []}

    login = os.getenv("DANBOORU_LOGIN", "")
    key   = os.getenv("DANBOORU_API_KEY", "")
    auth  = (login, key) if login else None

    try:
        r = _req.get(
            "https://danbooru.donmai.us/tags.json",
            params={
                "search[name_matches]": f"*{q}*",
                "search[order]":        "count",
                "search[category]":     4,          # 캐릭터 태그만
                "limit":                limit,
            },
            auth=auth,
            timeout=8,
            headers={"User-Agent": "HoloScope-Crawler/1.0"},
        )
        r.raise_for_status()
        return {
            "tags": [
                {"name": t["name"], "post_count": t["post_count"]}
                for t in r.json()
            ]
        }
    except Exception as e:
        return {"tags": [], "error": str(e)}


@router.get("/tags/validate/{tag}")
def validate_tag(tag: str):
    """태그 존재 여부 및 post_count 반환."""
    login = os.getenv("DANBOORU_LOGIN", "")
    key   = os.getenv("DANBOORU_API_KEY", "")
    auth  = (login, key) if login else None

    try:
        r = _req.get(
            "https://danbooru.donmai.us/tags.json",
            params={"search[name]": tag, "limit": 1},
            auth=auth,
            timeout=8,
            headers={"User-Agent": "HoloScope-Crawler/1.0"},
        )
        r.raise_for_status()
        data = r.json()
        if data:
            return {"valid": True, "post_count": data[0]["post_count"]}
        return {"valid": False, "post_count": 0}
    except Exception:
        return {"valid": None, "post_count": 0}
