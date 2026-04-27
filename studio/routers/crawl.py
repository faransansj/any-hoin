"""크롤링 라우터 — characters.json 기반 (Hololive 비의존)."""

import os
import re
import time
from pathlib import Path

import requests as _req
from fastapi import APIRouter, HTTPException, Query, WebSocket

import studio.characters as ch
from studio.jobs.crawl_job import CrawlJob

DATASET_DIR  = Path("./dataset/raw")
ALLOWED_EXT  = {".jpg", ".jpeg", ".png", ".webp"}
PROJECT_ROOT = "."
DANBOORU_TAGS_URL = "https://danbooru.donmai.us/tags.json"
DANBOORU_AUTOCOMPLETE_URL = "https://danbooru.donmai.us/autocomplete.json"
DANBOORU_ALIASES_URL = "https://danbooru.donmai.us/tag_aliases.json"
DANBOORU_POSTS_URL = "https://danbooru.donmai.us/posts.json"

router = APIRouter(prefix="/crawl", tags=["crawl"])
_job   = CrawlJob()


def _count(key: str) -> int:
    d = DATASET_DIR / key
    if not d.exists():
        return 0
    return sum(1 for f in d.iterdir() if f.suffix.lower() in ALLOWED_EXT)


def _danbooru_auth():
    login = os.getenv("DANBOORU_LOGIN", "")
    key   = os.getenv("DANBOORU_API_KEY", "")
    return (login, key) if login and key else None


def _normalize_tag_query(q: str) -> str:
    q = q.strip().lower()
    q = re.sub(r"\s+", "_", q)
    q = re.sub(r"_+", "_", q)
    return q.strip("_")


def _fetch_tag_matches(pattern: str, limit: int, category: int | None = 4) -> list[dict]:
    params: dict[str, str | int] = {
        "search[name_matches]": pattern,
        "search[order]": "count",
        "limit": limit,
    }
    if category is not None:
        params["search[category]"] = category

    r = _req.get(
        DANBOORU_TAGS_URL,
        params=params,
        auth=_danbooru_auth(),
        timeout=8,
        headers={"User-Agent": "HoloScope-Crawler/1.0"},
    )
    r.raise_for_status()
    return r.json()


def _fetch_autocomplete(query: str, limit: int) -> list[dict]:
    r = _req.get(
        DANBOORU_AUTOCOMPLETE_URL,
        params={
            "search[query]": query,
            "search[type]": "tag_query",
            "limit": limit,
        },
        auth=_danbooru_auth(),
        timeout=8,
        headers={"User-Agent": "HoloScope-Crawler/1.0"},
    )
    r.raise_for_status()
    return r.json()


def _fetch_alias_matches(pattern: str, limit: int) -> list[dict]:
    r = _req.get(
        DANBOORU_ALIASES_URL,
        params={
            "search[antecedent_name_matches]": pattern,
            "search[status]": "active",
            "limit": limit,
        },
        auth=_danbooru_auth(),
        timeout=8,
        headers={"User-Agent": "HoloScope-Crawler/1.0"},
    )
    r.raise_for_status()
    return r.json()


def _tag_search_patterns(q: str) -> list[str]:
    patterns = [f"{q}*", f"*{q}*"]
    tokens = [part for part in q.split("_") if part]
    if len(tokens) > 1:
        patterns.append("*" + "*".join(tokens) + "*")

    unique = []
    for pattern in patterns:
        if pattern not in unique:
            unique.append(pattern)
    return unique


def _autocomplete_queries(raw_q: str, normalized_q: str) -> list[str]:
    queries = []
    for query in (raw_q.strip(), normalized_q):
        if query and query not in queries:
            queries.append(query)
    return queries


def _add_tag_result(results: dict[str, dict], name: str, post_count: int = 0, **meta):
    if not name:
        return
    existing = results.get(name, {})
    results[name] = {
        "name": name,
        "post_count": max(int(existing.get("post_count") or 0), int(post_count or 0)),
        **{k: v for k, v in {**existing, **meta}.items() if v is not None},
    }


def _autocomplete_to_tags(items: list[dict], results: dict[str, dict]):
    for item in items:
        if item.get("category") != 4:
            continue
        name = item.get("value") or item.get("label")
        tag = item.get("tag") or {}
        if tag.get("is_deprecated"):
            continue
        _add_tag_result(
            results,
            name,
            item.get("post_count") or tag.get("post_count") or 0,
            label=item.get("label"),
            antecedent=item.get("antecedent"),
            source="autocomplete",
        )


def _tag_items_to_tags(items: list[dict], results: dict[str, dict]):
    for tag in items:
        if tag.get("category") != 4 or tag.get("is_deprecated"):
            continue
        if int(tag.get("post_count") or 0) <= 0:
            continue
        _add_tag_result(
            results,
            tag.get("name", ""),
            tag.get("post_count") or 0,
            label=tag.get("name"),
            source="tags",
        )


def _alias_items_to_tags(items: list[dict], results: dict[str, dict]):
    for alias in items:
        if alias.get("status") not in (None, "active"):
            continue
        consequent = alias.get("consequent_name", "")
        _add_tag_result(
            results,
            consequent,
            0,
            label=consequent,
            antecedent=alias.get("antecedent_name"),
            source="alias",
        )


def _default_display_name_from_tag(tag: str) -> str:
    # 단일 캐릭터 추가와 동일하게 기본 표시명은 Danbooru 태그 원문을 사용한다.
    return tag


def _genre_patterns(q: str) -> list[str]:
    patterns = [
        f"*_({q})",
        f"*_({q})*",
        f"*{q}*",
    ]
    unique = []
    for pattern in patterns:
        if pattern not in unique:
            unique.append(pattern)
    return unique


@router.get("/status")
def get_status():
    return _job.status()


@router.get("/health")
def crawl_health(remote: bool = True):
    """크롤러 heartbeat와 Danbooru 연결/rate-limit 상태를 점검."""
    payload = _job.health_snapshot()
    if not remote:
        return payload

    started = time.monotonic()
    remote_status = {
        "checked": True,
        "ok": False,
        "status_code": None,
        "rate_limited": False,
        "retry_after": None,
        "latency_ms": None,
        "error": None,
    }
    try:
        r = _req.get(
            DANBOORU_POSTS_URL,
            params={"limit": 1},
            auth=_danbooru_auth(),
            timeout=6,
            headers={"User-Agent": "HoloScope-Crawler/1.0"},
        )
        remote_status.update({
            "ok": 200 <= r.status_code < 400,
            "status_code": r.status_code,
            "rate_limited": r.status_code == 429,
            "retry_after": r.headers.get("Retry-After"),
            "latency_ms": int((time.monotonic() - started) * 1000),
        })
    except Exception as e:
        remote_status.update({
            "latency_ms": int((time.monotonic() - started) * 1000),
            "error": str(e),
        })

    return {**payload, "remote": remote_status}


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
    """Danbooru 태그 자동완성. 캐릭터 태그를 post_count 내림차순으로 반환."""
    raw_q = q.strip()
    q = _normalize_tag_query(q)
    limit = max(1, min(limit, 25))
    if len(q) < 2:
        return {"tags": []}

    try:
        merged: dict[str, dict] = {}
        for query in _autocomplete_queries(raw_q, q):
            _autocomplete_to_tags(_fetch_autocomplete(query, limit), merged)

        # prefix 검색이 빠르고 정확하다. 부족하면 contains 검색으로 보강한다.
        for pattern in _tag_search_patterns(q):
            _tag_items_to_tags(_fetch_tag_matches(pattern, limit), merged)
            if len(merged) >= limit:
                break

        # 실명/별칭 검색 보강: e.g. sorasaki_hina -> hina_(blue_archive)
        if len(merged) < limit:
            for pattern in _tag_search_patterns(q):
                _alias_items_to_tags(_fetch_alias_matches(pattern, limit), merged)
                if len(merged) >= limit:
                    break

        tags = sorted(
            merged.values(),
            key=lambda item: int(item.get("post_count") or 0),
            reverse=True,
        )[:limit]
        return {
            "tags": [
                {
                    "name": t["name"],
                    "post_count": t.get("post_count", 0),
                    "label": t.get("label"),
                    "antecedent": t.get("antecedent"),
                    "source": t.get("source"),
                }
                for t in tags
            ]
        }
    except Exception as e:
        raise HTTPException(502, f"Danbooru tag search failed: {e}") from e


@router.get("/tags/genre")
def search_genre_characters(q: str = Query(default="", min_length=0), limit: int = 80):
    """작품/장르명으로 관련 캐릭터 태그 후보를 반환한다."""
    raw_q = q.strip()
    q = _normalize_tag_query(q)
    limit = max(1, min(limit, 200))
    if len(q) < 2:
        return {"query": raw_q, "normalized": q, "characters": []}

    try:
        merged: dict[str, dict] = {}
        # Danbooru 캐릭터 태그는 대체로 hina_(blue_archive) 형태다.
        for pattern in _genre_patterns(q):
            _tag_items_to_tags(_fetch_tag_matches(pattern, limit, category=4), merged)
            if len(merged) >= limit:
                break

        characters = [
            {
                "key": item["name"],
                "tag": item["name"],
                "display_name": _default_display_name_from_tag(item["name"]),
                "post_count": int(item.get("post_count") or 0),
                "source": item.get("source"),
            }
            for item in sorted(
                merged.values(),
                key=lambda value: int(value.get("post_count") or 0),
                reverse=True,
            )[:limit]
        ]
        return {"query": raw_q, "normalized": q, "characters": characters}
    except Exception as e:
        raise HTTPException(502, f"Danbooru genre search failed: {e}") from e


@router.get("/tags/validate/{tag:path}")
def validate_tag(tag: str):
    """태그 존재 여부 및 post_count 반환."""
    tag = _normalize_tag_query(tag)

    try:
        r = _req.get(
            DANBOORU_TAGS_URL,
            params={"search[name]": tag, "limit": 1},
            auth=_danbooru_auth(),
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
