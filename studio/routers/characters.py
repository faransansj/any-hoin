"""캐릭터 정의 CRUD 라우터 (characters.json ↔ API)."""

from pathlib import Path

from fastapi import APIRouter, HTTPException

import studio.characters as ch

DATASET_DIR = Path("./dataset/raw")
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp"}

router = APIRouter(prefix="/characters", tags=["characters"])


def _count_images(d: Path, recursive: bool = False) -> int:
    if not d.exists():
        return 0
    files = d.rglob("*") if recursive else d.iterdir()
    return sum(1 for f in files if f.is_file() and f.suffix.lower() in ALLOWED_EXT)


def _with_count(char: dict) -> dict:
    count = _count_images(DATASET_DIR / char["key"])
    other_count = _count_images(DATASET_DIR / "others" / char["key"], recursive=True)
    return {
        **char,
        "count": count,
        "other_count": other_count,
        "total_count": count + other_count,
    }


# ── 목록 ──────────────────────────────────────────────────

@router.get("")
def list_characters():
    chars = ch.load()
    return {"characters": [_with_count(c) for c in chars.values()]}


# ── 생성 ─────────────────────────────────────────────────

@router.post("")
def create_character(body: dict):
    key          = (body.get("key") or "").strip()
    tag          = (body.get("tag") or "").strip()
    display_name = (body.get("display_name") or key).strip()

    if not key:
        raise HTTPException(400, "key is required")
    if not tag:
        raise HTTPException(400, "tag is required")
    # key는 폴더명이므로 경로 구분자 금지
    if any(c in key for c in "/\\."):
        raise HTTPException(400, "key must not contain path separators")

    chars = ch.load()
    if key in chars:
        raise HTTPException(409, f"'{key}' already exists")

    chars[key] = {"key": key, "tag": tag, "display_name": display_name}
    ch.save(chars)
    return _with_count(chars[key])


# ── 수정 ─────────────────────────────────────────────────

@router.put("/{key}")
def update_character(key: str, body: dict):
    chars = ch.load()
    if key not in chars:
        raise HTTPException(404, f"'{key}' not found")

    for field in ("tag", "display_name"):
        if field in body and body[field] is not None:
            chars[key][field] = str(body[field]).strip()

    ch.save(chars)
    return _with_count(chars[key])


# ── 삭제 ─────────────────────────────────────────────────

@router.delete("/{key}")
def delete_character(key: str):
    chars = ch.load()
    if key not in chars:
        raise HTTPException(404, f"'{key}' not found")
    del chars[key]
    ch.save(chars)
    return {"deleted": key}


# ── 일괄 가져오기 ─────────────────────────────────────────

@router.post("/import")
def import_characters(body: dict):
    """
    body = {"characters": [{"key": ..., "tag": ..., "display_name": ...}]}
    기존 key는 덮어쓰기.
    """
    chars   = ch.load()
    imported = []
    skipped  = []

    for item in body.get("characters", []):
        key = (item.get("key") or "").strip()
        tag = (item.get("tag") or "").strip()
        if not key or not tag:
            skipped.append(item)
            continue
        chars[key] = {
            "key":          key,
            "tag":          tag,
            "display_name": (item.get("display_name") or key).strip(),
        }
        imported.append(key)

    ch.save(chars)
    return {"imported": len(imported), "skipped": len(skipped), "keys": imported}
