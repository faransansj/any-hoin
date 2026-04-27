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


def _guard_key(key: str) -> str:
    key = key.strip()
    if not key:
        raise HTTPException(400, "key is required")
    if any(c in key for c in "/\\."):
        raise HTTPException(400, "key must not contain path separators")
    return key


def _dataset_candidates(include_others: bool = False) -> list[dict]:
    if not DATASET_DIR.exists():
        return []

    candidates = []
    for d in sorted(DATASET_DIR.iterdir()):
        if not d.is_dir() or d.name.startswith("."):
            continue
        if d.name == "others" and not include_others:
            continue
        count = _count_images(d, recursive=True)
        if count <= 0:
            continue
        candidates.append({
            "key": d.name,
            "tag": d.name,
            "display_name": d.name,
            "count": count,
        })
    return candidates


# ── 목록 ──────────────────────────────────────────────────

@router.get("")
def list_characters():
    chars = ch.load()
    return {"characters": [_with_count(c) for c in chars.values()]}


# ── 생성 ─────────────────────────────────────────────────

@router.post("")
def create_character(body: dict):
    key          = _guard_key(body.get("key") or "")
    tag          = (body.get("tag") or "").strip()
    display_name = (body.get("display_name") or key).strip()

    if not tag:
        raise HTTPException(400, "tag is required")

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
    overwrite=false이면 기존 key는 건너뜀.
    """
    chars   = ch.load()
    imported = []
    skipped  = []
    overwrite = bool(body.get("overwrite", True))

    for item in body.get("characters", []):
        try:
            key = _guard_key(item.get("key") or "")
        except HTTPException:
            skipped.append(item)
            continue
        tag = (item.get("tag") or "").strip()
        if not key or not tag:
            skipped.append(item)
            continue
        if key in chars and not overwrite:
            skipped.append(key)
            continue
        chars[key] = {
            "key":          key,
            "tag":          tag,
            "display_name": (item.get("display_name") or key).strip(),
        }
        imported.append(key)

    ch.save(chars)
    return {"imported": len(imported), "skipped": len(skipped), "keys": imported}


@router.get("/discover")
def discover_dataset_characters(include_others: bool = False):
    """dataset/raw에 이미지가 있지만 characters.json에 없는 폴더를 감지."""
    chars = ch.load()
    dataset_items = _dataset_candidates(include_others=include_others)
    missing = [item for item in dataset_items if item["key"] not in chars]
    return {
        "registered": len(chars),
        "dataset_labels": len(dataset_items),
        "missing": missing,
    }


@router.post("/recover")
def recover_dataset_characters(body: dict):
    """
    dataset/raw/<key> 폴더를 characters.json에 다시 등록한다.
    기본 tag/display_name은 폴더명으로 복구한다.
    """
    include_others = bool(body.get("include_others", False))
    requested = body.get("keys")
    requested_keys = {str(k).strip() for k in requested} if isinstance(requested, list) else None

    chars = ch.load()
    imported = []
    skipped = []

    for item in _dataset_candidates(include_others=include_others):
        key = _guard_key(item["key"])
        if requested_keys is not None and key not in requested_keys:
            continue
        if key in chars:
            skipped.append(key)
            continue
        chars[key] = {
            "key": key,
            "tag": item["tag"],
            "display_name": item["display_name"],
        }
        imported.append(key)

    ch.save(chars)
    return {"imported": len(imported), "skipped": len(skipped), "keys": imported}
