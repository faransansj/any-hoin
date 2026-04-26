"""라벨(폴더) CRUD 라우터."""

import shutil
from pathlib import Path

from fastapi import APIRouter, HTTPException

DATASET_DIR = Path("./dataset/raw")
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp"}

router = APIRouter(prefix="/labels", tags=["labels"])


def _count(d: Path) -> int:
    if not d.exists():
        return 0
    return sum(
        1
        for f in d.rglob("*")
        if f.is_file() and f.suffix.lower() in ALLOWED_EXT
    )


@router.get("")
def list_labels():
    if not DATASET_DIR.exists():
        return {"labels": []}

    labels = [
        {"name": d.name, "count": _count(d)}
        for d in sorted(DATASET_DIR.iterdir())
        if d.is_dir()
    ]

    # 불균형 경고: 최대 클래스의 20% 미만이면 warning=True
    if labels:
        max_count = max((l["count"] for l in labels), default=0)
        for l in labels:
            l["warning"] = bool(max_count > 0 and l["count"] < max_count * 0.2)

    return {"labels": labels}


def _guard_name(name: str):
    if any(c in name for c in "/\\.") or name.startswith("."):
        raise HTTPException(400, "Invalid label name")


@router.post("")
def create_label(body: dict):
    name = (body.get("name") or "").strip()
    if not name:
        raise HTTPException(400, "name is required")
    _guard_name(name)
    path = DATASET_DIR / name
    if path.exists():
        raise HTTPException(409, f"'{name}' already exists")
    path.mkdir(parents=True, exist_ok=True)
    return {"name": name, "count": 0, "warning": False}


@router.patch("/{name}")
def rename_label(name: str, body: dict):
    new_name = (body.get("name") or "").strip()
    if not new_name:
        raise HTTPException(400, "new name is required")
    _guard_name(name)
    _guard_name(new_name)
    src = DATASET_DIR / name
    dst = DATASET_DIR / new_name
    if not src.exists():
        raise HTTPException(404, f"'{name}' not found")
    if dst.exists():
        raise HTTPException(409, f"'{new_name}' already exists")
    src.rename(dst)
    return {"name": new_name, "count": _count(dst)}


@router.delete("/{name}")
def delete_label(name: str):
    _guard_name(name)
    path = DATASET_DIR / name
    if not path.exists():
        raise HTTPException(404, f"'{name}' not found")
    shutil.rmtree(path)
    return {"deleted": name}
