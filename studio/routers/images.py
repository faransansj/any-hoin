"""이미지 업로드 / 이동 / 삭제 / 썸네일 라우터."""

import hashlib
import shutil
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from PIL import Image

DATASET_DIR = Path("./dataset/raw")
THUMB_DIR   = Path("./.thumbnails")
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp"}
THUMB_SIZE  = (200, 200)

router = APIRouter(prefix="/images", tags=["images"])


def _image_files(d: Path) -> list[Path]:
    if not d.exists():
        return []
    return sorted(
        f for f in d.rglob("*")
        if f.is_file() and f.suffix.lower() in ALLOWED_EXT
    )


def _sort_images(files: list[Path], mode: str) -> list[Path]:
    if mode == "name_desc":
        return sorted(files, key=lambda p: p.as_posix().lower(), reverse=True)
    if mode == "newest":
        return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)
    if mode == "oldest":
        return sorted(files, key=lambda p: p.stat().st_mtime)
    return sorted(files, key=lambda p: p.as_posix().lower())


# ── 목록 조회 ──────────────────────────────────────────────

@router.get("")
def list_images(
    label: str = Query(...),
    page: int = Query(1, ge=1),
    per_page: int = Query(60, ge=1, le=200),
    sort: str = Query("name_asc", pattern="^(name_asc|name_desc|newest|oldest)$"),
):
    label_dir = _resolve_label_dir(label)
    files = _sort_images(_image_files(label_dir), sort)
    total = len(files)
    start = (page - 1) * per_page
    page_files = files[start : start + per_page]

    return {
        "total":    total,
        "page":     page,
        "per_page": per_page,
        "images": [
            {
                "id":        f"{label}/{f.relative_to(label_dir).as_posix()}",
                "name":      f.relative_to(label_dir).as_posix(),
                "label":     label,
                "url":       f"/api/images/file/{label}/{f.relative_to(label_dir).as_posix()}",
                "thumbnail": f"/api/images/thumb/{label}/{f.relative_to(label_dir).as_posix()}",
            }
            for f in page_files
        ],
    }


@router.get("/stats")
def get_stats():
    if not DATASET_DIR.exists():
        return {"total": 0, "labels": []}

    labels = []
    total = 0
    for d in sorted(DATASET_DIR.iterdir()):
        if d.is_dir():
            count = len(_image_files(d))
            labels.append({"label": d.name, "count": count})
            total += count

    return {"total": total, "labels": labels}


# ── 업로드 ────────────────────────────────────────────────

@router.post("/upload")
async def upload_images(
    label: str               = Form(...),
    files: list[UploadFile]  = File(...),
):
    label_dir = _resolve_label_dir(label)
    label_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for f in files:
        ext = Path(f.filename or "").suffix.lower()
        if ext not in ALLOWED_EXT:
            continue
        if not f.content_type or not f.content_type.startswith("image/"):
            continue
        content  = await f.read()
        md5      = hashlib.md5(content).hexdigest()
        dst_path = label_dir / f"{md5}{ext}"
        if not dst_path.exists():
            dst_path.write_bytes(content)
            saved.append(dst_path.name)

    return {"saved": len(saved), "files": saved}


# ── 이동 / 삭제 ───────────────────────────────────────────

@router.post("/move")
def move_images(body: dict):
    image_ids    = body.get("image_ids", [])
    target_label = body.get("target_label", "")
    if not target_label:
        raise HTTPException(400, "target_label required")

    target_dir = _resolve_label_dir(target_label)
    target_dir.mkdir(parents=True, exist_ok=True)

    moved = []
    for img_id in image_ids:
        src = _resolve_image_id(img_id)
        if src.exists() and src.is_file():
            dst = _unique_path(target_dir / src.name)
            shutil.move(str(src), str(dst))
            _evict_thumb(img_id)
            moved.append(img_id)

    return {"moved": len(moved)}


@router.delete("")
def delete_images(body: dict):
    image_ids = body.get("image_ids", [])
    deleted = []
    for img_id in image_ids:
        src = _resolve_image_id(img_id)
        if src.exists() and src.is_file():
            src.unlink()
            _evict_thumb(img_id)
            deleted.append(img_id)
    return {"deleted": len(deleted)}


# ── 파일 / 썸네일 제공 ──────────────────────────────────────

@router.get("/file/{label}/{filename:path}")
def get_file(label: str, filename: str):
    _guard(label, filename)
    path = _resolve_label_dir(label) / filename
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(str(path))


@router.get("/thumb/{label}/{filename:path}")
def get_thumb(label: str, filename: str):
    _guard(label, filename)
    src   = _resolve_label_dir(label) / filename
    thumb = THUMB_DIR / label / filename

    if not src.exists():
        raise HTTPException(404)

    if not thumb.exists():
        thumb.parent.mkdir(parents=True, exist_ok=True)
        try:
            img = Image.open(src).convert("RGB")
            img.thumbnail(THUMB_SIZE, Image.LANCZOS)
            img.save(str(thumb), "JPEG", quality=80)
        except Exception:
            raise HTTPException(400, "Corrupted or unreadable image")

    return FileResponse(str(thumb), media_type="image/jpeg")


# ── 헬퍼 ─────────────────────────────────────────────────

def _guard(label: str, filename: str):
    """Path traversal 방지."""
    _resolve_label_dir(label)
    _validate_relative_path(filename)


def _validate_relative_path(path: str):
    if not isinstance(path, str):
        raise HTTPException(400, "Invalid path")
    rel = Path(path)
    if rel.is_absolute() or path.startswith("/") or "\\" in path:
        raise HTTPException(400, "Invalid path")
    if any(part in {"", ".", ".."} for part in rel.parts):
        raise HTTPException(400, "Invalid path")


def _resolve_label_dir(label: str) -> Path:
    if not label:
        raise HTTPException(400, "Invalid label")
    _validate_relative_path(label)
    if len(Path(label).parts) != 1:
        raise HTTPException(400, "Invalid label")
    return DATASET_DIR / label


def _resolve_image_id(img_id: str) -> Path:
    if not img_id:
        raise HTTPException(400, "Invalid image id")
    _validate_relative_path(img_id)
    path = DATASET_DIR / img_id
    try:
        path.resolve().relative_to(DATASET_DIR.resolve())
    except ValueError:
        raise HTTPException(400, "Invalid image id")
    return path


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    idx = 1
    while True:
        candidate = path.with_name(f"{path.stem}_{idx}{path.suffix}")
        if not candidate.exists():
            return candidate
        idx += 1


def _evict_thumb(img_id: str):
    _validate_relative_path(img_id)
    thumb = THUMB_DIR / img_id
    if thumb.exists():
        thumb.unlink(missing_ok=True)
