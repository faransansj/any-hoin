"""이미지 업로드 / 이동 / 삭제 / 썸네일 라우터."""

from copy import deepcopy
import hashlib
import os
import shutil
import threading
import time
import warnings
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, ImageOps

DATASET_DIR = Path("./dataset/raw")
THUMB_DIR   = Path("./.thumbnails")
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp"}
THUMB_SIZE  = (200, 200)
MB = 1024 * 1024
DEFAULT_LARGE_THRESHOLD_MB = 16
DEFAULT_MAX_SIDE = 2048
DEFAULT_QUALITY = 88

router = APIRouter(prefix="/images", tags=["images"])


def _empty_preprocess_status() -> dict:
    return {
        "name": "image_preprocess",
        "state": "idle",
        "label": None,
        "threshold_mb": DEFAULT_LARGE_THRESHOLD_MB,
        "max_side": DEFAULT_MAX_SIDE,
        "quality": DEFAULT_QUALITY,
        "dry_run": False,
        "total": 0,
        "current": 0,
        "pct": 0.0,
        "current_image": None,
        "processed": 0,
        "skipped": 0,
        "before_bytes": 0,
        "after_bytes": 0,
        "saved_bytes": 0,
        "started_at": None,
        "finished_at": None,
        "elapsed_sec": None,
        "error": None,
        "result": None,
    }


_PREPROCESS_LOCK = threading.Lock()
_PREPROCESS_STATUS = _empty_preprocess_status()
_PREPROCESS_THREAD: threading.Thread | None = None


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


# ── 전처리 ───────────────────────────────────────────────

@router.get("/preprocess/scan")
def scan_large_images(
    label: str | None = Query(None),
    threshold_mb: float = Query(DEFAULT_LARGE_THRESHOLD_MB, ge=1, le=512),
    preview_limit: int = Query(20, ge=1, le=100),
):
    threshold_bytes = int(threshold_mb * MB)
    files = _scope_image_files(label)
    large_files = [f for f in files if _safe_size(f) >= threshold_bytes]
    large_files.sort(key=_safe_size, reverse=True)

    return {
        "label": label,
        "threshold_mb": threshold_mb,
        "threshold_bytes": threshold_bytes,
        "total_count": len(files),
        "large_count": len(large_files),
        "large_bytes": sum(_safe_size(f) for f in large_files),
        "largest": [_image_info(f) for f in large_files[:preview_limit]],
    }


@router.post("/preprocess")
def preprocess_large_images(body: dict):
    label, threshold_mb, max_side, quality, dry_run = _preprocess_params(body)
    files = _preprocess_targets(label, threshold_mb)
    return _run_preprocess_files(
        files,
        label=label,
        threshold_mb=threshold_mb,
        max_side=max_side,
        quality=quality,
        dry_run=dry_run,
    )


@router.get("/preprocess/status")
def get_preprocess_status():
    return _preprocess_status_snapshot()


@router.post("/preprocess/start")
def start_preprocess_large_images(body: dict):
    global _PREPROCESS_STATUS, _PREPROCESS_THREAD

    label, threshold_mb, max_side, quality, dry_run = _preprocess_params(body)
    files = _preprocess_targets(label, threshold_mb)
    started_at = time.time()

    with _PREPROCESS_LOCK:
        if _PREPROCESS_STATUS.get("state") == "running":
            return deepcopy(_PREPROCESS_STATUS)

        _PREPROCESS_STATUS = {
            **_empty_preprocess_status(),
            "state": "running",
            "label": label,
            "threshold_mb": threshold_mb,
            "max_side": max_side,
            "quality": quality,
            "dry_run": dry_run,
            "total": len(files),
            "started_at": started_at,
        }

        _PREPROCESS_THREAD = threading.Thread(
            target=_run_preprocess_background,
            args=(files,),
            kwargs={
                "label": label,
                "threshold_mb": threshold_mb,
                "max_side": max_side,
                "quality": quality,
                "dry_run": dry_run,
                "started_at": started_at,
            },
            daemon=True,
        )
        _PREPROCESS_THREAD.start()
        return deepcopy(_PREPROCESS_STATUS)


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

def _preprocess_params(body: dict) -> tuple[str | None, float, int, int, bool]:
    label = body.get("label") or None
    threshold_mb = float(body.get("threshold_mb", DEFAULT_LARGE_THRESHOLD_MB))
    max_side = int(body.get("max_side", DEFAULT_MAX_SIDE))
    quality = int(body.get("quality", DEFAULT_QUALITY))
    dry_run = bool(body.get("dry_run", False))

    if threshold_mb < 1 or threshold_mb > 512:
        raise HTTPException(400, "threshold_mb must be between 1 and 512")
    if max_side < 512 or max_side > 8192:
        raise HTTPException(400, "max_side must be between 512 and 8192")
    if quality < 50 or quality > 100:
        raise HTTPException(400, "quality must be between 50 and 100")

    return label, threshold_mb, max_side, quality, dry_run


def _preprocess_targets(label: str | None, threshold_mb: float) -> list[Path]:
    threshold_bytes = int(threshold_mb * MB)
    files = [f for f in _scope_image_files(label) if _safe_size(f) >= threshold_bytes]
    files.sort(key=_safe_size, reverse=True)
    return files


def _run_preprocess_files(
    files: list[Path],
    *,
    label: str | None,
    threshold_mb: float,
    max_side: int,
    quality: int,
    dry_run: bool,
    progress_cb=None,
) -> dict:
    results = []
    before_total = 0
    after_total = 0
    processed = 0
    skipped = 0
    total = len(files)

    for index, path in enumerate(files, start=1):
        before = _safe_size(path)
        before_total += before
        if progress_cb:
            progress_cb(
                current=index - 1,
                total=total,
                current_image=_image_id_for_path(path),
                processed=processed,
                skipped=skipped,
                before_bytes=before_total,
                after_bytes=after_total,
                saved_bytes=max(before_total - after_total, 0),
            )

        if dry_run:
            info = _image_info(path)
            info.update({
                "processed": False,
                "before_bytes": before,
                "after_bytes": before,
                "saved_bytes": 0,
                "reason": "dry_run",
            })
            result = info
        else:
            result = _preprocess_image_file(path, max_side=max_side, quality=quality)

        results.append(result)
        after_total += int(result.get("after_bytes", before))
        if result.get("processed"):
            processed += 1
        else:
            skipped += 1

        if progress_cb:
            progress_cb(
                current=index,
                total=total,
                current_image=_image_id_for_path(path),
                processed=processed,
                skipped=skipped,
                before_bytes=before_total,
                after_bytes=after_total,
                saved_bytes=max(before_total - after_total, 0),
            )

    return {
        "label": label,
        "threshold_mb": threshold_mb,
        "max_side": max_side,
        "quality": quality,
        "scanned": total,
        "processed": processed,
        "skipped": skipped,
        "before_bytes": before_total,
        "after_bytes": after_total,
        "saved_bytes": max(before_total - after_total, 0),
        "items": results[:100],
    }


def _preprocess_status_snapshot() -> dict:
    with _PREPROCESS_LOCK:
        return deepcopy(_PREPROCESS_STATUS)


def _update_preprocess_status(started_at: float, **updates):
    now = time.time()
    with _PREPROCESS_LOCK:
        total = int(updates.get("total", _PREPROCESS_STATUS.get("total") or 0))
        current = int(updates.get("current", _PREPROCESS_STATUS.get("current") or 0))
        pct = 100.0 if total == 0 else round(min(100.0, max(0.0, current / total * 100)), 1)
        _PREPROCESS_STATUS.update({
            **updates,
            "pct": pct,
            "elapsed_sec": round(now - started_at, 1),
        })


def _finish_preprocess_status(
    state: str,
    *,
    started_at: float,
    result: dict | None = None,
    error: str | None = None,
):
    finished_at = time.time()
    with _PREPROCESS_LOCK:
        total = int(_PREPROCESS_STATUS.get("total") or 0)
        _PREPROCESS_STATUS.update({
            "state": state,
            "current": total,
            "pct": 100.0 if state == "done" else _PREPROCESS_STATUS.get("pct", 0.0),
            "current_image": None,
            "finished_at": finished_at,
            "elapsed_sec": round(finished_at - started_at, 1),
            "error": error,
            "result": result,
        })
        if result:
            _PREPROCESS_STATUS.update({
                "processed": result["processed"],
                "skipped": result["skipped"],
                "before_bytes": result["before_bytes"],
                "after_bytes": result["after_bytes"],
                "saved_bytes": result["saved_bytes"],
            })


def _run_preprocess_background(
    files: list[Path],
    *,
    label: str | None,
    threshold_mb: float,
    max_side: int,
    quality: int,
    dry_run: bool,
    started_at: float,
):
    try:
        result = _run_preprocess_files(
            files,
            label=label,
            threshold_mb=threshold_mb,
            max_side=max_side,
            quality=quality,
            dry_run=dry_run,
            progress_cb=lambda **updates: _update_preprocess_status(started_at, **updates),
        )
        _finish_preprocess_status("done", started_at=started_at, result=result)
    except Exception as exc:
        _finish_preprocess_status("failed", started_at=started_at, error=str(exc))


def _scope_image_files(label: str | None) -> list[Path]:
    if label:
        return _image_files(_resolve_label_dir(label))
    if not DATASET_DIR.exists():
        return []
    files: list[Path] = []
    for d in sorted(DATASET_DIR.iterdir()):
        if d.is_dir() and not d.name.startswith("."):
            files.extend(_image_files(d))
    return files


def _safe_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _image_id_for_path(path: Path) -> str:
    try:
        return path.relative_to(DATASET_DIR).as_posix()
    except ValueError:
        return path.name


def _image_info(path: Path) -> dict:
    width = height = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            with Image.open(path) as img:
                width, height = img.size
    except Exception:
        pass

    rel = _image_id_for_path(path)
    parts = Path(rel).parts
    label = parts[0] if parts else ""
    name = "/".join(parts[1:]) if len(parts) > 1 else path.name
    size = _safe_size(path)
    return {
        "id": rel,
        "label": label,
        "name": name,
        "bytes": size,
        "size_mb": round(size / MB, 2),
        "width": width,
        "height": height,
    }


def _preprocess_image_file(path: Path, *, max_side: int, quality: int) -> dict:
    before = _safe_size(path)
    info = _image_info(path)
    result = {
        **info,
        "processed": False,
        "before_bytes": before,
        "after_bytes": before,
        "saved_bytes": 0,
        "reason": "",
    }

    tmp = path.with_name(f".{path.stem}.preprocess{path.suffix}")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            with Image.open(path) as src:
                img = ImageOps.exif_transpose(src)
                original_format = (src.format or path.suffix.lstrip(".") or "JPEG").upper()

                if original_format == "JPG":
                    original_format = "JPEG"
                if original_format not in {"JPEG", "PNG", "WEBP"}:
                    original_format = "JPEG"

                if max(img.width, img.height) > max_side:
                    img.thumbnail((max_side, max_side), Image.LANCZOS)

                save_kwargs = {"optimize": True}
                if original_format in {"JPEG", "WEBP"}:
                    if img.mode not in {"RGB", "L"}:
                        img = img.convert("RGB")
                    save_kwargs["quality"] = quality
                    if original_format == "JPEG":
                        save_kwargs["progressive"] = True
                elif original_format == "PNG":
                    save_kwargs["compress_level"] = 9

                img.save(tmp, original_format, **save_kwargs)

        after = _safe_size(tmp)
        if after <= 0:
            tmp.unlink(missing_ok=True)
            result["reason"] = "empty output"
            return result

        if after >= before:
            tmp.unlink(missing_ok=True)
            result["reason"] = "not smaller"
            return result

        os.replace(tmp, path)
        _evict_thumb(_image_id_for_path(path))
        result.update(_image_info(path))
        result.update({
            "processed": True,
            "before_bytes": before,
            "after_bytes": after,
            "saved_bytes": before - after,
            "reason": "resized",
        })
        return result
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        result["reason"] = exc.__class__.__name__
        return result

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
