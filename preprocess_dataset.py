"""
학습 최적화 이미지 캐시 생성 스크립트
dataset/raw/<class>/ → dataset/.cache/<class>/ (target_size px JPEG)

- 최장변 기준 리사이즈 (업스케일 없음)
- 소스보다 캐시가 최신이면 스킵
- PREPROCESS_EVENT_PREFIX 구조화 이벤트로 진행률 출력
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image, UnidentifiedImageError

PREPROCESS_EVENT_PREFIX = "__HOLOSCOPE_PREPROCESS_EVENT__ "
VALID_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def emit_event(type_: str, data: dict):
    payload = {"type": type_, "data": data}
    print(PREPROCESS_EVENT_PREFIX + json.dumps(payload, ensure_ascii=False), flush=True)


def _process_one(src: Path, dst: Path, target_size: int, quality: int) -> str:
    """'cached' | 'skipped' | 'error:<msg>' 반환."""
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        return "skipped"
    try:
        with Image.open(src) as img:
            img = img.convert("RGB")
            if max(img.width, img.height) > target_size:
                img.thumbnail((target_size, target_size), Image.LANCZOS)
            dst.parent.mkdir(parents=True, exist_ok=True)
            img.save(dst, "JPEG", quality=quality, optimize=True)
        return "cached"
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        return f"error:{exc.__class__.__name__}"


def _collect_tasks(data_dir: Path, cache_dir: Path) -> list[tuple[Path, Path]]:
    tasks: list[tuple[Path, Path]] = []
    for cls_dir in sorted(data_dir.iterdir()):
        if not cls_dir.is_dir() or cls_dir.name.startswith("."):
            continue
        for src in sorted(cls_dir.rglob("*")):
            if not src.is_file() or src.suffix.lower() not in VALID_EXT:
                continue
            rel = src.relative_to(data_dir)
            dst = cache_dir / rel.parent / (src.stem + ".jpg")
            tasks.append((src, dst))
    return tasks


def main():
    parser = argparse.ArgumentParser(description="학습 최적화 이미지 캐시 생성")
    parser.add_argument("--data-dir",    default="./dataset/raw")
    parser.add_argument("--cache-dir",   default="./dataset/.cache")
    parser.add_argument("--target-size", type=int, default=320,
                        help="최장변 픽셀 (기본: 320)")
    parser.add_argument("--quality",     type=int, default=95,
                        help="JPEG 품질 0-100 (기본: 95)")
    parser.add_argument("--workers",     type=int, default=None,
                        help="처리 스레드 수 (기본: CPU 수 기반 자동)")
    args = parser.parse_args()

    data_dir   = Path(args.data_dir)
    cache_dir  = Path(args.cache_dir)
    target_size = args.target_size
    quality    = max(1, min(100, args.quality))
    workers    = args.workers or min(8, os.cpu_count() or 4)

    if not data_dir.exists():
        print(f"[오류] 데이터 디렉토리가 없습니다: {data_dir}", file=sys.stderr)
        sys.exit(1)

    print(
        f"[전처리] data_dir={data_dir}  cache_dir={cache_dir}"
        f"  target_size={target_size}  quality={quality}  workers={workers}"
    )

    tasks = _collect_tasks(data_dir, cache_dir)
    total = len(tasks)

    if total == 0:
        print("[전처리] 처리할 이미지가 없습니다.")
        emit_event("done", {"total": 0, "cached": 0, "skipped": 0, "errors": 0,
                            "elapsed_sec": 0.0})
        return

    print(f"[전처리] {total}장 처리 예정")
    emit_event("start", {"total": total})

    cached = skipped = errors = 0
    started_at = time.monotonic()
    last_emit = 0.0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_src = {
            pool.submit(_process_one, src, dst, target_size, quality): src
            for src, dst in tasks
        }
        done_count = 0
        for future in as_completed(future_to_src):
            result = future.result()
            done_count += 1
            if result == "cached":
                cached += 1
            elif result == "skipped":
                skipped += 1
            else:
                errors += 1
                src = future_to_src[future]
                print(f"[전처리] 오류: {src} ({result})")

            now = time.monotonic()
            if done_count == total or now - last_emit >= 0.5:
                emit_event("progress", {
                    "done":        done_count,
                    "total":       total,
                    "cached":      cached,
                    "skipped":     skipped,
                    "errors":      errors,
                    "elapsed_sec": round(now - started_at, 1),
                })
                last_emit = now

    elapsed = round(time.monotonic() - started_at, 1)
    emit_event("done", {
        "total": total, "cached": cached, "skipped": skipped,
        "errors": errors, "elapsed_sec": elapsed,
    })
    print(f"[전처리] 완료 — 생성 {cached}장 · 스킵 {skipped}장 · 오류 {errors}장 ({elapsed}s)")


if __name__ == "__main__":
    main()
