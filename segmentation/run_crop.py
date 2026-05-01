"""
얼굴 크롭 전처리 CLI — Studio SegJob의 subprocess 대상.

사용:
  uv run python segmentation/run_crop.py \\
    --input-dir ./dataset/raw \\
    --output-dir ./dataset/raw_seg \\
    [--backend cascade|yolo] \\
    [--output-size 224] \\
    [--padding 0.3] \\
    [--min-face 48] \\
    [--sam-model vit_b] \\
    [--events]
"""

import argparse
import json
import sys
import time
from pathlib import Path

# subprocess로 직접 실행될 때 프로젝트 루트를 sys.path에 추가
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

SEG_EVENT_PREFIX = "__HOLOSCOPE_SEG_EVENT__ "


def emit(event: str, payload: dict, *, events: bool) -> None:
    if not events:
        return
    print(
        SEG_EVENT_PREFIX + json.dumps({"event": event, **payload}, ensure_ascii=False),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Anime face crop preprocessor")
    parser.add_argument("--input-dir",  required=True, metavar="DIR")
    parser.add_argument("--output-dir", required=True, metavar="DIR")
    parser.add_argument("--backend",    default="cascade", choices=["cascade", "yolo"])
    parser.add_argument("--output-size", type=int, default=224, metavar="PX")
    parser.add_argument("--padding",    type=float, default=0.3)
    parser.add_argument("--min-face",   type=int, default=48)
    parser.add_argument("--scale",      type=float, default=1.1,
                        help="cascade scaleFactor")
    parser.add_argument("--neighbors",  type=int, default=5,
                        help="cascade minNeighbors")
    parser.add_argument("--sam-model",  default="", metavar="TYPE",
                        help="SAM 모델 타입 (vit_b/vit_l/vit_h). 미지정 시 SAM 비활성.")
    parser.add_argument("--sam-checkpoint", default="", metavar="PATH")
    parser.add_argument("--sam-device",     default="cpu")
    parser.add_argument("--events", action="store_true",
                        help="Studio UI 용 구조화 이벤트를 출력합니다.")
    args = parser.parse_args()

    events = args.events
    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"[ERROR] 입력 디렉토리를 찾을 수 없습니다: {input_dir}", file=sys.stderr)
        sys.exit(1)

    # 얼굴 감지기
    from segmentation.face_detector import AnimeFaceDetector
    detector = AnimeFaceDetector(
        backend=args.backend,
        padding_ratio=args.padding,
        min_face_size=args.min_face,
        cascade_scale_factor=args.scale,
        cascade_min_neighbors=args.neighbors,
    )

    # SAM (선택)
    sam = None
    if args.sam_model:
        from segmentation.sam_segmenter import SAMSegmenter
        ckpt = Path(args.sam_checkpoint) if args.sam_checkpoint else None
        sam = SAMSegmenter(
            model_type=args.sam_model,
            checkpoint_path=ckpt,
            device=args.sam_device,
        )
        if not sam.is_available():
            print(
                f"[경고] SAM 체크포인트를 찾을 수 없습니다. SAM 없이 진행합니다.\n"
                f"  다운로드: {sam.checkpoint_url}"
            )
            sam = None

    # 처리기
    from segmentation.crop_processor import CropProcessor
    processor = CropProcessor(
        detector=detector,
        out_root=output_dir,
        output_size=args.output_size,
        skip_existing=True,
        sam_segmenter=sam,
    )

    # 클래스 디렉토리 목록 수집
    class_dirs = sorted(
        d for d in input_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )
    if not class_dirs:
        print(f"[ERROR] 클래스 디렉토리를 찾을 수 없습니다: {input_dir}", file=sys.stderr)
        sys.exit(1)

    valid_ext = {".jpg", ".jpeg", ".png", ".webp"}
    total_images = sum(
        len([f for f in d.iterdir() if f.is_file() and f.suffix.lower() in valid_ext])
        for d in class_dirs
    )

    print(f"얼굴 크롭 전처리 시작")
    print(f"  입력:  {input_dir}")
    print(f"  출력:  {output_dir}")
    print(f"  백엔드: {args.backend}")
    print(f"  클래스: {len(class_dirs)}개  |  이미지: {total_images}장")
    print(f"  SAM: {'활성' if sam else '비활성'}")
    print()

    emit("start", {
        "total_classes": len(class_dirs),
        "total_images": total_images,
        "backend": args.backend,
        "sam": bool(sam),
        "updated_at": time.time(),
    }, events=events)

    started_at = time.monotonic()
    global_processed = 0

    for cls_idx, cls_dir in enumerate(class_dirs, 1):
        cls_name = cls_dir.name
        cls_files = [
            f for f in cls_dir.iterdir()
            if f.is_file() and f.suffix.lower() in valid_ext
        ]
        cls_total = len(cls_files)

        print(f"  [{cls_idx}/{len(class_dirs)}] {cls_name}  ({cls_total}장)")
        emit("class_start", {
            "class": cls_name,
            "class_idx": cls_idx,
            "total_classes": len(class_dirs),
            "class_total": cls_total,
            "updated_at": time.time(),
        }, events=events)

        last_emit = 0.0

        def progress_cb(processed: int, total: int) -> None:
            nonlocal global_processed, last_emit
            global_processed += 1
            now = time.monotonic()
            if now - last_emit < 0.5 and processed < total:
                return
            last_emit = now
            elapsed = max(now - started_at, 1e-6)
            speed = global_processed / elapsed
            remaining = max(total_images - global_processed, 0)
            emit("progress", {
                "class": cls_name,
                "class_idx": cls_idx,
                "total_classes": len(class_dirs),
                "class_processed": processed,
                "class_total": total,
                "global_processed": global_processed,
                "total_images": total_images,
                "pct": round(global_processed / total_images * 100, 1) if total_images else 0,
                "eta_sec": round(remaining / speed) if speed > 0 else -1,
                "updated_at": time.time(),
            }, events=events)

        stats = processor.process_class(cls_dir, progress_cb=progress_cb)

        print(
            f"    single: {stats.single_face}  multi: {stats.multi_face}  "
            f"no-face: {stats.no_face}  error: {stats.errors}  "
            f"→ 크롭 {stats.saved_crops}장"
        )
        emit("class_done", {
            "class": cls_name,
            "class_idx": cls_idx,
            "total_classes": len(class_dirs),
            "single_face": stats.single_face,
            "multi_face": stats.multi_face,
            "no_face": stats.no_face,
            "errors": stats.errors,
            "saved_crops": stats.saved_crops,
            "updated_at": time.time(),
        }, events=events)

    elapsed = time.monotonic() - started_at
    print()
    print(f"전처리 완료 — {elapsed:.1f}초")
    print(f"결과 저장 위치: {output_dir}")

    emit("complete", {
        "total_images": total_images,
        "output_dir": str(output_dir),
        "elapsed_sec": round(elapsed, 1),
        "updated_at": time.time(),
    }, events=events)


if __name__ == "__main__":
    main()
