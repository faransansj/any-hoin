"""
CropProcessor — 데이터셋 디렉토리를 순회하며 애니메이션 얼굴을 감지·크롭한다.

처리 결과는 out_root/<class>/ 에 저장된다.
얼굴이 없는 이미지는 원본 그대로 복사한다.
얼굴이 여럿인 이미지는 _f0, _f1, ... 접미사로 복수의 크롭을 저장한다.

SAM 세그멘터가 활성화된 경우 직사각형 크롭 대신 정밀 마스크 크롭을 저장한다.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from .face_detector import AnimeFaceDetector, FaceBox

VALID_EXT = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass
class ClassStats:
    total: int = 0
    single_face: int = 0
    multi_face: int = 0
    no_face: int = 0
    errors: int = 0
    saved_crops: int = 0


@dataclass
class DatasetStats:
    by_class: dict[str, ClassStats] = field(default_factory=dict)

    @property
    def total_images(self) -> int:
        return sum(s.total for s in self.by_class.values())

    @property
    def total_crops(self) -> int:
        return sum(s.saved_crops for s in self.by_class.values())

    @property
    def multi_face_images(self) -> int:
        return sum(s.multi_face for s in self.by_class.values())


class CropProcessor:
    """데이터셋 디렉토리를 얼굴 크롭 버전으로 변환."""

    def __init__(
        self,
        detector: AnimeFaceDetector,
        out_root: Path,
        output_size: int = 224,
        skip_existing: bool = True,
        sam_segmenter=None,
    ):
        self.detector = detector
        self.out_root = Path(out_root)
        self.output_size = output_size
        self.skip_existing = skip_existing
        self.sam = sam_segmenter

    def _save_crop(self, crop: np.ndarray, dst: Path) -> None:
        crop_pil = Image.fromarray(crop)
        if min(crop_pil.width, crop_pil.height) > 0:
            crop_pil = crop_pil.resize(
                (self.output_size, self.output_size), Image.LANCZOS
            )
        ext = dst.suffix.lower()
        if ext in {".jpg", ".jpeg"}:
            crop_pil.save(dst, "JPEG", quality=92, optimize=True)
        elif ext == ".webp":
            crop_pil.save(dst, "WEBP", quality=92)
        else:
            crop_pil.save(dst, "PNG", optimize=True)

    def process_image(self, src: Path, out_dir: Path) -> tuple[list[Path], int]:
        """
        이미지 하나를 처리해 크롭을 out_dir에 저장한다.
        반환: (저장된 경로 목록, 감지된 얼굴 수)
        """
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            with Image.open(src) as pil:
                image = np.array(pil.convert("RGB"))
        except Exception:
            return [], -1

        boxes = self.detector.detect(image)

        if not boxes:
            dst = out_dir / src.name
            if not self.skip_existing or not dst.exists():
                shutil.copy2(src, dst)
            return [dst], 0

        saved: list[Path] = []
        stem = src.stem
        suffix = src.suffix.lower() if src.suffix.lower() in VALID_EXT else ".jpg"

        for idx, box in enumerate(boxes):
            name = f"{stem}{suffix}" if len(boxes) == 1 else f"{stem}_f{idx}{suffix}"
            dst = out_dir / name

            if self.skip_existing and dst.exists():
                saved.append(dst)
                continue

            if self.sam is not None:
                crop = self.sam.refine_crop(image, box, self.detector.padding_ratio)
            else:
                crop = self.detector.crop_face(image, box)

            if crop.size == 0 or min(crop.shape[:2]) == 0:
                continue

            self._save_crop(crop, dst)
            saved.append(dst)

        return saved, len(boxes)

    def process_class(
        self,
        class_dir: Path,
        *,
        progress_cb=None,
    ) -> ClassStats:
        """
        클래스 디렉토리를 처리한다.
        progress_cb(processed, total) — 진행 콜백 (선택)
        """
        out_dir = self.out_root / class_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)

        files = sorted(
            f for f in class_dir.iterdir()
            if f.is_file() and f.suffix.lower() in VALID_EXT
        )
        stats = ClassStats(total=len(files))

        for i, img_path in enumerate(files, 1):
            try:
                saved, n_faces = self.process_image(img_path, out_dir)
                if n_faces < 0:
                    stats.errors += 1
                elif n_faces == 0:
                    stats.no_face += 1
                    stats.saved_crops += len(saved)
                elif n_faces == 1:
                    stats.single_face += 1
                    stats.saved_crops += len(saved)
                else:
                    stats.multi_face += 1
                    stats.saved_crops += len(saved)
            except Exception:
                stats.errors += 1

            if progress_cb:
                progress_cb(i, stats.total)

        return stats

    def process_dataset(
        self,
        dataset_dir: Path,
        *,
        progress_cb=None,
    ) -> DatasetStats:
        """
        데이터셋 디렉토리 전체를 처리한다.
        progress_cb(class_name, processed_in_class, total_in_class)
        """
        dataset_dir = Path(dataset_dir)
        self.out_root.mkdir(parents=True, exist_ok=True)

        class_dirs = sorted(
            d for d in dataset_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )

        result = DatasetStats()
        for cls_dir in class_dirs:
            def _cb(p, t, name=cls_dir.name):
                if progress_cb:
                    progress_cb(name, p, t)

            stats = self.process_class(cls_dir, progress_cb=_cb)
            result.by_class[cls_dir.name] = stats

        return result
