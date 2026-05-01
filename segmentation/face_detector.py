"""
Anime face detector — lbpcascade_animeface (OpenCV) or YOLO (ultralytics).

기본 백엔드: cascade (OpenCV LBP, CPU, 추가 모델 없음)
선택 백엔드: yolo  (ultralytics, 더 높은 정확도)
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

LBPCASCADE_URL = (
    "https://raw.githubusercontent.com/nagadomi/lbpcascade_animeface"
    "/master/lbpcascade_animeface.xml"
)
LBPCASCADE_FILENAME = "lbpcascade_animeface.xml"
_MODELS_DIR = Path(__file__).parent / "models"


@dataclass
class FaceBox:
    x: int
    y: int
    w: int
    h: int

    @property
    def area(self) -> int:
        return self.w * self.h

    def padded(self, ratio: float, img_w: int, img_h: int) -> "FaceBox":
        px = int(self.w * ratio)
        py = int(self.h * ratio)
        x1 = max(0, self.x - px)
        y1 = max(0, self.y - py)
        x2 = min(img_w, self.x + self.w + px)
        y2 = min(img_h, self.y + self.h + py)
        return FaceBox(x1, y1, x2 - x1, y2 - y1)

    def as_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h}


class AnimeFaceDetector:
    """애니메이션 캐릭터 얼굴 감지기."""

    def __init__(
        self,
        backend: Literal["cascade", "yolo"] = "cascade",
        model_dir: Path | None = None,
        padding_ratio: float = 0.3,
        min_face_size: int = 48,
        cascade_scale_factor: float = 1.1,
        cascade_min_neighbors: int = 5,
    ):
        self.backend = backend
        self.model_dir = Path(model_dir) if model_dir else _MODELS_DIR
        self.padding_ratio = padding_ratio
        self.min_face_size = min_face_size
        self.cascade_scale_factor = cascade_scale_factor
        self.cascade_min_neighbors = cascade_min_neighbors
        self._cascade = None
        self._yolo = None

    def _ensure_cascade(self) -> Path:
        self.model_dir.mkdir(parents=True, exist_ok=True)
        xml_path = self.model_dir / LBPCASCADE_FILENAME
        if not xml_path.exists():
            print(f"[FaceDetector] lbpcascade_animeface.xml 다운로드 → {xml_path}")
            urllib.request.urlretrieve(LBPCASCADE_URL, xml_path)
            print("[FaceDetector] 다운로드 완료")
        return xml_path

    def _get_cascade(self):
        if self._cascade is None:
            try:
                import cv2  # type: ignore
            except ImportError as exc:
                raise ImportError(
                    "opencv-python-headless 가 필요합니다.\n"
                    "설치: uv sync --extra seg"
                ) from exc
            xml_path = self._ensure_cascade()
            self._cascade = cv2.CascadeClassifier(str(xml_path))
        return self._cascade

    def _detect_cascade(self, image: np.ndarray) -> list[FaceBox]:
        import cv2  # type: ignore
        cascade = self._get_cascade()
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray = cv2.equalizeHist(gray)
        detections = cascade.detectMultiScale(
            gray,
            scaleFactor=self.cascade_scale_factor,
            minNeighbors=self.cascade_min_neighbors,
            minSize=(self.min_face_size, self.min_face_size),
        )
        if len(detections) == 0:
            return []
        return [FaceBox(int(x), int(y), int(w), int(h)) for x, y, w, h in detections]

    def _detect_yolo(self, image: np.ndarray) -> list[FaceBox]:
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError:
            return []
        if self._yolo is None:
            self._yolo = YOLO("yolov8n-face.pt")
        results = self._yolo(image, verbose=False)
        boxes = []
        for r in results:
            for box in r.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = box[:4]
                boxes.append(FaceBox(int(x1), int(y1), int(x2 - x1), int(y2 - y1)))
        return boxes

    def detect(self, image: np.ndarray) -> list[FaceBox]:
        """RGB uint8 numpy 이미지에서 얼굴 박스 목록을 반환."""
        if self.backend == "yolo":
            return self._detect_yolo(image)
        return self._detect_cascade(image)

    def crop_face(self, image: np.ndarray, box: FaceBox) -> np.ndarray:
        h, w = image.shape[:2]
        pb = box.padded(self.padding_ratio, w, h)
        return image[pb.y : pb.y + pb.h, pb.x : pb.x + pb.w]

    def crop_all(self, image: np.ndarray) -> list[tuple[FaceBox, np.ndarray]]:
        """(box, crop) 리스트 반환."""
        return [(b, self.crop_face(image, b)) for b in self.detect(image)]
