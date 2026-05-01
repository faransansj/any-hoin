"""
Optional SAM (Segment Anything Model) backend for refined face segmentation.

Face bounding box → SAM mask → precise crop with background removed.

SAM 모델 다운로드:
  vit_b (~375MB):  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
  vit_l (~1.25GB): https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
  vit_h (~2.5GB):  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

SAM 설치:
  pip install git+https://github.com/facebookresearch/segment-anything.git
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np

from .face_detector import FaceBox

SAM_CHECKPOINT_URLS: dict[str, str] = {
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
}

_MODELS_DIR = Path(__file__).parent / "models"


class SAMSegmenter:
    """
    SAM 기반 얼굴 세그멘테이션.
    FaceBox를 프롬프트로 사용해 정밀한 얼굴 마스크를 얻는다.
    배경은 흰색(또는 지정 색상)으로 채운다.
    """

    def __init__(
        self,
        model_type: Literal["vit_b", "vit_l", "vit_h"] = "vit_b",
        checkpoint_path: Path | None = None,
        device: str = "cpu",
    ):
        self.model_type = model_type
        self.device = device
        self.checkpoint_path = (
            Path(checkpoint_path) if checkpoint_path
            else _MODELS_DIR / f"sam_{model_type}.pth"
        )
        self._predictor = None

    @property
    def checkpoint_url(self) -> str:
        return SAM_CHECKPOINT_URLS.get(self.model_type, "")

    def _load(self) -> None:
        try:
            from segment_anything import SamPredictor, sam_model_registry  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "segment-anything 이 필요합니다.\n"
                "설치: pip install git+https://github.com/facebookresearch/segment-anything.git"
            ) from exc

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"SAM 체크포인트를 찾을 수 없습니다: {self.checkpoint_path}\n"
                f"다운로드: {self.checkpoint_url}"
            )

        sam = sam_model_registry[self.model_type](checkpoint=str(self.checkpoint_path))
        sam.to(device=self.device)
        self._predictor = SamPredictor(sam)

    def is_available(self) -> bool:
        try:
            import segment_anything  # type: ignore  # noqa: F401
            return self.checkpoint_path.exists()
        except ImportError:
            return False

    def refine_crop(
        self,
        image: np.ndarray,
        box: FaceBox,
        padding_ratio: float = 0.3,
        fill_bg: tuple[int, int, int] = (255, 255, 255),
    ) -> np.ndarray:
        """
        SAM으로 얼굴 영역을 세그멘테이션하고, 배경을 fill_bg 색상으로 채운 크롭을 반환.
        SAM 미설치 / 체크포인트 없음 / 추론 실패 등 모든 경우에 단순 직사각형 크롭으로 폴백.
        """
        h, w = image.shape[:2]
        px = int(box.w * padding_ratio)
        py = int(box.h * padding_ratio)
        x1 = max(0, box.x - px)
        y1 = max(0, box.y - py)
        x2 = min(w, box.x + box.w + px)
        y2 = min(h, box.y + box.h + py)
        fallback = image[y1:y2, x1:x2].copy()

        try:
            if self._predictor is None:
                self._load()

            self._predictor.set_image(image)
            sam_box = np.array([box.x, box.y, box.x + box.w, box.y + box.h])
            masks, _, _ = self._predictor.predict(
                box=sam_box[None],
                multimask_output=False,
            )
            mask = masks[0]  # (H, W) bool

            crop = image[y1:y2, x1:x2].copy()
            mask_crop = mask[y1:y2, x1:x2]
            bg = np.array(fill_bg, dtype=np.uint8)
            crop[~mask_crop] = bg
            return crop
        except Exception:
            return fallback
