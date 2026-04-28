"""
Dataset & Augmentation Pipeline for Hololive Classifier
- 애니 이미지 특성 고려: 색조 변환 약하게
- Swin Transformer 입력 규격: 224x224
"""

import os
import json
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from PIL import Image, UnidentifiedImageError

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

# Danbooru 이미지 중 일부는 EXIF/TIFF 메타데이터가 깨져 있다. 이미지 본문은
# 정상 학습 가능하므로 PIL의 반복 경고만 숨겨 웹 로그가 오염되지 않게 한다.
warnings.filterwarnings("ignore", message=r".*Corrupt EXIF data.*", category=UserWarning)
warnings.filterwarnings("ignore", message=r".*Possibly corrupt EXIF data.*", category=UserWarning)
warnings.filterwarnings("ignore", message=r".*Truncated File Read.*", category=UserWarning)
warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)

VALID_EXT = {".jpg", ".jpeg", ".png", ".webp"}
MAX_IMAGE_PIXELS = Image.MAX_IMAGE_PIXELS
IMAGE_LOAD_ERRORS = (
    UnidentifiedImageError,
    OSError,
    ValueError,
    Image.DecompressionBombError,
)


@lru_cache(maxsize=250_000)
def _validate_image_cache(path: str, size: int, mtime_ns: int) -> tuple[bool, str]:
    try:
        with Image.open(path) as img:
            pixel_count = img.width * img.height
            if MAX_IMAGE_PIXELS is not None and pixel_count > MAX_IMAGE_PIXELS:
                return False, "too large"
            img.verify()
        return True, ""
    except IMAGE_LOAD_ERRORS as exc:
        return False, exc.__class__.__name__


def validate_image_file(path: str | Path, *, deep: bool = False) -> tuple[bool, str]:
    """Return whether an image file is readable enough for training."""
    path = Path(path)
    try:
        stat = path.stat()
    except OSError as exc:
        return False, exc.__class__.__name__

    if stat.st_size <= 0:
        return False, "empty file"

    if not deep:
        return True, ""

    return _validate_image_cache(str(path), stat.st_size, stat.st_mtime_ns)


def load_rgb_image_array(path: str | Path) -> np.ndarray:
    """Load an image as an RGB numpy array."""
    with Image.open(path) as image:
        pixel_count = image.width * image.height
        if MAX_IMAGE_PIXELS is not None and pixel_count > MAX_IMAGE_PIXELS:
            raise ValueError("image is too large")
        return np.array(image.convert("RGB"))


def _collect_samples_by_class(
    root_dir: str | Path,
    *,
    validate_images: bool = True,
    deep_validate_images: bool = False,
    validation_workers: int | None = None,
) -> tuple[dict[str, list[str]], list[tuple[str, str]]]:
    root_dir = Path(root_dir)
    top_dirs = sorted(
        d for d in root_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )
    top_level_classes = {d.name for d in top_dirs}

    # 전체 샘플 후보 수집. others/<class>는 해당 class가 top-level에 존재하면
    # 중복 라벨 오염을 피하기 위해 others 샘플에서 제외한다.
    candidate_paths_by_class: dict[str, list[str]] = {}
    for cls_dir in top_dirs:
        cls_candidates = []
        for f in sorted(cls_dir.rglob("*")):
            if not f.is_file() or f.suffix.lower() not in VALID_EXT:
                continue
            if cls_dir.name == "others":
                rel_parts = f.relative_to(cls_dir).parts
                if len(rel_parts) > 1 and rel_parts[0] in top_level_classes:
                    continue
            cls_candidates.append(str(f))

        if cls_candidates:
            candidate_paths_by_class[cls_dir.name] = cls_candidates

    if not validate_images:
        return candidate_paths_by_class, []

    all_paths = [
        path
        for paths in candidate_paths_by_class.values()
        for path in paths
    ]
    validation_results = _validate_image_paths(
        all_paths,
        validation_workers,
        deep=deep_validate_images,
    )

    sample_paths_by_class: dict[str, list[str]] = {}
    skipped_files: list[tuple[str, str]] = []
    for cls, paths in candidate_paths_by_class.items():
        cls_samples = []
        for path in paths:
            valid, reason = validation_results[path]
            if not valid:
                skipped_files.append((path, reason))
                continue
            cls_samples.append(path)
        if cls_samples:
            sample_paths_by_class[cls] = cls_samples

    return sample_paths_by_class, skipped_files


def _validate_image_paths(
    paths: list[str],
    validation_workers: int | None,
    *,
    deep: bool,
) -> dict[str, tuple[bool, str]]:
    if not paths:
        return {}

    if validation_workers is None:
        validation_workers = min(8, os.cpu_count() or 1, len(paths))
    validation_workers = max(1, min(validation_workers, len(paths)))

    if validation_workers == 1:
        return {path: validate_image_file(path, deep=deep) for path in paths}

    with ThreadPoolExecutor(max_workers=validation_workers) as executor:
        results = executor.map(
            lambda path: validate_image_file(path, deep=deep),
            paths,
        )
        return dict(zip(paths, results, strict=True))


def report_integrity_skips(skipped_files: list[tuple[str, str]]) -> None:
    if not skipped_files:
        return

    reason_counts = Counter(reason for _, reason in skipped_files)
    reason_text = ", ".join(
        f"{reason}: {count}" for reason, count in sorted(reason_counts.items())
    )
    print(f"데이터 무결성 검사: 문제 이미지 {len(skipped_files)}장 스킵 ({reason_text})")
    for path, reason in skipped_files[:5]:
        print(f"  - {path} ({reason})")
    if len(skipped_files) > 5:
        print(f"  ... 외 {len(skipped_files) - 5}장")


# ──────────────────────────────────────────────
# Augmentation 정의
# ──────────────────────────────────────────────


def get_train_transforms(img_size: int = 224):
    """
    학습용 증강
    - 색조 변환은 약하게 (캐릭터 구분에 색상이 핵심)
    - 기하학적 변환 위주
    """
    return A.Compose(
        [
            A.RandomResizedCrop(size=(img_size, img_size), scale=(0.7, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.3),
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=3, p=1.0),
                    A.MedianBlur(blur_limit=3, p=1.0),
                ],
                p=0.2,
            ),
            # 색조 변환: 아주 약하게
            A.HueSaturationValue(
                hue_shift_limit=5,  # 색조는 거의 안 건드림
                sat_shift_limit=20,  # 채도는 약간
                val_shift_limit=20,  # 밝기는 약간
                p=0.4,
            ),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
            A.CoarseDropout(
                num_holes_range=(1, 8),
                hole_height_range=(16, 32),
                hole_width_range=(16, 32),
                fill=0,
                p=0.3,
            ),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )


def get_val_transforms(img_size: int = 224):
    """검증/테스트용: 리사이즈 + 정규화만"""
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────


class HoloDataset(Dataset):
    """
    dataset/raw/ 하위 폴더 구조를 읽어 클래스 자동 생성
    폴더 구조:
        dataset/raw/
            houshou_marine/  ← 클래스명 = 폴더명
            usada_pekora/
            others/
    """

    def __init__(
        self,
        root_dir: str | Path,
        transform=None,
        split: str = "train",
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
        validate_images: bool = True,
        deep_validate_images: bool = False,
        sample_paths_by_class: dict[str, list[str]] | None = None,
        skipped_files: list[tuple[str, str]] | None = None,
        report_integrity: bool = True,
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.split = split
        self.skipped_files: list[tuple[str, str]] = skipped_files or []

        if sample_paths_by_class is None:
            sample_paths_by_class, self.skipped_files = _collect_samples_by_class(
                self.root_dir,
                validate_images=validate_images,
                deep_validate_images=deep_validate_images,
            )
            if validate_images and report_integrity:
                report_integrity_skips(self.skipped_files)

        # 클래스 목록 자동 생성 (샘플이 있는 폴더명 정렬)
        classes = sorted(sample_paths_by_class)
        self.class_to_idx = {cls: i for i, cls in enumerate(classes)}
        self.idx_to_class = {i: cls for cls, i in self.class_to_idx.items()}
        self.classes = classes

        # 전체 샘플 수집
        all_samples = []
        for cls in classes:
            for img_path in sample_paths_by_class[cls]:
                all_samples.append((img_path, self.class_to_idx[cls]))

        # 재현 가능한 train/val/test split
        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(all_samples))

        n_test = int(len(all_samples) * test_ratio)
        n_val = int(len(all_samples) * val_ratio)
        n_train = len(all_samples) - n_test - n_val

        if split == "train":
            split_indices = indices[:n_train]
        elif split == "val":
            split_indices = indices[n_train : n_train + n_val]
        else:  # test
            split_indices = indices[n_train + n_val :]

        self.samples = [all_samples[i] for i in split_indices]

        print(f"[{split}] {len(self.samples)}장 | {len(classes)}개 클래스")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        for offset in range(len(self.samples)):
            img_path, label = self.samples[(idx + offset) % len(self.samples)]
            try:
                img = load_rgb_image_array(img_path)
                break
            except IMAGE_LOAD_ERRORS as exc:
                warnings.warn(
                    f"Skipping unreadable image at load time: {img_path} "
                    f"({exc.__class__.__name__})",
                    RuntimeWarning,
                )
        else:
            raise RuntimeError("No readable images left in dataset split.")

        if self.transform:
            augmented = self.transform(image=img)
            img = augmented["image"]

        return img, label

    def get_class_weights(self) -> torch.Tensor:
        """클래스 불균형 보정용 가중치 계산"""
        class_counts = [0] * len(self.classes)
        for _, label in self.samples:
            class_counts[label] += 1

        weights = [1.0 / count if count > 0 else 0 for count in class_counts]
        sample_weights = [weights[label] for _, label in self.samples]
        return torch.tensor(sample_weights, dtype=torch.float)

    def save_class_map(self, save_path: str | Path):
        """클래스 매핑 저장 (API에서 사용)"""
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(self.idx_to_class, f, ensure_ascii=False, indent=2)
        print(f"클래스 맵 저장: {save_path}")


# ──────────────────────────────────────────────
# DataLoader 팩토리
# ──────────────────────────────────────────────


def build_dataloaders(
    root_dir: str | Path,
    img_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    use_weighted_sampler: bool = True,
    device_type: str = "cpu",
    validate_images: bool = True,
    deep_validate_images: bool = False,
):
    """train/val/test DataLoader 한번에 생성"""
    sample_paths_by_class, skipped_files = _collect_samples_by_class(
        root_dir,
        validate_images=validate_images,
        deep_validate_images=deep_validate_images,
    )
    if validate_images:
        report_integrity_skips(skipped_files)

    train_ds = HoloDataset(
        root_dir,
        get_train_transforms(img_size),
        split="train",
        sample_paths_by_class=sample_paths_by_class,
        skipped_files=skipped_files,
        report_integrity=False,
    )
    val_ds = HoloDataset(
        root_dir,
        get_val_transforms(img_size),
        split="val",
        sample_paths_by_class=sample_paths_by_class,
        skipped_files=skipped_files,
        report_integrity=False,
    )
    test_ds = HoloDataset(
        root_dir,
        get_val_transforms(img_size),
        split="test",
        sample_paths_by_class=sample_paths_by_class,
        skipped_files=skipped_files,
        report_integrity=False,
    )

    if len(train_ds.classes) == 0 or len(train_ds.samples) == 0:
        raise RuntimeError(
            f"No training images found in {root_dir}. "
            "Expected dataset/raw/<class>/ image folders."
        )

    # 클래스 불균형 처리
    if use_weighted_sampler:
        sample_weights = train_ds.get_class_weights()
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        shuffle = False
    else:
        sampler = None
        shuffle = True

    pin = device_type in ("cuda", "xpu")
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )

    return train_loader, val_loader, test_loader, train_ds
