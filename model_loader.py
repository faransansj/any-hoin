"""Shared model loading and inference utilities."""

import json
from pathlib import Path

import xpu_compat
import albumentations as A
import numpy as np
from PIL import Image

CHECKPOINT_DIR = Path("./checkpoints")
MODEL_PATH = CHECKPOINT_DIR / "best_model.pth"
ONNX_PATH = CHECKPOINT_DIR / "best_model.onnx"
CLASS_MAP_PATH = CHECKPOINT_DIR / "class_map.json"
CONFIG_PATH = CHECKPOINT_DIR / "config.json"
IMG_SIZE = 224
_DEFAULT_BACKBONE = "swin_tiny_patch4_window7_224"


def _read_backbone() -> str:
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, encoding="utf-8") as f:
                cfg = json.load(f)
            return cfg.get("backbone") or cfg.get("model") or _DEFAULT_BACKBONE
        except Exception:
            pass
    return _DEFAULT_BACKBONE

_ORT_PROVIDERS = [
    "ROCMExecutionProvider",
    "MIGraphXExecutionProvider",
    "CPUExecutionProvider",
]


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


class ModelLoader:
    _instance = None

    def __init__(self):
        self.session = None
        self.torch_model = None
        self.torch_device = None
        self.idx_to_class: dict[int, str] = {}
        self.transform = None
        self.backend: str = ""
        self._load()

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load(self):
        if not CLASS_MAP_PATH.exists():
            raise FileNotFoundError(f"class map not found: {CLASS_MAP_PATH}")

        with open(CLASS_MAP_PATH, encoding="utf-8") as f:
            raw = json.load(f)
        self.idx_to_class = {int(k): v for k, v in raw.items()}
        num_classes = len(self.idx_to_class)
        if num_classes <= 0:
            raise RuntimeError("class map is empty")

        if ONNX_PATH.exists():
            self._load_onnx()
        else:
            if not MODEL_PATH.exists():
                raise FileNotFoundError(f"model not found: {MODEL_PATH}")
            self._load_torch(num_classes)

        self.transform = A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        print(f"model loaded: {num_classes} classes, backend={self.backend}")

    def _load_onnx(self):
        import onnxruntime as ort

        available = ort.get_available_providers()
        providers = [p for p in _ORT_PROVIDERS if p in available]
        if not providers:
            providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(ONNX_PATH), providers=providers)
        self.backend = f"onnx+{providers[0]}"

    def _load_torch(self, num_classes: int):
        import timm
        import torch

        backbone = _read_backbone()
        self.torch_device = xpu_compat.best_device()
        model = timm.create_model(
            backbone,
            pretrained=False,
            num_classes=num_classes,
        )
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
        try:
            self.torch_model = model.to(self.torch_device).eval()
        except Exception as exc:
            print(f"[device] 모델을 {self.torch_device}로 올리지 못해 CPU로 폴백합니다: {exc}")
            self.torch_device = torch.device("cpu")
            self.torch_model = model.to(self.torch_device).eval()
        self.backend = f"torch+{self.torch_device}"

    def predict(self, img: Image.Image) -> tuple[str, float]:
        img_np = np.array(img.convert("RGB"))
        transformed = self.transform(image=img_np)["image"]
        inp = transformed.transpose(2, 0, 1)[np.newaxis].astype(np.float32)

        if self.session is not None:
            logits = self.session.run(["logits"], {"input": inp})[0][0]
        else:
            import torch

            try:
                tensor = torch.from_numpy(inp).to(self.torch_device)
                with torch.no_grad():
                    logits = self.torch_model(tensor).cpu().numpy()[0]
            except Exception as exc:
                if self.torch_device.type == "cpu":
                    raise
                print(f"[device] 추론 중 {self.torch_device} 실패 — CPU로 재시도합니다: {exc}")
                self.torch_device = torch.device("cpu")
                self.torch_model = self.torch_model.to(self.torch_device).eval()
                tensor = torch.from_numpy(inp).to(self.torch_device)
                with torch.no_grad():
                    logits = self.torch_model(tensor).cpu().numpy()[0]
                self.backend = f"torch+{self.torch_device}"

        probs = softmax(logits)
        top_idx = int(probs.argmax())
        return self.idx_to_class[top_idx], float(probs[top_idx])
