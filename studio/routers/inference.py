"""추론 테스트 라우터 — 학습된 모델로 실시간 분류."""

import io
import json
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image

import studio.characters as ch
from model_loader import ModelLoader

CHECKPOINT_DIR = Path("./checkpoints")
IMG_SIZE       = 224
TOP_K          = 5

router = APIRouter(prefix="/inference", tags=["inference"])

# 지연 로딩 — 추론 요청 시 최초 1회만 로드
_loader = None


def _get_loader():
    global _loader
    if _loader is None:
        _loader = ModelLoader.get()
    return _loader


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


@router.get("/model-info")
def model_info():
    fp32 = (CHECKPOINT_DIR / "best_model.pth").exists()
    onnx = (CHECKPOINT_DIR / "best_model.onnx").exists()
    fp16 = (CHECKPOINT_DIR / "best_model_fp16.pth").exists()
    int8 = (CHECKPOINT_DIR / "best_model_int8.pth").exists()
    int4 = (CHECKPOINT_DIR / "best_model_int4.pth").exists()
    int2 = (CHECKPOINT_DIR / "best_model_int2.pth").exists()

    config_path = CHECKPOINT_DIR / "config.json"
    config = {}
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    class_map_path = CHECKPOINT_DIR / "class_map.json"
    class_count = None
    if class_map_path.exists():
        try:
            with open(class_map_path, encoding="utf-8") as f:
                class_count = len(json.load(f))
        except (json.JSONDecodeError, OSError):
            class_count = None

    loaded_backend = None
    if _loader is not None:
        loaded_backend = getattr(_loader, "backend", None)

    return {
        "fp32_available": fp32,
        "fp16_available": fp16,
        "int8_available": int8,
        "int4_available": int4,
        "int2_available": int2,
        "onnx_available": onnx,
        "num_classes":    config.get("num_classes") or class_count,
        "best_val_acc":   config.get("best_val_acc"),
        "test_acc":       config.get("test_acc"),
        "preferred_backend": "onnx" if onnx else "torch" if fp32 else None,
        "loaded_backend": loaded_backend,
        "model_ready": bool((onnx or fp32) and class_count),
    }


@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "Image file required")

    if not (CHECKPOINT_DIR / "best_model.pth").exists() and \
       not (CHECKPOINT_DIR / "best_model.onnx").exists():
        raise HTTPException(503, "No model found. Train a model first.")

    try:
        loader = _get_loader()
    except Exception as e:
        raise HTTPException(503, f"Model load failed: {e}")

    content = await file.read()
    try:
        img = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Unreadable image file")

    # top-K 예측
    img_np      = np.array(img)
    transformed = loader.transform(image=img_np)["image"]
    inp         = transformed.transpose(2, 0, 1)[np.newaxis].astype(np.float32)

    if loader.session is not None:
        logits = loader.session.run(["logits"], {"input": inp})[0][0]
    else:
        import torch
        tensor = torch.from_numpy(inp).to(loader.torch_device)
        with torch.no_grad():
            logits = loader.torch_model(tensor).cpu().numpy()[0]

    probs    = _softmax(logits)
    top_idxs = probs.argsort()[::-1][:TOP_K]

    chars = ch.load()
    results = []
    for i, idx in enumerate(top_idxs):
        key  = loader.idx_to_class[int(idx)]
        meta = chars.get(key, {})
        results.append({
            "rank":         int(i + 1),
            "character":    key,
            "display_name": meta.get("display_name", key.replace("_", " ").title()),
            "confidence":   float(round(probs[idx], 4)),
        })

    return {"filename": file.filename, "top_k": results}
