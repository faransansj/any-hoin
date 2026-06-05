"""추론 테스트 라우터 — 학습된 모델로 실시간 분류."""

import io
import json
import shutil
from pathlib import Path

import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image

import studio.characters as ch
from model_loader import ModelLoader

CHECKPOINT_DIR = Path("./checkpoints")
IMG_SIZE       = 224
TOP_K          = 5
SELECTED_MODEL_DIR_NAME = "selected_inference"
MODEL_EXTENSIONS = {".pth", ".pt", ".onnx"}
METADATA_EXTENSIONS = {".json"}

router = APIRouter(prefix="/inference", tags=["inference"])

# 지연 로딩 — 추론 요청 시 최초 1회만 로드
_loader = None
_selected_model_path: Path | None = None


def _default_model_path() -> Path | None:
    onnx = CHECKPOINT_DIR / "best_model.onnx"
    fp32 = CHECKPOINT_DIR / "best_model.pth"
    if onnx.exists():
        return onnx
    if fp32.exists():
        return fp32
    return None


def _active_model_path() -> Path | None:
    if _selected_model_path is not None and _selected_model_path.exists():
        return _selected_model_path
    return _default_model_path()


def _has_selected_model() -> bool:
    return _selected_model_path is not None and _selected_model_path.exists()


def _active_class_map_path() -> Path:
    if _has_selected_model():
        selected_class_map = _selected_model_path.parent / "class_map.json"
        if selected_class_map.exists():
            return selected_class_map
    return CHECKPOINT_DIR / "class_map.json"


def _active_config_path() -> Path:
    if _has_selected_model():
        selected_config = _selected_model_path.parent / "config.json"
        if selected_config.exists():
            return selected_config
    return CHECKPOINT_DIR / "config.json"


def _loader_for(path: Path) -> ModelLoader:
    class_map_path = _active_class_map_path()
    config_path = _active_config_path()
    suffix = path.suffix.lower()
    if suffix == ".onnx":
        return ModelLoader(
            model_path=CHECKPOINT_DIR / "__unused_selected_model__.pth",
            onnx_path=path,
            class_map_path=class_map_path,
            config_path=config_path,
        )
    return ModelLoader(
        model_path=path,
        onnx_path=CHECKPOINT_DIR / "__unused_selected_model__.onnx",
        class_map_path=class_map_path,
        config_path=config_path,
    )


def _model_size_mb(path: Path | None) -> float | None:
    if path is None or not path.exists():
        return None
    return round(path.stat().st_size / (1024 * 1024), 2)


def _safe_upload_name(filename: str | None) -> str:
    name = Path(filename or "").name
    if not name or name in {".", ".."}:
        raise HTTPException(400, "Invalid model filename")
    if "/" in name or "\\" in name:
        raise HTTPException(400, "Invalid model filename")
    return name


def _selected_model_dir() -> Path:
    return CHECKPOINT_DIR / SELECTED_MODEL_DIR_NAME


def _clear_selected_model_dir() -> None:
    selected_dir = _selected_model_dir()
    if not selected_dir.exists():
        return
    for child in selected_dir.iterdir():
        if child.is_file() or child.is_symlink():
            child.unlink()
        elif child.is_dir():
            shutil.rmtree(child)


def _get_loader():
    global _loader
    if _loader is None:
        path = _active_model_path()
        if path is None:
            raise FileNotFoundError("model not found")
        if _has_selected_model():
            _loader = _loader_for(path)
        else:
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

    config_path = _active_config_path()
    config = {}
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    class_map_path = _active_class_map_path()
    class_count = None
    if class_map_path.exists():
        try:
            with open(class_map_path, encoding="utf-8") as f:
                class_count = len(json.load(f))
        except (json.JSONDecodeError, OSError):
            class_count = None

    loaded_backend = None
    loaded_model = None
    if _loader is not None:
        loaded_backend = getattr(_loader, "backend", None)
        loaded_path = getattr(_loader, "loaded_path", None)
        loaded_model = loaded_path.name if loaded_path else None

    active_path = _active_model_path()

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
        "loaded_model": loaded_model,
        "active_model": active_path.name if active_path else None,
        "active_model_size_mb": _model_size_mb(active_path),
        "custom_model_selected": _has_selected_model(),
        "model_ready": bool(active_path and class_count),
    }


@router.post("/model")
async def load_model(files: list[UploadFile] = File(...)):
    global _loader, _selected_model_path

    if not files:
        raise HTTPException(400, "Model file required")

    allowed_extensions = MODEL_EXTENSIONS | METADATA_EXTENSIONS
    unsupported = []
    for upload in files:
        name = _safe_upload_name(upload.filename)
        suffix = Path(name).suffix.lower()
        if suffix not in allowed_extensions and not name.endswith(".onnx.data"):
            unsupported.append(name)
    if unsupported:
        allowed = ", ".join(sorted(MODEL_EXTENSIONS | METADATA_EXTENSIONS)) + ", .onnx.data"
        raise HTTPException(400, f"Unsupported upload file. Allowed: {allowed}")

    model_files = [
        upload for upload in files
        if Path(upload.filename or "").suffix.lower() in MODEL_EXTENSIONS
    ]
    if len(model_files) != 1:
        allowed = ", ".join(sorted(MODEL_EXTENSIONS))
        raise HTTPException(400, f"Select exactly one model file. Allowed: {allowed}")

    main_upload = model_files[0]
    main_name = _safe_upload_name(main_upload.filename)
    suffix = Path(main_name).suffix.lower()

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    selected_dir = _selected_model_dir()
    backup_dir = CHECKPOINT_DIR / f"{SELECTED_MODEL_DIR_NAME}.bak"
    target = selected_dir / main_name
    try:
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        if selected_dir.exists():
            selected_dir.replace(backup_dir)
        selected_dir.mkdir(parents=True, exist_ok=True)

        for upload in files:
            name = _safe_upload_name(upload.filename)
            upload_path = selected_dir / name
            with open(upload_path, "wb") as out:
                shutil.copyfileobj(upload.file, out)

        if suffix == ".onnx":
            sidecar_name = f"{main_name}.data"
            sidecar_target = selected_dir / sidecar_name
            sidecar_source = CHECKPOINT_DIR / sidecar_name
            if not sidecar_target.exists() and sidecar_source.exists():
                shutil.copy2(sidecar_source, sidecar_target)
    except OSError as exc:
        if selected_dir.exists():
            shutil.rmtree(selected_dir)
        if backup_dir.exists():
            backup_dir.replace(selected_dir)
        raise HTTPException(500, f"Model upload failed: {exc}")
    finally:
        for upload in files:
            await upload.close()

    previous_selected = _selected_model_path
    previous_loader = _loader
    _selected_model_path = target
    _loader = None

    try:
        _get_loader()
    except Exception as exc:
        _selected_model_path = previous_selected
        _loader = previous_loader
        try:
            if selected_dir.exists():
                shutil.rmtree(selected_dir)
            if backup_dir.exists():
                backup_dir.replace(selected_dir)
        except OSError:
            pass
        raise HTTPException(400, f"Model load failed: {exc}")

    try:
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
    except OSError:
        pass

    return model_info()


@router.delete("/model")
def clear_model():
    global _loader, _selected_model_path
    _selected_model_path = None
    _loader = None
    ModelLoader.reset()
    try:
        _clear_selected_model_dir()
    except OSError as exc:
        raise HTTPException(500, f"Failed to clear selected model: {exc}")
    return model_info()


@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "Image file required")

    if _active_model_path() is None:
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
