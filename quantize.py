#!/usr/bin/env python3
"""통합 양자화 스크립트: fp16 / int8 / int4 / int2

Usage:
  python quantize.py --format fp16
  python quantize.py --format int8
  python quantize.py --format int4
  python quantize.py --format int2
"""
import argparse
import copy
import json
import os
from pathlib import Path

import xpu_compat

import timm
import torch
import torch.nn as nn

CHECKPOINT_DIR = Path("checkpoints")
FP32_PATH      = CHECKPOINT_DIR / "best_model.pth"
CLASS_MAP_PATH = CHECKPOINT_DIR / "class_map.json"
CONFIG_PATH    = CHECKPOINT_DIR / "config.json"
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

OUT_NAMES = {
    "fp16": "best_model_fp16.pth",
    "int8": "best_model_int8.pth",
    "int4": "best_model_int4.pth",
    "int2": "best_model_int2.pth",
}


# ── 모델 빌드 ──────────────────────────────────────────────

def _build_model(num_classes: int) -> nn.Module:
    model = timm.create_model(
        _read_backbone(),
        pretrained=False,
        num_classes=num_classes,
    )
    sd = torch.load(FP32_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(sd)
    return model.eval()


# ── FP16 ──────────────────────────────────────────────────

def _quant_fp16(model: nn.Module) -> dict:
    """float16 state dict — load_state_dict()와 직접 호환되는 plain state dict 반환."""
    return {k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()}


# ── INT8 ──────────────────────────────────────────────────

def _quant_int8(model: nn.Module) -> dict:
    """torch.quantization.quantize_dynamic — Linear 레이어 INT8 동적 양자화."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        qmodel = torch.quantization.quantize_dynamic(
            copy.deepcopy(model), {nn.Linear}, dtype=torch.qint8
        )
    return {"format": "int8", "model": qmodel.state_dict()}


# ── INT4 / INT2 packed ─────────────────────────────────────

def _pack_int4(w: torch.Tensor, scale: float) -> torch.Tensor:
    """symmetric INT4 ([-8, 7]) — 2 values per byte (high nibble | low nibble)."""
    flat = w.reshape(-1).float()
    q = (flat / scale).round().clamp(-8, 7).to(torch.int8)
    if q.numel() % 2:
        q = torch.cat([q, torch.zeros(1, dtype=torch.int8)])
    hi = (q[0::2] & 0x0F).to(torch.uint8)
    lo = (q[1::2] & 0x0F).to(torch.uint8)
    return (hi << 4) | lo


def _pack_int2(w: torch.Tensor, scale: float) -> torch.Tensor:
    """symmetric INT2 ([-2, 1]) — 4 values per byte."""
    flat = w.reshape(-1).float()
    q = (flat / scale).round().clamp(-2, 1).to(torch.int8) & 0x03
    pad = (-q.numel()) % 4
    if pad:
        q = torch.cat([q, torch.zeros(pad, dtype=torch.int8)])
    return (q[0::4].to(torch.uint8) << 6 |
            q[1::4].to(torch.uint8) << 4 |
            q[2::4].to(torch.uint8) << 2 |
            q[3::4].to(torch.uint8))


def _quant_packed(model: nn.Module, bits: int) -> dict:
    pack_fn = _pack_int4 if bits == 4 else _pack_int2
    levels  = (2 ** (bits - 1)) - 1   # int4 → 7, int2 → 1
    packed_sd: dict = {}

    for k, v in model.state_dict().items():
        if v.is_floating_point() and v.dim() >= 2:
            scale = v.abs().max().item() / levels if levels > 0 else 1.0
            if scale == 0.0:
                scale = 1.0
            packed_sd[k] = {
                "q":     pack_fn(v, scale),
                "scale": scale,
                "shape": list(v.shape),
                "bits":  bits,
            }
        else:
            packed_sd[k] = v

    return {"format": f"int{bits}", "packed_state_dict": packed_sd}


# ── 메인 ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=["fp16", "int8", "int4", "int2"], required=True)
    args = parser.parse_args()

    fmt      = args.format
    out_path = CHECKPOINT_DIR / OUT_NAMES[fmt]

    print(f"양자화 형식: {fmt.upper()}")
    print(f"FP32 소스:   {FP32_PATH}")

    with open(CLASS_MAP_PATH) as f:
        num_classes = len(json.load(f))
    print(f"클래스 수:   {num_classes}")

    model = _build_model(num_classes)

    print(f"\n[변환 중] {fmt.upper()} 양자화...")
    if fmt == "fp16":
        payload = _quant_fp16(model)
    elif fmt == "int8":
        payload = _quant_int8(model)
    elif fmt == "int4":
        payload = _quant_packed(model, 4)
    else:
        payload = _quant_packed(model, 2)

    torch.save(payload, out_path)

    fp32_mb = os.path.getsize(FP32_PATH) / 1024 ** 2
    out_mb  = os.path.getsize(out_path)  / 1024 ** 2
    ratio   = out_mb / fp32_mb * 100

    print(f"\n{'='*46}")
    print(f"  {'형식':<8} {'크기':>10}   {'압축률':>8}")
    print(f"  {'-'*42}")
    print(f"  {'FP32':<8} {fp32_mb:>8.1f} MB   {'100%':>8}")
    print(f"  {fmt.upper():<8} {out_mb:>8.1f} MB   {ratio:>7.1f}%")
    print(f"{'='*46}")
    print(f"\n저장 완료: {out_path}")

    # 구조화 메트릭 — export_job.py 가 파싱
    print(f"EXPORT_RESULT:{json.dumps({'format': fmt, 'fp32_mb': round(fp32_mb, 2), 'out_mb': round(out_mb, 2), 'ratio_pct': round(ratio, 1)})}")


if __name__ == "__main__":
    main()
