"""
이미지 분류 학습 스크립트
- Phase 1: Head만 학습 (5 epoch)
- Phase 2: 전체 fine-tune (낮은 lr)
- Mixup / CutMix augmentation
- EMA (Exponential Moving Average)
- WandB 로깅 (선택)
- 최고 val_acc 모델 자동 저장
- Early stopping 지원
- Intel Arc GPU (XPU) 지원 — IPEX 설치 시 자동 활성화
"""

import copy
import os
import sys
import json
import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import xpu_compat

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
warnings.filterwarnings("ignore", message=r".*Corrupt EXIF data.*", category=UserWarning)
warnings.filterwarnings("ignore", message=r".*Possibly corrupt EXIF data.*", category=UserWarning)
warnings.filterwarnings("ignore", message=r".*Truncated File Read.*", category=UserWarning)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
import timm
from tqdm import tqdm

from dataset import build_dataloaders

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

IPEX_AVAILABLE = False

TRAIN_EVENT_PREFIX = "__HOLOSCOPE_TRAIN_EVENT__ "


def emit_train_event(event_type: str, data: dict):
    """Studio가 tqdm 문자열 대신 구조화 이벤트를 파싱하도록 JSON 라인을 출력."""
    payload = {"type": event_type, "data": data}
    print(TRAIN_EVENT_PREFIX + json.dumps(payload, ensure_ascii=False), flush=True)


def emit_progress_event(
    split: str,
    batch_cur: int,
    batch_total: int,
    started_at: float,
    avg_loss: float | None = None,
    avg_acc: float | None = None,
):
    elapsed = max(time.monotonic() - started_at, 1e-6)
    speed_it_s = batch_cur / elapsed
    remaining = max(batch_total - batch_cur, 0)
    eta_sec = remaining / speed_it_s if speed_it_s > 0 else -1.0
    data = {
        "split": split,
        "pct": int((batch_cur / batch_total) * 100) if batch_total > 0 else 0,
        "batch_cur": batch_cur,
        "batch_total": batch_total,
        "eta_sec": round(eta_sec, 1),
        "speed_it_s": round(speed_it_s, 2),
    }
    if avg_loss is not None:
        data["avg_loss"] = round(avg_loss, 4)
    if avg_acc is not None:
        data["avg_acc"] = round(avg_acc, 4)
    emit_train_event("progress", data)


def emit_metric_event(
    phase: int,
    epoch: int,
    total_epochs: int,
    train_loss: float,
    train_acc: float,
    val_loss: float,
    val_acc: float,
):
    emit_train_event(
        "metric",
        {
            "phase": phase,
            "epoch": epoch,
            "total_epochs": total_epochs,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
        },
    )


# ──────────────────────────────────────────────
# 디바이스 감지
# ──────────────────────────────────────────────


class DeviceUnavailableError(RuntimeError):
    """요청한 가속 장치를 현재 PyTorch 환경에서 사용할 수 없음."""


def _require_device_available(device: torch.device, option: str):
    try:
        xpu_compat.require_device_available(device, option)
    except RuntimeError as exc:
        raise DeviceUnavailableError(str(exc)) from exc


def detect_device(
    force_xpu: bool = False, force_cpu: bool = False, device_str: str = ""
) -> torch.device:
    """디바이스 우선순위: --device 명시 > --xpu > --cpu > xpu > cuda > mps > cpu"""
    if force_cpu:
        return torch.device("cpu")
    if device_str:
        if device_str.lower() == "auto":
            device_str = ""
        else:
            device = torch.device(device_str)
            _require_device_available(device, f"--device {device_str}")
            return device
    if force_xpu:
        if xpu_compat.xpu_available():
            return torch.device("xpu")
        raise DeviceUnavailableError(xpu_compat.xpu_unavailable_message("--xpu"))
    if xpu_compat.xpu_available():
        return torch.device("xpu")
    if xpu_compat.cuda_available():
        return torch.device("cuda")
    if xpu_compat.mps_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_scaler(device: torch.device, enabled: bool) -> torch.amp.GradScaler:
    """device-aware GradScaler 생성 (XPU / CUDA / CPU)"""
    return xpu_compat.make_grad_scaler(device.type, enabled=enabled)


# ──────────────────────────────────────────────
# 모델
# ──────────────────────────────────────────────


DEFAULT_BACKBONE = "swin_tiny_patch4_window7_224"

AVAILABLE_BACKBONES = [
    {"key": "swin_tiny_patch4_window7_224",     "label": "Swin-T · 28M",     "params_m": 28,  "description": "기본 백본 — 빠른 학습"},
    {"key": "swin_small_patch4_window7_224",    "label": "Swin-S · 50M",     "params_m": 50,  "description": "Tiny 대비 +3~5% 정확도"},
    {"key": "swin_base_patch4_window7_224",     "label": "Swin-B · 88M",     "params_m": 88,  "description": "고정밀도 — VRAM 2배 이상"},
    {"key": "convnext_small.fb_in22k_ft_in1k",  "label": "ConvNeXt-S · 50M", "params_m": 50,  "description": "IN-22K 프리트레인, 빠른 수렴"},
    {"key": "convnext_base.fb_in22k_ft_in1k",   "label": "ConvNeXt-B · 89M", "params_m": 89,  "description": "IN-22K 프리트레인, 고정밀도"},
]


def build_model(num_classes: int, model_name: str = DEFAULT_BACKBONE, pretrained: bool = True) -> nn.Module:
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    return model


def freeze_backbone(model: nn.Module):
    """head만 학습 (Phase 1)"""
    for name, param in model.named_parameters():
        if "head" not in name:
            param.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Phase 1: head만 학습 | 학습 파라미터: {trainable:,}")


def unfreeze_all(model: nn.Module):
    """전체 학습 (Phase 2)"""
    for param in model.parameters():
        param.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Phase 2: 전체 fine-tune | 학습 파라미터: {trainable:,}")


# ──────────────────────────────────────────────
# Augmentation Mix (Mixup / CutMix)
# ──────────────────────────────────────────────


def _pick_aug(mixup_alpha: float, cutmix_alpha: float) -> str:
    """두 augmentation이 모두 활성화된 경우 50/50으로 선택."""
    if mixup_alpha > 0 and cutmix_alpha > 0:
        return "mixup" if np.random.random() < 0.5 else "cutmix"
    if mixup_alpha > 0:
        return "mixup"
    if cutmix_alpha > 0:
        return "cutmix"
    return ""


def mixup_data(imgs: torch.Tensor, labels: torch.Tensor, alpha: float):
    """배치 내 두 샘플을 선형 혼합. lam = Beta(alpha, alpha) 샘플."""
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(imgs.size(0), device=imgs.device)
    mixed = lam * imgs + (1.0 - lam) * imgs[idx]
    return mixed, labels, labels[idx], lam


def cutmix_data(imgs: torch.Tensor, labels: torch.Tensor, alpha: float):
    """무작위 직사각형 패치를 교체. 실제 교체 면적 비율로 lam 재계산."""
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(imgs.size(0), device=imgs.device)
    H, W = imgs.shape[2], imgs.shape[3]
    cut_rat = (1.0 - lam) ** 0.5
    cut_h, cut_w = int(H * cut_rat), int(W * cut_rat)
    cy, cx = np.random.randint(H), np.random.randint(W)
    y1 = max(0, cy - cut_h // 2);  y2 = min(H, cy + cut_h // 2)
    x1 = max(0, cx - cut_w // 2);  x2 = min(W, cx + cut_w // 2)
    mixed = imgs.clone()
    mixed[:, :, y1:y2, x1:x2] = imgs[idx, :, y1:y2, x1:x2]
    lam = 1.0 - (y2 - y1) * (x2 - x1) / (H * W)
    return mixed, labels, labels[idx], lam


# ──────────────────────────────────────────────
# EMA (Exponential Moving Average)
# ──────────────────────────────────────────────


class ModelEMA:
    """학습 파라미터의 지수 이동 평균 — 추론 시 일관된 정확도 향상."""

    def __init__(self, model: nn.Module, decay: float = 0.9998):
        self.ema = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for ema_p, model_p in zip(self.ema.parameters(), model.parameters()):
            ema_p.lerp_(model_p.detach().to(ema_p.device), 1.0 - self.decay)
        for ema_b, model_b in zip(self.ema.buffers(), model.buffers()):
            ema_b.copy_(model_b.to(ema_b.device))


# ──────────────────────────────────────────────
# 학습 루프
# ──────────────────────────────────────────────


def train_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    scaler,
    use_amp,
    accumulation_steps=1,
    progress_callback=None,
    mixup_alpha=0.0,
    cutmix_alpha=0.0,
    ema=None,
):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    optimizer.zero_grad(set_to_none=True)
    total_batches = len(loader)
    started_at = time.monotonic()
    last_progress_at = 0.0

    for i, (imgs, labels) in enumerate(tqdm(loader, desc="  train", leave=False)):
        imgs, labels = imgs.to(device), labels.to(device)

        # Mixup / CutMix
        aug = _pick_aug(mixup_alpha, cutmix_alpha)
        if aug == "mixup":
            imgs, y_a, y_b, lam = mixup_data(imgs, labels, mixup_alpha)
        elif aug == "cutmix":
            imgs, y_a, y_b, lam = cutmix_data(imgs, labels, cutmix_alpha)
        else:
            y_a, y_b, lam = labels, labels, 1.0

        with xpu_compat.autocast(device.type, enabled=use_amp):
            outputs = model(imgs)
            if aug:
                loss = lam * criterion(outputs, y_a) + (1.0 - lam) * criterion(outputs, y_b)
            else:
                loss = criterion(outputs, labels)
            loss = loss / accumulation_steps

        scaler.scale(loss).backward()

        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(loader):
            if use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update(model)

        total_loss += loss.item() * accumulation_steps * imgs.size(0)
        # 정확도는 primary label 기준 (Mixup/CutMix 적용 시 근사값)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += imgs.size(0)

        batch_cur = i + 1
        now = time.monotonic()
        if progress_callback and (
            batch_cur == total_batches or now - last_progress_at >= 1.0
        ):
            progress_callback(
                "train",
                batch_cur,
                total_batches,
                started_at,
                total_loss / max(total, 1),
                correct / max(total, 1),
            )
            last_progress_at = now

    return total_loss / total, correct / total


@torch.no_grad()
def val_epoch(model, loader, criterion, device, use_amp, progress_callback=None):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    total_batches = len(loader)
    started_at = time.monotonic()
    last_progress_at = 0.0

    for i, (imgs, labels) in enumerate(tqdm(loader, desc="  val", leave=False)):
        imgs, labels = imgs.to(device), labels.to(device)

        with xpu_compat.autocast(device.type, enabled=use_amp):
            outputs = model(imgs)
            loss = criterion(outputs, labels)

        total_loss += loss.item() * imgs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += imgs.size(0)

        batch_cur = i + 1
        now = time.monotonic()
        if progress_callback and (
            batch_cur == total_batches or now - last_progress_at >= 1.0
        ):
            progress_callback(
                "val",
                batch_cur,
                total_batches,
                started_at,
                total_loss / max(total, 1),
                correct / max(total, 1),
            )
            last_progress_at = now

    return total_loss / total, correct / total


@torch.no_grad()
def test_epoch(model, loader, criterion, device, use_amp, num_classes: int):
    """테스트: 전체 정확도 + 클래스별 정확도"""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    for imgs, labels in tqdm(loader, desc="  test", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)

        with xpu_compat.autocast(device.type, enabled=use_amp):
            outputs = model(imgs)
            loss = criterion(outputs, labels)

        preds = outputs.argmax(1)
        total_loss += loss.item() * imgs.size(0)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)

        for pred, label in zip(preds.cpu(), labels.cpu()):
            class_total[label] += 1
            if pred == label:
                class_correct[label] += 1

    per_class_acc = {
        i: class_correct[i] / class_total[i]
        for i in range(num_classes)
        if class_total[i] > 0
    }
    return total_loss / total, correct / total, per_class_acc


# ──────────────────────────────────────────────
# 체크포인트
# ──────────────────────────────────────────────


def save_checkpoint(
    path,
    phase,
    epoch,
    model,
    optimizer,
    scheduler,
    scaler,
    best_val_acc,
    patience_counter,
):
    torch.save(
        {
            "phase": phase,
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "best_val_acc": best_val_acc,
            "patience_counter": patience_counter,
        },
        path,
    )


def load_checkpoint(path, device):
    return torch.load(path, weights_only=False, map_location=device)


def load_config_best_val_acc(save_dir: Path) -> float:
    config_path = save_dir / "config.json"
    if not config_path.exists():
        return 0.0
    try:
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
        return max(0.0, float(config.get("best_val_acc") or 0.0))
    except Exception:
        return 0.0


def _optimizer_state_mismatch_reason(optimizer, state: dict) -> str | None:
    groups = state.get("param_groups")
    if not isinstance(groups, list):
        return "param_groups가 없습니다"
    if len(groups) != len(optimizer.param_groups):
        return (
            f"param group 수가 다릅니다 "
            f"(checkpoint={len(groups)}, current={len(optimizer.param_groups)})"
        )
    for idx, (saved_group, current_group) in enumerate(
        zip(groups, optimizer.param_groups)
    ):
        saved_params = saved_group.get("params", [])
        current_params = current_group.get("params", [])
        if len(saved_params) != len(current_params):
            return (
                f"param group {idx} 크기가 다릅니다 "
                f"(checkpoint={len(saved_params)}, current={len(current_params)})"
            )
    return None


def restore_runtime_state(ckpt, optimizer, scheduler, scaler, phase_label: str) -> None:
    """체크포인트 runtime state를 가능한 만큼 복원.

    모델 가중치 복원 뒤 optimizer/scheduler/scaler가 현재 환경과 안 맞아도
    학습을 중단하지 않고 새 runtime state로 이어간다.
    """
    optimizer_state = ckpt.get("optimizer_state")
    optimizer_loaded = False

    if optimizer_state:
        reason = _optimizer_state_mismatch_reason(optimizer, optimizer_state)
        if reason:
            print(
                f"[경고] {phase_label} optimizer_state 불일치 — "
                f"모델 가중치는 유지하고 optimizer/scheduler를 새로 시작합니다: {reason}"
            )
        else:
            try:
                optimizer.load_state_dict(optimizer_state)
                optimizer_loaded = True
            except Exception as exc:
                print(
                    f"[경고] {phase_label} optimizer_state 로드 실패 — "
                    f"모델 가중치는 유지하고 optimizer/scheduler를 새로 시작합니다: {exc}"
                )

    if optimizer_loaded:
        try:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        except Exception as exc:
            print(
                f"[경고] {phase_label} scheduler_state 로드 실패 — "
                f"scheduler를 새로 시작합니다: {exc}"
            )

    scaler_state = ckpt.get("scaler_state")
    if scaler_state:
        try:
            scaler.load_state_dict(scaler_state)
        except Exception as exc:
            print(
                f"[경고] {phase_label} scaler_state 로드 실패 — "
                f"scaler를 새로 시작합니다: {exc}"
            )


# ──────────────────────────────────────────────
# 메인 학습
# ──────────────────────────────────────────────


def train(args):
    global IPEX_AVAILABLE

    finetune = bool(getattr(args, "finetune", False))
    fresh = bool(getattr(args, "fresh", False))

    if finetune and fresh:
        raise ValueError("--finetune and --fresh cannot be used together.")

    device = detect_device(
        force_xpu=args.xpu,
        force_cpu=args.cpu,
        device_str=args.device,
    )
    IPEX_AVAILABLE = device.type == "xpu" and xpu_compat.ipex_available()
    use_amp = device.type in ("cuda", "xpu") and not args.no_amp
    if device.type == "mps" and not args.no_amp:
        print(
            "⚠️  주의: Apple Silicon MPS는 현재 torch.amp.autocast를 공식 지원하지 않습니다."
        )
        print("   - 기본적으로 FP32로 학습하며, AMP 관련 최적화는 적용되지 않습니다.")
        print("   - 더 빠른 학습을 원하시면 가급적 CUDA/XPU 환경을 권장합니다.")

    mixup_alpha  = float(getattr(args, "mixup_alpha",  0.0))
    cutmix_alpha = float(getattr(args, "cutmix_alpha", 0.0))
    ema_decay    = float(getattr(args, "ema_decay",    0.0))

    aug_tags = []
    if mixup_alpha  > 0: aug_tags.append(f"Mixup(α={mixup_alpha})")
    if cutmix_alpha > 0: aug_tags.append(f"CutMix(α={cutmix_alpha})")
    if ema_decay    > 0: aug_tags.append(f"EMA(decay={ema_decay})")

    print(
        f"Device: {device} | AMP: {use_amp}"
        + (" | IPEX" if (IPEX_AVAILABLE and device.type == "xpu") else "")
        + (" | Apple Silicon MPS" if device.type == "mps" else "")
        + (f" | {', '.join(aug_tags)}" if aug_tags else "")
    )

    # WandB 초기화
    use_wandb = args.wandb and WANDB_AVAILABLE
    if args.wandb and not WANDB_AVAILABLE:
        print("WandB 미설치: uv sync --extra logging 으로 설치 가능")
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            config=vars(args),
            name=args.wandb_run,
        )

    # 얼굴 크롭 전처리 (--face-crop-dir 지정 시 해당 디렉토리를 학습 데이터로 사용)
    effective_data_dir = args.data_dir
    face_crop_dir = getattr(args, "face_crop_dir", "")
    if face_crop_dir:
        import os
        if os.path.isdir(face_crop_dir):
            effective_data_dir = face_crop_dir
            print(f"[얼굴 크롭 데이터] {face_crop_dir} 를 학습 데이터로 사용합니다.")
        else:
            print(f"[경고] 얼굴 크롭 디렉토리를 찾을 수 없습니다: {face_crop_dir} — 원본 데이터 사용")

    # 데이터로더
    train_loader, val_loader, test_loader, train_ds = build_dataloaders(
        root_dir=effective_data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device_type=device.type,
        deep_validate_images=getattr(args, "deep_validate_images", False),
    )

    num_classes = len(train_ds.classes)
    if num_classes <= 0 or len(train_ds.samples) == 0:
        raise RuntimeError(
            f"No training images found in {args.data_dir}. "
            "Populate dataset/raw/<class>/ with images before training."
        )
    print(f"클래스 수: {num_classes}")
    backbone = getattr(args, "backbone", DEFAULT_BACKBONE)

    # 클래스 맵 저장
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = save_dir / "best_model.pth"
    train_ds.save_class_map(save_dir / "class_map.json")

    # 모델
    model = build_model(num_classes, backbone).to(device)

    # Loss: Label Smoothing으로 과적합 방지
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = make_scaler(device, enabled=use_amp)

    # IPEX XPU 최적화 적용 여부 플래그
    _use_ipex = IPEX_AVAILABLE and device.type == "xpu"

    best_val_acc = max(0.0, float(getattr(args, "initial_best_val_acc", 0.0)))
    patience_counter = 0
    resume_phase, resume_epoch = 1, 0
    ckpt = None

    # --finetune: best_model.pth를 로드해 Phase 2부터 추가학습
    if finetune:
        best_pth = best_model_path
        if not best_pth.exists():
            raise FileNotFoundError(f"--finetune 모드인데 {best_pth} 가 없습니다.")
        if best_val_acc <= 0:
            best_val_acc = load_config_best_val_acc(save_dir)
        state = torch.load(best_pth, weights_only=True, map_location=device)
        current = model.state_dict()
        mismatched = {k for k, v in state.items() if k in current and v.shape != current[k].shape}
        if mismatched:
            print(f"[finetune] 클래스 수 불일치 — head 레이어를 재초기화합니다: {mismatched}")
            state = {k: v for k, v in state.items() if k not in mismatched}
            best_val_acc = 0.0
        model.load_state_dict(state, strict=False)
        resume_phase = 2  # Phase 2(전체 fine-tune)부터 시작
        print(
            f"[finetune] {best_pth} 로드 완료 — Phase 2 추가학습 시작 "
            f"(initial best_val_acc={best_val_acc:.4f})"
        )

    # 체크포인트 로드 (--finetune 없을 때만)
    ckpt_path = save_dir / "checkpoint.pth"
    if not finetune and not fresh and ckpt_path.exists():
        ckpt = load_checkpoint(ckpt_path, device)
        resume_phase = ckpt["phase"]
        resume_epoch = ckpt["epoch"]
        best_val_acc = ckpt["best_val_acc"]
        state = ckpt["model_state"]
        current = model.state_dict()
        mismatched = {k for k, v in state.items() if k in current and v.shape != current[k].shape}
        if mismatched:
            print(f"[경고] 체크포인트 클래스 수 불일치 — head 레이어를 재초기화합니다: {mismatched}")
            state = {k: v for k, v in state.items() if k not in mismatched}
            resume_phase, resume_epoch, best_val_acc = 1, 0, 0.0
        model.load_state_dict(state, strict=False)
        print(
            f"체크포인트 로드: Phase {resume_phase}, "
            f"Epoch {resume_epoch}, best_val_acc={best_val_acc:.4f}"
        )
    elif fresh:
        print("처음부터 학습 — 기존 checkpoint.pth를 무시합니다.")
    elif not finetune:
        print("체크포인트 없음 — 처음부터 학습")

    # EMA 초기화 (체크포인트 로드 후, IPEX 최적화 전)
    ema = ModelEMA(model, decay=ema_decay) if ema_decay > 0 else None

    # ────────────────────────────────
    # Phase 1: Head만 학습
    # ────────────────────────────────
    if resume_phase == 1:
        freeze_backbone(model)
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1e-3,
            weight_decay=1e-4,
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=args.phase1_epochs)
        # Phase 1은 head만 학습하므로 IPEX 최적화 없이 진행.
        # ipex.optimize()는 Phase 2에서 전체 모델에 한 번만 호출한다.

        if ckpt is not None and resume_phase == 1 and resume_epoch > 0:
            patience_counter = ckpt["patience_counter"]
            restore_runtime_state(ckpt, optimizer, scheduler, scaler, "Phase 1")

        p1_start = resume_epoch + 1
        if p1_start <= args.phase1_epochs:
            print(f"\n{'─' * 40}")
            print(f"Phase 1 시작 (epoch {p1_start}~{args.phase1_epochs})")
            print(f"{'─' * 40}")

        for epoch in range(p1_start, args.phase1_epochs + 1):
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, device, scaler, use_amp,
                accumulation_steps=args.accumulation_steps,
                progress_callback=emit_progress_event,
                mixup_alpha=mixup_alpha,
                cutmix_alpha=cutmix_alpha,
                ema=ema,
            )
            val_model = ema.ema if ema is not None else model
            val_loss, val_acc = val_epoch(
                val_model, val_loader, criterion, device, use_amp, emit_progress_event
            )
            scheduler.step()

            print(
                f"  Epoch {epoch:2d}/{args.phase1_epochs} | "
                f"train_loss: {train_loss:.4f}  train_acc: {train_acc:.4f} | "
                f"val_loss: {val_loss:.4f}  val_acc: {val_acc:.4f}"
            )
            emit_metric_event(1, epoch, args.phase1_epochs, train_loss, train_acc, val_loss, val_acc)

            if use_wandb:
                wandb.log({
                    "phase": 1, "epoch": epoch,
                    "train/loss": train_loss, "train/acc": train_acc,
                    "val/loss": val_loss, "val/acc": val_acc,
                    "lr": scheduler.get_last_lr()[0],
                })

            # EMA 활성화 시 EMA 가중치를 best_model로 저장
            save_sd = (ema.ema if ema is not None else model).state_dict()
            if val_acc > best_val_acc or not best_model_path.exists():
                best_val_acc = val_acc
                patience_counter = 0
                torch.save(save_sd, best_model_path)
                print(f"  → best 저장 (val_acc: {val_acc:.4f})")
            else:
                patience_counter += 1
                if args.patience > 0 and patience_counter >= args.patience:
                    print(f"  Early stopping (patience={args.patience})")
                    break

            if epoch % args.save_interval == 0:
                save_checkpoint(
                    ckpt_path, 1, epoch, model, optimizer, scheduler, scaler,
                    best_val_acc, patience_counter,
                )
                print(f"  → periodic checkpoint 저장 (epoch {epoch})")
            elif epoch == args.phase1_epochs:
                save_checkpoint(
                    ckpt_path, 1, epoch, model, optimizer, scheduler, scaler,
                    best_val_acc, patience_counter,
                )

    # ────────────────────────────────
    # Phase 2: 전체 fine-tune
    unfreeze_all(model)

    optimizer = optim.AdamW(model.parameters(), lr=args.phase2_lr, weight_decay=1e-4)

    if _use_ipex:
        # 스케줄러 생성 전에 최적화: 스케줄러가 IPEX 래핑된 optimizer를 참조하도록.
        # 전체 파라미터가 언프리즈된 원본 모델에 한 번만 적용.
        model, optimizer = xpu_compat.try_ipex_optimize(
            model, optimizer, use_amp=use_amp,
        )

    # LR Warmup 설정: phase2_epochs에 따라 warmup 길이를 조정해 T_max >= 1 보장.
    warmup_epochs = min(2, max(0, args.phase2_epochs - 1))
    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=max(1, warmup_epochs)
    )
    main_scheduler = CosineAnnealingLR(
        optimizer, T_max=max(1, args.phase2_epochs - warmup_epochs)
    )

    # SequentialLR를 사용하여 Warmup 후 CosineAnnealing 적용
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_epochs],
    )

    p2_start = 1
    if resume_phase == 2 and ckpt is not None:
        # 중단된 Phase 2 체크포인트에서 재개
        patience_counter = ckpt["patience_counter"]
        restore_runtime_state(ckpt, optimizer, scheduler, scaler, "Phase 2")
        p2_start = resume_epoch + 1
    else:
        patience_counter = 0  # Phase 2 새로 시작 (--finetune 포함)

    if p2_start <= args.phase2_epochs:
        print(f"\n{'─' * 40}")
        print(
            f"Phase 2 시작 (epoch {p2_start}~{args.phase2_epochs}, patience={args.patience})"
        )
        print(f"{'─' * 40}")

    for epoch in range(p2_start, args.phase2_epochs + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler, use_amp,
            accumulation_steps=args.accumulation_steps,
            progress_callback=emit_progress_event,
            mixup_alpha=mixup_alpha,
            cutmix_alpha=cutmix_alpha,
            ema=ema,
        )
        val_model = ema.ema if ema is not None else model
        val_loss, val_acc = val_epoch(
            val_model, val_loader, criterion, device, use_amp, emit_progress_event
        )
        scheduler.step()

        print(
            f"  Epoch {epoch:2d}/{args.phase2_epochs} | "
            f"train_loss: {train_loss:.4f}  train_acc: {train_acc:.4f} | "
            f"val_loss: {val_loss:.4f}  val_acc: {val_acc:.4f}"
        )
        emit_metric_event(2, epoch, args.phase2_epochs, train_loss, train_acc, val_loss, val_acc)

        if use_wandb:
            wandb.log({
                "phase": 2, "epoch": args.phase1_epochs + epoch,
                "train/loss": train_loss, "train/acc": train_acc,
                "val/loss": val_loss, "val/acc": val_acc,
                "lr": scheduler.get_last_lr()[0],
            })

        save_sd = (ema.ema if ema is not None else model).state_dict()
        if val_acc > best_val_acc or not best_model_path.exists():
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(save_sd, best_model_path)
            print(f"  → best 저장 (val_acc: {val_acc:.4f})")
        else:
            patience_counter += 1
            if args.patience > 0 and patience_counter >= args.patience:
                print(f"  Early stopping (patience={args.patience})")
                break

        if epoch % args.save_interval == 0:
            save_checkpoint(
                ckpt_path, 2, epoch, model, optimizer, scheduler, scaler,
                best_val_acc, patience_counter,
            )
            print(f"  → periodic checkpoint 저장 (epoch {epoch})")
        elif epoch == args.phase2_epochs:
            save_checkpoint(
                ckpt_path, 2, epoch, model, optimizer, scheduler, scaler,
                best_val_acc, patience_counter,
            )

    # ────────────────────────────────
    # 최종 테스트
    # ────────────────────────────────
    print(f"\n{'─' * 40}")
    print("테스트 세트 평가")
    # IPEX 래핑 없는 깨끗한 모델로 평가
    test_model = build_model(num_classes, backbone).to(device)
    test_model.load_state_dict(
        torch.load(best_model_path, weights_only=True, map_location=device)
    )
    test_loss, test_acc, per_class_acc = test_epoch(
        test_model, test_loader, criterion, device, use_amp, num_classes
    )
    print(f"  test_loss: {test_loss:.4f}  test_acc: {test_acc:.4f}")

    # 정확도 낮은 클래스 출력 (디버깅용)
    low_acc_classes = sorted(
        [(train_ds.idx_to_class[i], acc) for i, acc in per_class_acc.items()],
        key=lambda x: x[1],
    )[:10]
    print("  [하위 10개 클래스]")
    for cls_name, acc in low_acc_classes:
        print(f"    {cls_name}: {acc:.4f}")
    print(f"{'─' * 40}\n")

    if use_wandb:
        wandb.log({"test/loss": test_loss, "test/acc": test_acc})
        wandb.finish()

    # 학습 설정 저장
    config = {
        "num_classes": num_classes,
        "img_size": args.img_size,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "backbone": backbone,
    }
    with open(save_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"학습 완료! 최고 val_acc: {best_val_acc:.4f}")
    print(f"모델 저장 위치: {save_dir}/best_model.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",  default="./dataset/raw")
    parser.add_argument("--save-dir",  default="./checkpoints")
    parser.add_argument("--backbone",  default=DEFAULT_BACKBONE, help="timm 백본 모델명")
    parser.add_argument("--img-size",  type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--phase1-epochs", type=int, default=5)
    parser.add_argument("--phase2-epochs", type=int, default=30)
    parser.add_argument("--phase2-lr",  type=float, default=1e-5)
    # ── 디바이스 옵션 ──────────────────────────────────
    parser.add_argument("--xpu",    action="store_true", help="Intel Arc GPU(XPU) 강제 사용 (IPEX 필요)")
    parser.add_argument("--cpu",    action="store_true", help="CPU 강제 사용")
    parser.add_argument("--device", default="", help="디바이스 직접 지정 (예: xpu, xpu:0, cuda:1)")
    parser.add_argument("--no-amp", action="store_true", help="AMP(mixed precision) 비활성화")
    # ── 학습 옵션 ──────────────────────────────────────
    parser.add_argument("--finetune", action="store_true",
                        help="best_model.pth 로드 후 Phase 2 추가학습 (새 데이터 추가 시)")
    parser.add_argument("--fresh",    action="store_true",
                        help="기존 checkpoint.pth를 무시하고 새 모델로 처음부터 학습합니다.")
    parser.add_argument("--save-interval", type=int, default=5, help="Save checkpoint every N epochs")
    parser.add_argument("--initial-best-val-acc", type=float, default=0.0,
                        help="Existing best val_acc to preserve when continuing from best_model.pth.")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience (0=비활성화)")
    # ── Augmentation ────────────────────────────────────
    parser.add_argument("--mixup-alpha",  type=float, default=0.0,
                        help="Mixup Beta 분포 α — 0=비활성화, 권장 0.4")
    parser.add_argument("--cutmix-alpha", type=float, default=0.0,
                        help="CutMix Beta 분포 α — 0=비활성화, 권장 1.0")
    # ── EMA ─────────────────────────────────────────────
    parser.add_argument("--ema-decay", type=float, default=0.0,
                        help="EMA 감쇠율 — 0=비활성화, 권장 0.9998")
    # ── WandB ─────────────────────────────────────────
    parser.add_argument("--wandb",         action="store_true", help="WandB 로깅 활성화")
    parser.add_argument("--wandb-project", default="holoscope")
    parser.add_argument("--wandb-run",     default=None, help="WandB 실행 이름 (미지정 시 자동)")
    parser.add_argument("--deep-validate-images", action="store_true",
                        help="학습 전 모든 이미지 헤더를 PIL로 검사합니다. 대형 데이터셋에서는 느릴 수 있습니다.")
    parser.add_argument("--face-crop-dir", default="",
                        help="얼굴 크롭 전처리된 데이터셋 경로. 지정 시 --data-dir 대신 사용.")
    parser.add_argument("--accumulation-steps", type=int, default=1,
                        help="Number of steps to accumulate gradients before updating")
    args = parser.parse_args()

    try:
        train(args)
    except DeviceUnavailableError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(2)
