from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from PIL import Image


class TinyClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.head = nn.Linear(8, num_classes)

    def forward(self, x):
        return self.head(self.features(x))


def _make_dataset(root):
    colors = [(220, 40, 40), (40, 220, 40)]
    for class_idx, color in enumerate(colors):
        class_dir = root / f"char_{class_idx}"
        class_dir.mkdir(parents=True)
        for i in range(12):
            img = Image.new("RGB", (48, 48), color=color)
            img.save(class_dir / f"{i}.jpg")


def test_training_smoke_runs_one_epoch_per_phase(tmp_path, monkeypatch, capsys):
    train_mod = pytest.importorskip("train")

    data_dir = tmp_path / "data"
    save_dir = tmp_path / "checkpoints"
    _make_dataset(data_dir)

    monkeypatch.setattr(
        train_mod,
        "build_model",
        lambda num_classes, pretrained=True: TinyClassifier(num_classes),
    )

    args = SimpleNamespace(
        data_dir=str(data_dir),
        save_dir=str(save_dir),
        img_size=32,
        batch_size=4,
        num_workers=0,
        phase1_epochs=1,
        phase2_epochs=1,
        phase2_lr=1e-3,
        xpu=False,
        cpu=True,
        device="",
        no_amp=True,
        finetune=False,
        save_interval=99,
        patience=0,
        wandb=False,
        wandb_project="holoscope-test",
        wandb_run=None,
        accumulation_steps=1,
    )

    train_mod.train(args)

    captured = capsys.readouterr()
    assert train_mod.TRAIN_EVENT_PREFIX in captured.out
    assert (save_dir / "best_model.pth").exists()
    assert (save_dir / "checkpoint.pth").exists()
    assert (save_dir / "class_map.json").exists()
    assert (save_dir / "config.json").exists()


def test_detect_device_rejects_explicit_xpu_without_xpu_build(monkeypatch):
    train_mod = pytest.importorskip("train")

    monkeypatch.setattr(train_mod.torch, "__version__", "2.11.0+cu130")

    with pytest.raises(RuntimeError, match="XPU 빌드가 아닙니다"):
        train_mod.detect_device(device_str="xpu")
