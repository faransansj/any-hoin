"""TrainJob — train.py subprocess 래퍼 + 에포크 메트릭 / 배치 진행률 파싱."""

import asyncio
import json
import re
import sys
import time
from .base_job import BaseJob

# "  Epoch  5/30 | train_loss: 0.4321  train_acc: 0.8765 | val_loss: 0.4123  val_acc: 0.8823"
_METRIC_RE = re.compile(
    r"Epoch\s+(\d+)/(\d+)"
    r".*?train_loss:\s*([\d.]+)\s+train_acc:\s*([\d.]+)"
    r".*?val_loss:\s*([\d.]+)\s+val_acc:\s*([\d.]+)"
)

# "  train:  70%|███████   | 7/10 [00:04<00:01,  2.39it/s]"
# "  val:   50%|█████     | 1/2 [00:03<00:03,  3.15s/it]"
_PROGRESS_RE = re.compile(
    r"^\s*(train|val):\s+(\d+)%\|[^|]*\|\s+(\d+)/(\d+)"
    r"\s+\[[\d:]+<([\d:?]+),\s+([\d.?]+)(it/s|s/it)\]"
)
TRAIN_EVENT_PREFIX = "__HOLOSCOPE_TRAIN_EVENT__ "


def _parse_eta(s: str) -> float:
    """'mm:ss' 문자열을 초로 변환. '?' 이면 -1 반환."""
    if "?" in s:
        return -1.0
    parts = s.split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    return float(s)


class TrainJob(BaseJob):
    def __init__(self):
        super().__init__("train")
        self.metrics: list[dict] = []
        self.current_phase: int = 1
        self.best_val_acc: float = 0.0
        self.phase1_epochs: int = 5
        self.phase2_epochs: int = 30
        self.current_progress: dict | None = None
        self.started_at: float | None = None
        self.finished_at: float | None = None

    async def start(self, params: dict, project_root: str):
        if self.state == "running":
            return
        self.state = "running"

        self.phase1_epochs = int(params.get("phase1_epochs", 5))
        self.phase2_epochs = int(params.get("phase2_epochs", 30))

        cmd = [sys.executable, "train.py"]
        cmd += ["--data-dir",      params.get("data_dir", "./dataset/raw")]
        cmd += ["--save-dir",      params.get("save_dir", "./checkpoints")]
        cmd += ["--batch-size",    str(params.get("batch_size", 32))]
        cmd += ["--phase1-epochs", str(self.phase1_epochs)]
        cmd += ["--phase2-epochs", str(self.phase2_epochs)]
        cmd += ["--phase2-lr",     str(params.get("phase2_lr", 1e-5))]
        cmd += ["--patience",      str(params.get("patience", 7))]

        device = params.get("device", "")
        if device and device != "auto":
            cmd += ["--device", device]
        if params.get("finetune"):
            cmd.append("--finetune")
        if params.get("no_amp"):
            cmd.append("--no-amp")

        self.metrics = []
        self.current_phase = 1
        self.best_val_acc = 0.0
        self.current_progress = None
        self.started_at = time.time()
        self.finished_at = None
        self._task = asyncio.create_task(self._run(cmd, cwd=project_root))

    async def _on_line(self, line: str):
        if TRAIN_EVENT_PREFIX in line:
            await self._on_structured_event(line.split(TRAIN_EVENT_PREFIX, 1)[1])
            return

        # 페이즈 전환 감지
        if "Phase 2 시작" in line:
            self.current_phase = 2

        # 에포크 완료 메트릭
        m = _METRIC_RE.search(line)
        if m:
            epoch, total, tl, ta, vl, va = m.groups()
            metric = {
                "epoch":        int(epoch),
                "total_epochs": int(total),
                "phase":        self.current_phase,
                "train_loss":   float(tl),
                "train_acc":    float(ta),
                "val_loss":     float(vl),
                "val_acc":      float(va),
            }
            self._record_metric(metric)
            await self._broadcast({"type": "metric", "data": metric})
            return

        # tqdm 배치 진행률
        p = _PROGRESS_RE.search(line)
        if p:
            split, pct, cur, total, eta_str, speed_val, unit = p.groups()
            speed = float(speed_val) if speed_val != "?" else 0.0
            # s/it → it/s 로 통일
            speed_it_s = (1.0 / speed) if (unit == "s/it" and speed > 0) else speed
            progress = {
                "split":       split,
                "pct":         int(pct),
                "batch_cur":   int(cur),
                "batch_total": int(total),
                "eta_sec":     _parse_eta(eta_str),
                "speed_it_s":  round(speed_it_s, 2),
            }
            self._record_progress(progress)
            await self._broadcast({
                "type": "progress",
                "data": progress,
            })

    async def _on_structured_event(self, payload: str):
        try:
            msg, _ = json.JSONDecoder().raw_decode(payload.strip())
        except json.JSONDecodeError:
            return

        event_type = msg.get("type")
        data = msg.get("data")
        if not isinstance(data, dict):
            return

        if event_type == "metric":
            metric = {
                "epoch": int(data["epoch"]),
                "total_epochs": int(data["total_epochs"]),
                "phase": int(data["phase"]),
                "train_loss": float(data["train_loss"]),
                "train_acc": float(data["train_acc"]),
                "val_loss": float(data["val_loss"]),
                "val_acc": float(data["val_acc"]),
            }
            self._record_metric(metric)
            await self._broadcast({"type": "metric", "data": metric})
            return

        if event_type == "progress":
            progress = {
                "split": data["split"],
                "pct": int(data["pct"]),
                "batch_cur": int(data["batch_cur"]),
                "batch_total": int(data["batch_total"]),
                "eta_sec": float(data["eta_sec"]),
                "speed_it_s": float(data["speed_it_s"]),
            }
            if "avg_loss" in data:
                progress["avg_loss"] = float(data["avg_loss"])
            if "avg_acc" in data:
                progress["avg_acc"] = float(data["avg_acc"])
            self._record_progress(progress)
            await self._broadcast({"type": "progress", "data": progress})

    def _record_metric(self, metric: dict):
        self.current_phase = int(metric.get("phase", self.current_phase))
        key = (metric["phase"], metric["epoch"], metric["total_epochs"])
        for idx, existing in enumerate(self.metrics):
            existing_key = (
                existing["phase"],
                existing["epoch"],
                existing["total_epochs"],
            )
            if existing_key == key:
                self.metrics[idx] = metric
                break
        else:
            self.metrics.append(metric)

        self.best_val_acc = max(self.best_val_acc, float(metric["val_acc"]))

    def _record_progress(self, progress: dict):
        self.current_progress = progress

    def _format_log_line(self, line: str) -> str | None:
        if TRAIN_EVENT_PREFIX in line:
            line = line.split(TRAIN_EVENT_PREFIX, 1)[0].rstrip()
        if not line.strip():
            return None
        if _PROGRESS_RE.search(line):
            return None
        if line.strip() in {"warnings.warn(", "warnings.warn(str(msg))"}:
            return None
        return line

    def status(self) -> dict:
        if self.state in ("done", "failed") and self.finished_at is None:
            self.finished_at = time.time()

        elapsed_sec = None
        if self.started_at is not None:
            end = self.finished_at if self.finished_at is not None else time.time()
            elapsed_sec = round(end - self.started_at, 1)

        return {
            **super().status(),
            "current_phase":  self.current_phase,
            "best_val_acc":   round(self.best_val_acc, 4),
            "epoch_count":    len(self.metrics),
            "phase1_epochs":  self.phase1_epochs,
            "phase2_epochs":  self.phase2_epochs,
            "current_progress": self.current_progress,
            "last_metric": self.metrics[-1] if self.metrics else None,
            "elapsed_sec": elapsed_sec,
        }
