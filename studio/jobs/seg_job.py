"""SegJob — segmentation/run_crop.py subprocess 래퍼."""

import asyncio
import json
import sys
import time

from .base_job import BaseJob

SEG_EVENT_PREFIX = "__HOLOSCOPE_SEG_EVENT__ "


class SegJob(BaseJob):
    def __init__(self):
        super().__init__("seg")
        self.total_images: int = 0
        self.processed: int = 0
        self.current_class: str = ""
        self.class_idx: int = 0
        self.total_classes: int = 0
        self.pct: float = 0.0
        self.eta_sec: int = -1
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self.output_dir: str = ""

    async def start(self, params: dict, project_root: str) -> None:
        if self.state == "running":
            return
        self.state = "running"

        input_dir  = params.get("input_dir",   "./dataset/raw")
        output_dir = params.get("output_dir",  "./dataset/raw_seg")
        backend    = params.get("backend",     "cascade")
        output_sz  = int(params.get("output_size", 224))
        padding    = float(params.get("padding",   0.3))
        min_face   = int(params.get("min_face",   48))
        sam_model  = params.get("sam_model",  "")
        sam_ckpt   = params.get("sam_checkpoint", "")
        sam_device = params.get("sam_device", "cpu")

        self.output_dir = output_dir
        self.total_images = 0
        self.processed = 0
        self.current_class = ""
        self.pct = 0.0
        self.eta_sec = -1
        self.started_at = time.time()
        self.finished_at = None

        cmd = [sys.executable, "segmentation/run_crop.py",
               "--input-dir",   input_dir,
               "--output-dir",  output_dir,
               "--backend",     backend,
               "--output-size", str(output_sz),
               "--padding",     str(padding),
               "--min-face",    str(min_face),
               "--events"]

        if sam_model:
            cmd += ["--sam-model", sam_model]
        if sam_ckpt:
            cmd += ["--sam-checkpoint", sam_ckpt]
        if sam_device:
            cmd += ["--sam-device", sam_device]

        self._task = asyncio.create_task(self._run(cmd, cwd=project_root))

    async def _on_line(self, line: str) -> None:
        if SEG_EVENT_PREFIX not in line:
            return
        payload = line.split(SEG_EVENT_PREFIX, 1)[1]
        try:
            msg, _ = json.JSONDecoder().raw_decode(payload.strip())
        except json.JSONDecodeError:
            return

        event = msg.get("event", "")

        if event == "start":
            self.total_images = int(msg.get("total_images", 0))
            self.total_classes = int(msg.get("total_classes", 0))
            await self._broadcast({"type": "seg_start", "data": {
                "total_images": self.total_images,
                "total_classes": self.total_classes,
            }})

        elif event == "class_start":
            self.current_class = msg.get("class", "")
            self.class_idx = int(msg.get("class_idx", 0))
            await self._broadcast({"type": "seg_class", "data": {
                "class": self.current_class,
                "class_idx": self.class_idx,
                "total_classes": self.total_classes,
            }})

        elif event == "progress":
            self.processed  = int(msg.get("global_processed", 0))
            self.pct        = float(msg.get("pct", 0))
            self.eta_sec    = int(msg.get("eta_sec", -1))
            self.current_class = msg.get("class", self.current_class)
            await self._broadcast({"type": "seg_progress", "data": {
                "class": self.current_class,
                "class_idx": self.class_idx,
                "class_processed": int(msg.get("class_processed", 0)),
                "class_total":     int(msg.get("class_total", 0)),
                "global_processed": self.processed,
                "total_images": self.total_images,
                "pct": self.pct,
                "eta_sec": self.eta_sec,
            }})

        elif event == "class_done":
            await self._broadcast({"type": "seg_class_done", "data": {
                "class":       msg.get("class", ""),
                "multi_face":  int(msg.get("multi_face", 0)),
                "single_face": int(msg.get("single_face", 0)),
                "no_face":     int(msg.get("no_face", 0)),
                "saved_crops": int(msg.get("saved_crops", 0)),
            }})

        elif event == "complete":
            self.pct = 100.0
            await self._broadcast({"type": "seg_complete", "data": {
                "total_images": int(msg.get("total_images", 0)),
                "output_dir":   msg.get("output_dir", ""),
                "elapsed_sec":  float(msg.get("elapsed_sec", 0)),
            }})

    def status(self) -> dict:
        if self.state in ("done", "failed") and self.finished_at is None:
            self.finished_at = time.time()
        elapsed = None
        if self.started_at is not None:
            end = self.finished_at if self.finished_at is not None else time.time()
            elapsed = round(end - self.started_at, 1)
        return {
            **super().status(),
            "total_images":  self.total_images,
            "processed":     self.processed,
            "current_class": self.current_class,
            "class_idx":     self.class_idx,
            "total_classes": self.total_classes,
            "pct":           round(self.pct, 1),
            "eta_sec":       self.eta_sec,
            "output_dir":    self.output_dir,
            "elapsed_sec":   elapsed,
        }
