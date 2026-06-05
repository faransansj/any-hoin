"""PreprocessJob — preprocess_dataset.py subprocess 래퍼."""

import asyncio
import json
import sys
import time

from .base_job import BaseJob

PREPROCESS_EVENT_PREFIX = "__HOLOSCOPE_PREPROCESS_EVENT__ "


class PreprocessJob(BaseJob):
    def __init__(self):
        super().__init__("preprocess")
        self.total: int = 0
        self.done: int = 0
        self.cached: int = 0
        self.skipped: int = 0
        self.errors: int = 0
        self.started_at: float | None = None
        self.finished_at: float | None = None

    async def start(self, params: dict, project_root: str):
        if self.state == "running":
            return
        self.state = "running"

        cmd = [sys.executable, "preprocess_dataset.py"]
        cmd += ["--data-dir",    params.get("data_dir",    "./dataset/raw")]
        cmd += ["--cache-dir",   params.get("cache_dir",   "./dataset/.cache")]
        cmd += ["--target-size", str(params.get("target_size", 320))]
        cmd += ["--quality",     str(params.get("quality",     95))]
        if params.get("workers"):
            cmd += ["--workers", str(params["workers"])]

        self.total = self.done = self.cached = self.skipped = self.errors = 0
        self.started_at = time.time()
        self.finished_at = None
        self._task = asyncio.create_task(self._run(cmd, cwd=project_root))

    async def _on_line(self, line: str):
        if PREPROCESS_EVENT_PREFIX not in line:
            return
        payload_str = line.split(PREPROCESS_EVENT_PREFIX, 1)[1]
        try:
            msg, _ = json.JSONDecoder().raw_decode(payload_str.strip())
        except json.JSONDecodeError:
            return

        event_type = msg.get("type")
        data = msg.get("data", {})

        if event_type == "start":
            self.total = int(data.get("total", 0))
        elif event_type in ("progress", "done"):
            self.done    = int(data.get("done",    self.done))
            self.total   = int(data.get("total",   self.total))
            self.cached  = int(data.get("cached",  self.cached))
            self.skipped = int(data.get("skipped", self.skipped))
            self.errors  = int(data.get("errors",  self.errors))
            await self._broadcast({"type": "progress", "data": self.status()})

    def _format_log_line(self, line: str) -> str | None:
        if PREPROCESS_EVENT_PREFIX in line:
            return None
        return line if line.strip() else None

    def status(self) -> dict:
        if self.state in ("done", "failed") and self.finished_at is None:
            self.finished_at = time.time()
        elapsed = None
        if self.started_at:
            end = self.finished_at if self.finished_at else time.time()
            elapsed = round(end - self.started_at, 1)
        pct = round(self.done / self.total * 100, 1) if self.total > 0 else 0.0
        return {
            **super().status(),
            "total":       self.total,
            "done":        self.done,
            "cached":      self.cached,
            "skipped":     self.skipped,
            "errors":      self.errors,
            "pct":         pct,
            "elapsed_sec": elapsed,
        }
