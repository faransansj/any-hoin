"""CrawlJob — danbooru_crawler.py subprocess 래퍼 (범용 캐릭터 지원)."""

import asyncio
import json
import os
import sys
import tempfile
import time

from fastapi import WebSocket

from .base_job import BaseJob

CRAWL_EVENT_PREFIX = "__HOLOSCOPE_CRAWL_EVENT__ "


class CrawlJob(BaseJob):
    def __init__(self):
        super().__init__("crawl")
        self._tmp_file: str | None = None
        self.current_progress: dict | None = None
        self.health: dict | None = None
        self.last_event_at: float | None = None

    async def start(self, params: dict, project_root: str):
        if self.state == "running":
            return
        self.state = "running"
        self.current_progress = None
        self.health = None
        self.last_event_at = None

        # 이전 임시 파일 정리
        self._cleanup_tmp()

        cmd = [sys.executable, "crawling/danbooru_crawler.py", "--events"]

        if params.get("username"):
            cmd += ["-u", params["username"]]
        if params.get("api_key"):
            cmd += ["-k", params["api_key"]]

        cmd += ["--min-images", str(params.get("min_images", 500))]
        cmd += ["--max-images", str(params.get("max_images", 1000))]
        cmd += ["--workers",    str(params.get("workers", 4))]
        cmd += ["--output-dir", params.get("output_dir", "./dataset/raw")]

        if params.get("resize_large_images"):
            cmd.append("--resize-large-images")
            cmd += ["--resize-threshold-mb", str(params.get("resize_threshold_mb", 4))]
            cmd += ["--resize-max-side", str(params.get("resize_max_side", 1536))]
            cmd += ["--resize-quality", str(params.get("resize_quality", 88))]

        # {key: tag} 딕셔너리를 임시 JSON 파일로 전달
        tags_dict: dict[str, str] = params.get("tags_dict", {})
        if tags_dict:
            tmp = tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False, prefix="studio_tags_"
            )
            json.dump(tags_dict, tmp, ensure_ascii=False)
            tmp.close()
            self._tmp_file = tmp.name
            cmd += ["--tags-file", self._tmp_file]

        self._task = asyncio.create_task(self._run(cmd, cwd=project_root))

    async def _on_line(self, line: str):
        if CRAWL_EVENT_PREFIX not in line:
            return
        raw = line.split(CRAWL_EVENT_PREFIX, 1)[1].strip()
        if not raw:
            return
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            return

        self.last_event_at = time.time()
        event_type = event.get("event")
        if event_type == "progress":
            self.current_progress = event
            self.health = event.get("health") or self.health
            await self._broadcast({"type": "crawl_progress", "data": event})
        elif event_type in {"start", "complete"}:
            self.health = event.get("health") or self.health
            await self._broadcast({"type": "crawl_health", "data": self.health_snapshot()})

    async def _on_ws_connected(self, ws: WebSocket):
        if self.current_progress is not None:
            await self._send_ws(ws, {"type": "crawl_progress", "data": self.current_progress})
        if self.health is not None:
            await self._send_ws(ws, {"type": "crawl_health", "data": self.health_snapshot()})

    def _format_log_line(self, line: str) -> str | None:
        if CRAWL_EVENT_PREFIX not in line:
            return line
        visible = line.split(CRAWL_EVENT_PREFIX, 1)[0].strip()
        return visible or None

    def _last_event_age_sec(self) -> float | None:
        if self.last_event_at is None:
            return None
        return round(time.time() - self.last_event_at, 1)

    def health_snapshot(self) -> dict:
        age = self._last_event_age_sec()
        heartbeat_ok = self.state != "running" or (age is not None and age < 45)
        return {
            "state": self.state,
            "heartbeat_ok": heartbeat_ok,
            "last_event_age_sec": age,
            "crawler": self.health,
            "current_progress": self.current_progress,
        }

    def status(self) -> dict:
        return {
            **super().status(),
            "current_progress": self.current_progress,
            "health": self.health,
            "last_event_age_sec": self._last_event_age_sec(),
        }

    def _cleanup_tmp(self):
        if self._tmp_file and os.path.exists(self._tmp_file):
            try:
                os.unlink(self._tmp_file)
            except OSError:
                pass
            self._tmp_file = None

    async def stop(self):
        await super().stop()
        self._cleanup_tmp()
