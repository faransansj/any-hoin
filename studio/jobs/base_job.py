"""
BaseJob — subprocess 실행 + WebSocket 로그 스트리밍 추상 기반 클래스
모든 Job(Crawl, Train, Export)이 이 클래스를 상속한다.
"""

import asyncio
import json
import re
from typing import Literal

from fastapi import WebSocket

JobState = Literal["idle", "running", "done", "failed"]
_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_WS_SEND_TIMEOUT_SEC = 0.5


def _clean_output_line(line: str) -> str:
    """ANSI 색상/캐리지 리턴을 제거해 웹 콘솔에서 진행바가 깨지지 않게 한다."""
    cleaned = _ANSI_RE.sub("", line.replace("\x00", ""))
    if "\r" in cleaned:
        cleaned = cleaned.split("\r")[-1]
    return cleaned.rstrip()


class BaseJob:
    def __init__(self, name: str):
        self.name = name
        self.state: JobState = "idle"
        self._proc: asyncio.subprocess.Process | None = None
        self._ws_clients: set[WebSocket] = set()
        self._log_buffer: list[str] = []  # 최근 500줄 보관
        self._task: asyncio.Task | None = None

    # ── WebSocket ────────────────────────────────────────────

    async def connect_ws(self, ws: WebSocket):
        """WebSocket 연결 수락 후 버퍼 재생 → 신규 메시지 수신까지 유지."""
        await ws.accept()
        self._ws_clients.add(ws)

        # 이미 쌓인 로그 재생
        for line in self._log_buffer[-500:]:
            if not await self._send_ws(ws, {"type": "log", "data": line}):
                return

        # 현재 상태 전송
        if not await self._send_ws(ws, {"type": "state", "data": self.state}):
            return
        await self._on_ws_connected(ws)

        # 연결 유지 (클라이언트가 끊을 때까지 대기)
        try:
            while True:
                await ws.receive_text()
        except Exception:
            pass
        finally:
            self._ws_clients.discard(ws)

    async def _broadcast(self, msg: dict):
        clients = list(self._ws_clients)
        if not clients:
            return

        payload = json.dumps(msg)
        await asyncio.gather(
            *(self._send_payload(ws, payload) for ws in clients),
            return_exceptions=True,
        )

    async def _send_ws(self, ws: WebSocket, msg: dict) -> bool:
        return await self._send_payload(ws, json.dumps(msg))

    async def _send_payload(self, ws: WebSocket, payload: str) -> bool:
        try:
            await asyncio.wait_for(ws.send_text(payload), timeout=_WS_SEND_TIMEOUT_SEC)
            return True
        except Exception:
            self._ws_clients.discard(ws)
            return False

    # ── 실행 ─────────────────────────────────────────────────

    async def _run(self, cmd: list[str], cwd: str = "."):
        """subprocess 실행 + stdout/stderr 스트리밍."""
        self.state = "running"
        self._log_buffer.clear()
        await self._broadcast({"type": "state", "data": "running"})

        self._proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=cwd,
        )

        if not self._proc.stdout:
            raise RuntimeError("Failed to open subprocess stdout")
        async for raw in self._proc.stdout:
            line = _clean_output_line(raw.decode("utf-8", errors="replace").rstrip("\n"))
            display_line = self._format_log_line(line)
            if display_line is not None:
                self._log_buffer.append(display_line)
                if len(self._log_buffer) > 500:
                    self._log_buffer.pop(0)
                await self._broadcast({"type": "log", "data": display_line})
            await self._on_line(line)

        await self._proc.wait()
        final_state: JobState = "done" if self._proc.returncode == 0 else "failed"
        self._proc = None
        self.state = final_state
        await self._broadcast({"type": "state", "data": final_state})

    async def _on_line(self, line: str):
        """서브클래스에서 오버라이드해 구조화 이벤트 파싱."""
        pass

    async def _on_ws_connected(self, ws: WebSocket):
        """서브클래스가 현재 진행 상태를 신규 WebSocket에 재전송할 때 사용."""
        pass

    def _format_log_line(self, line: str) -> str | None:
        """웹 콘솔에 표시할 로그 라인. None을 반환하면 숨긴다."""
        return line

    async def stop(self):
        if self._proc and self.state == "running":
            self._proc.terminate()
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self._proc.kill()
            self._proc = None
            self.state = "idle"
            await self._broadcast({"type": "state", "data": "idle"})
        if self._task and not self._task.done():
            self._task.cancel()

    def clear_buffer(self):
        self._log_buffer.clear()

    def status(self) -> dict:
        return {"name": self.name, "state": self.state}
