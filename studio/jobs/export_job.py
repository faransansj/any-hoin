"""ExportJob — quantize.py / export_onnx.py subprocess 래퍼."""

import asyncio
import json
import re
import sys
from .base_job import BaseJob

QUANT_FORMATS = ("fp16", "int8", "int4", "int2")

_RESULT_RE = re.compile(r"^EXPORT_RESULT:(.+)$")


class QuantJob(BaseJob):
    def __init__(self):
        super().__init__("quant")
        self.metrics: dict = {}

    async def start(self, fmt: str, project_root: str):
        if fmt not in QUANT_FORMATS:
            raise ValueError(f"Unknown format: {fmt}")
        if self.state == "running":
            return
        self.metrics = {}
        self.state = "running"
        cmd = [sys.executable, "quantize.py", "--format", fmt]
        self._task = asyncio.create_task(self._run(cmd, cwd=project_root))

    async def _on_line(self, line: str):
        m = _RESULT_RE.match(line)
        if m:
            try:
                self.metrics = json.loads(m.group(1))
            except json.JSONDecodeError:
                pass


class OnnxJob(BaseJob):
    def __init__(self):
        super().__init__("onnx")

    async def start(self, opset: int, project_root: str):
        if self.state == "running":
            return
        self.state = "running"
        cmd = [sys.executable, "export_onnx.py", "--opset", str(opset)]
        self._task = asyncio.create_task(self._run(cmd, cwd=project_root))


class HoinExportJob(BaseJob):
    def __init__(self):
        super().__init__("hoin")

    async def start(
        self,
        opset: int,
        project_root: str,
        model_name: str = "any-hoin",
        force_onnx: bool = False,
    ):
        if self.state == "running":
            return
        self.state = "running"
        cmd = [
            sys.executable,
            "scripts/export_hoin_model.py",
            "--opset",
            str(opset),
            "--checkpoint-dir",
            "./checkpoints",
            "--output-dir",
            f"./models/{model_name}",
            "--model-name",
            model_name,
            "--characters",
            "./characters.json",
            "--zip-path",
            f"./models/{model_name}-hoin-model.zip",
        ]
        if force_onnx:
            cmd.append("--force-onnx")
        self._task = asyncio.create_task(self._run(cmd, cwd=project_root))
