import json
from pathlib import Path

import studio.doctor as doctor


class _FakeDevice:
    type = "xpu"


def _stable_environment(monkeypatch):
    monkeypatch.setattr(doctor, "_port_open", lambda port: False)
    monkeypatch.setattr(doctor, "_groups", lambda: {"render", "video", "1000"})
    monkeypatch.setattr(doctor, "_package_version", lambda package: "2.8.0+xpu" if package == "triton-xpu" else None)
    monkeypatch.setattr(doctor.xpu_compat, "best_device", lambda: _FakeDevice())
    monkeypatch.setattr(doctor.xpu_compat, "device_label", lambda device: "Intel Arc GPU (XPU) - test")
    monkeypatch.setattr(doctor.xpu_compat, "cuda_available", lambda: False)
    monkeypatch.setattr(doctor.xpu_compat, "mps_available", lambda: False)


def _project_root(tmp_path: Path) -> Path:
    (tmp_path / "characters.json").write_text('{"characters": []}', encoding="utf-8")
    (tmp_path / "dataset" / "raw").mkdir(parents=True)
    (tmp_path / "checkpoints").mkdir()
    return tmp_path


def test_doctor_reports_xpu_available(monkeypatch, tmp_path):
    _stable_environment(monkeypatch)
    monkeypatch.setattr(doctor.xpu_compat, "torch_version", lambda: "2.8.0+xpu")
    monkeypatch.setattr(
        doctor.xpu_compat,
        "xpu_status",
        lambda: {"build": True, "available": True, "reason": None},
    )
    monkeypatch.setattr(doctor.xpu_compat, "ipex_version", lambda: "2.8.10+xpu")

    result = doctor.run_doctor(_project_root(tmp_path))
    checks = {check["name"]: check for check in result["checks"]}

    assert result["device"]["selected"] == "xpu"
    assert checks["xpu"]["status"] == "ok"
    assert checks["ipex"]["status"] == "ok"


def test_doctor_flags_ipex_torch_mismatch(monkeypatch, tmp_path):
    _stable_environment(monkeypatch)
    monkeypatch.setattr(doctor.xpu_compat, "torch_version", lambda: "2.8.0+xpu")
    monkeypatch.setattr(
        doctor.xpu_compat,
        "xpu_status",
        lambda: {"build": True, "available": True, "reason": None},
    )
    monkeypatch.setattr(doctor.xpu_compat, "ipex_version", lambda: "2.7.0+xpu")

    result = doctor.run_doctor(_project_root(tmp_path))
    checks = {check["name"]: check for check in result["checks"]}

    assert result["summary"] == "error"
    assert checks["ipex"]["status"] == "error"
    assert "major.minor" in checks["ipex"]["suggestion"]


def test_doctor_json_is_machine_readable(monkeypatch, tmp_path):
    _stable_environment(monkeypatch)
    monkeypatch.setattr(doctor.xpu_compat, "torch_version", lambda: "2.8.0+xpu")
    monkeypatch.setattr(
        doctor.xpu_compat,
        "xpu_status",
        lambda: {"build": True, "available": True, "reason": None},
    )
    monkeypatch.setattr(doctor.xpu_compat, "ipex_version", lambda: "2.8.10+xpu")

    payload = json.loads(doctor.doctor_json(_project_root(tmp_path)))

    assert payload["device"]["selected"] == "xpu"
    assert payload["system"]["cpu_count"] >= 0
    assert "ram_total" in payload["system"]
    assert isinstance(payload["checks"], list)
    assert {"ok", "warn", "error"} >= {check["status"] for check in payload["checks"]}


def test_system_info_contains_os_cpu_and_ram():
    info = doctor.system_info()

    assert info["system"]
    assert "cpu_count" in info
    assert "ram_total" in info
    assert info["ram_total"] == "unknown" or info["ram_total"].endswith("GiB")
