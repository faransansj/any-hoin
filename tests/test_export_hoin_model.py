import asyncio
import json
from pathlib import Path

from fastapi import HTTPException

from scripts.export_hoin_model import export_hoin_package
from studio.routers.export import export_hoin


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_export_hoin_route_rejects_unsafe_model_name():
    try:
        asyncio.run(export_hoin({"model_name": "../bad"}))
    except HTTPException as exc:
        assert exc.status_code == 400
    else:
        raise AssertionError("expected unsafe model_name to fail")


def test_export_hoin_package_uses_existing_onnx_and_writes_zip(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    output_dir = tmp_path / "models" / "any-hoin"
    zip_path = tmp_path / "models" / "any-hoin-hoin-model.zip"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "best_model.onnx").write_bytes(b"onnx")
    (checkpoint_dir / "best_model.onnx.data").write_bytes(b"external")
    write_json(checkpoint_dir / "class_map.json", {"0": "class_a"})

    result = export_hoin_package(
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir,
        model_name="any-hoin",
        characters_path=tmp_path / "characters.json",
        zip_path=zip_path,
        opset=18,
        force_onnx=False,
    )

    assert result["manifest"] == output_dir / "hoin-model.json"
    assert result["zip"] == zip_path
    assert zip_path.exists()
    assert (output_dir / "any-hoin.onnx").exists()
    assert (output_dir / "best_model.onnx.data").exists()


def test_export_hoin_package_requires_checkpoint_when_onnx_missing(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    write_json(checkpoint_dir / "class_map.json", {"0": "class_a"})

    try:
        export_hoin_package(
            checkpoint_dir=checkpoint_dir,
            output_dir=tmp_path / "models" / "any-hoin",
            model_name="any-hoin",
            characters_path=tmp_path / "characters.json",
            zip_path=tmp_path / "models" / "any-hoin-hoin-model.zip",
            opset=18,
            force_onnx=False,
        )
    except FileNotFoundError as exc:
        assert "best_model.pth" in str(exc)
    else:
        raise AssertionError("expected missing checkpoint to fail")
