import json
from pathlib import Path

import pytest

from scripts.package_hoin_model import build_manifest, package_model, write_package_zip


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_build_manifest_from_numeric_index_class_map(tmp_path):
    class_map_path = tmp_path / "class_map.json"
    characters_path = tmp_path / "characters.json"
    write_json(class_map_path, {"0": "class_a", "1": "class_b"})
    write_json(
        characters_path,
        {
            "characters": [
                {"key": "class_a", "display_name": "Class A"},
                {"key": "class_b", "display_name": "Class B", "display_name_ja": "クラスB"},
            ]
        },
    )

    manifest = build_manifest(
        model_name="any-hoin",
        onnx_name="any-hoin.onnx",
        class_map_path=class_map_path,
        characters_path=characters_path,
    )

    assert manifest == {
        "name": "any-hoin",
        "onnx": "any-hoin.onnx",
        "classes": [
            {"index": 0, "key": "class_a", "path": {"en": "Class A"}},
            {"index": 1, "key": "class_b", "path": {"en": "Class B", "ja": "クラスB"}},
        ],
    }


def test_build_manifest_from_key_to_index_class_map(tmp_path):
    class_map_path = tmp_path / "class_map.json"
    write_json(class_map_path, {"class_b": 1, "class_a": 0})

    manifest = build_manifest(
        model_name="any-hoin",
        onnx_name="any-hoin.onnx",
        class_map_path=class_map_path,
        characters_path=tmp_path / "missing_characters.json",
    )

    assert manifest["classes"] == [
        {"index": 0, "key": "class_a", "path": {"en": "class_a"}},
        {"index": 1, "key": "class_b", "path": {"en": "class_b"}},
    ]


def test_build_manifest_rejects_non_contiguous_indexes(tmp_path):
    class_map_path = tmp_path / "class_map.json"
    write_json(class_map_path, {"0": "class_a", "2": "class_c"})

    with pytest.raises(ValueError, match="contiguous"):
        build_manifest(
            model_name="any-hoin",
            onnx_name="any-hoin.onnx",
            class_map_path=class_map_path,
            characters_path=tmp_path / "characters.json",
        )


def test_package_model_writes_hoin_model_package(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    output_dir = tmp_path / "models" / "any-hoin"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "best_model.onnx").write_bytes(b"onnx")
    write_json(checkpoint_dir / "class_map.json", {"0": "class_a"})

    package_model(
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir,
        model_name="any-hoin",
        characters_path=tmp_path / "characters.json",
    )

    assert (output_dir / "any-hoin.onnx").read_bytes() == b"onnx"
    assert json.loads((output_dir / "class_map.json").read_text(encoding="utf-8")) == {"0": "class_a"}
    assert json.loads((output_dir / "hoin-model.json").read_text(encoding="utf-8")) == {
        "name": "any-hoin",
        "onnx": "any-hoin.onnx",
        "classes": [{"index": 0, "key": "class_a", "path": {"en": "class_a"}}],
    }


def test_package_model_preserves_external_data_filename(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    output_dir = tmp_path / "models" / "any-hoin"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "best_model.onnx").write_bytes(b"onnx")
    (checkpoint_dir / "best_model.onnx.data").write_bytes(b"external")
    write_json(checkpoint_dir / "class_map.json", {"0": "class_a"})

    package_model(
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir,
        model_name="any-hoin",
        characters_path=tmp_path / "characters.json",
    )

    manifest = json.loads((output_dir / "hoin-model.json").read_text(encoding="utf-8"))
    assert manifest["onnx_data"] == "best_model.onnx.data"
    assert (output_dir / "best_model.onnx.data").read_bytes() == b"external"
    assert not (output_dir / "any-hoin.onnx.data").exists()


def test_write_package_zip_rejects_manifest_path_traversal(tmp_path):
    output_dir = tmp_path / "package"
    output_dir.mkdir()
    write_json(
        output_dir / "hoin-model.json",
        {"name": "bad", "onnx": "../outside.onnx", "classes": []},
    )
    (output_dir / "class_map.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="unsafe"):
        write_package_zip(output_dir, tmp_path / "bad.zip")


def test_package_model_rejects_missing_onnx(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    write_json(checkpoint_dir / "class_map.json", {"0": "class_a"})

    with pytest.raises(FileNotFoundError, match="best_model.onnx"):
        package_model(
            checkpoint_dir=checkpoint_dir,
            output_dir=tmp_path / "models" / "any-hoin",
            model_name="any-hoin",
            characters_path=tmp_path / "characters.json",
        )
