#!/usr/bin/env python3
"""Build a hoin-compatible runtime model package from any-hoin checkpoints.

The package contract consumed by hoin is:

    <output-dir>/
      hoin-model.json
      <model-name>.onnx
      class_map.json

`hoin-model.json` owns the class routing metadata. The ONNX output index must
match each class `index` exactly.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import zipfile
from pathlib import Path
from typing import Any


DEFAULT_CHECKPOINT_DIR = Path("checkpoints")
DEFAULT_OUTPUT_DIR = Path("models/any-hoin")
DEFAULT_MODEL_NAME = "any-hoin"


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _load_character_metadata(characters_path: Path) -> dict[str, dict[str, Any]]:
    if not characters_path.exists():
        return {}

    raw = _read_json(characters_path)
    if isinstance(raw, dict) and isinstance(raw.get("characters"), list):
        entries = raw["characters"]
    elif isinstance(raw, list):
        entries = raw
    else:
        raise ValueError(f"unsupported characters metadata format: {characters_path}")

    metadata: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        key = entry.get("key")
        if isinstance(key, str) and key:
            metadata[key] = entry
    return metadata


def _class_pairs_from_map(class_map: dict[str, Any]) -> list[tuple[int, str]]:
    if not class_map:
        raise ValueError("class_map.json is empty")

    if all(str(key).isdigit() for key in class_map):
        pairs = [(int(index), str(class_key)) for index, class_key in class_map.items()]
    else:
        pairs = [(int(index), str(class_key)) for class_key, index in class_map.items()]

    pairs.sort(key=lambda item: item[0])
    indexes = [index for index, _ in pairs]
    expected = list(range(len(pairs)))
    if indexes != expected:
        raise ValueError(
            "class_map indexes must be contiguous from 0 to N-1 for hoin; "
            f"got {indexes}, expected {expected}"
        )

    keys = [key for _, key in pairs]
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        raise ValueError(f"class_map contains duplicate class keys: {duplicates}")

    return pairs


def _localized_path_for_key(class_key: str, metadata: dict[str, Any]) -> dict[str, str]:
    display_name = metadata.get("display_name") or metadata.get("char_name") or class_key
    path = {"en": str(display_name)}

    ja_name = metadata.get("display_name_ja") or metadata.get("ja_name") or metadata.get("name_ja")
    if ja_name:
        path["ja"] = str(ja_name)

    return path


def build_manifest(
    *,
    model_name: str,
    onnx_name: str,
    class_map_path: Path,
    characters_path: Path,
) -> dict[str, Any]:
    """Return a hoin-model.json payload for the given class map."""

    if not class_map_path.exists():
        raise FileNotFoundError(f"class map not found: {class_map_path}")

    raw_class_map = _read_json(class_map_path)
    if not isinstance(raw_class_map, dict):
        raise ValueError(f"class map must be a JSON object: {class_map_path}")

    characters = _load_character_metadata(characters_path)
    classes = []
    for index, class_key in _class_pairs_from_map(raw_class_map):
        classes.append(
            {
                "index": index,
                "key": class_key,
                "path": _localized_path_for_key(class_key, characters.get(class_key, {})),
            }
        )

    return {
        "name": model_name,
        "onnx": onnx_name,
        "classes": classes,
    }


def package_model(
    *,
    checkpoint_dir: Path,
    output_dir: Path,
    model_name: str,
    characters_path: Path,
) -> dict[str, Path]:
    """Copy ONNX/class map artifacts and write hoin-model.json."""

    checkpoint_dir = checkpoint_dir.resolve()
    output_dir = output_dir.resolve()
    source_onnx = checkpoint_dir / "best_model.onnx"
    source_class_map = checkpoint_dir / "class_map.json"

    if not source_onnx.exists():
        raise FileNotFoundError(f"ONNX artifact not found: {source_onnx}")
    if not source_class_map.exists():
        raise FileNotFoundError(f"class map not found: {source_class_map}")

    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_name = f"{model_name}.onnx"
    target_onnx = output_dir / onnx_name
    target_class_map = output_dir / "class_map.json"
    target_manifest = output_dir / "hoin-model.json"

    shutil.copy2(source_onnx, target_onnx)
    shutil.copy2(source_class_map, target_class_map)

    external_data = source_onnx.with_suffix(source_onnx.suffix + ".data")
    target_external_data: Path | None = None
    if external_data.exists():
        # ONNX external-data filenames are embedded inside the .onnx protobuf.
        # Preserve the source filename unless/until we explicitly rewrite the ONNX
        # external-data metadata; renaming to <model-name>.onnx.data breaks ORT.
        target_external_data = output_dir / external_data.name
        shutil.copy2(external_data, target_external_data)

    manifest = build_manifest(
        model_name=model_name,
        onnx_name=onnx_name,
        class_map_path=source_class_map,
        characters_path=characters_path,
    )
    if target_external_data is not None:
        manifest["onnx_data"] = target_external_data.name
    _write_json(target_manifest, manifest)

    return {
        "output_dir": output_dir,
        "onnx": target_onnx,
        "class_map": target_class_map,
        "manifest": target_manifest,
        **({"onnx_data": target_external_data} if target_external_data is not None else {}),
    }


def _safe_archive_name(name: str) -> str:
    path = Path(name)
    if (
        not name
        or path.is_absolute()
        or ".." in path.parts
        or "/" in name
        or "\\" in name
        or os.sep in name
        or (os.altsep is not None and os.altsep in name)
    ):
        raise ValueError(f"unsafe hoin package filename: {name!r}")
    return name


def write_package_zip(output_dir: Path, zip_path: Path) -> Path:
    """Create a zip archive containing the hoin package files at the archive root."""

    output_dir = output_dir.resolve()
    zip_path = zip_path.resolve()
    if not output_dir.exists():
        raise FileNotFoundError(f"hoin package directory not found: {output_dir}")

    required = ["hoin-model.json"]
    manifest_path = output_dir / "hoin-model.json"
    manifest = _read_json(manifest_path)
    if isinstance(manifest, dict):
        if manifest.get("onnx"):
            required.append(_safe_archive_name(str(manifest["onnx"])))
        if manifest.get("onnx_data"):
            required.append(_safe_archive_name(str(manifest["onnx_data"])))
    required.append("class_map.json")

    for name in required:
        if not (output_dir / name).is_file():
            raise FileNotFoundError(f"required hoin package file is missing: {output_dir / name}")

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in dict.fromkeys(required):
            zf.write(output_dir / name, arcname=name)
    return zip_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a hoin-compatible model package")
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--characters", type=Path, default=Path("characters.json"))
    parser.add_argument("--zip-path", type=Path, default=None, help="optional zip archive path to create")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = package_model(
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        characters_path=args.characters,
    )
    if args.zip_path is not None:
        zip_path = write_package_zip(args.output_dir, args.zip_path)
        outputs["zip"] = zip_path
    print(f"hoin model package written: {outputs['output_dir']}")
    print(f"manifest: {outputs['manifest']}")
    print(f"onnx: {outputs['onnx']}")
    if "onnx_data" in outputs:
        print(f"onnx_data: {outputs['onnx_data']}")
    if "zip" in outputs:
        print(f"zip: {outputs['zip']}")


if __name__ == "__main__":
    main()
