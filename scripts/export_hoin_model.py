#!/usr/bin/env python3
"""One-click export for hoin service/runtime use.

This script creates everything hoin needs:
- ONNX graph and external data, exported from checkpoints when needed
- hoin-model.json routing manifest
- class_map.json copy
- optional zip archive for publishing as a model repository/release asset
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.package_hoin_model import package_model, write_package_zip


DEFAULT_CHECKPOINT_DIR = Path("checkpoints")
DEFAULT_OUTPUT_DIR = Path("models/any-hoin")
DEFAULT_ZIP_PATH = Path("models/any-hoin-hoin-model.zip")
DEFAULT_MODEL_NAME = "any-hoin"


def _export_onnx(checkpoint_dir: Path, opset: int) -> None:
    from export_onnx import export

    export(SimpleNamespace(checkpoint_dir=str(checkpoint_dir), opset=opset))


def export_hoin_package(
    *,
    checkpoint_dir: Path,
    output_dir: Path,
    model_name: str,
    characters_path: Path,
    zip_path: Path | None,
    opset: int,
    force_onnx: bool,
) -> dict[str, Path]:
    """Export/package the current trained model for hoin."""

    checkpoint_dir = checkpoint_dir.resolve()
    onnx_path = checkpoint_dir / "best_model.onnx"
    if force_onnx or not onnx_path.exists():
        print(f"[hoin-export] exporting ONNX to {onnx_path} (opset={opset})")
        _export_onnx(checkpoint_dir, opset)
    elif not onnx_path.exists():
        raise FileNotFoundError(f"ONNX artifact not found: {onnx_path}")
    else:
        print(f"[hoin-export] using existing ONNX: {onnx_path}")

    outputs = package_model(
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir,
        model_name=model_name,
        characters_path=characters_path,
    )

    if zip_path is not None:
        outputs["zip"] = write_package_zip(output_dir, zip_path)

    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a hoin-compatible model package")
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--characters", type=Path, default=Path("characters.json"))
    parser.add_argument("--zip-path", type=Path, default=DEFAULT_ZIP_PATH)
    parser.add_argument("--no-zip", action="store_true")
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--force-onnx", action="store_true", help="re-export ONNX even if best_model.onnx exists")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = export_hoin_package(
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        characters_path=args.characters,
        zip_path=None if args.no_zip else args.zip_path,
        opset=args.opset,
        force_onnx=args.force_onnx,
    )
    print(f"hoin model package written: {outputs['output_dir']}")
    print(f"manifest: {outputs['manifest']}")
    print(f"onnx: {outputs['onnx']}")
    if "onnx_data" in outputs:
        print(f"onnx_data: {outputs['onnx_data']}")
    if "zip" in outputs:
        print(f"zip: {outputs['zip']}")


if __name__ == "__main__":
    main()
