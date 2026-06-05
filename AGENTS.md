# Repository Guidelines

## Project Structure & Module Organization

Python entry points live at the repo root: `train.py`, `dataset.py`, `main.py`, `studio_api.py`, plus export/quantization scripts. FastAPI lives under `studio/`, with routes in `studio/routers/` and jobs in `studio/jobs/`.

React frontend is in `frontend/`; pages are in `frontend/src/pages/`, shared UI in `frontend/src/components/`, API helpers in `frontend/src/api.ts`, and state in `frontend/src/store/`. Tests are in `tests/`. Runtime artifacts go under `dataset/`, `checkpoints/`, `models/`, and `.thumbnails/`.

## Build, Test, and Development Commands

- `uv sync`: install Python 3.11 dependencies.
- `uv sync --extra cuda|rocm|arc`: install one GPU backend.
- `python scripts/sync_backend.py arc --apply --check`: restore/check Intel Arc.
- `python start.py [--web|--doctor]`: cross-platform launcher; shell wrappers call this.
- `.venv/bin/python studio_api.py`: run only the backend.
- `cd frontend && npm install && npm run dev`: run Vite locally.
- `cd frontend && npm run build`: type-check/build assets.
- `uv run pytest`: run tests.

## Coding Style & Naming Conventions

Use Python 3.11, 4-space indentation, helpful type hints, and snake_case. Keep routers focused in HTTP concerns; put long-running work in `studio/jobs/`. Character keys must stay stable across `characters.json`, `dataset/raw/<key>/`, labels, and inference.

Frontend code uses TypeScript, React function components, Tailwind utilities, and PascalCase page/component filenames.

## Intel Arc XPU Compatibility

Intel Arc XPU is supported and expected to work on the Arc B560/B-series setup. Do not move `torch`, `torchvision`, `intel-extension-for-pytorch`, `oneccl-bind-pt`, or `triton-xpu` into base dependencies; keep them behind the mutually exclusive `cuda`, `rocm`, and `arc` extras in `pyproject.toml`.

For any XPU-capable entry point, import `xpu_compat` before `torch`. This preserves triton masking, safe IPEX loading, torch/IPEX version checks, and graceful fallback behavior. Avoid CUDA-only assumptions; route device checks through `xpu_compat.device_available()` or existing helpers.

Validate Arc changes with `uv sync --extra arc` and run via `.venv/bin/python ...` or `uv run --extra arc ...`. Plain `uv run python ...` can re-sync to a non-XPU torch build. After dependency edits, verify:

```bash
.venv/bin/python -c "import xpu_compat, torch; print(torch.__version__, torch.xpu.is_available())"
```

## Testing Guidelines

Python tests use pytest and `tests/test_*.py` naming. Prefer `tmp_path` and `monkeypatch`; avoid GPU or service requirements.

## Commit & Pull Request Guidelines

Use concise imperative commit subjects; occasional `fix:` prefixes are present.

PRs should describe changes, list verification commands, link issues, and include UI screenshots when relevant. Note GPU backend, dataset, or checkpoint assumptions.

## Security & Configuration Tips

Copy `.env.example` to `.env` for credentials. Do not commit secrets, large datasets, checkpoints, thumbnails, or local Studio settings.

# Agent Guidelines

## Model Export Packages

When adding or changing model export behavior, preserve the deployment package contract:

- Every deployable model folder or ZIP must include `class_map.json`.
- Every deployable PyTorch model must include `config.json` so the inference loader can rebuild the correct backbone and class count.
- Every deployable ONNX model that references external data must include the referenced `best_model.onnx.data` file next to `best_model.onnx`.
- `class_map.json` is model-specific. Do not reuse a class map across models unless the output class count and class order are known to match.
- The expected deployment folder shape is:

```text
models/<model-name>/
  best_model.pth
  best_model_fp16.pth
  best_model.onnx
  best_model.onnx.data
  class_map.json
  config.json
```

Optional quantized files such as `best_model_int8.pth`, `best_model_int4.pth`, and `best_model_int2.pth` may be included when present.

The Studio Export page should expose a single deployment ZIP download that packages the available model files together with required metadata. Do not add model-only downloads as the primary deployment path, because model weights without `class_map.json` are not usable for inference.
