# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HoloScope Studio — Swin Transformer-Tiny 기반 홀로라이브 캐릭터 분류기. 학술/비상업 데모 프로젝트.  
End-to-end ML studio: **Crawl → Label → Train → Export → Inference**, all controllable via a React web UI.

## Setup

```bash
uv sync                     # Python 3.11 venv + dependencies
uv sync --extra cuda        # NVIDIA GPU (PyPI CUDA build)
uv sync --extra rocm        # AMD GPU (ROCm build)
uv sync --extra arc         # Intel Arc XPU + IPEX
uv sync --extra logging     # + wandb
```

GPU extras are mutually exclusive (enforced by `[tool.uv.conflicts]`).

Auto-detect GPU and sync the right profile:
```bash
uv run python scripts/sync_backend.py --apply
```

Optional: copy `.env.example` → `.env` and set `DANBOORU_LOGIN` / `DANBOORU_API_KEY`.

## Running the Stack

**Recommended (both services, port management, health check):**
```bash
bash start.sh
```

**Manual — Backend** (port 8001):
```bash
.venv/bin/python studio_api.py   # prefer over 'uv run' to avoid dep re-resolution
```

**Manual — Frontend** (port 5173):
```bash
cd frontend && npm install && npm run dev
```

After `npm run build`, the backend serves `frontend/dist/` at `/` automatically.

**Legacy inference-only server** (port 8000):
```bash
uv run python main.py
```

## Testing

```bash
uv run pytest                          # all tests
uv run pytest tests/test_core.py       # single file
uv run pytest -k test_holo_dataset_split  # single test by name
```

Tests use `tmp_path` fixtures and `monkeypatch` for filesystem/module isolation; no external services or GPU required.

## CLI Commands

**Crawl** (standalone):
```bash
uv run python crawling/danbooru_crawler.py \
  -u USER -k KEY --min-images 500 --max-images 1000 \
  --workers 4 --output-dir ./dataset/raw
```

**Train** (standalone):
```bash
uv run python train.py \
  --data-dir ./dataset/raw --save-dir ./checkpoints \
  --batch-size 32 --phase1-epochs 5 --phase2-epochs 30 --phase2-lr 1e-5
```
Training modes: `--fresh` (scratch), `--finetune` (load weights, reset head), default = resume.

**Full release pipeline**:
```bash
bash pipeline/run_pipeline.sh --version v2.0.0 [--skip-crawl] [--skip-train] [--release]
```

## Architecture

### Backend: `studio_api.py`

FastAPI app on port 8001. Mounts all routers under `/api` and serves the built frontend from `frontend/dist/`.  
Routers: `characters`, `crawl`, `labels`, `images`, `training`, `export`, `inference`.

### Job System: `studio/jobs/`

All long-running operations (crawl, train, export) share the same pattern:

- **`BaseJob`** — spawns a subprocess, streams `stdout+stderr` line-by-line, broadcasts to WebSocket clients, buffers the last 500 log lines for reconnecting clients. State machine: `idle → running → done/failed`.
- **`CrawlJob`** — wraps `crawling/danbooru_crawler.py --events`. Passes a `--tags-file` temp JSON for character tag overrides. Parses `CRAWL_EVENT_PREFIX`-prefixed lines for structured progress/health events.
- **`TrainJob`** — wraps `train.py`. Parses `TRAIN_EVENT_PREFIX`-prefixed JSON events (epoch metrics and tqdm batch progress) from stdout; broadcasts structured `{type: "metric"|"progress", data: ...}` events alongside plain log lines.
- **`QuantJob` / `OnnxJob`** — wrap `quantize.py --format <fp16|int8|int4|int2>` / `export_onnx.py`.

Each router (e.g. `studio/routers/training.py`) holds a **module-level singleton job instance** and exposes `GET /status`, `POST /start`, `POST /stop`, `GET /metrics`, and `WS /logs`.

### Character Registry: `studio/characters.py` + `characters.json`

`characters.json` is the source of truth for the character list:
```json
{"characters": [{"key": "folder_name", "tag": "danbooru_tag", "display_name": "Display Name"}]}
```
`key` = `dataset/raw/<key>/` folder name = training class label = inference output key.  
`studio/characters.py` exposes `load() → dict[key, meta]` and `save()`.

### Training: `train.py` + `dataset.py`

Two-phase fine-tuning of `swin_tiny_patch4_window7_224` (~28M params, ImageNet-1K pretrained):
- **Phase 1** (default 5 epochs): frozen backbone, head only, lr=1e-3.
- **Phase 2** (default 30 epochs): full unfreeze, CosineAnnealingLR, lr=1e-5.
- AMP + label smoothing 0.1 + early stopping (`--patience`).
- Intel Arc XPU support is centralized in `xpu_compat.py`; it masks incompatible `triton-xpu`, checks IPEX major.minor before import, and falls back when IPEX optimization is unavailable.
- Outputs: `checkpoints/best_model.pth`, `class_map.json` (`{idx: char_key}`), `config.json`.

`dataset.py`: `HoloDataset` does 80/10/10 reproducible split by seed; `build_dataloaders()` uses `WeightedRandomSampler` for class imbalance. Augmentation uses geometric transforms with minimal hue shift (±5°) to preserve character-distinguishing colors.

### Frontend: `frontend/`

React + TypeScript + Vite + Tailwind + Recharts + Zustand.

- **`src/api.ts`** — typed fetch wrapper (`api.get/post/...`) and `api.ws(path)` for WebSocket construction. All requests go to `/api`.
- **`src/store/jobStore.ts`** — global Zustand store for job states (`crawlState`, `trainState`, `fp16State`, `onnxState`) and training metrics. Pages connect via WebSocket (`ws://localhost:8001/api/<job>/logs`) and push parsed events into this store.
- **Pages**: `Crawl`, `Dataset`, `Training`, `Export`, `Inference` — each corresponds to one pipeline stage.
- **`JobConsole`** component renders scrolling log output from the WebSocket stream.

### Model Export & Inference

- `quantize.py --format <fp16|int8|int4|int2>` — produces quantized checkpoints.
- `export_onnx.py` — exports `best_model.pth` → `best_model.onnx`.
- `studio/routers/inference.py` — lazy-loads `ModelLoader` from `main.py`; prefers ONNX session if available, falls back to PyTorch.

## Key Generated Files (not committed)

| Path | Description |
|------|-------------|
| `dataset/raw/<key>/` | Training images per character |
| `dataset/raw/others/` | Characters below `--min-images` threshold |
| `characters.json` | Character registry (edit via UI or directly) |
| `checkpoints/best_model.pth` | Best val_acc weights |
| `checkpoints/class_map.json` | `{idx: char_key}` for inference |
| `checkpoints/config.json` | Training config + final metrics |
| `checkpoints/best_model.onnx` | ONNX export |
