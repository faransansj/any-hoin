# 🌟 any-hoin — Integrated Hololive Character Classifier

[한국어](docs/README.kr.md) | [日本語](docs/README.ja.md) | [中文](docs/README.zh.md)

any-hoin is an end-to-end ML studio based on Swin Transformer-Tiny for Hololive character classification.  
**Crawl → Label → Train → Export → Inference** — all controlled from a single React web UI.

---

## ⚡ Quick Start

### Prerequisites

| Tool | Mac/Linux | Windows |
|------|-----------|---------|
| Git | `brew install git` / package manager | [git-scm.com](https://git-scm.com/download/win) |
| Node.js 18+ | `brew install node` / `nvm install --lts` | [nodejs.org](https://nodejs.org/) |
| uv | see below | see below |

> **Windows users**: Native PowerShell is supported for CPU/CUDA. For ROCm or Intel Arc (XPU), use **WSL2** — those PyTorch builds are Linux-only.

---

### Apple Silicon (M1 / M2 / M3 / M4) — One Shot

MPS (Metal Performance Shaders) acceleration is built into the standard PyTorch package — **no extra flags needed**.

```bash
# 1. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env   # or restart terminal

# 2. Clone and enter the repo
git clone https://github.com/faransansj/any-hoin.git
cd any-hoin

# 3. Install Python dependencies (MPS included in default torch build)
uv sync

# 4. One-shot start: backend + frontend
./start.sh
```

`start.sh` opens:
- Backend API → `http://localhost:8001`
- Frontend UI → `http://localhost:5173`

Press `Ctrl+C` to stop both at once.

> **MPS notes**:
> - The trainer auto-detects MPS — no `--device` flag required.
> - AMP (mixed precision) is not supported on MPS; training runs at FP32, which is still substantially faster than CPU.
> - If you see `mps` in the training log header, acceleration is active.

---

### Mac / Linux — One Shot

```bash
# 1. Install uv (Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env   # or restart terminal

# 2. Clone and enter the repo
git clone https://github.com/faransansj/any-hoin.git
cd any-hoin

# 3. Install Python dependencies (CPU / inference only)
uv sync

# 3a. (Optional) Unified backend setup
python scripts/sync_backend.py auto --apply --check

# Or explicitly pick ONE GPU backend — PyTorch accelerator wheels are mutually exclusive
uv sync --extra cuda   # NVIDIA GPU
uv sync --extra rocm   # AMD GPU  (Linux only)
uv sync --extra arc    # Intel Arc XPU (Linux only)

# 4. One-shot start: backend + frontend
./start.sh
```

`start.sh` opens:
- Backend API → `http://localhost:8001`
- Frontend UI → `http://localhost:5173`

Press `Ctrl+C` to stop both at once.

> **Unified device workflow**: In Studio, leave Training device as `auto`. Install/sync the backend once with `scripts/sync_backend.py`, then start with `./start.sh`. Avoid plain `uv run python` after selecting Arc/ROCm/CUDA because uv may re-sync the environment back to the default torch profile.

---

### Windows — One Shot (PowerShell)

```powershell
# 1. Install uv
powershell -ExecutionPolicy ByPass -Command "irm https://astral.sh/uv/install.ps1 | iex"
# Restart PowerShell after installation

# 2. Clone and enter the repo
git clone https://github.com/faransansj/any-hoin.git
cd any-hoin

# 3. Install Python dependencies (CPU / inference only)
uv sync

# 3a. (Optional) NVIDIA GPU only — ROCm/Arc require WSL2
uv sync --extra cuda

# 4. Install frontend dependencies
npm install --prefix frontend

# 5. Start backend (Terminal 1)
.venv\Scripts\python studio_api.py

# 5. Start frontend (Terminal 2 — open a new PowerShell window)
npm run dev --prefix frontend
```

Frontend UI → `http://localhost:5173`  
Backend API → `http://localhost:8001`

> **Tip**: You can run both in the same terminal using PowerShell jobs, but two separate windows makes logs easier to read.

---

### Windows (WSL2) — One Shot

WSL2 lets you use the same `start.sh` as Linux, and unlocks ROCm / Intel Arc support.

```bash
# Inside WSL2 terminal (Ubuntu recommended)

# 1. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env

# 2. Install Node.js (if not already)
curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
sudo apt-get install -y nodejs

# 3. Clone and enter the repo
git clone https://github.com/faransansj/any-hoin.git
cd any-hoin

# 4. Install dependencies
uv sync                  # CPU
uv sync --extra cuda     # NVIDIA (CUDA in WSL2)
uv sync --extra rocm     # AMD GPU
uv sync --extra arc      # Intel Arc XPU

# 5. One-shot start
./start.sh
```

---

## 🔑 Danbooru API (Optional)

Without credentials, anonymous rate limits and 2-tag search restrictions apply.

```bash
cp .env.example .env
# Edit .env — set DANBOORU_LOGIN and DANBOORU_API_KEY
# Get your key at: https://danbooru.donmai.us/profile
```

---

## 🛠 Web UI Features

| Page | Description |
|------|-------------|
| **Crawl** | Collect images for characters from Danbooru |
| **Dataset** | Check and manage collected dataset status |
| **Training** | Start training, monitor Loss/Accuracy in real-time |
| **Export** | Export trained model (FP16 / INT8 / INT4 / ONNX) |
| **Inference** | Upload images and classify characters instantly |

---

## 📂 Project Structure

```text
any-hoin/
├── studio/                 # Backend system
│   ├── jobs/               # Async job runners (crawl, train, export)
│   └── routers/            # FastAPI route handlers
├── frontend/               # React + TypeScript + Vite UI
│   ├── src/pages/          # Crawl, Dataset, Training, Export, Inference
│   └── src/store/          # Zustand job state
├── crawling/               # Danbooru crawler
├── train.py                # Two-phase Swin-T fine-tuning
├── dataset.py              # Dataset split + augmentation
├── export_onnx.py          # PyTorch → ONNX export
├── quantize.py             # FP16 / INT8 / INT4 / INT2 quantization
├── studio_api.py           # FastAPI entry point (port 8001)
└── start.sh                # One-shot launcher (Mac/Linux/WSL2)
```

---

## ⚙️ Manual Startup (if `start.sh` is unavailable)

**Backend**
```bash
# Mac/Linux/WSL2
.venv/bin/python studio_api.py

# Windows
.venv\Scripts\python studio_api.py
```

**Frontend** (new terminal)
```bash
cd frontend && npm install && npm run dev
```

---

## 📊 Model

| Property | Value |
|----------|-------|
| Architecture | Swin Transformer-Tiny |
| Input | 224 × 224 RGB |
| Pretrained | ImageNet-1K |
| Parameters | ~28M |
| Training | Two-phase fine-tuning (frozen head → full unfreeze) |

---

## ⚠️ Notice

- This project is a demo for academic and non-commercial purposes only.
- All character copyrights belong to © Cover Corp.
- Please comply with the [Danbooru Terms of Service](https://danbooru.donmai.us/terms_of_service) when crawling.
