# 🌟 any-hoin — Integrated Hololive Character Classifier

[한국어](README.kr.md) | [English](README.md) | [日本語](README.ja.md)

any-hoin 是基于 Swin Transformer-Tiny 的 Hololive 角色分类器，是一个将数据收集 (Crawling)、模型训练 (Training) 和推理服务 (Inference) 集成在单一 Web UI 中管理的综合平台。

## 🚀 快速开始

无论使用哪种操作系统，请依次复制并执行以下命令。

### 1. 环境准备 (通用)
首先，安装最新的 Python 包管理器 `uv`。

**macOS / Linux**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env
```

**Windows (PowerShell)**
```powershell
powershell -ExecutionPolicy ByPass -Command "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. 项目配置及依赖安装
```bash
# 克隆仓库并进入目录
git clone https://github.com/faransansj/any-hoin.git
cd any-hoin

# 创建虚拟环境并安装依赖 (自动设置 Python 3.11)
uv sync

# (可选) GPU 后端 — 根据硬件选择其中一个
uv sync --extra cuda   # NVIDIA GPU
uv sync --extra rocm   # AMD GPU
uv sync --extra arc    # Intel Arc GPU (XPU)
```

> **Intel Arc 用户注意**: 执行 `uv sync --extra arc` 后，请使用 `.venv/bin/python` 或 `uv run --extra arc python` 运行脚本，而非直接使用 `uv run python`。不带 `--extra arc` 运行时，uv 可能会重新安装 CUDA 版 PyTorch，导致 XPU 支持失效。

### 3. 启动 (一键)

```bash
./start.sh
```

同时启动后端 (`http://localhost:8001`) 和前端 (`http://localhost:5173`)。按 `Ctrl+C` 一次性关闭所有服务。

<details>
<summary>手动启动 (如有需要)</summary>

**后端**
```bash
.venv/bin/python studio_api.py
```

**前端** (新终端)
```bash
cd frontend && npm install && npm run dev
```
</details>

---

## 🛠 集成 Web UI 主要功能

您现在可以通过 Web UI 控制所有流程，无需运行单独的脚本。

- **🌐 Crawl Page**: 从 Danbooru 选择性地收集角色图像。
- **📚 Dataset Page**: 查看并管理已收集数据集的状态。
- **🏋️ Training Page**: 开始模型训练，并通过图表实时监控训练指标 (Loss, Accuracy)。
- **🔮 Inference Page**: 上传图像立即进行角色分类并查看元数据。
- **💾 Export Page**: 导出训练好的模型和配置文件。

---

## 📂 项目结构 (最新)

```text
any-hoin/
├── studio/                 # 集成后端系统
│   ├── jobs/               # 异步任务处理 (爬虫, 训练, 导出)
│   │   ├── base_job.py     # 任务基类
│   │   ├── crawl_job.py    # 爬虫任务逻辑
│   │   └── train_job.py    # 训练任务逻辑
│   ├── routers/            # API 端点 (FastAPI)
│   │   ├── characters.py   # 角色元数据管理
│   │   ├── crawl.py        # 爬虫控制
│   │   └── training.py     # 训练控制
│   └── characters.py       # 角色定义及数据模型
├── frontend/               # 基于 React + TypeScript + Vite 的 Web UI
│   ├── src/pages/          # 功能页面 (Crawl, Training, Inference 等)
│   └── src/store/          # 任务状态管理 (Zustand)
├── crawling/               # 核心爬虫引擎 (danbooru_crawler.py)
├── train.py                # 模型训练核心引擎
├── dataset.py              # 数据集及增强流水线
└── studio_api.py           # 集成 API 服务器入口
```

---

## ⚙️ 详细配置

### Danbooru 凭据设置
建议设置 `.env` 文件以避免速率限制。
```bash
cp .env.example .env
# 打开 .env 并输入 DANBOORU_LOGIN, DANBOORU_API_KEY
```

## 📊 模型信息
- **Architecture**: Swin Transformer-Tiny
- **Input Size**: 224 $\times$ 224 RGB
- **Pretrained**: ImageNet-1K
- **Key Feature**: 结合 ViT 系列的高数据效率和全局特征捕捉能力，即使在数据量较少的情况下也能实现高分类准确率。

---

## ⚠️ 注意事项
- 本项目仅用于学术及非商业目的。
- 所有角色版权归 © Cover Corp. 所有。
- 爬取 Danbooru 时请遵守其服务条款。
