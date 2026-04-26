# 🌟 any-hoin — Integrated Hololive Character Classifier

[한국어](README.kr.md) | [English](README.md) | [中文](README.zh.md)

any-hoinはSwin Transformer-Tinyベースのホロライブキャラクター分類器であり、データ収集(Crawling)、モデル学習(Training)、および推論サービス(Inference)を一つの統合Web UIで管理できる統合プラットフォームです。

## 🚀 クイックスタート

どのOSでも、以下のコマンドを順番にコピーして貼り付けてください。

### 1. 環境準備 (共通)
まず、最新のPythonパッケージマネージャーである `uv` をインストールします。

**macOS / Linux**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env
```

**Windows (PowerShell)**
```powershell
powershell -ExecutionPolicy ByPass -Command "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. プロジェクト設定および依存関係のインストール
```bash
# リポジトリのクローンおよびディレクトリ移動
git clone https://github.com/faransansj/any-hoin.git
cd any-hoin

# 仮想環境の作成および依存関係のインストール (Python 3.11 自動設定)
uv sync

# (オプション) GPUバックエンド — ハードウェアに合わせて一つ選択
uv sync --extra cuda   # NVIDIA GPU
uv sync --extra rocm   # AMD GPU
uv sync --extra arc    # Intel Arc GPU (XPU)
```

> **Intel Arcユーザーへ**: `uv sync --extra arc` 後、スクリプト実行には `uv run python` ではなく `.venv/bin/python` または `uv run --extra arc python` を使用してください。`--extra arc` なしで実行すると、uvがCUDAビルドのPyTorchを再インストールしてXPUサポートが壊れる場合があります。

### 3. 起動 (ワンショット)

```bash
./start.sh
```

バックエンド (`http://localhost:8001`) とフロントエンド (`http://localhost:5173`) を一度に起動します。`Ctrl+C` で全終了。

<details>
<summary>手動起動 (必要な場合)</summary>

**バックエンド**
```bash
.venv/bin/python studio_api.py
```

**フロントエンド** (新しいターミナル)
```bash
cd frontend && npm install && npm run dev
```
</details>

---

## 🛠 統合Web UI 主要機能

個別のスクリプトを実行することなく、Web UIからすべてのプロセスを制御できます。

- **🌐 Crawl Page**: Danbooruからキャラクター別に画像を限定的に収集します。
- **📚 Dataset Page**: 収集されたデータセットの状態を確認し、管理します。
- **🏋️ Training Page**: モデル学習を開始し、学習メトリクス(Loss, Accuracy)をリアルタイムにチャートで確認します。
- **🔮 Inference Page**: 画像をアップロードして即座にキャラクターを分類し、メタデータを確認します。
- **💾 Export Page**: 学習済みモデルと設定ファイルをエクスポートします。

---

## 📂 プロジェクト構造 (最新)

```text
any-hoin/
├── studio/                 # 統合バックエンドシステム
│   ├── jobs/               # 非同期タスク処理 (クローリング, 学習, エクスポート)
│   │   ├── base_job.py     # タスク基本クラス
│   │   ├── crawl_job.py    # クローリングタスクロジック
│   │   └── train_job.py    # 学習タスクロジック
│   ├── routers/            # APIエンドポイント (FastAPI)
│   │   ├── characters.py   # キャラクターメタデータ管理
│   │   ├── crawl.py        # クローリング制御
│   │   └── training.py     # 学習制御
│   └── characters.py       # キャラクター定義およびデータモデル
├── frontend/               # React + TypeScript + ViteベースのWeb UI
│   ├── src/pages/          # 機能別ページ (Crawl, Training, Inferenceなど)
│   └── src/store/          # タスク状態管理 (Zustand)
├── crawling/               # コアクローリングエンジン (danbooru_crawler.py)
├── train.py                # モデル学習コアエンジン
├── dataset.py              # データセットおよびAugmentationパイプライン
└── studio_api.py           # 統合APIサーバーエントリーポイント
```

---

## ⚙️ 詳細設定

### Danbooru認証設定
レートリミットを避けるため、`.env` ファイルの設定を推奨します。
```bash
cp .env.example .env
# .envファイルを開き、DANBOORU_LOGIN, DANBOORU_API_KEYを入力
```

## 📊 モデル情報
- **Architecture**: Swin Transformer-Tiny
- **Input Size**: 224 $\times$ 224 RGB
- **Pretrained**: ImageNet-1K
- **Key Feature**: ViT系の高いデータ効率とグローバルな特徴抽出能力を組み合わせ、少ないデータでも高い分類精度を実現します。

---

## ⚠️ 注意事項
- 本プロジェクトは学術および非商業目的のデモです。
- すべてのキャラクターの著作権は © Cover Corp.に帰属します。
- Danbooruクローリング時は、サイトの利用規約を遵守してください。
