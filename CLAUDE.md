# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Rules

Detailed rules for each subsystem live in `.claude/rules/`. These are auto-loaded by Claude Code but are also useful as human-readable documentation:

- `architecture.md` — code tree
- `tracker.md` — tracking pipeline patterns, metrics, viz, utilities
- `dataset-pipeline.md` — 3-step dataset build pipeline
- `classification.md` — training CLI, LOCO cross-validation, model table
- `frame-loading.md` — H.264 seeking bug, always use sequential loading
- `gui.md` — GUI development (active on `gui` branch)

Additional context for specific packages in `src/dataset/CLAUDE.md` and `src/classification/CLAUDE.md`.

## Project Overview

End-to-end pipeline for automated play-behaviour classification.

The pipeline (i) tracks individual birds (three per pen) across 15-minute recordings using chunked SAM 3 with YOLO-guided adaptive boundary selection and text-grounded re-initialisation, (ii) extracts per-bird DINOv3 and V-JEPA 2.1 appearance embeddings and handcrafted motion features from decoded masks, and (iii) classifies locomotor play, object play, and no-play windows.

## Environment & Setup

**Package manager**: Pixi (not pip/conda directly). Python 3.11 only.

```sh
# Fetch git submodules
git submodule update --init --recursive

# Install all environments
pixi install
```

Pixi environments (defined in `pixi.toml`):

- **`default`** — base deps (python, torch, pandas, loguru, matplotlib, pytest, ruff)
- **`tracker`** — SAM3 tracking + YOLO scan (transformers, accelerate, omegaconf, ultralytics)
- **`dataset`** — build_dataset + extract_features (CPU-only, lightweight)
- **`embeddings`** — DINOv3/V-JEPA extraction (transformers, accelerate, peft, timm, einops)
- **`classifier`** — classification training (lightning, wandb, torchmetrics, xgboost)
- **`videoprism`** — JAX + VideoPrism

Platform is Linux-only (CUDA 12.6).

## Running Scripts

Scripts are run as Python modules from the project root. CUDA device is specified in the YAML config:

```sh
# Tracker pipeline (defaults to config/tracker.yaml)
pixi run -e tracker tracker
# Custom config:
pixi run -e tracker tracker --config config/tracker_manual_chunking.yaml
```

**Pixi quoting caveat**: `pixi run -e <env> python -c "..."` breaks on spaces in paths, f-string curly braces, and other special characters. **Never use inline `-c` with pixi.** Instead, write a temporary `.py` file in `tmp/` (project-local, gitignored) and run it with `pixi run -e <env> python tmp/<script>.py`.

## Running Tests

```sh
CUDA_VISIBLE_DEVICES=1 pixi run -e tracker test_tracker  # SAM3 tracking test
pixi run -e dataset test_features                         # Dataset feature extraction tests (pytest)
```

Pixi tasks invoke them as `python -m test.<name>` (the `src/` package root is on the module path). Dataset tests in `tests/` use pytest.

## Data Layout

- `data/` — Small test data (images, short video clips, DLC annotations, ethogram parquets)
  - `data/labels/` — Registration protocols Excel files (behaviour labels + bird info)
  - `data/tracking/` — Symlinks to tracking run output dirs (gitignored)
  - `data/dataset/` — Combined dataset outputs from the dataset pipeline (tracks, labels, features, embeddings)
  - `data/postprocessing/` — Version-controlled per-video JSONs + `tracking_outputs.parquet`, organized as `day_{N}/{video_subdir}/`. Source of truth for `build_dataset`.
- `ext-data/` — Symlink to `/mnt/birds/rebecca2025/` (longer videos, output results, image sequences)
  - `ext-data/test/batch_mode_test_set/` — 3 × 2-min clips (`test_video_1/2/3.mp4`) for batch mode testing
- `data/video/` — Symlinks to video directories (`batch/`, `batch2/`, `week_1_day_2/` → `/mnt/birds/rebecca2025/raw/`)
- `Grounded-SAM-2-fork/` — Git submodule (backburner)

## Key Dependencies

- PyTorch 2.9.1 (CUDA 12.6 on Linux)
- HuggingFace Transformers v5.0.0rc2 (installed from git), Accelerate
- supervision, loguru, OmegaConf, pycocotools, matplotlib
- ultralytics (YOLO scan)
- scikit-learn (legacy KMeans in `src/utils.py`, used only by `viz.py`)
- PyTorch Lightning, torchmetrics, wandb (classifier environment)

## Linting

Ruff is available as a workspace dependency. **Do not run ruff automatically** — the user handles linting manually.

```sh
pixi run ruff check src/ script/
pixi run ruff format src/ script/
```

## Misc. notes

- SAM3 (both HF-transformers and native) is extremely unstable when run in Jupyter notebooks and frequently crashes the kernel. Prefer running SAM3 via scripts; for notebooks, load SAM3 first, free GPU memory, then load lighter models (e.g., DINOv3).
