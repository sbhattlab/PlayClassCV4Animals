# PlayClass

A pipeline for play behaviour recognition in videos of poultry with tracking, postprocessing, feature extraction and classification.

## Installation

**Package manager**: [Pixi](https://pixi.sh) (not pip/conda directly).

```sh
git submodule update --init --recursive
pixi install
```

Pixi environments: `default` (base), `tracker` (SAM3), `dataset` (build + features), `embeddings` (DINOv3/V-JEPA), `classifier` (training), `videoprism` (JAX), `gs2` (Grounded-SAM-2; used for tracker benchmarking), `tracker-evaluation` (CPU-only tracker scoring; motmetrics + pycocotools). Platform is Linux-only (CUDA 12.6).

## Data

```
data/
  labels/          Registration protocol Excel files (behaviour labels + bird info)
  tracking/        Symlinks to tracking run output dirs (gitignored)
  postprocessing/  Version-controlled per-video postprocessing JSONs + parquets (day_28/, day_29/)
  tracker_eval/    Version-controlled tracker benchmark artefacts (video manifest, keyframes, ablation configs, scored results)
ext-data/          Symlink to large data outputs (results, image sequences, embeddings, etc.)
```

### Dataset

Built from tracking outputs + registration protocol Excel files in three steps:

```sh
# 1. Labels, postprocessing, windows (fast, ~seconds)
pixi run -e dataset build_dataset

# 2. Mask features (CPU-only)
pixi run -e dataset extract_features

# 3. Embeddings (GPU required, multiple backbones available)
pixi run -e embeddings extract_embeddings_dinov3                                      # DINOv3 ViT-L (default)
pixi run -e embeddings python -m script.extract_embeddings_vjepa2 --temporal          # V-JEPA 2.1 ViT-L
pixi run -e videoprism extract_videoprism --temporal                                  # VideoPrism Base
```

All scripts default to `data/postprocessing/` (input) and `data/dataset/` (output). Video dirs are auto-discovered under `data/video/`.

Outputs in `data/dataset/`:

- `tracks.parquet` — postprocessed tracks with protocol bird IDs and window column
- `labels.parquet` — behaviour labels aligned to tracking windows
- `features_all.parquet` — per-frame mask features (spatial, temporal, pairwise)
- `features_windowed.parquet` — per-window feature summaries
- `embeddings_{backbone}_{size}[_{variant}].pt` — embeddings per (video, bird, window)

## Tasks

Scripts are organized as: executable scripts in `script/`, reusable library modules in `src/`.
Run via pixi tasks or as Python modules from the project root.

### Tracker

> [!IMPORTANT]
> Read the base config file (`config/tracker.yaml`) and modify appropriately (e.g. video path, CUDA device).

```sh
# Main SAM3 Tracker pipeline (defaults to config/tracker.yaml)
pixi run tracker

# Custom config
pixi run -e tracker python -m script.run_tracker --config config/tracker_manual_chunking.yaml
```

### Post-tracking

| Script | Description |
|--------|-------------|
| `script/build_dataset.py` | Postprocess tracking outputs, match bird IDs, build dataset parquets |
| `script/extract_features.py` | Extract mask features + window summaries from dataset tracks (CPU) |
| `script/extract_embeddings_dinov3.py` | Extract DINOv3 embeddings from dataset tracks (GPU) |
| `script/extract_embeddings_vjepa2.py` | Extract V-JEPA 2/2.1 video embeddings (GPU) |
| `script/extract_embeddings_videoprism.py` | Extract VideoPrism video embeddings (GPU, JAX) |
| `script/compute_chunk_boundaries.py` | Recompute YOLO scan metrics + chunk boundaries |
| `script/train.py` | Classification training with LOCO cross-validation (PyTorch Lightning) |
| `script/train_xgboost.py` | XGBoost baseline with LOCO cross-validation |

## Tests

```sh
# Dataset tests (pytest)
pixi run -e dataset test_features
pixi run -e dataset test_postprocessing
pixi run -e dataset test_post_build

# Tracker test (standalone, not pytest)
pixi run -e tracker test_tracker
```

## Classification

Behaviour classification using LOCO (Leave-One-Cage-Out) cross-validation.
Best result: **0.773 pooled macro F1** (TemporalCNNv2 on features + DINOv3 plain256 + V-JEPA 2.1).
See v0.2.0 release notes for full ablation tables.

```sh
# Features only (MLP baseline)
pixi run -e classifier train --model mlp --input features --exclude social

# Best model: temporal CNN on features + V-JEPA 2.1
pixi run -e classifier train --model temporal_cnn2 --input features+embeddings_vjepa21_vitl_temporal --exclude social --dropout 0.0 --n-segments 32

# XGBoost baseline
pixi run -e classifier train_xgboost --exclude social
```

## Tracker evaluation

Held-out tracker benchmark over 5 videos with sparse CVAT-annotated keyframes, scored with motmetrics + TrackEval. Compares SAM3 variants against Grounded-SAM-2 baselines. Preparation runs in the `tracker` env (needs torch); scoring runs in the CPU-only `tracker-evaluation` env.

```sh
# Prepare video manifest + keyframe schedule (tracker env)
pixi run -e tracker prepare-tracker-eval

# Convert tracker outputs to MOT format, then score against CVAT ground truth
pixi run -e tracker-evaluation convert_predictions
pixi run -e tracker-evaluation score-tracker-eval
```

See [`src/tracker_eval/README.md`](src/tracker_eval/README.md) for the full ablation table, CVAT handoff workflow, per-variant configs, and results schema.

## Data availability

The 30 video recordings analysed in the accompanying paper are part of an ongoing study of play behaviour in young chickens. The full dataset (videos, ethograms, tracking labels) will be released publicly upon completion of the broader study, subject to institutional review. For early access requests, please contact the corresponding authors.
