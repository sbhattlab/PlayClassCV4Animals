# Chicken behaviour classification

Multi-object tracking and segmentation of chickens in video data using SAM3,
with postprocessing, feature extraction, and behaviour classification.

## Installation

**Package manager**: [Pixi](https://pixi.sh) (not pip/conda directly).

```sh
git submodule update --init --recursive

# Install main (default) environment
pixi install

# Install SAM3 environment (main pipeline)
pixi install -e sam3-hf

# Launch shell
pixi shell -e sam3-hf
```

Other environments (`gs2`, `sam3-native`, `yolo`) exist but are not actively used. The `classifier` environment adds PyTorch Lightning + torchmetrics. The `videoprism` environment provides JAX + VideoPrism. Platform is Linux-only (CUDA 12.6).

## Data

```
data/
  labels/          Registration protocol Excel files (behaviour labels + bird info)
  tracking/        Symlinks to tracking run output dirs (gitignored)
  dataset/         Combined dataset outputs (tracks, labels, features, embeddings)
  postprocessing/  Version-controlled per-video postprocessing JSONs
  video/           Symlink to raw video files (gitignored)
  img/             Short test clips and diagnostic outputs (gitignored)
ext-data/          Symlink to /mnt/birds/rebecca2025/ (results, image sequences)
```

```sh
# ku-01: symlink video and external data
ln -s "/mnt/birds/rebecca2025/raw" data/video
ln -s "/mnt/birds/rebecca2025/" ext-data
```

### Dataset

Built from tracking outputs + registration protocol Excel files in three steps
(see `src/dataset/README.md` for full details):

```sh
# 1. Labels, postprocessing, windows (fast, ~seconds)
pixi run -e sam3-hf build_dataset

# 2. Mask features (CPU-only)
pixi run -e sam3-hf extract_features

# 3. DINOv3 embeddings (GPU required)
pixi run -e sam3-hf extract_embeddings --video-dir video-data/batch
```

All scripts auto-discover tracking runs under `data/tracking/` and write outputs to `data/dataset/`:

- `tracks.parquet` — postprocessed tracks with protocol bird IDs and window column
- `labels.parquet` — behaviour labels aligned to tracking windows
- `features_all.parquet` — per-frame mask features (spatial, temporal, pairwise)
- `features_windowed.parquet` — per-window feature summaries
- `embeddings.pt` — DINOv3 CLS-token embeddings per (video, bird, window)

## Tasks

Scripts are organized as: executable scripts in `script/`, reusable library modules in `src/`.
Run via pixi tasks or as Python modules from the project root.

### Tracking

> [!IMPORTANT]
> Read the base config file (`config/<tool name>_config.yaml`) and modify appropriately (e.g. CUDA device).

```sh
# Main SAM3-HF pipeline (defaults to config/sam3_hf_config.yaml)
pixi run -e sam3-hf sam3-hf-tracker

# Custom config
pixi run -e sam3-hf python -m script.sam3.run_sam3_hf --config config/sam3_hf_manual_chunking.yaml
```

### Post-tracking

| Script | Description |
|--------|-------------|
| `script/build_dataset.py` | Postprocess tracking outputs, match bird IDs, build dataset parquets |
| `script/extract_features.py` | Extract mask features + window summaries from dataset tracks (CPU) |
| `script/extract_embeddings.py` | Extract DINOv3 embeddings from dataset tracks (GPU) |
| `script/compute_chunk_boundaries.py` | Recompute YOLO scan metrics + chunk boundaries |
| `script/viz_chunk_boundaries.py` | Visualize chunk boundary frames |
| `script/viz_grounding.py` | Render grounding phase outputs onto video |
| `script/extract_embeddings_vjepa2.py` | Extract V-JEPA 2 video embeddings from dataset tracks (GPU) |
| `script/extract_embeddings_videoprism.py` | Extract VideoPrism video embeddings from dataset tracks (GPU, JAX) |
| `script/train.py` | Classification training with LOVO cross-validation (PyTorch Lightning) |
| `script/train_xgboost.py` | XGBoost baseline with LOVO cross-validation |

## Tests

> [!IMPORTANT]
> Set `CUDA_VISIBLE_DEVICES` explicitly before running GPU tests.

```sh
# Dataset tests (pytest)
pixi run -e sam3-hf test_features
pixi run -e sam3-hf test_postprocessing                     # Tracking postprocessing tests (pytest)
pixi run -e classifier pytest tests/test_pooling.py         # Segment pooling + attention tests (pytest)
pixi run -e sam3-hf pytest tests/test_post_build_dataset.py # Dataset integrity checks (pytest)
```

## Classification

Behaviour classification using LOVO (Leave-One-Video-Out) cross-validation.
Best result: **0.744 pooled macro F1** (TemporalCNNv2 on multi-scale DINOv3 embeddings + handcrafted features).
See `notes/ablation_final.md` for full ablation tables.

```sh
# Features only (MLP baseline)
pixi run -e classifier train --model mlp --input features --exclude social

# Best model: temporal CNN on features + multi-scale embeddings
pixi run -e classifier train --model temporal_cnn2 --input features+embeddings+embeddings_plain256+embeddings_union512 --exclude social --dropout 0.0 --n-segments 24

# XGBoost baseline
pixi run -e classifier python -m script.train_xgboost --exclude social
```
