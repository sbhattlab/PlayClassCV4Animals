# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

End-to-end pipeline for automated play-behaviour classiciation.

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

## Architecture

```
script/
  run_tracker.py            # Config-driven tracking pipeline (main script)

src/
  utils.py                  # Config/logging/output dirs, chunking, parquet export, video annotation
  masks.py                  # Mask/bbox/point extraction, tracking output normalization
  grounding.py              # Text-prompt grounding, best-frame selection, ID matching
  metrics.py                # Tracking metrics: mask-based, per-frame, per-id, per-run, summary
  viz.py                    # Visualizations: ID timeline, dashboard, score plots, mask evolution, prompt points
  chunk_boundaries.py       # Per-frame metrics, occlusion detection, separation windows, adaptive chunking
  yolo_scan.py              # YOLO inference only (run_yolo_scan); re-exports src.chunk_boundaries for compat
  ethogram.py               # Behavior label parsing from Excel registration protocols
  dataset/                  # Dataset construction package
    __init__.py             # Package marker (no re-exports; import from submodules directly)
    utils.py                # Shared helpers: fmt_time, get_video_fps, resolve_video_path, assert_embedding_label_alignment, load_video_frames_sequential
    tracking_issues.py      # Detection: ID switches, mask overlaps, low-score periods
    tracking_postprocessing.py  # Remediation: prefill from issues, ID-scoped trims, ID remaps, process_tracks
    labels.py               # Behaviour label parsing from Excel registration protocols
    features.py             # Handcrafted mask features: spatial, temporal, pairwise, window summarization (vectorized)
    embeddings/             # Embedding extraction package
      __init__.py           # Re-exports for backwards compat
      dinov3.py             # DINOv3 CLS-token embedding extraction from bbox crops
      vjepa2.py             # V-JEPA 2/2.1 video embedding extraction
      videoprism.py         # VideoPrism video embedding extraction
    crops.py              # Shared crop modes: crop_frame, compute_union_origin, compute_union_bbox
  classification/           # Behaviour classification package
    models.py               # Backbones: SimpleLinear, SimpleMLP, TemporalMLP, TemporalCNNv2. MODEL_REGISTRY maps names to (cls, temporal_flag).
    datamodule.py           # BehaviourDataset + BehaviourDataModule (LOVO split, segment pooling)
    trainer.py              # BehaviourClassifier LightningModule (weighted CE, AdamW, MetricCollection)
    stats.py                # LOVO aggregation: scalar summary CSV, summed confusion matrices
  debug/                    # Interactive debugging utilities and standalone grounding test script

script/
  build_dataset.py                 # Labels, postprocessing, windowing → tracks + labels parquets
  extract_features.py              # Mask feature extraction + window summarization (CPU-only)
  extract_embeddings_dinov3.py     # DINOv3 CLS-token embeddings (GPU required)
  extract_embeddings_vjepa2.py     # V-JEPA 2/2.1 video embeddings (GPU required)
  extract_embeddings_videoprism.py # VideoPrism video embeddings (JAX/GPU)
  save_cid_crops.py                # Save union384 crops to disk for CID pretraining
  cid_vjepa21.py                   # CID pretraining of V-JEPA 2.1
  train.py                         # Classification training CLI (PyTorch Lightning)
  train_xgboost.py                 # XGBoost baseline (LOCO/LOVO)
  compute_chunk_boundaries.py      # Recompute metrics + boundaries from yolo_tracking.parquet

config/                     # YAML configs (OmegaConf)
tests/                      # pytest tests (dataset features, labels, postprocessing)
src/test/                   # Test scripts for SAM3 inference (run via pixi tasks, not pytest)
notebook/                   # Jupyter notebooks for EDA and demos
```

### Key patterns

- **Config-driven**: `run_tracker.py` reads YAML configs via OmegaConf. Script entry point parses environment variables `CUDA_VISIBLE_DEVICES` and `PYTORCH_ALLOC_CONF` from config file before torch is imported. A `tracking:` section overrides `Sam3VideoConfig` parameters (keep-alive, IoU thresholds, reconditioning interval, etc.).
- **Timestamped output**: Each run creates `{output_dir}/{YYYYMMDD_HHMMSS}_{job_type}/` with subdirectories for `metrics/` and `visualizations/`. Config is copied for reproducibility.
- **Loguru logging**: Console (colored) + file handler in run directory. Replaces all `print()`.
- **Chunked processing**: Long videos are split into chunks via `chunk_video_frames_adaptive()`. Initial chunk size is set by `chunk_seconds` (default 60 s); when YOLO scan data is available, each boundary is shifted within a ±`adaptive_search_window_seconds` window to the frame with the highest separation score. Chunk 0 uses `Sam3VideoModel` (text-prompted); subsequent chunks use `Sam3TrackerVideoModel` (point-prompted). Small trailing remainders (<10% of chunk size) are absorbed into the last chunk. Point prompts are extracted from previous chunk's masks via `extract_equidistant_points_from_masks()`. `find_frame_with_enough_objects()` searches backwards for a frame with enough detected objects. `max_frames_to_track` limits how many frames are processed per video.
- **Two model phases**: `Sam3VideoModel` (text→segmentation) for initialization, `Sam3TrackerVideoModel` (point→tracking) for propagation. Each chunk loads its model fresh and cleans up GPU memory afterwards (`free_gpu_memory()` with triple `gc.collect` + CUDA cache clearing).
- **Adaptive chunking (YOLO scan)**: When `use_adaptive_chunking: true`, `run_yolo_scan()` (in `src/yolo_scan.py`) runs YOLO tracking on the full video and returns a raw `yolo_df`. Analysis — `compute_yolo_per_frame_metrics()` → `identify_occlusion_periods()` → `find_high_separation_windows()` — lives in `src/chunk_boundaries.py`. `chunk_video_frames_adaptive()` refines boundaries in priority order — separation-first (highest separation_score inside a high-separation window), then occlusion avoidance (farthest from occlusion with 90%/50% directional penalties), validated against `adaptive_max_chunk_seconds`. Scan outputs saved as `yolo_tracking.parquet`, `yolo_scan_metrics.parquet`, `yolo_scan_summary.parquet`. A `yolo_scan:` config section controls model, thresholds, and tracker config. To reanalyse boundaries without re-running YOLO, use `script/compute_chunk_boundaries.py`.
- **Manual chunking**: Set `manual_chunk_frames` to a list of `[start, end]` pairs to override fixed/adaptive chunking entirely. First pair → `Sam3VideoModel`; subsequent → `Sam3TrackerVideoModel`. Disables `yolo_scan_only`/`use_adaptive_chunking` with warnings. See `build_manual_chunks()` in `src/utils.py` and `config/tracker_manual_chunking.yaml`.
- **Batch processing**: Set `video_dir` instead of `video_path` to process all videos in a directory. Each video gets its own subdirectory under a shared timestamped batch dir. Errors caught per-video. `manual_chunk_frames` accepts a dict keyed by **basename** (e.g. `"video1.mp4": [[0,375],...]`) for per-video boundaries; unlisted videos use fixed chunking.
- **Device selection**: `Accelerator().device` from HuggingFace Accelerate.

### Metrics module (`src/metrics.py`)

Computes tracking quality post-hoc from `outputs_per_frame`. Four levels:

- **Per-frame** (`compute_per_frame_metrics`): Pairwise mask IoU, centroids, clustering coefficient, mask area stats, occlusion flags.
- **Per-run** (`compute_per_run_metrics`): Per contiguous segment per ID — runs, gaps, coverage, `mean_tracker_score`.
- **Summary** (`compute_summary_metrics`): Continuity, fragmentation, ID switch rate; occlusion-aware ID switch count when `per_frame_metrics` provided.
- **Occlusion-aware ID switch detection** (`detect_identity_switches`): Flags ID changes only when a recent high-occlusion event occurred within a sliding window.

### Visualizations module (`src/viz.py`)

Auto-generated on each run, saved to `run_dir/visualizations/`: ID timeline (tracker score color-coded), per-frame dashboard (5-panel timeseries), per-ID tracker scores, mask evolution at chunk boundaries, prompt points at boundaries, YOLO scan overview (when adaptive chunking enabled). All plots use MM:SS x-axis when FPS is available.

### Features module (`src/dataset/features.py`)

Extracts handcrafted features from tracking masks. All functions are vectorized (no row-by-row iteration):

- **`extract_spatial_features`**: Batch `pycocotools.mask.area()` for mask areas, numpy array math for bbox metrics, batch `mask_util.decode()` + `scipy.ndimage.center_of_mass` for centroids. Masks grouped by `size` (pycocotools requirement) and chunked at 256 to cap memory.
- **`extract_temporal_features`**: `pandas.groupby().diff()` / `.shift()` for frame-to-frame deltas.
- **`extract_pairwise_features`**: Numpy broadcasting for `(M, M)` distance matrices per frame.
- **`summarize_features_by_window`**: Single `groupby().agg()` call with `[mean, std, min, max, median]`.

Feature set (`_FEATURE_COLS`): `mask_area`, `aspect_ratio`, `velocity`, `area_change_rate`, `min_dist_to_other`, `mean_dist_to_other`. Additionally `bbox_area` is output by spatial features as a debugging column (not in `_FEATURE_COLS`) for detecting saltatory bbox spikes.

### Embeddings package (`src/dataset/embeddings/`)

Embedding extraction from tracked objects. Three backends:

- **`dinov3.py`**: DINOv3 CLS-token extraction. Collects all bbox crops across windows, batch-processes through the model. Uses bfloat16.
- **`vjepa2.py`**: V-JEPA 2/2.1 video embeddings. Processes non-overlapping clips of `num_frames`, spatially mean-pools per timestep. Includes `VJEPA21Wrapper` for torch.hub models.
- **`videoprism.py`**: VideoPrism (JAX/Flax). Same clip-based approach as V-JEPA but uses JAX arrays.

All backends output `{(video_id, bird_id, window): Tensor(F_w, D)}`. Embedding filenames follow `embeddings_{backbone}_{size}[_{variant}].pt` (e.g. `embeddings_dinov3_vitl.pt`, `embeddings_vjepa21_vitb_temporal.pt`). Model size is inferred via `parse_model_size()` in `src/dataset/utils.py`.

### Classification package (`src/classification/`)

Behaviour classification from dataset outputs. Uses PyTorch Lightning (pixi env `classifier`).

- **`models.py`**: Pure `nn.Module` backbones. `SimpleLinear` (linear probe), `SimpleMLP` (features MLP), `TemporalMLP` (gated attention pool), `TemporalCNNv2` (GELU bottleneck + conv). All temporal models expect `(B, F, D)` input. `MODEL_REGISTRY` maps model names to `(backbone_cls, temporal_flag)`.
- **`datamodule.py`**: `BehaviourDataModule` loads data once via `prepare_data()`, then re-splits per fold via `set_split_groups(test_video, val_video)` + `setup()`. Exposes `video_ids`, `class_weights` (inverse-sqrt from train split), `n_classes`, `data_dim`, `label_encoder`. Features are median-imputed (2 rows with NaN pairwise distances from solo-bird windows) then z-score normalized at load time. Embeddings are mean-pooled by default; when `temporal=True`, segment-pooled via adaptive average pooling.
- **`trainer.py`**: `BehaviourClassifier(LightningModule)` wraps any backbone. Weighted CE loss with label smoothing (0.1), AdamW. Uses `torchmetrics.MetricCollection` with `MulticlassF1Score` (macro) + `MulticlassConfusionMatrix`, cloned with prefixes for train/val/test stages. `evaluate_fold` in `train.py` loads the best checkpoint manually and re-evaluates all splits (Lightning resets metrics after fit/test).
- **`stats.py`**: LOVO aggregation. `aggregate_metrics()` dispatches to `aggregate_scalars()` (summary CSV with MEAN/STD/POOLED rows) and `aggregate_confusion_matrices()` (summed CMs as .txt + .npy for train/val/test, returns pooled F1 per split).

`LABEL_ORDER` and `DEFAULT_FPS` are defined in `src/_config.py`. `--exclude` removes a class. Training script: `script/train.py` (`pixi run -e classifier train`).

### Output schemas

- **`tracking_outputs.parquet`**: One row per (frame, object). MultiIndex `["frame_idx", "object_id"]`. Includes bbox, RLE mask, scores, tracker_score, chunk_idx, model_type.
- **`chunk_info.json`**: Per-chunk metadata: frame range, model type, prompt points, timing. Keys: `grounding_used`, `grounding_source_frame_idx`, `grounding_num_objects`, `grounding_fallback_reason`.
- **YOLO scan outputs**: `yolo_tracking.parquet`, `metrics/yolo_scan_metrics.parquet`, `metrics/yolo_scan_summary.parquet`, `visualizations/yolo_scan_overview.png`.
- **Metrics parquets**: `per_frame_metrics`, `per_id_metrics`, `summary_metrics`.
- **Dataset outputs** (saved to `data/dataset/` by the dataset pipeline):
  - `tracks.parquet` — cleaned tracks with window column
  - `labels.parquet` — aligned behaviour labels
  - `features_all.parquet` — per-frame mask features
  - `features_windowed.parquet` — per-window feature summaries
  - `embeddings.pt` — DINOv3 embeddings keyed by `(video_id, bird_id, window)`
- **Run directory**: `{timestamp}_{job_type}/` with config copy, log, parquets, `metrics/`, `visualizations/`. Batch mode: `{timestamp}_{job_type}/{sanitized_stem}/` per video.

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

## Dataset Pipeline

Three-step pipeline. Steps 2 and 3 are independent and can run in parallel (different resources).

```sh
# 1. Labels, postprocessing, windows (fast, ~seconds)
pixi run -e dataset build_dataset

# 2. Mask features (slow, ~2 min, CPU-only)
pixi run -e dataset extract_features

# 3. DINOv3 embeddings (slow, GPU required)
pixi run -e embeddings extract_embeddings \
    --video-dir data/video/batch
```

All scripts default to `data/postprocessing/` (input) and `data/dataset/` (output). Override with `--tracking-dir` / `--dataset-dir` / `--output-dir`.

### Step 1: `build_dataset`

`script/build_dataset.py` discovers all `tracking_outputs.parquet` files recursively under `data/postprocessing/` (across day directories), parses labels from Excel, detects issues, applies postprocessing, assigns windows, and filters incomplete windows. On first run it generates `tracking_postprocessing.json` templates; manually fill in `to` (id_switch) and `tracking_id` (id_match) fields, then rerun.

Per-video JSON files (`tracking_postprocessing.json`) support three entry types:

- **`trim`** — Remove track rows in a frame range, optionally scoped to one ID. Causes: `overlap`, `low_score`, `merged_object`.
- **`id_switch`** — Remap `from` ID to `to` ID for all rows before `frame` (merges split tracks).
- **`id_match`** — Rename tracker `tracking_id` to real `protocol_id` (bird identity).

Output: `tracks.parquet` + `labels.parquet` in `data/dataset/`.

### Step 2: `extract_features`

`script/extract_features.py` reads `tracks.parquet` from `data/dataset/`, decodes RLE masks, computes per-frame spatial/temporal/pairwise features, then summarizes per window. Also builds temporal feature tensors (per-frame features binned per window, same key format as embeddings).

Feature set (19 features): `mask_area`, `aspect_ratio`, `velocity`, `acceleration`, `velocity_autocorr`, `area_change_rate`, `min_dist_to_other`, `mean_dist_to_other`, plus additional spatial/temporal derivatives.

Output: `features_all.parquet` (per-frame) + `features_windowed.parquet` (per-window summaries) + `features_binned.pt` (temporal feature tensors keyed by `(video_id, bird_id, window)`).

### Step 3: `extract_embeddings`

`script/extract_embeddings.py` reads `tracks.parquet` from `data/dataset/`, crops bbox regions from video frames, and extracts DINOv3 CLS-token embeddings per (video_id, bird_id, window).

```sh
pixi run -e embeddings python -m script.extract_embeddings \
    --video-dir data/video --device 0

# Alternative backbone (ViT-B)
pixi run -e embeddings python -m script.extract_embeddings \
    --video-dir data/video --device 0 \
    --model-name facebook/dinov3-vitb16-pretrain-lvd1689m

# Custom resolution
pixi run -e embeddings python -m script.extract_embeddings \
    --video-dir data/video --device 0 \
    --resolution 256
```

Output filename encodes variant: `embeddings.pt` (default), `embeddings_vitb.pt`, `embeddings_r256.pt`. Dict keyed by `(video_id, bird_id, window)` with `Tensor(F_w, D)` values.

### Step 3b: Video embeddings (`extract_embeddings_vjepa2` / `extract_embeddings_videoprism`)

Two scripts for video-model embedding extraction. Both use shared `CROP_MODES` from `src/dataset/crops.py` (`--crop-mode`: bbox, plain256, union512, darken512, roi512).

- **`script/extract_embeddings_vjepa2.py`** (embeddings env) — V-JEPA 2 embeddings. Feeds K frames as a video clip through the backbone. Supports `--temporal` (per-timestep spatial-mean-pool).
- **`script/extract_embeddings_videoprism.py`** (videoprism env, JAX) — VideoPrism embeddings. Supports `--temporal`, `--raw` (full patch tokens).

```sh
# V-JEPA 2 temporal (embeddings env)
pixi run -e embeddings python -m script.extract_embeddings_vjepa2 \
    --video-dir data/video/batch data/video/batch2 --device 0 --temporal

# VideoPrism temporal (videoprism env)
pixi run -e videoprism extract_videoprism \
    --video-dir data/video/batch data/video/batch2 --device 0 --temporal

# VideoPrism raw tokens for trainable pooler (~88 GB)
pixi run -e videoprism extract_videoprism \
    --video-dir data/video/batch data/video/batch2 --device 0 --raw
```

Output: `embeddings_vjepa2_vitl.pt`, `embeddings_videoprism_temporal.pt`, `embeddings_videoprism_raw.pt`, etc.

## Classification Training

Runs in the `classifier` pixi environment. Requires dataset pipeline outputs (steps 1–3 above).

Always runs full LOVO (Leave-One-Video-Out) cross-validation across all videos. Val video is auto-selected from the next cage (cage-aware rotation to avoid environment leakage). A fresh model is created per fold.

```sh
# Features only (MLP)
pixi run -e classifier train --model mlp --input features --exclude social

# Mean-pooled embeddings (MLP)
pixi run -e classifier train --model mlp --input embeddings_250 --exclude social

# Features + embeddings combined
pixi run -e classifier train --model mlp --input features+embeddings --exclude social

# Temporal model on embeddings
pixi run -e classifier train --model temporal_mlp --input embeddings --exclude social

# Temporal hybrid mode (temporal embeddings + flat windowed features)
pixi run -e classifier train --model temporal_cnn2 --input features+embeddings --exclude social

# Dry run (first fold, 1 batch, no checkpoints)
pixi run -e classifier train --model mlp --input features --dry-run
```

LOVO output structure:

```
data/eval/20260304_174946_mlp/
├── cfg.json                          # CLI args for reproducibility
├── fold_0_C1G1/                      # per-fold checkpoints + CSVLogger
│   ├── checkpoints/                  # ModelCheckpoint (best val_loss)
│   └── lightning_logs/version_0/     # CSVLogger (metrics.csv per epoch)
├── fold_1_C1G3/
├── ...
├── lovo_summary.csv                  # scalar metrics per fold + MEAN/STD
├── lovo_train_confusion_matrix.txt   # summed train CM
├── lovo_val_confusion_matrix.txt     # summed val CM
├── lovo_test_confusion_matrix.txt    # summed test CM (disjoint test sets)
└── lovo_test_confusion_matrix.npy    # summed test CM as numpy array
```

| `--model` | Backbone | Temporal |
|-----------|----------|----------|
| `linear` | `SimpleLinear` | No (mean-pool / windowed stats) |
| `mlp` | `SimpleMLP` | No (mean-pool / windowed stats) |
| `temporal_mlp` | `TemporalMLP` | Yes (segment-pooled, gated attention) |
| `temporal_cnn2` | `TemporalCNNv2` | Yes (GELU bottleneck + single conv) |

Input is specified separately via `--input`: `features`, `embeddings`, `embeddings_250`, `features+embeddings`, etc. Any `embeddings[_variant]` maps to `{name}.pt` in the dataset dir. Multiple embedding files can be concatenated: `features+embeddings+embeddings_union512` loads both and concatenates along the feature dimension (multi-scale). Non-temporal models mean-pool embeddings and use windowed feature stats; temporal models segment-pool embeddings and use binned feature tensors. Temporal models also support **hybrid mode**: e.g. `--model temporal_mlp --input features+embeddings` passes temporal embeddings through the backbone and concatenates flat windowed features before the classification head.

Additional training args: `--dropout` (override model dropout), `--d-hidden` (override hidden dim), `--label-smoothing` (CE label smoothing), `--feature-dropout` (dropout on flat features only), `--n-segments` (temporal bins, default 12).

### Classification results (2026-03-23)

Dataset: 30 videos (15 day 28 + 15 day 29), 5 cages, 3-class (excluding social). LOCO cross-validation (leave-one-cage-out, 5 folds).

Training config: 5 fixed epochs, inv-sqrt class weights, label smoothing 0.1, dropout 0.0, best val_loss checkpoint. See `notes/ablation_v2.md` for full results.

**Best results**:

| Model | Input | Pooled Test F1 |
|-------|-------|----------------|
| TemporalCNNv2 (K=32) | features + plain256 + V-JEPA 2.1 | **0.773** |
| TemporalCNNv2 (K=32) | features + multi-scale DINOv3 + V-JEPA 2.1 | 0.771 |
| TemporalCNNv2 (K=32) | features + V-JEPA 2.1 | 0.770 |
| TemporalCNNv2 (K=24) | features + 3-scale DINOv3 | 0.765 |
| V-JEPA 2.1 only (TemporalCNNv2 K=32) | V-JEPA 2.1 | 0.763 |
| TemporalCNNv2 (K=24) | features + DINOv3 plain256 | 0.759 |
| MLP | features + V-JEPA 2.1 | 0.761 |
| MLP | features + DINOv3 | 0.752 |
| XGBoost (depth=3) | features only | 0.748 |

**Key findings**:

- **V-JEPA 2.1 is the strongest backbone** (0.763 alone, 0.770 with features), reversing v1 where DINOv3 led. The video model's temporal understanding now outperforms per-frame image embeddings.
- **Features remain essential**: +0.7 on top of V-JEPA 2.1 (0.763 → 0.770), +1.3 on DINOv3 (0.740 → 0.752). Handcrafted mask features provide complementary spatial/kinematic signal.
- **Multi-scale DINOv3 disappoints**: union512 alone hurts (0.743 < 0.745 single-scale). 3-scale (bbox+256+512) recovers to 0.765 but doesn't beat V-JEPA 2.1.
- **Temporal modeling matters**: MLP mean-pools V-JEPA 2.1 to 0.729; TemporalCNNv2 preserves temporal structure for 0.763 (+3.4 points).
- **XGBoost features-only at 0.748** remains a strong baseline, beating several embedding configurations.

## Utilities

### Recomputing Scan Results

Recompute scan metrics and chunk boundaries from an existing `yolo_tracking.parquet` without re-running YOLO inference:

```sh
pixi run -e tracker python -m script.compute_chunk_boundaries \
    --run-dir ext-data/output/results/yolo_scan/20260223_231859_yolo_scan
```

Or from Python (using individual steps from `src.chunk_boundaries`):

```python
import pandas as pd
from src.chunk_boundaries import (
    compute_yolo_per_frame_metrics,
    identify_occlusion_periods,
    find_high_separation_windows,
    chunk_video_frames_adaptive,
)

yolo_df = pd.read_parquet(run_dir / "yolo_tracking.parquet")
per_frame_metrics = compute_yolo_per_frame_metrics(yolo_df)
occlusion_periods = identify_occlusion_periods(per_frame_metrics, window_frames=25)
separation_windows = find_high_separation_windows(per_frame_metrics)
chunks = chunk_video_frames_adaptive(total_frames, fps,
    chunk_seconds=60,
    per_frame_metrics=per_frame_metrics,
)
```

Default thresholds (`occlusion_iou_threshold: 0.15`, `high_occlusion_threshold: 0.3`) provide balanced detection.

### YOLO-Scan-Only Mode

```sh
pixi run -e tracker python -m script.run_tracker --config config/tracker_scan.yaml
```

Outputs saved to `{output_dir}/{timestamp}_yolo_scan/`.

### Config Reference

| Config | Purpose |
|--------|---------|
| `config/tracker.yaml` | Main tracker config (single video, adaptive chunking on) |
| `config/tracker_manual_chunking.yaml` | SAM3: single video with explicit `manual_chunk_frames` list |
| `config/sam3_hf_yolo_scan_only.yaml` | YOLO scan only, no SAM3 |
| `config/botsort.yaml` | BoT-SORT tracker parameters for use with YOLO |
| `config/bytetrack.yaml` | ByteTrack tracker parameters for use with YOLO |

## TO-DO

### High priority

- Iteratively save outputs every chunk (tracking, metrics, visualizations)
- 'Resume' a partial run (i.e. a run on a video which progressed a third way through)
- Option to add point prompts, in manual chunking mode (list of tuples; first list is positive, second is negative; if a list is None, then ignore)

### Low priority

- Cache scan results for reuse across runs with same video
- Implement config ingest for test scripts
- Method for marking output run directory as 'incomplete'
  - Possible solution: placeholder name has `_incomplete` suffix, until completed, in which case the suffix is stripped.
- Benchmark tracking performance using frame streaming vs frame preloading

## Linting

Ruff is available as a workspace dependency. **Do not run ruff automatically** — the user handles linting manually.

```sh
pixi run ruff check src/ script/
pixi run ruff format src/ script/
```

## Misc. notes

- **H.264 frame seeking is broken**: `cv2.CAP_PROP_POS_FRAMES` seeking returns wrong frames for H.264-encoded videos (only accurate at keyframes/I-frames; inter-coded P/B-frames return incorrect data). Two frame-loading functions exist in `src/utils.py`:
  - `load_video_frames_range` — Uses `CAP_PROP_POS_FRAMES` seeking. **Fast but unreliable for H.264.**
  - `load_video_frames_sequential` — Reads sequentially from frame 0. **Slower but frame-accurate** for all codecs. Used by `extract_embeddings` and the diagnostic notebook.
  - **Known impact**: The SAM3 tracking pipeline (`run_sam3_hf.py`) uses `load_video_frames_range` to load each chunk's frames, so chunks after chunk 0 may start from slightly wrong frames due to seek inaccuracy. Tracking results appear fine in practice (SAM3 is robust to small frame offsets), but this should be fixed.
- SAM3 (both HF-transformers and native) is extremely unstable when run in Jupyter notebooks and frequently crashes the kernel. Prefer running SAM3 via scripts; for notebooks, load SAM3 first, free GPU memory, then load lighter models (e.g., DINOv3).
