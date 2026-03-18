# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multi-object tracking and segmentation of chickens in video data using SAM3 (HuggingFace) with an optional YOLO scan for adaptive chunking. Processes long videos by chunking them, running text-prompted segmentation on the first chunk, then propagating tracks via point prompts across subsequent chunks.

## Environment & Setup

**Package manager**: Pixi (not pip/conda directly). Python 3.11 only.

```sh
# Fetch git submodules
git submodule update --init --recursive

# Install main environment
pixi install -e sam3-hf        # SAM3 HuggingFace (main pipeline)
pixi shell -e sam3-hf          # Enter shell
```

Other environments exist (`sam3-native`, `gs2`, `yolo`) but are not actively used. The `classifier` environment adds PyTorch Lightning, wandb, and torchmetrics for classification training. The `videoprism` environment provides JAX + VideoPrism for video embedding extraction. Platform is Linux-only (CUDA 12.6).

## Running Scripts

Scripts are run as Python modules from the project root. CUDA device is specified in the YAML config:

```sh
# Main pipeline (defaults to config/sam3_hf_config.yaml)
pixi run -e sam3-hf sam3-hf-tracker
# Custom config:
pixi run -e sam3-hf python -m script.sam3.run_sam3_hf --config config/sam3_hf_manual_chunking.yaml
```

**Pixi quoting caveat**: `pixi run -e <env> python -c "..."` breaks on spaces in paths, f-string curly braces, and other special characters. **Never use inline `-c` with pixi.** Instead, write a temporary `.py` file in `tmp/` (project-local, gitignored) and run it with `pixi run -e <env> python tmp/<script>.py`.

## Running Tests

```sh
CUDA_VISIBLE_DEVICES=1 pixi run test-sam3-hf-image  # Single image inference
CUDA_VISIBLE_DEVICES=1 pixi run test-sam3-hf-video  # Video chunking test
pixi run -e sam3-hf test_features                    # Dataset feature extraction tests (pytest)
```

SAM3 tests are standalone scripts in `src/test/`, not pytest-based. Pixi tasks invoke them as `python -m test.<name>` (the `src/` package root is on the module path). Dataset tests in `tests/` use pytest.

## Architecture

```
script/sam3/
  run_sam3_hf.py            # Config-driven tracking pipeline (main script)

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
    embeddings.py           # DINOv3 CLS-token embedding extraction from bbox crops of tracked objects
    crops.py              # Shared crop modes: crop_frame, compute_union_origin, compute_union_bbox
  classification/           # Behaviour classification package
    models.py               # Backbones: SimpleLinear, SimpleMLP, TemporalMLP, TemporalCNNv2. MODEL_REGISTRY maps names to (cls, temporal_flag).
    datamodule.py           # BehaviourDataset + BehaviourDataModule (LOVO split, segment pooling)
    trainer.py              # BehaviourClassifier LightningModule (weighted CE, AdamW, MetricCollection)
    stats.py                # LOVO aggregation: scalar summary CSV, summed confusion matrices
  debug/                    # Interactive debugging utilities and standalone grounding test script

script/
  build_dataset.py             # Labels, postprocessing, windowing → tracks + labels parquets (output to data/dataset/)
  extract_features.py          # Mask feature extraction + window summarization (CPU-only, ~2 min)
  extract_embeddings.py        # DINOv3 CLS-token embeddings from bbox crops (GPU required)
  train.py                     # Classification training CLI (PyTorch Lightning)
  compute_chunk_boundaries.py  # User script: recompute metrics + boundaries from existing yolo_tracking.parquet
  dlc2yolo/                    # DLC-to-YOLO format converter for pose dataset creation
  convert_video_clean.py       # Video format/resolution conversion utility

config/                     # YAML configs (OmegaConf)
  video_specific/            # Per-video configs with tuned manual_chunk_frames
tests/                      # pytest tests (dataset features, labels, postprocessing)
src/test/                   # Test scripts for SAM3 inference (run via pixi tasks, not pytest)
notebook/                   # Jupyter notebooks for EDA and demos
```

### Key patterns

- **Config-driven**: `run_sam3_hf.py` reads YAML configs via OmegaConf. The `_early_init()` pattern parses config and sets `CUDA_VISIBLE_DEVICES` and `PYTORCH_ALLOC_CONF` before torch is imported. A `tracking:` section overrides `Sam3VideoConfig` parameters (keep-alive, IoU thresholds, reconditioning interval, etc.).
- **Timestamped output**: Each run creates `{output_dir}/{YYYYMMDD_HHMMSS}_{job_type}/` with subdirectories for `metrics/` and `visualizations/`. Config is copied for reproducibility.
- **Loguru logging**: Console (colored) + file handler in run directory. Replaces all `print()`.
- **Chunked processing**: Long videos are split into chunks via `chunk_video_frames_adaptive()`. Initial chunk size is set by `chunk_seconds` (default 60 s); when YOLO scan data is available, each boundary is shifted within a ±`adaptive_search_window_seconds` window to the frame with the highest separation score. Chunk 0 uses `Sam3VideoModel` (text-prompted); subsequent chunks use `Sam3TrackerVideoModel` (point-prompted). Small trailing remainders (<10% of chunk size) are absorbed into the last chunk. Point prompts are extracted from previous chunk's masks via `extract_equidistant_points_from_masks()`. `find_frame_with_enough_objects()` searches backwards for a frame with enough detected objects. `max_frames_to_track` limits how many frames are processed per video.
- **Two model phases**: `Sam3VideoModel` (text→segmentation) for initialization, `Sam3TrackerVideoModel` (point→tracking) for propagation. Each chunk loads its model fresh and cleans up GPU memory afterwards (`free_gpu_memory()` with triple `gc.collect` + CUDA cache clearing).
- **Adaptive chunking (YOLO scan)**: When `use_adaptive_chunking: true`, `run_yolo_scan()` (in `src/yolo_scan.py`) runs YOLO tracking on the full video and returns a raw `yolo_df`. Analysis — `compute_yolo_per_frame_metrics()` → `identify_occlusion_periods()` → `find_high_separation_windows()` — lives in `src/chunk_boundaries.py`. `chunk_video_frames_adaptive()` refines boundaries in priority order — separation-first (highest separation_score inside a high-separation window), then occlusion avoidance (farthest from occlusion with 90%/50% directional penalties), validated against `adaptive_max_chunk_seconds`. Scan outputs saved as `yolo_tracking.parquet`, `yolo_scan_metrics.parquet`, `yolo_scan_summary.parquet`. A `yolo_scan:` config section controls model, thresholds, and tracker config. To reanalyse boundaries without re-running YOLO, use `script/compute_chunk_boundaries.py`.
- **Manual chunking**: Set `manual_chunk_frames` to a list of `[start, end]` pairs to override fixed/adaptive chunking entirely. First pair → `Sam3VideoModel`; subsequent → `Sam3TrackerVideoModel`. Disables `yolo_scan_only`/`use_adaptive_chunking` with warnings. See `build_manual_chunks()` in `src/utils.py` and `config/sam3_hf_manual_chunking.yaml`.
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

- **`extract_spatial_features`**: Batch `pycocotools.mask.area()` for mask areas, numpy array math for bbox metrics, batch `mask_util.decode()` + `scipy.ndimage.center_of_mass` for centroids. Masks grouped by `size` (pycocotools requirement) and chunked at 500 to cap memory.
- **`extract_temporal_features`**: `pandas.groupby().diff()` / `.shift()` for frame-to-frame deltas.
- **`extract_pairwise_features`**: Numpy broadcasting for `(M, M)` distance matrices per frame.
- **`summarize_features_by_window`**: Single `groupby().agg()` call with `[mean, std, min, max, median]`.

Feature set (`_FEATURE_COLS`): `mask_area`, `aspect_ratio`, `velocity`, `area_change_rate`, `min_dist_to_other`, `mean_dist_to_other`. Additionally `bbox_area` is output by spatial features as a debugging column (not in `_FEATURE_COLS`) for detecting saltatory bbox spikes.

### Embeddings module (`src/dataset/embeddings.py`)

Extracts DINOv3 CLS-token embeddings from bbox crops of tracked objects. Output is `{(video_id, bird_id, window): Tensor(F_w, D)}` — one variable-length sequence per window, aligned with the feature summarization grouping. `D` depends on the model (768 for ViT-B, 1024 for ViT-L), read from `model.config.hidden_size`. Crops are plain bbox cutouts (no mask-based background attenuation). Bboxes are clamped to frame dimensions to handle SAM3 bbox overshoot (known upstream issue where center+size→corner conversion produces coordinates slightly outside the image). Extraction uses **bfloat16** (float16 produces all-NaN with DINOv3 ViT-L). Uses `load_video_frames_sequential` for frame-accurate loading (see Misc. notes on H.264 seeking). Extraction script: `script/extract_embeddings.py` (GPU required, run via `pixi run -e sam3-hf extract-embeddings`).

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
  - `data/postprocessing/` — Version-controlled copies of per-video JSONs (`tracking_postprocessing.json`, `tracking_issues.json`, `bird_info.json`, `chunk_info.json`), organized as `{run_id}/{video_subdir}/`
- `ext-data/` — Symlink to `/mnt/birds/rebecca2025/` (longer videos, output results, image sequences)
  - `ext-data/test/batch_mode_test_set/` — 3 × 2-min clips (`test_video_1/2/3.mp4`) for batch mode testing
- `video-data/` — Symlink to `/mnt/birds/rebecca2025/raw` (raw video files)
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
pixi run -e sam3-hf build_dataset

# 2. Mask features (slow, ~2 min, CPU-only)
pixi run -e sam3-hf extract_features

# 3. DINOv3 embeddings (slow, GPU required)
pixi run -e sam3-hf extract_embeddings \
    --video-dir video-data/batch
```

All scripts default to `data/tracking/` (input) and `data/dataset/` (output). Override with `--tracking-dir` / `--dataset-dir` / `--output-dir`.

### Step 1: `build_dataset`

`script/build_dataset.py` discovers all `tracking_outputs.parquet` files recursively under `data/tracking/` (across multiple tracking runs), parses labels from Excel, detects issues, applies postprocessing, assigns windows, and filters incomplete windows. On first run it generates `tracking_postprocessing.json` templates; manually fill in `to` (id_switch) and `tracking_id` (id_match) fields, then rerun.

Per-video JSON files (`tracking_postprocessing.json`) support three entry types:

- **`trim`** — Remove track rows in a frame range, optionally scoped to one ID. Causes: `overlap`, `low_score`, `merged_object`.
- **`id_switch`** — Remap `from` ID to `to` ID for all rows before `frame` (merges split tracks).
- **`id_match`** — Rename tracker `tracking_id` to real `protocol_id` (bird identity).

Output: `tracks.parquet` + `labels.parquet` in `data/dataset/`. JSON files are copied to `data/postprocessing/` for version control.

### Step 2: `extract_features`

`script/extract_features.py` reads `tracks.parquet` from `data/dataset/`, decodes RLE masks, computes per-frame spatial/temporal/pairwise features, then summarizes per window. Also builds temporal feature tensors (per-frame features binned per window, same key format as embeddings).

Feature set (19 features): `mask_area`, `aspect_ratio`, `velocity`, `acceleration`, `velocity_autocorr`, `area_change_rate`, `min_dist_to_other`, `mean_dist_to_other`, plus additional spatial/temporal derivatives.

Output: `features_all.parquet` (per-frame) + `features_windowed.parquet` (per-window summaries) + `features_binned.pt` (temporal feature tensors keyed by `(video_id, bird_id, window)`).

### Step 3: `extract_embeddings`

`script/extract_embeddings.py` reads `tracks.parquet` from `data/dataset/`, crops bbox regions from video frames, and extracts DINOv3 CLS-token embeddings per (video_id, bird_id, window).

```sh
pixi run -e sam3-hf python -m script.extract_embeddings \
    --video-dir data/video --device 0

# Alternative backbone (ViT-B)
pixi run -e sam3-hf python -m script.extract_embeddings \
    --video-dir data/video --device 0 \
    --model-name facebook/dinov3-vitb16-pretrain-lvd1689m

# Custom resolution / bbox scale / LoRA adapter
pixi run -e sam3-hf python -m script.extract_embeddings \
    --video-dir data/video --device 0 \
    --resolution 256 --bbox-scale 1.25 \
    --lora-weights data/eval/<timestamp>_finetune/lora_adapter
```

Output filename encodes variant: `embeddings.pt` (default), `embeddings_vitb.pt`, `embeddings_125.pt`, `embeddings_r256.pt`, `embeddings_lora.pt`. Dict keyed by `(video_id, bird_id, window)` with `Tensor(F_w, D)` values.

### Step 3b: Video embeddings (`extract_embeddings_vjepa2` / `extract_embeddings_videoprism`)

Two scripts for video-model embedding extraction. Both use shared `CROP_MODES` from `src/dataset/crops.py` (`--crop-mode`: bbox, plain256, union512, darken512, roi512).

- **`script/extract_embeddings_vjepa2.py`** (sam3-hf env) — V-JEPA 2 embeddings. Feeds K frames as a video clip through the backbone. Supports `--temporal` (per-timestep spatial-mean-pool).
- **`script/extract_embeddings_videoprism.py`** (videoprism env, JAX) — VideoPrism embeddings. Supports `--temporal`, `--raw` (full patch tokens).

```sh
# V-JEPA 2 temporal (sam3-hf env)
pixi run -e sam3-hf python -m script.extract_embeddings_vjepa2 \
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

# LoRA-finetuned embeddings
pixi run -e classifier train --model mlp --input features+embeddings_lora --exclude social

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

Input is specified separately via `--input`: `features`, `embeddings`, `embeddings_250`, `features+embeddings`, `features+embeddings_lora`, etc. Any `embeddings[_variant]` maps to `{name}.pt` in the dataset dir. Multiple embedding files can be concatenated: `features+embeddings+embeddings_union512` loads both and concatenates along the feature dimension (multi-scale). Non-temporal models mean-pool embeddings and use windowed feature stats; temporal models segment-pool embeddings and use binned feature tensors. Temporal models also support **hybrid mode**: e.g. `--model temporal_mlp --input features+embeddings` passes temporal embeddings through the backbone and concatenates flat windowed features before the classification head.

Additional training args: `--dropout` (override model dropout), `--d-hidden` (override hidden dim), `--label-smoothing` (CE label smoothing), `--feature-dropout` (dropout on flat features only), `--n-segments` (temporal bins, default 12).

### Classification results (2026-03-17)

Dataset: 3 tracking runs (incl. re-tracked C1G1), 15 videos, 3-class (excluding social — only 67 samples, 0% recall). LOVO with group-matched val rotation (each video is val at most once).

Training config: 5 fixed epochs, inv-sqrt class weights, label smoothing 0.1, best val_loss checkpoint, no early stopping. See `notes/ablation_features.md` and `notes/ablation_embeddings.md` for full ablation studies.

**Best results** (v2 val rotation, proper group-matched):

| Model | Input | Mean Test F1 | Pooled Test F1 | Run ID | Notes |
|-------|-------|-------------|----------------|--------|-------|
| TemporalCNNv2 | feat + multi-scale DINOv3 (bbox+512) | 0.716 | 0.737 | `20260315_180947` | **best overall**, K=24 |
| TemporalCNNv2 | feat + multi-scale DINOv3 (bbox+512) | 0.716 | 0.735 | `20260315_174859` | K=12 |
| Temporal MLP | feat + multi-scale DINOv3 (bbox+512) | 0.714 | 0.732 | `20260315_173902` | |
| BiGRU | feat + multi-scale + feat_drop=0.2 | 0.718 | 0.732 | `20260315_175138` | |
| Temporal MLP | feat + DINOv3 ViT-L, dropout=0.0 | 0.714 | 0.731 | `20260315_123332` | best single-scale |
| Temporal GRU | feat + DINOv3 ViT-L, dropout=0.0 | 0.714 | 0.727 | `20260315_124030` | |
| XGBoost | features (max_depth=3) | 0.699 | 0.718 | `20260315_115343` | |
| MLP | features | 0.693 | 0.711 | `20260315_115542` | |
| V-JEPA2 | feat + SSv2 pooler (MLP) | 0.707 | 0.722 | `20260314_134337` | best video model |
| VP-Base | feat + temporal | 0.705 | 0.714 | `20260314_145632` | |

**Key findings**:

- **Best result: 0.737 pooled F1** — TemporalCNNv2 (GELU bottleneck + single conv) on multi-scale DINOv3 embeddings (tight bbox + union512 crops) with K=24 temporal segments.
- **Multi-scale embeddings help**: concatenating tight bbox CLS + 512-crop CLS adds spatial context that improves worm detection. 512 alone is terrible (0.569) but complementary with tight bbox.
- **Dropout 0.0 is optimal**: models were over-regularized at default dropout=0.3. Removing dropout gained +0.010 pooled.
- **LoRA fine-tuning doesn't help**: LOVO LoRA (0.699) matches frozen embeddings. Biased LoRA gains (0.766) were entirely data leakage.
- **Video models (V-JEPA 2, VideoPrism) underperform DINOv3**: domain shift from full-scene pretraining to small bbox crops. Best video model: V-JEPA 2 at 0.722 pooled (SSv2 pooler).
- **Raw token classifiers (PriVi, attentive pooler) fail**: too many tokens, too few training samples.
- **Worm class is kinematically incoherent**: mixes stationary pecking (looks like none) and active running/chasing (looks like locomotor).
- XGBoost max_depth=3 (0.718) is a strong features-only baseline.

Previous best (old val rotation): Temporal GRU + feat + DINOv3 ViT-L = 0.732 pooled (inflated by biased val splits).

## Utilities

### Recomputing Scan Results

Recompute scan metrics and chunk boundaries from an existing `yolo_tracking.parquet` without re-running YOLO inference:

```sh
pixi run -e sam3-hf python -m script.compute_chunk_boundaries \
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
pixi run -e sam3-hf python -m script.sam3.run_sam3_hf --config config/sam3_hf_yolo_scan_only.yaml
```

Note: the pixi task `yolo-prescan` exists but references a non-existent config (`prescan_only.yaml`); use the command above instead.

Outputs saved to `{output_dir}/{timestamp}_yolo_scan/`.

### Config Reference

| Config | Purpose |
|--------|---------|
| `config/sam3_hf_config.yaml` | Main production config (single video, adaptive chunking on) |
| `config/sam3_hf_manual_chunking.yaml` | SAM3-HF: single video with explicit `manual_chunk_frames` list |
| `config/sam3_hf_yolo_scan_only.yaml` | YOLO scan only, no SAM3 |
| `config/gs2_manual_chunking.yaml` | GS2: single video with explicit `manual_chunk_frames` list |
| `config/video_specific/*.yaml` | Per-video configs with tuned chunk boundaries (e.g., C1G1 day 28, C5G2 day 28) |

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
  - `load_video_frames_range` — Uses `CAP_PROP_POS_FRAMES` seeking. **Fast but unreliable for H.264.** Still used by `run_sam3_hf.py` (per-chunk loading at line 716) and `run_gs2_manual_chunking.py`.
  - `load_video_frames_sequential` — Reads sequentially from frame 0. **Slower but frame-accurate** for all codecs. Used by `extract_embeddings` and the diagnostic notebook.
  - **Known impact**: The SAM3 tracking pipeline (`run_sam3_hf.py`) uses `load_video_frames_range` to load each chunk's frames, so chunks after chunk 0 may start from slightly wrong frames due to seek inaccuracy. Tracking results appear fine in practice (SAM3 is robust to small frame offsets), but this should be fixed.
- SAM3 (both HF-transformers and native) is extremely unstable when run in Jupyter notebooks and frequently crashes the kernel. Prefer running SAM3 via scripts; for notebooks, load SAM3 first, free GPU memory, then load lighter models (e.g., DINOv3).
- YOLO tracker configs live in `data/yolo/` (botsort, bytetrack variants); selected via `yolo_scan.tracker_config` in the YAML config.
