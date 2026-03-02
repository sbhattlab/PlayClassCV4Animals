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

Other environments exist (`sam3-native`, `gs2`, `yolo`) but are not actively used. Platform is Linux-only (CUDA 12.6).

## Running Scripts

Scripts are run as Python modules from the project root. CUDA device is specified in the YAML config:

```sh
# Main pipeline (defaults to config/sam3_hf_config.yaml)
pixi run -e sam3-hf sam3-hf-tracker
# Custom config:
pixi run -e sam3-hf python -m script.sam3.run_sam3_hf --config config/sam3_hf_manual_chunking.yaml
```

## Running Tests

```sh
CUDA_VISIBLE_DEVICES=1 pixi run test-sam3-hf-image  # Single image inference
CUDA_VISIBLE_DEVICES=1 pixi run test-sam3-hf-video  # Video chunking test
```

Tests are standalone scripts in `src/test/`, not pytest-based. Pixi tasks invoke them as `python -m test.<name>` (the `src/` package root is on the module path).

## Architecture

```
script/sam3/
  run_sam3_hf.py            # Config-driven tracking pipeline (main script)

src/
  utils.py                  # Config/logging/output dirs, chunking, parquet export, video annotation
  processing.py             # Mask/bbox/point extraction, tracking output normalization
  grounding.py              # Text-prompt grounding, best-frame selection, ID matching
  metrics.py                # Tracking metrics: mask-based, per-frame, per-id, per-run, summary
  viz.py                    # Visualizations: ID timeline, dashboard, score plots, mask evolution, prompt points
  chunk_boundaries.py       # Per-frame metrics, occlusion detection, separation windows, adaptive chunking
  yolo_scan.py              # YOLO inference only (run_yolo_scan); re-exports src.chunk_boundaries for compat
  ethogram.py               # Behavior label parsing from Excel registration protocols
  dataset/                  # Dataset construction package
    __init__.py             # Package marker (no re-exports; import from submodules directly)
    utils.py                # Shared helpers: fmt_time, _decode_rle_mask
    tracking_issues.py      # ID switch detection, overlap detection, ID remapping, overlap removal
    labels.py               # Behaviour label parsing from Excel registration protocols
    features.py             # Handcrafted mask features: spatial, temporal, pairwise, summarization
    embeddings.py           # DINOv3 CLS-token embedding extraction from tracked objects
    base.py                 # Glue: process_tracks (overlap removal + ID merge)
  debug/                    # Interactive debugging utilities and standalone grounding test script

script/
  preprocess.py                # Build tracking_issues.json per subdir, then build dataset (single-pass)
  compute_chunk_boundaries.py  # User script: recompute metrics + boundaries from existing yolo_tracking.parquet
  dlc2yolo/                    # DLC-to-YOLO format converter for pose dataset creation
  convert_video_clean.py       # Video format/resolution conversion utility

config/                     # YAML configs (OmegaConf)
  video_specific/            # Per-video configs with tuned manual_chunk_frames
src/test/                   # Test scripts (run via pixi tasks)
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

### Output schemas

- **`tracking_outputs.parquet`**: One row per (frame, object). MultiIndex `["frame_idx", "object_id"]`. Includes bbox, RLE mask, scores, tracker_score, chunk_idx, model_type.
- **`chunk_info.json`**: Per-chunk metadata: frame range, model type, prompt points, timing. Keys: `grounding_used`, `grounding_source_frame_idx`, `grounding_num_objects`, `grounding_fallback_reason`.
- **YOLO scan outputs**: `yolo_tracking.parquet`, `metrics/yolo_scan_metrics.parquet`, `metrics/yolo_scan_summary.parquet`, `visualizations/yolo_scan_overview.png`.
- **Metrics parquets**: `per_frame_metrics`, `per_id_metrics`, `summary_metrics`.
- **Run directory**: `{timestamp}_{job_type}/` with config copy, log, parquets, `metrics/`, `visualizations/`. Batch mode: `{timestamp}_{job_type}/{sanitized_stem}/` per video.

## Data Layout

- `data/` — Small test data (images, short video clips, DLC annotations, ethogram parquets)
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

Ruff is available as a workspace dependency:

```sh
pixi run ruff check src/ script/
pixi run ruff format src/ script/
```

## Misc. notes

- SAM3 (both HF-transformers and native) is extremely unstable when run in Jupyter notebooks and frequently crashes the kernel. Prefer running SAM3 via scripts; for notebooks, load SAM3 first, free GPU memory, then load lighter models (e.g., DINOv3).
- YOLO tracker configs live in `data/yolo/` (botsort, bytetrack variants); selected via `yolo_scan.tracker_config` in the YAML config.
