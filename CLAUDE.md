# CLAUDE.md

## Project Overview

Multi-object tracking and segmentation of chickens in video data using SAM3 (HuggingFace) with an optional YOLO pre-scan for adaptive chunking. Processes long videos by chunking them, running text-prompted segmentation on the first chunk, then propagating tracks via point prompts across subsequent chunks.

## Environment & Setup

**Package manager**: Pixi (not pip/conda directly). Python 3.11 only.

```sh
# Fetch git submodules
git submodule update --init --recursive

# Install main environment
pixi install -e sam3-hf        # SAM3 HuggingFace (main pipeline)
pixi shell -e sam3-hf          # Enter shell
```

Other environments exist (`sam3-native`, `gs2`, macOS variants) but are not actively used. CUDA environments are Linux-only.

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

Tests are standalone scripts in `test/`, not pytest-based.

## Architecture

```
script/sam3/
  run_sam3_hf.py            # Config-driven tracking pipeline (main script)

src/
  utils.py                  # Config/logging/output dirs, chunking, parquet export, video annotation
  metrics.py                # Tracking metrics: mask-based, per-frame, per-id, per-run, summary
  viz.py                    # Visualizations: ID timeline, dashboard, score plots, mask evolution, prompt points
  yolo_prescan.py           # YOLO-based occlusion prescan: inference, per-frame metrics, transition detection

config/                     # YAML configs (OmegaConf)
test/                       # Test scripts (run via pixi tasks)
notebook/                   # Jupyter notebooks for EDA and demos
```

### Key patterns

- **Config-driven**: `run_sam3_hf.py` reads YAML configs via OmegaConf. The `_early_init()` pattern parses config and sets `CUDA_VISIBLE_DEVICES` and `PYTORCH_ALLOC_CONF` before torch is imported. A `tracking:` section overrides `Sam3VideoConfig` parameters (keep-alive, IoU thresholds, reconditioning interval, etc.).
- **Timestamped output**: Each run creates `{output_dir}/{YYYYMMDD_HHMMSS}_{job_type}/` with subdirectories for `metrics/` and `visualizations/`. Config is copied for reproducibility.
- **Loguru logging**: Console (colored) + file handler in run directory. Replaces all `print()`.
- **Chunked processing**: Long videos are split into chunks via `chunk_video_frames_dual()`. Chunk 0 uses `Sam3VideoModel` (text-prompted, shorter: `video_model_chunk_seconds`). Subsequent chunks use `Sam3TrackerVideoModel` (point-prompted, longer: `tracker_chunk_seconds`). Small trailing remainders (<10% of chunk size) are absorbed into the last chunk. Point prompts are extracted from previous chunk's masks via `extract_equidistant_points_from_masks()`. `find_frame_with_enough_objects()` searches backwards for a frame with enough detected objects. `max_frames_to_track` limits how many frames are processed per video.
- **Two model phases**: `Sam3VideoModel` (text→segmentation) for initialization, `Sam3TrackerVideoModel` (point→tracking) for propagation. Each chunk loads its model fresh and cleans up GPU memory afterwards (`free_gpu_memory()` with triple `gc.collect` + CUDA cache clearing).
- **Adaptive chunking (YOLO pre-scan)**: When `use_adaptive_chunking: true`, `run_yolo_prescan()` runs YOLO tracking on the full video, computes per-frame spatial metrics, identifies occlusion periods, and extracts transition frames. `chunk_video_frames_adaptive()` scores candidate boundaries by distance to nearest occlusion (90% penalty if occlusion is ahead, 50% if just ended), validated against `adaptive_min/max_chunk_seconds`. Pre-scan outputs saved as `yolo_tracking.parquet`, `yolo_prescan_metrics.parquet`, `yolo_prescan_summary.parquet`. A `yolo_prescan:` config section controls model, thresholds, and tracker config.
- **Manual chunking**: Set `manual_chunk_frames` to a list of `[start, end]` pairs to override fixed/adaptive chunking entirely. First pair → `Sam3VideoModel`; subsequent → `Sam3TrackerVideoModel`. Disables `prescan_only`/`use_adaptive_chunking` with warnings. See `build_manual_chunks()` in `src/utils.py` and `config/sam3_hf_manual_chunking.yaml`.
- **Batch processing**: Set `video_dir` instead of `video_path` to process all videos in a directory. Each video gets its own subdirectory under a shared timestamped batch dir. Errors caught per-video. `manual_chunk_frames` accepts a dict keyed by **basename** (e.g. `"video1.mp4": [[0,375],...]`) for per-video boundaries; unlisted videos use fixed chunking. See `config/batch_mode.yaml`.
- **Device selection**: `Accelerator().device` from HuggingFace Accelerate.

### Metrics module (`src/metrics.py`)

Computes tracking quality post-hoc from `outputs_per_frame`. Four levels:

- **Per-frame** (`compute_per_frame_metrics`): Pairwise mask IoU, centroids, clustering coefficient, mask area stats, occlusion flags.
- **Per-run** (`compute_per_run_metrics`): Per contiguous segment per ID — runs, gaps, coverage, `mean_tracker_score`.
- **Summary** (`compute_summary_metrics`): Continuity, fragmentation, ID switch rate; occlusion-aware ID switch count when `per_frame_metrics` provided.
- **Occlusion-aware ID switch detection** (`detect_identity_switches`): Flags ID changes only when a recent high-occlusion event occurred within a sliding window.

### Visualizations module (`src/viz.py`)

Auto-generated on each run, saved to `run_dir/visualizations/`: ID timeline (tracker score color-coded), per-frame dashboard (5-panel timeseries), per-ID tracker scores, mask evolution at chunk boundaries, prompt points at boundaries, YOLO prescan overview (when adaptive chunking enabled). All plots use MM:SS x-axis when FPS is available.

### Output schemas

- **`tracking_outputs.parquet`**: One row per (frame, object). MultiIndex `["frame_idx", "object_id"]`. Includes bbox, RLE mask, scores, tracker_score, chunk_idx, model_type.
- **`chunk_info.json`**: Per-chunk metadata: frame range, model type, prompt points, timing.
- **YOLO prescan outputs**: `yolo_tracking.parquet`, `metrics/yolo_prescan_metrics.parquet`, `metrics/yolo_prescan_summary.parquet`, `visualizations/yolo_prescan_overview.png`.
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
- ultralytics (YOLO pre-scan)
- scikit-learn (legacy KMeans pre-scan in `src/utils.py`, used only by `viz.py`)

## Utilities

### Parameter Sensitivity Testing

Recompute prescan metrics and chunk boundaries from an existing `yolo_tracking.parquet` without re-running YOLO inference:

```python
yolo_df = pd.read_parquet(run_dir / "yolo_tracking.parquet")
from src.yolo_prescan import compute_yolo_prescan_results
prescan_results = compute_yolo_prescan_results(
    yolo_df, fps=25.0,
    occlusion_iou_threshold=0.15,
    high_occlusion_threshold=0.3,
)
```

Default thresholds (`occlusion_iou_threshold: 0.15`, `high_occlusion_threshold: 0.3`) provide balanced detection. See commit 498cd11 for full regeneration pattern.

### Prescan-Only Mode

```sh
pixi run yolo-prescan  # uses config/prescan_only.yaml
```

Outputs saved to `{output_dir}/{timestamp}_yolo_prescan/`.

### Config Reference

| Config | Purpose |
|--------|---------|
| `config/sam3_hf_config.yaml` | Main production config (single video, adaptive chunking on) |
| `config/sam3_hf_manual_chunking.yaml` | SAM3-HF: single video with explicit `manual_chunk_frames` list |
| `config/gs2_manual_chunking.yaml` | GS2: single video with explicit `manual_chunk_frames` list |
| `config/batch_mode.yaml` | Batch mode (`video_dir`) with per-video `manual_chunk_frames` dict |
| `config/prescan_only.yaml` | YOLO pre-scan only, no SAM3 |

## TO-DO

### High priority
- Iteratively save outputs every chunk (tracking, metrics, visualizations)
- 'Resume' a partial run (i.e. a run on a video which progressed a third way through)
- Remove inactive code: remove 'random' point sampling legacy method
- Option to add point prompts, in manual chunking mode (list of tuples; first list is positive, second is negative; if a list is None, then ignore)

### Low priority
- Cache prescan results for reuse across runs with same video
- Implement config ingest for test scripts
- Method for marking output run directory as 'incomplete' 
  - Possible solution: placeholder name has `_incomplete` suffix, until completed, in which case the suffix is stripped.
- `video_model_chunk_seconds` and `tracker_chunk_seconds` config keys should *not* be required - currently, failure to provide them during e.g. manual mode causes run to fail
- Benchmark tracking performance using frame streaming vs frame preloading 

# Misc. notes
- For whatever reason, Sam3 both hf-transformers and the native implementations are extremely unstable when being run in Jupyter Notebooks, and frequently kernel