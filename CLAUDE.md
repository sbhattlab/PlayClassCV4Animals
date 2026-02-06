# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multi-object tracking and segmentation of chickens in video data using SAM3, Grounded-SAM-2, and YOLO. The system processes long videos by chunking them, running text-prompted segmentation on the first chunk, then propagating tracks via point prompts across subsequent chunks.

## Environment & Setup

**Package manager**: Pixi (not pip/conda directly). Python 3.11 only.

```sh
# Fetch git submodules (Grounded-SAM-2 fork)
git submodule update --init --recursive

# Install an environment
pixi install -e sam3-hf        # SAM3 HuggingFace (main pipeline)
pixi install -e sam3-native    # SAM3 native (Facebook Research)
pixi run -e gs2 setup-gs2      # Grounded-SAM-2 (downloads checkpoints + installs)

# Enter an environment shell
pixi shell -e sam3-hf
```

Environments: `sam3-hf`, `sam3-hf-macos`, `sam3-native`, `gs2`, `gs2-macos`. CUDA environments are Linux-only; macOS variants use CPU/MPS.

## Running Scripts

Scripts are run as Python modules from the project root. CUDA device is specified in the YAML config (no env var prefix needed):

```sh
# Main SAM3 pipeline (config-driven, reads CUDA_VISIBLE_DEVICES from YAML)
python -m script.sam3.run_sam3_hf --config config/sam3_hf_config.yaml

# Grounded-SAM-2
python -m script.gs2.chicken_tracking_demo --gs2-repo-path Grounded-SAM-2-fork -i ext-data/imgs/imgs_1min --text ".chicken.bird."
```

## Running Tests

```sh
CUDA_VISIBLE_DEVICES=1 pixi run test-sam3-hf-image     # Single image inference (sam3-hf env)
CUDA_VISIBLE_DEVICES=1 pixi run test-sam3-hf-video     # Video chunking test (sam3-hf env)
CUDA_VISIBLE_DEVICES=1 pixi run test-sam3-native-video  # Native SAM3 video test (sam3-native env)
CUDA_VISIBLE_DEVICES=1 pixi run test-gs2                # Grounded-SAM-2 chicken tracking demo (gs2 env)
```

Tests are standalone scripts in `test/`, not pytest-based.

## Architecture

```
script/                       # Executable scripts (run as python -m script.X.Y)
  sam3/                       # SAM3 pipelines
    run_sam3_hf.py            # Config-driven tracking pipeline (main script)
  gs2/                        # Grounded-SAM-2 pipelines
    chicken_tracking_demo.py  # GS2 image-sequence tracking
  yolo/                       # YOLO format conversion and visualization

src/                          # SAM3 library modules
  utils.py                    # Config/logging/output dirs, parquet export, video annotation
  metrics.py                  # Tracking metrics: mask-based, per-frame, per-id, per-run, summary
  viz.py                      # Matplotlib visualizations: ID timeline, metrics dashboard, score plots

config/                       # YAML configs (OmegaConf) for pipeline runs
test/                         # Test scripts (run via pixi tasks)
notebook/                     # Jupyter notebooks for EDA and demos
```

### Key patterns

- **Config-driven**: `run_sam3_hf.py` reads YAML configs via OmegaConf (`config/sam3_hf_config.yaml`). The `_early_init()` pattern parses config and sets `CUDA_VISIBLE_DEVICES` and `PYTORCH_ALLOC_CONF` before torch is imported. A `tracking:` section overrides `Sam3VideoConfig` parameters (keep-alive, IoU thresholds, reconditioning interval, etc.). A `metrics:` section controls occlusion/clustering thresholds.
- **Timestamped output**: Each run creates `{output_dir}/{YYYYMMDD_HHMMSS}_{job_type}/` with subdirectories for `metrics/` and `visualizations/`. Config is copied for reproducibility.
- **Loguru logging**: Console (colored) + file handler in run directory. Replaces all `print()`.
- **Chunked processing**: Long videos are split into chunks via `chunk_video_frames_dual()`. Chunk 0 uses `Sam3VideoModel` (text-prompted, shorter: `video_model_chunk_seconds`). Subsequent chunks use `Sam3TrackerVideoModel` (point-prompted, longer: `tracker_chunk_seconds`). Small trailing remainders (<10% of chunk size) are absorbed into the last chunk. Point prompts are extracted from previous chunk's masks via `sample_points_from_masks()`. `find_frame_with_enough_objects()` searches backwards for a frame with enough detected objects. Object identities are preserved across chunks by passing the same object IDs.
- **Two model phases**: `Sam3VideoModel` (text→segmentation) for initialization, `Sam3TrackerVideoModel` (point→tracking) for propagation. Each chunk loads its model fresh and cleans up GPU memory afterwards (`free_gpu_memory()` with triple `gc.collect` + CUDA cache clearing).
- **Known issue — VideoModel→TrackerModel transition**: Tracking quality degrades across the chunk boundary when handing off from `Sam3VideoModel` to `Sam3TrackerVideoModel`. A suspected cause is the `custom_resolution` override (e.g. 560px) distorting the point prompts or mask quality for the tracker. Reverting to native resolution for the tracker chunks may help, but more testing is needed to verify.
- **Device selection**: `Accelerator().device` from HuggingFace Accelerate.

### Metrics module (`src/metrics.py`)

The simplified metrics module computes tracking quality post-hoc from `outputs_per_frame` (the SAM3 output dict). Three levels of analysis:

- **Per-frame** (`compute_per_frame_metrics`): Mask-based spatial metrics per frame — pairwise mask IoU, centroids, centroid distances, clustering coefficient, mask area stats, occlusion flags. Works directly on `outputs_per_frame` including masks.
- **Per-ID** (`compute_per_id_metrics`): Object lifecycle — runs, gaps, coverage, identity-aware `self_iou` and greedy `spatial_continuity_iou`. Box-based, via `_normalize_frame_dict`.
- **Per-run** (`compute_per_run_metrics`): Same stats scoped to each contiguous segment, plus `mean_tracker_score`.
- **Summary** (`compute_summary_metrics`): Aggregates — continuity, fragmentation, ID switch rate. Optionally includes occlusion-aware ID switch count when `per_frame_metrics` is provided.
- **Occlusion-aware ID switch detection** (`detect_identity_switches`): Heuristic that flags ID changes only when a recent high-occlusion event occurred within a sliding window.

Raw model fields (`obj_id_to_tracker_score`, `removed_obj_ids`, `suppressed_obj_ids`) are preserved from the inference loop through to metrics and parquet output.

### Visualizations module (`src/viz.py`)

Auto-generated on each run, saved to `run_dir/visualizations/`:

- **ID timeline** (`plot_id_timeline`): Eventplot showing object ID existence over time, bars color-coded by tracker score (RdYlGn), with occlusion/count-change trouble-spot overlays.
- **Per-frame dashboard** (`plot_per_frame_dashboard`): 5-panel timeseries — object count, max mask IoU, min centroid distance, clustering coefficient, mean mask area with min/max band.
- **Per-ID scores** (`plot_per_id_scores`): Tracker score over time per object ID.

All plots use MM:SS x-axis when FPS is available.

### Output schemas & data structures

- **Model outputs**: See `Sam3VideoSegmentationOutput` in HF Transformers for raw fields. Post-processing in `run_sam3_hf.py:_process_video_chunk()` and `_process_tracker_chunk()`. Key difference: VideoModel outputs GPU tensors with `removed_obj_ids`/`suppressed_obj_ids`; TrackerModel outputs numpy arrays with sigmoid scores.
- **`tracking_outputs.parquet`**: Written by `src.utils:process_tracking_outputs()`. One row per (frame, object). MultiIndex `["frame_idx", "object_id"]`. Includes bbox, RLE mask, scores, tracker_score, chunk_idx, model_type.
- **`chunk_info.json`**: Written by `run_sam3_hf.py`. Per-chunk metadata: frame range, model type, prompt points, timing.
- **Metrics parquets**: Written by `src.metrics`. `per_frame_metrics` (spatial/occlusion per frame), `per_id_metrics` (per contiguous run per ID), `summary_metrics` (single-row aggregates).
- **Run directory**: `{timestamp}_{job_type}/` containing config copy, log, chunk_info.json, annotated_video.mp4, tracking_outputs.parquet, `metrics/`, `visualizations/`.

## Data Layout

- `data/` — Small test data (images, short video clips, DLC annotations, ethogram parquets)
- `ext-data/` — Symlink to `/mnt/birds/rebecca2025/` (longer videos, output results, image sequences)
- `video-data/` — Symlink to `/mnt/birds/rebecca2025/raw` (raw video files)
- `Grounded-SAM-2-fork/` — Git submodule with SAM2/Grounding DINO code and checkpoints

## Key Dependencies

- PyTorch 2.9.1 (CUDA 12.6 on Linux)
- HuggingFace Transformers v5.0.0rc2 (installed from git), Accelerate (device management)
- supervision (annotation/visualization), loguru (logging), OmegaConf (config)
- pycocotools (RLE mask encoding for parquet storage), matplotlib (visualizations)
