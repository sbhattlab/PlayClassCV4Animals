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
python -m script.sam3.demo --config config/sam3_hf_config.yaml

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
    demo.py                   # Config-driven tracking pipeline (active development on dev branch)
    metrics.py                # Tracking metrics: mask-based, per-frame, per-id, per-run, summary
    utils.py                  # Config/logging/output dirs, parquet export, video annotation
    viz.py                    # Matplotlib visualizations: ID timeline, metrics dashboard, score plots
    run_sam3_hf_chunking.py   # Original chunked pipeline (being replaced by demo.py)
  gs2/                        # Grounded-SAM-2 pipelines
    chicken_tracking_demo.py  # GS2 image-sequence tracking
  yolo/                       # YOLO format conversion and visualization

src/                          # Original core library modules (legacy, being superseded by script/sam3/)
  utils.py                    # Config (OmegaConf), device selection, mask utils, video I/O, rendering
  sam3_hf.py                  # SAM3 HuggingFace pipeline: chunking, text→segmentation, point→tracking
  tracking_metrics.py         # Original per-frame/chunk metrics (dataclass-based)

config/                       # YAML configs (OmegaConf) for pipeline runs
test/                         # Test scripts (run via pixi tasks)
notebook/                     # Jupyter notebooks for EDA and demos
```

### Key patterns

- **Config-driven**: `demo.py` reads YAML configs via OmegaConf (`config/sam3_hf_config.yaml`). The `_early_init()` pattern parses config and sets `CUDA_VISIBLE_DEVICES` and `PYTORCH_ALLOC_CONF` before torch is imported. A `tracking:` section overrides `Sam3VideoConfig` parameters (keep-alive, IoU thresholds, reconditioning interval, etc.). A `metrics:` section controls occlusion/clustering thresholds.
- **Timestamped output**: Each run creates `{output_dir}/{YYYYMMDD_HHMMSS}_{job_type}/` with subdirectories for `metrics/` and `visualizations/`. Config is copied for reproducibility.
- **Loguru logging**: Console (colored) + file handler in run directory. Replaces all `print()`.
- **Chunked processing**: Long videos are split into chunks via `chunk_video_frames_dual()`. Chunk 0 uses `Sam3VideoModel` (text-prompted, shorter: `video_model_chunk_seconds`). Subsequent chunks use `Sam3TrackerVideoModel` (point-prompted, longer: `tracker_chunk_seconds`). Small trailing remainders (<10% of chunk size) are absorbed into the last chunk. Point prompts are extracted from previous chunk's masks via `sample_points_from_masks()`. `find_frame_with_enough_objects()` searches backwards for a frame with enough detected objects. Object identities are preserved across chunks by passing the same object IDs.
- **Two model phases**: `Sam3VideoModel` (text→segmentation) for initialization, `Sam3TrackerVideoModel` (point→tracking) for propagation. Each chunk loads its model fresh and cleans up GPU memory afterwards (`free_gpu_memory()` with triple `gc.collect` + CUDA cache clearing).
- **Known issue — VideoModel→TrackerModel transition**: Tracking quality degrades across the chunk boundary when handing off from `Sam3VideoModel` to `Sam3TrackerVideoModel`. A suspected cause is the `custom_resolution` override (e.g. 560px) distorting the point prompts or mask quality for the tracker. Reverting to native resolution for the tracker chunks may help, but more testing is needed to verify.
- **Device selection**: `Accelerator().device` from HuggingFace Accelerate (replaces manual CUDA > MPS > CPU logic in `src/utils.py`).

### Metrics module (`script/sam3/metrics.py`)

The simplified metrics module computes tracking quality post-hoc from `outputs_per_frame` (the SAM3 output dict). Three levels of analysis:

- **Per-frame** (`compute_per_frame_metrics`): Mask-based spatial metrics per frame — pairwise mask IoU, centroids, centroid distances, clustering coefficient, mask area stats, occlusion flags. Works directly on `outputs_per_frame` including masks.
- **Per-ID** (`compute_per_id_metrics`): Object lifecycle — runs, gaps, coverage, identity-aware `self_iou` and greedy `spatial_continuity_iou`. Box-based, via `_normalize_frame_dict`.
- **Per-run** (`compute_per_run_metrics`): Same stats scoped to each contiguous segment, plus `mean_tracker_score`.
- **Summary** (`compute_summary_metrics`): Aggregates — continuity, fragmentation, ID switch rate. Optionally includes occlusion-aware ID switch count when `per_frame_metrics` is provided.
- **Occlusion-aware ID switch detection** (`detect_identity_switches`): Heuristic that flags ID changes only when a recent high-occlusion event occurred within a sliding window.

Raw model fields (`obj_id_to_tracker_score`, `removed_obj_ids`, `suppressed_obj_ids`) are preserved from the inference loop through to metrics and parquet output.

### Visualizations module (`script/sam3/viz.py`)

Auto-generated on each run, saved to `run_dir/visualizations/`:

- **ID timeline** (`plot_id_timeline`): Eventplot showing object ID existence over time, bars color-coded by tracker score (RdYlGn), with occlusion/count-change trouble-spot overlays.
- **Per-frame dashboard** (`plot_per_frame_dashboard`): 5-panel timeseries — object count, max mask IoU, min centroid distance, clustering coefficient, mean mask area with min/max band.
- **Per-ID scores** (`plot_per_id_scores`): Tracker score over time per object ID.

All plots use MM:SS x-axis when FPS is available.

### Model output data structures

**Raw model output (`Sam3VideoSegmentationOutput` from `model.propagate_in_video_iterator`):**
- `object_ids`: `list[int]` — detected object IDs
- `obj_id_to_mask`: `dict[int, tensor]` — per-object logit masks on GPU
- `obj_id_to_score`: `dict[int, float]` — detection confidence
- `obj_id_to_tracker_score`: `dict[int, float]` — tracking confidence
- `removed_obj_ids`: `set` — IDs removed this frame
- `suppressed_obj_ids`: `set` — IDs suppressed this frame
- `frame_idx`: `int`

**Processed output (from `processor.postprocess_outputs`):**
- `object_ids`: `tensor (N,)` — detected object IDs
- `scores`: `tensor (N,)` — detection confidence
- `boxes`: `tensor (N, 4)` — XYXY absolute coordinates
- `masks`: `tensor (N, H, W)` — boolean masks at original resolution
- `prompt_to_obj_ids`: `dict` — e.g. `{"bird": [0, 1, 2]}`

**Sam3TrackerVideoModel output (normalized in `_process_tracker_chunk`):**
- Same keys as processed output, but: masks are `numpy (N, H, W)`, scores are `sigmoid(object_score_logits)` probabilities, boxes computed from masks. Includes `obj_id_to_tracker_score` (built from sigmoid scores). No `removed_obj_ids`/`suppressed_obj_ids`.

### Run output directory structure

```
{timestamp}_{job_type}/
├── sam3_hf_config.yaml           # Copy of config used
├── sam3_hf_{timestamp}.log       # Loguru log file
├── chunk_info.json               # Per-chunk metadata (model type, prompt points, source frame)
├── annotated_video.mp4           # Video with mask/box/label overlays
├── tracking_outputs.parquet      # Per-detection results with RLE masks
├── metrics/
│   ├── summary_metrics.parquet   # Single-row aggregate metrics
│   ├── per_id_metrics.parquet    # Per-object per-run metrics
│   └── per_frame_metrics.parquet # Per-frame spatial metrics
└── visualizations/
    ├── id_timeline.png
    ├── per_frame_dashboard.png
    └── per_id_scores.png
```

### Canonical output schemas

#### `chunk_info.json`

Written by `demo.py`. Root key `"chunks"` containing an array of per-chunk objects.

| Key | Type | Description |
|-----|------|-------------|
| `chunk_idx` | `int` | Zero-indexed chunk number |
| `frame_range` | `[int, int]` | `[start_frame, end_frame]` (exclusive end) |
| `model_type` | `str` | `"Sam3VideoModel"` or `"Sam3TrackerVideoModel"` |
| `prompt_points` | `dict \| null` | `{str(obj_id): [[x, y], ...]}`. `null` for chunk 0 |
| `num_objects_tracked` | `int` | Number of objects with point prompts |
| `source_frame_idx` | `int \| null` | Frame from previous chunk where points were sampled. `null` for chunk 0 |
| `fallback_reason` | `str \| null` | `"first_chunk"`, `"no_objects_found"`, or `"could_not_extract_points"`. `null` if no fallback |
| `timing.elapsed_seconds` | `float` | Total processing time for this chunk |
| `timing.avg_seconds_per_frame` | `float` | Average time per frame |
| `timing.fps` | `float` | Effective processing speed |
| `timing.num_frames` | `int` | Number of frames in this chunk |

#### `tracking_outputs.parquet`

Written by `utils.py:process_tracking_outputs()`. One row per (frame, object) detection. MultiIndex: `["frame_idx", "object_id"]`.

| Column | Dtype | Description |
|--------|-------|-------------|
| `bbox` | `object` (list of float) | `[x1, y1, x2, y2]` XYXY absolute pixel coordinates |
| `counts` | `str` | RLE-encoded mask (pycocotools format, decoded to string) |
| `size` | `object` (list of int) | `[height, width]` of original frame |
| `scores` | `float64` | Detection confidence (VideoModel) or `sigmoid(object_score_logits)` (TrackerModel) |
| `tracker_score` | `float64` | Tracking confidence — `obj_id_to_tracker_score` (VideoModel) or `sigmoid(object_score_logits)` (TrackerModel) |
| `chunk_idx` | `int64` | Which chunk this detection belongs to |
| `model_type` | `str` | `"Sam3VideoModel"` or `"Sam3TrackerVideoModel"` |
| `is_chunk_start` | `bool` | `True` if this frame is the first frame of its chunk |

#### `metrics/per_frame_metrics.parquet`

Written by `metrics.py:compute_per_frame_metrics()` → `per_frame_metrics_to_df()`. One row per frame.

| Column | Dtype | Description |
|--------|-------|-------------|
| `frame_idx` | `int64` | Frame index |
| `num_objects` | `int64` | Objects detected in this frame |
| `objects_present` | `str` | Comma-separated object IDs |
| `min_centroid_distance` | `float64` | Min pairwise centroid distance (px). `inf` if ≤1 object |
| `mean_centroid_distance` | `float64` | Mean pairwise centroid distance (px). `inf` if ≤1 object |
| `clustering_coefficient` | `float64` | Fraction of centroid pairs within `clustering_distance_threshold` |
| `max_pairwise_mask_iou` | `float64` | Max IoU between any two masks |
| `mean_pairwise_mask_iou` | `float64` | Mean pairwise mask IoU |
| `num_overlapping_pairs` | `int64` | Pairs with IoU > `occlusion_iou_threshold` |
| `mean_mask_area` | `float64` | Mean mask area (px²) |
| `min_mask_area` | `float64` | Smallest mask area |
| `max_mask_area` | `float64` | Largest mask area |
| `mask_area_variance` | `float64` | Variance of mask areas |
| `is_high_occlusion` | `bool` | `True` if max IoU > threshold or clustering > 0.5 |
| `is_object_count_change` | `bool` | `True` if object count differs from previous frame |

#### `metrics/per_id_metrics.parquet`

Written by `metrics.py:compute_per_run_metrics()` → `per_run_metrics_to_multiindex_df()`. One row per contiguous run per object ID. MultiIndex: `["id", "run_idx"]`.

| Column | Dtype | Description |
|--------|-------|-------------|
| `first_frame` | `int64` | First frame of this run |
| `last_frame` | `int64` | Last frame of this run |
| `length` | `int64` | Number of frames in this run |
| `low_count_total` | `int64` | Frames in run with total detections < `low_count_threshold` |
| `low_count_fraction` | `float64` | `low_count_total / length` |
| `mean_iou` | `float64` | Mean consecutive-frame box IoU within run |
| `mean_bbox_area` | `float64` | Mean bounding box area (px²) |
| `mean_score` | `float64` | Mean detection confidence |
| `mean_tracker_score` | `float64` | Mean tracking confidence |
| `frames` | `object` (list of int) | All frame indices in this run |
| `low_count_frames` | `object` (list of int) | Frame indices with low detection count |

#### `metrics/summary_metrics.parquet`

Written by `metrics.py:compute_summary_metrics()` → `summary_metrics_to_df()`. Single row aggregating the entire run.

| Column | Dtype | Description |
|--------|-------|-------------|
| `n_frames` | `int64` | Total frames processed |
| `avg_detections_per_frame` | `float64` | Mean detections per frame |
| `total_unique_ids` | `int64` | Count of unique object IDs |
| `mean_track_length` | `float64` | Mean frames per track |
| `persistence_rate_>=5` | `float64` | Fraction of tracks with ≥ `persistence_k` frames |
| `mean_fragmentation_runs_per_id` | `float64` | Mean contiguous segments per ID |
| `continuity_fraction` | `float64` | Fraction of detections that persist to the next frame |
| `iou_match_count` | `int64` | Greedy box-IoU matches between consecutive frames |
| `id_switches` | `int64` | ID changes among matched boxes |
| `id_switch_rate` | `float64` | `id_switches / iou_match_count` |
| `low_frame_count` | `int64` | Frames with < `low_count_threshold` detections |
| `low_frame_fraction` | `float64` | `low_frame_count / n_frames` |
| `mean_coverage_per_id` | `float64` | Mean of `(frames_present / span)` per ID |
| `mean_per_id_iou` | `float64` | Mean self-IoU across all IDs |
| `occlusion_aware_id_switches` | `int64` | **Conditional** — only present when `per_frame_metrics` is provided |
| `occlusion_aware_id_switch_rate` | `float64` | **Conditional** — only present when `per_frame_metrics` is provided |

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
