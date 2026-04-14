# Tracker module

The tracker pipeline (`script/run_tracker.py`) is the main entry point for SAM3 tracking. The metrics module (`src/metrics.py`) and visualizations module (`src/viz.py`) are primarily used by the tracker pipeline.

## Key patterns

- **Config-driven**: `run_tracker.py` reads YAML configs via OmegaConf. Script entry point parses environment variables `CUDA_VISIBLE_DEVICES` and `PYTORCH_ALLOC_CONF` from config file before torch is imported. A `tracking:` section overrides `Sam3VideoConfig` parameters (keep-alive, IoU thresholds, reconditioning interval, etc.).
- **Dataset-aligned output**: Each run creates `{output_dir}/{YYYYMMDD_HHMMSS}_{job_type}/day_{N}/{sanitized_video_stem}/` with subdirectories for `metrics/` and `visualizations/`. The timestamped parent isolates runs from each other. The day number and video ID are parsed from the video filename via `extract_video_id()`. Falls back to `{output_dir}/{timestamp}_{job_type}/{sanitized_stem}/` if parsing fails. Config is copied for reproducibility.
- **Loguru logging**: Console (colored) + file handler in run directory. Replaces all `print()`.
- **Chunked processing**: Long videos are split into chunks via `chunk_video_frames_adaptive()`. Initial chunk size is set by `chunk_seconds` (default 60 s); when YOLO scan data is available, each boundary is shifted within a +/-`adaptive_search_window_seconds` window to the frame with the highest separation score. Chunk 0 uses `Sam3VideoModel` (text-prompted); subsequent chunks use `Sam3TrackerVideoModel` (point-prompted). Small trailing remainders (<10% of chunk size) are absorbed into the last chunk. Point prompts are extracted from previous chunk's masks via `extract_equidistant_points_from_masks()`. `find_frame_with_enough_objects()` searches backwards for a frame with enough detected objects. `max_frames_to_track` limits how many frames are processed per video.
- **Two model phases**: `Sam3VideoModel` (text->segmentation) for initialization, `Sam3TrackerVideoModel` (point->tracking) for propagation. Each chunk loads its model fresh and cleans up GPU memory afterwards (`free_gpu_memory()` with triple `gc.collect` + CUDA cache clearing).
- **Adaptive chunking (YOLO scan)**: When `use_adaptive_chunking: true`, `run_yolo_scan()` (in `src/yolo_scan.py`) runs YOLO tracking on the full video and returns a raw `yolo_df`. Analysis -- `compute_yolo_per_frame_metrics()` -> `identify_occlusion_periods()` -> `find_high_separation_windows()` -- lives in `src/chunk_boundaries.py`. `chunk_video_frames_adaptive()` refines boundaries in priority order -- separation-first (highest separation_score inside a high-separation window), then occlusion avoidance (farthest from occlusion with 90%/50% directional penalties), validated against `adaptive_max_chunk_seconds`. Scan outputs saved as `yolo_tracking.parquet`, `yolo_scan_metrics.parquet`, `yolo_scan_summary.parquet`. A `yolo_scan:` config section controls model, thresholds, and tracker config. To reanalyse boundaries without re-running YOLO, use `script/compute_chunk_boundaries.py`.
- **Manual chunking**: Set `manual_chunk_frames` to a list of `[start, end]` pairs to override fixed/adaptive chunking entirely. First pair -> `Sam3VideoModel`; subsequent -> `Sam3TrackerVideoModel`. Disables `yolo_scan_only`/`use_adaptive_chunking` with warnings. See `build_manual_chunks()` in `src/utils.py` and `config/tracker_manual_chunking.yaml`.
- **Batch processing**: Set `video_dir` instead of `video_path` to process all videos in a directory. All videos share one timestamped parent; each video creates its own `day_{N}/{sanitized_stem}/` subdirectory inside it. Errors caught per-video. `manual_chunk_frames` accepts a dict keyed by **basename** (e.g. `"video1.mp4": [[0,375],...]`) for per-video boundaries; unlisted videos use fixed chunking.
- **Device selection**: `Accelerator().device` from HuggingFace Accelerate.

## Metrics module (`src/metrics.py`)

Computes tracking quality post-hoc from `outputs_per_frame`. Four levels:

- **Per-frame** (`compute_per_frame_metrics`): Pairwise mask IoU, centroids, clustering coefficient, mask area stats, occlusion flags.
- **Per-run** (`compute_per_run_metrics`): Per contiguous segment per ID -- runs, gaps, coverage, `mean_tracker_score`.
- **Summary** (`compute_summary_metrics`): Continuity, fragmentation, ID switch rate; occlusion-aware ID switch count when `per_frame_metrics` provided.
- **Occlusion-aware ID switch detection** (`detect_identity_switches`): Flags ID changes only when a recent high-occlusion event occurred within a sliding window.

## Visualizations module (`src/viz.py`)

Auto-generated on each run, saved to `run_dir/visualizations/`: ID timeline (tracker score color-coded), per-frame dashboard (5-panel timeseries), per-ID tracker scores, mask evolution at chunk boundaries, prompt points at boundaries, YOLO scan overview (when adaptive chunking enabled). All plots use MM:SS x-axis when FPS is available.

## Output schemas

- **`tracking_outputs.parquet`**: One row per (frame, object). MultiIndex `["frame_idx", "object_id"]`. Includes bbox, RLE mask, scores, tracker_score, chunk_idx, model_type.
- **`chunk_info.json`**: Per-chunk metadata: frame range, model type, prompt points, timing. Keys: `grounding_used`, `grounding_source_frame_idx`, `grounding_num_objects`, `grounding_fallback_reason`.
- **YOLO scan outputs**: `yolo_tracking.parquet`, `metrics/yolo_scan_metrics.parquet`, `metrics/yolo_scan_summary.parquet`, `visualizations/yolo_scan_overview.png`.
- **Metrics parquets**: `per_frame_metrics`, `per_id_metrics`, `summary_metrics`.
- **Run directory**: `{timestamp}_{job_type}/day_{N}/{sanitized_stem}/` with config copy, log, parquets, `metrics/`, `visualizations/`. Both single-video and batch mode use the same `day_{N}/{sanitized_stem}/` layout inside a timestamped parent, matching the `data/postprocessing/` structure expected by `build_dataset`.

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
