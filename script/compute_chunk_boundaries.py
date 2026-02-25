"""
Recompute chunk boundary metrics from an existing run dir.

Given a run dir that contains ``yolo_tracking.parquet``, this script:

1. Recomputes per-frame metrics, occlusion periods, and separation windows
2. Recomputes adaptive chunk boundaries
3. Saves updated parquets in-place (metrics/yolo_scan_metrics.parquet,
   metrics/yolo_scan_summary.parquet)
4. Overwrites chunk_info.json
5. Regenerates visualizations/yolo_scan_overview.png
6. Generates visualizations/chunk_boundaries_<run>.png (frame screengrab grid)

Tunable parameters live in the YAML config under ``yolo_scan:`` and at the
top level (``chunk_seconds``, ``adaptive_max_chunk_seconds``, etc.).
To tweak: edit the YAML copy in the run dir, then re-run this script.

Usage::

    pixi run -e sam3-hf python -m script.compute_chunk_boundaries \\
        --run-dir ext-data/output/results/yolo_scan/20260223_231859_yolo_scan

    # With a different config (tweaked params):
    pixi run -e sam3-hf python -m script.compute_chunk_boundaries \\
        --run-dir ext-data/output/results/yolo_scan/20260223_231859_yolo_scan \\
        --config config/yolo_scan_only.yaml
"""

import json
from argparse import ArgumentParser
from pathlib import Path

import cv2
import pandas as pd
from loguru import logger
from omegaconf import OmegaConf

from src.chunk_boundaries import (
    chunk_video_frames_adaptive,
    compute_yolo_per_frame_metrics,
    yolo_scan_to_df,
)
from src.viz import plot_chunk_boundary_frames, plot_yolo_scan_overview


def parse_args():
    parser = ArgumentParser(
        description="Recompute YOLO scan metrics and chunk boundaries from an existing run dir"
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Existing run directory containing yolo_tracking.parquet",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Override YAML config; defaults to the .yaml found in --run-dir",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    assert run_dir.exists(), f"Run dir not found: {run_dir}"

    # -------------------------------------------------------------------------
    # 1. Find and load config
    # -------------------------------------------------------------------------
    if args.config:
        config_path = Path(args.config)
        assert config_path.exists(), f"Config not found: {config_path}"
    else:
        yaml_files = list(run_dir.glob("*.yaml"))
        assert yaml_files, f"No YAML config found in: {run_dir}"
        assert len(yaml_files) == 1, (
            f"Expected exactly one YAML in {run_dir}, found: {yaml_files}"
        )
        config_path = yaml_files[0]

    logger.info(f"Loading config: {config_path}")
    cfg = OmegaConf.load(config_path)
    yolo_scan_cfg = cfg.get("yolo_scan", {})

    # -------------------------------------------------------------------------
    # 2. Load raw YOLO tracking data
    # -------------------------------------------------------------------------
    yolo_parquet_path = run_dir / "yolo_tracking.parquet"
    assert yolo_parquet_path.exists(), f"yolo_tracking.parquet not found in: {run_dir}"
    logger.info(f"Loading: {yolo_parquet_path}")
    yolo_df = pd.read_parquet(yolo_parquet_path)

    # -------------------------------------------------------------------------
    # 3. Get fps + total_frames
    #    Primary: existing yolo_scan_summary.parquet  Fallback: cv2
    # -------------------------------------------------------------------------
    summary_path = run_dir / "metrics" / "yolo_scan_summary.parquet"
    fps: float
    total_frames: int

    if summary_path.exists():
        summary_df = pd.read_parquet(summary_path)
        fps = float(summary_df["fps"].iloc[0])
        total_frames = int(summary_df["total_frames"].iloc[0])
        logger.info(f"fps={fps}, total_frames={total_frames} (from summary parquet)")
    else:
        logger.warning(f"{summary_path} not found — falling back to cv2")
        video_path = cfg.get("video_path")
        assert video_path, "video_path not in config and no summary parquet available"
        cap_meta = cv2.VideoCapture(str(video_path))
        assert cap_meta.isOpened(), f"Could not open video: {video_path}"
        fps = cap_meta.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap_meta.get(cv2.CAP_PROP_FRAME_COUNT))
        cap_meta.release()
        logger.info(f"fps={fps}, total_frames={total_frames} (from cv2)")

    # -------------------------------------------------------------------------
    # 4. Per-frame metrics
    # -------------------------------------------------------------------------
    occlusion_iou_threshold = float(yolo_scan_cfg.get("occlusion_iou_threshold", 0.15))
    clustering_distance_threshold = float(
        yolo_scan_cfg.get("clustering_distance_threshold", 0.15)
    )
    separation_min_objects = int(yolo_scan_cfg.get("separation_min_objects", 3))
    separation_min_distance = float(yolo_scan_cfg.get("separation_min_distance", 0.15))

    logger.info("Computing per-frame metrics...")
    per_frame_metrics = compute_yolo_per_frame_metrics(
        yolo_df,
        occlusion_iou_threshold=occlusion_iou_threshold,
        clustering_distance_threshold=clustering_distance_threshold,
        use_normalized_coords=True,
        separation_min_objects=separation_min_objects,
        separation_min_distance=separation_min_distance,
    )
    logger.info(f"  {len(per_frame_metrics)} frames with metrics")

    # -------------------------------------------------------------------------
    # 5. Adaptive chunk boundaries
    # -------------------------------------------------------------------------
    chunk_seconds = float(cfg.get("chunk_seconds", 60))
    search_window_seconds = float(cfg.get("adaptive_search_window_seconds", 10.0))
    max_chunk_seconds = float(cfg.get("adaptive_max_chunk_seconds", 150))

    logger.info("Computing adaptive chunk boundaries...")
    chunks = chunk_video_frames_adaptive(
        total_frames,
        fps,
        chunk_seconds,
        per_frame_metrics=per_frame_metrics,
        search_window_seconds=search_window_seconds,
        max_chunk_seconds=max_chunk_seconds,
    )
    logger.info(f"  {len(chunks)} chunks:")
    for i, (s, e, mtype) in enumerate(chunks):
        logger.info(f"    Chunk {i}: frames {s}–{e} ({(e - s) / fps:.1f}s) [{mtype}]")

    # -------------------------------------------------------------------------
    # 8. Save parquets in-place
    # -------------------------------------------------------------------------
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)

    yolo_scan_metrics_df = yolo_scan_to_df(per_frame_metrics)
    yolo_scan_metrics_path = metrics_dir / "yolo_scan_metrics.parquet"
    yolo_scan_metrics_df.to_parquet(yolo_scan_metrics_path, index=False)
    logger.info(f"Saved: {yolo_scan_metrics_path}")

    yolo_scan_summary_df = pd.DataFrame(
        [
            {
                "total_frames": total_frames,
                "video_duration_seconds": total_frames / fps,
                "fps": fps,
            }
        ]
    )
    yolo_scan_summary_path = metrics_dir / "yolo_scan_summary.parquet"
    yolo_scan_summary_df.to_parquet(yolo_scan_summary_path, index=False)
    logger.info(f"Saved: {yolo_scan_summary_path}")

    # -------------------------------------------------------------------------
    # 9. Save chunk_info.json in-place
    # -------------------------------------------------------------------------
    chunk_info = {
        "chunks": [
            {
                "chunk_idx": i,
                "frame_range": [s, e],
                "model_type": (
                    "Sam3VideoModel" if mtype == "video" else "Sam3TrackerVideoModel"
                ),
            }
            for i, (s, e, mtype) in enumerate(chunks)
        ]
    }
    chunk_info_path = run_dir / "chunk_info.json"
    with open(chunk_info_path, "w") as f:
        json.dump(chunk_info, f, indent=2)
    logger.info(f"Saved: {chunk_info_path}")

    # -------------------------------------------------------------------------
    # 10. Regenerate yolo_scan_overview.png
    # -------------------------------------------------------------------------
    viz_dir = run_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)

    tracker_chunk_starts = [s for s, e, mtype in chunks[1:]]

    plot_yolo_scan_overview(
        yolo_scan_metrics_df,
        occlusion_periods=None,
        chunk_boundaries=tracker_chunk_starts,
        fps=fps,
        save_path=viz_dir / "yolo_scan_overview.png",
    )
    logger.info(f"Saved: {viz_dir / 'yolo_scan_overview.png'}")

    # -------------------------------------------------------------------------
    # 11. Generate chunk_boundaries_<run>.png (N rows × 2 cols: start + end)
    # -------------------------------------------------------------------------
    video_path = cfg.get("video_path")
    plot_chunk_boundary_frames(
        chunk_info=chunk_info,
        video_path=video_path,
        fps=fps,
        yolo_scan_df=yolo_scan_metrics_df,
        save_path=viz_dir / f"chunk_boundaries_{run_dir.stem}.png",
    )


if __name__ == "__main__":
    main()
