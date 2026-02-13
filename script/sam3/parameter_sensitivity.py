#!/usr/bin/env python
"""
Parameter Sensitivity Testing for YOLO Prescan

This script demonstrates how to test different occlusion detection parameters
without re-running YOLO inference (~5 min). Instead, it recomputes prescan
metrics and chunk boundaries from existing yolo_tracking.parquet data.

Usage:
    python -m script.sam3.parameter_sensitivity_example ext-data/output/results/sam3-hf/20260213_142342_yolo_prescan

Modify the PARAMETER_SETS below to test different threshold combinations.
"""

import argparse
import subprocess
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger
from omegaconf import OmegaConf

from src.utils import chunk_video_frames_adaptive, chunk_video_frames_dual, setup_logger
from src.viz import plot_yolo_prescan_overview
from src.yolo_prescan import compute_yolo_prescan_results, yolo_prescan_to_df


def generate_occlusion_overlay_video(
    video_path: Path,
    occlusion_periods: list,
    fps: float,
    output_path: Path,
    opacity: float = 0.3,
):
    """
    Generate video with red overlay during occlusion periods using ffmpeg.

    Args:
        video_path: Path to source video
        occlusion_periods: List of (start_frame, end_frame) tuples
        fps: Video frame rate
        output_path: Path for output video
        opacity: Red overlay opacity (0-1), default 0.3
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Build ffmpeg filter expression for occlusion periods
    if not occlusion_periods:
        # No occlusions, just copy the video
        cmd = [
            "ffmpeg",
            "-i",
            str(video_path),
            "-c",
            "copy",
            "-y",
            str(output_path),
        ]
    else:
        # Build enable expression: between(n,start,end) for each period
        enable_expressions = []
        for start_frame, end_frame in occlusion_periods:
            enable_expressions.append(f"between(n,{start_frame},{end_frame})")
        enable_expr = "+".join(enable_expressions)

        # drawbox filter: draw full-screen red box with alpha
        filter_complex = (
            f"drawbox=x=0:y=0:w=iw:h=ih:color=red@{opacity}:t=fill"
            f":enable='{enable_expr}'"
        )

        cmd = [
            "ffmpeg",
            "-i",
            str(video_path),
            "-vf",
            filter_complex,
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "23",
            "-c:a",
            "copy",
            "-y",
            str(output_path),
        ]

    logger.info(f"    Running ffmpeg (this may take a few minutes)...")
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")


# Parameter sets to test
# Strict → Moderate → Relaxed
PARAMETER_SETS = [
    {
        "name": "strict",
        "occlusion_iou_threshold": 0.08,
        "high_occlusion_threshold": 0.15,
        "clustering_distance_threshold": 0.15,
    },
    {
        "name": "moderate (default)",
        "occlusion_iou_threshold": 0.15,
        "high_occlusion_threshold": 0.3,
        "clustering_distance_threshold": 0.15,
    },
    {
        "name": "relaxed",
        "occlusion_iou_threshold": 0.20,
        "high_occlusion_threshold": 0.4,
        "clustering_distance_threshold": 0.20,
    },
]

# =============================================================================
# Main Analysis
# =============================================================================


def run_parameter_sensitivity_analysis(
    yolo_df: pd.DataFrame,
    run_dir: Path,
    fps: float,
    total_frames: int,
    video_model_chunk_seconds: float = 15,
    tracker_chunk_seconds: float = 45,
    adaptive_min_chunk_seconds: float = 15,
    adaptive_max_chunk_seconds: float = 90,
    video_path: Path | None = None,
    generate_video: bool = False,
    parameter_sets: list[dict] | None = None,
) -> pd.DataFrame:
    """
    Run parameter sensitivity analysis on existing YOLO tracking data.

    Tests multiple parameter combinations without re-running YOLO inference.
    Generates comparison visualizations and optionally video overlays.

    Args:
        yolo_df: YOLO tracking DataFrame from prescan
        run_dir: Run directory where outputs will be saved
        fps: Video frame rate
        total_frames: Total number of frames
        video_model_chunk_seconds: Duration of first chunk
        tracker_chunk_seconds: Duration of subsequent chunks
        adaptive_min_chunk_seconds: Minimum chunk duration
        adaptive_max_chunk_seconds: Maximum chunk duration
        video_path: Optional path to source video for overlay generation
        generate_video: Whether to generate video overlays
        parameter_sets: List of parameter dicts to test (uses defaults if None)

    Returns:
        DataFrame with comparison summary
    """
    # Use default parameter sets if none provided
    if parameter_sets is None:
        parameter_sets = PARAMETER_SETS

    # Create output directory for parameter comparison
    output_dir = run_dir / "parameter_comparison"
    output_dir.mkdir(exist_ok=True)

    logger.info("=" * 70)
    logger.info("YOLO Prescan Parameter Sensitivity Testing")
    logger.info("=" * 70)
    logger.info(f"Output directory: {output_dir}")
    logger.info(
        f"  → {len(yolo_df)} detections across {yolo_df['frame'].nunique()} frames"
    )
    logger.info(f"  → Frame range: {yolo_df['frame'].min()} to {yolo_df['frame'].max()}")
    logger.info(f"  → FPS: {fps}")
    logger.info(f"  → Total frames: {total_frames}")
    logger.info("")

    # Warn if video generation was requested but no video path available
    if generate_video and video_path is None:
        logger.warning("Video generation requested but video path not available")
        logger.warning("Video overlays will be skipped.")
        generate_video = False

    # Test each parameter set
    results_summary = []

    for param_set in parameter_sets:
        name = param_set["name"]
        logger.info("-" * 70)
        logger.info(f"Testing: {name}")
        logger.info(
            f"  occlusion_iou_threshold:        {param_set['occlusion_iou_threshold']}"
        )
        logger.info(
            f"  high_occlusion_threshold:       {param_set['high_occlusion_threshold']}"
        )
        logger.info(
            f"  clustering_distance_threshold:  {param_set['clustering_distance_threshold']}"
        )

        # Recompute prescan metrics with new parameters
        prescan_results = compute_yolo_prescan_results(
            yolo_df,
            fps=fps,
            window_seconds=1.0,
            high_occlusion_threshold=param_set["high_occlusion_threshold"],
            occlusion_iou_threshold=param_set["occlusion_iou_threshold"],
            clustering_distance_threshold=param_set["clustering_distance_threshold"],
        )

        # Extract results
        occlusion_periods = prescan_results["occlusion_periods"]
        transition_frames = prescan_results["transition_frames"]
        per_frame_metrics = prescan_results["per_frame_metrics"]

        logger.info(f"  → {len(occlusion_periods)} occlusion periods detected")
        logger.info(f"  → {len(transition_frames)} transition frames identified")

        # Compute original (fixed) chunks
        chunks = chunk_video_frames_dual(
            total_frames,
            fps,
            video_model_chunk_seconds,
            tracker_chunk_seconds,
        )

        # Compute adaptive chunks with new parameters
        adaptive_chunks = chunk_video_frames_adaptive(
            chunks,
            transition_frames,
            fps,
            occlusion_periods=occlusion_periods,
            min_chunk_seconds=adaptive_min_chunk_seconds,
            max_chunk_seconds=adaptive_max_chunk_seconds,
        )

        # Extract tracker chunk boundaries for visualization
        chunk_boundaries = [
            c[0] for c in adaptive_chunks if c[2] == "tracker"  # (start, end, type)
        ]

        logger.info(f"  → {len(adaptive_chunks)} chunks after adaptive adjustment")

        # Convert metrics to DataFrame
        prescan_metrics_df = yolo_prescan_to_df(per_frame_metrics)

        # Generate visualization
        save_path = output_dir / f"yolo_prescan_overview_{name.replace(' ', '_')}.png"
        plot_yolo_prescan_overview(
            prescan_metrics_df,
            occlusion_periods=occlusion_periods,
            chunk_boundaries=chunk_boundaries,
            fps=fps,
            save_path=save_path,
            name_suffix=name,
        )
        logger.info(f"  → Visualization saved: {save_path.name}")

        # Generate video overlay if requested and video path is available
        if generate_video and video_path is not None:
            video_output_path = (
                output_dir / f"occlusion_overlay_{name.replace(' ', '_')}.mp4"
            )
            logger.info(f"  → Generating video overlay: {video_output_path.name}")

            try:
                generate_occlusion_overlay_video(
                    video_path=video_path,
                    occlusion_periods=occlusion_periods,
                    fps=fps,
                    output_path=video_output_path,
                    opacity=0.3,
                )
                logger.info(f"  ✓ Video overlay saved: {video_output_path.name}")
            except Exception as e:
                logger.warning(f"  ✗ Video generation failed: {e}")

        logger.info("")

        # Store summary
        results_summary.append(
            {
                "parameter_set": name,
                "occlusion_iou_threshold": param_set["occlusion_iou_threshold"],
                "high_occlusion_threshold": param_set["high_occlusion_threshold"],
                "num_occlusion_periods": len(occlusion_periods),
                "num_transition_frames": len(transition_frames),
                "num_chunks": len(adaptive_chunks),
                "num_boundaries_adjusted": sum(
                    1
                    for i, (orig, adapt) in enumerate(zip(chunks, adaptive_chunks))
                    if orig[0] != adapt[0] and i > 0  # skip chunk 0
                ),
            }
        )

    # Print comparison table
    logger.info("=" * 70)
    logger.info("SUMMARY COMPARISON")
    logger.info("=" * 70)
    summary_df = pd.DataFrame(results_summary)
    logger.info(summary_df.to_string(index=False))
    logger.info("")

    # Save summary to CSV
    summary_path = output_dir / "parameter_comparison_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Summary saved to: {summary_path}")
    logger.info("")
    logger.info("✓ Parameter sensitivity testing complete!")
    logger.info(f"  Review visualizations in: {output_dir}")

    return summary_df


def main():
    parser = argparse.ArgumentParser(
        description="Test YOLO prescan parameter sensitivity on existing run",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Path to existing prescan run directory containing yolo_tracking.parquet",
    )
    parser.add_argument(
        "--generate-video",
        action="store_true",
        help="Generate video overlays with red opacity for occlusion periods (requires source video path in config)",
    )
    args = parser.parse_args()

    run_dir = args.run_dir
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    # Setup logger with timestamped file in output directory
    output_dir = run_dir / "parameter_comparison"
    output_dir.mkdir(exist_ok=True)
    log_file = setup_logger(log_dir=output_dir, job_type="parameter_sensitivity")

    logger.info("=" * 70)
    logger.info("STANDALONE Parameter Sensitivity Testing")
    logger.info("=" * 70)
    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Log file: {log_file}")
    logger.info("")

    # Load existing YOLO tracking data (no re-inference needed!)
    yolo_tracking_path = run_dir / "yolo_tracking.parquet"
    if not yolo_tracking_path.exists():
        raise FileNotFoundError(f"YOLO tracking data not found: {yolo_tracking_path}")

    logger.info(f"Loading YOLO tracking data from: {yolo_tracking_path}")
    yolo_df = pd.read_parquet(yolo_tracking_path)
    total_frames = yolo_df["frame"].max() + 1  # frames are 0-indexed
    logger.info("")

    # Load config from run directory to get original parameters
    config_files = list(run_dir.glob("*.yaml"))
    if config_files:
        config_path = config_files[0]
        logger.info(f"Loading config from: {config_path.name}")
        cfg = OmegaConf.load(config_path)

        fps = cfg.get("yolo_prescan", {}).get(
            "fps", 25.0
        )  # Try to get from yolo_prescan
        if fps is None:
            # Fallback: infer from prescan summary if available
            summary_path = run_dir / "metrics" / "yolo_prescan_summary.parquet"
            if summary_path.exists():
                summary_df = pd.read_parquet(summary_path)
                fps = summary_df.iloc[0]["fps"]
            else:
                fps = 25.0  # Last resort default

        video_model_chunk_seconds = cfg.get("video_model_chunk_seconds", 15)
        tracker_chunk_seconds = cfg.get("tracker_chunk_seconds", 45)
        adaptive_min_chunk_seconds = cfg.get("adaptive_min_chunk_seconds", 15)
        adaptive_max_chunk_seconds = cfg.get("adaptive_max_chunk_seconds", 90)

        logger.info(f"  → FPS: {fps}")
        logger.info(f"  → Total frames: {total_frames}")
        logger.info(f"  → Video model chunk: {video_model_chunk_seconds}s")
        logger.info(f"  → Tracker chunk: {tracker_chunk_seconds}s")

        # Extract video path for video generation if requested
        video_path = None
        if args.generate_video:
            video_path_str = cfg.get("video_path")
            if video_path_str:
                video_path = Path(video_path_str)
                if not video_path.exists():
                    logger.info(f"  ⚠ Video path from config not found: {video_path}")
                    video_path = None
                else:
                    logger.info(f"  → Video path: {video_path}")
            else:
                logger.info("  ⚠ No video_path in config, skipping video generation")
    else:
        # Fallback to prescan summary
        summary_path = run_dir / "metrics" / "yolo_prescan_summary.parquet"
        if summary_path.exists():
            summary_df = pd.read_parquet(summary_path)
            fps = summary_df.iloc[0]["fps"]
            logger.info(f"Config not found, using values from prescan summary")
            logger.info(f"  → FPS: {fps}")
            logger.info(f"  → Total frames: {total_frames}")
        else:
            logger.info("WARNING: Could not find config or summary, using defaults")
            fps = 25.0

        # Use defaults for chunk parameters
        video_model_chunk_seconds = 15
        tracker_chunk_seconds = 45
        adaptive_min_chunk_seconds = 15
        adaptive_max_chunk_seconds = 90
        video_path = None

    logger.info("")

    # Run the parameter sensitivity analysis using the refactored function
    run_parameter_sensitivity_analysis(
        yolo_df=yolo_df,
        run_dir=run_dir,
        fps=fps,
        total_frames=total_frames,
        video_model_chunk_seconds=video_model_chunk_seconds,
        tracker_chunk_seconds=tracker_chunk_seconds,
        adaptive_min_chunk_seconds=adaptive_min_chunk_seconds,
        adaptive_max_chunk_seconds=adaptive_max_chunk_seconds,
        video_path=video_path,
        generate_video=args.generate_video,
    )


if __name__ == "__main__":
    main()
