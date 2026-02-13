#!/usr/bin/env python
"""
Parameter Sensitivity Testing for YOLO Prescan

This script demonstrates how to test different occlusion detection parameters
without re-running YOLO inference (~5 min). Instead, it recomputes prescan
metrics and chunk boundaries from existing yolo_tracking.parquet data.

Usage:
    python notebook/parameter_sensitivity_example.py ext-data/output/results/sam3-hf/20260213_142342_yolo_prescan

Modify the PARAMETER_SETS below to test different threshold combinations.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from omegaconf import OmegaConf

from src.utils import chunk_video_frames_adaptive, chunk_video_frames_dual
from src.viz import plot_yolo_prescan_overview
from src.yolo_prescan import compute_yolo_prescan_results, yolo_prescan_to_df

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
    args = parser.parse_args()

    run_dir = args.run_dir
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    print("=" * 70)
    print("YOLO Prescan Parameter Sensitivity Testing")
    print("=" * 70)
    print(f"Run directory: {run_dir}")
    print()

    # Load existing YOLO tracking data (no re-inference needed!)
    yolo_tracking_path = run_dir / "yolo_tracking.parquet"
    if not yolo_tracking_path.exists():
        raise FileNotFoundError(f"YOLO tracking data not found: {yolo_tracking_path}")

    print(f"Loading YOLO tracking data from: {yolo_tracking_path}")
    yolo_df = pd.read_parquet(yolo_tracking_path)
    total_frames = yolo_df["frame"].max() + 1  # frames are 0-indexed
    print(f"  → {len(yolo_df)} detections across {yolo_df['frame'].nunique()} frames")
    print(f"  → Frame range: {yolo_df['frame'].min()} to {yolo_df['frame'].max()}")
    print()

    # Load config from run directory to get original parameters
    config_files = list(run_dir.glob("*.yaml"))
    if config_files:
        config_path = config_files[0]
        print(f"Loading config from: {config_path.name}")
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

        print(f"  → FPS: {fps}")
        print(f"  → Total frames: {total_frames}")
        print(f"  → Video model chunk: {video_model_chunk_seconds}s")
        print(f"  → Tracker chunk: {tracker_chunk_seconds}s")
    else:
        # Fallback to prescan summary
        summary_path = run_dir / "metrics" / "yolo_prescan_summary.parquet"
        if summary_path.exists():
            summary_df = pd.read_parquet(summary_path)
            fps = summary_df.iloc[0]["fps"]
            print(f"Config not found, using values from prescan summary")
            print(f"  → FPS: {fps}")
            print(f"  → Total frames: {total_frames}")
        else:
            print("WARNING: Could not find config or summary, using defaults")
            fps = 25.0

        # Use defaults for chunk parameters
        video_model_chunk_seconds = 15
        tracker_chunk_seconds = 45
        adaptive_min_chunk_seconds = 15
        adaptive_max_chunk_seconds = 90

    print()

    # Create output directory for parameter comparison
    output_dir = run_dir / "parameter_comparison"
    output_dir.mkdir(exist_ok=True)
    print(f"Saving results to: {output_dir}")
    print()

    # Test each parameter set
    results_summary = []

    for param_set in PARAMETER_SETS:
        name = param_set["name"]
        print("-" * 70)
        print(f"Testing: {name}")
        print(
            f"  occlusion_iou_threshold:        {param_set['occlusion_iou_threshold']}"
        )
        print(
            f"  high_occlusion_threshold:       {param_set['high_occlusion_threshold']}"
        )
        print(
            f"  clustering_distance_threshold:  {param_set['clustering_distance_threshold']}"
        )
        print()

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

        print(f"  → {len(occlusion_periods)} occlusion periods detected")
        print(f"  → {len(transition_frames)} transition frames identified")

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
            c[0]
            for c in adaptive_chunks
            if c[2] == "tracker"  # (start, end, type)
        ]

        print(f"  → {len(adaptive_chunks)} chunks after adaptive adjustment")
        print()

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
        print(f"  → Visualization saved: {save_path.name}")
        print()

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
    print("=" * 70)
    print("SUMMARY COMPARISON")
    print("=" * 70)
    summary_df = pd.DataFrame(results_summary)
    print(summary_df.to_string(index=False))
    print()

    # Save summary to CSV
    summary_path = output_dir / "parameter_comparison_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary saved to: {summary_path}")
    print()
    print("✓ Parameter sensitivity testing complete!")
    print(f"  Review visualizations in: {output_dir}")


if __name__ == "__main__":
    main()
