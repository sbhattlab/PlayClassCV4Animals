"""
Visualize tracking metrics around chunk boundaries.

For each boundary, creates a 4-panel time-series plot (±20 frames window):
1. Bbox sizes per object ID
2. Centroid positions (X and Y) per object ID
3. Tracker scores per object ID
4. Min centroid distance (from per_frame_metrics)

Usage:
    python -m script.sam3.plot_chunk_boundary_metrics \
        --run-dir sandbox/output/results/sam3-hf/20260206_164707_sam3_hf \
        --window 20
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from script.sam3.diagnostic_utils import (
    create_output_directory,
    get_chunk_boundaries,
    get_video_fps,
    load_run_results,
)

# Reuse viz helpers
from src.viz import _format_mmss


def plot_boundary_metrics(
    tracking_df, per_frame_df, boundary: dict,
    fps: float, window: int, output_dir: Path,
):
    """Create 4-panel metric plot around a single chunk boundary."""
    boundary_frame = boundary["boundary_frame"]
    chunk_idx = boundary["chunk_idx"]

    frame_lo = boundary_frame - window
    frame_hi = boundary_frame + window

    # Filter tracking data to window
    all_frames = tracking_df.index.get_level_values("frame_idx")
    mask = (all_frames >= frame_lo) & (all_frames <= frame_hi)
    window_df = tracking_df[mask]

    if window_df.empty:
        print(f"  No data in window [{frame_lo}, {frame_hi}]")
        return

    object_ids = sorted(window_df.index.get_level_values("object_id").unique().tolist())
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(object_ids), 1)))
    oid_colors = {oid: colors[i] for i, oid in enumerate(object_ids)}

    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(14, 14))
    boundary_x = boundary_frame / fps if fps else boundary_frame

    def frame_to_x(f):
        return f / fps if fps else f

    # --- Panel 1: Bbox sizes ---
    ax = axes[0]
    for oid in object_ids:
        try:
            oid_data = window_df.xs(oid, level="object_id")
        except KeyError:
            continue
        frames = sorted(oid_data.index.tolist())
        x = [frame_to_x(f) for f in frames]
        bbox_areas = []
        for f in frames:
            bbox = oid_data.loc[f, "bbox"]
            if isinstance(bbox, list) and len(bbox) == 4:
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                bbox_areas.append(w * h)
            else:
                bbox_areas.append(0)
        ax.plot(x, bbox_areas, color=oid_colors[oid], linewidth=1.2,
                label=f"ID {oid}", marker=".", markersize=3)
    ax.axvline(boundary_x, color="red", linewidth=2, linestyle="--", alpha=0.8,
               label="Boundary")
    ax.set_ylabel("Bbox Area (px²)")
    ax.set_title(f"Chunk Boundary {chunk_idx-1}→{chunk_idx} (frame {boundary_frame})")
    ax.legend(fontsize=7, loc="upper right")

    # --- Panel 2: Centroid positions ---
    ax = axes[1]
    for oid in object_ids:
        try:
            oid_data = window_df.xs(oid, level="object_id")
        except KeyError:
            continue
        frames = sorted(oid_data.index.tolist())
        x = [frame_to_x(f) for f in frames]
        cx_list, cy_list = [], []
        for f in frames:
            bbox = oid_data.loc[f, "bbox"]
            if isinstance(bbox, list) and len(bbox) == 4:
                cx_list.append((bbox[0] + bbox[2]) / 2)
                cy_list.append((bbox[1] + bbox[3]) / 2)
            else:
                cx_list.append(np.nan)
                cy_list.append(np.nan)
        ax.plot(x, cx_list, color=oid_colors[oid], linewidth=1.2,
                linestyle="-", label=f"ID {oid} X")
        ax.plot(x, cy_list, color=oid_colors[oid], linewidth=1.0,
                linestyle="--", alpha=0.6)
    ax.axvline(boundary_x, color="red", linewidth=2, linestyle="--", alpha=0.8)
    ax.set_ylabel("Centroid Position (px)")
    ax.legend(fontsize=7, loc="upper right", title="solid=X, dashed=Y")

    # --- Panel 3: Tracker scores ---
    ax = axes[2]
    for oid in object_ids:
        try:
            oid_data = window_df.xs(oid, level="object_id")
        except KeyError:
            continue
        frames = sorted(oid_data.index.tolist())
        x = [frame_to_x(f) for f in frames]
        scores = [oid_data.loc[f, "tracker_score"] for f in frames]
        ax.plot(x, scores, color=oid_colors[oid], linewidth=1.2,
                label=f"ID {oid}", marker=".", markersize=3)
    ax.axvline(boundary_x, color="red", linewidth=2, linestyle="--", alpha=0.8)
    ax.set_ylabel("Tracker Score")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=7, loc="lower right")

    # --- Panel 4: Min centroid distance ---
    ax = axes[3]
    if per_frame_df is not None and "min_centroid_distance" in per_frame_df.columns:
        pf_mask = (per_frame_df["frame_idx"] >= frame_lo) & \
                  (per_frame_df["frame_idx"] <= frame_hi)
        pf_window = per_frame_df[pf_mask]
        x = [frame_to_x(f) for f in pf_window["frame_idx"].values]
        dists = pf_window["min_centroid_distance"].values.copy()
        dists[np.isinf(dists)] = np.nan
        ax.plot(x, dists, color="mediumpurple", linewidth=1.2)
        ax.set_ylabel("Min Centroid Dist (px)")
    else:
        ax.text(0.5, 0.5, "No per-frame metrics available", transform=ax.transAxes,
                ha="center", va="center", fontsize=10, color="gray")
    ax.axvline(boundary_x, color="red", linewidth=2, linestyle="--", alpha=0.8)

    # X-axis formatting
    if fps:
        axes[-1].set_xlabel("Time (MM:SS)")
        axes[-1].xaxis.set_major_formatter(mticker.FuncFormatter(_format_mmss))
    else:
        axes[-1].set_xlabel("Frame Index")

    fig.tight_layout()
    save_path = output_dir / f"boundary_metrics_chunk{chunk_idx}.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize metrics around chunk boundaries"
    )
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--window", type=int, default=20, help="Frames ± boundary")
    parser.add_argument("--video-path", type=str, default=None,
                        help="Path to video (for FPS; optional if per_frame has it)")
    args = parser.parse_args()

    run_results = load_run_results(args.run_dir)
    tracking_df = run_results["tracking_df"]
    chunk_info = run_results["chunk_info"]
    per_frame_df = run_results["per_frame_df"]

    if tracking_df is None or chunk_info is None:
        print("Missing tracking_outputs.parquet or chunk_info.json")
        return

    boundaries = get_chunk_boundaries(chunk_info)
    output_dir = create_output_directory(args.run_dir)

    fps = None
    if args.video_path:
        fps = get_video_fps(args.video_path)
    if fps is None or fps == 0:
        # Try to infer from frame count and chunk info
        fps = 25.0  # fallback
        print(f"Using fallback FPS: {fps}")

    for boundary in boundaries:
        chunk_idx = boundary["chunk_idx"]
        print(f"Boundary {chunk_idx-1}→{chunk_idx} at frame {boundary['boundary_frame']}")
        plot_boundary_metrics(
            tracking_df, per_frame_df, boundary, fps, args.window, output_dir
        )


if __name__ == "__main__":
    main()
