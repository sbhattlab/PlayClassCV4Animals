"""
Compare equidistant vs random point extraction methods on actual tracking data.

Loads masks from a previous run's tracking_outputs.parquet, applies both methods,
and generates visual comparisons and statistics to prove the equidistant method
produces better prompt points for chunk-boundary handoff.

Usage:
    python -m script.sam3.compare_point_extraction \
        --run-dir sandbox/output/results/sam3-hf/20260206_164707_sam3_hf \
        --video-path ext-data/test/video_1_5_min.mp4 \
        --source-frame 1369
"""

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from script.sam3.diagnostic_utils import (
    create_output_directory,
    decode_rle_mask,
    get_masks_for_frame,
    load_run_results,
    read_video_frame,
)
from src.utils import extract_equidistant_points_from_mask, sample_points_from_masks


def find_source_frames(run_results: dict) -> list[int]:
    """Find all source frame indices from chunk_info."""
    chunk_info = run_results["chunk_info"]
    if chunk_info is None:
        return []
    frames = []
    for chunk in chunk_info.get("chunks", []):
        src = chunk.get("source_frame_idx")
        if src is not None:
            frames.append(int(src))
    return frames


def compute_border_distance(point: list, width: int, height: int) -> float:
    """Minimum distance from a point to any frame border."""
    x, y = point
    return min(x, y, width - 1 - x, height - 1 - y)


def run_comparison(
    tracking_df: pd.DataFrame,
    frame_idx: int,
    video_path: str | None,
    output_dir: Path,
    num_random_runs: int = 10,
    num_points: int = 3,
):
    """
    Compare point extraction methods for all objects at a given frame.

    Args:
        tracking_df: MultiIndex DataFrame (frame_idx, object_id).
        frame_idx: Source frame to extract points from.
        video_path: Optional path to video for background image.
        output_dir: Directory to save output images and stats.
        num_random_runs: Number of random sampling runs for statistics.
        num_points: Points per mask.
    """
    masks = get_masks_for_frame(tracking_df, frame_idx)
    if not masks:
        print(f"No masks found at frame {frame_idx}")
        return

    # Read background frame if video available
    bg_frame = None
    if video_path and Path(video_path).exists():
        bg_frame = read_video_frame(video_path, frame_idx)
        if bg_frame is not None:
            bg_frame = cv2.cvtColor(bg_frame, cv2.COLOR_BGR2RGB)

    object_ids = sorted(masks.keys())
    h, w = next(iter(masks.values())).shape

    # Collect statistics
    stats_rows = []

    for obj_id in object_ids:
        mask = masks[obj_id]

        # Equidistant method
        equi_pts = extract_equidistant_points_from_mask(mask, num_points)
        if equi_pts is None:
            continue

        # Random method - multiple runs
        mask_batch = mask[np.newaxis, :, :]  # (1, H, W)
        random_all_pts = []
        for _ in range(num_random_runs):
            rand_pts = sample_points_from_masks(mask_batch, num_points)
            random_all_pts.append(rand_pts[0].tolist())  # (num_points, 2)

        # Compute statistics for equidistant
        equi_border_dists = [compute_border_distance(p, w, h) for p in equi_pts]
        equi_x = [p[0] for p in equi_pts]
        equi_spread_x = np.std(equi_x) if len(equi_x) > 1 else 0.0

        stats_rows.append({
            "object_id": obj_id,
            "method": "equidistant",
            "min_border_dist": min(equi_border_dists),
            "mean_border_dist": np.mean(equi_border_dists),
            "spread_x": equi_spread_x,
            "consistency": 0.0,  # deterministic
            "points": str(equi_pts),
        })

        # Compute statistics for random (aggregated)
        all_border_dists = []
        all_spreads = []
        for pts in random_all_pts:
            dists = [compute_border_distance(p, w, h) for p in pts]
            all_border_dists.append(min(dists))
            xs = [p[0] for p in pts]
            all_spreads.append(np.std(xs) if len(xs) > 1 else 0.0)

        # Consistency = std of x-coordinates across runs (higher = less consistent)
        all_first_x = [pts[0][0] for pts in random_all_pts]
        consistency = np.std(all_first_x) if len(all_first_x) > 1 else 0.0

        stats_rows.append({
            "object_id": obj_id,
            "method": f"random (avg of {num_random_runs})",
            "min_border_dist": np.mean(all_border_dists),
            "mean_border_dist": np.mean([
                np.mean([compute_border_distance(p, w, h) for p in pts])
                for pts in random_all_pts
            ]),
            "spread_x": np.mean(all_spreads),
            "consistency": consistency,
            "points": f"{num_random_runs} runs",
        })

        stats_rows.append({
            "object_id": obj_id,
            "method": "random (worst)",
            "min_border_dist": min(all_border_dists),
            "mean_border_dist": min([
                np.mean([compute_border_distance(p, w, h) for p in pts])
                for pts in random_all_pts
            ]),
            "spread_x": min(all_spreads),
            "consistency": consistency,
            "points": "worst single run",
        })

        # --- Visualization for this object ---
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for ax_idx, (ax, title) in enumerate(zip(
            axes, [f"Equidistant (ID {obj_id})", f"Random x{num_random_runs} (ID {obj_id})"]
        )):
            # Background
            if bg_frame is not None:
                ax.imshow(bg_frame, alpha=0.7)
            # Mask overlay
            mask_rgba = np.zeros((*mask.shape, 4))
            mask_rgba[mask > 0] = [0.2, 0.6, 1.0, 0.3]
            ax.imshow(mask_rgba)

            if ax_idx == 0:
                # Equidistant points
                pts_arr = np.array(equi_pts)
                ax.scatter(
                    pts_arr[:, 0], pts_arr[:, 1],
                    c="lime", s=120, marker="*", edgecolors="black",
                    linewidths=1.0, zorder=10, label="Equidistant"
                )
                for i, pt in enumerate(equi_pts):
                    ax.annotate(
                        f"({pt[0]}, {pt[1]})", (pt[0], pt[1]),
                        textcoords="offset points", xytext=(5, 5),
                        fontsize=7, color="lime",
                        bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.7),
                    )
            else:
                # Random points from all runs
                for run_idx, pts in enumerate(random_all_pts):
                    pts_arr = np.array(pts)
                    alpha = 0.4
                    ax.scatter(
                        pts_arr[:, 0], pts_arr[:, 1],
                        c="red", s=40, marker="o", alpha=alpha, edgecolors="darkred",
                        linewidths=0.5, zorder=5,
                    )
                # Highlight border points (< 50px from edge)
                for pts in random_all_pts:
                    for pt in pts:
                        if compute_border_distance(pt, w, h) < 50:
                            ax.scatter(
                                pt[0], pt[1], c="yellow", s=100, marker="x",
                                linewidths=2, zorder=11,
                            )

            # Border danger zones
            border_thresh = 50
            ax.axvline(border_thresh, color="red", linestyle="--", alpha=0.3)
            ax.axvline(w - border_thresh, color="red", linestyle="--", alpha=0.3)
            ax.axhline(border_thresh, color="red", linestyle="--", alpha=0.3)
            ax.axhline(h - border_thresh, color="red", linestyle="--", alpha=0.3)

            ax.set_xlim(0, w)
            ax.set_ylim(h, 0)
            ax.set_title(title)
            ax.set_xlabel("x")
            ax.set_ylabel("y")

        fig.suptitle(
            f"Point Extraction Comparison — Frame {frame_idx}, Object ID {obj_id}\n"
            f"Frame size: {w}x{h} | Red dashed = 50px border zone",
            fontsize=11,
        )
        fig.tight_layout()
        save_path = output_dir / f"comparison_frame{frame_idx}_obj{obj_id}.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {save_path}")

    # Save statistics table
    stats_df = pd.DataFrame(stats_rows)
    stats_path = output_dir / f"comparison_stats_frame{frame_idx}.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"\nStatistics saved: {stats_path}")
    print(stats_df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description="Compare equidistant vs random point extraction methods"
    )
    parser.add_argument("--run-dir", type=str, required=True, help="Path to SAM3 run directory")
    parser.add_argument("--video-path", type=str, default=None, help="Path to source video")
    parser.add_argument(
        "--source-frame", type=int, default=None,
        help="Frame index to analyze (default: use all source frames from chunk_info)",
    )
    parser.add_argument("--num-random-runs", type=int, default=10, help="Random sampling iterations")
    parser.add_argument("--num-points", type=int, default=3, help="Points per mask")
    args = parser.parse_args()

    run_results = load_run_results(args.run_dir)
    tracking_df = run_results["tracking_df"]
    if tracking_df is None:
        print(f"No tracking_outputs.parquet found in {args.run_dir}")
        return

    output_dir = create_output_directory(args.run_dir, "diagnostic_visualizations")

    # Determine which frames to analyze
    if args.source_frame is not None:
        source_frames = [args.source_frame]
    else:
        source_frames = find_source_frames(run_results)
        if not source_frames:
            # Fall back to last frame
            all_frames = sorted(tracking_df.index.get_level_values("frame_idx").unique())
            source_frames = [all_frames[-1]] if all_frames else []

    print(f"Analyzing {len(source_frames)} source frames: {source_frames}")
    for frame_idx in source_frames:
        print(f"\n{'='*60}")
        print(f"Frame {frame_idx}")
        print(f"{'='*60}")
        run_comparison(
            tracking_df, frame_idx, args.video_path, output_dir,
            num_random_runs=args.num_random_runs,
            num_points=args.num_points,
        )


if __name__ == "__main__":
    main()
