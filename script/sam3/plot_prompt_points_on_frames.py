"""
Visualize prompt points on source and target frames at chunk boundaries.

For each chunk transition, creates a 2-panel figure:
- Left: Source frame (end of previous chunk) with mask overlay and sampled points
- Right: First frame of next chunk with same point markers

Border points (within 50px of frame edge) are highlighted in red.

Usage:
    python -m script.sam3.plot_prompt_points_on_frames \
        --run-dir sandbox/output/results/sam3-hf/20260206_164707_sam3_hf \
        --video-path ext-data/test/video_1_5_min.mp4
"""

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from script.sam3.diagnostic_utils import (
    create_output_directory,
    frame_to_mmss,
    get_chunk_boundaries,
    get_masks_for_frame,
    get_video_fps,
    load_run_results,
    read_video_frame,
)

# Distinct colors for object IDs
OBJ_COLORS = [
    (0.2, 0.8, 0.2),   # green
    (0.2, 0.4, 1.0),   # blue
    (1.0, 0.6, 0.0),   # orange
    (0.8, 0.2, 0.8),   # purple
    (0.0, 0.8, 0.8),   # cyan
    (1.0, 0.2, 0.2),   # red
]

BORDER_THRESHOLD = 50  # pixels


def is_border_point(x: float, y: float, width: int, height: int) -> bool:
    """Check if a point is within BORDER_THRESHOLD of the frame edge."""
    return x < BORDER_THRESHOLD or x > width - BORDER_THRESHOLD or \
           y < BORDER_THRESHOLD or y > height - BORDER_THRESHOLD


def plot_frame_with_points(
    ax, frame_rgb, masks: dict, prompt_points: dict | None,
    title: str, width: int, height: int
):
    """Plot a single frame with mask overlay and prompt points."""
    ax.imshow(frame_rgb)

    # Overlay masks (semi-transparent)
    for i, (obj_id, mask) in enumerate(sorted(masks.items())):
        color = OBJ_COLORS[i % len(OBJ_COLORS)]
        mask_rgba = np.zeros((*mask.shape, 4))
        mask_rgba[mask > 0] = [*color, 0.3]
        ax.imshow(mask_rgba)

        # Draw bbox
        ys, xs = np.where(mask > 0)
        if len(ys) > 0:
            x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
            rect = plt.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=1.5, edgecolor=color, facecolor="none"
            )
            ax.add_patch(rect)
            ax.text(x1, y1 - 3, f"ID {obj_id}", fontsize=7, color=color,
                    bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.6))

    # Overlay prompt points
    if prompt_points:
        for i, (obj_id, pts) in enumerate(sorted(prompt_points.items())):
            color = OBJ_COLORS[i % len(OBJ_COLORS)]
            for pt in pts:
                x, y = pt[0], pt[1]
                if is_border_point(x, y, width, height):
                    # Border point - highlight in red
                    ax.scatter(x, y, c="red", s=150, marker="X", edgecolors="yellow",
                               linewidths=1.5, zorder=12)
                    ax.annotate(
                        f"BORDER ({x:.0f},{y:.0f})", (x, y),
                        textcoords="offset points", xytext=(8, 8), fontsize=6,
                        color="red", fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.2", fc="yellow", alpha=0.8),
                    )
                else:
                    # Interior point
                    ax.scatter(x, y, c=[color], s=100, marker="*", edgecolors="white",
                               linewidths=0.8, zorder=10)
                    ax.annotate(
                        f"({x:.0f},{y:.0f})", (x, y),
                        textcoords="offset points", xytext=(5, 5), fontsize=6,
                        color="white",
                        bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.6),
                    )

    # Border danger zones
    ax.axvline(BORDER_THRESHOLD, color="red", linestyle="--", alpha=0.2)
    ax.axvline(width - BORDER_THRESHOLD, color="red", linestyle="--", alpha=0.2)
    ax.axhline(BORDER_THRESHOLD, color="red", linestyle="--", alpha=0.2)
    ax.axhline(height - BORDER_THRESHOLD, color="red", linestyle="--", alpha=0.2)

    ax.set_title(title, fontsize=9)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize prompt points at chunk boundaries"
    )
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--video-path", type=str, required=True)
    parser.add_argument("--boundary-idx", type=int, default=None,
                        help="Specific boundary index (default: all)")
    args = parser.parse_args()

    run_results = load_run_results(args.run_dir)
    tracking_df = run_results["tracking_df"]
    chunk_info = run_results["chunk_info"]

    if tracking_df is None or chunk_info is None:
        print("Missing tracking_outputs.parquet or chunk_info.json")
        return

    boundaries = get_chunk_boundaries(chunk_info)
    fps = get_video_fps(args.video_path)
    output_dir = create_output_directory(args.run_dir)

    # Get frame dimensions from first mask
    all_frames = sorted(tracking_df.index.get_level_values("frame_idx").unique())
    sample_masks = get_masks_for_frame(tracking_df, all_frames[0])
    if sample_masks:
        h, w = next(iter(sample_masks.values())).shape
    else:
        frame = read_video_frame(args.video_path, 0)
        h, w = frame.shape[:2]

    # Filter boundaries if requested
    if args.boundary_idx is not None:
        boundaries = [b for b in boundaries if b["chunk_idx"] == args.boundary_idx]

    for boundary in boundaries:
        chunk_idx = boundary["chunk_idx"]
        source_frame = boundary["source_frame_idx"]
        target_frame = boundary["boundary_frame"]
        prompt_points = boundary["prompt_points"]

        if source_frame is None:
            continue

        print(f"Chunk {chunk_idx}: source={source_frame}, target={target_frame}")

        # Read frames
        src_frame_bgr = read_video_frame(args.video_path, source_frame)
        tgt_frame_bgr = read_video_frame(args.video_path, target_frame)
        if src_frame_bgr is None or tgt_frame_bgr is None:
            print(f"  Could not read frames, skipping")
            continue

        src_frame_rgb = cv2.cvtColor(src_frame_bgr, cv2.COLOR_BGR2RGB)
        tgt_frame_rgb = cv2.cvtColor(tgt_frame_bgr, cv2.COLOR_BGR2RGB)

        # Get masks
        src_masks = get_masks_for_frame(tracking_df, source_frame)
        tgt_masks = get_masks_for_frame(tracking_df, target_frame)

        # Count border points
        n_border = 0
        n_total = 0
        if prompt_points:
            for pts in prompt_points.values():
                for pt in pts:
                    n_total += 1
                    if is_border_point(pt[0], pt[1], w, h):
                        n_border += 1

        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        src_time = frame_to_mmss(source_frame, fps) if fps else str(source_frame)
        tgt_time = frame_to_mmss(target_frame, fps) if fps else str(target_frame)

        plot_frame_with_points(
            axes[0], src_frame_rgb, src_masks, prompt_points,
            f"Source: frame {source_frame} ({src_time}) — end of chunk {chunk_idx-1}",
            w, h,
        )
        plot_frame_with_points(
            axes[1], tgt_frame_rgb, tgt_masks, prompt_points,
            f"Target: frame {target_frame} ({tgt_time}) — start of chunk {chunk_idx}",
            w, h,
        )

        border_pct = f"{n_border}/{n_total} ({100*n_border/max(n_total,1):.0f}%)"
        fig.suptitle(
            f"Chunk Boundary {chunk_idx-1}→{chunk_idx} | "
            f"Border points: {border_pct} | "
            f"Objects: {len(prompt_points) if prompt_points else 0}",
            fontsize=11,
        )
        fig.tight_layout()

        save_path = output_dir / f"prompt_points_boundary_{chunk_idx}.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {save_path}")


if __name__ == "__main__":
    main()
