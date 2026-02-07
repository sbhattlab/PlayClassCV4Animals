"""
Visualize mask quality evolution across chunk boundaries.

Creates a 2x3 grid showing frames around a chunk boundary with
segmentation masks overlaid, bounding boxes, and annotation text
(frame index, tracker scores, bbox dimensions).

Usage:
    python -m script.sam3.plot_mask_evolution \
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

OBJ_COLORS = [
    (0.2, 0.8, 0.2),
    (0.2, 0.4, 1.0),
    (1.0, 0.6, 0.0),
    (0.8, 0.2, 0.8),
    (0.0, 0.8, 0.8),
    (1.0, 0.2, 0.2),
]


def get_tracker_scores_for_frame(tracking_df, frame_idx: int) -> dict:
    """Get object_id -> tracker_score mapping for a frame."""
    if frame_idx not in tracking_df.index.get_level_values("frame_idx"):
        return {}
    frame_data = tracking_df.xs(frame_idx, level="frame_idx")
    scores = {}
    for obj_id in frame_data.index:
        row = frame_data.loc[obj_id]
        scores[int(obj_id)] = row.get("tracker_score")
    return scores


def get_bboxes_for_frame(tracking_df, frame_idx: int) -> dict:
    """Get object_id -> bbox mapping for a frame."""
    if frame_idx not in tracking_df.index.get_level_values("frame_idx"):
        return {}
    frame_data = tracking_df.xs(frame_idx, level="frame_idx")
    bboxes = {}
    for obj_id in frame_data.index:
        row = frame_data.loc[obj_id]
        bboxes[int(obj_id)] = row.get("bbox")
    return bboxes


def plot_single_frame(
    ax, frame_rgb, masks: dict, scores: dict, bboxes: dict,
    frame_idx: int, fps: float, is_boundary: bool,
):
    """Plot a single frame with mask overlay, bboxes, and annotations."""
    ax.imshow(frame_rgb)

    for i, (obj_id, mask) in enumerate(sorted(masks.items())):
        color = OBJ_COLORS[i % len(OBJ_COLORS)]

        # Mask overlay
        mask_rgba = np.zeros((*mask.shape, 4))
        mask_rgba[mask > 0] = [*color, 0.35]
        ax.imshow(mask_rgba)

        # Bbox
        bbox = bboxes.get(obj_id)
        if bbox and isinstance(bbox, list) and len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            bw, bh = x2 - x1, y2 - y1
            rect = plt.Rectangle(
                (x1, y1), bw, bh,
                linewidth=1.5, edgecolor=color, facecolor="none",
            )
            ax.add_patch(rect)

            # Label with score and bbox dims
            score = scores.get(obj_id)
            score_str = f"{score:.3f}" if score is not None else "N/A"
            label = f"ID{obj_id} s={score_str}\n{int(bw)}x{int(bh)}"
            ax.text(
                x1, y1 - 2, label, fontsize=6, color="white",
                verticalalignment="bottom",
                bbox=dict(boxstyle="round,pad=0.2", fc=(*color, 0.7), ec="none"),
            )

    time_str = frame_to_mmss(frame_idx, fps) if fps else ""
    border_tag = " [BOUNDARY]" if is_boundary else ""
    ax.set_title(f"Frame {frame_idx} ({time_str}){border_tag}", fontsize=9,
                 color="red" if is_boundary else "black",
                 fontweight="bold" if is_boundary else "normal")
    ax.axis("off")


def plot_mask_evolution_for_boundary(
    tracking_df, boundary: dict, video_path: str,
    fps: float, output_dir: Path,
):
    """Create 2x3 grid showing mask evolution across a chunk boundary."""
    boundary_frame = boundary["boundary_frame"]
    source_frame = boundary["source_frame_idx"]
    chunk_idx = boundary["chunk_idx"]

    # Pick 6 frames to display: source, boundary-1, boundary, boundary+1, +2, +5
    if source_frame is not None:
        display_frames = [
            source_frame,
            boundary_frame - 1,
            boundary_frame,
            boundary_frame + 1,
            boundary_frame + 2,
            boundary_frame + 5,
        ]
    else:
        display_frames = [
            boundary_frame - 2,
            boundary_frame - 1,
            boundary_frame,
            boundary_frame + 1,
            boundary_frame + 2,
            boundary_frame + 5,
        ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes_flat = axes.flatten()

    for ax_idx, frame_idx in enumerate(display_frames):
        ax = axes_flat[ax_idx]

        frame_bgr = read_video_frame(video_path, frame_idx)
        if frame_bgr is None:
            ax.text(0.5, 0.5, f"Frame {frame_idx}\nnot available",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=10, color="gray")
            ax.axis("off")
            continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        masks = get_masks_for_frame(tracking_df, frame_idx)
        scores = get_tracker_scores_for_frame(tracking_df, frame_idx)
        bboxes = get_bboxes_for_frame(tracking_df, frame_idx)

        plot_single_frame(
            ax, frame_rgb, masks, scores, bboxes,
            frame_idx, fps, is_boundary=(frame_idx == boundary_frame),
        )

    fig.suptitle(
        f"Mask Evolution — Chunk {chunk_idx-1}→{chunk_idx} "
        f"(boundary at frame {boundary_frame})",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()

    save_path = output_dir / f"mask_evolution_chunk{chunk_idx}.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize mask evolution across chunk boundaries"
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

    if args.boundary_idx is not None:
        boundaries = [b for b in boundaries if b["chunk_idx"] == args.boundary_idx]

    for boundary in boundaries:
        chunk_idx = boundary["chunk_idx"]
        print(f"Boundary {chunk_idx-1}→{chunk_idx} at frame {boundary['boundary_frame']}")
        plot_mask_evolution_for_boundary(
            tracking_df, boundary, args.video_path, fps, output_dir,
        )


if __name__ == "__main__":
    main()
