"""
Visualization functions for SAM3 tracking metrics.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _frame_to_seconds(frame_idx, fps):
    """Convert frame index to seconds."""
    return frame_idx / fps


def _format_mmss(seconds, _pos=None):
    """Format seconds as MM:SS for matplotlib tick labels."""
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def _setup_time_xaxis(ax, frames, fps):
    """Configure x-axis as MM:SS if fps is available, else frame index."""
    if fps is not None:
        ax.set_xlabel("Time (MM:SS)")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(_format_mmss))
    else:
        ax.set_xlabel("Frame Index")


def _save_or_show(fig, save_path):
    """Save figure to path (and close) or show interactively."""
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def _frame_to_x(frame_idx, fps):
    """Convert frame index to x-axis value (seconds if fps, else frame)."""
    if fps is not None:
        return _frame_to_seconds(frame_idx, fps)
    return frame_idx


# ---------------------------------------------------------------------------
# Plot 1: ID Timeline with trouble-spot overlay
# ---------------------------------------------------------------------------


def plot_id_timeline(tracking_df, per_frame_df=None, fps=None, save_path=None):
    """
    Event-plot style ID timeline showing when each object ID exists.

    Bars are color-coded by tracker_score (green=high, red=low).
    Optional trouble-spot overlay shades occlusion and count-change frames.

    Args:
        tracking_df: MultiIndex DataFrame (frame_idx, object_id) with
                     'tracker_score' column.
        per_frame_df: DataFrame with 'frame_idx', 'is_high_occlusion',
                      'is_object_count_change' columns. Optional.
        fps: Video frames per second (for MM:SS x-axis). None = frame index.
        save_path: Path to save PNG. None = plt.show().
    """
    fig, ax = plt.subplots(figsize=(14, 4))

    # Overlay trouble spots first (behind bars)
    if per_frame_df is not None:
        occlusion_frames = per_frame_df.loc[
            per_frame_df["is_high_occlusion"], "frame_idx"
        ].values
        count_change_frames = per_frame_df.loc[
            per_frame_df["is_object_count_change"], "frame_idx"
        ].values

        for f in occlusion_frames:
            x = _frame_to_x(f, fps)
            w = _frame_to_x(1, fps) if fps else 1
            ax.axvspan(x - w / 2, x + w / 2, color="red", alpha=0.10, linewidth=0)
        for f in count_change_frames:
            x = _frame_to_x(f, fps)
            w = _frame_to_x(1, fps) if fps else 1
            ax.axvspan(
                x - w / 2, x + w / 2, color="orange", alpha=0.15, linewidth=0
            )

    # Extract unique object IDs and sort
    object_ids = sorted(
        tracking_df.index.get_level_values("object_id").unique().tolist()
    )
    id_to_y = {oid: i for i, oid in enumerate(object_ids)}

    # Colormap for tracker_score
    cmap = plt.cm.RdYlGn  # red (low) → yellow → green (high)
    norm = Normalize(vmin=0.0, vmax=1.0)

    bar_height = 0.6
    has_tracker_score = "tracker_score" in tracking_df.columns

    for oid in object_ids:
        oid_data = tracking_df.xs(oid, level="object_id")
        frames = sorted(oid_data.index.tolist())
        if not frames:
            continue

        # Find contiguous segments
        segments = []
        seg_start = frames[0]
        prev = frames[0]
        for f in frames[1:]:
            if f != prev + 1:
                segments.append((seg_start, prev))
                seg_start = f
            prev = f
        segments.append((seg_start, prev))

        y = id_to_y[oid]
        for start, end in segments:
            x_start = _frame_to_x(start, fps)
            x_end = _frame_to_x(end, fps)
            width = x_end - x_start + (_frame_to_x(1, fps) if fps else 1)

            # Color by mean tracker_score in this segment
            if has_tracker_score:
                seg_data = oid_data.loc[start:end, "tracker_score"]
                mean_score = seg_data.mean()
                if pd.isna(mean_score):
                    mean_score = 0.5
                color = cmap(norm(mean_score))
            else:
                color = cmap(norm(0.7))

            ax.barh(
                y,
                width,
                left=x_start,
                height=bar_height,
                color=color,
                edgecolor="none",
            )

    # Y-axis: object IDs
    ax.set_yticks(range(len(object_ids)))
    ax.set_yticklabels([f"ID {oid}" for oid in object_ids])
    ax.set_ylim(-0.5, len(object_ids) - 0.5)

    # X-axis
    _setup_time_xaxis(ax, None, fps)

    # Colorbar for tracker score
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, aspect=30)
    cbar.set_label("Tracker Score")

    # Legend for trouble spots
    legend_handles = []
    if per_frame_df is not None:
        if len(occlusion_frames) > 0:
            legend_handles.append(
                plt.Rectangle((0, 0), 1, 1, fc="red", alpha=0.15, label="Occlusion")
            )
        if len(count_change_frames) > 0:
            legend_handles.append(
                plt.Rectangle(
                    (0, 0), 1, 1, fc="orange", alpha=0.2, label="Count Change"
                )
            )
    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper right", fontsize=8)

    ax.set_title("Object ID Timeline")
    fig.tight_layout()
    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Plot 2: Per-frame metrics dashboard
# ---------------------------------------------------------------------------


def plot_per_frame_dashboard(per_frame_df, fps=None, save_path=None):
    """
    5-panel timeseries dashboard of per-frame tracking metrics.

    Panels: object count, max mask IoU, min centroid distance,
    clustering coefficient, mean mask area (with min/max band).

    Args:
        per_frame_df: DataFrame with per-frame metric columns.
        fps: Video FPS for MM:SS x-axis. None = frame index.
        save_path: Path to save PNG. None = plt.show().
    """
    frames = per_frame_df["frame_idx"].values
    x = np.array([_frame_to_x(f, fps) for f in frames])

    fig, axes = plt.subplots(5, 1, sharex=True, figsize=(14, 12))

    # Panel 1: Object count (step plot)
    ax = axes[0]
    ax.step(x, per_frame_df["num_objects"].values, where="mid", color="steelblue")
    ax.set_ylabel("Object Count")
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.set_title("Per-Frame Tracking Metrics Dashboard")

    # Panel 2: Max pairwise mask IoU
    ax = axes[1]
    iou_vals = per_frame_df["max_pairwise_mask_iou"].values
    ax.plot(x, iou_vals, color="tomato", linewidth=0.8)
    ax.axhline(0.15, color="red", linestyle="--", linewidth=0.8, alpha=0.6, label="Threshold (0.15)")
    ax.fill_between(x, iou_vals, 0.15, where=iou_vals > 0.15, color="red", alpha=0.15)
    ax.set_ylabel("Max Mask IoU")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8, loc="upper right")

    # Panel 3: Min centroid distance
    ax = axes[2]
    min_dist = per_frame_df["min_centroid_distance"].values.copy()
    # Replace inf with NaN for plotting
    min_dist[np.isinf(min_dist)] = np.nan
    ax.plot(x, min_dist, color="mediumpurple", linewidth=0.8)
    ax.set_ylabel("Min Centroid Dist (px)")

    # Panel 4: Clustering coefficient
    ax = axes[3]
    ax.plot(x, per_frame_df["clustering_coefficient"].values, color="teal", linewidth=0.8)
    ax.set_ylabel("Clustering Coeff")
    ax.set_ylim(-0.05, 1.05)

    # Panel 5: Mean mask area with min/max band
    ax = axes[4]
    mean_area = per_frame_df["mean_mask_area"].values
    min_area = per_frame_df["min_mask_area"].values
    max_area = per_frame_df["max_mask_area"].values
    ax.plot(x, mean_area, color="darkorange", linewidth=0.8, label="Mean")
    ax.fill_between(x, min_area, max_area, color="orange", alpha=0.2, label="Min–Max")
    ax.set_ylabel("Mask Area (px)")
    ax.legend(fontsize=8, loc="upper right")

    # Shared x-axis
    _setup_time_xaxis(axes[-1], frames, fps)

    fig.tight_layout()
    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Plot 3: Per-ID tracker scores over time
# ---------------------------------------------------------------------------


def plot_per_id_scores(tracking_df, fps=None, save_path=None):
    """
    Line plot of tracker_score over time for each object ID.

    Args:
        tracking_df: MultiIndex DataFrame (frame_idx, object_id) with
                     'tracker_score' column.
        fps: Video FPS for MM:SS x-axis. None = frame index.
        save_path: Path to save PNG. None = plt.show().
    """
    fig, ax = plt.subplots(figsize=(14, 4))

    object_ids = sorted(
        tracking_df.index.get_level_values("object_id").unique().tolist()
    )
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(object_ids), 1)))

    for oid, color in zip(object_ids, colors):
        oid_data = tracking_df.xs(oid, level="object_id")
        frames = oid_data.index.values
        x = np.array([_frame_to_x(f, fps) for f in frames])
        scores = oid_data["tracker_score"].values
        ax.plot(x, scores, color=color, linewidth=0.8, alpha=0.8, label=f"ID {oid}")

    ax.set_ylabel("Tracker Score")
    ax.set_ylim(-0.05, 1.05)
    _setup_time_xaxis(ax, None, fps)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Per-ID Tracker Score Over Time")

    fig.tight_layout()
    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------


def generate_all_visualizations(tracking_df, per_frame_df, output_dir, fps=None):
    """
    Generate all tracking visualizations and save to output_dir.

    Args:
        tracking_df: MultiIndex DataFrame from process_tracking_outputs.
        per_frame_df: DataFrame from per_frame_metrics_to_df.
        output_dir: Directory to save PNG files.
        fps: Video FPS for MM:SS axis labels.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_id_timeline(
        tracking_df,
        per_frame_df,
        fps=fps,
        save_path=output_dir / "id_timeline.png",
    )
    plot_per_frame_dashboard(
        per_frame_df,
        fps=fps,
        save_path=output_dir / "per_frame_dashboard.png",
    )
    plot_per_id_scores(
        tracking_df,
        fps=fps,
        save_path=output_dir / "per_id_scores.png",
    )
