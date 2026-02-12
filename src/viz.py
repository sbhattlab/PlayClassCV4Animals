"""
Visualization functions for SAM3 tracking metrics and diagnostics.
"""

from collections import Counter
from pathlib import Path
from typing import Optional, Sequence

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import pycocotools.mask as mask_util
import seaborn as sns
from loguru import logger
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from sklearn.decomposition import PCA

from src.utils import prescan_occlusion_periods

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_OBJ_COLORS = [
    (0.2, 0.8, 0.2),  # green
    (0.2, 0.4, 1.0),  # blue
    (1.0, 0.6, 0.0),  # orange
    (0.8, 0.2, 0.8),  # purple
    (0.0, 0.8, 0.8),  # cyan
    (1.0, 0.2, 0.2),  # red
]

_BORDER_THRESHOLD = 50  # pixels


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
# Diagnostic helpers (from diagnostic_utils)
# ---------------------------------------------------------------------------


def _decode_rle_mask(counts, size):
    """Decode an RLE-encoded mask to a binary numpy array."""
    if isinstance(counts, str):
        counts = counts.encode("utf-8")
    rle = {"counts": counts, "size": size}
    return mask_util.decode(rle).astype(np.uint8)


def _read_video_frame(video_path, frame_idx):
    """Extract a single frame from a video file. Returns BGR or None."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    return frame


def _get_chunk_boundaries(chunk_info):
    """Parse chunk_info dict and return boundary transition metadata."""
    chunks = chunk_info.get("chunks", [])
    boundaries = []
    for i, chunk in enumerate(chunks):
        if i == 0:
            continue
        frame_range = chunk.get("frame_range", [0, 0])
        prev_range = chunks[i - 1].get("frame_range", [0, 0])
        prompt_points = chunk.get("prompt_points")
        if prompt_points:
            prompt_points = {int(k): v for k, v in prompt_points.items()}
        boundaries.append(
            {
                "chunk_idx": chunk.get("chunk_idx", i),
                "source_frame_idx": chunk.get("source_frame_idx"),
                "boundary_frame": frame_range[0],
                "prompt_points": prompt_points,
                "prev_chunk_end": prev_range[1] - 1,
                "model_type": chunk.get("model_type"),
            }
        )
    return boundaries


def _get_masks_for_frame(tracking_df, frame_idx):
    """Decode all masks for a given frame. Returns dict of obj_id -> mask."""
    if frame_idx not in tracking_df.index.get_level_values("frame_idx"):
        return {}
    frame_data = tracking_df.xs(frame_idx, level="frame_idx")
    masks = {}
    for obj_id in frame_data.index:
        row = frame_data.loc[obj_id]
        mask = _decode_rle_mask(row["counts"], row["size"])
        masks[int(obj_id)] = mask
    return masks


def _get_tracker_scores_for_frame(tracking_df, frame_idx):
    """Get object_id -> tracker_score mapping for a frame."""
    if frame_idx not in tracking_df.index.get_level_values("frame_idx"):
        return {}
    frame_data = tracking_df.xs(frame_idx, level="frame_idx")
    scores = {}
    for obj_id in frame_data.index:
        row = frame_data.loc[obj_id]
        scores[int(obj_id)] = row.get("tracker_score")
    return scores


def _get_bboxes_for_frame(tracking_df, frame_idx):
    """Get object_id -> bbox mapping for a frame."""
    if frame_idx not in tracking_df.index.get_level_values("frame_idx"):
        return {}
    frame_data = tracking_df.xs(frame_idx, level="frame_idx")
    bboxes = {}
    for obj_id in frame_data.index:
        row = frame_data.loc[obj_id]
        bboxes[int(obj_id)] = row.get("bbox")
    return bboxes


def _is_border_point(x, y, width, height):
    """Check if a point is within _BORDER_THRESHOLD of the frame edge."""
    return (
        x < _BORDER_THRESHOLD
        or x > width - _BORDER_THRESHOLD
        or y < _BORDER_THRESHOLD
        or y > height - _BORDER_THRESHOLD
    )


def plot_prescan_overview(
    # data: np.ndarray,
    # frame_indices: np.ndarray,
    # labels: Optional[np.ndarray] = None,
    # kmeans=None,
    small_image_shape: Optional[tuple[int, int]] = None,
    # data_mean: Optional[np.ndarray] = None,
    sample_images: Optional[Sequence[np.ndarray]] = None,
    max_clusters_to_show: int = 12,
):
    """
    Visualise prescan inputs and kmeans outputs.

    Args:
        data: (N, P) array used for clustering (samples x pixels).
        frame_indices: (N,) sampled frame indices in original video coords.
        labels: (N,) cluster labels (if not provided, will try to use kmeans.labels_).
        kmeans: fitted MiniBatchKMeans instance (optional).
        small_image_shape: (h, w) shape to reshape centers back into images
                           (used to display cluster centres). If None, centres
                           shown as 1D lines.
        data_mean: mean vector that was subtracted from data (to reconstruct
                   true intensity of cluster centres). If None, centres shown
                   as-is.
        sample_images: optional list/array of thumbnail frames corresponding to
                       frame_indices (useful to show representative frames per cluster).
        max_clusters_to_show: cap how many cluster-centre thumbnails to draw.
    """
    from src.debug import load_inputs

    cfg, video_info = load_inputs()
    sns.set(style="whitegrid")
    small_h = 24  # set to the height used when downsampling (new_h)
    small_w = 30

    prescan_results = prescan_occlusion_periods(
        video_path=cfg.video_path, fps=video_info.fps
    )
    # Use explicit fields returned by PrescanResult: original, centred and mean
    # Keep data (centred) as the clustering input for visualisations and
    # use data_mean to reconstruct cluster-centre thumbnails when needed.
    data_original = prescan_results.data_original
    data_centered = prescan_results.data_centered
    data_mean = prescan_results.data_mean
    data = data_centered
    kmeans = prescan_results.kmeans
    # backward-compatible label accessor
    if hasattr(prescan_results, "cluster_labels"):
        labels = prescan_results.cluster_labels
    else:
        labels = prescan_results.labels()
    frame_indices = prescan_results.frame_indices

    assert data.ndim == 2, "data should be 2D (samples x pixels)"
    n, p = data.shape

    if labels is None and kmeans is not None:
        labels = np.asarray(kmeans.labels_)
    if labels is None:
        labels = np.zeros(n, dtype=int)

    # PCA scatter (2 components)
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(data)
    fig = plt.figure(constrained_layout=True, figsize=(14, 10))
    gs = fig.add_gridspec(3, 2)

    ax_scatter = fig.add_subplot(gs[0, 0])
    scatter_colors = labels if labels is not None else frame_indices
    sc = ax_scatter.scatter(pcs[:, 0], pcs[:, 1], c=scatter_colors, cmap="tab20", s=18)
    ax_scatter.set_title("PCA scatter (samples) — colored by cluster")
    ax_scatter.set_xlabel("PC1")
    ax_scatter.set_ylabel("PC2")
    if labels is not None:
        # create legend for up to 12 clusters
        unique = np.unique(labels)
        if unique.size <= 12:
            for u in unique:
                pts = pcs[labels == u]
                if pts.size:
                    ax_scatter.scatter([], [], label=f"c{u}", c=[plt.cm.tab20(u % 20)])
            ax_scatter.legend(ncol=2, fontsize="small", loc="best")

    # Cluster timeline (1-row color bar)
    ax_timeline = fig.add_subplot(gs[0, 1])
    cmap = plt.get_cmap("tab20")
    # Ensure `labels` becomes a 2D array of shape (1, N) so imshow accepts it.
    timeline_img = np.atleast_2d(
        np.asarray(labels)
    )  # shape (1, N) or (M, N) if labels already 2D
    ax_timeline.imshow(timeline_img, aspect="auto", cmap=cmap, interpolation="nearest")
    ax_timeline.set_yticks([])
    ax_timeline.set_xticks(np.linspace(0, n - 1, min(10, n)).astype(int))
    ax_timeline.set_xticklabels(
        [
            str(int(frame_indices[i]))
            for i in np.linspace(0, n - 1, min(10, n)).astype(int)
        ],
        rotation=45,
    )
    ax_timeline.set_title("Cluster assignment timeline (sampled frames)")

    # Heatmap of data (rows sorted by cluster)
    ax_heat = fig.add_subplot(gs[1, :])
    order = np.argsort(labels)
    sns.heatmap(
        data[order, :],
        ax=ax_heat,
        cmap="viridis",
        cbar_kws={"label": "gray-level"},
        xticklabels=False,
        yticklabels=False,
    )
    ax_heat.set_title("Sampled-frame × pixel heatmap (rows sorted by cluster)")

    # Cluster size histogram
    ax_hist = fig.add_subplot(gs[2, 0])
    counts = Counter(labels)
    xs = sorted(counts.keys())
    ys = [counts[x] for x in xs]
    ax_hist.bar(xs, ys, color=[plt.cm.tab20(x % 20) for x in xs])
    ax_hist.set_xlabel("cluster")
    ax_hist.set_ylabel("count")
    ax_hist.set_title("Cluster sizes")

    # Cluster centres (thumbnail grid or 1D)
    ax_centres = fig.add_subplot(gs[2, 1])
    if kmeans is not None and hasattr(kmeans, "cluster_centers_"):
        centers = np.asarray(kmeans.cluster_centers_)
        if data_mean is not None:
            centers = centers + data_mean.reshape(1, -1)
        n_centers = centers.shape[0]
        show_n = min(n_centers, max_clusters_to_show)
        # If small_image_shape provided, render as images grid
        if (
            small_image_shape is not None
            and small_image_shape[0] * small_image_shape[1] == p
        ):
            # build small montage
            h, w = small_image_shape
            # normalize each centre for display
            imgs = centers[:show_n].reshape(show_n, h, w)
            # plot as grid using imshow
            cols = int(np.ceil(show_n / 2))
            rows = int(np.ceil(show_n / cols))
            ax_centres.remove()
            fig2, axs = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
            axs = np.atleast_2d(axs)
            for idx in range(rows * cols):
                r = idx // cols
                c = idx % cols
                ax = axs[r, c]
                if idx < show_n:
                    im = imgs[idx]
                    vmin, vmax = np.percentile(im, (2, 98))
                    ax.imshow(im, cmap="gray", vmin=vmin, vmax=vmax)
                    ax.set_title(f"center {idx}")
                ax.axis("off")
            fig2.suptitle("Cluster centres (first N)")
            plt.tight_layout()
        else:
            # 1D plot of first show_n centres
            for i in range(show_n):
                ax_centres.plot(
                    centers[i], alpha=0.8, label=f"c{i}", color=plt.cm.tab20(i % 20)
                )
            ax_centres.set_title("Cluster centres (1D)")
            ax_centres.legend(fontsize="x-small", ncol=2)
    else:
        ax_centres.text(
            0.5, 0.5, "No kmeans.cluster_centers_ available", ha="center", va="center"
        )
        ax_centres.axis("off")

    plt.show()

    # If sample_images provided, show representative frames per cluster
    if sample_images is not None:
        unique_clusters = np.unique(labels)
        show_clusters = unique_clusters[:max_clusters_to_show]
        n_show = len(show_clusters)
        cols = min(6, n_show)
        rows = int(np.ceil(n_show / cols))
        fig_samp, axs = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
        axs = np.atleast_2d(axs)
        for i, c in enumerate(show_clusters):
            idxs = np.where(labels == c)[0]
            if len(idxs) == 0:
                continue
            # choose median index in cluster
            sel = idxs[len(idxs) // 2]
            img = sample_images[sel]
            r = i // cols
            ccol = i % cols
            ax = axs[r, ccol]
            ax.imshow(img[..., ::-1] if img.shape[-1] == 3 else img, cmap="gray")
            ax.set_title(f"cluster {c}\nframe {int(frame_indices[sel])}")
            ax.axis("off")
        # hide remaining axes
        for j in range(i + 1, rows * cols):
            r = j // cols
            ccol = j % cols
            axs[r, ccol].axis("off")
        fig_samp.suptitle("Representative thumbnails per cluster")
        plt.tight_layout()
        plt.show()


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
            ax.axvspan(x - w / 2, x + w / 2, color="orange", alpha=0.15, linewidth=0)

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

            # Color by mean score in this segment (prefer tracker_score, fall back to scores)
            if has_tracker_score:
                seg_data = oid_data.loc[start:end, "tracker_score"]
                mean_score = seg_data.mean()
                if pd.isna(mean_score) and "scores" in oid_data.columns:
                    mean_score = oid_data.loc[start:end, "scores"].mean()
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
    cbar.set_label("Object Score")

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
    ax.axhline(
        0.15,
        color="red",
        linestyle="--",
        linewidth=0.8,
        alpha=0.6,
        label="Threshold (0.15)",
    )
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
    ax.plot(
        x, per_frame_df["clustering_coefficient"].values, color="teal", linewidth=0.8
    )
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
    Line plot of object scores over time for each object ID.

    Uses the 'scores' column which is populated for both Sam3VideoModel
    (detection confidence) and Sam3TrackerVideoModel (sigmoid of
    object_score_logits). Falls back to 'tracker_score' if 'scores' has
    no valid data.

    Args:
        tracking_df: MultiIndex DataFrame (frame_idx, object_id) with
                     'scores' and/or 'tracker_score' columns.
        fps: Video FPS for MM:SS x-axis. None = frame index.
        save_path: Path to save PNG. None = plt.show().
    """
    fig, ax = plt.subplots(figsize=(14, 4))

    # Pick the best available score column
    score_col = "scores"
    if score_col not in tracking_df.columns or tracking_df[score_col].isna().all():
        score_col = "tracker_score"

    object_ids = sorted(
        tracking_df.index.get_level_values("object_id").unique().tolist()
    )
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(object_ids), 1)))

    for oid, color in zip(object_ids, colors):
        oid_data = tracking_df.xs(oid, level="object_id")
        frames = oid_data.index.values
        x = np.array([_frame_to_x(f, fps) for f in frames])
        scores = oid_data[score_col].values
        ax.plot(x, scores, color=color, linewidth=0.8, alpha=0.8, label=f"ID {oid}")

    ax.set_ylabel("Object Score")
    ax.set_ylim(-0.05, 1.05)
    _setup_time_xaxis(ax, None, fps)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Per-ID Object Score Over Time")

    fig.tight_layout()
    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Plot 4: Mask evolution across chunk boundaries
# ---------------------------------------------------------------------------


def _plot_single_frame(
    ax,
    frame_rgb,
    masks,
    scores,
    bboxes,
    frame_idx,
    fps,
    is_boundary,
):
    """Plot a single frame with mask overlay, bboxes, and annotations."""
    ax.imshow(frame_rgb)

    for i, (obj_id, mask) in enumerate(sorted(masks.items())):
        color = _OBJ_COLORS[i % len(_OBJ_COLORS)]

        # Mask overlay
        mask_rgba = np.zeros((*mask.shape, 4))
        mask_rgba[mask > 0] = [*color, 0.35]
        ax.imshow(mask_rgba)

        # Bbox
        bbox = bboxes.get(obj_id)
        if bbox is not None and len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            bw = x2 - x1
            bh = y2 - y1
            rect = mpatches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor=color,
                facecolor="none",
            )
            ax.add_patch(rect)

            score = scores.get(obj_id)
            score_str = f"{score:.3f}" if score is not None else "N/A"
            label = f"ID{obj_id} s={score_str}\n{int(bw)}x{int(bh)}"
            ax.text(
                x1,
                y1 - 2,
                label,
                fontsize=6,
                color="white",
                verticalalignment="bottom",
                bbox=dict(boxstyle="round,pad=0.2", fc=(*color, 0.7), ec="none"),
            )

    time_str = _format_mmss(_frame_to_seconds(frame_idx, fps)) if fps else ""
    border_tag = " [BOUNDARY]" if is_boundary else ""
    ax.set_title(
        f"Frame {frame_idx} ({time_str}){border_tag}",
        fontsize=9,
        color="red" if is_boundary else "black",
        fontweight="bold" if is_boundary else "normal",
    )
    ax.axis("off")


def plot_mask_evolution(tracking_df, chunk_info, video_path, fps=None, output_dir=None):
    """
    Visualize mask quality evolution across all chunk boundaries.

    Creates a 2x3 grid per boundary showing frames around the transition
    with segmentation masks overlaid, bounding boxes, and tracker scores.

    Args:
        tracking_df: MultiIndex DataFrame (frame_idx, object_id).
        chunk_info: Dict with 'chunks' key (from chunk_info.json).
        video_path: Path to the source video file.
        fps: Video FPS for time labels.
        output_dir: Directory to save PNGs. None = plt.show().
    """
    boundaries = _get_chunk_boundaries(chunk_info)
    if not boundaries:
        return

    for boundary in boundaries:
        boundary_frame = boundary["boundary_frame"]
        source_frame = boundary["source_frame_idx"]
        chunk_idx = boundary["chunk_idx"]

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

        for ax_idx, fidx in enumerate(display_frames):
            ax = axes_flat[ax_idx]

            frame_bgr = _read_video_frame(video_path, fidx)
            if frame_bgr is None:
                ax.text(
                    0.5,
                    0.5,
                    f"Frame {fidx}\nnot available",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="gray",
                )
                ax.axis("off")
                continue

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            masks = _get_masks_for_frame(tracking_df, fidx)
            scores = _get_tracker_scores_for_frame(tracking_df, fidx)
            bboxes = _get_bboxes_for_frame(tracking_df, fidx)

            _plot_single_frame(
                ax,
                frame_rgb,
                masks,
                scores,
                bboxes,
                fidx,
                fps,
                is_boundary=(fidx == boundary_frame),
            )

        fig.suptitle(
            f"Mask Evolution — Chunk {chunk_idx - 1}\u2192{chunk_idx} "
            f"(boundary at frame {boundary_frame})",
            fontsize=12,
            fontweight="bold",
        )
        fig.tight_layout()

        if output_dir is not None:
            save_path = Path(output_dir) / f"mask_evolution_chunk{chunk_idx}.png"
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"Saved mask evolution plot: {save_path}")
        else:
            plt.show()


# ---------------------------------------------------------------------------
# Plot 5: Prompt points on source/target frames
# ---------------------------------------------------------------------------


def _plot_frame_with_points(ax, frame_rgb, masks, prompt_points, title, width, height):
    """Plot a single frame with mask overlay and prompt points."""
    ax.imshow(frame_rgb)

    for i, (obj_id, mask) in enumerate(sorted(masks.items())):
        color = _OBJ_COLORS[i % len(_OBJ_COLORS)]
        mask_rgba = np.zeros((*mask.shape, 4))
        mask_rgba[mask > 0] = [*color, 0.3]
        ax.imshow(mask_rgba)

        ys, xs = np.where(mask > 0)
        if len(ys) > 0:
            x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
            rect = plt.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=1.5,
                edgecolor=color,
                facecolor="none",
            )
            ax.add_patch(rect)
            ax.text(
                x1,
                y1 - 3,
                f"ID {obj_id}",
                fontsize=7,
                color=color,
                bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.6),
            )

    if prompt_points:
        for i, (obj_id, pts) in enumerate(sorted(prompt_points.items())):
            color = _OBJ_COLORS[i % len(_OBJ_COLORS)]
            for pt in pts:
                x, y = pt[0], pt[1]
                if _is_border_point(x, y, width, height):
                    ax.scatter(
                        x,
                        y,
                        c="red",
                        s=150,
                        marker="X",
                        edgecolors="yellow",
                        linewidths=1.5,
                        zorder=12,
                    )
                    ax.annotate(
                        f"BORDER ({x:.0f},{y:.0f})",
                        (x, y),
                        textcoords="offset points",
                        xytext=(8, 8),
                        fontsize=6,
                        color="red",
                        fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.2", fc="yellow", alpha=0.8),
                    )
                else:
                    ax.scatter(
                        x,
                        y,
                        c=[color],
                        s=100,
                        marker="*",
                        edgecolors="white",
                        linewidths=0.8,
                        zorder=10,
                    )
                    ax.annotate(
                        f"({x:.0f},{y:.0f})",
                        (x, y),
                        textcoords="offset points",
                        xytext=(5, 5),
                        fontsize=6,
                        color="white",
                        bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.6),
                    )

    ax.axvline(_BORDER_THRESHOLD, color="red", linestyle="--", alpha=0.2)
    ax.axvline(width - _BORDER_THRESHOLD, color="red", linestyle="--", alpha=0.2)
    ax.axhline(_BORDER_THRESHOLD, color="red", linestyle="--", alpha=0.2)
    ax.axhline(height - _BORDER_THRESHOLD, color="red", linestyle="--", alpha=0.2)

    ax.set_title(title, fontsize=9)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)


def plot_prompt_points(tracking_df, chunk_info, video_path, fps=None, output_dir=None):
    """
    Visualize prompt points on source and target frames at chunk boundaries.

    For each chunk transition, creates a 2-panel figure:
    - Left: Source frame with mask overlay and sampled points
    - Right: First frame of next chunk with same point markers
    Border points (within 50px of frame edge) are highlighted in red.

    Args:
        tracking_df: MultiIndex DataFrame (frame_idx, object_id).
        chunk_info: Dict with 'chunks' key (from chunk_info.json).
        video_path: Path to the source video file.
        fps: Video FPS for time labels.
        output_dir: Directory to save PNGs. None = plt.show().
    """
    boundaries = _get_chunk_boundaries(chunk_info)
    if not boundaries:
        return

    # Get frame dimensions
    all_frames = sorted(tracking_df.index.get_level_values("frame_idx").unique())
    sample_masks = _get_masks_for_frame(tracking_df, all_frames[0])
    if sample_masks:
        h, w = next(iter(sample_masks.values())).shape
    else:
        frame = _read_video_frame(video_path, 0)
        if frame is None:
            return
        h, w = frame.shape[:2]

    for boundary in boundaries:
        chunk_idx = boundary["chunk_idx"]
        source_frame = boundary["source_frame_idx"]
        target_frame = boundary["boundary_frame"]
        prompt_points = boundary["prompt_points"]

        if source_frame is None:
            continue

        src_frame_bgr = _read_video_frame(video_path, source_frame)
        tgt_frame_bgr = _read_video_frame(video_path, target_frame)
        if src_frame_bgr is None or tgt_frame_bgr is None:
            continue

        src_frame_rgb = cv2.cvtColor(src_frame_bgr, cv2.COLOR_BGR2RGB)
        tgt_frame_rgb = cv2.cvtColor(tgt_frame_bgr, cv2.COLOR_BGR2RGB)

        src_masks = _get_masks_for_frame(tracking_df, source_frame)
        tgt_masks = _get_masks_for_frame(tracking_df, target_frame)

        # Count border points
        n_border = 0
        n_total = 0
        if prompt_points:
            for pts in prompt_points.values():
                for pt in pts:
                    n_total += 1
                    if _is_border_point(pt[0], pt[1], w, h):
                        n_border += 1

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        src_time = (
            _format_mmss(_frame_to_seconds(source_frame, fps))
            if fps
            else str(source_frame)
        )
        tgt_time = (
            _format_mmss(_frame_to_seconds(target_frame, fps))
            if fps
            else str(target_frame)
        )

        _plot_frame_with_points(
            axes[0],
            src_frame_rgb,
            src_masks,
            prompt_points,
            f"Source: frame {source_frame} ({src_time}) \u2014 end of chunk {chunk_idx - 1}",
            w,
            h,
        )
        _plot_frame_with_points(
            axes[1],
            tgt_frame_rgb,
            tgt_masks,
            prompt_points,
            f"Target: frame {target_frame} ({tgt_time}) \u2014 start of chunk {chunk_idx}",
            w,
            h,
        )

        border_pct = f"{n_border}/{n_total} ({100 * n_border / max(n_total, 1):.0f}%)"
        fig.suptitle(
            f"Chunk Boundary {chunk_idx - 1}\u2192{chunk_idx} | "
            f"Border points: {border_pct} | "
            f"Objects: {len(prompt_points) if prompt_points else 0}",
            fontsize=11,
        )
        fig.tight_layout()

        if output_dir is not None:
            save_path = Path(output_dir) / f"prompt_points_boundary_{chunk_idx}.png"
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"Saved prompt points plot: {save_path}")
        else:
            plt.show()


# ---------------------------------------------------------------------------
# Plot 6: YOLO prescan overview
# ---------------------------------------------------------------------------


def plot_yolo_prescan_overview(
    yolo_prescan_df,
    occlusion_periods=None,
    fps=None,
    save_path=None,
):
    """
    4-panel timeseries overview of YOLO-based prescan metrics.

    Panels: object count, max pairwise bbox IoU, clustering coefficient,
    high-occlusion flag. Occlusion periods are shaded in red across all panels.

    Args:
        yolo_prescan_df: DataFrame from yolo_prescan_to_df() with columns
            frame_idx, num_objects, max_pairwise_bbox_iou,
            clustering_coefficient, is_high_occlusion, mean_confidence.
        occlusion_periods: List of (start_frame, end_frame) tuples. Optional.
        fps: Video FPS for MM:SS x-axis. None = frame index.
        save_path: Path to save PNG. None = plt.show().
    """
    frames = yolo_prescan_df["frame_idx"].values
    x = np.array([_frame_to_x(f, fps) for f in frames])

    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(14, 10))

    def _shade_occlusion(ax):
        if occlusion_periods:
            for start, end in occlusion_periods:
                x_start = _frame_to_x(start, fps)
                x_end = _frame_to_x(end, fps)
                ax.axvspan(x_start, x_end, alpha=0.15, color="red", linewidth=0)

    # Panel 1: Object count
    ax = axes[0]
    ax.step(x, yolo_prescan_df["num_objects"].values, where="mid", color="steelblue",
            linewidth=0.8, alpha=0.8)
    ax.set_ylabel("# Objects")
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.set_title("YOLO Pre-scan Overview", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    _shade_occlusion(ax)

    # Panel 2: Max pairwise bbox IoU
    ax = axes[1]
    iou_vals = yolo_prescan_df["max_pairwise_bbox_iou"].values
    ax.plot(x, iou_vals, linewidth=0.8, color="tomato", alpha=0.8)
    ax.axhline(0.15, color="red", linestyle="--", linewidth=0.8, alpha=0.5,
               label="Threshold (0.15)")
    ax.fill_between(x, iou_vals, 0.15, where=iou_vals > 0.15, color="red", alpha=0.15)
    ax.set_ylabel("Max Bbox IoU")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    _shade_occlusion(ax)

    # Panel 3: Clustering coefficient
    ax = axes[2]
    ax.plot(x, yolo_prescan_df["clustering_coefficient"].values,
            linewidth=0.8, color="mediumpurple", alpha=0.8)
    ax.axhline(0.5, color="red", linestyle="--", linewidth=0.8, alpha=0.5,
               label="Threshold (0.5)")
    ax.set_ylabel("Clustering\nCoefficient")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    _shade_occlusion(ax)

    # Panel 4: High occlusion flag
    ax = axes[3]
    occ_flags = yolo_prescan_df["is_high_occlusion"].astype(int).values
    ax.fill_between(x, 0, occ_flags, alpha=0.5, color="red", step="mid")
    ax.set_ylabel("High\nOcclusion")
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["No", "Yes"])
    ax.grid(True, alpha=0.3)
    _shade_occlusion(ax)

    # Add legend for occlusion period shading
    if occlusion_periods:
        axes[0].legend(
            handles=[mpatches.Patch(fc="red", alpha=0.15, label="Occlusion Period")],
            fontsize=8, loc="upper right",
        )

    _setup_time_xaxis(axes[-1], frames, fps)

    fig.tight_layout()
    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------


def generate_all_visualizations(
    tracking_df,
    per_frame_df,
    output_dir,
    fps=None,
    chunk_info=None,
    video_path=None,
    yolo_prescan_df=None,
    yolo_occlusion_periods=None,
):
    """
    Generate all tracking visualizations and save to output_dir.

    Args:
        tracking_df: MultiIndex DataFrame from process_tracking_outputs.
        per_frame_df: DataFrame from per_frame_metrics_to_df.
        output_dir: Directory to save PNG files.
        fps: Video FPS for MM:SS axis labels.
        chunk_info: Dict with 'chunks' key for diagnostic plots. Optional.
        video_path: Path to source video for diagnostic plots. Optional.
        yolo_prescan_df: DataFrame from yolo_prescan_to_df. Optional.
        yolo_occlusion_periods: List of (start, end) frame tuples. Optional.
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

    if chunk_info is not None and video_path is not None:
        plot_mask_evolution(
            tracking_df,
            chunk_info,
            video_path,
            fps=fps,
            output_dir=output_dir,
        )
        plot_prompt_points(
            tracking_df,
            chunk_info,
            video_path,
            fps=fps,
            output_dir=output_dir,
        )

    if yolo_prescan_df is not None and not yolo_prescan_df.empty:
        plot_yolo_prescan_overview(
            yolo_prescan_df,
            occlusion_periods=yolo_occlusion_periods,
            fps=fps,
            save_path=output_dir / "yolo_prescan_overview.png",
        )
