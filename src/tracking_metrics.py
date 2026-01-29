"""
Tracking Metrics Module

Provides functions to compute and log metrics for diagnosing identity loss
in multi-object tracking scenarios.

This module helps identify:
- When objects clump together (high occlusion)
- When identities may have switched
- Object lifecycle across chunks
- Tracking stability metrics

Usage:
    from src.tracking_metrics import (
        compute_frame_metrics,
        compute_chunk_metrics,
        FrameMetrics,
        ChunkMetrics,
        save_frame_metrics_to_parquet,
        save_chunk_summary_to_parquet,
    )
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from loguru import logger


def _ensure_numpy(arr: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """Convert tensor to numpy array if needed, handling CUDA tensors."""
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    return arr


@dataclass
class FrameMetrics:
    """Metrics computed for a single frame."""

    frame_idx: int
    num_objects: int

    # Spatial clustering metrics
    min_centroid_distance: float  # Minimum distance between any two object centroids
    mean_centroid_distance: float  # Mean pairwise centroid distance
    clustering_coefficient: float  # 0-1, higher = more clustered

    # Overlap metrics
    max_pairwise_iou: float  # Maximum IoU between any two objects
    mean_pairwise_iou: float  # Mean pairwise IoU
    num_overlapping_pairs: int  # Count of pairs with IoU > threshold

    # Mask quality metrics
    mean_mask_area: float  # Average mask area in pixels
    min_mask_area: float  # Smallest mask
    max_mask_area: float  # Largest mask
    mask_area_variance: float  # Variance in mask sizes

    # Object identity stability
    objects_present: List[int] = field(default_factory=list)  # Object IDs in this frame

    # Flags
    is_high_occlusion_frame: bool = False  # True if significant overlap detected
    is_object_count_change: bool = False  # True if object count changed from previous


@dataclass
class ChunkMetrics:
    """Aggregated metrics for a video chunk."""

    chunk_idx: int
    start_frame: int
    end_frame: int

    # Object lifecycle
    objects_at_start: List[int] = field(default_factory=list)
    objects_at_end: List[int] = field(default_factory=list)
    objects_lost: List[int] = field(
        default_factory=list
    )  # Present at start, gone at end
    objects_gained: List[int] = field(default_factory=list)  # New objects appeared

    # Identity stability
    identity_switches: int = 0  # Detected ID switches (heuristic)
    max_continuous_tracking: dict[int, int] = field(
        default_factory=dict
    )  # obj_id -> max consecutive frames

    # Occlusion events
    high_occlusion_frames: List[int] = field(default_factory=list)
    total_occlusion_events: int = 0

    # Statistics
    mean_objects_per_frame: float = 0.0
    object_count_changes: int = 0  # Number of frames where count changed


def compute_pairwise_iou(masks: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """
    Compute pairwise IoU matrix for all masks.

    Args:
        masks: Array of shape (N, H, W) with N binary masks (numpy or torch tensor)

    Returns:
        NxN matrix of IoU values
    """
    masks = _ensure_numpy(masks)
    n = len(masks)
    if n == 0:
        return np.array([])

    iou_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            intersection = np.logical_and(masks[i], masks[j]).sum()
            union = np.logical_or(masks[i], masks[j]).sum()
            iou = intersection / union if union > 0 else 0.0
            iou_matrix[i, j] = iou
            iou_matrix[j, i] = iou

    return iou_matrix


def compute_centroids(masks: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """
    Compute centroids for all masks.

    Args:
        masks: Array of shape (N, H, W) with N binary masks (numpy or torch tensor)

    Returns:
        Array of shape (N, 2) with [x, y] centroids
    """
    masks = _ensure_numpy(masks)
    centroids = []
    for mask in masks:
        y_coords, x_coords = np.where(mask > 0)
        if len(y_coords) > 0:
            centroids.append([np.mean(x_coords), np.mean(y_coords)])
        else:
            centroids.append([np.nan, np.nan])
    return np.array(centroids)


def compute_pairwise_centroid_distances(centroids: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Euclidean distances between centroids.

    Args:
        centroids: Array of shape (N, 2) with [x, y] centroids

    Returns:
        NxN distance matrix
    """
    n = len(centroids)
    if n == 0:
        return np.array([])

    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(centroids[i] - centroids[j])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist

    return dist_matrix


def compute_frame_metrics(
    frame_idx: int,
    masks: Union[np.ndarray, torch.Tensor],
    object_ids: Union[np.ndarray, torch.Tensor],
    prev_num_objects: Optional[int] = None,
    occlusion_iou_threshold: float = 0.15,
    clustering_distance_threshold: float = 50.0,  # pixels
) -> FrameMetrics:
    """
    Compute comprehensive metrics for a single frame.

    Args:
        frame_idx: Frame index
        masks: Array of shape (N, H, W) with N binary masks (numpy or torch tensor)
        object_ids: Array of object IDs corresponding to masks (numpy or torch tensor)
        prev_num_objects: Number of objects in previous frame (for change detection)
        occlusion_iou_threshold: IoU threshold to consider objects as overlapping
        clustering_distance_threshold: Distance threshold for clustering detection

    Returns:
        FrameMetrics dataclass with computed values
    """
    # Ensure numpy arrays (handles CUDA tensors)
    masks = _ensure_numpy(masks)
    object_ids = _ensure_numpy(object_ids)

    n = len(masks)

    if n == 0:
        return FrameMetrics(
            frame_idx=frame_idx,
            num_objects=0,
            min_centroid_distance=float("inf"),
            mean_centroid_distance=float("inf"),
            clustering_coefficient=0.0,
            max_pairwise_iou=0.0,
            mean_pairwise_iou=0.0,
            num_overlapping_pairs=0,
            mean_mask_area=0.0,
            min_mask_area=0.0,
            max_mask_area=0.0,
            mask_area_variance=0.0,
            objects_present=[],
            is_high_occlusion_frame=False,
            is_object_count_change=prev_num_objects is not None
            and prev_num_objects != 0,
        )

    # Compute centroids and distances
    centroids = compute_centroids(masks)
    dist_matrix = compute_pairwise_centroid_distances(centroids)

    # Extract upper triangle (excluding diagonal) for pairwise stats
    if n > 1:
        upper_tri_indices = np.triu_indices(n, k=1)
        pairwise_distances = dist_matrix[upper_tri_indices]
        min_dist = (
            float(np.nanmin(pairwise_distances))
            if len(pairwise_distances) > 0
            else float("inf")
        )
        mean_dist = (
            float(np.nanmean(pairwise_distances))
            if len(pairwise_distances) > 0
            else float("inf")
        )
    else:
        min_dist = float("inf")
        mean_dist = float("inf")
        pairwise_distances = np.array([])

    # Clustering coefficient: proportion of pairs within threshold distance
    if len(pairwise_distances) > 0:
        close_pairs = np.sum(pairwise_distances < clustering_distance_threshold)
        clustering_coef = close_pairs / len(pairwise_distances)
    else:
        clustering_coef = 0.0

    # Compute IoU matrix
    iou_matrix = compute_pairwise_iou(masks)

    if n > 1:
        upper_tri_iou = iou_matrix[upper_tri_indices]
        max_iou = float(np.max(upper_tri_iou)) if len(upper_tri_iou) > 0 else 0.0
        mean_iou = float(np.mean(upper_tri_iou)) if len(upper_tri_iou) > 0 else 0.0
        num_overlapping = int(np.sum(upper_tri_iou > occlusion_iou_threshold))
    else:
        max_iou = 0.0
        mean_iou = 0.0
        num_overlapping = 0

    # Mask area statistics
    mask_areas = np.array([mask.sum() for mask in masks])

    # Determine flags
    is_high_occlusion = max_iou > occlusion_iou_threshold or clustering_coef > 0.5
    is_count_change = prev_num_objects is not None and prev_num_objects != n

    return FrameMetrics(
        frame_idx=frame_idx,
        num_objects=n,
        min_centroid_distance=min_dist,
        mean_centroid_distance=mean_dist,
        clustering_coefficient=float(clustering_coef),
        max_pairwise_iou=max_iou,
        mean_pairwise_iou=mean_iou,
        num_overlapping_pairs=num_overlapping,
        mean_mask_area=float(np.mean(mask_areas)),
        min_mask_area=float(np.min(mask_areas)),
        max_mask_area=float(np.max(mask_areas)),
        mask_area_variance=float(np.var(mask_areas)),
        objects_present=list(object_ids.astype(int)),
        is_high_occlusion_frame=is_high_occlusion,
        is_object_count_change=is_count_change,
    )


def detect_identity_switches(
    frame_metrics_list: List[FrameMetrics],
    window_size: int = 5,
) -> List[Tuple[int, int, int]]:
    """
    Heuristically detect potential identity switches.

    An identity switch is suspected when:
    1. Object count remains stable but object IDs change
    2. A high-occlusion event is followed by ID changes

    Args:
        frame_metrics_list: List of FrameMetrics in frame order
        window_size: Number of frames to look back for occlusion events

    Returns:
        List of (frame_idx, old_id, new_id) suspected switches
    """
    switches = []

    for i, metrics in enumerate(frame_metrics_list):
        if i == 0:
            continue

        prev_metrics = frame_metrics_list[i - 1]

        # Check if object count is stable but IDs changed
        if metrics.num_objects == prev_metrics.num_objects > 0:
            prev_ids = set(prev_metrics.objects_present)
            curr_ids = set(metrics.objects_present)

            disappeared = prev_ids - curr_ids
            appeared = curr_ids - prev_ids

            # If IDs changed but count didn't, suspect a switch
            if disappeared and appeared and len(disappeared) == len(appeared):
                # Check if there was a recent occlusion event
                recent_occlusion = any(
                    frame_metrics_list[j].is_high_occlusion_frame
                    for j in range(max(0, i - window_size), i)
                )

                if recent_occlusion:
                    for old_id, new_id in zip(sorted(disappeared), sorted(appeared)):
                        switches.append((metrics.frame_idx, old_id, new_id))
                        logger.warning(
                            f"Suspected identity switch at frame {metrics.frame_idx}: "
                            f"ID {old_id} -> {new_id} (after occlusion)"
                        )

    return switches


def compute_chunk_metrics(
    chunk_idx: int,
    start_frame: int,
    end_frame: int,
    frame_metrics_list: List[FrameMetrics],
) -> ChunkMetrics:
    """
    Aggregate frame metrics into chunk-level metrics.

    Args:
        chunk_idx: Chunk index
        start_frame: Global start frame index
        end_frame: Global end frame index
        frame_metrics_list: List of FrameMetrics for this chunk

    Returns:
        ChunkMetrics dataclass with aggregated values
    """
    if not frame_metrics_list:
        return ChunkMetrics(
            chunk_idx=chunk_idx, start_frame=start_frame, end_frame=end_frame
        )

    # Object lifecycle
    objects_at_start = frame_metrics_list[0].objects_present
    objects_at_end = frame_metrics_list[-1].objects_present

    all_objects_seen = set()
    for fm in frame_metrics_list:
        all_objects_seen.update(fm.objects_present)

    objects_lost = [oid for oid in objects_at_start if oid not in objects_at_end]
    objects_gained = [oid for oid in objects_at_end if oid not in objects_at_start]

    # Identity switches
    switches = detect_identity_switches(frame_metrics_list)

    # Continuous tracking per object
    max_continuous = {}
    for obj_id in all_objects_seen:
        current_streak = 0
        max_streak = 0
        for fm in frame_metrics_list:
            if obj_id in fm.objects_present:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        max_continuous[obj_id] = max_streak

    # Occlusion events
    high_occlusion_frames = [
        fm.frame_idx for fm in frame_metrics_list if fm.is_high_occlusion_frame
    ]

    # Count separate occlusion events (consecutive high-occlusion frames = 1 event)
    occlusion_events = 0
    in_occlusion = False
    for fm in frame_metrics_list:
        if fm.is_high_occlusion_frame and not in_occlusion:
            occlusion_events += 1
            in_occlusion = True
        elif not fm.is_high_occlusion_frame:
            in_occlusion = False

    # Statistics
    mean_objects = np.mean([fm.num_objects for fm in frame_metrics_list])
    count_changes = sum(1 for fm in frame_metrics_list if fm.is_object_count_change)

    return ChunkMetrics(
        chunk_idx=chunk_idx,
        start_frame=start_frame,
        end_frame=end_frame,
        objects_at_start=objects_at_start,
        objects_at_end=objects_at_end,
        objects_lost=objects_lost,
        objects_gained=objects_gained,
        identity_switches=len(switches),
        max_continuous_tracking=max_continuous,
        high_occlusion_frames=high_occlusion_frames,
        total_occlusion_events=occlusion_events,
        mean_objects_per_frame=float(mean_objects),
        object_count_changes=count_changes,
    )


def log_frame_metrics(metrics: FrameMetrics, log_level: str = "DEBUG") -> None:
    """Log frame metrics at specified level."""
    log_fn = getattr(logger, log_level.lower(), logger.debug)

    if metrics.is_high_occlusion_frame:
        logger.warning(
            f"⚠️  Frame {metrics.frame_idx}: HIGH OCCLUSION DETECTED - "
            f"max_iou={metrics.max_pairwise_iou:.3f}, "
            f"min_dist={metrics.min_centroid_distance:.1f}px, "
            f"clustering={metrics.clustering_coefficient:.2f}"
        )

    if metrics.is_object_count_change:
        logger.info(
            f"📊 Frame {metrics.frame_idx}: Object count changed to {metrics.num_objects}"
        )

    log_fn(
        f"Frame {metrics.frame_idx}: "
        f"n={metrics.num_objects}, "
        f"min_dist={metrics.min_centroid_distance:.1f}, "
        f"max_iou={metrics.max_pairwise_iou:.3f}, "
        f"cluster_coef={metrics.clustering_coefficient:.2f}"
    )


def log_chunk_metrics(metrics: ChunkMetrics) -> None:
    """Log chunk-level metrics summary."""
    logger.info(
        f"\n{'=' * 60}\n"
        f"CHUNK {metrics.chunk_idx} METRICS SUMMARY (frames {metrics.start_frame}-{metrics.end_frame})\n"
        f"{'=' * 60}"
    )

    logger.info(f"Objects at start: {metrics.objects_at_start}")
    logger.info(f"Objects at end:   {metrics.objects_at_end}")

    if metrics.objects_lost:
        logger.warning(f"⚠️  Objects LOST: {metrics.objects_lost}")
    if metrics.objects_gained:
        logger.info(f"📈 Objects GAINED: {metrics.objects_gained}")

    if metrics.identity_switches > 0:
        logger.warning(f"🔄 Suspected identity switches: {metrics.identity_switches}")

    logger.info(f"Mean objects per frame: {metrics.mean_objects_per_frame:.2f}")
    logger.info(f"Object count changes: {metrics.object_count_changes}")
    logger.info(f"Total occlusion events: {metrics.total_occlusion_events}")
    logger.info(f"High occlusion frames: {len(metrics.high_occlusion_frames)}")

    if metrics.max_continuous_tracking:
        logger.info("Continuous tracking (frames):")
        for obj_id, frames in sorted(metrics.max_continuous_tracking.items()):
            total_frames = metrics.end_frame - metrics.start_frame
            pct = 100 * frames / total_frames if total_frames > 0 else 0
            logger.info(f"  Object {obj_id}: {frames} frames ({pct:.1f}%)")


def metrics_to_dict(metrics: FrameMetrics | ChunkMetrics) -> dict:
    """Convert metrics dataclass to JSON-serializable dict."""
    return asdict(metrics)


# =============================================================================
# Pandas-based save functions
# =============================================================================


def save_frame_metrics_to_parquet(
    frame_metrics_list: List[FrameMetrics],
    output_path: Path,
) -> None:
    """
    Save frame metrics to Parquet file using pandas.

    Args:
        frame_metrics_list: List of FrameMetrics objects
        output_path: Path to save the parquet file
    """
    if not frame_metrics_list:
        logger.warning("No frame metrics to save")
        return

    # Convert to list of dicts
    records = []
    for fm in frame_metrics_list:
        record = asdict(fm)
        # Convert list of objects to comma-separated string for easier analysis
        record["objects_present"] = ",".join(map(str, record["objects_present"]))
        records.append(record)

    df = pd.DataFrame(records)

    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to parquet (efficient columnar format)
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved frame metrics ({len(df)} rows) to {output_path}")


def save_frame_metrics_to_csv(
    frame_metrics_list: List[FrameMetrics],
    output_path: Path,
) -> None:
    """
    Save frame metrics to CSV file using pandas.

    Args:
        frame_metrics_list: List of FrameMetrics objects
        output_path: Path to save the CSV file
    """
    if not frame_metrics_list:
        logger.warning("No frame metrics to save")
        return

    # Convert to list of dicts
    records = []
    for fm in frame_metrics_list:
        record = asdict(fm)
        # Convert list of objects to comma-separated string for easier analysis
        record["objects_present"] = ",".join(map(str, record["objects_present"]))
        records.append(record)

    df = pd.DataFrame(records)

    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    df.to_csv(output_path, index=False)
    logger.info(f"Saved frame metrics ({len(df)} rows) to {output_path}")


def save_chunk_summary_to_parquet(
    chunk_metrics_list: List[ChunkMetrics],
    output_path: Path,
) -> None:
    """
    Save chunk-level summary to Parquet file using pandas.

    Args:
        chunk_metrics_list: List of ChunkMetrics objects
        output_path: Path to save the parquet file
    """
    if not chunk_metrics_list:
        logger.warning("No chunk metrics to save")
        return

    records = []
    for cm in chunk_metrics_list:
        record = {
            "chunk_idx": cm.chunk_idx,
            "start_frame": cm.start_frame,
            "end_frame": cm.end_frame,
            "objects_at_start": ",".join(map(str, cm.objects_at_start)),
            "objects_at_end": ",".join(map(str, cm.objects_at_end)),
            "objects_lost": ",".join(map(str, cm.objects_lost)),
            "objects_gained": ",".join(map(str, cm.objects_gained)),
            "identity_switches": cm.identity_switches,
            "total_occlusion_events": cm.total_occlusion_events,
            "high_occlusion_frame_count": len(cm.high_occlusion_frames),
            "mean_objects_per_frame": cm.mean_objects_per_frame,
            "object_count_changes": cm.object_count_changes,
            # Store max_continuous_tracking as JSON string for flexibility
            "max_continuous_tracking": str(cm.max_continuous_tracking),
        }
        records.append(record)

    df = pd.DataFrame(records)

    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to parquet
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved chunk summary ({len(df)} rows) to {output_path}")


def save_chunk_summary_to_csv(
    chunk_metrics_list: List[ChunkMetrics],
    output_path: Path,
) -> None:
    """
    Save chunk-level summary to CSV file using pandas.

    Args:
        chunk_metrics_list: List of ChunkMetrics objects
        output_path: Path to save the CSV file
    """
    if not chunk_metrics_list:
        logger.warning("No chunk metrics to save")
        return

    records = []
    for cm in chunk_metrics_list:
        record = {
            "chunk_idx": cm.chunk_idx,
            "start_frame": cm.start_frame,
            "end_frame": cm.end_frame,
            "objects_at_start": ",".join(map(str, cm.objects_at_start)),
            "objects_at_end": ",".join(map(str, cm.objects_at_end)),
            "objects_lost": ",".join(map(str, cm.objects_lost)),
            "objects_gained": ",".join(map(str, cm.objects_gained)),
            "identity_switches": cm.identity_switches,
            "total_occlusion_events": cm.total_occlusion_events,
            "high_occlusion_frame_count": len(cm.high_occlusion_frames),
            "mean_objects_per_frame": cm.mean_objects_per_frame,
            "object_count_changes": cm.object_count_changes,
            # Store max_continuous_tracking as JSON string for flexibility
            "max_continuous_tracking": str(cm.max_continuous_tracking),
        }
        records.append(record)

    df = pd.DataFrame(records)

    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    df.to_csv(output_path, index=False)
    logger.info(f"Saved chunk summary ({len(df)} rows) to {output_path}")
