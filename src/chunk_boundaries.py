"""
Per-frame metrics, occlusion detection, separation windows, and adaptive chunk boundary computation.

This module is the analysis counterpart to ``src.yolo_scan``, which handles
raw YOLO inference.  Given a YOLO tracking DataFrame (one row per detection per
frame), it computes spatial metrics, identifies occlusion / high-separation
periods, and produces adaptive chunk boundaries suitable for
``script.sam3.run_sam3_hf``.

Typical usage (from a tracking run or reanalysis script)::

    from src.chunk_boundaries import (
        compute_yolo_scan_results,
        chunk_video_frames_adaptive,
        yolo_scan_to_df,
    )

    scan_results = compute_yolo_scan_results(yolo_df, fps=25.0)
    chunks = chunk_video_frames_adaptive(
        total_frames, fps,
        video_model_seconds=15,
        tracker_seconds=60,
        **scan_results,
    )
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from src.metrics import (
    compute_bbox_area_stats,
    compute_bbox_centroids,
    compute_clustering_coefficient,
    compute_pairwise_bbox_iou,
    compute_pairwise_centroid_distances,
)


# ---------------------------------------------------------------------------
# Per-frame metrics
# ---------------------------------------------------------------------------


def compute_separation_score(
    num_objects: int,
    min_centroid_distance: float,
    clustering_coefficient: float,
    num_overlapping_pairs: int,
    min_objects: int = 3,
    min_separation_distance: float = 0.15,
) -> float:
    """
    Per-frame separation quality score in [0, 1].

    Returns 0 for frames that fail the hard gates (too few objects, any
    overlap, or subjects too close). Otherwise:
        min(min_centroid_distance, 1.0) * (1 - clustering_coefficient)

    Args:
        num_objects: Number of detected objects in the frame.
        min_centroid_distance: Minimum pairwise centroid distance (normalised).
        clustering_coefficient: Fraction of pairs within the clustering threshold.
        num_overlapping_pairs: Count of bbox pairs with IoU above threshold.
        min_objects: Minimum objects required to score above zero.
        min_separation_distance: Minimum min_centroid_distance required.

    Returns:
        Float in [0, 1]. Higher = subjects more spread out and unambiguous.
    """
    if (
        num_objects < min_objects
        or num_overlapping_pairs > 0
        or min_centroid_distance == float("inf")
        or min_centroid_distance < min_separation_distance
    ):
        return 0.0
    return min(min_centroid_distance, 1.0) * (1.0 - clustering_coefficient)


def compute_yolo_per_frame_metrics(
    yolo_df: pd.DataFrame,
    occlusion_iou_threshold: float = 0.15,
    clustering_distance_threshold: float = 0.15,
    use_normalized_coords: bool = True,
    separation_min_objects: int = 3,
    separation_min_distance: float = 0.15,
) -> List[Dict[str, Any]]:
    """
    Compute per-frame spatial, overlap, and bbox quality metrics from YOLO tracking.

    Args:
        yolo_df: DataFrame with columns [frame, track_id, x1, y1, x2, y2,
                 cx_norm, cy_norm, confidence] (from YOLO tracking)
        occlusion_iou_threshold: bbox IoU above which a pair is "overlapping"
        clustering_distance_threshold: centroid distance for clustering coefficient
        use_normalized_coords: use cx_norm/cy_norm for distances (recommended)
        separation_min_objects: minimum objects for a non-zero separation_score
        separation_min_distance: minimum min_centroid_distance for separation_score

    Returns:
        List of dicts (sorted by frame), one per frame with metrics:
        - num_objects, objects_present, min/mean_centroid_distance,
          clustering_coefficient, max/mean_pairwise_bbox_iou,
          num_overlapping_pairs, mean/min/max/variance bbox_area,
          is_high_occlusion, is_object_count_change, mean_confidence,
          separation_score
    """
    results: List[Dict[str, Any]] = []
    prev_num_objects: Optional[int] = None

    yolo_df = yolo_df.sort_values("frame").reset_index(drop=True)

    for frame_idx in sorted(yolo_df["frame"].unique()):
        frame_dets = yolo_df[yolo_df["frame"] == frame_idx]
        n = len(frame_dets)

        if n == 0:
            results.append(
                {
                    "frame_idx": int(frame_idx),
                    "num_objects": 0,
                    "objects_present": [],
                    "min_centroid_distance": float("inf"),
                    "mean_centroid_distance": float("inf"),
                    "clustering_coefficient": 0.0,
                    "max_pairwise_bbox_iou": 0.0,
                    "mean_pairwise_bbox_iou": 0.0,
                    "num_overlapping_pairs": 0,
                    "mean_bbox_area": 0.0,
                    "min_bbox_area": 0.0,
                    "max_bbox_area": 0.0,
                    "bbox_area_variance": 0.0,
                    "is_high_occlusion": False,
                    "is_object_count_change": prev_num_objects is not None
                    and prev_num_objects != 0,
                    "mean_confidence": 0.0,
                    "separation_score": 0.0,
                }
            )
            prev_num_objects = 0
            continue

        boxes = frame_dets[["x1", "y1", "x2", "y2"]].values

        if use_normalized_coords and "cx_norm" in frame_dets.columns:
            centroids = frame_dets[["cx_norm", "cy_norm"]].values
        else:
            centroids = compute_bbox_centroids(boxes)

        dist_matrix = compute_pairwise_centroid_distances(centroids)
        if n > 1:
            upper_dists = dist_matrix[np.triu_indices(n, k=1)]
            min_dist = float(np.nanmin(upper_dists)) if len(upper_dists) else float("inf")
            mean_dist = float(np.nanmean(upper_dists)) if len(upper_dists) else float("inf")
        else:
            min_dist = float("inf")
            mean_dist = float("inf")

        clust_coef = compute_clustering_coefficient(centroids, clustering_distance_threshold)

        iou_matrix = compute_pairwise_bbox_iou(boxes)
        if n > 1:
            upper_iou = iou_matrix[np.triu_indices(n, k=1)]
            max_iou = float(np.max(upper_iou)) if len(upper_iou) else 0.0
            mean_iou = float(np.mean(upper_iou)) if len(upper_iou) else 0.0
            num_overlap = int(np.sum(upper_iou > occlusion_iou_threshold))
        else:
            max_iou = 0.0
            mean_iou = 0.0
            num_overlap = 0

        area_stats = compute_bbox_area_stats(boxes)
        mean_conf = float(frame_dets["confidence"].mean())
        is_high_occ = max_iou > occlusion_iou_threshold or clust_coef > 0.5
        is_count_change = prev_num_objects is not None and prev_num_objects != n

        sep_score = compute_separation_score(
            num_objects=n,
            min_centroid_distance=min_dist,
            clustering_coefficient=float(clust_coef),
            num_overlapping_pairs=num_overlap,
            min_objects=separation_min_objects,
            min_separation_distance=separation_min_distance,
        )

        results.append(
            {
                "frame_idx": int(frame_idx),
                "num_objects": n,
                "objects_present": frame_dets["track_id"].tolist(),
                "min_centroid_distance": min_dist,
                "mean_centroid_distance": mean_dist,
                "clustering_coefficient": float(clust_coef),
                "max_pairwise_bbox_iou": max_iou,
                "mean_pairwise_bbox_iou": mean_iou,
                "num_overlapping_pairs": num_overlap,
                "mean_bbox_area": area_stats["mean"],
                "min_bbox_area": area_stats["min"],
                "max_bbox_area": area_stats["max"],
                "bbox_area_variance": area_stats["variance"],
                "is_high_occlusion": is_high_occ,
                "is_object_count_change": is_count_change,
                "mean_confidence": mean_conf,
                "separation_score": sep_score,
            }
        )
        prev_num_objects = n

    return results


# ---------------------------------------------------------------------------
# Window / period detection
# ---------------------------------------------------------------------------


def find_high_separation_windows(
    per_frame_metrics: List[Dict[str, Any]],
    min_objects: int = 3,
    min_separation_distance: float = 0.15,
    min_window_frames: int = 25,
    gap_tolerance_frames: int = 5,
) -> List[Tuple[int, int]]:
    """
    Find sustained periods of high subject separation.

    A frame qualifies when its separation_score > 0.  Contiguous runs of
    qualifying frames — bridged by gaps <= gap_tolerance_frames — that span
    >= min_window_frames total are returned as (start_frame, end_frame) tuples.

    Args:
        per_frame_metrics: Output of compute_yolo_per_frame_metrics.
        min_objects: Minimum detections required for a frame to qualify.
        min_separation_distance: Normalised centroid distance threshold.
        min_window_frames: Minimum span (frames) for a window to be kept.
        gap_tolerance_frames: Gap frames that can be bridged within a run.

    Returns:
        List of (start_frame, end_frame) tuples, sorted by start_frame.
    """
    if not per_frame_metrics:
        return []

    qualifying = [
        m["frame_idx"]
        for m in sorted(per_frame_metrics, key=lambda m: m["frame_idx"])
        if compute_separation_score(
            num_objects=m["num_objects"],
            min_centroid_distance=m["min_centroid_distance"],
            clustering_coefficient=m["clustering_coefficient"],
            num_overlapping_pairs=m["num_overlapping_pairs"],
            min_objects=min_objects,
            min_separation_distance=min_separation_distance,
        )
        > 0.0
    ]

    if not qualifying:
        return []

    windows: List[Tuple[int, int]] = []
    run_start = qualifying[0]
    run_end = qualifying[0]

    for frame in qualifying[1:]:
        if frame - run_end <= gap_tolerance_frames:
            run_end = frame
        else:
            if run_end - run_start + 1 >= min_window_frames:
                windows.append((run_start, run_end))
            run_start = frame
            run_end = frame

    if run_end - run_start + 1 >= min_window_frames:
        windows.append((run_start, run_end))

    return windows


def identify_occlusion_periods(
    per_frame_metrics: List[Dict[str, Any]],
    window_frames: int = 25,
    high_occlusion_threshold: float = 0.3,
) -> List[Tuple[int, int]]:
    """
    Identify contiguous periods of high occlusion using a sliding window.

    Args:
        per_frame_metrics: output of compute_yolo_per_frame_metrics
        window_frames: sliding window size
        high_occlusion_threshold: fraction of frames in window that must be
                                  flagged to mark the period as high-occlusion

    Returns:
        List of (start_frame, end_frame) tuples for occlusion periods
    """
    if not per_frame_metrics:
        return []

    frames = np.array([m["frame_idx"] for m in per_frame_metrics])
    flags = np.array([m["is_high_occlusion"] for m in per_frame_metrics])

    periods: List[Tuple[int, int]] = []
    i = 0

    while i < len(frames):
        window_end_idx = min(i + window_frames, len(frames))
        window_flags = flags[i:window_end_idx]

        if np.mean(window_flags) >= high_occlusion_threshold:
            start_frame = frames[i]

            j = i
            while j < len(frames):
                window_end_idx = min(j + window_frames, len(frames))
                window_flags = flags[j:window_end_idx]
                if np.mean(window_flags) < high_occlusion_threshold:
                    break
                j += 1

            end_frame = frames[min(j + window_frames - 1, len(frames) - 1)]
            periods.append((int(start_frame), int(end_frame)))
            i = j + window_frames
        else:
            i += 1

    return periods


# ---------------------------------------------------------------------------
# High-level scan orchestration
# ---------------------------------------------------------------------------


def compute_yolo_scan_results(
    yolo_df: pd.DataFrame,
    fps: float = 25.0,
    window_seconds: float = 1.0,
    high_occlusion_threshold: float = 0.3,
    occlusion_iou_threshold: float = 0.15,
    clustering_distance_threshold: float = 0.15,
    separation_min_objects: int = 3,
    separation_min_distance: float = 0.15,
    separation_min_window_seconds: float = 1.0,
    separation_gap_tolerance_frames: int = 5,
) -> Dict[str, Any]:
    """
    Run full YOLO-based scan analysis: occlusion periods + separation windows.

    Convenience wrapper that sequences:
        compute_yolo_per_frame_metrics → identify_occlusion_periods
        → find_high_separation_windows

    Args:
        yolo_df: YOLO tracking DataFrame
        fps: video frame rate
        window_seconds: sliding window duration for occlusion detection
        high_occlusion_threshold: fraction of frames flagged in window
        occlusion_iou_threshold: bbox IoU threshold for overlap
        clustering_distance_threshold: normalized centroid distance threshold
        separation_min_objects: minimum detections for a frame to qualify as
            high-separation
        separation_min_distance: minimum normalised centroid distance for
            a frame to qualify as high-separation
        separation_min_window_seconds: minimum duration (seconds) of a
            sustained high-separation run to be returned as a window
        separation_gap_tolerance_frames: frames with no qualifying detections
            that can be bridged without breaking a separation run

    Returns:
        Dict with keys: per_frame_metrics, occlusion_periods, transition_frames,
        separation_windows, total_frames, video_duration_seconds, fps,
        window_frames, high_occlusion_threshold
    """
    window_frames = int(window_seconds * fps)
    separation_min_window_frames = max(1, int(separation_min_window_seconds * fps))

    per_frame_metrics = compute_yolo_per_frame_metrics(
        yolo_df,
        occlusion_iou_threshold=occlusion_iou_threshold,
        clustering_distance_threshold=clustering_distance_threshold,
        use_normalized_coords=True,
        separation_min_objects=separation_min_objects,
        separation_min_distance=separation_min_distance,
    )

    occlusion_periods = identify_occlusion_periods(
        per_frame_metrics,
        window_frames=window_frames,
        high_occlusion_threshold=high_occlusion_threshold,
    )

    transition_frames_list = []
    for start, end in occlusion_periods:
        transition_frames_list.extend([start, end])
    transition_frames = np.array(sorted(set(transition_frames_list)))

    separation_windows = find_high_separation_windows(
        per_frame_metrics,
        min_objects=separation_min_objects,
        min_separation_distance=separation_min_distance,
        min_window_frames=separation_min_window_frames,
        gap_tolerance_frames=separation_gap_tolerance_frames,
    )

    total_frames = int(yolo_df["frame"].max())
    video_duration = total_frames / fps

    logger.info(
        f"Scan: {len(occlusion_periods)} occlusion periods, "
        f"{len(separation_windows)} separation windows "
        f"(min_distance={separation_min_distance}, min_objects={separation_min_objects})"
    )

    return {
        "per_frame_metrics": per_frame_metrics,
        "occlusion_periods": occlusion_periods,
        "transition_frames": transition_frames,
        "separation_windows": separation_windows,
        "total_frames": total_frames,
        "video_duration_seconds": float(video_duration),
        "fps": float(fps),
        "window_frames": int(window_frames),
        "high_occlusion_threshold": float(high_occlusion_threshold),
    }


# ---------------------------------------------------------------------------
# DataFrame export
# ---------------------------------------------------------------------------


def yolo_scan_to_df(per_frame_metrics: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert per-frame metrics list to a DataFrame for analysis/export."""
    if not per_frame_metrics:
        return pd.DataFrame()

    rows = []
    for d in per_frame_metrics:
        r = d.copy()
        r["objects_present"] = ",".join(str(x) for x in r.get("objects_present", []))
        rows.append(r)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Adaptive chunking
# ---------------------------------------------------------------------------


def chunk_video_frames_adaptive(
    total_frames: int,
    fps: float,
    video_model_seconds: int,
    tracker_seconds: int,
    per_frame_metrics: list[dict] | None = None,
    search_window_seconds: float = 10.0,
    max_chunk_seconds: float = 150.0,
    # Legacy / unused parameters kept for call-site compatibility
    separation_windows: list[tuple[int, int]] | None = None,
    occlusion_periods: list[tuple[int, int]] | None = None,
    transition_frames: np.ndarray | None = None,
    min_chunk_seconds: float = 0.0,
    margin_seconds: float = 3.0,
) -> list[tuple[int, int, str]]:
    """
    Split video into chunks, with tracker-chunk boundaries placed at the frame
    of highest subject separation within a search window around each nominal cut.

    Generates initial fixed chunks (chunk 0 → video model, remainder → tracker),
    then for each tracker-chunk boundary searches ±``search_window_seconds`` for
    the frame with the highest ``separation_score``.  A frame's separation score
    is > 0 only when all hard gates pass (≥ min_objects detected, no overlapping
    bbox pairs above the IoU threshold, min centroid distance ≥ threshold); the
    score itself is ``min(min_centroid_distance, 1) × (1 − clustering_coeff)``.

    If no frame in the window has a positive separation score the nominal
    boundary is kept unchanged and a debug message is logged.

    Small trailing remainders (< 10 % of tracker chunk size) are absorbed into
    the preceding chunk.

    Args:
        total_frames: Total number of frames in the video.
        fps: Video frame rate.
        video_model_seconds: Duration of the first (text-prompted) chunk in seconds.
        tracker_seconds: Duration of subsequent (point-prompted) chunks in seconds.
        per_frame_metrics: Per-frame metric dicts from ``compute_yolo_per_frame_metrics``.
            Must contain ``frame_idx`` and ``separation_score``.  When ``None``
            the nominal fixed-chunk boundaries are returned unchanged.
        search_window_seconds: Search radius (±seconds) around each nominal boundary.
        max_chunk_seconds: Hard upper limit on chunk duration (VRAM constraint).
        separation_windows, occlusion_periods, transition_frames,
        min_chunk_seconds, margin_seconds: Unused legacy parameters kept for
            call-site compatibility.

    Returns:
        List of (start_frame, end_frame, model_type) tuples where model_type is
        ``"video"`` (first chunk) or ``"tracker"`` (all others).
    """
    video_model_frames = int(fps * video_model_seconds)
    tracker_frames = int(fps * tracker_seconds)
    search_window_frames = int(search_window_seconds * fps)
    max_frames = int(max_chunk_seconds * fps)

    # Step 1: Generate initial fixed chunks
    fixed_chunks: list[tuple[int, int, str]] = []
    end = min(video_model_frames, total_frames)
    fixed_chunks.append((0, end, "video"))

    start = end
    while start < total_frames:
        end = min(start + tracker_frames, total_frames)
        remaining_after = total_frames - end
        if 0 < remaining_after < tracker_frames * 0.1:
            end = total_frames
        fixed_chunks.append((start, end, "tracker"))
        start = end

    if len(fixed_chunks) <= 1 or per_frame_metrics is None:
        return fixed_chunks

    # Step 2: Build per-frame lookups
    # Primary: separation score (> 0 only when all hard gates pass)
    sep_lookup: dict[int, float] = {
        m["frame_idx"]: m.get("separation_score", 0.0)
        for m in per_frame_metrics
    }
    # Fallback: for frames that fail the sep gates (e.g. only 2 of 3 birds detected),
    # rank by (zero_iou, min_centroid_distance) so we still avoid high-overlap frames.
    # Requires at least 2 objects for pairwise metrics to exist.
    fallback_lookup: dict[int, tuple[float, float]] = {
        m["frame_idx"]: (
            -m.get("max_pairwise_bbox_iou", 1.0),
            m.get("min_centroid_distance", 0.0)
            if m.get("min_centroid_distance", float("inf")) != float("inf")
            else 0.0,
        )
        for m in per_frame_metrics
        if m.get("num_objects", 0) >= 2
    }

    boundaries = [c[0] for c in fixed_chunks]
    total_end = fixed_chunks[-1][1]
    adjusted_boundaries = list(boundaries)

    def _validate(candidate: int, idx: int) -> bool:
        prev_start = adjusted_boundaries[idx - 1]
        next_end = (
            adjusted_boundaries[idx + 1]
            if idx + 1 < len(adjusted_boundaries)
            else total_end
        )
        return (candidate - prev_start) <= max_frames and (next_end - candidate) <= max_frames

    # Step 3: Refine each tracker-chunk boundary (index 1+)
    for i in range(1, len(adjusted_boundaries)):
        original = adjusted_boundaries[i]
        search_start = max(0, original - search_window_frames)
        search_end = min(total_end, original + search_window_frames)

        # Find the frame with the highest separation score in the window.
        best_frame: int | None = None
        best_score: float = 0.0
        for frame, score in sep_lookup.items():
            if search_start <= frame <= search_end and score > best_score:
                if _validate(frame, i):
                    best_score = score
                    best_frame = frame

        if best_frame is not None and best_frame != original:
            shift = best_frame - original
            adjusted_boundaries[i] = best_frame
            logger.info(
                f"Boundary {i}: frame {original} → {best_frame} "
                f"(shifted {shift:+d} frames, separation_score={best_score:.3f})"
            )
        else:
            # No frame with positive separation score found — fall back to the
            # least-bad frame: among n_obj>=2 candidates, minimise IoU then
            # maximise centroid distance.
            fb_frame: int | None = None
            fb_score: tuple[float, float] = (-float("inf"), -float("inf"))
            for frame, score in fallback_lookup.items():
                if search_start <= frame <= search_end and score > fb_score:
                    if _validate(frame, i):
                        fb_score = score
                        fb_frame = frame

            if fb_frame is not None and fb_frame != original:
                shift = fb_frame - original
                adjusted_boundaries[i] = fb_frame
                logger.warning(
                    f"Boundary {i}: frame {original} → {fb_frame} "
                    f"(shifted {shift:+d} frames, fallback: "
                    f"iou={-fb_score[0]:.3f}, dist={fb_score[1]:.3f})"
                )
            else:
                logger.debug(
                    f"Boundary {i}: frame {original} kept — no better frame found "
                    f"(best sep_score in window = {best_score:.3f})"
                )

    # Step 4: Rebuild chunks from adjusted boundaries
    adjusted_chunks: list[tuple[int, int, str]] = []
    for i in range(len(adjusted_boundaries)):
        s = adjusted_boundaries[i]
        e = (
            adjusted_boundaries[i + 1]
            if i + 1 < len(adjusted_boundaries)
            else total_end
        )
        adjusted_chunks.append((s, e, fixed_chunks[i][2]))

    # Step 5: Absorb small trailing chunks (< 10% of tracker chunk size)
    if len(adjusted_chunks) > 1:
        last_start, last_end, _ = adjusted_chunks[-1]
        if last_end - last_start < tracker_frames * 0.1:
            prev_start, _, prev_type = adjusted_chunks[-2]
            logger.info(
                f"Absorbing small trailing chunk "
                f"({last_end - last_start} frames, {(last_end - last_start) / fps:.1f}s) "
                f"into previous chunk"
            )
            adjusted_chunks[-2] = (prev_start, last_end, prev_type)
            adjusted_chunks.pop()

    return adjusted_chunks
