"""
YOLO-based occlusion period detection for adaptive chunking pre-scan.

Computes per-frame spatial and overlap metrics from YOLO tracking outputs,
providing a semantic alternative to pixel-clustering (KMeans) for identifying
high-occlusion regions.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def compute_bbox_iou(boxA: np.ndarray, boxB: np.ndarray) -> float:
    """
    Compute IoU between two bounding boxes in [x1, y1, x2, y2] format.
    
    Args:
        boxA: (4,) array [x1, y1, x2, y2]
        boxB: (4,) array [x1, y1, x2, y2]
    
    Returns:
        IoU in [0, 1]
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    inter = interW * interH
    
    areaA = max(0, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    areaB = max(0, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    union = areaA + areaB - inter
    
    return inter / union if union > 0 else 0.0


def compute_pairwise_bbox_iou(boxes: np.ndarray) -> np.ndarray:
    """
    Compute pairwise IoU matrix for all bounding boxes.
    
    Args:
        boxes: (N, 4) array of [x1, y1, x2, y2] boxes
    
    Returns:
        (N, N) symmetric IoU matrix with zeros on diagonal
    """
    n = len(boxes)
    if n == 0:
        return np.array([])
    
    iou_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            iou = compute_bbox_iou(boxes[i], boxes[j])
            iou_matrix[i, j] = iou
            iou_matrix[j, i] = iou
    return iou_matrix


def compute_bbox_centroids(boxes: np.ndarray) -> np.ndarray:
    """
    Compute centroids from bounding boxes.
    
    Args:
        boxes: (N, 4) array of [x1, y1, x2, y2] boxes
    
    Returns:
        (N, 2) array of [cx, cy] centroids
    """
    if len(boxes) == 0:
        return np.array([])
    
    cx = (boxes[:, 0] + boxes[:, 2]) / 2
    cy = (boxes[:, 1] + boxes[:, 3]) / 2
    return np.column_stack([cx, cy])


def compute_pairwise_centroid_distances(centroids: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Euclidean distances between centroids.
    
    Args:
        centroids: (N, 2) array of [cx, cy] centroids
    
    Returns:
        (N, N) symmetric distance matrix
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


def compute_clustering_coefficient(
    centroids: np.ndarray, 
    threshold: float
) -> float:
    """
    Fraction of centroid pairs within threshold distance.
    
    Args:
        centroids: (N, 2) array
        threshold: distance in pixels (or normalized coordinates)
    
    Returns:
        Float in [0, 1]. 0.0 when fewer than 2 objects.
    """
    n = len(centroids)
    if n < 2:
        return 0.0
    
    dists = compute_pairwise_centroid_distances(centroids)
    upper = dists[np.triu_indices(n, k=1)]
    if len(upper) == 0:
        return 0.0
    
    return float(np.sum(upper < threshold) / len(upper))


def compute_bbox_area_stats(boxes: np.ndarray) -> Dict[str, Any]:
    """
    Compute bounding box area statistics.
    
    Args:
        boxes: (N, 4) array of [x1, y1, x2, y2] boxes
    
    Returns:
        Dict with keys: areas (N,), mean, min, max, variance
    """
    if len(boxes) == 0:
        return {
            "areas": np.array([]),
            "mean": 0.0,
            "min": 0.0,
            "max": 0.0,
            "variance": 0.0,
        }
    
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    areas = np.maximum(areas, 0.0)
    
    return {
        "areas": areas,
        "mean": float(np.mean(areas)),
        "min": float(np.min(areas)),
        "max": float(np.max(areas)),
        "variance": float(np.var(areas)),
    }


def compute_yolo_per_frame_metrics(
    yolo_df: pd.DataFrame,
    occlusion_iou_threshold: float = 0.15,
    clustering_distance_threshold: float = 0.15,  # normalized coordinates
    use_normalized_coords: bool = True,
) -> List[Dict[str, Any]]:
    """
    Compute per-frame spatial, overlap, and bbox quality metrics from YOLO tracking.
    
    Similar to metrics.compute_per_frame_metrics but operates on YOLO bbox data.
    
    Args:
        yolo_df: DataFrame with columns [frame, track_id, x1, y1, x2, y2, 
                 cx_norm, cy_norm, confidence] (from YOLO tracking)
        occlusion_iou_threshold: bbox IoU above which a pair is "overlapping"
        clustering_distance_threshold: centroid distance for clustering coefficient
        use_normalized_coords: use cx_norm/cy_norm for distances (recommended)
    
    Returns:
        List of dicts (sorted by frame), one per frame with metrics:
        - num_objects: detection count
        - objects_present: list of track IDs
        - min_centroid_distance: minimum pairwise centroid distance
        - mean_centroid_distance: mean pairwise centroid distance
        - clustering_coefficient: fraction of pairs within threshold
        - max_pairwise_bbox_iou: maximum bbox overlap
        - mean_pairwise_bbox_iou: mean bbox overlap
        - num_overlapping_pairs: count of pairs above IoU threshold
        - mean_bbox_area: mean detection area
        - min_bbox_area: smallest detection
        - max_bbox_area: largest detection
        - bbox_area_variance: area variance
        - is_high_occlusion: flag (high IoU or high clustering)
        - is_object_count_change: object count changed from prev frame
        - mean_confidence: mean YOLO detection confidence
    """
    results: List[Dict[str, Any]] = []
    prev_num_objects: Optional[int] = None
    
    # Sort by frame to ensure sequential processing
    yolo_df = yolo_df.sort_values("frame").reset_index(drop=True)
    
    for frame_idx in sorted(yolo_df["frame"].unique()):
        frame_dets = yolo_df[yolo_df["frame"] == frame_idx]
        n = len(frame_dets)
        
        if n == 0:
            results.append({
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
                "is_object_count_change": prev_num_objects is not None and prev_num_objects != 0,
                "mean_confidence": 0.0,
            })
            prev_num_objects = 0
            continue
        
        # Extract bounding boxes and centroids
        boxes = frame_dets[["x1", "y1", "x2", "y2"]].values
        
        if use_normalized_coords and "cx_norm" in frame_dets.columns:
            centroids = frame_dets[["cx_norm", "cy_norm"]].values
        else:
            centroids = compute_bbox_centroids(boxes)
        
        # Centroid distances
        dist_matrix = compute_pairwise_centroid_distances(centroids)
        if n > 1:
            upper_dists = dist_matrix[np.triu_indices(n, k=1)]
            min_dist = float(np.nanmin(upper_dists)) if len(upper_dists) else float("inf")
            mean_dist = float(np.nanmean(upper_dists)) if len(upper_dists) else float("inf")
        else:
            min_dist = float("inf")
            mean_dist = float("inf")
        
        clust_coef = compute_clustering_coefficient(centroids, clustering_distance_threshold)
        
        # Bbox IoU
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
        
        # Bbox areas
        area_stats = compute_bbox_area_stats(boxes)
        
        # Confidence
        mean_conf = float(frame_dets["confidence"].mean())
        
        # Flags
        is_high_occ = max_iou > occlusion_iou_threshold or clust_coef > 0.5
        is_count_change = prev_num_objects is not None and prev_num_objects != n
        
        results.append({
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
        })
        prev_num_objects = n
    
    return results


def identify_occlusion_periods(
    per_frame_metrics: List[Dict[str, Any]],
    window_frames: int = 125,  # 1 second at 25fps, 5 seconds at 25fps
    high_occlusion_threshold: float = 0.3,  # fraction of frames flagged
) -> List[Tuple[int, int]]:
    """
    Identify contiguous periods of high occlusion using a sliding window.
    
    Args:
        per_frame_metrics: output of compute_yolo_per_frame_metrics
        window_frames: sliding window size (e.g., 125 frames = 1s @ 25fps)
        high_occlusion_threshold: fraction of frames in window that must be
                                  flagged to mark the period as high-occlusion
    
    Returns:
        List of (start_frame, end_frame) tuples for occlusion periods
    """
    if not per_frame_metrics:
        return []
    
    # Extract frame indices and occlusion flags
    frames = np.array([m["frame_idx"] for m in per_frame_metrics])
    flags = np.array([m["is_high_occlusion"] for m in per_frame_metrics])
    
    periods: List[Tuple[int, int]] = []
    i = 0
    
    while i < len(frames):
        # Define window end
        window_end_idx = min(i + window_frames, len(frames))
        window_flags = flags[i:window_end_idx]
        
        # Check if this window exceeds threshold
        if np.mean(window_flags) >= high_occlusion_threshold:
            # Start of occlusion period
            start_frame = frames[i]
            
            # Extend until occlusion drops below threshold
            j = i
            while j < len(frames):
                window_end_idx = min(j + window_frames, len(frames))
                window_flags = flags[j:window_end_idx]
                if np.mean(window_flags) < high_occlusion_threshold:
                    break
                j += 1
            
            end_frame = frames[min(j + window_frames - 1, len(frames) - 1)]
            periods.append((int(start_frame), int(end_frame)))
            
            # Skip past this period
            i = j + window_frames
        else:
            i += 1
    
    return periods


def compute_yolo_prescan_results(
    yolo_df: pd.DataFrame,
    fps: float = 25.0,
    window_seconds: float = 1.0,
    high_occlusion_threshold: float = 0.3,
    occlusion_iou_threshold: float = 0.15,
    clustering_distance_threshold: float = 0.15,
) -> Dict[str, Any]:
    """
    Run full YOLO-based pre-scan analysis to identify occlusion periods.
    
    This is the main entry point that parallels the KMeans pre-scan but uses
    semantic object detection instead of pixel clustering.
    
    Args:
        yolo_df: YOLO tracking DataFrame
        fps: video frame rate
        window_seconds: sliding window duration for occlusion detection
        high_occlusion_threshold: fraction of frames flagged in window
        occlusion_iou_threshold: bbox IoU threshold for overlap
        clustering_distance_threshold: normalized centroid distance threshold
    
    Returns:
        Dict containing:
        - per_frame_metrics: List[Dict] with spatial metrics per frame
        - occlusion_periods: List[(start_frame, end_frame)] high-occlusion periods
        - transition_frames: np.ndarray of frames where occlusion changes
        - total_frames: int, total frame count
        - video_duration_seconds: float
        - fps: float
    """
    window_frames = int(window_seconds * fps)
    
    # Compute per-frame metrics
    per_frame_metrics = compute_yolo_per_frame_metrics(
        yolo_df,
        occlusion_iou_threshold=occlusion_iou_threshold,
        clustering_distance_threshold=clustering_distance_threshold,
        use_normalized_coords=True,
    )
    
    # Identify occlusion periods
    occlusion_periods = identify_occlusion_periods(
        per_frame_metrics,
        window_frames=window_frames,
        high_occlusion_threshold=high_occlusion_threshold,
    )
    
    # Extract transition frames (start/end of each period)
    transition_frames = []
    for start, end in occlusion_periods:
        transition_frames.extend([start, end])
    transition_frames = np.array(sorted(set(transition_frames)))
    
    total_frames = yolo_df["frame"].max()
    video_duration = total_frames / fps
    
    return {
        "per_frame_metrics": per_frame_metrics,
        "occlusion_periods": occlusion_periods,
        "transition_frames": transition_frames,
        "total_frames": int(total_frames),
        "video_duration_seconds": float(video_duration),
        "fps": float(fps),
        "window_frames": int(window_frames),
        "high_occlusion_threshold": float(high_occlusion_threshold),
    }


def yolo_prescan_to_df(
    per_frame_metrics: List[Dict[str, Any]]
) -> pd.DataFrame:
    """
    Convert per-frame metrics list to a DataFrame for analysis/export.
    
    Similar to metrics.per_frame_metrics_to_df.
    """
    if not per_frame_metrics:
        return pd.DataFrame()
    
    rows = []
    for d in per_frame_metrics:
        r = d.copy()
        # Convert list to comma-separated string
        r["objects_present"] = ",".join(str(x) for x in r.get("objects_present", []))
        rows.append(r)
    
    return pd.DataFrame(rows)
