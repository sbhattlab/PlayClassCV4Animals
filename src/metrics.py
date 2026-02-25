"""
Tracking metrics computation for SAM3 video outputs.
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.processing import (
    _normalize_frame_dict,
    to_numpy,)


# ---------------------------------------------------------------------------
# Low-level metric helpers (moved from src.processing and src.yolo_scan)
# ---------------------------------------------------------------------------


def compute_bbox_iou(boxA, boxB) -> float:
    """
    Compute IoU between two bounding boxes in [x1, y1, x2, y2] format.

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


def compute_pairwise_centroid_distances(centroids: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Euclidean distances between centroids.

    Args:
        centroids: (N, 2) array of [x, y] or [cx, cy] centroids.

    Returns:
        (N, N) symmetric distance matrix.
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


def compute_clustering_coefficient(centroids: np.ndarray, threshold: float) -> float:
    """
    Fraction of centroid pairs within *threshold* distance.

    Args:
        centroids: (N, 2) array.
        threshold: distance (pixels or normalized coordinates).

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


def compute_max_pairwise_iou(masks_np: np.ndarray) -> float:
    """Return max pixel-IoU over all pairs of binary masks (N, H, W)."""
    n = len(masks_np)
    max_iou = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            inter = float((masks_np[i] & masks_np[j]).sum())
            union = float((masks_np[i] | masks_np[j]).sum())
            if union > 0:
                max_iou = max(max_iou, inter / union)
    return max_iou


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


# ---------------------------------------------------------------------------
# Mask-based spatial primitives
# ---------------------------------------------------------------------------


def compute_pairwise_maskcompute_bbox_iou(masks: np.ndarray) -> np.ndarray:
    """
    Compute pairwise pixel-level IoU for all masks.

    Args:
        masks: (N, H, W) bool or uint8 array.

    Returns:
        (N, N) symmetric IoU matrix with zeros on diagonal.
    """
    masks = to_numpy(masks)
    if masks.ndim == 2:
        masks = masks[None, ...]
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


def compute_mask_centroids(masks: np.ndarray) -> np.ndarray:
    """
    Compute centroids from mask pixel coordinates.

    Args:
        masks: (N, H, W) bool array.

    Returns:
        (N, 2) array of [x, y] centroids.  NaN for empty masks.
    """
    masks = to_numpy(masks)
    if masks.ndim == 2:
        masks = masks[None, ...]
    centroids = []
    for mask in masks:
        y_coords, x_coords = np.where(mask > 0)
        if len(y_coords) > 0:
            centroids.append([float(np.mean(x_coords)), float(np.mean(y_coords))])
        else:
            centroids.append([np.nan, np.nan])
    return np.array(centroids)


def compute_mask_area_stats(masks: np.ndarray) -> Dict[str, Any]:
    """
    Compute mask area statistics.

    Args:
        masks: (N, H, W) bool array.

    Returns:
        Dict with keys: areas (N,), mean, min, max, variance.
    """
    masks = to_numpy(masks)
    if masks.ndim == 2:
        masks = masks[None, ...]
    areas = np.array([m.sum() for m in masks], dtype=float)
    if len(areas) == 0:
        return {"areas": areas, "mean": 0.0, "min": 0.0, "max": 0.0, "variance": 0.0}
    return {
        "areas": areas,
        "mean": float(np.mean(areas)),
        "min": float(np.min(areas)),
        "max": float(np.max(areas)),
        "variance": float(np.var(areas)),
    }


# ---------------------------------------------------------------------------
# Per-frame timeseries metrics (operates directly on outputs_per_frame)
# ---------------------------------------------------------------------------


def compute_per_frame_metrics(
    outputs_per_frame: Dict[int, Dict],
    occlusion_iou_threshold: float = 0.15,
    clustering_distance_threshold: float = 50.0,
) -> List[Dict[str, Any]]:
    """
    Compute per-frame spatial, overlap, and mask quality metrics.

    Works directly on ``outputs_per_frame`` (the SAM3 output dict) so that
    it can access masks — which ``_normalize_frame_dict`` discards.

    Args:
        outputs_per_frame: {frame_idx: {"object_ids", "masks", "boxes", "scores", ...}}
        occlusion_iou_threshold: mask IoU above which a pair is "overlapping".
        clustering_distance_threshold: centroid distance (px) for clustering.

    Returns:
        List of dicts (sorted by frame_idx), one per frame.
    """
    sorted_idxs = sorted(int(k) for k in outputs_per_frame.keys())
    results: List[Dict[str, Any]] = []
    prev_num_objects: Optional[int] = None

    for frame_idx in sorted_idxs:
        v = outputs_per_frame[frame_idx]

        # Extract arrays
        obj_ids = to_numpy(v.get("object_ids", []))
        masks = to_numpy(v.get("masks", np.empty((0, 0, 0))))
        if masks.ndim == 2:
            masks = masks[None, ...]

        n = len(obj_ids)

        if n == 0:
            results.append(
                {
                    "frame_idx": int(frame_idx),
                    "num_objects": 0,
                    "objects_present": [],
                    "min_centroid_distance": float("inf"),
                    "mean_centroid_distance": float("inf"),
                    "clustering_coefficient": 0.0,
                    "max_pairwise_mask_iou": 0.0,
                    "mean_pairwise_mask_iou": 0.0,
                    "num_overlapping_pairs": 0,
                    "mean_mask_area": 0.0,
                    "min_mask_area": 0.0,
                    "max_mask_area": 0.0,
                    "mask_area_variance": 0.0,
                    "is_high_occlusion": False,
                    "is_object_count_change": prev_num_objects is not None
                    and prev_num_objects != 0,
                }
            )
            prev_num_objects = 0
            continue

        # Centroids & distances
        centroids = compute_mask_centroids(masks)
        dist_matrix = compute_pairwise_centroid_distances(centroids)

        if n > 1:
            upper_dists = dist_matrix[np.triu_indices(n, k=1)]
            min_dist = float(np.nanmin(upper_dists)) if len(upper_dists) else float("inf")
            mean_dist = (
                float(np.nanmean(upper_dists)) if len(upper_dists) else float("inf")
            )
        else:
            min_dist = float("inf")
            mean_dist = float("inf")

        clust_coef = compute_clustering_coefficient(centroids, clustering_distance_threshold)

        # Mask IoU
        iou_matrix = compute_pairwise_maskcompute_bbox_iou(masks)
        if n > 1:
            upper_iou = iou_matrix[np.triu_indices(n, k=1)]
            max_iou = float(np.max(upper_iou)) if len(upper_iou) else 0.0
            mean_iou = float(np.mean(upper_iou)) if len(upper_iou) else 0.0
            num_overlap = int(np.sum(upper_iou > occlusion_iou_threshold))
        else:
            max_iou = 0.0
            mean_iou = 0.0
            num_overlap = 0

        # Mask areas
        area_stats = compute_mask_area_stats(masks)

        # Flags
        is_high_occ = max_iou > occlusion_iou_threshold or clust_coef > 0.5
        is_count_change = prev_num_objects is not None and prev_num_objects != n

        results.append(
            {
                "frame_idx": int(frame_idx),
                "num_objects": n,
                "objects_present": [int(oid) for oid in obj_ids],
                "min_centroid_distance": min_dist,
                "mean_centroid_distance": mean_dist,
                "clustering_coefficient": float(clust_coef),
                "max_pairwise_mask_iou": max_iou,
                "mean_pairwise_mask_iou": mean_iou,
                "num_overlapping_pairs": num_overlap,
                "mean_mask_area": area_stats["mean"],
                "min_mask_area": area_stats["min"],
                "max_mask_area": area_stats["max"],
                "mask_area_variance": area_stats["variance"],
                "is_high_occlusion": is_high_occ,
                "is_object_count_change": is_count_change,
            }
        )
        prev_num_objects = n

    return results


def per_frame_metrics_to_df(
    per_frame_metrics: List[Dict[str, Any]],
) -> pd.DataFrame:
    """Convert per-frame metrics list to a DataFrame."""
    if not per_frame_metrics:
        return pd.DataFrame()
    rows = []
    for d in per_frame_metrics:
        r = d.copy()
        # Convert list to comma-separated string for storage
        r["objects_present"] = ",".join(str(x) for x in r.get("objects_present", []))
        rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Occlusion-aware identity switch detection
# ---------------------------------------------------------------------------


def detect_identity_switches(
    per_frame_metrics: List[Dict[str, Any]],
    window_size: int = 5,
) -> List[Tuple[int, int, int]]:
    """
    Detect suspected identity switches using an occlusion-aware heuristic.

    A switch is flagged when:
      1. Object count is stable between consecutive frames, but IDs changed.
      2. A high-occlusion event occurred within the preceding *window_size* frames.

    Args:
        per_frame_metrics: output of :func:`compute_per_frame_metrics`.
        window_size: number of frames to look back for occlusion events.

    Returns:
        List of ``(frame_idx, old_id, new_id)`` tuples.
    """
    switches: List[Tuple[int, int, int]] = []

    for i, metrics in enumerate(per_frame_metrics):
        if i == 0:
            continue

        prev = per_frame_metrics[i - 1]

        # Stable count but different IDs?
        if metrics["num_objects"] == prev["num_objects"] > 0:
            prev_ids = set(prev["objects_present"])
            curr_ids = set(metrics["objects_present"])

            disappeared = prev_ids - curr_ids
            appeared = curr_ids - prev_ids

            if disappeared and appeared and len(disappeared) == len(appeared):
                # Check for recent occlusion
                recent_occlusion = any(
                    per_frame_metrics[j].get("is_high_occlusion", False)
                    for j in range(max(0, i - window_size), i)
                )

                if recent_occlusion:
                    for old_id, new_id in zip(sorted(disappeared), sorted(appeared)):
                        switches.append((metrics["frame_idx"], old_id, new_id))

    return switches


# ---------------------------------------------------------------------------
# Per-ID / Per-run / Summary metrics (box-based, via _normalize_frame_dict)
# ---------------------------------------------------------------------------


def compute_per_id_metrics(
    frame_dict: Dict[Any, Any], low_count_threshold: int = 3, iou_thresh: float = 0.5
) -> Dict[Any, Dict[str, Any]]:
    """
    Returns: { id: {first_frame, last_frame, length, runs, gaps_total, frames,
                     coverage, low_count_frames, low_count_total, low_count_fraction,
                     self_iou, spatial_continuity_iou, mean_iou (alias for self_iou),
                     mean_bbox_area (optional)} }
    """
    idxs, frames = _normalize_frame_dict(frame_dict)
    if not idxs:
        return {}

    counts = {idx: len(f) for idx, f in zip(idxs, frames)}
    low_frames_set = {idx for idx, c in counts.items() if c < low_count_threshold}

    # id -> frames list and bbox areas aggregator
    id_to_frames = defaultdict(list)
    id_bbox_areas = defaultdict(list)
    for idx, f in zip(idxs, frames):
        for det in f:
            if not isinstance(det, dict):
                continue
            uid = det.get("id")
            if uid is None:
                continue
            id_to_frames[uid].append(idx)
            if "bbox" in det:
                b = det["bbox"]
                area = max(0.0, (b[2] - b[0]) * (b[3] - b[1]))
                id_bbox_areas[uid].append(area)

    # compute runs/gaps per id
    def _runs_and_gaps(fl: List[int]) -> Tuple[int, int]:
        if not fl:
            return 0, 0
        fl_sorted = sorted(fl)
        runs = 1
        gaps = 0
        for a, b in zip(fl_sorted, fl_sorted[1:]):
            if b != a + 1:
                runs += 1
                gaps += b - a - 1
        return runs, gaps

    # Greedy spatial continuity IoU (matches ANY detection by proximity, ignoring ID)
    id_spatial_ious = defaultdict(list)
    for A, B in zip(frames, frames[1:]):
        used_b = set()
        for a in A:
            if not isinstance(a, dict) or "bbox" not in a:
                continue
            best_j, best_iou = None, 0.0
            for j, b in enumerate(B):
                if j in used_b or not isinstance(b, dict) or "bbox" not in b:
                    continue
                val = compute_bbox_iou(a["bbox"], b["bbox"])
                if val > best_iou:
                    best_iou, best_j = val, j
            if best_j is not None and best_iou >= iou_thresh:
                aid = a.get("id")
                id_spatial_ious[aid].append(best_iou)

    # Identity-aware self-IoU (same ID across consecutive frames)
    frame_id_map: Dict[int, Dict[Any, Dict]] = {}
    for idx, f in zip(idxs, frames):
        frame_id_map[idx] = {
            d.get("id"): d for d in f if isinstance(d, dict) and "id" in d
        }

    id_self_ious: Dict[Any, List[float]] = defaultdict(list)
    for uid, flist in id_to_frames.items():
        fl_sorted = sorted(flist)
        for a, b in zip(fl_sorted, fl_sorted[1:]):
            da = frame_id_map.get(a, {}).get(uid)
            db = frame_id_map.get(b, {}).get(uid)
            if da and db and "bbox" in da and "bbox" in db:
                id_self_ious[uid].append(compute_bbox_iou(da["bbox"], db["bbox"]))

    per_id = {}
    for uid, flist in id_to_frames.items():
        fl_sorted = sorted(flist)
        first, last = fl_sorted[0], fl_sorted[-1]
        length = len(fl_sorted)
        runs, gaps = _runs_and_gaps(fl_sorted)
        span = last - first + 1 if last >= first else 0
        coverage = length / span if span > 0 else 0.0
        low_in_span = [f for f in fl_sorted if f in low_frames_set]
        low_total = len(low_in_span)
        low_frac = low_total / span if span > 0 else 0.0
        self_iou = float(np.mean(id_self_ious[uid])) if id_self_ious.get(uid) else None
        spatial_continuity_iou = (
            float(np.mean(id_spatial_ious[uid]))
            if id_spatial_ious.get(uid)
            else None
        )
        mean_area = (
            float(np.mean(id_bbox_areas[uid])) if id_bbox_areas.get(uid) else None
        )

        per_id[uid] = {
            "id": uid,
            "first_frame": int(first),
            "last_frame": int(last),
            "length": int(length),
            "runs": int(runs),
            "gaps_total": int(gaps),
            "frames": fl_sorted,
            "span": int(span),
            "coverage": float(coverage),
            "low_count_frames": low_in_span,
            "low_count_total": int(low_total),
            "low_count_fraction": float(low_frac),
            "self_iou": self_iou,
            "spatial_continuity_iou": spatial_continuity_iou,
            "mean_iou": self_iou,  # backward-compat alias
            "mean_bbox_area": mean_area,
        }
    return per_id


def compute_summary_metrics(
    frame_dict: Dict[Any, Any],
    persistence_k: int = 5,
    iou_match_thresh: float = 0.5,
    low_count_threshold: int = 3,
    per_frame_metrics: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Returns a flat summary dict aggregating the usual tracking proxies.

    If *per_frame_metrics* (from :func:`compute_per_frame_metrics`) is
    supplied, occlusion-aware identity switch counts are included.
    """
    idxs, frames = _normalize_frame_dict(frame_dict)
    n_frames = len(idxs)
    dets_per_frame = [len(f) for f in frames]
    counts = {idx: len(f) for idx, f in zip(idxs, frames)}
    low_frame_count = sum(1 for c in counts.values() if c < low_count_threshold)

    # build id->frames
    id_to_frames = defaultdict(list)
    for idx, f in zip(idxs, frames):
        for det in f:
            if not isinstance(det, dict):
                continue
            uid = det.get("id")
            if uid is None:
                continue
            id_to_frames[uid].append(idx)

    track_lengths = (
        np.array([len(v) for v in id_to_frames.values()])
        if id_to_frames
        else np.array([])
    )
    mean_track_length = float(track_lengths.mean()) if track_lengths.size else 0.0
    persistence_rate = float((track_lengths >= persistence_k).sum()) / (
        track_lengths.size or 1
    )

    # fragmentation
    def _count_runs(frames_list):
        if not frames_list:
            return 0
        fl = sorted(frames_list)
        runs = 1
        for a, b in zip(fl, fl[1:]):
            if b != a + 1:
                runs += 1
        return runs

    runs_per_id = (
        np.array([_count_runs(v) for v in id_to_frames.values()])
        if id_to_frames
        else np.array([])
    )
    mean_runs = float(runs_per_id.mean()) if runs_per_id.size else 0.0

    # continuity (fraction of detections that persist to next frame)
    transitions_possible = 0
    transitions_preserved = 0
    for f_a, f_b in zip(frames, frames[1:]):
        map_b = {d.get("id"): d for d in f_b if isinstance(d, dict) and "id" in d}
        for d in f_a:
            if not isinstance(d, dict):
                continue
            uid = d.get("id")
            if uid is None:
                continue
            transitions_possible += 1
            if uid in map_b:
                transitions_preserved += 1
    continuity = transitions_preserved / (transitions_possible or 1)

    # IoU-based id-switch proxy: greedy matching
    switches = 0
    matches = 0
    for A, B in zip(frames, frames[1:]):
        if not A or not B:
            continue
        used_b = set()
        for a in A:
            if not isinstance(a, dict) or "bbox" not in a:
                continue
            best_j, best_iou = None, 0.0
            for j, b in enumerate(B):
                if j in used_b or not isinstance(b, dict) or "bbox" not in b:
                    continue
                val = compute_bbox_iou(a["bbox"], b["bbox"])
                if val > best_iou:
                    best_iou, best_j = val, j
            if best_j is not None and best_iou >= iou_match_thresh:
                matches += 1
                used_b.add(best_j)
                if a.get("id") != B[best_j].get("id"):
                    switches += 1
    id_switch_rate = switches / (matches or 1)

    # aggregate per-id mean_iou and mean coverage
    per_id = compute_per_id_metrics(
        frame_dict, low_count_threshold=low_count_threshold, iou_thresh=iou_match_thresh
    )
    mean_coverage = (
        float(np.mean([v["coverage"] for v in per_id.values()])) if per_id else None
    )
    mean_per_id_iou = (
        float(
            np.mean(
                [v["mean_iou"] for v in per_id.values() if v["mean_iou"] is not None]
            )
        )
        if per_id
        else None
    )

    summary = {
        "n_frames": int(n_frames),
        "avg_detections_per_frame": float(np.mean(dets_per_frame))
        if dets_per_frame
        else 0.0,
        "total_unique_ids": int(len(id_to_frames)),
        "mean_track_length": mean_track_length,
        f"persistence_rate_>={persistence_k}": persistence_rate,
        "mean_fragmentation_runs_per_id": mean_runs,
        "continuity_fraction": float(continuity),
        "iou_match_count": int(matches),
        "id_switches": int(switches),
        "id_switch_rate": float(id_switch_rate),
        "low_frame_count": int(low_frame_count),
        "low_frame_fraction": float(low_frame_count / n_frames) if n_frames else 0.0,
        "mean_coverage_per_id": mean_coverage,
        "mean_per_id_iou": mean_per_id_iou,
    }

    # Occlusion-aware switch count (requires per_frame_metrics from mask analysis)
    if per_frame_metrics is not None:
        occlusion_switch_events = detect_identity_switches(per_frame_metrics)
        summary["occlusion_aware_id_switches"] = len(occlusion_switch_events)
        summary["occlusion_aware_id_switch_rate"] = len(occlusion_switch_events) / (
            matches or 1
        )

    return summary


def per_id_metrics_to_df(per_id_metrics: Dict[Any, Dict[str, Any]]) -> pd.DataFrame:
    """
    Converts compute_per_id_metrics output into a DataFrame (one row per id).
    """
    if not per_id_metrics:
        return pd.DataFrame()
    rows = []
    for uid, d in per_id_metrics.items():
        r = d.copy()
        # ensure id included as column
        r["id"] = uid
        rows.append(r)
    df = pd.DataFrame(rows)
    # sensible column ordering
    cols = [
        "id",
        "first_frame",
        "last_frame",
        "length",
        "span",
        "coverage",
        "runs",
        "gaps_total",
        "low_count_total",
        "low_count_fraction",
        "low_count_frames",
        "self_iou",
        "spatial_continuity_iou",
        "mean_iou",
        "mean_bbox_area",
        "frames",
    ]
    existing = [c for c in cols if c in df.columns]
    other = [c for c in df.columns if c not in existing]
    return df[existing + other]


def summary_metrics_to_df(summary_metrics: Dict[str, Any]) -> pd.DataFrame:
    """
    Converts compute_summary_metrics output into a single-row DataFrame.
    """
    if not summary_metrics:
        return pd.DataFrame()
    return pd.DataFrame([summary_metrics])


def compute_per_run_metrics(
    frame_dict: Dict[Any, Any], low_count_threshold: int = 3, iou_thresh: float = 0.5
) -> Dict[Any, List[Dict[str, Any]]]:
    """
    Return per-object runs:
      { object_id: [ { run_idx, first_frame, last_frame, length, frames,
                       low_count_frames, low_count_total, low_count_fraction,
                       mean_iou, mean_bbox_area, mean_score }, ... ] }
    A "run" = contiguous frames where the object appears.
    """
    idxs, frames = _normalize_frame_dict(frame_dict)
    if not idxs:
        return {}

    # per-frame detection counts -> low-frame set
    counts = {idx: len(f) for idx, f in zip(idxs, frames)}
    low_frames_set = {idx for idx, c in counts.items() if c < low_count_threshold}

    # build mapping: frame_idx -> {id: det_dict}
    frame_map = {
        idx: {d.get("id"): d for d in f if isinstance(d, dict) and "id" in d}
        for idx, f in zip(idxs, frames)
    }

    # id -> sorted list of frames where it appears
    id_to_frames = defaultdict(list)
    for idx, f in zip(idxs, frames):
        for det in f:
            if not isinstance(det, dict):
                continue
            uid = det.get("id")
            if uid is None:
                continue
            id_to_frames[uid].append(idx)

    per_run = {}
    for uid, flist in id_to_frames.items():
        fl_sorted = sorted(flist)
        runs = []
        run_idx = 0
        i = 0
        while i < len(fl_sorted):
            # start new run at fl_sorted[i]
            run_frames = [fl_sorted[i]]
            j = i + 1
            while j < len(fl_sorted) and fl_sorted[j] == run_frames[-1] + 1:
                run_frames.append(fl_sorted[j])
                j += 1
            # compute run-level stats
            first = run_frames[0]
            last = run_frames[-1]
            length = len(run_frames)
            low_in_run = [f for f in run_frames if f in low_frames_set]
            low_total = len(low_in_run)
            low_frac = low_total / length if length else 0.0

            # mean IoU for this id across consecutive frames inside the run
            ious = []
            areas = []
            scores = []
            tracker_scores = []
            for a, b in zip(run_frames, run_frames[1:]):
                da = frame_map.get(a, {}).get(uid)
                db = frame_map.get(b, {}).get(uid)
                if da and db and "bbox" in da and "bbox" in db:
                    ious.append(compute_bbox_iou(da["bbox"], db["bbox"]))
            # gather area/score/tracker_score across run
            for fidx in run_frames:
                d = frame_map.get(fidx, {}).get(uid)
                if d:
                    if "bbox" in d:
                        b = d["bbox"]
                        areas.append(max(0.0, (b[2] - b[0]) * (b[3] - b[1])))
                    if "score" in d and d["score"] is not None:
                        try:
                            scores.append(float(d["score"]))
                        except Exception:
                            pass
                    if "tracker_score" in d and d["tracker_score"] is not None:
                        try:
                            tracker_scores.append(float(d["tracker_score"]))
                        except Exception:
                            pass

            mean_iou = float(np.mean(ious)) if ious else None
            mean_area = float(np.mean(areas)) if areas else None
            mean_score = float(np.mean(scores)) if scores else None
            mean_tracker_score = (
                float(np.mean(tracker_scores)) if tracker_scores else None
            )

            runs.append(
                {
                    "run_idx": int(run_idx),
                    "first_frame": int(first),
                    "last_frame": int(last),
                    "length": int(length),
                    "frames": run_frames,
                    "low_count_frames": low_in_run,
                    "low_count_total": int(low_total),
                    "low_count_fraction": float(low_frac),
                    "mean_iou": mean_iou,
                    "mean_bbox_area": mean_area,
                    "mean_score": mean_score,
                    "mean_tracker_score": mean_tracker_score,
                }
            )

            run_idx += 1
            i = j
        per_run[uid] = runs
    return per_run


def per_run_metrics_to_multiindex_df(
    per_run_metrics: Dict[Any, List[Dict[str, Any]]],
) -> pd.DataFrame:
    """
    Flatten per-run metrics into a DataFrame indexed by MultiIndex (object_id, run_idx).
    Columns: first_frame, last_frame, length, frames, low_count_total, low_count_fraction, mean_iou, mean_bbox_area, mean_score, ...
    """
    rows = []
    for uid, runs in per_run_metrics.items():
        for r in runs:
            row = r.copy()
            row["id"] = uid
            rows.append(row)
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # ensure run_idx exists and is int
    if "run_idx" not in df.columns:
        df["run_idx"] = 0
    df["run_idx"] = df["run_idx"].astype(int)
    df["id"] = df["id"].astype(
        type(list(per_run_metrics.keys())[0]) if per_run_metrics else int
    )

    # set MultiIndex (object_id outer, run_idx inner)
    df = df.set_index(["id", "run_idx"]).sort_index()
    # order columns sensibly
    cols = [
        "first_frame",
        "last_frame",
        "length",
        "low_count_total",
        "low_count_fraction",
        "mean_iou",
        "mean_bbox_area",
        "mean_score",
        "mean_tracker_score",
        "frames",
        "low_count_frames",
    ]
    existing = [c for c in cols if c in df.columns]
    other = [c for c in df.columns if c not in existing]
    return df[existing + other]
