"""
YOLO-based occlusion period detection for adaptive chunking scan.

Computes per-frame spatial and overlap metrics from YOLO tracking outputs,
providing a semantic alternative to pixel-clustering (KMeans) for identifying
high-occlusion regions.

The ``run_yolo_scan`` function runs YOLO+ByteTrack inference inline on a
video file, builds a tracking DataFrame, and calls ``compute_yolo_scan_results``
to produce transition frames suitable for ``chunk_video_frames_adaptive``.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from loguru import logger
from omegaconf import OmegaConf
from tqdm import tqdm


def _draw_label(img: np.ndarray, label: str, x1: float, y1: float) -> None:
    """
    Draw a filled rectangle behind text for readability, positioned above the box.

    Args:
        img: Image array (modified in-place)
        label: Text to draw
        x1, y1: Top-left corner of bounding box
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 1
    (w, h), _ = cv2.getTextSize(label, font, scale, thickness)
    top_left = (int(x1), int(y1) - h - 6)
    bottom_right = (int(x1) + w + 6, int(y1))
    cv2.rectangle(img, top_left, bottom_right, (0, 0, 0), -1)
    cv2.putText(
        img,
        label,
        (int(x1) + 3, int(y1) - 4),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


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


def compute_clustering_coefficient(centroids: np.ndarray, threshold: float) -> float:
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
    separation_min_objects: int = 3,
    separation_min_distance: float = 0.15,
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
        separation_min_objects: minimum objects for a non-zero separation_score
        separation_min_distance: minimum min_centroid_distance for separation_score

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
        - separation_score: per-frame subject separation quality in [0, 1]
    """
    results: List[Dict[str, Any]] = []
    prev_num_objects: Optional[int] = None

    # Sort by frame to ensure sequential processing
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
            min_dist = (
                float(np.nanmin(upper_dists)) if len(upper_dists) else float("inf")
            )
            mean_dist = (
                float(np.nanmean(upper_dists)) if len(upper_dists) else float("inf")
            )
        else:
            min_dist = float("inf")
            mean_dist = float("inf")

        clust_coef = compute_clustering_coefficient(
            centroids, clustering_distance_threshold
        )

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


def find_high_separation_windows(
    per_frame_metrics: List[Dict[str, Any]],
    min_objects: int = 3,
    min_separation_distance: float = 0.15,
    min_window_frames: int = 25,
    gap_tolerance_frames: int = 5,
) -> List[Tuple[int, int]]:
    """
    Find sustained periods of high subject separation.

    A frame qualifies when its separation score > 0 (i.e. num_objects >=
    min_objects, num_overlapping_pairs == 0, and min_centroid_distance >=
    min_separation_distance).  Contiguous runs of qualifying frames — where
    the gap to the next qualifying frame is <= gap_tolerance_frames — that
    span >= min_window_frames total are returned as (start_frame, end_frame)
    tuples.

    Args:
        per_frame_metrics: Output of compute_yolo_per_frame_metrics.
        min_objects: Minimum detections required for a frame to qualify.
        min_separation_distance: Normalised centroid distance threshold.
        min_window_frames: Minimum span (in frames) for a window to be kept.
        gap_tolerance_frames: Frames with no detections that can be bridged
            within a run without breaking it.

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
        ) > 0.0
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

    # Flush trailing run
    if run_end - run_start + 1 >= min_window_frames:
        windows.append((run_start, run_end))

    return windows


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


def run_yolo_scan(
    video_path: str | Path,
    fps: float,
    total_frames: int,
    device: str,
    model_name: str = "model/yolo26x.pt",
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.45,
    tracker_config: str = "data/yolo/bytetrack.yaml",
    allowed_classes: set[str] | None = None,
    window_seconds: float = 1.0,
    high_occlusion_threshold: float = 0.3,
    occlusion_iou_threshold: float = 0.15,
    clustering_distance_threshold: float = 0.15,
    separation_min_objects: int = 3,
    separation_min_distance: float = 0.15,
    separation_min_window_seconds: float = 1.0,
    separation_gap_tolerance_frames: int = 5,
    output_video_path: str | Path | None = None,
) -> Dict[str, Any]:
    """
    Run YOLO+ByteTrack inference on a video and compute scan results.

    Loads a YOLO model, runs tracking with ``model.track(stream=True)``,
    builds a per-detection DataFrame, then delegates to
    ``compute_yolo_scan_results`` for occlusion/transition analysis.
    The YOLO model and GPU memory are released before returning.

    Args:
        video_path: Path to the input video file.
        fps: Video frame rate (used for windowed occlusion detection).
        total_frames: Total frame count (for logging only).
        device: Device string (e.g. ``"cuda:0"``, ``"cpu"``). Typically
            derived from ``Accelerator().device`` which respects the
            ``CUDA_VISIBLE_DEVICES`` env var set from the config YAML.
        model_name: YOLO model weight file (e.g. ``"yolo11x.pt"``).
        conf_thresh: Confidence threshold for detections.
        iou_thresh: IoU threshold for NMS.
        tracker_config: Path to ByteTrack YAML config.
        allowed_classes: Set of class name strings to keep (e.g. ``{"bird"}``).
            ``None`` keeps all classes.
        window_seconds: Sliding window for occlusion period detection.
        high_occlusion_threshold: Fraction of flagged frames in window.
        occlusion_iou_threshold: Bbox IoU above which a pair is overlapping.
        clustering_distance_threshold: Normalized centroid distance threshold.
        output_video_path: Optional path to save annotated video with bounding boxes.
            If ``None``, no video is written.

    Returns:
        Dict with keys:
        - ``yolo_df``: Raw tracking DataFrame (one row per detection per frame).
        - ``scan_results``: Output of ``compute_yolo_scan_results``.
        - ``model_name``, ``conf_thresh``, ``iou_thresh``: Echo of config.
    """
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "ultralytics is required for YOLO scan. "
            "Install it with: pixi install -e sam3-hf"
        ) from exc

    from src.utils import free_gpu_memory

    video_path = str(video_path)

    logger.info(
        f"YOLO scan: model={model_name}, device={device}, tracker={tracker_config}"
    )
    logger.info("Tracker config:")
    logger.info(OmegaConf.load(tracker_config))

    model = YOLO(model_name)

    # Build allowed class IDs from names if specified
    allowed_class_ids: set[int] | None = None
    if allowed_classes is not None:
        allowed_class_ids = {
            cid for cid, name in model.names.items() if name in allowed_classes
        }
        logger.info(
            f"YOLO scan: filtering to classes {allowed_classes} "
            f"(IDs: {allowed_class_ids})"
        )

    # Set up video writer if output path is provided
    video_writer = None
    if output_video_path is not None:
        import cv2

        # Get video metadata for writer
        cap_meta = cv2.VideoCapture(str(video_path))
        video_fps = cap_meta.get(cv2.CAP_PROP_FPS)
        video_fps = float(video_fps) if video_fps and video_fps > 0 else fps
        width = int(cap_meta.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap_meta.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap_meta.release()

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            str(output_video_path), fourcc, video_fps, (width, height)
        )
        logger.info(
            f"YOLO scan: writing annotated video to {output_video_path} "
            f"({width}x{height} @ {video_fps:.1f} FPS)"
        )

    # Run tracking with stream mode for memory efficiency
    rows: list[dict] = []
    frame_count = 0

    results_gen = model.track(
        source=video_path,
        stream=True,
        persist=True,
        tracker=tracker_config,
        device=device,
        verbose=False,
    )

    pbar = tqdm(
        total=total_frames,
        desc="YOLO scan",
        unit="frames",
        dynamic_ncols=True,
    )

    for result in results_gen:
        # Stop after processing total_frames
        if frame_count >= total_frames:
            break

        # Get frame image for video writing
        frame_img = result.orig_img.copy() if video_writer is not None else None

        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            # Write blank frame if video writer is active
            if video_writer is not None and frame_img is not None:
                video_writer.write(frame_img)
            frame_count += 1
            pbar.update(1)
            continue

        # Get image dimensions for normalization
        img_h, img_w = result.orig_shape

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        track_ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None

        for i in range(len(xyxy)):
            # Filter by allowed classes
            if allowed_class_ids is not None and cls_ids[i] not in allowed_class_ids:
                continue

            x1, y1, x2, y2 = xyxy[i]
            cx_norm = ((x1 + x2) / 2) / img_w
            cy_norm = ((y1 + y2) / 2) / img_h
            tid = int(track_ids[i]) if track_ids is not None else -1

            rows.append(
                {
                    "frame": frame_count,
                    "track_id": tid,
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                    "cx_norm": float(cx_norm),
                    "cy_norm": float(cy_norm),
                    "confidence": float(confs[i]),
                }
            )

            # Draw bounding box and label on frame for video
            if video_writer is not None and frame_img is not None:
                color = (0, 255, 0)  # Green
                cv2.rectangle(
                    frame_img,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    color,
                    2,
                )
                label = f"ID {tid} conf={confs[i]:.2f}"
                _draw_label(frame_img, label, x1, y1)

        # Write annotated frame to video
        if video_writer is not None and frame_img is not None:
            video_writer.write(frame_img)

        frame_count += 1
        pbar.update(1)

    pbar.close()
    logger.info(f"YOLO scan: processed {frame_count} frames, {len(rows)} detections")

    # Release video writer if it was used
    if video_writer is not None:
        video_writer.release()
        logger.info(f"YOLO scan: annotated video saved to {output_video_path}")

    # Clean up YOLO model and free GPU
    del model
    free_gpu_memory()
    logger.info("YOLO scan: model unloaded, GPU memory freed")

    yolo_df = pd.DataFrame(rows)

    if yolo_df.empty:
        logger.warning("YOLO scan: no detections found")
        return {
            "yolo_df": yolo_df,
            "scan_results": {
                "per_frame_metrics": [],
                "occlusion_periods": [],
                "transition_frames": np.array([], dtype=int),
                "total_frames": total_frames,
                "video_duration_seconds": total_frames / fps,
                "fps": fps,
                "window_frames": int(window_seconds * fps),
                "high_occlusion_threshold": high_occlusion_threshold,
            },
            "model_name": model_name,
            "conf_thresh": conf_thresh,
            "iou_thresh": iou_thresh,
        }

    # Compute scan results (per-frame metrics, occlusion periods, transitions)
    scan_results = compute_yolo_scan_results(
        yolo_df,
        fps=fps,
        window_seconds=window_seconds,
        high_occlusion_threshold=high_occlusion_threshold,
        occlusion_iou_threshold=occlusion_iou_threshold,
        clustering_distance_threshold=clustering_distance_threshold,
        separation_min_objects=separation_min_objects,
        separation_min_distance=separation_min_distance,
        separation_min_window_seconds=separation_min_window_seconds,
        separation_gap_tolerance_frames=separation_gap_tolerance_frames,
    )

    logger.info(
        f"YOLO scan: {len(scan_results['occlusion_periods'])} occlusion periods, "
        f"{len(scan_results['transition_frames'])} transition frames"
    )

    return {
        "yolo_df": yolo_df,
        "scan_results": scan_results,
        "model_name": model_name,
        "conf_thresh": conf_thresh,
        "iou_thresh": iou_thresh,
    }


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

    This is the main entry point that uses semantic object detection to identify
    high-occlusion regions and high-separation windows for adaptive chunking.

    Args:
        yolo_df: YOLO tracking DataFrame
        fps: video frame rate
        window_seconds: sliding window duration for occlusion detection
        high_occlusion_threshold: fraction of frames flagged in window
        occlusion_iou_threshold: bbox IoU threshold for overlap
        clustering_distance_threshold: normalized centroid distance threshold
        separation_min_objects: minimum detections for a frame to qualify as
            high-separation (used for both per-frame score and window detection)
        separation_min_distance: minimum normalised centroid distance for
            a frame to qualify as high-separation
        separation_min_window_seconds: minimum duration (seconds) of a
            sustained high-separation run to be returned as a window
        separation_gap_tolerance_frames: frames with no qualifying detections
            that can be bridged without breaking a separation run

    Returns:
        Dict containing:
        - per_frame_metrics: List[Dict] with spatial metrics + separation_score
        - occlusion_periods: List[(start_frame, end_frame)] high-occlusion periods
        - transition_frames: np.ndarray of frames where occlusion changes
        - separation_windows: List[(start_frame, end_frame)] high-separation windows
        - total_frames: int, total frame count
        - video_duration_seconds: float
        - fps: float
    """
    window_frames = int(window_seconds * fps)
    separation_min_window_frames = max(1, int(separation_min_window_seconds * fps))

    # Compute per-frame metrics (includes separation_score)
    per_frame_metrics = compute_yolo_per_frame_metrics(
        yolo_df,
        occlusion_iou_threshold=occlusion_iou_threshold,
        clustering_distance_threshold=clustering_distance_threshold,
        use_normalized_coords=True,
        separation_min_objects=separation_min_objects,
        separation_min_distance=separation_min_distance,
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

    # Identify high-separation windows
    separation_windows = find_high_separation_windows(
        per_frame_metrics,
        min_objects=separation_min_objects,
        min_separation_distance=separation_min_distance,
        min_window_frames=separation_min_window_frames,
        gap_tolerance_frames=separation_gap_tolerance_frames,
    )

    total_frames = yolo_df["frame"].max()
    video_duration = total_frames / fps

    logger.info(
        f"YOLO scan: {len(separation_windows)} high-separation windows found "
        f"(min_distance={separation_min_distance}, min_objects={separation_min_objects})"
    )

    return {
        "per_frame_metrics": per_frame_metrics,
        "occlusion_periods": occlusion_periods,
        "transition_frames": transition_frames,
        "separation_windows": separation_windows,
        "total_frames": int(total_frames),
        "video_duration_seconds": float(video_duration),
        "fps": float(fps),
        "window_frames": int(window_frames),
        "high_occlusion_threshold": float(high_occlusion_threshold),
    }


def yolo_scan_to_df(per_frame_metrics: List[Dict[str, Any]]) -> pd.DataFrame:
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
