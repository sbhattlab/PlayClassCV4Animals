"""
YOLO-based raw inference scan.

Runs YOLO+ByteTrack on a video and returns a per-detection DataFrame
(``yolo_tracking.parquet``).  All downstream analysis — per-frame metrics,
occlusion periods, separation windows, and adaptive chunking — lives in
``src.yolo.boundaries``.

Re-exports from ``src.yolo.boundaries`` are provided so that existing callers continue to work unchanged.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
from loguru import logger
from omegaconf import OmegaConf
from tqdm import tqdm

from src.memory import free_gpu_memory
from src.metrics import compute_separation_score, compute_yolo_per_frame_metrics


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
# Main inference entry point
# ---------------------------------------------------------------------------


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
    ``compute_yolo_scan_results`` (from ``src.chunk_boundaries``) for
    occlusion/transition analysis.  The YOLO model and GPU memory are
    released before returning.

    Args:
        video_path: Path to the input video file.
        fps: Video frame rate.
        total_frames: Total frame count (for logging only).
        device: Device string (e.g. ``"cuda:0"``).
        model_name: YOLO model weight file.
        conf_thresh: Confidence threshold for detections.
        iou_thresh: IoU threshold for NMS.
        tracker_config: Path to ByteTrack YAML config.
        allowed_classes: Set of class name strings to keep. ``None`` keeps all.
        window_seconds: Sliding window for occlusion period detection.
        high_occlusion_threshold: Fraction of flagged frames in window.
        occlusion_iou_threshold: Bbox IoU above which a pair is overlapping.
        clustering_distance_threshold: Normalized centroid distance threshold.
        separation_min_objects: Min detections for a frame to qualify as separated.
        separation_min_distance: Min normalised centroid distance for separation.
        separation_min_window_seconds: Min duration for a separation window.
        separation_gap_tolerance_frames: Bridgeable gap within a separation run.
        output_video_path: Optional path to save annotated video.

    Returns:
        Dict with keys: ``yolo_df``, ``scan_results``, ``model_name``,
        ``conf_thresh``, ``iou_thresh``.
    """
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "ultralytics is required for YOLO scan. "
            "Install it with: pixi install -e sam3-hf"
        ) from exc

    video_path = str(video_path)

    logger.info(
        f"YOLO scan: model={model_name}, device={device}, tracker={tracker_config}"
    )
    logger.info("Tracker config:")
    logger.info(OmegaConf.load(tracker_config))

    model = YOLO(model_name)

    allowed_class_ids: set[int] | None = None
    if allowed_classes is not None:
        allowed_class_ids = {
            cid for cid, name in model.names.items() if name in allowed_classes
        }
        logger.info(
            f"YOLO scan: filtering to classes {allowed_classes} "
            f"(IDs: {allowed_class_ids})"
        )

    video_writer = None
    if output_video_path is not None:
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
        if frame_count >= total_frames:
            break

        frame_img = result.orig_img.copy() if video_writer is not None else None

        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            if video_writer is not None and frame_img is not None:
                video_writer.write(frame_img)
            frame_count += 1
            pbar.update(1)
            continue

        img_h, img_w = result.orig_shape

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        track_ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None

        for i in range(len(xyxy)):
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

            if video_writer is not None and frame_img is not None:
                cv2.rectangle(
                    frame_img,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 255, 0),
                    2,
                )
                _draw_label(frame_img, f"ID {tid} conf={confs[i]:.2f}", x1, y1)

        if video_writer is not None and frame_img is not None:
            video_writer.write(frame_img)

        frame_count += 1
        pbar.update(1)

    pbar.close()
    logger.info(f"YOLO scan: processed {frame_count} frames, {len(rows)} detections")

    if video_writer is not None:
        video_writer.release()
        logger.info(f"YOLO scan: annotated video saved to {output_video_path}")

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
                "separation_windows": [],
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
