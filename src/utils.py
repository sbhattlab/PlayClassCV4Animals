"""
Utility functions for processing SAM3 tracking outputs.
"""

import gc
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pycocotools.mask as mask_util
import supervision as sv
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from sklearn.cluster import MiniBatchKMeans


# ---------------------------------------------------------------------------
# Config, environment, logging, and output directory utilities
# ---------------------------------------------------------------------------


def load_config(config_path: str | Path) -> DictConfig:
    """Load configuration from a YAML file using OmegaConf."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    cfg = OmegaConf.load(config_path)
    return cfg


def set_env_vars(cfg):
    """Set environment variables from config (must run before torch import)."""
    if cfg.get("CUDA_VISIBLE_DEVICES"):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.CUDA_VISIBLE_DEVICES)
    if cfg.get("PYTORCH_ALLOC_CONF"):
        os.environ["PYTORCH_ALLOC_CONF"] = str(cfg.PYTORCH_ALLOC_CONF)


def setup_logger(
    log_dir: Path, job_type: str = "sam3_demo", debug: bool = False
) -> Path:
    """
    Configure loguru logger with both console and file output.

    Returns:
        Path to the log file.
    """
    level = "DEBUG" if debug else "INFO"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = log_dir / f"{job_type}_{timestamp}.log"

    logger.remove()

    # Console handler (colored)
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> "
        "[<level>{level}</level>] {message}",
        level=level,
    )

    # File handler
    logger.add(
        str(log_filename),
        format="{time:YYYY-MM-DD HH:mm:ss} [{level}] {message}",
        level=level,
        enqueue=True,
        backtrace=True,
        diagnose=True,
    )

    return log_filename


def create_run_directory(base_output_dir: Path, job_type: str) -> Path:
    """Create a timestamped run directory for this job."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_output_dir / f"{timestamp}_{job_type}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


# ---------------------------------------------------------------------------
# Chunking and model transition utilities
# ---------------------------------------------------------------------------


@dataclass
class PrescanResult:
    """Result of KMeans-based video pre-scan for occlusion period detection."""

    frame_indices: np.ndarray  # sampled frame indices (in original video coordinates)
    cluster_labels: np.ndarray  # cluster label per sampled frame
    transition_frames: np.ndarray  # frame indices where cluster label changes


def prescan_occlusion_periods(
    video_path: str | Path,
    fps: float,
    total_frames: int | None = None,
) -> PrescanResult:
    """
    Pre-scan video with MiniBatchKMeans to identify visual scene-state transitions.

    Samples ~1fps, downscales to 30px width grayscale, clusters frames, and
    identifies transition points where the visual state changes. Occlusion
    periods (chickens clustered/overlapping) form a distinct cluster.

    Inspired by DeepLabCut's MiniBatchKMeans frame selection.

    Args:
        video_path: Path to the video file.
        fps: Video frame rate (used to compute sample interval).
        total_frames: If set, only scan up to this many frames.

    Returns:
        PrescanResult with frame_indices, cluster_labels, and transition_frames.
    """
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    sample_interval = max(1, int(fps))  # ~1fps
    target_width = 30  # DLC convention

    frame_indices = []
    frame_data = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if total_frames is not None and frame_idx >= total_frames:
            break
        if frame_idx % sample_interval == 0:
            # Downsample to target_width, preserving aspect ratio
            h, w = frame.shape[:2]
            scale = target_width / w
            new_h = max(1, int(h * scale))
            small = cv2.resize(frame, (target_width, new_h), interpolation=cv2.INTER_AREA)
            # Grayscale via mean across channels
            gray = np.mean(small, axis=2).flatten()
            frame_data.append(gray)
            frame_indices.append(frame_idx)
        frame_idx += 1

    cap.release()

    frame_indices = np.array(frame_indices)
    data = np.array(frame_data, dtype=np.float32)

    # Mean-center
    data -= data.mean(axis=0)

    # Auto-determine number of clusters
    video_duration = frame_idx / fps
    n_clusters = max(2, int(video_duration // 5))
    # Cap to avoid more clusters than samples
    n_clusters = min(n_clusters, len(data))

    logger.info(
        f"Pre-scan: {len(data)} sampled frames, {video_duration:.1f}s duration, "
        f"{n_clusters} clusters"
    )

    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=256)
    labels = kmeans.fit_predict(data)

    # Identify transitions (where label changes between consecutive samples)
    transitions = []
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            transitions.append(frame_indices[i])

    transition_frames = np.array(transitions, dtype=int)

    # Log cluster statistics
    unique, counts = np.unique(labels, return_counts=True)
    largest_pct = counts.max() / len(labels) * 100
    logger.info(
        f"Pre-scan: {len(unique)} unique clusters, "
        f"largest cluster = {largest_pct:.1f}% of frames"
    )
    logger.info(f"Pre-scan: {len(transition_frames)} transition frames: {transition_frames.tolist()}")

    return PrescanResult(
        frame_indices=frame_indices,
        cluster_labels=labels,
        transition_frames=transition_frames,
    )


def chunk_video_frames_adaptive(
    fixed_chunks: list[tuple[int, int, str]],
    prescan: PrescanResult,
    fps: float,
    search_window_seconds: float = 10.0,
    min_chunk_seconds: float = 15.0,
    max_chunk_seconds: float = 90.0,
) -> list[tuple[int, int, str]]:
    """
    Adjust fixed chunk boundaries to align with cluster transitions from a pre-scan.

    For each tracker-chunk boundary (skip chunk 0), searches within ±search_window
    for the nearest cluster transition frame. Validates that adjusted chunks stay
    within min/max duration constraints. Falls back to original boundary if
    constraints are violated.

    Args:
        fixed_chunks: Output from chunk_video_frames_dual().
        prescan: PrescanResult from prescan_occlusion_periods().
        fps: Video frame rate.
        search_window_seconds: Search radius (seconds) around each boundary.
        min_chunk_seconds: Minimum allowed chunk duration.
        max_chunk_seconds: Maximum allowed chunk duration.

    Returns:
        Adjusted chunks in the same format as chunk_video_frames_dual().
    """
    if len(fixed_chunks) <= 1:
        return fixed_chunks

    search_window_frames = int(search_window_seconds * fps)
    min_frames = int(min_chunk_seconds * fps)
    max_frames = int(max_chunk_seconds * fps)
    transitions = prescan.transition_frames

    # Work with mutable list of boundaries
    # Boundaries are the start frames of chunks 1..N (i.e. the end of the previous chunk)
    boundaries = [c[0] for c in fixed_chunks]  # start of each chunk
    total_end = fixed_chunks[-1][1]  # final frame

    adjusted_boundaries = list(boundaries)

    # Only adjust tracker chunk boundaries (indices 1+)
    for i in range(1, len(adjusted_boundaries)):
        original = adjusted_boundaries[i]

        # Find nearest transition within search window
        candidates = transitions[
            (transitions >= original - search_window_frames)
            & (transitions <= original + search_window_frames)
        ]

        if len(candidates) == 0:
            logger.debug(
                f"Boundary {i}: frame {original} — no transitions within "
                f"±{search_window_seconds}s, keeping original"
            )
            continue

        nearest = candidates[np.argmin(np.abs(candidates - original))]
        shift = int(nearest) - original

        # Validate: check chunk sizes with this adjustment
        prev_start = adjusted_boundaries[i - 1]
        next_end = adjusted_boundaries[i + 1] if i + 1 < len(adjusted_boundaries) else total_end
        prev_chunk_len = int(nearest) - prev_start
        next_chunk_len = next_end - int(nearest)

        if prev_chunk_len < min_frames or prev_chunk_len > max_frames:
            logger.debug(
                f"Boundary {i}: frame {original} → {nearest} rejected "
                f"(prev chunk would be {prev_chunk_len / fps:.1f}s, "
                f"limits: {min_chunk_seconds}-{max_chunk_seconds}s)"
            )
            continue

        if next_chunk_len < min_frames or next_chunk_len > max_frames:
            logger.debug(
                f"Boundary {i}: frame {original} → {nearest} rejected "
                f"(next chunk would be {next_chunk_len / fps:.1f}s, "
                f"limits: {min_chunk_seconds}-{max_chunk_seconds}s)"
            )
            continue

        adjusted_boundaries[i] = int(nearest)
        logger.info(
            f"Boundary {i}: frame {original} → {nearest} "
            f"(shifted by {shift} frames, nearest transition)"
        )

    # Rebuild chunks from adjusted boundaries
    adjusted_chunks = []
    for i in range(len(adjusted_boundaries)):
        start = adjusted_boundaries[i]
        end = adjusted_boundaries[i + 1] if i + 1 < len(adjusted_boundaries) else total_end
        model_type = fixed_chunks[i][2]  # preserve original model type
        adjusted_chunks.append((start, end, model_type))

    # Absorb small trailing chunks (<10% of tracker chunk size) into last chunk
    if len(adjusted_chunks) > 1:
        last_start, last_end, last_type = adjusted_chunks[-1]
        last_len = last_end - last_start
        prev_start, prev_end, prev_type = adjusted_chunks[-2]
        prev_len = prev_end - prev_start
        # Use the tracker chunk seconds from the original fixed chunks as reference
        ref_tracker_frames = fixed_chunks[1][1] - fixed_chunks[1][0] if len(fixed_chunks) > 1 else last_len
        if last_len < ref_tracker_frames * 0.1:
            adjusted_chunks[-2] = (prev_start, last_end, prev_type)
            adjusted_chunks.pop()
            logger.info(
                f"Absorbed small trailing chunk ({last_len} frames) into previous chunk"
            )

    return adjusted_chunks


def chunk_video_frames_dual(
    total_frames: int,
    fps: float,
    video_model_seconds: int,
    tracker_seconds: int,
) -> list[tuple[int, int, str]]:
    """
    Split video into chunks with different durations for each model type.

    The first chunk uses Sam3VideoModel (text-prompted, shorter duration) and
    all subsequent chunks use Sam3TrackerVideoModel (point-prompted, longer).

    Returns list of (start_idx, end_idx, model_type) tuples where
    model_type is "video" (Sam3VideoModel) or "tracker" (Sam3TrackerVideoModel).
    """
    video_model_frames = int(fps * video_model_seconds)
    tracker_frames = int(fps * tracker_seconds)

    chunks = []
    # Chunk 0: video model (short)
    end = min(video_model_frames, total_frames)
    chunks.append((0, end, "video"))

    # Remaining chunks: tracker (longer)
    # Absorb small trailing remainders (< 10% of chunk size) into the last chunk
    start = end
    while start < total_frames:
        end = min(start + tracker_frames, total_frames)
        remaining_after = total_frames - end
        if 0 < remaining_after < tracker_frames * 0.1:
            end = total_frames  # absorb small tail
        chunks.append((start, end, "tracker"))
        start = end

    return chunks


def get_all_objects_from_results(results: dict) -> tuple[list, list, list]:
    """
    Extract masks, boxes, and object_ids from a single frame's output dict.

    Handles both torch tensors and numpy arrays.

    Returns:
        (masks_list, boxes_list, object_ids_list) where each is a list of
        per-object arrays.
    """
    masks = results.get("masks")
    boxes = results.get("boxes")
    object_ids = results.get("object_ids")

    if masks is None or object_ids is None:
        return [], [], []

    masks_np = to_numpy(masks)
    boxes_np = to_numpy(boxes) if boxes is not None else None
    ids_np = to_numpy(object_ids)

    masks_list = [masks_np[i] for i in range(len(ids_np))]
    boxes_list = (
        [boxes_np[i] for i in range(len(ids_np))] if boxes_np is not None else []
    )
    object_ids_list = ids_np.tolist()

    return masks_list, boxes_list, object_ids_list


def find_frame_with_enough_objects(
    outputs_per_frame: dict,
    min_objects: int = 3,
    max_lookback: int = 10,
) -> tuple[int | None, list, list, list]:
    """
    Search backwards through frame outputs to find a frame with enough objects.

    Args:
        outputs_per_frame: Dict mapping frame_idx -> processed output dict.
        min_objects: Minimum number of objects required.
        max_lookback: Maximum number of frames to search backwards from the end.

    Returns:
        (frame_idx, masks_list, boxes_list, object_ids_list) or
        (None, [], [], []) if no suitable frame found.
    """
    sorted_frames = sorted(outputs_per_frame.keys(), reverse=True)

    for frame_idx in sorted_frames[:max_lookback]:
        masks_list, boxes_list, object_ids_list = get_all_objects_from_results(
            outputs_per_frame[frame_idx]
        )
        if len(object_ids_list) >= min_objects:
            logger.debug(
                f"Found {len(object_ids_list)} objects at frame {frame_idx}"
            )
            return frame_idx, masks_list, boxes_list, object_ids_list

    logger.warning(
        f"No frame with >= {min_objects} objects found in last {max_lookback} frames"
    )
    return None, [], [], []


def extract_equidistant_points_from_mask(
    mask: np.ndarray, num_points: int = 3
) -> list[list[int]] | None:
    """
    Extract equidistant points along the mask's major axis (x-axis).

    Samples points deterministically by sorting pixels by x-coordinate
    and selecting evenly-spaced points using linspace. Provides better
    spatial coverage than random sampling and naturally avoids border pixels.

    Originally from commit d5b4cda (the "magic run" with perfect tracking).

    Args:
        mask: Binary mask (H, W) with 1s indicating the object.
        num_points: Number of equidistant points to extract.

    Returns:
        List of [x, y] points, or None if mask is empty.
    """
    y_coords, x_coords = np.where(mask > 0)
    if len(y_coords) == 0:
        return None

    center_x = int(np.mean(x_coords))
    center_y = int(np.mean(y_coords))

    if num_points == 1:
        return [[center_x, center_y]]

    sorted_indices = np.argsort(x_coords)
    n_pixels = len(sorted_indices)

    indices = np.linspace(0, n_pixels - 1, num_points, dtype=int)

    points = []
    for idx in indices:
        sorted_idx = sorted_indices[idx]
        x = int(x_coords[sorted_idx])
        y = int(y_coords[sorted_idx])
        points.append([x, y])

    return points


def extract_equidistant_points_from_masks(
    masks: np.ndarray, num_points: int = 3
) -> np.ndarray:
    """
    Batch wrapper for extract_equidistant_points_from_mask().

    Args:
        masks: np.array with shape (N, H, W), binary masks.
        num_points: Number of points to extract per mask.

    Returns:
        points: np.array with shape (N, num_points, 2) in (x, y) format.
    """
    n = masks.shape[0]
    points = []
    for i in range(n):
        pts = extract_equidistant_points_from_mask(masks[i], num_points)
        if pts is None:
            points.append(np.zeros((num_points, 2)))
        else:
            points.append(np.array(pts))
    return np.array(points, dtype=np.float32)


def sample_points_from_masks(masks: np.ndarray, num_points: int = 3) -> np.ndarray:
    """
    Sample random points from mask-positive pixels and return absolute coordinates.

    Adapted from: IDEA-Research/Grounded-SAM-2
    Source: https://github.com/IDEA-Research/Grounded-SAM-2/blob/main/utils/track_utils.py
    Also available locally: Grounded-SAM-2-fork/utils/track_utils.py

    Args:
        masks: np.array with shape (N, H, W), binary masks.
        num_points: Number of points to sample per mask.

    Returns:
        points: np.array with shape (N, num_points, 2) in (x, y) format.
    """
    n, h, w = masks.shape
    points = []
    for i in range(n):
        indices = np.argwhere(masks[i] == 1)
        indices = indices[:, ::-1]  # (y, x) to (x, y)
        if len(indices) == 0:
            points.append(np.zeros((num_points, 2)))
            continue
        if len(indices) < num_points:
            sampled_indices = np.random.choice(len(indices), num_points, replace=True)
        else:
            sampled_indices = np.random.choice(
                len(indices), num_points, replace=False
            )
        sampled_points = indices[sampled_indices]
        points.append(sampled_points)
    points = np.array(points, dtype=np.float32)
    return points


def free_gpu_memory(log_stats: bool = False):
    """Free GPU memory between chunks via garbage collection and cache clearing."""
    if log_stats and torch.cuda.is_available():
        before = torch.cuda.memory_allocated() / 1024**2
        logger.debug(f"GPU memory before cleanup: {before:.1f} MB")

    gc.collect()
    gc.collect()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    if log_stats and torch.cuda.is_available():
        after = torch.cuda.memory_allocated() / 1024**2
        logger.debug(f"GPU memory after cleanup: {after:.1f} MB")


# ---------------------------------------------------------------------------
# Data processing and annotation utilities
# ---------------------------------------------------------------------------


def to_numpy(x):
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.array(x)


def process_tracking_outputs(outputs_per_frame):
    index_tuples = []
    bboxes = []
    counts_list = []
    sizes = []
    scores_list = []
    tracker_scores_list = []
    chunk_idx_list = []
    model_type_list = []
    is_chunk_start_list = []

    for frame_idx, proc in outputs_per_frame.items():
        object_ids = to_numpy(proc["object_ids"])
        boxes = to_numpy(proc["boxes"])
        masks = proc["masks"]  # keep lazy until conversion per-item
        scores = to_numpy(proc.get("scores", np.zeros(len(object_ids))))
        tracker_scores_dict = proc.get("obj_id_to_tracker_score") or {}

        # Chunk metadata (stamped by chunked pipeline, None for single-pass)
        chunk_idx = proc.get("_chunk_idx")
        model_type = proc.get("_model_type")
        is_chunk_start = proc.get("_is_chunk_start")

        for i, oid in enumerate(object_ids):
            # bbox -> list (x1,y1,x2,y2)
            bbox = boxes[i].tolist()

            # mask -> RLE
            mask_item = masks[i]
            # if mask already RLE-like (dict with 'counts'/'size'), use it
            if (
                isinstance(mask_item, dict)
                and "counts" in mask_item
                and "size" in mask_item
            ):
                rle = mask_item
                counts = rle["counts"]
                try:
                    counts = counts.decode("utf-8")
                except Exception:
                    pass
                size = rle["size"]
            else:
                m = to_numpy(mask_item).astype(np.uint8)
                # squeeze singleton channel dimension if present
                if m.ndim == 3 and m.shape[0] in (1,):
                    m = m.squeeze(0)
                # ensure Fortran order required by pycocotools
                rle = mask_util.encode(np.asfortranarray(m))
                counts = rle["counts"]
                try:
                    counts = counts.decode("ascii")
                except Exception:
                    pass
                size = rle["size"]

            score = float(scores[i])
            tracker_score = (
                float(tracker_scores_dict[int(oid)])
                if tracker_scores_dict and int(oid) in tracker_scores_dict
                else None
            )
            index_tuples.append((int(frame_idx), int(oid)))
            bboxes.append(bbox)
            counts_list.append(counts)
            sizes.append(size)
            scores_list.append(score)
            tracker_scores_list.append(tracker_score)
            chunk_idx_list.append(chunk_idx)
            model_type_list.append(model_type)
            is_chunk_start_list.append(is_chunk_start)

    mi = pd.MultiIndex.from_tuples(index_tuples, names=["frame_idx", "object_id"])
    df_results = pd.DataFrame(
        {
            "bbox": bboxes,
            "counts": counts_list,
            "size": sizes,
            "scores": scores_list,
            "tracker_score": tracker_scores_list,
            "chunk_idx": chunk_idx_list,
            "model_type": model_type_list,
            "is_chunk_start": is_chunk_start_list,
        },
        index=mi,
    )
    return df_results


def create_annotation_callback(outputs_per_frame: dict):
    """
    Creates a callback function for sv.process_video that annotates frames
    using pre-computed SAM3 tracking outputs.
    """
    mask_annotator = sv.MaskAnnotator()
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator()

    def callback(frame: np.ndarray, frame_idx: int) -> np.ndarray:
        # Get outputs for this frame (may be missing for some frames)
        if frame_idx not in outputs_per_frame:
            return frame  # return original frame if no detections

        frame_out = outputs_per_frame[frame_idx]

        # Convert to tensors (handles both torch tensor and numpy array inputs)
        masks_raw = frame_out["masks"]
        boxes_raw = frame_out["boxes"]
        ids_raw = frame_out["object_ids"]
        scores_raw = frame_out["scores"]

        if isinstance(masks_raw, np.ndarray):
            masks_t = torch.from_numpy(masks_raw)
        else:
            masks_t = masks_raw.detach().cpu()

        if isinstance(boxes_raw, np.ndarray):
            boxes_t = torch.from_numpy(boxes_raw)
        else:
            boxes_t = boxes_raw.detach().cpu()

        if isinstance(ids_raw, np.ndarray):
            ids_t = torch.from_numpy(ids_raw)
        else:
            ids_t = ids_raw.detach().cpu()

        if isinstance(scores_raw, np.ndarray):
            scores_t = torch.from_numpy(scores_raw)
        else:
            scores_t = scores_raw.detach().cpu()

        # Prepare masks: ensure shape (N, 1, H, W)
        if masks_t.ndim == 3:  # (N, H, W)
            masks_t = masks_t.unsqueeze(1)  # -> (N, 1, H, W)
        masks_t = masks_t.to(torch.uint8)

        # Build transformers-style results
        transformers_res = {
            "boxes": boxes_t,
            "masks": masks_t,
            "labels": ids_t,
            "scores": scores_t,
        }

        # Create id2label mapping
        id2label = {
            int(i): f"id:{int(i)}" for i in to_numpy(ids_t)
        }

        # Build detections
        detections = sv.Detections.from_transformers(
            transformers_results=transformers_res, id2label=id2label
        )

        # Create labels with ID and confidence
        labels = [
            f"#{int(obj_id)} {confidence:.2f}"
            for obj_id, confidence in zip(
                to_numpy(ids_t), detections.confidence
            )
        ]

        # Apply annotations
        annotated = mask_annotator.annotate(scene=frame.copy(), detections=detections)
        annotated = box_annotator.annotate(scene=annotated, detections=detections)
        annotated = label_annotator.annotate(
            scene=annotated, detections=detections, labels=labels
        )

        return annotated

    return callback


def annotate_video_with_sam3_outputs(
    source_path: str, target_path: str, outputs_per_frame: dict
):
    """
    Process entire video with SAM3 tracking outputs and save annotated version.
    """
    callback = create_annotation_callback(outputs_per_frame)
    sv.process_video(
        source_path=source_path,
        target_path=target_path,
        callback=callback,
        show_progress=True,
    )
