"""
Utility functions for processing SAM3 tracking outputs.
"""

import gc
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pycocotools.mask as mask_util
import supervision as sv
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf


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
