"""
Utility functions for processing SAM3 tracking outputs.
"""

import gc
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import psutil
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
    log_dir: Path = Path("sandbox/logs/sam3-hf"),
    job_type: str = "sam3_hf",
    debug: bool = False,
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


def chunk_video_frames_adaptive(
    total_frames: int,
    fps: float,
    video_model_seconds: int,
    tracker_seconds: int,
    separation_windows: list[tuple[int, int]] | None = None,
    per_frame_metrics: list[dict] | None = None,
    occlusion_periods: list[tuple[int, int]] | None = None,
    transition_frames: np.ndarray | None = None,
    search_window_seconds: float = 10.0,
    min_chunk_seconds: float = 15.0,
    max_chunk_seconds: float = 90.0,
    margin_seconds: float = 3.0,
) -> list[tuple[int, int, str]]:
    """
    Split video into adaptive chunks, optionally refined by separation windows
    or occlusion avoidance.

    Generates initial fixed chunks (chunk 0 → video model, remainder → tracker),
    then for each tracker-chunk boundary refines in priority order:

      1. Separation-first: if ``separation_windows`` and ``per_frame_metrics``
         are provided, search ±search_window_seconds for frames inside a
         separation window and pick the highest separation_score.
      2. Occlusion avoidance: if ``occlusion_periods`` are provided, search for
         frames outside occlusion periods (+margin_seconds buffer) and prefer
         frames farthest from any occlusion.
      3. No-op: keep original boundary if neither applies or constraints are
         violated.

    Validates against min_chunk_seconds / max_chunk_seconds and absorbs trailing
    remainders < 10% of tracker chunk size.

    Args:
        total_frames: Total number of frames in the video.
        fps: Video frame rate.
        video_model_seconds: Duration of the first (text-prompted) chunk in seconds.
        tracker_seconds: Duration of subsequent (point-prompted) chunks in seconds.
        separation_windows: List of (start_frame, end_frame) high-separation windows.
            When provided together with per_frame_metrics, enables separation-first
            boundary refinement.
        per_frame_metrics: Per-frame metric dicts (must contain 'frame_idx' and
            'separation_score'). Required for separation-first refinement.
        occlusion_periods: List of (start_frame, end_frame) high-occlusion periods.
            Used as a fallback when separation-first is unavailable.
        transition_frames: Legacy parameter kept for backwards compatibility.
            Ignored when occlusion_periods or separation_windows are provided.
        search_window_seconds: Search radius (seconds) around each boundary.
        min_chunk_seconds: Minimum allowed chunk duration after adjustment.
        max_chunk_seconds: Maximum allowed chunk duration after adjustment.
        margin_seconds: Safety buffer (seconds) around occlusion period edges.

    Returns:
        List of (start_frame, end_frame, model_type) tuples where model_type is
        "video" (first chunk) or "tracker" (all others).
    """
    video_model_frames = int(fps * video_model_seconds)
    tracker_frames = int(fps * tracker_seconds)
    search_window_frames = int(search_window_seconds * fps)
    min_frames = int(min_chunk_seconds * fps)
    max_frames = int(max_chunk_seconds * fps)
    margin_frames = int(margin_seconds * fps)

    # Step 1: Generate initial fixed chunks (same logic as old chunk_video_frames_dual)
    fixed_chunks: list[tuple[int, int, str]] = []
    end = min(video_model_frames, total_frames)
    fixed_chunks.append((0, end, "video"))

    start = end
    while start < total_frames:
        end = min(start + tracker_frames, total_frames)
        remaining_after = total_frames - end
        if 0 < remaining_after < tracker_frames * 0.1:
            end = total_frames  # absorb small tail
        fixed_chunks.append((start, end, "tracker"))
        start = end

    if len(fixed_chunks) <= 1:
        return fixed_chunks

    # Step 2: Collect refinement data
    use_separation = bool(separation_windows and per_frame_metrics)
    use_occlusion = bool(occlusion_periods)

    score_lookup: dict[int, float] = {}
    if use_separation and per_frame_metrics:
        score_lookup = {
            m["frame_idx"]: m["separation_score"]
            for m in per_frame_metrics
            if m.get("separation_score", 0.0) > 0.0
        }

    boundaries = [c[0] for c in fixed_chunks]
    total_end = fixed_chunks[-1][1]
    adjusted_boundaries = list(boundaries)

    def _is_in_occlusion(frame: int) -> bool:
        """Check if frame falls within any occlusion period (with margin)."""
        if not occlusion_periods:
            return False
        for start_f, end_f in occlusion_periods:
            if (start_f - margin_frames) <= frame <= (end_f + margin_frames):
                return True
        return False

    def _occlusion_quality(frame: int) -> float:
        """
        Compute quality score for a candidate boundary (higher = better).

        Scoring rules:
        1. Distance to nearest occlusion (farther = better)
        2. Strong penalty if occlusion is ahead (within next margin_frames)
        3. Moderate penalty if occlusion is behind (within prev margin_frames)
        """
        if not occlusion_periods:
            return 1000.0  # Arbitrary high score if no occlusions

        min_dist = float("inf")
        occlusion_ahead = False
        occlusion_behind = False

        for start_f, end_f in occlusion_periods:
            # Check if occlusion is ahead (about to start)
            if start_f > frame and (start_f - frame) <= margin_frames:
                occlusion_ahead = True
            # Check if occlusion just ended
            if end_f < frame and (frame - end_f) <= margin_frames:
                occlusion_behind = True

            # Distance to nearest edge
            dist = min(abs(frame - start_f), abs(frame - end_f))
            min_dist = min(min_dist, dist)

        # Base score is distance
        score = float(min_dist)

        # Heavy penalty if occlusion is just ahead
        if occlusion_ahead:
            score *= 0.1  # 90% penalty

        # Moderate penalty if occlusion just ended
        if occlusion_behind:
            score *= 0.5  # 50% penalty

        return score

    # Step 3: Refine tracker-chunk boundaries (indices 1+)
    for i in range(1, len(adjusted_boundaries)):
        original = adjusted_boundaries[i]
        search_start = max(0, original - search_window_frames)
        search_end = min(total_end, original + search_window_frames)
        best_candidate = None

        if use_separation:
            # Separation-first: find highest-score frame inside a separation window
            sep_candidates: dict[int, float] = {}
            for win_start, win_end in separation_windows:  # type: ignore[union-attr]
                overlap_start = max(win_start, search_start)
                overlap_end = min(win_end, search_end)
                if overlap_start > overlap_end:
                    continue
                for frame, score in score_lookup.items():
                    if overlap_start <= frame <= overlap_end:
                        sep_candidates[frame] = score

            if sep_candidates:
                best_candidate = max(sep_candidates, key=lambda f: sep_candidates[f])
                shift = best_candidate - original
                prev_start = adjusted_boundaries[i - 1]
                next_end = (
                    adjusted_boundaries[i + 1]
                    if i + 1 < len(adjusted_boundaries)
                    else total_end
                )
                prev_len = best_candidate - prev_start
                next_len = next_end - best_candidate

                if prev_len < min_frames or prev_len > max_frames:
                    logger.debug(
                        f"Separation boundary {i}: frame {original} → {best_candidate} rejected "
                        f"(prev chunk {prev_len / fps:.1f}s outside "
                        f"[{min_chunk_seconds}-{max_chunk_seconds}]s)"
                    )
                    best_candidate = None
                elif next_len < min_frames or next_len > max_frames:
                    logger.debug(
                        f"Separation boundary {i}: frame {original} → {best_candidate} rejected "
                        f"(next chunk {next_len / fps:.1f}s outside "
                        f"[{min_chunk_seconds}-{max_chunk_seconds}]s)"
                    )
                    best_candidate = None
                else:
                    adjusted_boundaries[i] = best_candidate
                    logger.info(
                        f"Separation boundary {i}: frame {original} → {best_candidate} "
                        f"(shifted {shift:+d} frames, "
                        f"separation_score={sep_candidates[best_candidate]:.3f})"
                    )
                    continue  # Done with this boundary
            else:
                logger.debug(
                    f"Separation boundary {i}: frame {original} — no separation windows "
                    f"in ±{search_window_seconds}s, trying occlusion fallback"
                )

        if use_occlusion and best_candidate is None:
            # Occlusion avoidance fallback
            candidates_list = list(range(search_start, search_end + 1))
            safe_candidates = [f for f in candidates_list if not _is_in_occlusion(f)]

            if not safe_candidates:
                logger.warning(
                    f"Boundary {i}: frame {original} — all candidates within occlusion periods "
                    f"(±{search_window_seconds}s window), keeping original (RISKY)"
                )
                continue

            best_candidate = max(safe_candidates, key=_occlusion_quality)
            shift = int(best_candidate) - original
            prev_start = adjusted_boundaries[i - 1]
            next_end = (
                adjusted_boundaries[i + 1]
                if i + 1 < len(adjusted_boundaries)
                else total_end
            )
            prev_len = int(best_candidate) - prev_start
            next_len = next_end - int(best_candidate)

            if prev_len < min_frames or prev_len > max_frames:
                logger.debug(
                    f"Boundary {i}: frame {original} → {best_candidate} rejected "
                    f"(prev chunk would be {prev_len / fps:.1f}s, "
                    f"limits: {min_chunk_seconds}-{max_chunk_seconds}s)"
                )
                continue

            if next_len < min_frames or next_len > max_frames:
                logger.debug(
                    f"Boundary {i}: frame {original} → {best_candidate} rejected "
                    f"(next chunk would be {next_len / fps:.1f}s, "
                    f"limits: {min_chunk_seconds}-{max_chunk_seconds}s)"
                )
                continue

            adjusted_boundaries[i] = int(best_candidate)
            quality = _occlusion_quality(int(best_candidate))
            logger.info(
                f"Boundary {i}: frame {original} → {best_candidate} "
                f"(shifted by {shift:+d} frames, quality score: {quality:.1f})"
            )

    # Step 4: Rebuild chunks from adjusted boundaries
    adjusted_chunks = []
    for i in range(len(adjusted_boundaries)):
        s = adjusted_boundaries[i]
        e = (
            adjusted_boundaries[i + 1]
            if i + 1 < len(adjusted_boundaries)
            else total_end
        )
        model_type = fixed_chunks[i][2]
        adjusted_chunks.append((s, e, model_type))

    # Step 5: Absorb small trailing chunks (<10% of tracker chunk size)
    if len(adjusted_chunks) > 1:
        last_start, last_end, _ = adjusted_chunks[-1]
        last_len = last_end - last_start
        prev_start, _, prev_type = adjusted_chunks[-2]
        should_absorb = False

        if last_len < tracker_frames * 0.1:
            should_absorb = True
            logger.info(
                f"Absorbing small trailing chunk ({last_len} frames, "
                f"{last_len / fps:.1f}s) into previous chunk"
            )
        elif occlusion_periods and _is_in_occlusion(last_start):
            combined_len = last_end - prev_start
            if combined_len <= max_frames:
                should_absorb = True
                logger.info(
                    f"Absorbing last chunk (boundary at {last_start} too close to occlusion) "
                    f"into previous chunk (combined: {combined_len / fps:.1f}s)"
                )

        if should_absorb:
            adjusted_chunks[-2] = (prev_start, last_end, prev_type)
            adjusted_chunks.pop()

    return adjusted_chunks


def build_manual_chunks(frame_pairs: list[list[int]]) -> list[tuple[int, int, str]]:
    """Build chunk list from user-supplied (start_frame, end_frame) pairs.
    First chunk → "video" model; all subsequent → "tracker" model.
    """
    chunks = []
    for i, (start, end) in enumerate(frame_pairs):
        model_type = "video" if i == 0 else "tracker"
        chunks.append((int(start), int(end), model_type))
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
            logger.debug(f"Found {len(object_ids_list)} objects at frame {frame_idx}")
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
            sampled_indices = np.random.choice(len(indices), num_points, replace=False)
        sampled_points = indices[sampled_indices]
        points.append(sampled_points)
    points = np.array(points, dtype=np.float32)
    return points


def free_gpu_memory(log_stats: bool = False):
    """Free GPU memory between chunks via garbage collection and cache clearing."""
    if log_stats and torch.cuda.is_available():
        before_alloc = torch.cuda.memory_allocated() / 1024**2
        before_res = torch.cuda.memory_reserved() / 1024**2
        logger.info(
            f"GPU memory before cleanup: {before_alloc:.1f} MB allocated, {before_res:.1f} MB reserved"
        )

    gc.collect()
    gc.collect()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()  # second pass clears anything freed by first-pass destructors
        torch.cuda.ipc_collect()

    if log_stats and torch.cuda.is_available():
        after_alloc = torch.cuda.memory_allocated() / 1024**2
        after_res = torch.cuda.memory_reserved() / 1024**2
        logger.info(
            f"GPU memory after cleanup: {after_alloc:.1f} MB allocated, {after_res:.1f} MB reserved"
        )


def free_system_memory(label: str = ""):
    """Run gc.collect() and log system RAM before/after."""
    prefix = f"{label}: " if label else ""

    vm = psutil.virtual_memory()
    before_gb = vm.used / 1024**3
    available_gb = vm.available / 1024**3
    logger.info(f"RAM before gc ({prefix}{before_gb:.1f} GB used, {available_gb:.1f} GB available)")

    gc.collect()

    vm = psutil.virtual_memory()
    after_gb = vm.used / 1024**3
    available_gb = vm.available / 1024**3
    freed_gb = before_gb - after_gb
    logger.info(
        f"RAM after gc ({prefix}{after_gb:.1f} GB used, {available_gb:.1f} GB available, freed {freed_gb:.1f} GB)"
    )


def get_video_metadata(video_path: str | Path) -> tuple[float, int]:
    """Return (fps, total_frames) for a video without loading all frames into RAM."""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, total_frames


def load_video_frames_range(video_path: str | Path, start_frame: int, end_frame: int) -> list:
    """Load frames [start_frame, end_frame) from a video file as a list of RGB numpy arrays."""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    for _ in range(end_frame - start_frame):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


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
        id2label = {int(i): f"id:{int(i)}" for i in to_numpy(ids_t)}

        # Build detections
        detections = sv.Detections.from_transformers(
            transformers_results=transformers_res, id2label=id2label
        )

        # Create labels with ID and confidence
        labels = [
            f"#{int(obj_id)} {confidence:.2f}"
            for obj_id, confidence in zip(to_numpy(ids_t), detections.confidence)
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


# ---------------------------------------------------------------------------
# Grounding / Tracker Utilities
# ---------------------------------------------------------------------------


def sanitize_filename(name: str) -> str:
    """Sanitize a stem string for use as a directory name."""
    sanitized = re.sub(r"[^\w\-]", "_", name)
    sanitized = re.sub(r"_+", "_", sanitized)
    return sanitized.strip("_") or "video"


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


def find_best_grounding_frame(
    grounding_outputs: dict,
    min_objects: int = 3,
    method: str = "combined",
) -> tuple[int | None, list, list, list]:
    """
    Select best frame from grounding outputs by detection quality.

    Filters frames with >= min_objects, then ranks by:
      "min_occlusion"  — lowest max pairwise IoU
      "best_scores"    — highest mean detection score
      "combined"       — low-occlusion frames first (max_iou < 0.10),
                         then by descending mean score within that tier

    Returns:
        (frame_idx, masks_list, boxes_list, object_ids_list) or
        (None, [], [], []) if no frame qualifies.
    """
    candidates = []
    for frame_idx, results in grounding_outputs.items():
        masks_list, boxes_list, object_ids_list = get_all_objects_from_results(results)
        if len(object_ids_list) < min_objects:
            continue
        masks_array = np.stack(
            [m.squeeze(0) if m.ndim == 3 and m.shape[0] == 1 else m for m in masks_list]
        ).astype(bool)
        max_iou = compute_max_pairwise_iou(masks_array)

        scores = results.get("scores")
        if scores is not None:
            scores_arr = scores if isinstance(scores, np.ndarray) else np.array(scores)
            mean_score = float(scores_arr.mean()) if scores_arr.size > 0 else 0.0
        else:
            tracker_scores = results.get("obj_id_to_tracker_score", {})
            if tracker_scores:
                mean_score = float(np.mean(list(tracker_scores.values())))
            else:
                mean_score = 0.0

        candidates.append(
            (frame_idx, masks_list, boxes_list, object_ids_list, max_iou, mean_score)
        )

    if not candidates:
        return None, [], [], []

    if method == "min_occlusion":
        candidates.sort(key=lambda x: x[4])
    elif method == "best_scores":
        candidates.sort(key=lambda x: -x[5])
    else:  # "combined"
        candidates.sort(key=lambda x: (x[4] >= 0.10, -x[5]))

    best = candidates[0]
    return best[0], best[1], best[2], best[3]


def match_grounding_ids_to_previous(
    grounding_masks: list,
    grounding_ids: list,
    prev_masks: list,
    prev_ids: list,
    iou_threshold: float = 0.10,
) -> dict[int, int]:
    """
    Greedy IoU-based assignment of grounding IDs to previous-chunk IDs.

    Builds P×Q IoU matrix, iteratively picks highest-IoU pair, removes both
    from pool, stops when below iou_threshold. Unmatched grounding IDs keep
    their original value if it does not collide with an already-mapped ID;
    otherwise they are assigned a fresh ID to avoid silent key collisions in
    grounding_prompt_points.
    """
    if not grounding_masks or not prev_masks:
        return {int(pid): int(pid) for pid in grounding_ids}

    def _to_bool(m):
        arr = m.squeeze(0) if m.ndim == 3 and m.shape[0] == 1 else m
        return arr.astype(bool)

    gs_masks = [_to_bool(m) for m in grounding_masks]
    pv_masks = [_to_bool(m) for m in prev_masks]

    P, Q = len(gs_masks), len(pv_masks)
    iou_matrix = np.zeros((P, Q), dtype=np.float32)
    for i in range(P):
        for j in range(Q):
            inter = float((gs_masks[i] & pv_masks[j]).sum())
            union = float((gs_masks[i] | pv_masks[j]).sum())
            iou_matrix[i, j] = inter / union if union > 0 else 0.0

    id_map: dict[int, int] = {}
    assigned_gs = set()
    assigned_pv = set()

    flat_indices = np.argsort(iou_matrix.ravel())[::-1]
    for flat_idx in flat_indices:
        i, j = divmod(int(flat_idx), Q)
        if iou_matrix[i, j] < iou_threshold:
            break
        if i in assigned_gs or j in assigned_pv:
            continue
        id_map[int(grounding_ids[i])] = int(prev_ids[j])
        assigned_gs.add(i)
        assigned_pv.add(j)

    # Unmatched grounding IDs: pass through with original value, but remap to a
    # fresh ID if that value is already claimed by a matched assignment to avoid
    # silent dict-key collisions in grounding_prompt_points.
    claimed = set(id_map.values())
    next_fresh = max(claimed | {int(pid) for pid in grounding_ids}, default=-1) + 1
    for i, pid in enumerate(grounding_ids):
        if int(pid) not in id_map:
            if int(pid) not in claimed:
                id_map[int(pid)] = int(pid)
                claimed.add(int(pid))
            else:
                id_map[int(pid)] = next_fresh
                claimed.add(next_fresh)
                next_fresh += 1

    return id_map


def reseed_tracker_memory(
    model, inference_session, frame_idx: int, masks_np: np.ndarray, device
):
    """
    Inject current binary masks as a fresh conditioning frame.

    Removes frame_idx from frames_tracked so forward() treats it
    as is_init_cond_frame=True, re-encoding the memory bank.

    Args:
        model: Sam3TrackerVideoModel instance.
        inference_session: Active inference session.
        frame_idx: Local frame index to reseed.
        masks_np: (N, H, W) bool/float numpy array of current masks.
        device: Torch device.
    """
    masks_tensor = torch.from_numpy(masks_np.astype(np.float32)).to(device)

    obj_ids = list(inference_session.obj_ids)[: len(masks_np)]
    for i, obj_id in enumerate(obj_ids):
        obj_idx = inference_session.obj_id_to_idx(obj_id)
        mask_t = masks_tensor[i].unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        inference_session.add_mask_inputs(obj_idx, frame_idx, mask_t)
        # Clear tracked status so forward treats this as an init-cond frame
        inference_session.frames_tracked_per_obj[obj_idx].pop(frame_idx, None)
        if obj_id not in inference_session.obj_with_new_inputs:
            inference_session.obj_with_new_inputs.append(obj_id)

    # Re-run forward for this frame — encodes fresh memory anchor
    model(inference_session, frame_idx=frame_idx, run_mem_encoder=True)


def run_grounding(
    chunk_frames: list,
    global_start_idx: int,
    grounding_frames: int,
    process_video_chunk_fn: Callable,
    cfg,
    device,
) -> dict:
    """
    Run Sam3VideoModel on chunk_frames[:grounding_frames] for fresh detections.

    Clamps grounding_frames to len(chunk_frames). Frees GPU memory before
    returning.

    Args:
        chunk_frames: List of RGB frames for the current chunk.
        global_start_idx: Global frame index of the first frame in chunk_frames.
        grounding_frames: Number of frames to run grounding on.
        process_video_chunk_fn: Callable matching the signature of
            _process_video_chunk(chunk_frames, start_idx, cfg, device).
        cfg: OmegaConf config.
        device: Torch device.

    Returns:
        Dict mapping global frame indices to processed output dicts.
    """
    n = min(grounding_frames, len(chunk_frames))
    outputs = process_video_chunk_fn(chunk_frames[:n], global_start_idx, cfg, device)
    free_gpu_memory()
    return outputs
