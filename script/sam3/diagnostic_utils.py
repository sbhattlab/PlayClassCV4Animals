"""
Shared utilities for SAM3 diagnostic visualization scripts.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pycocotools.mask as mask_util


def load_run_results(run_dir: str | Path) -> dict:
    """
    Load all results from a SAM3 run directory.

    Args:
        run_dir: Path to the timestamped run directory.

    Returns:
        Dict with keys: chunk_info, tracking_df, per_frame_df (any may be None).
    """
    run_dir = Path(run_dir)

    chunk_info = None
    chunk_info_path = run_dir / "chunk_info.json"
    if chunk_info_path.exists():
        with open(chunk_info_path) as f:
            chunk_info = json.load(f)

    tracking_df = None
    tracking_path = run_dir / "tracking_outputs.parquet"
    if tracking_path.exists():
        tracking_df = pd.read_parquet(tracking_path)

    per_frame_df = None
    per_frame_path = run_dir / "metrics" / "per_frame_metrics.parquet"
    if per_frame_path.exists():
        per_frame_df = pd.read_parquet(per_frame_path)

    return {
        "chunk_info": chunk_info,
        "tracking_df": tracking_df,
        "per_frame_df": per_frame_df,
    }


def decode_rle_mask(counts, size) -> np.ndarray:
    """
    Decode an RLE-encoded mask to a binary numpy array.

    Args:
        counts: RLE counts string.
        size: [height, width] of the mask.

    Returns:
        Binary mask array of shape (H, W).
    """
    if isinstance(counts, str):
        counts = counts.encode("utf-8")
    rle = {"counts": counts, "size": size}
    return mask_util.decode(rle).astype(np.uint8)


def read_video_frame(video_path: str | Path, frame_idx: int) -> np.ndarray | None:
    """
    Extract a single frame from a video file.

    Args:
        video_path: Path to the video file.
        frame_idx: 0-based frame index to extract.

    Returns:
        BGR frame as numpy array, or None if extraction fails.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    return frame


def get_chunk_boundaries(chunk_info: dict) -> list[dict]:
    """
    Parse chunk_info.json and return boundary transition metadata.

    Args:
        chunk_info: Parsed chunk_info.json dict with 'chunks' key.

    Returns:
        List of dicts, one per boundary transition, with keys:
        - chunk_idx: Index of the target (tracker) chunk
        - source_frame_idx: Last frame of previous chunk used for point extraction
        - boundary_frame: First frame of the new chunk
        - prompt_points: Dict of object_id -> [[x,y], ...] point prompts
        - prev_chunk_end: Last frame index of previous chunk
    """
    chunks = chunk_info.get("chunks", [])
    boundaries = []

    for i, chunk in enumerate(chunks):
        if i == 0:
            continue  # first chunk has no boundary transition

        frame_range = chunk.get("frame_range", [0, 0])
        prev_range = chunks[i - 1].get("frame_range", [0, 0])

        prompt_points = chunk.get("prompt_points")
        # Convert string keys to int
        if prompt_points:
            prompt_points = {int(k): v for k, v in prompt_points.items()}

        boundaries.append({
            "chunk_idx": chunk.get("chunk_idx", i),
            "source_frame_idx": chunk.get("source_frame_idx"),
            "boundary_frame": frame_range[0],
            "prompt_points": prompt_points,
            "prev_chunk_end": prev_range[1] - 1,
            "model_type": chunk.get("model_type"),
        })

    return boundaries


def create_output_directory(run_dir: str | Path, subdir: str = "diagnostic_visualizations") -> Path:
    """
    Create a diagnostic output subdirectory within the run directory.

    Args:
        run_dir: Path to the run directory.
        subdir: Name of the subdirectory to create.

    Returns:
        Path to the created directory.
    """
    out_dir = Path(run_dir) / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def get_masks_for_frame(tracking_df: pd.DataFrame, frame_idx: int) -> dict:
    """
    Extract and decode all masks for a given frame from tracking_outputs.

    Args:
        tracking_df: MultiIndex DataFrame (frame_idx, object_id).
        frame_idx: Frame index to extract masks for.

    Returns:
        Dict mapping object_id -> decoded binary mask (H, W).
    """
    if frame_idx not in tracking_df.index.get_level_values("frame_idx"):
        return {}

    frame_data = tracking_df.xs(frame_idx, level="frame_idx")
    masks = {}
    for obj_id in frame_data.index:
        row = frame_data.loc[obj_id]
        mask = decode_rle_mask(row["counts"], row["size"])
        masks[int(obj_id)] = mask

    return masks


def get_video_fps(video_path: str | Path) -> float:
    """Get FPS from a video file."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


def frame_to_mmss(frame_idx: int, fps: float) -> str:
    """Convert frame index to MM:SS string."""
    seconds = frame_idx / fps
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"
