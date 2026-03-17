import os
from pathlib import Path

import pandas as pd
from loguru import logger

from src._config import DEFAULT_FPS


def get_video_fps(tracking_dir):
    """Read video FPS from yolo_scan_summary.parquet in the tracking dir."""
    tracking_dir = Path(tracking_dir)
    summary_path = tracking_dir / "metrics" / "yolo_scan_summary.parquet"
    if summary_path.exists():
        summary = pd.read_parquet(summary_path)
        if "fps" in summary.columns and len(summary) > 0:
            return float(summary["fps"].iloc[0])
    logger.warning(
        f"Could not read FPS from {summary_path}, falling back to {DEFAULT_FPS}"
    )
    return DEFAULT_FPS


def fmt_time(frame_idx, fps=25.0):
    """Frame index → MM:SS.f timestamp string."""
    t = frame_idx / fps
    m, s = divmod(t, 60)
    return f"{int(m):02d}:{s:05.2f}"


def resolve_video_path(
    video_id: str, tracking_dir: Path, video_dirs: list[Path]
) -> Path | None:
    """Find the video file for a given video_id.

    Searches tracking subdirs under *tracking_dir* for directories whose name
    starts with *video_id*, then looks for ``{subdir_name}.mp4`` in each of
    *video_dirs* (flat and one level deep).
    """
    for root, _dirs, files in os.walk(tracking_dir, followlinks=True):
        if "tracking_outputs.parquet" in files:
            name = Path(root).name
            if name.startswith(video_id):
                mp4_name = f"{name}.mp4"
                for video_dir in video_dirs:
                    video_path = video_dir / mp4_name
                    if video_path.exists():
                        return video_path
                    for subdir in video_dir.iterdir():
                        if subdir.is_dir():
                            video_path = subdir / mp4_name
                            if video_path.exists():
                                return video_path
                logger.warning(
                    f"Expected video not found in any video dir: {mp4_name}"
                )
                return None
    logger.warning(f"No tracking subdir found starting with '{video_id}'")
    return None


def assert_embedding_label_alignment(
    embedding_keys: set[tuple], dataset_dir: Path
) -> None:
    """Assert that embedding keys match label keys exactly."""
    labels = pd.read_parquet(dataset_dir / "labels.parquet")
    label_keys = set(
        labels[["video_id", "bird_id", "window"]].itertuples(
            index=False, name=None
        )
    )
    assert embedding_keys == label_keys, (
        f"Window key mismatch with labels: "
        f"{len(embedding_keys - label_keys)} in embeddings only, "
        f"{len(label_keys - embedding_keys)} in labels only"
    )


def load_video_frames_sequential(video_path, start_frame, end_frame):
    """Load frames [start_frame, end_frame) sequentially via cv2.

    Slower than seek-based loading but frame-accurate for all codecs
    (including H.264). Use this when torchcodec is not available (e.g.
    JAX environments).
    """
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    frames = []
    for i in range(end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        if i >= start_frame:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames
