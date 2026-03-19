import os
import re
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


def extract_video_id(tracking_dir_name: str) -> str | None:
    """Extract video_id (e.g. ``C1G3D28``) from a tracking directory name.

    Parses the cage (``CxGy``) and day (``day_Z``) from names like
    ``C1G3_Test_1_day_28_1_Camera_8_2025_02_04_10_59_56_3``.
    """
    m = re.search(r"(C\dG\d).*day_(\d+)", tracking_dir_name)
    if m:
        return f"{m.group(1)}D{m.group(2)}"
    return None


def cage_id_from_video_id(video_id: str) -> str:
    """Extract cage_id (e.g. ``C1``) from a video_id like ``C1G3D28``."""
    return video_id[:2]


def resolve_video_path(
    video_id: str, tracking_dir: Path, video_dirs: list[Path]
) -> Path | None:
    """Find the video file for a given video_id.

    Searches tracking subdirs under *tracking_dir* for directories whose
    extracted video_id matches, then looks for ``{subdir_name}.mp4`` in each
    of *video_dirs* (flat and one level deep).
    """
    for root, _dirs, files in os.walk(tracking_dir, followlinks=True):
        if "tracking_outputs.parquet" in files:
            name = Path(root).name
            if extract_video_id(name) == video_id:
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
                logger.warning(f"Expected video not found in any video dir: {mp4_name}")
                return None
    logger.warning(f"No tracking subdir found for video_id '{video_id}'")
    return None


def assert_embedding_label_alignment(
    embedding_keys: set[tuple], dataset_dir: Path
) -> None:
    """Assert that embedding keys match label keys exactly."""
    labels = pd.read_parquet(dataset_dir / "labels.parquet")
    label_keys = set(
        labels[["video_id", "bird_id", "window"]].itertuples(index=False, name=None)
    )
    assert embedding_keys == label_keys, (
        f"Window key mismatch with labels: "
        f"{len(embedding_keys - label_keys)} in embeddings only, "
        f"{len(label_keys - embedding_keys)} in labels only"
    )
