from pathlib import Path

import pandas as pd
import pycocotools.mask as mask_util
from loguru import logger

DEFAULT_FPS = 25.0


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


def _decode_rle_mask(counts, size):
    """Decode an RLE-encoded mask to a binary numpy array."""
    if isinstance(counts, str):
        counts = counts.encode("utf-8")
    rle = {"counts": counts, "size": size}
    return mask_util.decode(rle).astype(bool)
