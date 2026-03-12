"""
Config, environment, logging, output directory, and IO utilities.
"""

import gc
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import psutil
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


def sanitize_filename(name: str) -> str:
    """Sanitize a stem string for use as a directory name."""
    sanitized = re.sub(r"[^\w\-]", "_", name)
    sanitized = re.sub(r"_+", "_", sanitized)
    return sanitized.strip("_") or "video"


def build_manual_chunks(
    frame_pairs: list[list[int]],
    tracker_override_indices: set[int] | None = None,
) -> list[tuple[int, int, str]]:
    """
    Build chunk list from user-supplied (start_frame, end_frame) pairs.

    If tracker_override_indices is provided, those chunk indices are forced to
    "tracker" type regardless of position (e.g. chunk 0 with manual prompts).
    """
    chunks = []
    for i, (start, end) in enumerate(frame_pairs):
        if tracker_override_indices and i in tracker_override_indices:
            model_type = "tracker"
        else:
            model_type = "video" if i == 0 else "tracker"
        chunks.append((int(start), int(end), model_type))
    return chunks


def load_chunks_from_chunk_info(
    run_dir: Path,
) -> tuple[list[list[int]], dict[int, dict[int, list]]]:
    """
    Load chunk boundaries and optional prompt points from a previous run's
    chunk_info.json.

    Returns:
        frame_pairs: list of [start_frame, end_frame] pairs
        prompt_points: dict mapping chunk_idx → {obj_id: [[x,y], ...]} for
            chunks that have non-empty prompt_points
    """
    import json

    chunk_info_path = Path(run_dir) / "chunk_info.json"
    if not chunk_info_path.exists():
        raise FileNotFoundError(f"chunk_info.json not found in: {run_dir}")

    with open(chunk_info_path) as f:
        chunk_info = json.load(f)

    frame_pairs = [chunk["frame_range"] for chunk in chunk_info["chunks"]]

    prompt_points: dict[int, dict[int, list]] = {}
    for i, chunk in enumerate(chunk_info["chunks"]):
        pp = chunk.get("prompt_points")
        if pp:
            # JSON serializes dict keys as strings; convert to int
            prompt_points[i] = {int(k): v for k, v in pp.items()}

    return frame_pairs, prompt_points


# ---------------------------------------------------------------------------
# GPU and system memory utilities
# ---------------------------------------------------------------------------


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
    logger.info(
        f"RAM before gc ({prefix}{before_gb:.1f} GB used, {available_gb:.1f} GB available)"
    )

    gc.collect()

    vm = psutil.virtual_memory()
    after_gb = vm.used / 1024**3
    available_gb = vm.available / 1024**3
    freed_gb = before_gb - after_gb
    logger.info(
        f"RAM after gc ({prefix}{after_gb:.1f} GB used, {available_gb:.1f} GB available, freed {freed_gb:.1f} GB)"
    )


# ---------------------------------------------------------------------------
# IO utilities
# ---------------------------------------------------------------------------


def get_video_metadata(video_path: str | Path) -> tuple[float, int]:
    """Return (fps, total_frames) for a video without loading all frames into RAM."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, total_frames


def load_video_frames_range(
    video_path: str | Path, start_frame: int, end_frame: int
) -> list:
    """Load frames [start_frame, end_frame) from a video file as a list of RGB numpy arrays.

    Uses cv2 seek (CAP_PROP_POS_FRAMES). Note: seek-based indexing can be
    unreliable on some codecs/containers — prefer load_video_frames_torchcodec
    for frame-accurate results.
    """
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


def load_video_frames_torchcodec(
    video_path: str | Path, start_frame: int, end_frame: int
) -> list:
    """Load frames [start_frame, end_frame) as a list of RGB numpy arrays.

    Uses torchcodec for frame-accurate decoding (seek_mode="exact").
    The first call on a video incurs a ~18 s index scan; subsequent calls
    (or calls after the OS has cached the file) are fast.
    """
    from torchcodec.decoders import VideoDecoder

    decoder = VideoDecoder(str(video_path))
    batch = decoder.get_frames_in_range(start=start_frame, stop=end_frame)
    # batch.data shape: (N, C, H, W) uint8 tensor
    return [frame.permute(1, 2, 0).numpy() for frame in batch.data]
