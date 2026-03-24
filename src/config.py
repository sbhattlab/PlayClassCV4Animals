"""
Configuration and environment setup utilities
"""

import os
import re
import sys
from datetime import datetime
from pathlib import Path

from loguru import logger
from omegaconf import DictConfig, OmegaConf


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
    log_dir: Path = Path("tmp/logs"),
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


def create_video_run_directory(base_output_dir: Path, video_stem: str) -> Path:
    """Create a dataset-aligned run directory: ``{base}/day_{N}/{sanitized_stem}/``.

    Parses the video ID from *video_stem* via
    :func:`~src.dataset.utils.extract_video_id` to obtain the day number.
    Falls back to a flat ``{base}/{sanitized_stem}/`` if parsing fails.
    """
    from src.dataset.utils import extract_video_id

    sanitized = sanitize_filename(video_stem)
    video_id = extract_video_id(video_stem)
    if video_id is not None:
        day = video_id[5:]  # "C1G3D28" → "28"
        run_dir = base_output_dir / f"day_{day}" / sanitized
    else:
        logger.warning(
            f"Could not parse video ID from '{video_stem}'; "
            "output will not be grouped by day"
        )
        run_dir = base_output_dir / sanitized
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def sanitize_filename(name: str) -> str:
    """Sanitize a stem string for use as a directory name."""
    sanitized = re.sub(r"[^\w\-]", "_", name)
    sanitized = re.sub(r"_+", "_", sanitized)
    return sanitized.strip("_") or "video"
