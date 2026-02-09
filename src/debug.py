"""
Utilities for use in interactive debugging sessions (e.g. in Jupyter notebooks or IPython shell).
"""

import json
from collections import namedtuple
from pathlib import Path

import pandas as pd
import supervision as sv
from omegaconf import DictConfig, OmegaConf

from src.utils import load_config

JOB_TYPE = "sam3_hf"
CONFIG_PATH = f"config/{JOB_TYPE}_config.yaml"

Metrics = namedtuple("Metrics", ["per_frame", "per_id", "summary"])


def load_inputs(config_path=CONFIG_PATH) -> tuple[DictConfig, sv.VideoInfo]:

    assert Path(config_path).exists(), f"Config file {config_path} does not exist."
    cfg = load_config(config_path)
    video_info = sv.VideoInfo.from_video_path(cfg.video_path)
    return cfg, video_info


def load_outputs(config_path=CONFIG_PATH, run_id="20260207_185451_sam3_hf"):
    cfg = load_config(config_path)
    run_dir = Path(f"{cfg.output_dir}/{run_id}")
    assert run_dir.exists(), f"Run directory {run_dir} does not exist."
    metrics_dir = run_dir / "metrics"
    # viz_dir = run_dir / "visualizations"

    tracking_outputs = pd.read_parquet(run_dir / "tracking_outputs.parquet")

    with open(run_dir / "chunk_info.json", "r") as f:
        chunk_info = json.load(f)

    metrics = Metrics(
        pd.read_parquet(metrics_dir / "per_frame_metrics.parquet"),
        pd.read_parquet(metrics_dir / "per_id_metrics.parquet"),
        pd.read_parquet(metrics_dir / "summary_metrics.parquet"),
    )

    return metrics, chunk_info, tracking_outputs
