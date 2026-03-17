"""
Launcher for the SAM3 Tracker pipeline.

Sets CUDA_VISIBLE_DEVICES and other env vars from the YAML config
BEFORE torch is imported, then hands off to the real pipeline module.

Usage:
    python -m script.run_tracker --config config/sam3_hf_config.yaml
"""

import argparse
import os

from omegaconf import OmegaConf


def main():
    parser = argparse.ArgumentParser(description="SAM3 Tracker")
    parser.add_argument(
        "--config",
        type=str,
        default="config/sam3_hf_config.yaml",
        help="Path to config file (default: config/tracker_config.yaml)",
    )
    args, _ = parser.parse_known_args()

    # Load config and set env vars BEFORE any torch import
    cfg = OmegaConf.load(args.config)
    if cfg.get("CUDA_VISIBLE_DEVICES"):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.CUDA_VISIBLE_DEVICES)
    if cfg.get("PYTORCH_ALLOC_CONF"):
        os.environ["PYTORCH_ALLOC_CONF"] = str(cfg.PYTORCH_ALLOC_CONF)

    # Now safe to import torch-dependent code
    from src.tracker.tracker import run

    run(cfg, config_path=args.config)


if __name__ == "__main__":
    main()
