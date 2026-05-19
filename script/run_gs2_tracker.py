"""
Launcher for the Grounded-SAM-2 (gs2) tracker pipeline.

Sets CUDA_VISIBLE_DEVICES and PYTORCH_ALLOC_CONF from the YAML config BEFORE
torch is imported, then hands off to src.tracker.grounded_sam_2.run.

Usage:
    pixi run -e gs2 python -m script.run_gs2_tracker --config config/gs2_fixed_day_28.yaml
"""

import argparse
import os

from omegaconf import OmegaConf


def main():
    parser = argparse.ArgumentParser(description="Grounded-SAM-2 Tracker")
    parser.add_argument(
        "--config",
        type=str,
        default="config/gs2_fixed_day_28.yaml",
        help="Path to config file (default: config/gs2_fixed_day_28.yaml)",
    )
    args, _ = parser.parse_known_args()

    cfg = OmegaConf.load(args.config)
    if cfg.get("CUDA_VISIBLE_DEVICES"):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.CUDA_VISIBLE_DEVICES)
    if cfg.get("PYTORCH_ALLOC_CONF"):
        os.environ["PYTORCH_ALLOC_CONF"] = str(cfg.PYTORCH_ALLOC_CONF)

    from src.tracker.grounded_sam_2 import run

    run(cfg, config_path=args.config)


if __name__ == "__main__":
    main()
