"""
Extract handcrafted mask features from tracked objects.

Separate from ``build_dataset.py`` because feature extraction (decoding RLE
masks, computing centroids) is expensive (~7 min) and dominates runtime.
Splitting lets us rerun the lightweight steps (labels, windows, filtering)
without re-extracting features.

Usage::

    pixi run -e sam3-hf extract_features \
        --tracking-dir data/tracking/20260225_214929_sam3_hf
"""

import sys
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from loguru import logger

from src.dataset.features import extract_mask_features, summarize_features_by_window


def parse_args():
    parser = ArgumentParser(
        description="Extract mask features from dataset_tracks.parquet."
    )
    parser.add_argument(
        "--tracking-dir",
        type=Path,
        required=True,
        help="Batch tracking dir containing dataset_tracks.parquet",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    tracks_path = args.tracking_dir / "dataset_tracks.parquet"
    if not tracks_path.exists():
        logger.error(f"dataset_tracks.parquet not found in {args.tracking_dir}")
        logger.error("Run build_dataset first to generate it.")
        sys.exit(1)

    tracks = pd.read_parquet(tracks_path)

    if "window" not in tracks.columns:
        logger.error(
            "dataset_tracks.parquet missing 'window' column. "
            "Run build_dataset first to assign windows."
        )
        sys.exit(1)

    # Extract per-frame mask features
    logger.info("Extracting mask features...")
    features = extract_mask_features(tracks)
    features = features.merge(
        tracks[["video_id", "bird_id", "frame_idx", "window"]],
        on=["video_id", "bird_id", "frame_idx"],
    )

    # Summarize features per window
    logger.info("Summarizing features by window...")
    features_windowed = summarize_features_by_window(features)

    logger.info(
        f"Extracted {len(features)} per-frame features, "
        f"{len(features_windowed)} per-window features"
    )

    # Save
    all_path = args.tracking_dir / "all_features.parquet"
    features.to_parquet(all_path)
    logger.info(f"Saved: {all_path}")

    windowed_path = args.tracking_dir / "dataset_features.parquet"
    features_windowed.to_parquet(windowed_path)
    logger.info(f"Saved: {windowed_path}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
