"""
Extract handcrafted mask features from tracked objects.

Separate from ``build_dataset.py`` because feature extraction (decoding RLE
masks, computing centroids) is expensive (~7 min) and dominates runtime.
Splitting lets us rerun the lightweight steps (labels, windows, filtering)
without re-extracting features.

Usage::

    pixi run -e sam3-hf extract_features
"""

import sys
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import torch
from loguru import logger

from src._config import DEFAULT_DATASET_DIR
from src.dataset.features import (
    bin_features_per_window,
    extract_mask_features,
    summarize_features_by_window,
)


def parse_args():
    parser = ArgumentParser(
        description="Extract mask features from tracks.parquet."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Directory containing tracks.parquet and labels.parquet (default: %(default)s)",
    )
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Skip mask decoding; re-summarize from existing features_all.parquet",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    all_path = args.dataset_dir / "features_all.parquet"
    windowed_path = args.dataset_dir / "features_windowed.parquet"

    if args.summarize_only:
        if not all_path.exists():
            logger.error(f"features_all.parquet not found in {args.dataset_dir}")
            logger.error("Run without --summarize-only first to extract per-frame features.")
            sys.exit(1)
        features = pd.read_parquet(all_path)
        logger.info(f"Loaded {len(features)} per-frame features from {all_path}")
    else:
        tracks_path = args.dataset_dir / "tracks.parquet"
        if not tracks_path.exists():
            logger.error(f"tracks.parquet not found in {args.dataset_dir}")
            logger.error("Run build_dataset first to generate it.")
            sys.exit(1)

        tracks = pd.read_parquet(tracks_path)

        if "window" not in tracks.columns:
            logger.error(
                "tracks.parquet missing 'window' column. "
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
        features.to_parquet(all_path)
        logger.info(f"Saved: {all_path}")

    # Summarize features per window
    logger.info("Summarizing features by window...")
    features_windowed = summarize_features_by_window(features)

    logger.info(
        f"{len(features)} per-frame features → "
        f"{len(features_windowed)} per-window features"
    )

    # Check alignment with labels
    labels = pd.read_parquet(args.dataset_dir / "labels.parquet")
    _key_cols = ["video_id", "bird_id", "window"]
    feature_keys = set(features_windowed[_key_cols].itertuples(index=False, name=None))
    label_keys = set(labels[_key_cols].itertuples(index=False, name=None))
    assert feature_keys == label_keys, (
        f"Window key mismatch with labels: {len(feature_keys - label_keys)} in features only, "
        f"{len(label_keys - feature_keys)} in labels only"
    )

    features_windowed.to_parquet(windowed_path)
    logger.info(f"Saved: {windowed_path}")

    # Build temporal feature tensors (same format as embeddings.pt)
    logger.info("Building temporal feature tensors...")
    temporal_dict = bin_features_per_window(features)
    temporal_path = args.dataset_dir / "features_binned.pt"
    torch.save(temporal_dict, temporal_path)
    logger.info(
        f"Saved: {temporal_path} ({len(temporal_dict)} windows, "
        f"{next(iter(temporal_dict.values())).shape[-1]} features)"
    )

    logger.info("Done.")


if __name__ == "__main__":
    main()
