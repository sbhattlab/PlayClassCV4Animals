"""Smoke test for built dataset: coverage table + duplicate frame check.

Usage::

    pixi run -e dataset python -m script.smoke_test_dataset
"""

import sys
from pathlib import Path

import pandas as pd
from loguru import logger

from src._config import DEFAULT_DATASET_DIR, DEFAULT_FPS

EXPECTED_FRAMES_PER_WINDOW = int(5 * DEFAULT_FPS)  # 125


def main():
    tracks_path = Path(DEFAULT_DATASET_DIR) / "tracks.parquet"
    if not tracks_path.exists():
        logger.error(f"tracks.parquet not found in {DEFAULT_DATASET_DIR}")
        sys.exit(1)

    tracks = pd.read_parquet(tracks_path)
    n_videos = tracks["video_id"].nunique()
    n_rows = len(tracks)
    logger.info(f"Loaded {n_rows} rows across {n_videos} videos")

    # ---- Per-bird coverage ----
    print()
    worst_cov = 1.0
    worst_label = ""
    for video_id, vg in sorted(tracks.groupby("video_id")):
        parts = []
        for bird_id, bg in sorted(vg.groupby("bird_id")):
            n_windows = bg["window"].nunique()
            n_frames = len(bg)
            expected = n_windows * EXPECTED_FRAMES_PER_WINDOW
            cov = n_frames / expected if expected > 0 else 0.0
            if cov < worst_cov:
                worst_cov = cov
                worst_label = f"{video_id} bird {bird_id}"
            parts.append(f"bird {bird_id} {cov:.2f}x")
        print(f"  {video_id}  {'  '.join(parts)}")

    print(f"\n  Worst coverage: {worst_cov:.2f}x ({worst_label})")

    # ---- Duplicate frames ----
    dupes = tracks[tracks.duplicated(subset=["video_id", "bird_id", "frame_idx"], keep=False)]
    n_dupes = len(dupes)
    if n_dupes == 0:
        print("  No duplicate frames!")
    else:
        print(f"\n  WARNING: {n_dupes} duplicate (video_id, bird_id, frame_idx) rows:")
        print(
            dupes[["video_id", "bird_id", "frame_idx", "window", "chunk_idx"]]
            .to_string(index=False)
        )

    print()


if __name__ == "__main__":
    main()
