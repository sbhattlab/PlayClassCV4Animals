"""
Build tracking_issues.json and tracking_postprocessing.json for each tracking
subdirectory, then build the dataset in a single pass (no redundant disk
re-reads).

Given a batch tracking directory (containing per-video subdirectories, each with
``tracking_outputs.parquet``), this script:

1. Discovers all tracking subdirectories
2. Parses labels and bird info from Excel files (once)
3. For each subdirectory:
   a. Reads FPS from ``yolo_scan_summary.parquet`` (falls back to 25.0)
   b. Loads ``tracking_outputs.parquet``
   c. Detects ID transitions and mask overlaps → ``tracking_issues.json``
   d. Creates or loads ``tracking_postprocessing.json``; if any remap ``to``
      values are null, stops.
4. Once all remaps are filled in, runs ``process_tracks`` on the in-memory data
   and concatenates the results.

Typical workflow::

    # First run: generates tracking_issues.json + tracking_postprocessing.json
    pixi run -e sam3-hf python -m script.preprocess \\
        --tracking-dir data/tracking/20260225_214929_sam3_hf \\
        --label-dir data/labels

    # Manually fill in "to" values and add trim entries in each
    # tracking_postprocessing.json

    # Second run: validates remaps, builds dataset
    pixi run -e sam3-hf python -m script.preprocess \\
        --tracking-dir data/tracking/20260225_214929_sam3_hf \\
        --label-dir data/labels
"""

import json
from argparse import ArgumentParser
from glob import glob
from pathlib import Path

import pandas as pd
from loguru import logger

from src.dataset.labels import process_labels
from src.dataset.tracking_issues import detect_tracking_issues
from src.dataset.tracking_postprocessing import (
    check_postprocessing,
    prefill_postprocessing,
    process_tracks,
)
from src.dataset.utils import fmt_time, get_video_fps

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _log_issues(issues, fps):
    """Log a summary of detected tracking issues."""
    n_switches = sum(1 for v in issues.values() if v["type"] == "id_switch")
    n_overlaps = sum(1 for v in issues.values() if v["type"] == "overlap")
    n_low = sum(1 for v in issues.values() if v["type"] == "low_score")
    logger.info(
        f"  Found {n_switches} ID switch(es), {n_overlaps} overlap(s), "
        f"{n_low} low-score period(s)"
    )

    for name, ev in issues.items():
        if ev["type"] == "overlap":
            dur = ev["end_time"] - ev["start_time"]
            logger.info(
                f"    {name}: IDs {ev['ids']}  "
                f"{fmt_time(ev['start'], fps)} - {fmt_time(ev['end'], fps)}  "
                f"({dur:.1f}s)"
            )
        elif ev["type"] == "low_score":
            dur = ev["end_time"] - ev["start_time"]
            logger.info(
                f"    {name}: ID {ev['id']}  "
                f"{fmt_time(ev['start'], fps)} - {fmt_time(ev['end'], fps)}  "
                f"({dur:.1f}s, min_score={ev['min_score']:.3f}, "
                f"{ev['low_score_rows']}/{ev['total_rows']} rows)"
            )
        else:
            logger.info(
                f"    {name}: frame {ev['start']}  "
                f"({fmt_time(ev['start'], fps)})  "
                f"{ev['ids_before']} -> {ev['ids_after']}"
            )


def process_tracking_subdir(tracking_dir, bird_info):
    """Detect issues, write JSONs, return in-memory results.

    Returns
    -------
    dict
        ``tracks``: DataFrame from tracking_outputs.parquet
        ``issues``: detected tracking issues dict
        ``postprocessing``: list of postprocessing entries (trims + remaps)
        ``fps``: video FPS
        ``ready``: True if all remaps have ``to`` values filled in
    """
    tracking_dir = Path(tracking_dir)
    logger.info(f"Processing {tracking_dir.name}")

    fps = get_video_fps(tracking_dir)
    tracks = pd.read_parquet(tracking_dir / "tracking_outputs.parquet")
    logger.info(f"  FPS: {fps:.2f}, {len(tracks)} tracking rows")

    issues = detect_tracking_issues(tracks, fps)
    _log_issues(issues, fps)

    _save_json(tracking_dir / "tracking_issues.json", issues)

    # Match subdir name to video_id (e.g. "C1G1_day28" starts with "C1G1")
    video_birds = {}
    for video_id, birds in bird_info.items():
        if tracking_dir.name.startswith(video_id):
            video_birds = birds
            break
    if video_birds:
        _save_json(tracking_dir / "bird_info.json", video_birds)
        logger.info(f"  Saved bird_info.json ({len(video_birds)} bird(s))")
    else:
        logger.warning(f"  No bird info matched for {tracking_dir.name}")

    # Load or create tracking_postprocessing.json
    pp_path = tracking_dir / "tracking_postprocessing.json"
    if pp_path.exists():
        postprocessing = _load_json(pp_path)
        logger.info(
            f"  tracking_postprocessing.json already exists "
            f"({len(postprocessing)} entry/ies)"
        )
    else:
        postprocessing = prefill_postprocessing(issues, tracks, video_birds, fps)
        _save_json(pp_path, postprocessing)
        logger.info(
            f"  Created tracking_postprocessing.json with "
            f"{len(postprocessing)} template(s)"
        )

    # Check readiness
    try:
        check_postprocessing(postprocessing)
        ready = True
    except ValueError as e:
        logger.warning(f"  {tracking_dir.name}: {e}")
        ready = False

    return {
        "tracks": tracks,
        "postprocessing": postprocessing,
        "fps": fps,
        "ready": ready,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = ArgumentParser(
        description="Build tracking_issues.json per subdirectory, then build dataset."
    )
    parser.add_argument(
        "--tracking-dir",
        type=Path,
        required=True,
        help="Path to batch tracking dir (contains per-video subdirs)",
    )
    parser.add_argument(
        "--label-dir",
        type=Path,
        required=True,
        help="Path to directory containing Registration protocols Excel files",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Glob label files
    label_files = sorted(glob(str(args.label_dir / "*.xlsx")))
    if not label_files:
        logger.error(f"No .xlsx files found in {args.label_dir}")
        return
    logger.info(f"Found {len(label_files)} label file(s)")

    # 2. Parse labels + bird info
    labels, bird_info = process_labels(label_files)

    # 3. Discover tracking subdirs
    tracking_dirs = sorted(
        [p.parent for p in args.tracking_dir.rglob("tracking_outputs.parquet")]
    )
    if not tracking_dirs:
        logger.error(f"No tracking_outputs.parquet found under {args.tracking_dir}")
        return
    logger.info(f"Found {len(tracking_dirs)} tracking subdirectory(ies)")

    # 4. Detect issues + prefill remaps per subdir (keeps data in memory)
    results = [process_tracking_subdir(d, bird_info) for d in tracking_dirs]

    if not all(r["ready"] for r in results):
        logger.warning(
            "Cannot build dataset — fill in 'to' (id_switch) and "
            "'tracking_id' (id_match) fields, then re-run."
        )
        return

    # 5. Build dataset from in-memory data
    logger.info("All remaps validated. Building dataset...")
    all_tracks = []
    for r in results:
        tracks_clean, labels = process_tracks(
            tracks=r["tracks"],
            labels=labels,
            postprocessing=r["postprocessing"],
            fps=r["fps"],
        )
        all_tracks.append(tracks_clean)

    tracks = pd.concat(all_tracks, ignore_index=True)
    logger.info(f"Dataset built: {len(tracks)} track rows, {len(labels)} label rows")

    # 6. Save dataset
    tracks.to_parquet(args.tracking_dir / "dataset_tracks.parquet")
    labels.to_parquet(args.tracking_dir / "dataset_labels.parquet")
    logger.info(
        f"Saved dataset_tracks.parquet and dataset_labels.parquet "
        f"to {args.tracking_dir}"
    )


if __name__ == "__main__":
    main()
