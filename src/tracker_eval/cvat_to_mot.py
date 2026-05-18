"""Convert a CVAT project Backup export to sparse MOTChallenge ground truth.

Reads each task's `annotations.json` (CVAT native format, which stores only
human-drawn keyframes — no interpolation), assigns track ids 1, 2, 3 in
track-declaration order, and writes `<video_id>.txt` files in MOTChallenge
1.1 format. `outside=true` keyframes (bird fully hidden) are dropped.

Usage:
    pixi run -e tracker-evaluation python -m src.tracker_eval cvat-to-mot \
        --backup-root ext-data/output/results/tracker_benchmark/cvat_backup/playclass-tracker-eval \
        --out-dir ext-data/output/results/tracker_benchmark/ground_truth \
        [--keyframes-csv data/tracker_eval/annotation_frames.csv]

If --keyframes-csv is passed, the converter additionally reports which
scheduled keyframes are missing and which drawn shapes were unscheduled
(extras). The MOT output still contains every drawn (non-outside) shape;
filtering can be re-applied later if needed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .paths import ANNOTATION_FRAMES, CVAT_BACKUP_DIR, GROUND_TRUTH_DIR

MOT_COLS = ["frame", "id", "x", "y", "w", "h", "conf", "_a", "_b", "_c"]


def load_scheduled_keyframes(csv_path: Path) -> dict[str, set[int]]:
    df = pd.read_csv(csv_path, usecols=["video_id", "frame_idx"])
    return {
        str(vid): set(g["frame_idx"].astype(int).tolist())
        for vid, g in df.groupby("video_id")
    }


def convert_task(task_dir: Path) -> tuple[str, pd.DataFrame, dict]:
    """Return (video_id, mot_df, per_bird_stats)."""
    task_meta = json.loads((task_dir / "task.json").read_text())
    video_id = task_meta["name"]
    ann = json.loads((task_dir / "annotations.json").read_text())
    tracks = ann[0]["tracks"]

    records: list[dict] = []
    per_bird: dict[int, dict[str, int]] = {}
    for idx, tr in enumerate(tracks, start=1):  # CVAT track order → id 1..N
        drawn = 0
        outside_count = 0
        for shape in tr["shapes"]:
            if shape.get("outside", False):
                outside_count += 1
                continue
            if shape.get("type") != "rectangle":
                continue
            x1, y1, x2, y2 = shape["points"]
            records.append(
                {
                    "frame": int(shape["frame"]),
                    "id": idx,
                    "x": x1,
                    "y": y1,
                    "w": x2 - x1,
                    "h": y2 - y1,
                    "conf": 1,
                    "_a": -1,
                    "_b": -1,
                    "_c": -1,
                }
            )
            drawn += 1
        per_bird[idx] = {"drawn": drawn, "outside": outside_count}

    df = (
        pd.DataFrame(records, columns=MOT_COLS)
        .sort_values(["frame", "id"])
        .reset_index(drop=True)
    )
    return video_id, df, per_bird


def _add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backup-root", type=Path, default=CVAT_BACKUP_DIR)
    parser.add_argument("--out-dir", type=Path, default=GROUND_TRUTH_DIR)
    parser.add_argument("--keyframes-csv", type=Path, default=ANNOTATION_FRAMES)


def add_subparser(subparsers) -> argparse.ArgumentParser:
    p = subparsers.add_parser(
        "cvat-to-mot",
        help="Convert a CVAT project Backup export to sparse MOTChallenge ground truth.",
    )
    _add_args(p)
    p.set_defaults(func=run)
    return p


def run(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sched = (
        load_scheduled_keyframes(args.keyframes_csv)
        if args.keyframes_csv is not None and Path(args.keyframes_csv).exists()
        else None
    )

    task_dirs = sorted(
        d
        for d in args.backup_root.iterdir()
        if d.is_dir() and d.name.startswith("task_")
    )
    print(f"Found {len(task_dirs)} task dir(s) under {args.backup_root}")

    print()
    print(
        f"{'video_id':<14} {'bird':>5} {'drawn':>7} {'outside':>8} {'sched':>6} {'missing':>8} {'extra':>6}"
    )
    print("-" * 80)

    summary = []
    for td in task_dirs:
        video_id, mot_df, per_bird = convert_task(td)
        out_path = args.out_dir / f"{video_id}.txt"
        mot_df.to_csv(out_path, header=False, index=False, float_format="%.2f")
        summary.append((video_id, len(mot_df), out_path))

        drawn_frames = set(mot_df["frame"].astype(int).unique())
        sched_frames = sched.get(video_id, set()) if sched else None

        for bird_id, stats in per_bird.items():
            n_sched = len(sched_frames) if sched_frames is not None else "-"
            if sched_frames is not None:
                missing = len(sched_frames - drawn_frames)
                extra = len(drawn_frames - sched_frames)
            else:
                missing = extra = "-"
            print(
                f"{video_id:<14} {bird_id:>5} {stats['drawn']:>7} {stats['outside']:>8} {n_sched!s:>6} {missing!s:>8} {extra!s:>6}"
            )

    print()
    print("MOT row counts (sum across 3 birds per video):")
    for video_id, n, path in summary:
        print(f"  {video_id:<14} {n:>5} rows -> {path}")


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    _add_args(parser)
    run(parser.parse_args())


if __name__ == "__main__":
    _main()
