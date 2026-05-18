"""Build `tracking_eval/video_manifest.csv` from the cached YOLO scan parquets.

Ranks the 30 candidate videos (5 cages × 3 groups × 2 days) by a composite
difficulty score computed across 7 proxies derived from
`metrics/yolo_scan_metrics.parquet` and `metrics/yolo_scan_summary.parquet`
of each video's YOLO scan run. The hardest group within each cage
(rank_in_cage == 1) is marked `selected=True` and forms the 5-video
held-out evaluation set.

Proxies (each independently ranked across all 30 candidates; ↑ harder means
ascending rank, ↓ harder means descending rank):

  - frac_high_occlusion              ↑
  - mean_overlapping_pairs           ↑
  - mean_pairwise_bbox_iou           ↑
  - frac_object_count_change         ↑
  - num_occlusion_periods_per_min    ↑
  - mean_centroid_distance           ↓
  - mean_separation_score            ↓

`difficulty` is the sum of per-proxy ranks (range 7–210 across 30 rows;
higher = harder). `rank_in_cage` ranks the 6 candidates within each cage
by `difficulty` (1 = hardest).

Sanity check (`--min-day-28`): the script fails if fewer than N of the 5
selected videos come from day 28. The joint cross-day ranking has
historically produced 2 day-28 picks without forcing.

Scan-dir discovery walks `--scan-runs-root` (default
`ext-data/output/results/sam3-hf`) for `{YYYYMMDD_HHMMSS}_sam3_hf/{stem}/`
directories containing `yolo_tracking.parquet`. When the same video stem
appears in multiple timestamped runs, the lexicographically latest run is
used (timestamp prefix orders correctly).

Usage:
    pixi run -e tracker python -m tracking_eval build-manifest \
        --scan-runs-root ext-data/output/results/sam3-hf \
        --raw-video-root ext-data/raw \
        --out tracking_eval/video_manifest.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from .paths import MANIFEST_CSV, RAW_VIDEO_ROOT, ROOT, SCAN_RUNS_ROOT

STEM_PATTERN = re.compile(r"^(?P<cage>C\d)(?P<group>G\d)_Test_\d+_day_(?P<day>\d+)_")

# (proxy column, higher_is_harder)
PROXIES: tuple[tuple[str, bool], ...] = (
    ("frac_high_occlusion", True),
    ("mean_overlapping_pairs", True),
    ("mean_pairwise_bbox_iou", True),
    ("frac_object_count_change", True),
    ("num_occlusion_periods_per_min", True),
    ("mean_centroid_distance", False),
    ("mean_separation_score", False),
)

MANIFEST_COLUMNS = [
    "video_id",
    "cage",
    "group",
    "day",
    "path",
    "scan_dir",
    "selected",
    "notes",
    "difficulty",
    "rank_in_cage",
]


def parse_stem(stem: str) -> dict | None:
    m = STEM_PATTERN.match(stem)
    if not m:
        return None
    return {
        "cage": m.group("cage"),
        "group": m.group("group"),
        "day": int(m.group("day")),
    }


def discover_scan_dirs(scan_runs_root: Path) -> dict[str, Path]:
    """Return {video_stem: latest_scan_dir} across timestamped run dirs."""
    by_stem: dict[str, Path] = {}
    for run_dir in sorted(scan_runs_root.iterdir()):
        if not run_dir.is_dir():
            continue
        for video_dir in sorted(run_dir.iterdir()):
            if not video_dir.is_dir():
                continue
            if not (video_dir / "yolo_tracking.parquet").exists():
                continue
            existing = by_stem.get(video_dir.name)
            if existing is None or video_dir.parent.name > existing.parent.name:
                by_stem[video_dir.name] = video_dir
    return by_stem


def compute_difficulty_proxies(scan_dir: Path) -> dict[str, float]:
    pf = pd.read_parquet(scan_dir / "metrics" / "yolo_scan_metrics.parquet")
    summary = pd.read_parquet(scan_dir / "metrics" / "yolo_scan_summary.parquet").iloc[0]
    duration_min = float(summary["video_duration_seconds"]) / 60.0

    finite_mcd = pf.loc[pf["mean_centroid_distance"] != float("inf"), "mean_centroid_distance"]
    return {
        "frac_high_occlusion": float(pf["is_high_occlusion"].mean()),
        "mean_overlapping_pairs": float(pf["num_overlapping_pairs"].mean()),
        "mean_pairwise_bbox_iou": float(pf["mean_pairwise_bbox_iou"].mean()),
        "frac_object_count_change": float(pf["is_object_count_change"].mean()),
        "num_occlusion_periods_per_min": (
            float(summary["num_occlusion_periods"]) / duration_min
            if duration_min > 0
            else 0.0
        ),
        "mean_centroid_distance": float(finite_mcd.mean()) if len(finite_mcd) else 0.0,
        "mean_separation_score": float(pf["separation_score"].mean()),
    }


def _add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--scan-runs-root", type=Path, default=SCAN_RUNS_ROOT)
    parser.add_argument("--raw-video-root", type=Path, default=RAW_VIDEO_ROOT)
    parser.add_argument("--out", type=Path, default=MANIFEST_CSV)
    parser.add_argument(
        "--days",
        type=int,
        nargs="+",
        default=[28, 29],
        help="Candidate days to include (day-37 leftover scan dirs are filtered out by default).",
    )
    parser.add_argument(
        "--min-day-28",
        type=int,
        default=2,
        help="Fail if fewer than N day-28 videos end up selected.",
    )


def add_subparser(subparsers) -> argparse.ArgumentParser:
    p = subparsers.add_parser(
        "build-manifest",
        help="Rank 30 candidate videos by composite YOLO-scan difficulty; pick hardest group per cage.",
    )
    _add_args(p)
    p.set_defaults(func=run)
    return p


def run(args: argparse.Namespace) -> None:
    stem_to_scan = discover_scan_dirs(args.scan_runs_root)
    print(f"Discovered {len(stem_to_scan)} scan dirs under {args.scan_runs_root}")

    rows: list[dict] = []
    for stem, scan_dir in sorted(stem_to_scan.items()):
        parsed = parse_stem(stem)
        if parsed is None:
            print(f"  [skip] unparseable stem: {stem}")
            continue
        if parsed["day"] not in args.days:
            continue
        proxies = compute_difficulty_proxies(scan_dir)
        raw_video_path = args.raw_video_root / f"day_{parsed['day']}" / f"{stem}.mp4"
        rows.append(
            {
                "video_id": f"{parsed['cage']}{parsed['group']}_day_{parsed['day']}",
                "cage": parsed["cage"],
                "group": parsed["group"],
                "day": parsed["day"],
                "path": str(raw_video_path.relative_to(ROOT)),
                "scan_dir": str(scan_dir.relative_to(ROOT)),
                **proxies,
            }
        )

    df = pd.DataFrame(rows)
    counts_by_day = df["day"].value_counts().sort_index().to_dict()
    print(f"Built {len(df)} candidate rows ({counts_by_day})")
    expected = 5 * 3 * len(args.days)
    if len(df) != expected:
        print(
            f"  [WARN] expected {expected} candidates (5 cages × 3 groups × {len(args.days)} days); got {len(df)}"
        )

    for col, higher_is_harder in PROXIES:
        df[f"rank_{col}"] = df[col].rank(method="min", ascending=higher_is_harder)
    rank_cols = [f"rank_{c}" for c, _ in PROXIES]
    df["difficulty"] = df[rank_cols].sum(axis=1).astype(int)

    df["rank_in_cage"] = (
        df.groupby("cage")["difficulty"].rank(method="min", ascending=False).astype(int)
    )
    df["selected"] = df["rank_in_cage"] == 1
    df["notes"] = df["selected"].map(
        {True: "hardest group in cage across both days", False: ""}
    )

    n_day_28 = int(((df["selected"]) & (df["day"] == 28)).sum())
    if n_day_28 < args.min_day_28:
        raise SystemExit(
            f"Only {n_day_28} of the {int(df['selected'].sum())} selected videos are from day 28 "
            f"(expected ≥ {args.min_day_28}). Inspect ranking before overriding."
        )

    out = (
        df[MANIFEST_COLUMNS]
        .sort_values(["cage", "rank_in_cage"])
        .reset_index(drop=True)
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Wrote {len(out)} rows to {args.out}")

    print()
    print("=== Selected (rank_in_cage == 1) ===")
    with pd.option_context("display.width", 200, "display.max_colwidth", 100):
        print(out[out["selected"]].to_string(index=False))


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    _add_args(parser)
    run(parser.parse_args())


if __name__ == "__main__":
    _main()
