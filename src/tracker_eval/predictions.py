"""Convert raw tracker predictions to MOTChallenge 1.1 .txt files.

Three-way ablation:
  - Variant A — YOLO + BoT-SORT only (`yolo_tracking.parquet` under
    `--predictions-root`).
  - Variant B — SAM3 with fixed chunking (`tracking_outputs.parquet`
    under `--predictions-root-fixed`, optional).
  - Variant C — SAM3 with adaptive, occlusion-informed chunking
    (`tracking_outputs.parquet` under `--predictions-root`).

Writes:
  <out-dir>/A_yolo_botsort/<video_id>.txt
  <out-dir>/B_sam3_fixed/<video_id>.txt       (if --predictions-root-fixed given)
  <out-dir>/C_sam3_adaptive/<video_id>.txt

MOTChallenge row format:
    frame, id, bb_left, bb_top, bb_width, bb_height, conf, -1, -1, -1

video_id is resolved from `--manifest` (matches the GT files emitted by
cvat_backup_to_mot.py). The MOT files emitted here are dense (every
predicted frame); motmetrics restricts scoring to GT-present frames
during evaluation, so we do **not** filter predictions here.

Usage:
    pixi run -e tracker-evaluation python -m src.tracker_eval convert-preds \
        --predictions-root ext-data/output/results/tracker_benchmark/tracker_outputs_adaptive \
        --predictions-root-fixed ext-data/output/results/tracker_benchmark/tracker_outputs_fixed \
        --manifest data/tracker_eval/video_manifest.csv \
        --out-dir ext-data/output/results/tracker_benchmark/predictions_mot
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .paths import (
    MANIFEST_CSV,
    PREDICTIONS_MOT_DIR,
    TRACKER_RUNS_ADAPTIVE,
    TRACKER_RUNS_FIXED,
)


def load_video_id_map(manifest_csv: Path) -> dict[str, str]:
    """Return {stem -> video_id} for rows with selected=True."""
    df = pd.read_csv(manifest_csv)
    selected = df[df["selected"].astype(str).str.lower().isin({"true", "1"})]
    return {Path(p).stem: vid for p, vid in zip(selected["path"], selected["video_id"])}


def sam3_to_mot_rows(parquet_path: Path) -> list[str]:
    """tracking_outputs.parquet (SAM3) → MOTChallenge rows.

    Index is MultiIndex(frame_idx, object_id). Column `bbox` is [x1,y1,x2,y2].
    Confidence is `tracker_score` (per-frame tracker quality).
    """
    df = pd.read_parquet(parquet_path, columns=["bbox", "tracker_score"])
    rows: list[tuple[int, int, str]] = []
    for (frame_idx, object_id), row in df.iterrows():
        x1, y1, x2, y2 = row["bbox"]
        w = x2 - x1
        h = y2 - y1
        conf = float(row["tracker_score"]) if pd.notna(row["tracker_score"]) else 1.0
        rows.append(
            (
                int(frame_idx),
                int(object_id),
                f"{int(frame_idx)},{int(object_id)},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{conf:.4f},-1,-1,-1",
            )
        )
    rows.sort(key=lambda r: (r[0], r[1]))
    return [r[2] for r in rows]


def yolo_to_mot_rows(parquet_path: Path) -> list[str]:
    """yolo_tracking.parquet (YOLO+BoT-SORT) → MOTChallenge rows.

    Flat schema: [frame, track_id, x1, y1, x2, y2, ..., confidence].
    Rows with `track_id == -1` are untracked detections (BoT-SORT did
    not assign an identity) — dropped, since they cannot contribute to
    identity-association metrics.
    """
    df = pd.read_parquet(parquet_path)
    df = df[df["track_id"] != -1]
    rows: list[tuple[int, int, str]] = []
    for r in df.itertuples(index=False):
        w = r.x2 - r.x1
        h = r.y2 - r.y1
        conf = float(r.confidence) if pd.notna(r.confidence) else 1.0
        rows.append(
            (
                int(r.frame),
                int(r.track_id),
                f"{int(r.frame)},{int(r.track_id)},{r.x1:.2f},{r.y1:.2f},{w:.2f},{h:.2f},{conf:.4f},-1,-1,-1",
            )
        )
    rows.sort(key=lambda r: (r[0], r[1]))
    return [r[2] for r in rows]


def write_rows(out_path: Path, rows: list[str]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(rows) + ("\n" if rows else ""))


def _add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--predictions-root",
        type=Path,
        default=TRACKER_RUNS_ADAPTIVE,
        help="Adaptive SAM3 + YOLO scan run dir (supplies A_yolo_botsort and C_sam3_adaptive).",
    )
    parser.add_argument(
        "--predictions-root-fixed",
        type=Path,
        default=TRACKER_RUNS_FIXED,
        help="Fixed-chunking SAM3 run dir (supplies B_sam3_fixed). Skipped if dir is missing or empty.",
    )
    parser.add_argument("--manifest", type=Path, default=MANIFEST_CSV)
    parser.add_argument("--out-dir", type=Path, default=PREDICTIONS_MOT_DIR)


def add_subparser(subparsers) -> argparse.ArgumentParser:
    p = subparsers.add_parser(
        "convert-preds",
        help="Convert tracker prediction parquets (3-way ablation) to MOTChallenge .txt files.",
    )
    _add_args(p)
    p.set_defaults(func=run)
    return p


def run(args: argparse.Namespace) -> None:
    stem_to_video = load_video_id_map(args.manifest)
    print(f"{len(stem_to_video)} selected videos in manifest")

    summary = []
    for stem, video_id in sorted(stem_to_video.items(), key=lambda kv: kv[1]):
        run_dir = args.predictions_root / stem
        if not run_dir.exists():
            print(f"  [WARN] {video_id}: adaptive run dir missing at {run_dir}; skipping")
            continue

        sam3_pq = run_dir / "tracking_outputs.parquet"
        yolo_pq = run_dir / "yolo_tracking.parquet"

        a_rows = yolo_to_mot_rows(yolo_pq) if yolo_pq.exists() else []
        c_rows = sam3_to_mot_rows(sam3_pq) if sam3_pq.exists() else []

        a_out = args.out_dir / "A_yolo_botsort" / f"{video_id}.txt"
        c_out = args.out_dir / "C_sam3_adaptive" / f"{video_id}.txt"
        write_rows(a_out, a_rows)
        write_rows(c_out, c_rows)

        b_rows: list[str] = []
        if args.predictions_root_fixed is not None and args.predictions_root_fixed.exists():
            fixed_run_dir = args.predictions_root_fixed / stem
            sam3_fixed_pq = fixed_run_dir / "tracking_outputs.parquet"
            if sam3_fixed_pq.exists():
                b_rows = sam3_to_mot_rows(sam3_fixed_pq)
                b_out = args.out_dir / "B_sam3_fixed" / f"{video_id}.txt"
                write_rows(b_out, b_rows)
            else:
                print(
                    f"  [skip] {video_id}: fixed-chunking parquet missing at {sam3_fixed_pq}"
                )

        summary.append((video_id, len(a_rows), len(b_rows), len(c_rows)))

    print()
    print(
        f"{'video_id':<14} {'A (YOLO+BoT-SORT)':>20} {'B (SAM3 fixed)':>18} {'C (SAM3 adaptive)':>20}"
    )
    print("-" * 80)
    for vid, na, nb, nc in summary:
        print(f"{vid:<14} {na:>20} {nb:>18} {nc:>20}")


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    _add_args(parser)
    run(parser.parse_args())


if __name__ == "__main__":
    _main()
