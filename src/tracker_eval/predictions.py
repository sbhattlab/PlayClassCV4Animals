"""Convert raw tracker predictions to MOTChallenge 1.1 .txt files.

Six-way ablation (ordered by Family → Recovery):
  - Variant A         — YOLO + BoT-SORT only.
  - Variant B-strict  — Grounded-SAM-2 strict (no recovery scaffolding).
  - Variant B-parity  — Grounded-SAM-2 with parity recovery.
  - Variant C-strict  — SAM 3 frame-zero (no scan, no fallbacks).
  - Variant D         — SAM 3 adaptive grounding + fixed chunking.
  - Variant E         — SAM 3 adaptive grounding + adaptive chunking.

Writes one .txt per variant per selected video, under `<out-dir>/<bucket>/`.
Variants whose source dir is missing produce an empty MOT file so the
scorer still sees the variant (rather than silently dropping its row).

MOTChallenge row format:
    frame, id, bb_left, bb_top, bb_width, bb_height, conf, -1, -1, -1

video_id is resolved from `--manifest`. Predictions emitted here are
dense (every predicted frame); motmetrics / TrackEval restrict scoring
to GT-present frames at evaluation time, so we do not filter here.
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
    TRACKER_RUNS_FRAME_ZERO,
    TRACKER_RUNS_GS2,
    TRACKER_RUNS_GS2_STRICT,
)


def load_video_id_map(manifest_csv: Path) -> dict[str, str]:
    """Return {stem -> video_id} for rows with selected=True."""
    df = pd.read_csv(manifest_csv)
    selected = df[df["selected"].astype(str).str.lower().isin({"true", "1"})]
    return {Path(p).stem: vid for p, vid in zip(selected["path"], selected["video_id"])}


def sam3_to_mot_rows(parquet_path: Path) -> list[str]:
    """tracking_outputs.parquet (SAM 3 / gs2) → MOTChallenge rows."""
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
    """yolo_tracking.parquet (YOLO+BoT-SORT) → MOTChallenge rows."""
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


def _convert_sam3_bucket(
    args: argparse.Namespace,
    stem: str,
    video_id: str,
    root_attr: str,
    bucket: str,
) -> int:
    """Convert one stem's tracking_outputs.parquet from a SAM3-family root.

    Writes an empty MOT file if the root or the per-stem parquet is missing,
    so the scorer always sees this variant (and counts it as a zero-
    prediction penalty rather than dropping the row).
    """
    root = getattr(args, root_attr)
    out_path = args.out_dir / bucket / f"{video_id}.txt"
    if root is None or not root.exists():
        write_rows(out_path, [])
        return 0
    run_dir = root / stem
    pq = run_dir / "tracking_outputs.parquet"
    if not pq.exists():
        print(
            f"  [empty] {video_id}: {bucket} parquet missing at {pq} "
            f"(writing empty MOT file so the variant is scored)"
        )
        write_rows(out_path, [])
        return 0
    rows = sam3_to_mot_rows(pq)
    write_rows(out_path, rows)
    return len(rows)


def _add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--predictions-root",
        type=Path,
        default=TRACKER_RUNS_ADAPTIVE,
        help="Adaptive SAM 3 + YOLO scan run dir (supplies A_yolo_botsort and E_sam3_adaptive).",
    )
    parser.add_argument(
        "--predictions-root-fixed",
        type=Path,
        default=TRACKER_RUNS_FIXED,
        help="Fixed-chunking SAM 3 run dir (supplies D_sam3_fixed). Skipped if dir is missing or empty.",
    )
    parser.add_argument(
        "--predictions-root-frame-zero",
        type=Path,
        default=TRACKER_RUNS_FRAME_ZERO,
        help="SAM 3 frame-zero run dir (supplies C_sam3_frame_zero). Skipped if dir is missing or empty.",
    )
    parser.add_argument(
        "--predictions-root-gs2",
        type=Path,
        default=TRACKER_RUNS_GS2,
        help="Grounded-SAM-2 parity-recovery run dir (supplies B_gs2_fixed). Skipped if dir is missing or empty.",
    )
    parser.add_argument(
        "--predictions-root-gs2-strict",
        type=Path,
        default=TRACKER_RUNS_GS2_STRICT,
        help="Grounded-SAM-2 strict run dir (supplies B_gs2_strict). Skipped if dir is missing or empty.",
    )
    parser.add_argument("--manifest", type=Path, default=MANIFEST_CSV)
    parser.add_argument("--out-dir", type=Path, default=PREDICTIONS_MOT_DIR)


def add_subparser(subparsers) -> argparse.ArgumentParser:
    p = subparsers.add_parser(
        "convert-preds",
        help="Convert tracker prediction parquets (6-way ablation) to MOTChallenge .txt files.",
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

        yolo_pq = run_dir / "yolo_tracking.parquet"
        sam3_adaptive_pq = run_dir / "tracking_outputs.parquet"

        a_rows = yolo_to_mot_rows(yolo_pq) if yolo_pq.exists() else []
        e_rows = sam3_to_mot_rows(sam3_adaptive_pq) if sam3_adaptive_pq.exists() else []
        write_rows(args.out_dir / "A_yolo_botsort" / f"{video_id}.txt", a_rows)
        write_rows(args.out_dir / "E_sam3_adaptive" / f"{video_id}.txt", e_rows)

        n_b_strict = _convert_sam3_bucket(
            args, stem, video_id, "predictions_root_gs2_strict", "B_gs2_strict"
        )
        n_b_parity = _convert_sam3_bucket(
            args, stem, video_id, "predictions_root_gs2", "B_gs2_fixed"
        )
        n_c_strict = _convert_sam3_bucket(
            args, stem, video_id, "predictions_root_frame_zero", "C_sam3_frame_zero"
        )
        n_d = _convert_sam3_bucket(
            args, stem, video_id, "predictions_root_fixed", "D_sam3_fixed"
        )

        summary.append(
            (video_id, len(a_rows), n_b_strict, n_b_parity, n_c_strict, n_d, len(e_rows))
        )

    print()
    header = (
        f"{'video_id':<14} {'A':>10} {'B-strict':>10} {'B-parity':>10} "
        f"{'C-strict':>10} {'D':>10} {'E':>10}"
    )
    print(header)
    print("-" * len(header))
    for vid, na, nbs, nbp, ncs, nd, ne in summary:
        print(f"{vid:<14} {na:>10} {nbs:>10} {nbp:>10} {ncs:>10} {nd:>10} {ne:>10}")


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    _add_args(parser)
    run(parser.parse_args())


if __name__ == "__main__":
    _main()
