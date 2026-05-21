"""Sparse-keyframe MOT evaluation for the 6-way ablation.

Reads sparse GT (from `cvat_to_mot.py`) and dense predictions (from
`predictions.py`) for all 5 evaluation videos × 6 tracker variants
(A_yolo_botsort, B_gs2_strict, B_gs2_fixed, C_sam3_frame_zero,
D_sam3_fixed, E_sam3_adaptive) and emits:

    data/tracker_eval/results/metrics_per_video.csv
    data/tracker_eval/results/metrics_per_cage.csv
    data/tracker_eval/results/metrics_aggregate.csv

Two metric libraries are used:

- **TrackEval** (canonical HOTA implementation, Luiten et al. 2021) for
  HOTA / DetA / AssA / LocA / DetRe / DetPr / AssRe / AssPr / OWTA.
- **motmetrics** for IDF1 / IDP / IDR / MOTA / MOTP / IDsw / precision /
  recall / mostly-tracked / mostly-lost / fragmentations.

Matching: IoU ≥ 0.5 (MOTChallenge default) for the motmetrics side. HOTA
sweeps α ∈ {0.05, 0.10, ..., 0.95} internally and averages.

Sparse-eval pattern: predictions stay dense; the accumulator and TrackEval
data dict are built only over GT-present frames so we only score frames
that have human-verified ground truth.

Usage:
    pixi run -e tracker-evaluation python -m src.tracker_eval evaluate \
        --gt-dir ext-data/tracker_benchmark/ground_truth \
        --predictions-mot-dir ext-data/tracker_benchmark/predictions_mot \
        --manifest data/tracker_eval/video_manifest.csv \
        --out-dir data/tracker_eval/results
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

from .paths import (
    GROUND_TRUTH_DIR,
    MANIFEST_CSV,
    PREDICTIONS_MOT_DIR,
    RESULTS_DIR,
    TRACKEVAL_DIR,
)

mm = None
np = None
HOTA = None


def _ensure_eval_deps() -> None:
    """Import evaluation-only dependencies lazily.

    The `prepare` command runs in the tracker pixi env, which does not include
    motmetrics. Keeping these imports lazy lets the shared CLI dispatcher
    register pre-CVAT commands without requiring post-CVAT dependencies.
    """
    global HOTA, mm, np
    if mm is not None and np is not None and HOTA is not None:
        return

    import motmetrics as _mm
    import numpy as _np

    # TrackEval is vendored as a git submodule at ext/TrackEval, not pip-installed.
    # Insert it on sys.path so `import trackeval` resolves to the submodule.
    if str(TRACKEVAL_DIR) not in sys.path:
        sys.path.insert(0, str(TRACKEVAL_DIR))

    # TrackEval (last upstream change ~2021) uses deprecated numpy aliases
    # np.float / np.int / np.bool / np.object, removed in numpy 1.24+. Restore as
    # builtin aliases before TrackEval is imported.
    for _name, _alias in (
        ("float", float),
        ("int", int),
        ("bool", bool),
        ("object", object),
    ):
        if not hasattr(_np, _name):
            setattr(_np, _name, _alias)

    from trackeval.metrics.hota import HOTA as _HOTA

    mm = _mm
    np = _np
    HOTA = _HOTA

VARIANTS = (
    "A_yolo_botsort",
    "B_gs2_strict",
    "B_gs2_fixed",
    "C_sam3_frame_zero",
    "D_sam3_fixed",
    "E_sam3_adaptive",
)

# motmetrics fields. (HOTA-family fields come from TrackEval.)
MOTMETRICS_FIELDS = [
    "num_frames",
    "num_unique_objects",
    "num_objects",
    "mota",
    "motp",
    "idf1",
    "idp",
    "idr",
    "precision",
    "recall",
    "num_switches",
    "num_false_positives",
    "num_misses",
    "mostly_tracked",
    "partially_tracked",
    "mostly_lost",
    "num_fragmentations",
]

# TrackEval HOTA summary fields we keep in the output. HOTA itself is the
# headline metric; DetA/AssA/LocA decompose detection vs association vs
# localisation; the others are precision/recall sub-components.
HOTA_SUMMARY_FIELDS = [
    "HOTA",
    "DetA",
    "AssA",
    "LocA",
    "DetRe",
    "DetPr",
    "AssRe",
    "AssPr",
    "OWTA",
]

MOT_COLS = ["FrameId", "Id", "X", "Y", "W", "H", "Conf", "_a", "_b", "_c"]


def load_mot(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, names=MOT_COLS, header=None)
    return df[["FrameId", "Id", "X", "Y", "W", "H"]]


def selected_videos(manifest_csv: Path) -> list[tuple[str, str]]:
    """Return [(video_id, cage), ...] for selected rows."""
    df = pd.read_csv(manifest_csv)
    selected = df[df["selected"].astype(str).str.lower().isin({"true", "1"})]
    return list(zip(selected["video_id"], selected["cage"]))


def iou_xywh(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """IoU between two arrays of [x, y, w, h] bboxes; returns NxM matrix."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)))
    a = np.asarray(boxes_a, dtype=float).copy()
    b = np.asarray(boxes_b, dtype=float).copy()
    a_x2 = a[:, 0] + a[:, 2]
    a_y2 = a[:, 1] + a[:, 3]
    b_x2 = b[:, 0] + b[:, 2]
    b_y2 = b[:, 1] + b[:, 3]
    xx1 = np.maximum(a[:, None, 0], b[None, :, 0])
    yy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    xx2 = np.minimum(a_x2[:, None], b_x2[None, :])
    yy2 = np.minimum(a_y2[:, None], b_y2[None, :])
    inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
    area_a = a[:, 2] * a[:, 3]
    area_b = b[:, 2] * b[:, 3]
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def build_motmetrics_accumulator(
    gt: pd.DataFrame, pred: pd.DataFrame
) -> mm.MOTAccumulator:
    """motmetrics accumulator at IoU ≥ 0.5, over GT-present frames."""
    acc = mm.MOTAccumulator(auto_id=False)
    gt_by_frame = {fr: g for fr, g in gt.groupby("FrameId")}
    pred_by_frame = {fr: p for fr, p in pred.groupby("FrameId")}
    for fr in sorted(gt_by_frame):
        g = gt_by_frame[fr]
        p = pred_by_frame.get(fr)
        gt_ids = g["Id"].astype(int).tolist()
        gt_boxes = g[["X", "Y", "W", "H"]].values
        if p is not None and not p.empty:
            pred_ids = p["Id"].astype(int).tolist()
            pred_boxes = p[["X", "Y", "W", "H"]].values
            dists = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=0.5)
        else:
            pred_ids = []
            dists = mm.distances.iou_matrix(gt_boxes, [], max_iou=0.5)
        acc.update(gt_ids, pred_ids, dists, frameid=int(fr))
    return acc


def build_trackeval_data(gt: pd.DataFrame, pred: pd.DataFrame) -> dict:
    """Build the per-sequence dict TrackEval HOTA expects.

    Only GT-present frames are included (sparse-eval convention). IDs are
    re-indexed to dense 0..N-1 ranges as TrackEval requires.
    """
    gt_frames = sorted(gt["FrameId"].unique())
    gt_by_frame = {fr: g for fr, g in gt.groupby("FrameId")}
    pred_by_frame = {fr: p for fr, p in pred.groupby("FrameId")}

    gt_ids_sorted = sorted(gt["Id"].astype(int).unique())
    pred_ids_sorted = sorted(pred["Id"].astype(int).unique())
    gt_id_map = {gid: i for i, gid in enumerate(gt_ids_sorted)}
    pred_id_map = {pid: i for i, pid in enumerate(pred_ids_sorted)}

    gt_ids_per_t: list[np.ndarray] = []
    tracker_ids_per_t: list[np.ndarray] = []
    similarity_per_t: list[np.ndarray] = []
    num_gt_dets = 0
    num_tracker_dets = 0

    for fr in gt_frames:
        g = gt_by_frame[fr]
        gt_arr = np.array([gt_id_map[int(x)] for x in g["Id"]], dtype=int)
        gt_boxes = g[["X", "Y", "W", "H"]].values.astype(float)
        p = pred_by_frame.get(fr)
        if p is not None and not p.empty:
            pred_arr = np.array([pred_id_map[int(x)] for x in p["Id"]], dtype=int)
            pred_boxes = p[["X", "Y", "W", "H"]].values.astype(float)
            sim = iou_xywh(gt_boxes, pred_boxes)
        else:
            pred_arr = np.array([], dtype=int)
            sim = np.zeros((len(gt_arr), 0))
        gt_ids_per_t.append(gt_arr)
        tracker_ids_per_t.append(pred_arr)
        similarity_per_t.append(sim)
        num_gt_dets += len(gt_arr)
        num_tracker_dets += len(pred_arr)

    return {
        "num_timesteps": len(gt_frames),
        "num_gt_ids": len(gt_ids_sorted),
        "num_tracker_ids": len(pred_ids_sorted),
        "num_gt_dets": num_gt_dets,
        "num_tracker_dets": num_tracker_dets,
        "gt_ids": gt_ids_per_t,
        "tracker_ids": tracker_ids_per_t,
        "similarity_scores": similarity_per_t,
    }


def hota_summary(res: dict) -> dict[str, float]:
    """Reduce HOTA per-α arrays to a single mean (per the standard convention)."""
    out: dict[str, float] = {}
    for field in HOTA_SUMMARY_FIELDS:
        v = res[field]
        out[field] = float(np.mean(v)) if isinstance(v, np.ndarray) else float(v)
    return out


def _add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--gt-dir", type=Path, default=GROUND_TRUTH_DIR)
    parser.add_argument("--predictions-mot-dir", type=Path, default=PREDICTIONS_MOT_DIR)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_CSV)
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)


def add_subparser(subparsers) -> argparse.ArgumentParser:
    p = subparsers.add_parser(
        "evaluate",
        help="Sparse-keyframe MOT evaluation (HOTA via TrackEval; IDF1/MOTA via motmetrics).",
    )
    _add_args(p)
    p.set_defaults(func=run)
    return p


def run(args: argparse.Namespace) -> None:
    _ensure_eval_deps()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    videos = selected_videos(args.manifest)

    # Skip variants whose predictions are not (yet) on disk for every video.
    variants = [
        v
        for v in VARIANTS
        if all(
            (args.predictions_mot_dir / v / f"{vid}.txt").exists() for vid, _ in videos
        )
    ]
    for v in VARIANTS:
        if v not in variants:
            print(f"[skip] {v}: predictions incomplete under {args.predictions_mot_dir / v}")
    print(
        f"{len(videos)} videos x {len(variants)} variants = {len(videos) * len(variants)} runs"
    )

    mh = mm.metrics.create()
    hota_metric = HOTA()

    accs: dict[str, list[mm.MOTAccumulator]] = {v: [] for v in variants}
    names: dict[str, list[str]] = {v: [] for v in variants}
    hota_per_seq: dict[str, dict[str, dict]] = {
        v: {} for v in variants
    }  # variant -> {video_id -> raw HOTA result dict}
    cage_by_video: dict[str, str] = {}

    for video_id, cage in videos:
        cage_by_video[video_id] = cage
        gt_path = args.gt_dir / f"{video_id}.txt"
        gt = load_mot(gt_path)
        for variant in variants:
            pred_path = args.predictions_mot_dir / variant / f"{video_id}.txt"
            pred = load_mot(pred_path)
            accs[variant].append(build_motmetrics_accumulator(gt, pred))
            names[variant].append(video_id)
            te_data = build_trackeval_data(gt, pred)
            hota_per_seq[variant][video_id] = hota_metric.eval_sequence(te_data)
            print(f"  built: {variant} / {video_id}")

    # === per-video ===
    rows_per_video: list[dict] = []
    for variant in variants:
        mm_df = mh.compute_many(
            accs[variant], names=names[variant], metrics=MOTMETRICS_FIELDS
        )
        for video_id in mm_df.index:
            row = mm_df.loc[video_id].to_dict()
            row["variant"] = variant
            row["video_id"] = video_id
            row["cage"] = cage_by_video[video_id]
            row.update(hota_summary(hota_per_seq[variant][video_id]))
            rows_per_video.append(row)
    per_video = pd.DataFrame(rows_per_video)
    front = ["variant", "video_id", "cage", "HOTA", "DetA", "AssA", "LocA"]
    per_video = per_video[front + [c for c in per_video.columns if c not in front]]
    per_video.to_csv(args.out_dir / "metrics_per_video.csv", index=False)

    # === per-cage === (one video per cage in this study, but generalised)
    rows_per_cage: list[dict] = []
    for variant in variants:
        groups: dict[str, list[mm.MOTAccumulator]] = defaultdict(list)
        names_in_cage: dict[str, list[str]] = defaultdict(list)
        seqs_in_cage: dict[str, list[dict]] = defaultdict(list)
        for acc, vid in zip(accs[variant], names[variant]):
            c = cage_by_video[vid]
            groups[c].append(acc)
            names_in_cage[c].append(vid)
            seqs_in_cage[c].append(hota_per_seq[variant][vid])
        for cage in sorted(groups):
            mm_df = mh.compute_many(
                groups[cage],
                names=names_in_cage[cage],
                metrics=MOTMETRICS_FIELDS,
                generate_overall=True,
            )
            overall = mm_df.loc["OVERALL"].to_dict()
            overall["variant"] = variant
            overall["cage"] = cage
            cage_hota_res = hota_metric.combine_sequences(
                {vid: r for vid, r in zip(names_in_cage[cage], seqs_in_cage[cage])}
            )
            overall.update(hota_summary(cage_hota_res))
            rows_per_cage.append(overall)
    per_cage = pd.DataFrame(rows_per_cage)
    front = ["variant", "cage", "HOTA", "DetA", "AssA", "LocA"]
    per_cage = per_cage[front + [c for c in per_cage.columns if c not in front]]
    per_cage.to_csv(args.out_dir / "metrics_per_cage.csv", index=False)

    # === aggregate ===
    rows_agg: list[dict] = []
    for variant in variants:
        mm_df = mh.compute_many(
            accs[variant],
            names=names[variant],
            metrics=MOTMETRICS_FIELDS,
            generate_overall=True,
        )
        overall = mm_df.loc["OVERALL"].to_dict()
        overall["variant"] = variant
        agg_hota_res = hota_metric.combine_sequences(
            {vid: r for vid, r in hota_per_seq[variant].items()}
        )
        overall.update(hota_summary(agg_hota_res))
        rows_agg.append(overall)
    aggregate = pd.DataFrame(rows_agg)
    front = ["variant", "HOTA", "DetA", "AssA", "LocA"]
    aggregate = aggregate[front + [c for c in aggregate.columns if c not in front]]
    aggregate.to_csv(args.out_dir / "metrics_aggregate.csv", index=False)

    # Headline
    print()
    print("=" * 80)
    print("AGGREGATE (5 videos, sparse GT)")
    print("=" * 80)
    headline = ["variant", "HOTA", "DetA", "AssA", "idf1", "mota", "num_switches"]
    print(aggregate[headline].to_string(index=False))


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    _add_args(parser)
    run(parser.parse_args())


if __name__ == "__main__":
    _main()
