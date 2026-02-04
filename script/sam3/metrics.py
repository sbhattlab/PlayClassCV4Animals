"""
Tracking metrics computation for SAM3 video outputs.
"""

from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


def _tensor_to_numpy(x):
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    if isinstance(x, np.ndarray):
        return x
    try:
        return np.array(x)
    except Exception:
        return np.array([])


def _normalize_frame_dict(
    frame_dict: Dict[Any, Any],
) -> Tuple[List[int], List[List[Dict[str, Any]]]]:
    idxs = sorted(int(k) for k in frame_dict.keys())
    frames: List[List[Dict[str, Any]]] = []
    for i in idxs:
        v = frame_dict[i]
        if isinstance(v, list):
            frames.append(v)
            continue
        if isinstance(v, dict):
            if "detections" in v:
                frames.append(v["detections"] or [])
                continue
            if "instances" in v:
                frames.append(v["instances"] or [])
                continue
            if "object_ids" in v and "boxes" in v:
                obj_ids = _tensor_to_numpy(v["object_ids"])
                boxes = _tensor_to_numpy(v["boxes"])
                scores = _tensor_to_numpy(v.get("scores", np.zeros(len(obj_ids))))
                dets = []
                for j in range(len(obj_ids)):
                    try:
                        oid = int(obj_ids[j])
                    except Exception:
                        oid = int(np.asarray(obj_ids)[j])
                    bbox = (
                        boxes[j].tolist()
                        if hasattr(boxes[j], "tolist")
                        else list(np.asarray(boxes[j]))
                    )
                    score = float(scores[j]) if len(scores) > j else None
                    dets.append({"id": oid, "bbox": bbox, "score": score})
                frames.append(dets)
                continue
            if all(isinstance(k, (int, np.integer)) for k in v.keys()):
                vals = list(v.values())
                frames.append(vals if vals and isinstance(vals[0], dict) else [])
                continue
        frames.append([])
    return idxs, frames


def _iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    inter = interW * interH
    areaA = max(0, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    areaB = max(0, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0


def compute_per_id_metrics(
    frame_dict: Dict[Any, Any], low_count_threshold: int = 3, iou_thresh: float = 0.5
) -> Dict[Any, Dict[str, Any]]:
    """
    Returns: { id: {first_frame, last_frame, length, runs, gaps_total, frames,
                     coverage, low_count_frames, low_count_total, low_count_fraction,
                     mean_iou, mean_bbox_area (optional)} }
    """
    idxs, frames = _normalize_frame_dict(frame_dict)
    if not idxs:
        return {}

    counts = {idx: len(f) for idx, f in zip(idxs, frames)}
    low_frames_set = {idx for idx, c in counts.items() if c < low_count_threshold}

    # id -> frames list and bbox areas aggregator
    id_to_frames = defaultdict(list)
    id_bbox_areas = defaultdict(list)
    for idx, f in zip(idxs, frames):
        for det in f:
            if not isinstance(det, dict):
                continue
            uid = det.get("id")
            if uid is None:
                continue
            id_to_frames[uid].append(idx)
            if "bbox" in det:
                b = det["bbox"]
                area = max(0.0, (b[2] - b[0]) * (b[3] - b[1]))
                id_bbox_areas[uid].append(area)

    # compute runs/gaps per id
    def _runs_and_gaps(fl: List[int]) -> Tuple[int, int]:
        if not fl:
            return 0, 0
        fl_sorted = sorted(fl)
        runs = 1
        gaps = 0
        for a, b in zip(fl_sorted, fl_sorted[1:]):
            if b != a + 1:
                runs += 1
                gaps += b - a - 1
        return runs, gaps

    # compute per-id mean IoU by greedy matching across consecutive frames
    id_ious = defaultdict(list)
    for A, B in zip(frames, frames[1:]):
        used_b = set()
        for a in A:
            if not isinstance(a, dict) or "bbox" not in a:
                continue
            best_j, best_iou = None, 0.0
            for j, b in enumerate(B):
                if j in used_b or not isinstance(b, dict) or "bbox" not in b:
                    continue
                val = _iou(a["bbox"], b["bbox"])
                if val > best_iou:
                    best_iou, best_j = val, j
            if best_j is not None and best_iou >= iou_thresh:
                aid = a.get("id")
                id_ious[aid].append(best_iou)

    per_id = {}
    for uid, flist in id_to_frames.items():
        fl_sorted = sorted(flist)
        first, last = fl_sorted[0], fl_sorted[-1]
        length = len(fl_sorted)
        runs, gaps = _runs_and_gaps(fl_sorted)
        span = last - first + 1 if last >= first else 0
        coverage = length / span if span > 0 else 0.0
        low_in_span = [f for f in fl_sorted if f in low_frames_set]
        low_total = len(low_in_span)
        low_frac = low_total / span if span > 0 else 0.0
        mean_iou = float(np.mean(id_ious[uid])) if id_ious.get(uid) else None
        mean_area = (
            float(np.mean(id_bbox_areas[uid])) if id_bbox_areas.get(uid) else None
        )

        per_id[uid] = {
            "id": uid,
            "first_frame": int(first),
            "last_frame": int(last),
            "length": int(length),
            "runs": int(runs),
            "gaps_total": int(gaps),
            "frames": fl_sorted,
            "span": int(span),
            "coverage": float(coverage),
            "low_count_frames": low_in_span,
            "low_count_total": int(low_total),
            "low_count_fraction": float(low_frac),
            "mean_iou": mean_iou,
            "mean_bbox_area": mean_area,
        }
    return per_id


def compute_summary_metrics(
    frame_dict: Dict[Any, Any],
    persistence_k: int = 5,
    iou_match_thresh: float = 0.5,
    low_count_threshold: int = 3,
) -> Dict[str, Any]:
    """
    Returns a flat summary dict aggregating the usual tracking proxies.
    """
    idxs, frames = _normalize_frame_dict(frame_dict)
    n_frames = len(idxs)
    dets_per_frame = [len(f) for f in frames]
    counts = {idx: len(f) for idx, f in zip(idxs, frames)}
    low_frame_count = sum(1 for c in counts.values() if c < low_count_threshold)

    # build id->frames
    id_to_frames = defaultdict(list)
    for idx, f in zip(idxs, frames):
        for det in f:
            if not isinstance(det, dict):
                continue
            uid = det.get("id")
            if uid is None:
                continue
            id_to_frames[uid].append(idx)

    track_lengths = (
        np.array([len(v) for v in id_to_frames.values()])
        if id_to_frames
        else np.array([])
    )
    mean_track_length = float(track_lengths.mean()) if track_lengths.size else 0.0
    persistence_rate = float((track_lengths >= persistence_k).sum()) / (
        track_lengths.size or 1
    )

    # fragmentation
    def _count_runs(frames_list):
        if not frames_list:
            return 0
        fl = sorted(frames_list)
        runs = 1
        for a, b in zip(fl, fl[1:]):
            if b != a + 1:
                runs += 1
        return runs

    runs_per_id = (
        np.array([_count_runs(v) for v in id_to_frames.values()])
        if id_to_frames
        else np.array([])
    )
    mean_runs = float(runs_per_id.mean()) if runs_per_id.size else 0.0

    # continuity (fraction of detections that persist to next frame)
    transitions_possible = 0
    transitions_preserved = 0
    for f_a, f_b in zip(frames, frames[1:]):
        map_b = {d.get("id"): d for d in f_b if isinstance(d, dict) and "id" in d}
        for d in f_a:
            if not isinstance(d, dict):
                continue
            uid = d.get("id")
            if uid is None:
                continue
            transitions_possible += 1
            if uid in map_b:
                transitions_preserved += 1
    continuity = transitions_preserved / (transitions_possible or 1)

    # IoU-based id-switch proxy: greedy matching
    switches = 0
    matches = 0
    for A, B in zip(frames, frames[1:]):
        if not A or not B:
            continue
        used_b = set()
        for a in A:
            if not isinstance(a, dict) or "bbox" not in a:
                continue
            best_j, best_iou = None, 0.0
            for j, b in enumerate(B):
                if j in used_b or not isinstance(b, dict) or "bbox" not in b:
                    continue
                val = _iou(a["bbox"], b["bbox"])
                if val > best_iou:
                    best_iou, best_j = val, j
            if best_j is not None and best_iou >= iou_match_thresh:
                matches += 1
                used_b.add(best_j)
                if a.get("id") != B[best_j].get("id"):
                    switches += 1
    id_switch_rate = switches / (matches or 1)

    # aggregate per-id mean_iou and mean coverage
    per_id = compute_per_id_metrics(
        frame_dict, low_count_threshold=low_count_threshold, iou_thresh=iou_match_thresh
    )
    mean_coverage = (
        float(np.mean([v["coverage"] for v in per_id.values()])) if per_id else None
    )
    mean_per_id_iou = (
        float(
            np.mean(
                [v["mean_iou"] for v in per_id.values() if v["mean_iou"] is not None]
            )
        )
        if per_id
        else None
    )

    summary = {
        "n_frames": int(n_frames),
        "avg_detections_per_frame": float(np.mean(dets_per_frame))
        if dets_per_frame
        else 0.0,
        "total_unique_ids": int(len(id_to_frames)),
        "mean_track_length": mean_track_length,
        f"persistence_rate_>={persistence_k}": persistence_rate,
        "mean_fragmentation_runs_per_id": mean_runs,
        "continuity_fraction": float(continuity),
        "iou_match_count": int(matches),
        "id_switches": int(switches),
        "id_switch_rate": float(id_switch_rate),
        "low_frame_count": int(low_frame_count),
        "low_frame_fraction": float(low_frame_count / n_frames) if n_frames else 0.0,
        "mean_coverage_per_id": mean_coverage,
        "mean_per_id_iou": mean_per_id_iou,
    }
    return summary


def per_id_metrics_to_df(per_id_metrics: Dict[Any, Dict[str, Any]]) -> pd.DataFrame:
    """
    Converts compute_per_id_metrics output into a DataFrame (one row per id).
    """
    if not per_id_metrics:
        return pd.DataFrame()
    rows = []
    for uid, d in per_id_metrics.items():
        r = d.copy()
        # ensure id included as column
        r["id"] = uid
        rows.append(r)
    df = pd.DataFrame(rows)
    # sensible column ordering
    cols = [
        "id",
        "first_frame",
        "last_frame",
        "length",
        "span",
        "coverage",
        "runs",
        "gaps_total",
        "low_count_total",
        "low_count_fraction",
        "low_count_frames",
        "mean_iou",
        "mean_bbox_area",
        "frames",
    ]
    existing = [c for c in cols if c in df.columns]
    other = [c for c in df.columns if c not in existing]
    return df[existing + other]


def summary_metrics_to_df(summary_metrics: Dict[str, Any]) -> pd.DataFrame:
    """
    Converts compute_summary_metrics output into a single-row DataFrame.
    """
    if not summary_metrics:
        return pd.DataFrame()
    return pd.DataFrame([summary_metrics])


def compute_per_run_metrics(
    frame_dict: Dict[Any, Any], low_count_threshold: int = 3, iou_thresh: float = 0.5
) -> Dict[Any, List[Dict[str, Any]]]:
    """
    Return per-object runs:
      { object_id: [ { run_idx, first_frame, last_frame, length, frames,
                       low_count_frames, low_count_total, low_count_fraction,
                       mean_iou, mean_bbox_area, mean_score }, ... ] }
    A "run" = contiguous frames where the object appears.
    """
    idxs, frames = _normalize_frame_dict(frame_dict)
    if not idxs:
        return {}

    # per-frame detection counts -> low-frame set
    counts = {idx: len(f) for idx, f in zip(idxs, frames)}
    low_frames_set = {idx for idx, c in counts.items() if c < low_count_threshold}

    # build mapping: frame_idx -> {id: det_dict}
    frame_map = {
        idx: {d.get("id"): d for d in f if isinstance(d, dict) and "id" in d}
        for idx, f in zip(idxs, frames)
    }

    # id -> sorted list of frames where it appears
    id_to_frames = defaultdict(list)
    for idx, f in zip(idxs, frames):
        for det in f:
            if not isinstance(det, dict):
                continue
            uid = det.get("id")
            if uid is None:
                continue
            id_to_frames[uid].append(idx)

    per_run = {}
    for uid, flist in id_to_frames.items():
        fl_sorted = sorted(flist)
        runs = []
        run_idx = 0
        i = 0
        while i < len(fl_sorted):
            # start new run at fl_sorted[i]
            run_frames = [fl_sorted[i]]
            j = i + 1
            while j < len(fl_sorted) and fl_sorted[j] == run_frames[-1] + 1:
                run_frames.append(fl_sorted[j])
                j += 1
            # compute run-level stats
            first = run_frames[0]
            last = run_frames[-1]
            length = len(run_frames)
            low_in_run = [f for f in run_frames if f in low_frames_set]
            low_total = len(low_in_run)
            low_frac = low_total / length if length else 0.0

            # mean IoU for this id across consecutive frames inside the run
            ious = []
            areas = []
            scores = []
            for a, b in zip(run_frames, run_frames[1:]):
                da = frame_map.get(a, {}).get(uid)
                db = frame_map.get(b, {}).get(uid)
                if da and db and "bbox" in da and "bbox" in db:
                    ious.append(_iou(da["bbox"], db["bbox"]))
            # gather area/score across run
            for fidx in run_frames:
                d = frame_map.get(fidx, {}).get(uid)
                if d:
                    if "bbox" in d:
                        b = d["bbox"]
                        areas.append(max(0.0, (b[2] - b[0]) * (b[3] - b[1])))
                    if "score" in d and d["score"] is not None:
                        try:
                            scores.append(float(d["score"]))
                        except Exception:
                            pass

            mean_iou = float(np.mean(ious)) if ious else None
            mean_area = float(np.mean(areas)) if areas else None
            mean_score = float(np.mean(scores)) if scores else None

            runs.append(
                {
                    "run_idx": int(run_idx),
                    "first_frame": int(first),
                    "last_frame": int(last),
                    "length": int(length),
                    "frames": run_frames,
                    "low_count_frames": low_in_run,
                    "low_count_total": int(low_total),
                    "low_count_fraction": float(low_frac),
                    "mean_iou": mean_iou,
                    "mean_bbox_area": mean_area,
                    "mean_score": mean_score,
                }
            )

            run_idx += 1
            i = j
        per_run[uid] = runs
    return per_run


def per_run_metrics_to_multiindex_df(
    per_run_metrics: Dict[Any, List[Dict[str, Any]]],
) -> pd.DataFrame:
    """
    Flatten per-run metrics into a DataFrame indexed by MultiIndex (object_id, run_idx).
    Columns: first_frame, last_frame, length, frames, low_count_total, low_count_fraction, mean_iou, mean_bbox_area, mean_score, ...
    """
    rows = []
    for uid, runs in per_run_metrics.items():
        for r in runs:
            row = r.copy()
            row["id"] = uid
            rows.append(row)
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # ensure run_idx exists and is int
    if "run_idx" not in df.columns:
        df["run_idx"] = 0
    df["run_idx"] = df["run_idx"].astype(int)
    df["id"] = df["id"].astype(
        type(list(per_run_metrics.keys())[0]) if per_run_metrics else int
    )

    # set MultiIndex (object_id outer, run_idx inner)
    df = df.set_index(["id", "run_idx"]).sort_index()
    # order columns sensibly
    cols = [
        "first_frame",
        "last_frame",
        "length",
        "low_count_total",
        "low_count_fraction",
        "mean_iou",
        "mean_bbox_area",
        "mean_score",
        "frames",
        "low_count_frames",
    ]
    existing = [c for c in cols if c in df.columns]
    other = [c for c in df.columns if c not in existing]
    return df[existing + other]
