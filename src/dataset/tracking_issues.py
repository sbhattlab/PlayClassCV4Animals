"""Tracking issue detection: ID switches, mask overlaps, and remediation."""

from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pycocotools.mask as mask_util
from loguru import logger

from .utils import fmt_time


def get_video_fps(tracking_dir):
    """Read video FPS from yolo_scan_summary.parquet in the tracking dir. Fall back to 25.0."""
    tracking_dir = Path(tracking_dir)
    summary_path = tracking_dir / "metrics" / "yolo_scan_summary.parquet"
    if summary_path.exists():
        summary = pd.read_parquet(summary_path)
        if "fps" in summary.columns and len(summary) > 0:
            return float(summary["fps"].iloc[0])
    logger.warning(f"Could not read FPS from {summary_path}, falling back to 25.0")
    return 25.0


def detect_id_switches(tracks, fps):
    """Find frames where the set of tracked IDs changes.

    Returns a dict of transition events keyed by "transition0", "transition1", etc.
    Each event has: frame, time, ids_before, ids_after, lost, gained.
    """
    ids_per_frame = tracks.groupby("frame_idx").apply(
        lambda g: set(g.index.get_level_values("object_id"))
    )

    transitions = []
    for (prev_frame, prev_ids), (curr_frame, curr_ids) in zip(
        ids_per_frame.items(), ids_per_frame.iloc[1:].items()
    ):
        if prev_ids != curr_ids:
            transitions.append((prev_frame, curr_frame, prev_ids, curr_ids))

    transition_events = {}
    for i, (prev_frame, curr_frame, prev_ids, curr_ids) in enumerate(transitions):
        lost = prev_ids - curr_ids
        gained = curr_ids - prev_ids
        transition_events[f"transition{i}"] = {
            "frame": int(curr_frame),
            "time": round(curr_frame / fps, 2),
            "ids_before": sorted(prev_ids),
            "ids_after": sorted(curr_ids),
            "lost": sorted(lost),
            "gained": sorted(gained),
        }

    return transition_events


def detect_overlaps(tracks, fps, iou_threshold=0.7, merge_gap_seconds=1.0):
    """Find frames where different IDs have overlapping masks (double-segmentation).

    Computes pairwise mask IoU per frame, groups by ID pair, and merges
    consecutive overlap blocks within ``merge_gap_seconds`` of each other.

    Returns a dict of overlap events keyed by "overlap0", "overlap1", etc.
    Each event has: start, end, start_time, end_time, ids.
    """
    merge_gap_frames = int(merge_gap_seconds * fps)

    overlaps = []
    for frame_idx, group in tracks.groupby("frame_idx"):
        if len(group) < 2:
            continue

        obj_ids = group.index.get_level_values("object_id").tolist()
        rles = []
        for _, row in group.iterrows():
            rle = {"counts": row["counts"], "size": row["size"]}
            if isinstance(rle["counts"], str):
                rle["counts"] = rle["counts"].encode("utf-8")
            rles.append(rle)

        iou_matrix = mask_util.iou(rles, rles, [0] * len(rles))

        for (i, id_a), (j, id_b) in combinations(enumerate(obj_ids), 2):
            iou_val = iou_matrix[i][j]
            if iou_val >= iou_threshold:
                overlaps.append((frame_idx, id_a, id_b, iou_val))

    # Group per ID pair, merge blocks within gap
    pair_frames = defaultdict(list)
    for frame_idx, id_a, id_b, iou_val in overlaps:
        key = (min(id_a, id_b), max(id_a, id_b))
        pair_frames[key].append((frame_idx, iou_val))

    events = {}
    event_idx = 0

    for (id_a, id_b), frames_ious in sorted(pair_frames.items()):
        frames_ious.sort()

        # Build consecutive blocks
        blocks = []
        blk_start, blk_end = frames_ious[0][0], frames_ious[0][0]
        for frame, _ in frames_ious[1:]:
            if frame <= blk_end + 1:
                blk_end = frame
            else:
                blocks.append((blk_start, blk_end))
                blk_start = blk_end = frame
        blocks.append((blk_start, blk_end))

        # Merge blocks within merge_gap_frames of each other
        merged = [blocks[0]]
        for start, end in blocks[1:]:
            if start - merged[-1][1] <= merge_gap_frames:
                merged[-1] = (merged[-1][0], end)
            else:
                merged.append((start, end))

        for start, end in merged:
            events[f"overlap{event_idx}"] = {
                "start": int(start),
                "end": int(end),
                "start_time": round(start / fps, 2),
                "end_time": round(end / fps, 2),
                "ids": [id_a, id_b],
            }
            event_idx += 1

    return events


def detect_tracking_issues(tracks, fps, iou_threshold=0.7, merge_gap_seconds=1.0):
    """Detect transitions and overlaps, merge into a unified tracking_issues dict.

    Each entry has a ``type`` field ("id_switch" or "overlap") and is sorted
    by start frame.
    """
    id_switch_events = detect_id_switches(tracks, fps)
    overlap_events = detect_overlaps(
        tracks, fps, iou_threshold=iou_threshold, merge_gap_seconds=merge_gap_seconds
    )

    tracking_issues = {}

    for name, ev in overlap_events.items():
        tracking_issues[name] = {
            "type": "overlap",
            "start": ev["start"],
            "end": ev["end"],
            "start_time": ev["start_time"],
            "end_time": ev["end_time"],
            "ids": ev["ids"],
        }

    for name, ev in id_switch_events.items():
        tracking_issues[name] = {
            "type": "id_switch",
            "start": ev["frame"],
            "end": ev["frame"],
            "start_time": ev["time"],
            "end_time": ev["time"],
            "ids_before": ev["ids_before"],
            "ids_after": ev["ids_after"],
            "lost": ev["lost"],
            "gained": ev["gained"],
        }

    # Sort by start frame
    tracking_issues = dict(
        sorted(tracking_issues.items(), key=lambda kv: kv[1]["start"])
    )

    return tracking_issues


def prefill_id_remaps(tracking_issues, fps=25.0):
    """Build an ``id_remaps`` template from detected ID switch events.

    For each gained ID in a switch event, creates an entry with ``"to"``
    set to ``null`` for the user to fill in.

    Parameters
    ----------
    tracking_issues : dict
        Output of :func:`detect_tracking_issues`.
    fps : float
        Video frame rate (used for the ``_time`` hint).

    Returns
    -------
    list[dict]
        List of remap entries ready to be written to ``id_remaps.json``.
        Each has ``frame``, ``from``, ``to`` (null), and ``_time`` (hint).
    """
    remaps = []
    for ev in tracking_issues.values():
        if ev["type"] != "id_switch":
            continue
        for gained_id in ev.get("gained", []):
            remaps.append(
                {
                    "frame": ev["start"],
                    "from": gained_id,
                    "to": None,
                    "_time": fmt_time(ev["start"], fps),
                }
            )
    return remaps


def remove_overlaps(tracks, issues, labels):
    overlap_events = [v for v in issues.values() if v["type"] == "overlap"]

    # 1. tracks: drop rows for the overlapping IDs within the overlap frame range
    frame_idx = tracks.index.get_level_values("frame_idx")
    object_id = tracks.index.get_level_values("object_id")

    tracks_mask = np.zeros(len(tracks), dtype=bool)
    for ev in overlap_events:
        tracks_mask |= (
            (frame_idx >= ev["start"])
            & (frame_idx <= ev["end"])
            & np.isin(object_id, ev["ids"])
        )

    tracks_dropped = tracks_mask.sum()
    tracks = tracks[~tracks_mask]

    # 2. labels: drop all birds' rows whose 5s window [t-5, t] intersects any overlap period
    #    (no bird_id <-> object_id mapping yet, so we must drop all birds)
    labels_mask = pd.Series(False, index=labels.index)
    for ev in overlap_events:
        # window [t-5, t] intersects [start, end] when t > start and t-5 < end
        labels_mask |= (labels["time"] > ev["start_time"]) & (
            labels["time"] - 5 < ev["end_time"]
        )

    labels_dropped = labels_mask.sum()
    labels = labels[~labels_mask]

    print(f"tracks: dropped {tracks_dropped} rows -> {len(tracks)} remaining")
    print(f"labels: dropped {labels_dropped} rows -> {len(labels)} remaining")
    for ev in overlap_events:
        dur = ev["end_time"] - ev["start_time"]
        print(
            f"  IDs {ev['ids']}: {fmt_time(ev['start'])} - {fmt_time(ev['end'])} ({dur:.1f}s)"
        )

    return tracks, labels


def merge_id_on_switch(tracks, id_remaps):
    """Remap object IDs across switch events to produce continuous per-bird tracks.

    Parameters
    ----------
    tracks : pd.DataFrame
        Tracking outputs with MultiIndex ``["frame_idx", "object_id"]``.
    id_remaps : list[dict]
        Each entry has ``"frame"`` (int), ``"from"`` (int), ``"to"`` (int).
        All rows with ``object_id == from`` at ``frame_idx >= frame`` are
        remapped to ``to``.  Remaps are applied in order, so later entries
        can build on earlier ones.

        Example::

            [
                {"frame": 1547, "from": 1, "to": 4},
                {"frame": 3200, "from": 6, "to": 0}
            ]

    Returns
    -------
    pd.DataFrame
        Tracks with the same columns but updated ``object_id`` index.
    """
    if not id_remaps:
        return tracks

    df = tracks.reset_index()

    for remap in id_remaps:
        frame = remap["frame"]
        id_from = remap["from"]
        id_to = remap["to"]

        mask = (df["frame_idx"] >= frame) & (df["object_id"] == id_from)
        n = mask.sum()
        if n == 0:
            logger.warning(
                f"ID remap {id_from}→{id_to} at frame {frame}: no matching rows"
            )
            continue
        df.loc[mask, "object_id"] = id_to
        logger.info(
            f"Remapped {n} rows: object_id {id_from}→{id_to} from frame {frame}"
        )

    df = df.set_index(["frame_idx", "object_id"]).sort_index()
    return df
