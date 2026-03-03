"""Tracking issue detection: ID switches, mask overlaps."""

from collections import defaultdict
from itertools import combinations

import pycocotools.mask as mask_util


def detect_id_switches(tracks, fps):
    """Find frames where the set of tracked IDs changes.

    Returns a dict of transition events keyed by "transition0", "transition1", etc.
    Each event has: frame, time, ids_before, ids_after, lost, gained.
    """
    ids_per_frame = tracks.groupby("frame_idx")["tracking_id"].apply(set)

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

        obj_ids = group["tracking_id"].tolist()
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


def detect_low_score_periods(
    tracks, fps, score_threshold=0.8, min_duration_seconds=1.0, merge_gap_seconds=1.0
):
    """Find per-ID stretches where tracker_score stays below *score_threshold*.

    Only periods lasting at least *min_duration_seconds* are reported.
    Consecutive low-score blocks within *merge_gap_seconds* are merged.

    Returns a dict keyed by ``"low_score0"``, ``"low_score1"``, etc.
    """
    merge_gap_frames = int(merge_gap_seconds * fps)
    min_frames = int(min_duration_seconds * fps)

    events = {}
    idx = 0

    for obj_id, grp in tracks.groupby("tracking_id"):
        low = grp.query("tracker_score < @score_threshold")
        if low.empty:
            continue

        frames = sorted(low["frame_idx"])

        # Build consecutive blocks
        blocks = []
        blk_start = blk_end = frames[0]
        for f in frames[1:]:
            if f <= blk_end + 1:
                blk_end = f
            else:
                blocks.append((blk_start, blk_end))
                blk_start = blk_end = f
        blocks.append((blk_start, blk_end))

        # Merge blocks within gap
        merged = [blocks[0]]
        for start, end in blocks[1:]:
            if start - merged[-1][1] <= merge_gap_frames:
                merged[-1] = (merged[-1][0], end)
            else:
                merged.append((start, end))

        for start, end in merged:
            if end - start + 1 < min_frames:
                continue
            n_low = sum(1 for f in frames if start <= f <= end)
            in_range = grp.query("@start <= frame_idx <= @end")
            events[f"low_score{idx}"] = {
                "start": int(start),
                "end": int(end),
                "start_time": round(start / fps, 2),
                "end_time": round(end / fps, 2),
                "id": int(obj_id),
                "low_score_rows": n_low,
                "total_rows": len(in_range),
                "min_score": round(float(in_range["tracker_score"].min()), 3),
            }
            idx += 1

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
    low_score_events = detect_low_score_periods(
        tracks, fps, merge_gap_seconds=merge_gap_seconds
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

    for name, ev in low_score_events.items():
        tracking_issues[name] = {
            "type": "low_score",
            **ev,
        }

    # Sort by start frame
    tracking_issues = dict(
        sorted(tracking_issues.items(), key=lambda kv: kv[1]["start"])
    )

    return tracking_issues
