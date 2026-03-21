"""Tracking issue detection: remediation."""

import numpy as np
import pandas as pd
from loguru import logger

from src._config import DEFAULT_MIN_WINDOW_COVERAGE

from .utils import fmt_time


def _worst_id_in_range(tracks, ids, start, end):
    """Return the ID with the lowest mean tracker_score in [start, end]."""
    in_range = tracks.query("@start <= frame_idx <= @end")

    worst_id, worst_score = ids[0], float("inf")
    for oid in ids:
        subset = in_range.query("tracking_id == @oid")
        if subset.empty:
            continue
        mean_score = subset["tracker_score"].mean()
        if mean_score < worst_score:
            worst_id, worst_score = oid, mean_score
    return worst_id


def prefill_postprocessing(tracking_issues, tracks, video_birds, fps=25.0):
    """Build a ``tracking_postprocessing.json`` template from detected issues.

    Generates one entry per detected issue:

    - **overlap** → ``{"type": "trim", "cause": "overlap", ...}``
      The ``id`` is set to the ID with the lower mean tracker_score in the
      overlap range (user can swap).
    - **low_score** → ``{"type": "trim", "cause": "low_score", ...}``
    - **id_switch** → ``{"type": "id_switch", "to": null, ...}`` (user fills in)

    The user can review, adjust, or remove entries before re-running.

    Parameters
    ----------
    tracking_issues : dict
        Output of :func:`detect_tracking_issues`.
    tracks : pd.DataFrame
        Tracking outputs with ``tracking_id`` column.
    fps : float
        Video frame rate (used for the ``_time`` hint).

    Returns
    -------
    list[dict]
        List of postprocessing entries ready to be written to
        ``tracking_postprocessing.json``.
    """
    entries = []
    for ev in tracking_issues.values():
        if ev["type"] == "overlap":
            entries.append(
                {
                    "type": "trim",
                    "cause": "overlap",
                    "from": ev["start"],
                    "to": ev["end"],
                    "id": _worst_id_in_range(tracks, ev["ids"], ev["start"], ev["end"]),
                    "_time": (
                        f"{fmt_time(ev['start'], fps)}-{fmt_time(ev['end'], fps)}"
                    ),
                }
            )
        elif ev["type"] == "low_score":
            entries.append(
                {
                    "type": "trim",
                    "cause": "low_score",
                    "from": ev["start"],
                    "to": ev["end"],
                    "id": ev["id"],
                    "_time": (
                        f"{fmt_time(ev['start'], fps)}-{fmt_time(ev['end'], fps)}"
                    ),
                }
            )
        elif ev["type"] == "id_switch":
            for gained_id in ev.get("gained", []):
                entries.append(
                    {
                        "type": "id_switch",
                        "frame": ev["start"],
                        "from": None,
                        "to": gained_id,
                        "_time": fmt_time(ev["start"], fps),
                    }
                )
    for protocol_id, description in video_birds.items():
        entries.append(
            {
                "type": "id_match",
                "protocol_id": protocol_id,
                "description": description,
                "tracking_id": None,
                "frame": None,
            }
        )
    return entries


def merge_id_on_switch(tracks, id_remaps):
    """Remap tracking IDs across switch events to produce continuous per-bird tracks.

    Each entry says: "at ``frame``, the chicken that **was** ``from`` gets
    reassigned to ``to``."  Rows with ``tracking_id == from`` **before** the
    switch frame are renamed to ``to``.  Rows at or after the switch keep
    their current ID (a different chicken may now occupy the ``from`` slot).

    Remaps at the same frame are applied simultaneously (masks computed before
    any renames) so that swaps (e.g. A→B and B→C at the same frame) work
    correctly.  Across different frames, groups are applied earliest-first.

    Parameters
    ----------
    tracks : pd.DataFrame
        Tracking outputs with ``tracking_id`` column.
    id_remaps : list[dict]
        Each entry has ``"frame"`` (int), ``"from"`` (int), ``"to"`` (int).

    Returns
    -------
    pd.DataFrame
        Tracks with the same columns but updated ``tracking_id`` values.
    """
    if not id_remaps:
        return tracks

    # Group remaps by frame so same-frame swaps are applied simultaneously
    from collections import defaultdict

    by_frame = defaultdict(list)
    for remap in id_remaps:
        by_frame[remap["frame"]].append(remap)

    for frame in sorted(by_frame):
        # Compute all masks before any renames
        planned = []
        for remap in by_frame[frame]:
            id_from = remap["from"]
            id_to = remap["to"]
            mask = (tracks["frame_idx"] < frame) & (tracks["tracking_id"] == id_from)
            n = mask.sum()
            if n == 0:
                logger.warning(
                    f"ID remap {id_from}→{id_to} (switch at frame {frame}): "
                    "no matching rows before switch frame"
                )
                continue
            planned.append((id_from, id_to, mask, n))

        # Apply all renames for this frame
        for id_from, id_to, mask, n in planned:
            tracks.loc[mask, "tracking_id"] = id_to
            logger.info(
                f"Remapped {n} rows: tracking_id {id_from}→{id_to} "
                f"(before frame {frame})"
            )

    return tracks


def trim(tracks, labels, trims, fps, video_id=""):
    """Drop track rows within each trim's ``[from, to]`` frame range.

    If ``id`` is present, only rows for that tracking ID are dropped.
    Otherwise all rows in the range are dropped (global trim) and
    labels whose time falls within the range are also dropped (scoped
    to the current ``video_id``).
    """
    if not trims:
        return tracks, labels

    tracks_drop = pd.Series(False, index=tracks.index)
    labels_drop = pd.Series(False, index=labels.index)
    video_mask = labels["video_id"] == video_id if video_id else True

    for t in trims:
        f_from, f_to = t["from"], t["to"]
        t_from, t_to = f_from / fps, f_to / fps
        cause = t.get("cause", "")
        tag = f" ({cause})" if cause else ""

        in_range = tracks["frame_idx"].between(f_from, f_to)
        if "id" in t:
            in_range = in_range & (tracks["tracking_id"] == t["id"])
            tag += f" ID {t['id']}"
        else:
            labels_drop |= video_mask & labels["time"].between(t_from, t_to)

        tracks_drop |= in_range

        logger.info(
            f"Trim{tag} frames {f_from}-{f_to} "
            f"({fmt_time(f_from, fps)}-{fmt_time(f_to, fps)})"
        )

    n_t, n_l = tracks_drop.sum(), labels_drop.sum()
    tracks = tracks[~tracks_drop]
    labels = labels[~labels_drop]
    logger.info(f"Trims: dropped {n_t} track rows, {n_l} label rows")

    return tracks, labels


def match_bird_ids(tracks, id_matches):
    """Rename tracking IDs to protocol (bird) IDs using verified id_match entries.

    Runs **after** ``merge_id_on_switch``, so the ``"tracking_id"`` field in
    each match entry must reference the post-merge surviving IDs, not the
    original tracker-assigned IDs.

    Renames the ``tracking_id`` column to ``bird_id`` in the output.

    Parameters
    ----------
    tracks : pd.DataFrame
        Tracking outputs with ``tracking_id`` column.
    id_matches : list[dict]
        Each entry has ``"protocol_id"`` (int/str), ``"tracking_id"``
        (int/str — the **post-merge** tracking ID), and ``"frame"`` (int,
        reference frame for verification).

    Returns
    -------
    pd.DataFrame
        Tracks with ``bird_id`` column (renamed from ``tracking_id``,
        values updated to protocol IDs where matched).
    """
    for match in id_matches:
        tid = int(match["tracking_id"])
        protocol_id = int(match["protocol_id"])
        frame = match.get("frame", None)

        # Verify tracking_id exists (optionally at a specific frame)
        if frame is not None:
            at_frame = tracks.query("frame_idx == @frame and tracking_id == @tid")
            if at_frame.empty:
                logger.warning(
                    f"ID match: tracking_id {tid} not found at frame {frame} "
                    f"(target protocol_id {protocol_id})"
                )
                continue

        mask = tracks["tracking_id"] == tid
        n = mask.sum()
        tracks.loc[mask, "tracking_id"] = protocol_id
        logger.info(
            f"Matched {n} rows: tracking_id {tid} → bird_id {protocol_id} "
            f"(verified at frame {frame})"
        )

    tracks = tracks.rename(columns={"tracking_id": "bird_id"})
    return tracks


def check_postprocessing(entries):
    """Validate that all user-editable fields in postprocessing entries are filled in.

    Raises ``ValueError`` if any required fields are null.
    """
    problems = []
    for i, e in enumerate(entries):
        etype = e.get("type")
        if etype == "id_switch" and e.get("to") is None:
            problems.append(
                f"  entry {i}: id_switch at frame {e.get('frame')} — 'to' is null"
            )
        if etype == "id_switch" and e.get("from") is None:
            problems.append(
                f"  entry {i}: id_switch at frame {e.get('frame')} — 'from' is null"
            )
        if etype == "id_match" and e.get("tracking_id") is None:
            problems.append(
                f"  entry {i}: id_match for protocol_id {e.get('protocol_id')} "
                "— 'tracking_id' is null"
            )
    if problems:
        raise ValueError("Postprocessing has unfilled entries:\n" + "\n".join(problems))


def process_tracks(tracks, labels, postprocessing=None, fps=25.0, video_id=""):
    entries = postprocessing or []
    check_postprocessing(entries)

    trims = [e for e in entries if e["type"] == "trim"]
    remaps = [e for e in entries if e["type"] == "id_switch"]
    id_matches = [e for e in entries if e["type"] == "id_match"]

    tracks, labels = trim(tracks, labels, trims, fps, video_id=video_id)
    tracks = merge_id_on_switch(tracks, remaps)
    tracks = match_bird_ids(tracks, id_matches)
    return tracks, labels


def align_labels(tracks, labels, fps_lookup):
    """Crop labels to each bird's actual track coverage.

    Computes the min/max ``frame_idx`` per ``(video_id, bird_id)`` from
    *tracks*, converts to time via *fps_lookup*, and drops any label rows
    that fall outside that time range or have no matching bird in the tracks.

    Both *tracks* and *labels* must share a ``bird_id`` column.

    Parameters
    ----------
    tracks : pd.DataFrame
        Post-processed tracks with ``video_id``, ``bird_id``, ``frame_idx``.
    labels : pd.DataFrame
        Labels with ``video_id``, ``bird_id``, ``time``.
    fps_lookup : dict[str, float]
        Maps ``video_id`` to frames-per-second.

    Returns
    -------
    pd.DataFrame
        Filtered labels (same columns, fewer rows).
    """
    track_ranges = (
        tracks.groupby(["video_id", "bird_id"])["frame_idx"]
        .agg(["min", "max"])
        .reset_index()
    )
    track_ranges["t_min"] = track_ranges.apply(
        lambda r: r["min"] / fps_lookup[r["video_id"]], axis=1
    )
    track_ranges["t_max"] = track_ranges.apply(
        lambda r: r["max"] / fps_lookup[r["video_id"]], axis=1
    )

    labels_before = len(labels)
    labels = labels.merge(
        track_ranges[["video_id", "bird_id", "t_min", "t_max"]],
        on=["video_id", "bird_id"],
        how="inner",
    )
    labels = labels.query("t_min <= time <= t_max").drop(columns=["t_min", "t_max"])
    logger.info(
        f"Label alignment: kept {len(labels)}/{labels_before} rows "
        f"with matching track coverage"
    )
    return labels


def filter_incomplete_windows(
    tracks,
    labels,
    fps_lookup,
    *,
    min_coverage=DEFAULT_MIN_WINDOW_COVERAGE,
    max_coverage=1.02,
):
    """Drop windows where actual frame count is outside a coverage threshold.

    After trimming, some windows may have very few frames remaining. Feature
    statistics (std, velocity) from 2 frames are meaningless, and a temporal
    model seeing 2 frames instead of ~125 is problematic. Windows with too
    many frames indicate a windowing bug and are also dropped.

    Assumes constant FPS per video and constant window duration (label spacing)
    throughout the dataset.

    Parameters
    ----------
    tracks : pd.DataFrame
        Tracks with ``video_id``, ``bird_id``, ``frame_idx``, ``window``.
    labels : pd.DataFrame
        Labels with ``video_id``, ``bird_id``, ``window``.
    fps_lookup : dict[str, float]
        Maps ``video_id`` to frames-per-second.
    min_coverage : float
        Minimum fraction of expected frames a window must have to be kept.
        Default 0.5 (50%).
    max_coverage : float
        Maximum fraction of expected frames a window may have to be kept.
        Default 1.02 (102%) — allows 1–2 extra frames from fps rounding.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ``(tracks, labels)`` with incomplete windows removed from both.
    """
    # Derive window duration from median consecutive label time delta
    deltas = []
    for _, grp in labels.groupby(["video_id", "bird_id"]):
        times = grp["time"].sort_values().values
        if len(times) >= 2:
            deltas.extend(np.diff(times).tolist())
    if not deltas:
        logger.info("Window filter: only single-window groups, nothing to filter")
        return tracks, labels
    window_duration = float(np.median(deltas))

    # Count actual frames per window
    actual = (
        tracks.groupby(["video_id", "bird_id", "window"])["frame_idx"]
        .count()
        .rename("actual_frames")
        .reset_index()
    )
    actual["expected_frames"] = actual["video_id"].map(
        lambda vid: window_duration * fps_lookup[vid]
    )
    actual["coverage"] = actual["actual_frames"] / actual["expected_frames"]

    drop = actual.query("coverage < @min_coverage or coverage > @max_coverage")
    keep = actual.query("coverage >= @min_coverage and coverage <= @max_coverage")

    n_too_short = (actual["coverage"] < min_coverage).sum()
    n_too_long = (actual["coverage"] > max_coverage).sum()

    keep_keys = set(zip(keep["video_id"], keep["bird_id"], keep["window"]))
    tracks_key = list(zip(tracks["video_id"], tracks["bird_id"], tracks["window"]))
    labels_key = list(zip(labels["video_id"], labels["bird_id"], labels["window"]))

    tracks = tracks[[k in keep_keys for k in tracks_key]]
    labels = labels[[k in keep_keys for k in labels_key]]

    logger.info(
        f"Window filter (coverage={min_coverage:.0%}–{max_coverage:.0%}): "
        f"window_duration={window_duration:.1f}s, "
        f"dropped {len(drop)}/{len(drop) + len(keep)} windows "
        f"({n_too_short} too short, {n_too_long} too long)"
    )

    return tracks, labels


def assign_windows(tracks, labels, fps_lookup):
    """Assign a ``window`` column to both tracks and labels.

    Each label defines a window ``(prev_time, time]``; track frames whose
    time falls in that interval share the same window index.  Windows are
    numbered per ``(video_id, bird_id)`` starting from 0.

    Parameters
    ----------
    tracks : pd.DataFrame
        Post-processed tracks with ``video_id``, ``bird_id``, ``frame_idx``.
    labels : pd.DataFrame
        Aligned labels with ``video_id``, ``bird_id``, ``time``.
    fps_lookup : dict[str, float]
        Maps ``video_id`` to frames-per-second.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ``(tracks, labels)`` both with a ``window`` column.
    """
    labels = labels.sort_values(["video_id", "bird_id", "time"]).reset_index(drop=True)
    labels["window"] = labels.groupby(["video_id", "bird_id"]).cumcount()
    tracks["window"] = -1

    for (vid, bid), lbl_group in labels.groupby(["video_id", "bird_id"]):
        label_times = lbl_group["time"].values  # already sorted
        track_mask = (tracks["video_id"] == vid) & (tracks["bird_id"] == bid)
        frame_times = tracks.loc[track_mask, "frame_idx"].values / fps_lookup[vid]
        window_idx = np.searchsorted(label_times, frame_times, side="left")
        # Frames past the last label time get index == len(label_times);
        # mark them as unassigned (-1) instead of clipping into the last window
        in_range = window_idx < len(label_times)
        assigned = np.full(len(window_idx), -1, dtype=int)
        assigned[in_range] = lbl_group["window"].values[window_idx[in_range]]
        tracks.loc[track_mask, "window"] = assigned

    tracks["window"] = tracks["window"].astype(int)
    n_unmatched = (tracks["window"] == -1).sum()
    if n_unmatched:
        logger.info(f"Dropping {n_unmatched} track rows past the last label time")
        tracks = tracks[tracks["window"] != -1].reset_index(drop=True)
    logger.info(
        f"Assigned {labels['window'].nunique()} unique windows "
        f"across {labels.groupby('video_id').ngroups} video(s)"
    )
    return tracks, labels
