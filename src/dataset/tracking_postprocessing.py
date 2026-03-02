"""Tracking issue detection: remediation."""

import numpy as np
import pandas as pd
from loguru import logger

from .utils import fmt_time


def _worst_id_in_range(tracks, ids, start, end):
    """Return the ID with the lowest mean tracker_score in [start, end]."""
    frame_idx = tracks.index.get_level_values("frame_idx")
    object_id = tracks.index.get_level_values("object_id")
    in_range = (frame_idx >= start) & (frame_idx <= end)

    worst_id, worst_score = ids[0], float("inf")
    for oid in ids:
        subset = tracks[in_range & (object_id == oid)]
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
        Tracking outputs with MultiIndex ``["frame_idx", "object_id"]``.
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
    for object_id, object_description in video_birds.items():
        entries.append(
            {
                "type": "id_match",
                "protocol_id": object_id,
                "description": object_description,
                "tracking_id": None,
                "frame": None,
            }
        )
    return entries


def merge_id_on_switch(tracks, id_remaps):
    """Remap object IDs across switch events to produce continuous per-bird tracks.

    Each entry says: "at ``frame``, the chicken that **was** ``from`` gets
    reassigned to ``to``."  Rows with ``object_id == from`` **before** the
    switch frame are renamed to ``to``.  Rows at or after the switch keep
    their current ID (a different chicken may now occupy the ``from`` slot).

    Remaps at the same frame are applied simultaneously (masks computed before
    any renames) so that swaps (e.g. A→B and B→C at the same frame) work
    correctly.  Across different frames, groups are applied earliest-first.

    Parameters
    ----------
    tracks : pd.DataFrame
        Tracking outputs with MultiIndex ``["frame_idx", "object_id"]``.
    id_remaps : list[dict]
        Each entry has ``"frame"`` (int), ``"from"`` (int), ``"to"`` (int).

    Returns
    -------
    pd.DataFrame
        Tracks with the same columns but updated ``object_id`` index.
    """
    if not id_remaps:
        return tracks

    df = tracks.reset_index()

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
            mask = (df["frame_idx"] < frame) & (df["object_id"] == id_from)
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
            df.loc[mask, "object_id"] = id_to
            logger.info(
                f"Remapped {n} rows: object_id {id_from}→{id_to} "
                f"(before frame {frame})"
            )

    df = df.set_index(["frame_idx", "object_id"]).sort_index()
    return df


def trim(tracks, labels, trims, fps):
    """Drop track rows within each trim's ``[from, to]`` frame range.

    If ``id`` is present, only rows for that object ID are dropped.
    Otherwise all rows in the range are dropped (global trim).
    Labels whose time falls within the range are always dropped.
    """
    if not trims:
        return tracks, labels

    frame_idx = tracks.index.get_level_values("frame_idx")
    object_id = tracks.index.get_level_values("object_id")
    tracks_mask = np.zeros(len(tracks), dtype=bool)
    labels_mask = pd.Series(False, index=labels.index)

    for t in trims:
        f_from, f_to = t["from"], t["to"]
        t_from, t_to = f_from / fps, f_to / fps
        cause = t.get("cause", "")
        tag = f" ({cause})" if cause else ""

        in_range = (frame_idx >= f_from) & (frame_idx <= f_to)
        if "id" in t:
            in_range = in_range & (object_id == t["id"])
            tag += f" ID {t['id']}"

        tracks_mask |= in_range
        labels_mask |= (labels["time"] >= t_from) & (labels["time"] <= t_to)

        logger.info(
            f"Trim{tag} frames {f_from}-{f_to} "
            f"({fmt_time(f_from, fps)}-{fmt_time(f_to, fps)})"
        )

    n_t, n_l = tracks_mask.sum(), labels_mask.sum()
    tracks = tracks[~tracks_mask]
    labels = labels[~labels_mask]
    logger.info(f"Trims: dropped {n_t} track rows, {n_l} label rows")

    return tracks, labels


def match_bird_ids(tracks, id_matches):
    """Rename object_ids to protocol_ids using verified id_match entries.

    Runs **after** ``merge_id_on_switch``, so ``tracking_id`` must reference
    the post-merge surviving IDs, not the original tracker-assigned IDs.

    Parameters
    ----------
    tracks : pd.DataFrame
        Tracking outputs with MultiIndex ``["frame_idx", "object_id"]``.
    id_matches : list[dict]
        Each entry has ``"protocol_id"`` (int/str), ``"tracking_id"``
        (int/str — the **post-merge** object ID), and ``"frame"`` (int,
        reference frame for verification).

    Returns
    -------
    pd.DataFrame
        Tracks with ``object_id`` values renamed to ``protocol_id``.
    """
    if not id_matches:
        return tracks

    df = tracks.reset_index()

    for match in id_matches:
        tracking_id = int(match["tracking_id"])
        protocol_id = int(match["protocol_id"])
        frame = match.get("frame", None)

        # Verify tracking_id exists (optionally at a specific frame)
        if frame is not None:
            at_frame = df[(df["frame_idx"] == frame) & (df["object_id"] == tracking_id)]
            if at_frame.empty:
                logger.warning(
                    f"ID match: tracking_id {tracking_id} not found at frame {frame} "
                    f"(target protocol_id {protocol_id})"
                )
                continue

        mask = df["object_id"] == tracking_id
        n = mask.sum()
        df.loc[mask, "object_id"] = protocol_id
        logger.info(
            f"Matched {n} rows: object_id {tracking_id} → protocol_id {protocol_id} "
            f"(verified at frame {frame})"
        )

    df = df.set_index(["frame_idx", "object_id"]).sort_index()
    return df


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
        if etype == "id_match" and e.get("tracking_id") is None:
            problems.append(
                f"  entry {i}: id_match for protocol_id {e.get('protocol_id')} "
                "— 'tracking_id' is null"
            )
    if problems:
        raise ValueError("Postprocessing has unfilled entries:\n" + "\n".join(problems))


def process_tracks(tracks, labels, postprocessing=None, fps=25.0):
    entries = postprocessing or []
    check_postprocessing(entries)

    trims = [e for e in entries if e["type"] == "trim"]
    remaps = [e for e in entries if e["type"] == "id_switch"]
    id_matches = [e for e in entries if e["type"] == "id_match"]

    tracks, labels = trim(tracks, labels, trims, fps)
    tracks = merge_id_on_switch(tracks, remaps)
    tracks = match_bird_ids(tracks, id_matches)
    return tracks, labels
