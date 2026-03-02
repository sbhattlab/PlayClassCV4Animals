import json

import numpy as np
import pandas as pd

BEHAVIOUR_COLS = [
    "Running",
    "Frolicking",
    "Wing flapping",
    "Spinning",
    "Spinning while wing flapping",
    "Sparring jumping no contact",
    "Sparring jumping contact",
    "Sparring stand-off no contact",
    "Sparring stand-off contact",
    "Worm running",
    "Worm running mealworm",
    "Worm chasing",
    "Worm exchange",
    "Worm pecking",
]

MERGED_CLASSES = {
    "locomotor": [
        "Running",
        "Frolicking",
        "Wing flapping",
        "Spinning",
        "Spinning while wing flapping",
    ],
    "social": [
        "Sparring jumping no contact",
        "Sparring jumping contact",
        "Sparring stand-off no contact",
        "Sparring stand-off contact",
    ],
    "worm": [
        "Worm running",
        "Worm running mealworm",
        "Worm chasing",
        "Worm exchange",
        "Worm pecking",
    ],
}


def fmt_time(frame_idx, fps=25.0):
    """Frame index → MM:SS.f timestamp string."""
    t = frame_idx / fps
    m, s = divmod(t, 60)
    return f"{int(m):02d}:{s:05.2f}"


def merge_behaviours(behav_str):
    if behav_str == "none":
        return "none"
    merged = set()
    for merged_label, components in MERGED_CLASSES.items():
        if any(comp in behav_str for comp in components):
            merged.add(merged_label)
    return ", ".join(sorted(merged)) if merged else "other"


def process_labels(label_files):
    # files = glob("data/Registration protocols*.xlsx")

    all_labels = []

    for f in label_files:
        xl = pd.ExcelFile(f)

        for sheet in xl.sheet_names:
            raw = xl.parse(sheet, header=None)
            bird_id = raw.iloc[0, 0]
            bird_description = raw.iloc[0, 1]

            raw.columns = raw.iloc[1].astype(str).str.strip()
            raw = raw.iloc[2:].reset_index(drop=True)
            raw.columns.name = None

            # Only keep behaviour columns that exist in this sheet
            available_behav = [c for c in BEHAVIOUR_COLS if c in raw.columns]
            raw = raw[["min", "sek"] + available_behav].copy()

            # Fill missing behaviour columns with 0
            for col in BEHAVIOUR_COLS:
                if col not in raw.columns:
                    raw[col] = 0

            raw = raw[pd.to_numeric(raw["min"], errors="coerce").notna()].copy()
            raw["bird_id"] = bird_id
            raw["bird_description"] = bird_description
            raw["sheet"] = sheet.strip()
            all_labels.append(raw)

    labels = pd.concat(all_labels, ignore_index=True)
    labels["min"] = pd.to_numeric(labels["min"]) - 1
    labels["sek"] = pd.to_numeric(labels["sek"])
    labels["time"] = labels["min"] * 60 + labels["sek"]  # seconds
    labels["time_str"] = (
        labels["min"].astype(int).astype(str).str.zfill(2)
        + ":"
        + labels["sek"].astype(float).map("{:05.2f}".format)
    )

    # Currently, we have present/absence of each behaviour. Let's transform that to a string separated by ","
    labels["behav"] = labels[BEHAVIOUR_COLS].apply(
        lambda row: ", ".join(col for col in BEHAVIOUR_COLS if row[col] == 1), axis=1
    )
    # Replace "" by "none"
    labels["behav"] = labels["behav"].replace("", "none")

    labels["video_id"] = labels["sheet"].str.extract(r"^(C\dG\d)")[0]
    labels["behav_group"] = labels["behav"].apply(merge_behaviours)

    return labels.loc[
        :,
        [
            "video_id",
            "bird_id",
            "bird_description",
            "time",
            "time_str",
            "behav",
            "behav_group",
        ],
    ]


def remove_overlaps(tracks, issues, labels):
    overlap_events = [v for v in issues.values() if v["type"] == "overlap"]

    # 1. tracks: drop rows for the overlapping IDs within the overlap frame range
    frame_idx = tracks.index.get_level_values("frame_idx")
    object_id = tracks.index.get_level_values("bird_id")

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


def merge_id_on_switch(tracks, issues):
    raise NotImplementedError("TODO: merge tracks on id switch events in issues.json")


def process_tracks(tracks, issues, labels):
    tracks, labels = remove_overlaps(tracks, issues, labels)
    tracks = merge_id_on_switch(tracks, issues)
    return tracks, labels


def make_Xy(tracking_dirs, label_files):
    labels = process_labels(label_files)

    all_tracks = []

    for tracking_dir in tracking_dirs:
        tracks_i = pd.read_parquet(f"{tracking_dir}/tracking_outputs.parquet")
        with open(f"{tracking_dir}/tracking_issues.json", "r", encoding="utf-8") as f:
            issues_i = json.load(f)

        tracks_i_clean, labels = process_tracks(
            tracks=tracks_i, issues=issues_i, labels=labels
        )

        all_tracks.append(tracks_i_clean)

    tracks = pd.concat(all_tracks, ignore_index=True)

    # TODO: check that it's actually aligned

    return tracks, labels
