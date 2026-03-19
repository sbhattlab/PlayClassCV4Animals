"""Behaviour label processing from Registration protocols Excel files."""

import re
from pathlib import Path

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


def merge_behaviours(behav_str):
    if behav_str == "none":
        return "none"
    merged = set()
    for merged_label, components in MERGED_CLASSES.items():
        if any(comp in behav_str for comp in components):
            merged.add(merged_label)
    return ", ".join(sorted(merged)) if merged else "other"


def _extract_age(filepath) -> int:
    """Extract bird age in days from a label filename.

    Expects filenames like ``Registration protocols week 1 day 2 (age 29 days).xlsx``.
    """
    m = re.search(r"age (\d+) days", Path(filepath).name)
    if not m:
        raise ValueError(
            f"Cannot extract age from filename: {Path(filepath).name!r}. "
            "Expected pattern 'age <N> days'."
        )
    return int(m.group(1))


def process_labels(label_files):
    """Parse behaviour labels and bird info from Registration protocols Excel files.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, dict[int, str]]]
        ``(labels_df, bird_info)`` where *bird_info* maps
        ``{video_id: {bird_id: bird_description}}``.
        Video IDs have the form ``CxGyDz`` (e.g. ``C1G3D28``).
    """
    all_labels = []
    bird_info: dict[str, dict[int, str]] = {}

    for f in label_files:
        age = _extract_age(f)
        xl = pd.ExcelFile(f)

        for sheet in xl.sheet_names:
            raw = xl.parse(sheet, header=None)
            bird_id = raw.iloc[0, 0]
            bird_description = raw.iloc[0, 1]

            # Accumulate bird_info while we already have the sheet open
            cage_group = sheet.strip().split()[0]  # "C1G1 2664" -> "C1G1"
            video_id = f"{cage_group}D{age}"
            if video_id not in bird_info:
                bird_info[video_id] = {}
            bird_info[video_id][int(bird_id)] = str(bird_description).strip()

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
            raw["_age"] = age
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

    cage_group = labels["sheet"].str.extract(r"^(C\dG\d)")[0]
    labels["video_id"] = cage_group + "D" + labels["_age"].astype(str)
    labels["behav_group"] = labels["behav"].apply(merge_behaviours)

    labels = labels.loc[
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

    return labels, bird_info


def resolve_dual_groups(labels: pd.DataFrame) -> pd.DataFrame:
    """Resolve multi-valued ``behav_group`` entries to a single class.

    When a label row has active behaviours in multiple merged classes,
    ``behav_group`` contains a comma-separated string (e.g. ``"locomotor,
    worm"``).  This function picks the rarest component class (across all
    rows) as the resolved label, since rare behaviours are more informative
    as classification targets.

    Parameters
    ----------
    labels : pd.DataFrame
        Must contain a ``behav_group`` column.

    Returns
    -------
    pd.DataFrame
        Same DataFrame with a new ``behav_label`` column.
    """
    # Count frequency of each individual group across all rows
    from collections import Counter

    counts: Counter[str] = Counter()
    for bg in labels["behav_group"]:
        for part in bg.split(", "):
            counts[part.strip()] += 1

    def _resolve(behav_group: str) -> str:
        parts = [p.strip() for p in behav_group.split(", ")]
        if len(parts) == 1:
            return parts[0]
        # Pick the rarest component
        return min(parts, key=lambda p: counts[p])

    labels["behav_label"] = labels["behav_group"].apply(_resolve)
    return labels
