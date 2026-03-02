"""Behaviour label processing from Registration protocols Excel files."""

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


def process_labels(label_files):
    """Parse behaviour labels and bird info from Registration protocols Excel files.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, dict[int, str]]]
        ``(labels_df, bird_info)`` where *bird_info* maps
        ``{video_id: {bird_id: bird_description}}``.
    """
    all_labels = []
    bird_info: dict[str, dict[int, str]] = {}

    for f in label_files:
        xl = pd.ExcelFile(f)

        for sheet in xl.sheet_names:
            raw = xl.parse(sheet, header=None)
            bird_id = raw.iloc[0, 0]
            bird_description = raw.iloc[0, 1]

            # Accumulate bird_info while we already have the sheet open
            video_id = sheet.strip().split()[0]  # "C1G1 2664" -> "C1G1"
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
