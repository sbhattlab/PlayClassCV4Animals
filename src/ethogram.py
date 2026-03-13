"""
Ethogram parsing utilities for chicken behaviour classification.

Parses the Registration protocols Excel file into a clean DataFrame with
merged behaviour classes suitable for downstream classification.

Behaviour columns are merged into three coarse classes:
  - locomotor: Running, Frolicking, Wing flapping, Spinning,
                Spinning while wing flapping
  - social:    Sparring jumping no contact, Sparring jumping contact,
                Sparring stand-off no contact, Sparring stand-off contact
  - worm:      Worm running, Worm running mealworm, Worm chasing,
                Worm exchange, Worm pecking

Windows where all behaviour cells are empty are labelled 'none'.
Windows where all behaviour cells are explicitly zero are labelled 'inactive'.
Windows with more than one merged class active are labelled 'multi' and
are typically dropped before classification.
"""

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


def parse_ethogram(path: str) -> pd.DataFrame:
    """Parse the Registration protocols Excel file.

    Each sheet corresponds to one individual-session recording.
    Sheet naming convention: {Camera}{Group} {IndividualID}
    e.g. 'C5G3 2662' → camera 5, group 3, individual 2662.

    Returns a DataFrame with columns:
        min, sek, <BEHAVIOUR_COLS>, bird_id, sheet,
        locomotor, social, worm, merged_n, merged_label
    """
    xl = pd.ExcelFile(path)
    all_dfs = []

    for sheet in xl.sheet_names:
        raw = xl.parse(sheet, header=None)
        bird_id = raw.iloc[0, 0]

        raw.columns = raw.iloc[1].str.strip()
        raw = raw.iloc[2:].reset_index(drop=True)
        raw.columns.name = None

        # Normalise variant column name across Excel versions
        raw = raw.rename(columns={"Object running": "Worm running"})

        raw = raw[["min", "sek"] + BEHAVIOUR_COLS].copy()
        raw = raw[pd.to_numeric(raw["min"], errors="coerce").notna()].copy()
        raw["bird_id"] = bird_id
        raw["sheet"] = sheet
        all_dfs.append(raw)

    df = pd.concat(all_dfs, ignore_index=True)
    behaviour_numeric = df[BEHAVIOUR_COLS].apply(pd.to_numeric, errors="coerce")
    all_empty = behaviour_numeric.isna().all(axis=1)
    df[BEHAVIOUR_COLS] = behaviour_numeric.fillna(0).astype(int)
    df["min"] = pd.to_numeric(df["min"])
    df["sek"] = pd.to_numeric(df["sek"])

    for label, cols in MERGED_CLASSES.items():
        df[label] = df[cols].sum(axis=1).clip(upper=1)

    df["merged_n"] = df[list(MERGED_CLASSES)].sum(axis=1)

    df["merged_label"] = "inactive"
    df.loc[all_empty, "merged_label"] = "none"
    df.loc[df["merged_n"] > 1, "merged_label"] = "multi"
    for label in MERGED_CLASSES:
        df.loc[(df["merged_n"] == 1) & (df[label] == 1), "merged_label"] = label

    return df


def get_single_label(df: pd.DataFrame, include_inactive: bool = True) -> pd.DataFrame:
    """Return only unambiguous single-label windows.

    Args:
        df: Output of parse_ethogram().
        include_inactive: If True, retain inactive windows as a class.
                          If False, return only windows with one active behaviour.
    """
    if include_inactive:
        return df[df["merged_label"] != "multi"].copy()
    return df[df["merged_n"] == 1].copy()
