"""Handcrafted mask feature extraction: spatial, temporal, and pairwise."""

import numpy as np
import pandas as pd

from .utils import _decode_rle_mask


def extract_spatial_features(tracks: pd.DataFrame) -> pd.DataFrame:
    """Compute per-object per-frame spatial features from masks and bboxes.

    Parameters
    ----------
    tracks : pd.DataFrame
        Tracking outputs with MultiIndex ``["frame_idx", "object_id"]`` and columns
        ``bbox`` (list ``[x1, y1, x2, y2]``), ``counts``, ``size``.

    Returns
    -------
    pd.DataFrame
        Same MultiIndex as *tracks*, with columns:

        - ``mask_area``: pixel count of the mask
        - ``bbox_area``: bounding box area in pixels
        - ``aspect_ratio``: bbox width / height (NaN if height == 0)
        - ``centroid_x``, ``centroid_y``: mask centroid coordinates
    """
    rows = []
    for (fidx, oid), row in tracks.iterrows():
        mask = _decode_rle_mask(row["counts"], row["size"])
        mask_area = int(mask.sum())

        bbox = row["bbox"]
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        w = x2 - x1
        h = y2 - y1
        bbox_area = w * h
        aspect_ratio = w / h if h > 0 else np.nan

        y_coords, x_coords = np.where(mask)
        if len(y_coords) > 0:
            cx = float(np.mean(x_coords))
            cy = float(np.mean(y_coords))
        else:
            cx, cy = np.nan, np.nan

        rows.append(
            {
                "frame_idx": fidx,
                "object_id": oid,
                "mask_area": mask_area,
                "bbox_area": float(bbox_area),
                "aspect_ratio": float(aspect_ratio),
                "centroid_x": cx,
                "centroid_y": cy,
            }
        )

    df = pd.DataFrame(rows).set_index(["frame_idx", "object_id"])
    return df


def extract_temporal_features(spatial: pd.DataFrame) -> pd.DataFrame:
    """Compute per-object temporal features from spatial features.

    Parameters
    ----------
    spatial : pd.DataFrame
        Output of :func:`extract_spatial_features` (MultiIndex
        ``["frame_idx", "object_id"]``).

    Returns
    -------
    pd.DataFrame
        Same MultiIndex, with columns:

        - ``velocity_x``, ``velocity_y``: centroid displacement from previous frame
        - ``velocity``: Euclidean displacement magnitude
        - ``area_change``: absolute mask area change from previous frame
        - ``area_change_rate``: relative mask area change (NaN if previous area == 0)

        First frame per object has NaN for all temporal features.
    """
    records = []
    for oid, group in spatial.groupby("object_id"):
        group = group.sort_index(level="frame_idx")
        cx = group["centroid_x"].values
        cy = group["centroid_y"].values
        area = group["mask_area"].values
        fidxs = group.index.get_level_values("frame_idx").values

        for i in range(len(group)):
            if i == 0:
                vx, vy, vel, darea, darea_rate = (
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                )
            else:
                vx = cx[i] - cx[i - 1]
                vy = cy[i] - cy[i - 1]
                vel = np.sqrt(vx**2 + vy**2)
                darea = float(area[i] - area[i - 1])
                darea_rate = darea / area[i - 1] if area[i - 1] > 0 else np.nan

            records.append(
                {
                    "frame_idx": fidxs[i],
                    "object_id": oid,
                    "velocity_x": float(vx),
                    "velocity_y": float(vy),
                    "velocity": float(vel),
                    "area_change": float(darea),
                    "area_change_rate": float(darea_rate),
                }
            )

    df = pd.DataFrame(records).set_index(["frame_idx", "object_id"])
    return df


def extract_pairwise_features(spatial: pd.DataFrame) -> pd.DataFrame:
    """Compute per-object pairwise interaction features within each frame.

    Parameters
    ----------
    spatial : pd.DataFrame
        Output of :func:`extract_spatial_features`.

    Returns
    -------
    pd.DataFrame
        Same MultiIndex, with columns:

        - ``min_dist_to_other``: min centroid distance to any other object (inf if alone)
        - ``mean_dist_to_other``: mean centroid distance to all other objects (inf if alone)
        - ``nearest_neighbor_id``: object_id of nearest neighbor (NaN if alone)
    """
    records = []
    for fidx, group in spatial.groupby("frame_idx"):
        oids = group.index.get_level_values("object_id").values
        cxs = group["centroid_x"].values
        cys = group["centroid_y"].values

        for i, oid in enumerate(oids):
            if len(oids) < 2:
                records.append(
                    {
                        "frame_idx": fidx,
                        "object_id": oid,
                        "min_dist_to_other": np.inf,
                        "mean_dist_to_other": np.inf,
                        "nearest_neighbor_id": np.nan,
                    }
                )
                continue

            dists = np.sqrt((cxs - cxs[i]) ** 2 + (cys - cys[i]) ** 2)
            dists[i] = np.inf  # exclude self
            min_idx = np.argmin(dists)

            records.append(
                {
                    "frame_idx": fidx,
                    "object_id": oid,
                    "min_dist_to_other": float(dists[min_idx]),
                    "mean_dist_to_other": float(np.mean(dists[dists < np.inf])),
                    "nearest_neighbor_id": int(oids[min_idx]),
                }
            )

    df = pd.DataFrame(records).set_index(["frame_idx", "object_id"])
    return df


def extract_mask_features(tracks: pd.DataFrame) -> pd.DataFrame:
    """Extract all handcrafted features from tracking outputs.

    Convenience wrapper that calls :func:`extract_spatial_features`,
    :func:`extract_temporal_features`, and :func:`extract_pairwise_features`,
    then joins them into a single DataFrame.

    Parameters
    ----------
    tracks : pd.DataFrame
        Tracking outputs with MultiIndex ``["frame_idx", "object_id"]``.

    Returns
    -------
    pd.DataFrame
        Same MultiIndex, with all spatial + temporal + pairwise columns.
    """
    spatial = extract_spatial_features(tracks)
    temporal = extract_temporal_features(spatial)
    pairwise = extract_pairwise_features(spatial)
    return spatial.join(temporal).join(pairwise)


_SUMMARY_AGGS = ["mean", "std", "min", "max", "median"]

# Columns that are numeric features (exclude nearest_neighbor_id which is categorical)
_FEATURE_COLS = [
    "mask_area",
    "bbox_area",
    "aspect_ratio",
    "centroid_x",
    "centroid_y",
    "velocity_x",
    "velocity_y",
    "velocity",
    "area_change",
    "area_change_rate",
    "min_dist_to_other",
    "mean_dist_to_other",
]


def summarize_features_per_object(features: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-frame features into per-object summary statistics.

    Parameters
    ----------
    features : pd.DataFrame
        Output of :func:`extract_mask_features` (MultiIndex
        ``["frame_idx", "object_id"]``).

    Returns
    -------
    pd.DataFrame
        Indexed by ``object_id``. Columns are ``{feature}_{agg}`` for each
        feature and aggregation (mean, std, min, max, median), plus
        ``n_frames`` (frame count per object).
    """
    cols = [c for c in _FEATURE_COLS if c in features.columns]
    grouped = features[cols].groupby("object_id")
    summary = grouped.agg(_SUMMARY_AGGS)
    # Flatten MultiIndex columns: ("velocity", "mean") -> "velocity_mean"
    summary.columns = [f"{col}_{agg}" for col, agg in summary.columns]
    summary["n_frames"] = grouped.size()
    return summary


def summarize_features_per_window(
    features: pd.DataFrame,
    window_frames: int = 125,
    step_frames: int | None = None,
) -> pd.DataFrame:
    """Aggregate per-frame features into fixed-size temporal windows.

    Parameters
    ----------
    features : pd.DataFrame
        Output of :func:`extract_mask_features`.
    window_frames : int
        Window size in frames (default 125 = 5 s at 25 fps).
    step_frames : int | None
        Step between windows. Defaults to *window_frames* (non-overlapping).

    Returns
    -------
    pd.DataFrame
        MultiIndex ``["object_id", "window_start"]``. Columns are
        ``{feature}_{agg}`` plus ``n_frames`` (actual frames in window).
    """
    if step_frames is None:
        step_frames = window_frames

    cols = [c for c in _FEATURE_COLS if c in features.columns]
    all_frames = features.index.get_level_values("frame_idx")
    f_min, f_max = int(all_frames.min()), int(all_frames.max())

    records = []
    for oid, obj_feat in features.groupby("object_id"):
        obj_frames = obj_feat.index.get_level_values("frame_idx")

        for win_start in range(f_min, f_max + 1, step_frames):
            win_end = win_start + window_frames
            mask = (obj_frames >= win_start) & (obj_frames < win_end)
            window = obj_feat.loc[mask, cols]

            if len(window) == 0:
                continue

            row = {"object_id": oid, "window_start": win_start, "n_frames": len(window)}
            for col in cols:
                vals = window[col].dropna()
                if len(vals) == 0:
                    for agg in _SUMMARY_AGGS:
                        row[f"{col}_{agg}"] = np.nan
                else:
                    row[f"{col}_mean"] = float(vals.mean())
                    row[f"{col}_std"] = float(vals.std())
                    row[f"{col}_min"] = float(vals.min())
                    row[f"{col}_max"] = float(vals.max())
                    row[f"{col}_median"] = float(vals.median())
            records.append(row)

    df = pd.DataFrame(records).set_index(["object_id", "window_start"])
    return df
