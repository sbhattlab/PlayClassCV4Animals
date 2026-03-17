"""Handcrafted mask feature extraction: spatial, temporal, and pairwise."""

import multiprocessing
import os

import cv2
import numpy as np
import pandas as pd
import pycocotools.mask as mask_util
from loguru import logger
from tqdm.auto import tqdm

_N_WORKERS = 16

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maximum masks to decode at once per size-group (caps memory ~1 GB at 1080p)
_DECODE_CHUNK_SIZE = 256

_ID_COLS = ["video_id", "bird_id"]
_KEY_COLS = ["video_id", "bird_id", "frame_idx"]
_FRAME_COLS = ["video_id", "frame_idx"]

# Columns that are numeric features (exclude nearest_neighbor_id which is categorical)
_FEATURE_COLS = [
    "mask_area",
    "aspect_ratio",
    "velocity",
    "acceleration",
    "area_change_rate",
    "orientation_velocity",
    "solidity_change",
    "elongation_change",
    "turning_angle",
    "velocity_autocorr",
    "min_dist_to_other",
    "mean_dist_to_other",
    "dist_change_rate",
    "elongation",
    "orientation",
    "solidity",
    "eccentricity",
    "perimeter",
    "circularity",
]


# ---------------------------------------------------------------------------
# Aggregation helpers & config
# ---------------------------------------------------------------------------


def mad(x):
    """Median absolute deviation."""
    return (x - x.median()).abs().median()


def cv(x):
    """Coefficient of variation (std / mean)."""
    m = x.mean()
    return x.std() / m if m != 0 else np.nan


def q10(x):
    """10th percentile."""
    return x.quantile(0.1)


def q90(x):
    """90th percentile."""
    return x.quantile(0.9)


_SUMMARY_AGGS = ["mean", "std", "median", mad, "skew", "kurt", cv, q10, q90]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _prepare_rle_list(tracks):
    """Convert counts/size columns to list-of-dicts for pycocotools batch ops."""
    rle_list = []
    for counts, size in zip(tracks["counts"], tracks["size"]):
        # pycocotools requires bytes, but parquet can deserialize counts as str
        if isinstance(counts, str):
            counts = counts.encode("utf-8")
        rle_list.append({"counts": counts, "size": size})
    return rle_list


def _mask_geometry(masks_3d):
    """Compute centroids and shape features from decoded binary masks.

    Uses ``cv2.moments()`` for centroids and second-order moments, and
    ``cv2.convexHull`` for solidity.

    Parameters
    ----------
    masks_3d : np.ndarray
        (H, W, N) uint8 array of binary masks (from ``mask_util.decode``).

    Returns
    -------
    dict[str, np.ndarray]
        Keys: ``centroid`` (N, 2) [cx, cy], ``elongation``, ``orientation``,
        ``solidity``, ``eccentricity`` — each (N,) float64. NaN for empty masks.
    """
    n = masks_3d.shape[2]
    centroid = np.full((n, 2), np.nan)
    elongation = np.full(n, np.nan)
    orientation = np.full(n, np.nan)
    solidity = np.full(n, np.nan)
    eccentricity = np.full(n, np.nan)
    perimeter = np.full(n, np.nan)
    circularity = np.full(n, np.nan)

    for k in range(n):
        mask = masks_3d[:, :, k]
        m = cv2.moments(mask, binaryImage=True)
        area = m["m00"]
        if area == 0:
            continue

        # Centroid
        centroid[k] = (m["m10"] / area, m["m01"] / area)

        # Second-order central moments → eigenvalues of covariance matrix
        mu20 = m["mu20"] / area
        mu02 = m["mu02"] / area
        mu11 = m["mu11"] / area

        diff = mu20 - mu02
        discriminant = np.sqrt(diff**2 + 4 * mu11**2)
        lam1 = 0.5 * (mu20 + mu02 + discriminant)  # major
        lam2 = max(0.5 * (mu20 + mu02 - discriminant), 0.0)  # minor

        # Elongation: major / minor axis ratio
        if lam2 > 0:
            elongation[k] = np.sqrt(lam1 / lam2)

        # Orientation: angle of major axis (radians, -pi/2 to pi/2)
        orientation[k] = 0.5 * np.arctan2(2 * mu11, diff)

        # Eccentricity: sqrt(1 - minor/major), 0=circle, 1=line
        if lam1 > 0:
            eccentricity[k] = np.sqrt(1.0 - lam2 / lam1)

        # Solidity: mask area / convex hull area
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            all_pts = np.vstack(contours)
            hull = cv2.convexHull(all_pts)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity[k] = area / hull_area

            # Perimeter and circularity
            perim = cv2.arcLength(all_pts, closed=True)
            perimeter[k] = perim
            if perim > 0:
                circularity[k] = 4.0 * np.pi * area / (perim**2)

    return {
        "centroid": centroid,
        "elongation": elongation,
        "orientation": orientation,
        "solidity": solidity,
        "eccentricity": eccentricity,
        "perimeter": perimeter,
        "circularity": circularity,
    }


def _decode_and_compute_geometry(chunk_rles):
    """Worker function: decode RLEs and compute geometry in one step."""
    masks_3d = mask_util.decode(chunk_rles)
    if masks_3d.ndim == 2:
        masks_3d = masks_3d[:, :, np.newaxis]
    return _mask_geometry(masks_3d)


# ---------------------------------------------------------------------------
# Per-frame feature extraction
# ---------------------------------------------------------------------------


def extract_spatial_features(tracks: pd.DataFrame) -> pd.DataFrame:
    """Compute per-object per-frame spatial features from masks.

    Parameters
    ----------
    tracks : pd.DataFrame
        Flat DataFrame with columns ``video_id``, ``bird_id``, ``frame_idx``,
        ``bbox`` (list ``[x1, y1, x2, y2]``), ``counts``, ``size``.

    Returns
    -------
    pd.DataFrame
        Flat DataFrame with key columns (``video_id``, ``bird_id``,
        ``frame_idx``) plus:

        - ``mask_area``: pixel count of the mask
        - ``bbox_area``: bounding box area in pixels
          (not a feature — useful for detecting bbox spikes vs mask_area)
        - ``aspect_ratio``: bbox width / height (may capture posture)
        - ``centroid_x``, ``centroid_y``: mask centroid coordinates
          (intermediates for temporal/pairwise features, not used as features)
    """
    n = len(tracks)

    # --- Mask areas ---
    rle_list = _prepare_rle_list(tracks)
    mask_areas = np.array(mask_util.area(rle_list), dtype=np.float64)

    # --- Aspect ratio from bbox ---
    bbox_arr = np.array(tracks["bbox"].tolist(), dtype=np.float64)
    w = bbox_arr[:, 2] - bbox_arr[:, 0]
    h = bbox_arr[:, 3] - bbox_arr[:, 1]
    bbox_areas = w * h
    aspect_ratios = np.full_like(w, np.nan)
    np.divide(w, h, out=aspect_ratios, where=h > 0)

    # --- Geometry: centroids + shape features ---
    centroid_x = np.full(n, np.nan)
    centroid_y = np.full(n, np.nan)
    elongation = np.full(n, np.nan)
    orientation = np.full(n, np.nan)
    solidity = np.full(n, np.nan)
    eccentricity = np.full(n, np.nan)
    perimeter = np.full(n, np.nan)
    circularity = np.full(n, np.nan)

    # pycocotools.mask.decode requires all masks to share the same size,
    # so group by size first, then decode in chunks to cap memory.
    sizes = tracks["size"].apply(tuple)
    # Use positional indices (iloc-style) since rle_list is a plain 0-based list,
    # but the DataFrame index may not be contiguous/0-based after filtering.
    idx_to_pos = {idx: pos for pos, idx in enumerate(tracks.index)}

    # Build list of (chunk_rles, chunk_positions) work items
    work_items = []
    for _size, group in tracks.groupby(sizes):
        indices = group.index
        positions = [idx_to_pos[i] for i in indices]
        group_rles = [rle_list[p] for p in positions]
        for chunk_start in range(0, len(group_rles), _DECODE_CHUNK_SIZE):
            chunk_end = chunk_start + _DECODE_CHUNK_SIZE
            work_items.append(
                (
                    group_rles[chunk_start:chunk_end],
                    positions[chunk_start:chunk_end],
                )
            )

    logger.info(f"Decoding {len(work_items)} mask chunks with {_N_WORKERS} workers")
    with multiprocessing.Pool(processes=_N_WORKERS) as pool:
        results = list(
            tqdm(
                pool.imap(_decode_and_compute_geometry, [w[0] for w in work_items]),
                total=len(work_items),
                desc="Decoding masks",
            )
        )

    for (_, chunk_positions), geom in zip(work_items, results):
        centroid_x[chunk_positions] = geom["centroid"][:, 0]
        centroid_y[chunk_positions] = geom["centroid"][:, 1]
        elongation[chunk_positions] = geom["elongation"]
        orientation[chunk_positions] = geom["orientation"]
        solidity[chunk_positions] = geom["solidity"]
        eccentricity[chunk_positions] = geom["eccentricity"]
        perimeter[chunk_positions] = geom["perimeter"]
        circularity[chunk_positions] = geom["circularity"]

    return pd.DataFrame(
        {
            "video_id": tracks["video_id"].values,
            "bird_id": tracks["bird_id"].values,
            "frame_idx": tracks["frame_idx"].values,
            "mask_area": mask_areas,
            "bbox_area": bbox_areas,  # not a feature — for detecting bbox spikes
            "aspect_ratio": aspect_ratios,
            "centroid_x": centroid_x,
            "centroid_y": centroid_y,
            "elongation": elongation,
            "orientation": orientation,
            "solidity": solidity,
            "eccentricity": eccentricity,
            "perimeter": perimeter,
            "circularity": circularity,
        }
    )


def extract_temporal_features(spatial: pd.DataFrame) -> pd.DataFrame:
    """Compute per-object temporal features from spatial features.

    Parameters
    ----------
    spatial : pd.DataFrame
        Output of :func:`extract_spatial_features` (flat DataFrame).

    Returns
    -------
    pd.DataFrame
        Flat DataFrame with key columns plus:

        - ``velocity``: Euclidean centroid displacement from previous frame
        - ``area_change_rate``: relative mask area change (NaN if previous area == 0)

        First frame per (video_id, bird_id) has NaN for all temporal features.
    """
    # Sort so diff() operates in correct frame order within each group
    spatial = spatial.sort_values(_KEY_COLS)

    grouped = spatial.groupby(_ID_COLS, sort=False)

    # Velocity
    velocity_x = grouped["centroid_x"].diff()
    velocity_y = grouped["centroid_y"].diff()
    velocity = np.sqrt(velocity_x**2 + velocity_y**2)

    # Relative area change: (current - prev) / prev
    area_prev = grouped["mask_area"].shift(1)
    area_change_rate = np.full(len(spatial), np.nan)
    np.divide(
        (spatial["mask_area"] - area_prev).values,
        area_prev.values,
        out=area_change_rate,
        where=(area_prev > 0).values,
    )

    # Shape dynamics
    orientation_velocity = grouped["orientation"].diff()
    solidity_change = grouped["solidity"].diff()
    elongation_change = grouped["elongation"].diff()

    # Acceleration: change in velocity (captures bursty starts/stops)
    acceleration = velocity.groupby([spatial["video_id"], spatial["bird_id"]]).diff()

    # Turning angle: angle between consecutive velocity vectors
    # Set to NaN when velocity is near zero (stationary bird → noisy angle)
    vx_prev = velocity_x.shift(1)
    vy_prev = velocity_y.shift(1)
    cross = vx_prev * velocity_y - vy_prev * velocity_x
    dot = vx_prev * velocity_x + vy_prev * velocity_y
    turning_angle = np.arctan2(cross, dot).abs()
    # Mask out frames where either velocity vector is near zero
    speed_prev = np.sqrt(vx_prev**2 + vy_prev**2)
    min_speed = 1.0  # pixels — below this, direction is meaningless
    turning_angle[velocity < min_speed] = np.nan
    turning_angle[speed_prev < min_speed] = np.nan

    # Velocity autocorrelation: product of consecutive velocities
    # High when motion is sustained (locomotor), low/variable for bursty worm behaviour
    velocity_prev = velocity.groupby([spatial["video_id"], spatial["bird_id"]]).shift(1)
    velocity_autocorr = velocity * velocity_prev

    return pd.DataFrame(
        {
            "video_id": spatial["video_id"].values,
            "bird_id": spatial["bird_id"].values,
            "frame_idx": spatial["frame_idx"].values,
            "velocity": velocity.values,
            "acceleration": acceleration.values,
            "area_change_rate": area_change_rate,
            "orientation_velocity": orientation_velocity.values,
            "solidity_change": solidity_change.values,
            "elongation_change": elongation_change.values,
            "turning_angle": turning_angle.values,
            "velocity_autocorr": velocity_autocorr.values,
        }
    )


def extract_pairwise_features(spatial: pd.DataFrame) -> pd.DataFrame:
    """Compute per-object pairwise interaction features within each frame.

    Parameters
    ----------
    spatial : pd.DataFrame
        Output of :func:`extract_spatial_features` (flat DataFrame).

    Returns
    -------
    pd.DataFrame
        Flat DataFrame with key columns plus:

        - ``min_dist_to_other``: min centroid distance to any other bird (NaN if alone)
        - ``mean_dist_to_other``: mean centroid distance to all other birds (NaN if alone)
        - ``nearest_neighbor_id``: bird_id of nearest neighbor (NaN if alone)
    """
    n = len(spatial)

    # Min distance to nearest neighbor
    min_dist = np.full(n, np.nan)
    # Mean distance to all other birds (may capture crowding)
    mean_dist = np.full(n, np.nan)
    # ID of nearest neighbor (categorical feature, may capture social interactions)
    nn_id = np.full(n, np.nan, dtype=object)

    frame_groups = spatial.groupby(_FRAME_COLS)
    for _key, group in tqdm(frame_groups, desc="Pairwise distances", unit="frame"):
        idx = group.index.values
        # Skip frames with only one bird (no pairwise features to compute)
        M = len(group)
        if M < 2:
            continue

        cxs = group["centroid_x"].values
        cys = group["centroid_y"].values
        bird_ids = group["bird_id"].values

        # Skip birds with NaN centroids (empty masks)
        valid = ~(np.isnan(cxs) | np.isnan(cys))
        if valid.sum() < 2:
            continue

        # (M, M) distance matrix; NaN on diagonal to exclude self
        dx = cxs[:, None] - cxs[None, :]
        dy = cys[:, None] - cys[None, :]
        dist_matrix = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dist_matrix, np.nan)

        # Only compute for rows with valid centroids
        valid_idx = idx[valid]
        valid_rows = np.where(valid)[0]
        min_indices = np.nanargmin(dist_matrix[valid_rows], axis=1)
        min_dist[valid_idx] = dist_matrix[valid_rows, min_indices]
        nn_id[valid_idx] = bird_ids[min_indices]
        mean_dist[valid_idx] = np.nanmean(dist_matrix[valid_rows], axis=1)

    return pd.DataFrame(
        {
            "video_id": spatial["video_id"].values,
            "bird_id": spatial["bird_id"].values,
            "frame_idx": spatial["frame_idx"].values,
            "min_dist_to_other": min_dist,
            "mean_dist_to_other": mean_dist,
            "nearest_neighbor_id": nn_id,
        }
    )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def extract_mask_features(tracks: pd.DataFrame) -> pd.DataFrame:
    """Extract all handcrafted features from tracking outputs.

    Convenience wrapper that calls :func:`extract_spatial_features`,
    :func:`extract_temporal_features`, and :func:`extract_pairwise_features`,
    then merges them into a single DataFrame.

    Parameters
    ----------
    tracks : pd.DataFrame
        Flat DataFrame with ``video_id``, ``bird_id``, ``frame_idx``,
        ``bbox``, ``counts``, ``size``.

    Returns
    -------
    pd.DataFrame
        Flat DataFrame with all spatial + temporal + pairwise columns.
    """
    # Spatial: mask area, aspect ratio, shape features
    spatial = extract_spatial_features(tracks)
    # Temporal: velocity, area change rate
    temporal = extract_temporal_features(spatial)
    # Pairwise: min and mean distance to nearest neighbor
    pairwise = extract_pairwise_features(spatial)
    # Merge all relevant features
    merged = spatial.merge(temporal, on=_KEY_COLS).merge(pairwise, on=_KEY_COLS)
    # Distance change rate: frame-to-frame change in nearest-bird distance
    merged = merged.sort_values(_KEY_COLS)
    merged["dist_change_rate"] = merged.groupby(_ID_COLS)["min_dist_to_other"].diff()
    return merged


# ---------------------------------------------------------------------------
# Window summarisation
# ---------------------------------------------------------------------------


def summarize_features_by_window(features: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-frame features using label-aligned ``window`` column.

    Parameters
    ----------
    features : pd.DataFrame
        Output of :func:`extract_mask_features` with an additional ``window``
        column (from :func:`assign_windows`).

    Returns
    -------
    pd.DataFrame
        One row per ``(video_id, bird_id, window)``. Columns are
        ``{feature}_{agg}`` for each feature and aggregation, plus
        ``n_frames`` (frame count per window).
    """
    group_cols = _ID_COLS + ["window"]
    grouped = features.groupby(group_cols)

    agg_dict = {col: _SUMMARY_AGGS for col in _FEATURE_COLS}
    agg_result = grouped[_FEATURE_COLS].agg(agg_dict)
    agg_result.columns = [f"{col}_{agg}" for col, agg in agg_result.columns]
    agg_result["n_frames"] = grouped.size()

    return agg_result.reset_index()


def bin_features_per_window(
    features: pd.DataFrame,
) -> dict[tuple, "torch.Tensor"]:
    """Convert per-frame features into per-window tensors for temporal models.

    Same output format as ``embeddings.pt``: dict keyed by
    ``(video_id, bird_id, window)`` with ``Tensor(T, D)`` values,
    where T is the number of frames in that window and D is ``len(_FEATURE_COLS)``.

    Parameters
    ----------
    features : pd.DataFrame
        Output of :func:`extract_mask_features` with ``window`` column.

    Returns
    -------
    dict[tuple, torch.Tensor]
    """
    import torch

    group_cols = _ID_COLS + ["window"]
    result = {}
    for key, group in features.groupby(group_cols):
        vals = group[_FEATURE_COLS].values.astype(np.float32)
        # Replace NaN with 0 (first-frame temporal features)
        vals = np.nan_to_num(vals, nan=0.0)
        result[key] = torch.from_numpy(vals)
    return result
