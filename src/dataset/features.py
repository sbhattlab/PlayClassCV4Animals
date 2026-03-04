"""Handcrafted mask feature extraction: spatial, temporal, and pairwise."""

import numpy as np
import pandas as pd
import pycocotools.mask as mask_util
from scipy.ndimage import center_of_mass
from tqdm.auto import tqdm

# Maximum masks to decode at once per size-group (caps memory ~1 GB at 1080p)
_DECODE_CHUNK_SIZE = 500

_ID_COLS = ["video_id", "bird_id"]
_KEY_COLS = ["video_id", "bird_id", "frame_idx"]
_FRAME_COLS = ["video_id", "frame_idx"]


def _prepare_rle_list(tracks):
    """Convert counts/size columns to list-of-dicts for pycocotools batch ops."""
    rle_list = []
    for counts, size in zip(tracks["counts"], tracks["size"]):
        # pycocotools requires bytes, but parquet can deserialize counts as str
        if isinstance(counts, str):
            counts = counts.encode("utf-8")
        rle_list.append({"counts": counts, "size": size})
    return rle_list


def _centroids_from_masks(masks_3d):
    """Compute centroids from a batch of decoded binary masks.

    Parameters
    ----------
    masks_3d : np.ndarray
        (H, W, N) uint8 array of binary masks (from ``mask_util.decode``).

    Returns
    -------
    cx, cy : np.ndarray
        (N,) centroid coordinates. NaN for empty masks.
    """
    N = masks_3d.shape[2]
    cx = np.full(N, np.nan)
    cy = np.full(N, np.nan)
    for k in range(N):
        if masks_3d[:, :, k].any():
            cy[k], cx[k] = center_of_mass(masks_3d[:, :, k])
    return cx, cy


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

    # --- Centroids (needed for velocity and pairwise distances) ---
    centroid_x = np.full(n, np.nan)
    centroid_y = np.full(n, np.nan)

    # pycocotools.mask.decode requires all masks to share the same size,
    # so group by size first, then decode in chunks to cap memory.
    sizes = tracks["size"].apply(tuple)
    # Use positional indices (iloc-style) since rle_list is a plain 0-based list,
    # but the DataFrame index may not be contiguous/0-based after filtering.
    idx_to_pos = {idx: pos for pos, idx in enumerate(tracks.index)}
    for _size, group in tqdm(tracks.groupby(sizes), desc="Decoding masks"):
        indices = group.index
        positions = [idx_to_pos[i] for i in indices]
        group_rles = [rle_list[p] for p in positions]

        for chunk_start in tqdm(
            range(0, len(group_rles), _DECODE_CHUNK_SIZE), leave=False
        ):
            chunk_end = chunk_start + _DECODE_CHUNK_SIZE
            chunk_rles = group_rles[chunk_start:chunk_end]
            chunk_positions = positions[chunk_start:chunk_end]

            # Batch decode: (H, W, N) binary array, one mask per N slice
            masks_3d = mask_util.decode(chunk_rles)
            if masks_3d.ndim == 2:  # single mask returns (H, W)
                masks_3d = masks_3d[:, :, np.newaxis]

            cx, cy = _centroids_from_masks(masks_3d)
            centroid_x[chunk_positions] = cx
            centroid_y[chunk_positions] = cy

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

    return pd.DataFrame(
        {
            "video_id": spatial["video_id"].values,
            "bird_id": spatial["bird_id"].values,
            "frame_idx": spatial["frame_idx"].values,
            "velocity": velocity.values,
            "area_change_rate": area_change_rate,
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
    # Spatial: mask area, aspect ratio
    spatial = extract_spatial_features(tracks)
    # Temporal: velocity, area change rate
    temporal = extract_temporal_features(spatial)
    # Pairwise: min and mean distance to nearest neighbor
    pairwise = extract_pairwise_features(spatial)
    # Merge all relevant features
    return spatial.merge(temporal, on=_KEY_COLS).merge(pairwise, on=_KEY_COLS)


_SUMMARY_AGGS = ["mean", "std", "min", "max", "median"]

# Columns that are numeric features (exclude nearest_neighbor_id which is categorical)
_FEATURE_COLS = [
    "mask_area",
    "aspect_ratio",
    "velocity",
    "area_change_rate",
    "min_dist_to_other",
    "mean_dist_to_other",
]


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
        ``{feature}_{agg}`` for each feature and aggregation (mean, std, min,
        max, median), plus ``n_frames`` (frame count per window).
    """
    group_cols = _ID_COLS + ["window"]
    grouped = features.groupby(group_cols)

    agg_dict = {col: _SUMMARY_AGGS for col in _FEATURE_COLS}
    agg_result = grouped[_FEATURE_COLS].agg(agg_dict)
    agg_result.columns = [f"{col}_{agg}" for col, agg in agg_result.columns]
    agg_result["n_frames"] = grouped.size()

    return agg_result.reset_index()
