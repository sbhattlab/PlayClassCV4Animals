"""
Data processing utilities for SAM3 tracking outputs.

Functions for extracting, converting, and processing mask/bbox/point data
from model outputs.
"""

import numpy as np
import pandas as pd
import pycocotools.mask as mask_util
import torch
from loguru import logger


def to_numpy(x):
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.array(x)


def get_all_objects_from_results(results: dict) -> tuple[list, list, list]:
    """
    Extract masks, boxes, and object_ids from a single frame's output dict.

    Handles both torch tensors and numpy arrays.

    Returns:
        (masks_list, boxes_list, object_ids_list) where each is a list of
        per-object arrays.
    """
    masks = results.get("masks")
    boxes = results.get("boxes")
    object_ids = results.get("object_ids")

    if masks is None or object_ids is None:
        return [], [], []

    masks_np = to_numpy(masks)
    boxes_np = to_numpy(boxes) if boxes is not None else None
    ids_np = to_numpy(object_ids)

    masks_list = [masks_np[i] for i in range(len(ids_np))]
    boxes_list = (
        [boxes_np[i] for i in range(len(ids_np))] if boxes_np is not None else []
    )
    object_ids_list = ids_np.tolist()

    return masks_list, boxes_list, object_ids_list


def find_frame_with_enough_objects(
    outputs_per_frame: dict,
    min_objects: int = 3,
    max_lookback: int = 10,
) -> tuple[int | None, list, list, list]:
    """
    Search backwards through frame outputs to find a frame with enough objects.

    Used as fallback, if no qualifying frame in grounding outputs

    Args:
        outputs_per_frame: Dict mapping frame_idx -> processed output dict.
        min_objects: Minimum number of objects required.
        max_lookback: Maximum number of frames to search backwards from the end.

    Returns:
        (frame_idx, masks_list, boxes_list, object_ids_list) or
        (None, [], [], []) if no suitable frame found.
    """
    sorted_frames = sorted(outputs_per_frame.keys(), reverse=True)

    for frame_idx in sorted_frames[:max_lookback]:
        masks_list, boxes_list, object_ids_list = get_all_objects_from_results(
            outputs_per_frame[frame_idx]
        )
        if len(object_ids_list) >= min_objects:
            logger.debug(f"Found {len(object_ids_list)} objects at frame {frame_idx}")
            return frame_idx, masks_list, boxes_list, object_ids_list

    logger.warning(
        f"No frame with >= {min_objects} objects found in last {max_lookback} frames"
    )
    return None, [], [], []


def extract_equidistant_points_from_mask(
    mask: np.ndarray, num_points: int = 3
) -> list[list[int]] | None:
    """
    Extract equidistant points along the mask's major axis (x-axis).

    Samples points deterministically by sorting pixels by x-coordinate
    and selecting evenly-spaced points using linspace. Provides better
    spatial coverage than random sampling and naturally avoids border pixels.

    Originally from commit d5b4cda (the "magic run" with perfect tracking).

    Args:
        mask: Binary mask (H, W) with 1s indicating the object.
        num_points: Number of equidistant points to extract.

    Returns:
        List of [x, y] points, or None if mask is empty.
    """
    y_coords, x_coords = np.where(mask > 0)
    if len(y_coords) == 0:
        return None

    center_x = int(np.mean(x_coords))
    center_y = int(np.mean(y_coords))

    if num_points == 1:
        return [[center_x, center_y]]

    sorted_indices = np.argsort(x_coords)
    n_pixels = len(sorted_indices)

    indices = np.linspace(0, n_pixels - 1, num_points, dtype=int)

    points = []
    for idx in indices:
        sorted_idx = sorted_indices[idx]
        x = int(x_coords[sorted_idx])
        y = int(y_coords[sorted_idx])
        points.append([x, y])

    return points


def extract_equidistant_points_from_masks(
    masks: np.ndarray, num_points: int = 3
) -> np.ndarray:
    """
    Batch wrapper for extract_equidistant_points_from_mask().

    Args:
        masks: np.array with shape (N, H, W), binary masks.
        num_points: Number of points to extract per mask.

    Returns:
        points: np.array with shape (N, num_points, 2) in (x, y) format.
    """
    n = masks.shape[0]
    points = []
    for i in range(n):
        pts = extract_equidistant_points_from_mask(masks[i], num_points)
        if pts is None:
            points.append(np.zeros((num_points, 2)))
        else:
            points.append(np.array(pts))
    return np.array(points, dtype=np.float32)


def sample_points_from_masks(masks: np.ndarray, num_points: int = 3) -> np.ndarray:
    """
    Sample random points from mask-positive pixels and return absolute coordinates.

    Adapted from: IDEA-Research/Grounded-SAM-2
    Source: https://github.com/IDEA-Research/Grounded-SAM-2/blob/main/utils/track_utils.py
    Also available locally: Grounded-SAM-2-fork/utils/track_utils.py

    Args:
        masks: np.array with shape (N, H, W), binary masks.
        num_points: Number of points to sample per mask.

    Returns:
        points: np.array with shape (N, num_points, 2) in (x, y) format.
    """
    n, h, w = masks.shape
    points = []
    for i in range(n):
        indices = np.argwhere(masks[i] == 1)
        indices = indices[:, ::-1]  # (y, x) to (x, y)
        if len(indices) == 0:
            points.append(np.zeros((num_points, 2)))
            continue
        if len(indices) < num_points:
            sampled_indices = np.random.choice(len(indices), num_points, replace=True)
        else:
            sampled_indices = np.random.choice(len(indices), num_points, replace=False)
        sampled_points = indices[sampled_indices]
        points.append(sampled_points)
    points = np.array(points, dtype=np.float32)
    return points


def compute_bbox_iou(boxA, boxB) -> float:
    """
    Compute IoU between two bounding boxes in [x1, y1, x2, y2] format.

    Returns:
        IoU in [0, 1]
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    inter = interW * interH

    areaA = max(0, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    areaB = max(0, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    union = areaA + areaB - inter

    return inter / union if union > 0 else 0.0


def compute_pairwise_centroid_distances(centroids: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Euclidean distances between centroids.

    Args:
        centroids: (N, 2) array of [x, y] or [cx, cy] centroids.

    Returns:
        (N, N) symmetric distance matrix.
    """
    n = len(centroids)
    if n == 0:
        return np.array([])
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(centroids[i] - centroids[j])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
    return dist_matrix


def compute_clustering_coefficient(centroids: np.ndarray, threshold: float) -> float:
    """
    Fraction of centroid pairs within *threshold* distance.

    Args:
        centroids: (N, 2) array.
        threshold: distance (pixels or normalized coordinates).

    Returns:
        Float in [0, 1]. 0.0 when fewer than 2 objects.
    """
    n = len(centroids)
    if n < 2:
        return 0.0
    dists = compute_pairwise_centroid_distances(centroids)
    upper = dists[np.triu_indices(n, k=1)]
    if len(upper) == 0:
        return 0.0
    return float(np.sum(upper < threshold) / len(upper))


def compute_max_pairwise_iou(masks_np: np.ndarray) -> float:
    """Return max pixel-IoU over all pairs of binary masks (N, H, W)."""
    n = len(masks_np)
    max_iou = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            inter = float((masks_np[i] & masks_np[j]).sum())
            union = float((masks_np[i] | masks_np[j]).sum())
            if union > 0:
                max_iou = max(max_iou, inter / union)
    return max_iou


def reseed_tracker_memory(
    model, inference_session, frame_idx: int, masks_np: np.ndarray, device
):
    """
    Inject current binary masks as a fresh conditioning frame.

    Removes frame_idx from frames_tracked so forward() treats it
    as is_init_cond_frame=True, re-encoding the memory bank.

    Args:
        model: Sam3TrackerVideoModel instance.
        inference_session: Active inference session.
        frame_idx: Local frame index to reseed.
        masks_np: (N, H, W) bool/float numpy array of current masks.
        device: Torch device.
    """
    masks_tensor = torch.from_numpy(masks_np.astype(np.float32)).to(device)

    obj_ids = list(inference_session.obj_ids)[: len(masks_np)]
    for i, obj_id in enumerate(obj_ids):
        obj_idx = inference_session.obj_id_to_idx(obj_id)
        mask_t = masks_tensor[i].unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        inference_session.add_mask_inputs(obj_idx, frame_idx, mask_t)
        # Clear tracked status so forward treats this as an init-cond frame
        inference_session.frames_tracked_per_obj[obj_idx].pop(frame_idx, None)
        if obj_id not in inference_session.obj_with_new_inputs:
            inference_session.obj_with_new_inputs.append(obj_id)

    # Re-run forward for this frame — encodes fresh memory anchor
    model(inference_session, frame_idx=frame_idx, run_mem_encoder=True)


def process_tracking_outputs(outputs_per_frame):
    index_tuples = []
    bboxes = []
    counts_list = []
    sizes = []
    scores_list = []
    tracker_scores_list = []
    chunk_idx_list = []
    model_type_list = []
    is_chunk_start_list = []

    for frame_idx, proc in outputs_per_frame.items():
        object_ids = to_numpy(proc["object_ids"])
        boxes = to_numpy(proc["boxes"])
        masks = proc["masks"]  # keep lazy until conversion per-item
        scores = to_numpy(proc.get("scores", np.zeros(len(object_ids))))
        tracker_scores_dict = proc.get("obj_id_to_tracker_score") or {}

        # Chunk metadata (stamped by chunked pipeline, None for single-pass)
        chunk_idx = proc.get("_chunk_idx")
        model_type = proc.get("_model_type")
        is_chunk_start = proc.get("_is_chunk_start")

        for i, oid in enumerate(object_ids):
            # bbox -> list (x1,y1,x2,y2)
            bbox = boxes[i].tolist()

            # mask -> RLE
            mask_item = masks[i]
            # if mask already RLE-like (dict with 'counts'/'size'), use it
            if (
                isinstance(mask_item, dict)
                and "counts" in mask_item
                and "size" in mask_item
            ):
                rle = mask_item
                counts = rle["counts"]
                try:
                    counts = counts.decode("utf-8")
                except Exception:
                    pass
                size = rle["size"]
            else:
                m = to_numpy(mask_item).astype(np.uint8)
                # squeeze singleton channel dimension if present
                if m.ndim == 3 and m.shape[0] in (1,):
                    m = m.squeeze(0)
                # ensure Fortran order required by pycocotools
                rle = mask_util.encode(np.asfortranarray(m))
                counts = rle["counts"]
                try:
                    counts = counts.decode("ascii")
                except Exception:
                    pass
                size = rle["size"]

            score = float(scores[i])
            tracker_score = (
                float(tracker_scores_dict[int(oid)])
                if tracker_scores_dict and int(oid) in tracker_scores_dict
                else None
            )
            index_tuples.append((int(frame_idx), int(oid)))
            bboxes.append(bbox)
            counts_list.append(counts)
            sizes.append(size)
            scores_list.append(score)
            tracker_scores_list.append(tracker_score)
            chunk_idx_list.append(chunk_idx)
            model_type_list.append(model_type)
            is_chunk_start_list.append(is_chunk_start)

    mi = pd.MultiIndex.from_tuples(index_tuples, names=["frame_idx", "object_id"])
    df_results = pd.DataFrame(
        {
            "bbox": bboxes,
            "counts": counts_list,
            "size": sizes,
            "scores": scores_list,
            "tracker_score": tracker_scores_list,
            "chunk_idx": chunk_idx_list,
            "model_type": model_type_list,
            "is_chunk_start": is_chunk_start_list,
        },
        index=mi,
    )
    return df_results
