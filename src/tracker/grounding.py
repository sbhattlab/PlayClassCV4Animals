"""
Grounding and ID-matching utilities for SAM3 chunked tracking.

Functions for running text-prompted grounding, selecting the best grounding
frame, and matching grounding IDs across chunks.
"""

from typing import Callable

import numpy as np
from loguru import logger

from src.memory import free_gpu_memory
from src.metrics import compute_max_pairwise_iou
from src.tracker.masks import get_all_objects_from_results


def run_grounding(
    chunk_frames: list,
    global_start_idx: int,
    grounding_frames: int,
    process_video_chunk_fn: Callable,
    cfg,
    device,
) -> dict:
    """
    Run Sam3VideoModel on chunk_frames[:grounding_frames] for fresh detections.

    Clamps grounding_frames to len(chunk_frames). Frees GPU memory before
    returning.


    Args:
        chunk_frames: List of RGB frames for the current chunk.
        global_start_idx: Global frame index of the first frame in chunk_frames.
        grounding_frames: Number of frames to run grounding on.
        process_video_chunk_fn: Callable matching the signature of
            _process_video_chunk(chunk_frames, start_idx, cfg, device).
        cfg: OmegaConf config.
        device: Torch device.

    Returns:
        Dict mapping global frame indices to processed output dicts.
    """
    n = min(grounding_frames, len(chunk_frames))
    outputs = process_video_chunk_fn(chunk_frames[:n], global_start_idx, cfg, device)
    free_gpu_memory()
    return outputs


def find_best_grounding_frame(
    grounding_outputs: dict,
    min_objects: int = 3,
    method: str = "combined",
) -> tuple[int | None, list, list, list]:
    """
    Select best frame from grounding outputs by detection quality.

    Filters frames with >= min_objects, then ranks by:
      "min_occlusion"  — lowest max pairwise IoU
      "best_scores"    — highest mean detection score
      "combined"       — low-occlusion frames first (max_iou < 0.10),
                         then by descending mean score within that tier

    Returns:
        (frame_idx, masks_list, boxes_list, object_ids_list) or
        (None, [], [], []) if no frame qualifies.
    """
    candidates = []
    for frame_idx, results in grounding_outputs.items():
        masks_list, boxes_list, object_ids_list = get_all_objects_from_results(results)
        if len(object_ids_list) < min_objects:
            continue
        masks_array = np.stack(
            [m.squeeze(0) if m.ndim == 3 and m.shape[0] == 1 else m for m in masks_list]
        ).astype(bool)
        max_iou = compute_max_pairwise_iou(masks_array)

        scores = results.get("scores")
        if scores is not None:
            scores_arr = scores if isinstance(scores, np.ndarray) else np.array(scores)
            mean_score = float(scores_arr.mean()) if scores_arr.size > 0 else 0.0
        else:
            tracker_scores = results.get("obj_id_to_tracker_score", {})
            if tracker_scores:
                mean_score = float(np.mean(list(tracker_scores.values())))
            else:
                mean_score = 0.0

        candidates.append(
            (frame_idx, masks_list, boxes_list, object_ids_list, max_iou, mean_score)
        )

    if not candidates:
        return None, [], [], []

    if method == "min_occlusion":
        candidates.sort(key=lambda x: x[4])
    elif method == "best_scores":
        candidates.sort(key=lambda x: -x[5])
    else:  # "combined"
        candidates.sort(key=lambda x: (x[4] >= 0.10, -x[5]))

    best = candidates[0]
    return best[0], best[1], best[2], best[3]


def match_grounding_ids_to_previous(
    grounding_masks: list,
    grounding_ids: list,
    prev_masks: list,
    prev_ids: list,
    iou_threshold: float = 0.10,
) -> dict[int, int]:
    """
    Greedy IoU-based assignment of grounding IDs to previous-chunk IDs.

    Builds P×Q IoU matrix, iteratively picks highest-IoU pair, removes both
    from pool, stops when below iou_threshold. Unmatched grounding IDs keep
    their original value if it does not collide with an already-mapped ID;
    otherwise they are assigned a fresh ID to avoid silent key collisions in
    grounding_prompt_points.
    """
    if not grounding_masks or not prev_masks:
        return {int(pid): int(pid) for pid in grounding_ids}

    def _to_bool(m):
        arr = m.squeeze(0) if m.ndim == 3 and m.shape[0] == 1 else m
        return arr.astype(bool)

    gs_masks = [_to_bool(m) for m in grounding_masks]
    pv_masks = [_to_bool(m) for m in prev_masks]

    P, Q = len(gs_masks), len(pv_masks)
    iou_matrix = np.zeros((P, Q), dtype=np.float32)
    for i in range(P):
        for j in range(Q):
            inter = float((gs_masks[i] & pv_masks[j]).sum())
            union = float((gs_masks[i] | pv_masks[j]).sum())
            iou_matrix[i, j] = inter / union if union > 0 else 0.0

    id_map: dict[int, int] = {}
    assigned_gs = set()
    assigned_pv = set()

    flat_indices = np.argsort(iou_matrix.ravel())[::-1]
    for flat_idx in flat_indices:
        i, j = divmod(int(flat_idx), Q)
        if iou_matrix[i, j] < iou_threshold:
            break
        if i in assigned_gs or j in assigned_pv:
            continue
        id_map[int(grounding_ids[i])] = int(prev_ids[j])
        assigned_gs.add(i)
        assigned_pv.add(j)

    # Unmatched grounding IDs: pass through with original value, but remap to a
    # fresh ID if that value is already claimed by a matched assignment to avoid
    # silent dict-key collisions in grounding_prompt_points.
    claimed = set(id_map.values())
    next_fresh = max(claimed | {int(pid) for pid in grounding_ids}, default=-1) + 1
    for i, pid in enumerate(grounding_ids):
        if int(pid) not in id_map:
            if int(pid) not in claimed:
                id_map[int(pid)] = int(pid)
                claimed.add(int(pid))
            else:
                id_map[int(pid)] = next_fresh
                claimed.add(next_fresh)
                next_fresh += 1

    return id_map
