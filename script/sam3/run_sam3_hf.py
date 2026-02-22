"""
SAM3 HuggingFace Video Tracking Pipeline

Processes video in chunks:
- Chunk 0: Sam3VideoModel with text prompt (short, e.g. 15s)
- Chunks 1+: Sam3TrackerVideoModel with point prompts (longer, e.g. 45s)

Usage:
    python -m script.sam3.run_sam3_hf
"""

import argparse
import json
import re
import shutil
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from omegaconf import OmegaConf
from simpler_timer import SimplerTimer


def _early_init():
    """Parse config and set env vars BEFORE torch import."""
    parser = argparse.ArgumentParser(description="SAM3 HuggingFace Video Tracking")
    parser.add_argument(
        "--config",
        type=str,
        default="config/sam3_hf_config.yaml",
        help="Path to config file (default: config/sam3_hf_config.yaml)",
    )
    args, _ = parser.parse_known_args()

    from src.utils import load_config, set_env_vars

    cfg = load_config(args.config)
    set_env_vars(cfg)
    return args, cfg


# Call early init BEFORE importing torch/transformers
_args, _cfg = _early_init()

import torch  # noqa: E402

# Allow TF32 on Ampere+ GPUs — ~2x faster matmul with negligible precision loss
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from accelerate import Accelerator  # noqa: E402
from transformers import (  # noqa: E402
    Sam3TrackerVideoConfig,
    Sam3TrackerVideoModel,
    Sam3TrackerVideoProcessor,
    Sam3VideoConfig,
    Sam3VideoModel,
    Sam3VideoProcessor,
)

from src.metrics import (  # noqa: E402
    compute_per_frame_metrics,
    compute_per_run_metrics,
    compute_summary_metrics,
    per_frame_metrics_to_df,
    per_run_metrics_to_multiindex_df,
    summary_metrics_to_df,
)
from src.utils import (  # noqa: E402
    annotate_video_with_sam3_outputs,
    build_manual_chunks,
    chunk_video_frames_adaptive,
    chunk_video_frames_dual,
    create_run_directory,
    extract_equidistant_points_from_masks,
    find_frame_with_enough_objects,
    free_gpu_memory,
    free_system_memory,
    get_video_metadata,
    load_video_frames_range,
    process_tracking_outputs,
    sample_points_from_masks,
    setup_logger,
)
from src.viz import generate_all_visualizations  # noqa: E402
from src.yolo_prescan import run_yolo_prescan, yolo_prescan_to_df  # noqa: E402

# Import parameter sensitivity function for optional testing
try:
    from script.sam3.parameter_sensitivity import run_parameter_sensitivity_analysis
except ImportError:
    run_parameter_sensitivity_analysis = None  # noqa: E402

# ---------------------------------------------------------------------------
# Per-chunk processing helpers
# ---------------------------------------------------------------------------


def _compute_max_pairwise_iou(masks_np: np.ndarray) -> float:
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


def _find_best_prescreen_frame(
    prescreen_outputs: dict,
    min_objects: int = 3,
    method: str = "combined",
) -> tuple[int | None, list, list, list]:
    """
    Select best frame from prescreen outputs by detection quality.
    Filters frames with >= min_objects, then ranks by:
      "min_occlusion"  — lowest max pairwise IoU
      "best_scores"    — highest mean detection score
      "combined"       — low-occlusion frames first (max_iou < 0.10),
                         then by descending mean score within that tier
    Returns (frame_idx, masks_list, boxes_list, object_ids_list)
    or (None, [], [], []) if no frame qualifies.
    """
    from src.utils import get_all_objects_from_results

    candidates = []
    for frame_idx, results in prescreen_outputs.items():
        masks_list, boxes_list, object_ids_list = get_all_objects_from_results(results)
        if len(object_ids_list) < min_objects:
            continue
        masks_array = np.stack(
            [m.squeeze(0) if m.ndim == 3 and m.shape[0] == 1 else m for m in masks_list]
        ).astype(bool)
        max_iou = _compute_max_pairwise_iou(masks_array)

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


def _match_prescreen_ids_to_previous(
    prescreen_masks: list,
    prescreen_ids: list,
    prev_masks: list,
    prev_ids: list,
    iou_threshold: float = 0.10,
) -> dict[int, int]:
    """
    Greedy IoU-based assignment of prescreen IDs to previous-chunk IDs.
    Builds P×Q IoU matrix, iteratively picks highest-IoU pair, removes both
    from pool, stops when below iou_threshold. Unmatched prescreen IDs keep
    their original value in the returned dict.
    """
    if not prescreen_masks or not prev_masks:
        return {int(pid): int(pid) for pid in prescreen_ids}

    def _to_bool(m):
        arr = m.squeeze(0) if m.ndim == 3 and m.shape[0] == 1 else m
        return arr.astype(bool)

    ps_masks = [_to_bool(m) for m in prescreen_masks]
    pv_masks = [_to_bool(m) for m in prev_masks]

    P, Q = len(ps_masks), len(pv_masks)
    iou_matrix = np.zeros((P, Q), dtype=np.float32)
    for i in range(P):
        for j in range(Q):
            inter = float((ps_masks[i] & pv_masks[j]).sum())
            union = float((ps_masks[i] | pv_masks[j]).sum())
            iou_matrix[i, j] = inter / union if union > 0 else 0.0

    id_map: dict[int, int] = {}
    assigned_ps = set()
    assigned_pv = set()

    flat_indices = np.argsort(iou_matrix.ravel())[::-1]
    for flat_idx in flat_indices:
        i, j = divmod(int(flat_idx), Q)
        if iou_matrix[i, j] < iou_threshold:
            break
        if i in assigned_ps or j in assigned_pv:
            continue
        id_map[int(prescreen_ids[i])] = int(prev_ids[j])
        assigned_ps.add(i)
        assigned_pv.add(j)

    # Unmatched prescreen IDs pass through unchanged
    for i, pid in enumerate(prescreen_ids):
        if int(pid) not in id_map:
            id_map[int(pid)] = int(pid)

    return id_map


def _run_text_prescreen(
    chunk_frames: list,
    global_start_idx: int,
    prescreen_frames: int,
    cfg,
    device,
) -> dict:
    """
    Run Sam3VideoModel on chunk_frames[:prescreen_frames] for fresh detections.
    Clamps prescreen_frames to len(chunk_frames). Frees GPU memory before returning.
    """
    n = min(prescreen_frames, len(chunk_frames))
    outputs = _process_video_chunk(chunk_frames[:n], global_start_idx, cfg, device)
    free_gpu_memory()
    return outputs


def _reseed_tracker_memory(
    model, inference_session, frame_idx: int, masks_np: np.ndarray, device
):
    """
    Inject current binary masks as a fresh conditioning frame.
    Removes frame_idx from frames_tracked so forward() treats it
    as is_init_cond_frame=True, re-encoding the memory bank.
    masks_np: (N, H, W) bool/float numpy array.
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


def _process_video_chunk(chunk_frames, start_idx, cfg, device):
    """
    Process a chunk with Sam3VideoModel (text-prompted segmentation).

    Loads model fresh, runs inference, cleans up. Returns dict mapping
    global frame indices to processed output dicts.
    """
    text_prompt = cfg.text_prompt
    custom_resolution = cfg.get("custom_resolution", None)

    # Load config for tracking parameter overrides
    config = Sam3VideoConfig.from_pretrained("facebook/sam3")

    if custom_resolution is not None:
        config.image_size = custom_resolution

        model = Sam3VideoModel.from_pretrained("facebook/sam3", config=config).to(
            device, dtype=torch.bfloat16
        )
        processor = Sam3VideoProcessor.from_pretrained(
            "facebook/sam3",
            size={"height": custom_resolution, "width": custom_resolution},
        )
        logger.info(
            f"Custom resolution {custom_resolution}x{custom_resolution} applied to Sam3Video model and processor"
        )
    else:
        model = Sam3VideoModel.from_pretrained("facebook/sam3", config=config).to(
            device, dtype=torch.bfloat16
        )
        processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")

    tracking_cfg = cfg.get("tracking", {})
    for key in [
        "init_trk_keep_alive",
        "max_trk_keep_alive",
        "min_trk_keep_alive",
        "trk_assoc_iou_thresh",
        "hotstart_dup_thresh",
        "suppress_overlap_thresh",
        "recondition_every_nth_frame",
    ]:
        val = tracking_cfg.get(key)
        if val is not None:
            setattr(config, key, val)

    # Initialize video inference session
    inference_session = processor.init_video_session(
        video=chunk_frames,
        inference_device=device,
        processing_device="cpu",
        video_storage_device="cpu",
        dtype=torch.bfloat16,
    )

    # Add text prompt
    inference_session = processor.add_text_prompt(
        inference_session=inference_session,
        text=text_prompt,
    )

    # Process all frames
    total_frames = len(chunk_frames)
    outputs_per_frame = {}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for model_outputs in model.propagate_in_video_iterator(
            inference_session=inference_session,
            max_frame_num_to_track=total_frames,
        ):
            processed_outputs = processor.postprocess_outputs(
                inference_session, model_outputs
            )
            # Preserve raw tracking fields
            processed_outputs["obj_id_to_tracker_score"] = dict(
                model_outputs.obj_id_to_tracker_score
            )
            processed_outputs["removed_obj_ids"] = set(model_outputs.removed_obj_ids)
            processed_outputs["suppressed_obj_ids"] = set(
                model_outputs.suppressed_obj_ids
            )
            global_frame_idx = start_idx + model_outputs.frame_idx
            outputs_per_frame[global_frame_idx] = processed_outputs
            local_frame_idx = model_outputs.frame_idx
            if local_frame_idx % 25 == 0 or local_frame_idx == total_frames - 1:
                logger.info(
                    f"  [text] frame {local_frame_idx + 1}/{total_frames} "
                    f"({100 * (local_frame_idx + 1) / total_frames:.0f}%)"
                )

    # Cleanup
    if hasattr(inference_session, "reset_inference_session"):
        inference_session.reset_inference_session()
    del inference_session, model, processor

    return outputs_per_frame


def _process_tracker_chunk(chunk_frames, start_idx, all_prompt_points, cfg, device):
    """
    Process a chunk with Sam3TrackerVideoModel (point-prompted tracking).

    Loads model fresh, runs inference with point prompts for all objects,
    cleans up. Returns dict mapping global frame indices to output dicts
    normalized to match Sam3VideoModel output format.
    """
    custom_resolution = cfg.get("custom_resolution", None)

    config = Sam3TrackerVideoConfig.from_pretrained("facebook/sam3")

    tracking_cfg = cfg.get("tracking", {})
    for key in [
        "init_trk_keep_alive",
        "max_trk_keep_alive",
        "min_trk_keep_alive",
        "trk_assoc_iou_thresh",
        "hotstart_dup_thresh",
        "suppress_overlap_thresh",
        "recondition_every_nth_frame",
    ]:
        val = tracking_cfg.get(key)
        if val is not None:
            setattr(config, key, val)

    reseed_cfg = cfg.get("mid_chunk_reseed", {})
    reseed_enabled = reseed_cfg.get("enabled", False)
    reseed_max_iou = reseed_cfg.get("max_iou_threshold", 0.08)
    reseed_window = reseed_cfg.get("window_frames", 10)
    reseed_cooldown = reseed_cfg.get("cooldown_frames", 50)

    if custom_resolution is not None:
        config.image_size = custom_resolution

        model = Sam3TrackerVideoModel.from_pretrained(
            "facebook/sam3", config=config
        ).to(device, dtype=torch.bfloat16)
        processor = Sam3TrackerVideoProcessor.from_pretrained(
            "facebook/sam3",
            size={"height": custom_resolution, "width": custom_resolution},
        )
        logger.info(
            f"Custom resolution {custom_resolution}x{custom_resolution} applied to Sam3TrackerVideo model and processor"
        )
    else:
        model = Sam3TrackerVideoModel.from_pretrained(
            "facebook/sam3", config=config
        ).to(device, dtype=torch.bfloat16)
        processor = Sam3TrackerVideoProcessor.from_pretrained("facebook/sam3")

    # Initialize video inference session
    inference_session = processor.init_video_session(
        video=chunk_frames,
        inference_device=device,
        processing_device="cpu",
        video_storage_device="cpu",
        dtype=torch.bfloat16,
    )

    # Add point inputs for ALL objects on first frame
    ann_frame_idx = 0
    obj_ids = list(all_prompt_points.keys())

    input_points = [[all_prompt_points[oid] for oid in obj_ids]]
    input_labels = [[[1] * len(all_prompt_points[oid]) for oid in obj_ids]]

    logger.info(f"Adding point prompts for {len(obj_ids)} objects on frame 0")
    processor.add_inputs_to_inference_session(
        inference_session=inference_session,
        frame_idx=ann_frame_idx,
        obj_ids=obj_ids,
        input_points=input_points,
        input_labels=input_labels,
    )

    # Run inference on first frame to initialize tracking
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        _ = model(
            inference_session=inference_session,
            frame_idx=ann_frame_idx,
        )

    # Rolling state for mid-chunk re-seeding
    iou_window = deque(maxlen=reseed_window)
    frames_since_reseed = (
        reseed_cooldown  # start "ready" — allows first eligible reseed immediately
    )

    # Process all frames
    total_frames = len(chunk_frames)
    outputs_per_frame = {}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for tracker_output in model.propagate_in_video_iterator(
            inference_session, show_progress_bar=False
        ):
            # Post-process masks
            video_res_masks = processor.post_process_masks(
                [tracker_output.pred_masks],
                original_sizes=[
                    [
                        inference_session.video_height,
                        inference_session.video_width,
                    ]
                ],
                binarize=True,
            )[0]

            # Convert to numpy
            masks_np = (
                video_res_masks.cpu().numpy()
                if isinstance(video_res_masks, torch.Tensor)
                else video_res_masks
            )
            # (N, 1, H, W) -> (N, H, W)
            if masks_np.ndim == 4:
                masks_np = masks_np.squeeze(1)

            local_frame_idx = tracker_output.frame_idx

            # --- mid-chunk re-seeding ---
            if reseed_enabled and masks_np.shape[0] >= 2:
                current_max_iou = _compute_max_pairwise_iou(masks_np)
                iou_window.append(current_max_iou)
                frames_since_reseed += 1

                window_full = len(iou_window) == reseed_window
                window_stable = (
                    window_full and (sum(iou_window) / reseed_window) < reseed_max_iou
                )
                cooldown_ok = frames_since_reseed >= reseed_cooldown
                not_occluded = current_max_iou < reseed_max_iou

                if window_stable and cooldown_ok and not_occluded:
                    _reseed_tracker_memory(
                        model, inference_session, local_frame_idx, masks_np, device
                    )
                    frames_since_reseed = 0
                    iou_window.clear()
                    logger.info(
                        f"  [tracker] re-seeded memory at local frame {local_frame_idx} "
                        f"(max_iou={current_max_iou:.3f})"
                    )

            # Compute bounding boxes from masks
            boxes = []
            for m in masks_np:
                ys, xs = np.where(m > 0)
                if len(ys) > 0:
                    boxes.append(
                        [
                            float(xs.min()),
                            float(ys.min()),
                            float(xs.max()),
                            float(ys.max()),
                        ]
                    )
                else:
                    boxes.append([0.0, 0.0, 0.0, 0.0])

            # Use actual tracked object IDs from session
            tracked_obj_ids = (
                list(inference_session.obj_ids)
                if hasattr(inference_session, "obj_ids")
                else obj_ids
            )

            # Convert object_score_logits → probabilities via sigmoid
            # These are raw logits from the mask decoder's obj_score_head;
            # the video model applies sigmoid internally before storing as
            # tracker_score (see modeling_sam3_video.py).
            if tracker_output.object_score_logits is not None:
                obj_scores = (
                    torch.sigmoid(tracker_output.object_score_logits)
                    .squeeze(-1)
                    .float()
                    .cpu()
                    .numpy()
                )
            else:
                obj_scores = np.ones(len(masks_np))

            # Build obj_id_to_tracker_score dict from sigmoid scores
            active_ids = tracked_obj_ids[: len(masks_np)]
            active_scores = obj_scores[: len(masks_np)]
            obj_id_to_tracker_score = {
                int(oid): float(active_scores[i]) for i, oid in enumerate(active_ids)
            }

            global_frame_idx = start_idx + local_frame_idx
            outputs_per_frame[global_frame_idx] = {
                "masks": masks_np,
                "boxes": np.array(boxes),
                "object_ids": np.array(active_ids),
                "scores": active_scores,
                "obj_id_to_tracker_score": obj_id_to_tracker_score,
            }
            if local_frame_idx % 25 == 0 or local_frame_idx == total_frames - 1:
                logger.info(
                    f"  [tracker] frame {local_frame_idx + 1}/{total_frames} "
                    f"({100 * (local_frame_idx + 1) / total_frames:.0f}%)"
                )

    # Cleanup
    if hasattr(inference_session, "reset_inference_session"):
        inference_session.reset_inference_session()
    del inference_session, model, processor

    return outputs_per_frame


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sanitize_filename(name: str) -> str:
    """Sanitize a stem string for use as a directory name."""
    sanitized = re.sub(r"[^\w\-]", "_", name)
    sanitized = re.sub(r"_+", "_", sanitized)
    return sanitized.strip("_") or "video"


# ---------------------------------------------------------------------------
# Single-video pipeline
# ---------------------------------------------------------------------------


def _run_single_video(cfg, run_dir: Path, config_path: Path | None = None):
    """Run the full SAM3 tracking pipeline for a single video."""
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    job_type = cfg.job_type
    log_file = setup_logger(run_dir, job_type=job_type)

    logger.info("=" * 60)
    logger.info("SAM3 HuggingFace Video Tracking (Chunked Pipeline)")
    logger.info("=" * 60)

    if config_path is not None and config_path.exists():
        shutil.copy(config_path, run_dir / config_path.name)
        logger.info(f"Config copied to {run_dir / config_path.name}")
    if config_path is not None:
        logger.info(f"Loaded config file: {config_path}")
    logger.info("CONFIGURATION")
    logger.info(f"\n{OmegaConf.to_yaml(cfg, resolve=True)}")
    logger.info("=" * 60)
    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Log file: {log_file}")

    # Start total pipeline timer
    total_timer = SimplerTimer()

    # Read config values
    video_path = cfg.video_path

    # Read video metadata only (frames are loaded per-chunk to avoid holding the full video in RAM)
    device = Accelerator().device
    logger.info(f"Using device: {device}")
    logger.info(f"Reading video metadata: {video_path}")
    fps, total_frames = get_video_metadata(video_path)

    # Apply start_frame offset (before capping, so max_frames_to_track is relative to start_frame)
    start_frame = cfg.get("start_frame", 0)
    if start_frame > 0:
        if start_frame >= total_frames:
            logger.error(
                f"start_frame ({start_frame}) >= total_frames ({total_frames}), aborting"
            )
            return
        total_frames = total_frames - start_frame
        logger.info(
            f"Starting from frame {start_frame}: {total_frames} frames remaining"
        )

    # Cap total frames if max_frames_to_track is set
    max_frames = cfg.get("max_frames_to_track", 0)
    if max_frames and max_frames > 0:
        total_frames = min(total_frames, max_frames)
        logger.info(
            f"Capped to {total_frames} frames (max_frames_to_track={max_frames})"
        )

    # Read prescan/chunking flags early so manual chunking can override them
    prescan_only = cfg.get("prescan_only", False)
    use_adaptive_chunking = cfg.get("use_adaptive_chunking", False)

    # Compute fixed chunks (baseline, may be overridden by manual chunking)
    chunks = chunk_video_frames_dual(
        total_frames,
        fps,
        cfg.video_model_chunk_seconds,
        cfg.tracker_chunk_seconds,
    )

    # Manual chunking override — supports both a flat list (all videos) and
    # a dict keyed by basename (per-video, e.g. {"video1.mp4": [[0,375],...], ...})
    raw_manual_chunks = cfg.get("manual_chunk_frames", None)
    if raw_manual_chunks is None:
        manual_chunk_frames = None
    else:
        converted = OmegaConf.to_container(raw_manual_chunks, resolve=True)
        if isinstance(converted, dict):
            video_basename = Path(video_path).name
            manual_chunk_frames = converted.get(video_basename, None)
        else:
            manual_chunk_frames = converted
    use_manual_chunking = manual_chunk_frames is not None

    if use_manual_chunking:
        if prescan_only:
            logger.warning(
                "prescan_only is invalid with manual_chunk_frames — ignoring"
            )
            prescan_only = False
        if use_adaptive_chunking:
            logger.warning(
                "use_adaptive_chunking is invalid with manual_chunk_frames — ignoring"
            )
            use_adaptive_chunking = False
        if cfg.get("run_parameter_sensitivity", False):
            logger.warning(
                "run_parameter_sensitivity is invalid with manual_chunk_frames — ignoring"
            )
        chunks = build_manual_chunks(manual_chunk_frames)
        logger.info(f"Manual chunking: {len(chunks)} chunks from config")

    # YOLO prescan data (populated when prescan or adaptive chunking is enabled)
    yolo_prescan_metrics_df = None
    yolo_occlusion_periods = None

    # -----------------------------------------------------------------------
    # Run YOLO prescan if requested (prescan_only or use_adaptive_chunking)
    # -----------------------------------------------------------------------
    if prescan_only or use_adaptive_chunking:
        if prescan_only:
            logger.info("=" * 60)
            logger.info("PRESCAN-ONLY MODE — running YOLO pre-scan only")
            logger.info("=" * 60)
        else:
            logger.info("Running YOLO pre-scan for adaptive chunking...")

        yolo_cfg = cfg.get("yolo_prescan", {})
        yolo_result = run_yolo_prescan(
            video_path=video_path,
            fps=fps,
            total_frames=total_frames,
            device=str(device),
            model_name=yolo_cfg.get("model", "yolo11x.pt"),
            conf_thresh=yolo_cfg.get("conf_thresh", 0.25),
            iou_thresh=yolo_cfg.get("iou_thresh", 0.45),
            tracker_config=yolo_cfg.get("tracker_config", "data/yolo/bytetrack.yaml"),
            allowed_classes=(
                set(yolo_cfg.get("allowed_classes", []))
                if yolo_cfg.get("allowed_classes")
                else None
            ),
            window_seconds=yolo_cfg.get("window_seconds", 1.0),
            high_occlusion_threshold=yolo_cfg.get("high_occlusion_threshold", 0.3),
            occlusion_iou_threshold=yolo_cfg.get("occlusion_iou_threshold", 0.15),
            clustering_distance_threshold=yolo_cfg.get(
                "clustering_distance_threshold", 0.15
            ),
            output_video_path=run_dir / "yolo_tracking.mp4",
        )

        # Save YOLO prescan artifacts
        yolo_df = yolo_result["yolo_df"]
        prescan_results = yolo_result["prescan_results"]

        if not yolo_df.empty:
            yolo_tracking_path = run_dir / "yolo_tracking.parquet"
            yolo_df.to_parquet(yolo_tracking_path, index=False)
            logger.info(f"YOLO tracking saved to: {yolo_tracking_path}")

            prescan_metrics_df = yolo_prescan_to_df(
                prescan_results["per_frame_metrics"]
            )
            prescan_metrics_path = metrics_dir / "yolo_prescan_metrics.parquet"
            prescan_metrics_df.to_parquet(prescan_metrics_path, index=False)
            logger.info(f"YOLO prescan metrics saved to: {prescan_metrics_path}")

            # Store for visualization
            yolo_prescan_metrics_df = prescan_metrics_df
            yolo_occlusion_periods = prescan_results["occlusion_periods"]

        # Save summary as single-row Parquet
        prescan_summary_df = pd.DataFrame(
            [
                {
                    "occlusion_periods": str(prescan_results["occlusion_periods"]),
                    "transition_frames": str(
                        prescan_results["transition_frames"].tolist()
                    ),
                    "num_occlusion_periods": len(prescan_results["occlusion_periods"]),
                    "num_transition_frames": len(prescan_results["transition_frames"]),
                    "total_frames": prescan_results.get("total_frames", total_frames),
                    "video_duration_seconds": prescan_results.get(
                        "video_duration_seconds", total_frames / fps
                    ),
                    "fps": fps,
                    "model_name": yolo_result["model_name"],
                    "conf_thresh": yolo_result["conf_thresh"],
                    "iou_thresh": yolo_result["iou_thresh"],
                }
            ]
        )
        prescan_summary_path = metrics_dir / "yolo_prescan_summary.parquet"
        prescan_summary_df.to_parquet(prescan_summary_path, index=False)
        logger.info(f"YOLO prescan summary saved to: {prescan_summary_path}")
        logger.info(f"YOLO prescan output directory: {run_dir}")

        # Run parameter sensitivity testing if requested
        if cfg.get("run_parameter_sensitivity", False) and not yolo_df.empty:
            logger.info("")
            logger.info("=" * 60)
            logger.info("Running parameter sensitivity analysis...")
            logger.info("=" * 60)

            if run_parameter_sensitivity_analysis is not None:
                try:
                    video_path_for_sensitivity = (
                        Path(video_path) if video_path else None
                    )
                    run_parameter_sensitivity_analysis(
                        yolo_df=yolo_df,
                        run_dir=run_dir,
                        fps=fps,
                        total_frames=total_frames,
                        video_model_chunk_seconds=cfg.video_model_chunk_seconds,
                        tracker_chunk_seconds=cfg.tracker_chunk_seconds,
                        adaptive_min_chunk_seconds=cfg.get(
                            "adaptive_min_chunk_seconds", 15
                        ),
                        adaptive_max_chunk_seconds=cfg.get(
                            "adaptive_max_chunk_seconds", 90
                        ),
                        video_path=video_path_for_sensitivity,
                        generate_video=False,
                    )
                    logger.info("=" * 60)
                    logger.info("Parameter sensitivity analysis complete")
                    logger.info("=" * 60)
                    logger.info("")
                except Exception as e:
                    logger.error(f"Parameter sensitivity analysis failed: {e}")
                    logger.info("Continuing with main pipeline...")
                    logger.info("")
            else:
                logger.warning(
                    "Parameter sensitivity module not available (import failed)"
                )
                logger.info("")

        # Adjust chunk boundaries if adaptive chunking is enabled
        if use_adaptive_chunking and not yolo_df.empty:
            transition_frames = prescan_results["transition_frames"]
            occlusion_periods = prescan_results["occlusion_periods"]
            chunks = chunk_video_frames_adaptive(
                chunks,
                transition_frames,
                fps,
                occlusion_periods=occlusion_periods,
                min_chunk_seconds=cfg.get("adaptive_min_chunk_seconds", 15),
                max_chunk_seconds=cfg.get("adaptive_max_chunk_seconds", 90),
            )

        # Exit early if prescan-only mode
        if prescan_only:
            if not yolo_df.empty:
                # Build chunk_info for visualization
                chunk_info_for_viz = {
                    "chunks": [
                        {
                            "chunk_idx": i,
                            "frame_range": [s, e],
                            "model_type": (
                                "Sam3VideoModel"
                                if mtype == "video"
                                else "Sam3TrackerVideoModel"
                            ),
                        }
                        for i, (s, e, mtype) in enumerate(chunks)
                    ]
                }

                viz_dir = run_dir / "visualizations"
                generate_all_visualizations(
                    tracking_df=None,
                    per_frame_df=None,
                    output_dir=viz_dir,
                    fps=fps,
                    chunk_info=chunk_info_for_viz,
                    yolo_prescan_df=yolo_prescan_metrics_df,
                    yolo_occlusion_periods=yolo_occlusion_periods,
                )
                logger.info(f"Visualizations saved to: {viz_dir}")
            else:
                logger.warning("YOLO prescan returned no detections!")

            free_gpu_memory()
            total_timer.end()
            logger.info(f"Total prescan time: {total_timer.timestamp()}")
            logger.info("Prescan-only mode complete. Exiting.")
            return

    # Continue with SAM3 pipeline (only reached if not prescan_only)

    logger.info(f"Video: {total_frames} frames at {fps:.1f} FPS")
    _prescreen_on = OmegaConf.to_container(
        cfg.get("text_prescreen", {}), resolve=True
    ).get("enabled", False)
    if _prescreen_on:
        logger.info(
            f"Chunks: {len(chunks)} (all via video model text-prompt prescreen for {cfg.get('prescreen_frames')} frames -> video tracker model)"
        )
    else:
        logger.info(
            f"Chunks: {len(chunks)} ({chunks[0][2]} + {len(chunks) - 1} tracker)"
        )
    for i, (s, e, mtype) in enumerate(chunks):
        display_type = "prescreen→tracker" if _prescreen_on else mtype
        logger.info(
            f"  Chunk {i}: frames {s}-{e} ({display_type}, {(e - s) / fps:.1f}s)"
        )

    # -----------------------------------------------------------------------
    # Chunk processing loop
    # -----------------------------------------------------------------------
    all_outputs_per_frame = {}
    chunk_info_list = []
    previous_chunk_outputs = None
    results_path = run_dir / "tracking_outputs.parquet"
    prescreen_results_path = run_dir / "prescreen_outputs.parquet"

    for chunk_idx, (start_idx, end_idx, chunk_type) in enumerate(chunks):
        # Load only this chunk's frames from disk — avoids holding the full video in RAM
        global_chunk_start = start_frame + start_idx
        global_chunk_end = start_frame + end_idx
        chunk_frames = load_video_frames_range(
            video_path, global_chunk_start, global_chunk_end
        )
        num_frames = len(chunk_frames)

        # Start timing this chunk
        timer = SimplerTimer()

        if torch.cuda.is_available():
            alloc_mb = torch.cuda.memory_allocated() / 1024**2
            res_mb = torch.cuda.memory_reserved() / 1024**2
            logger.info(
                f"GPU memory before chunk {chunk_idx}: {alloc_mb:.1f} MB allocated, {res_mb:.1f} MB reserved"
            )
        logger.info(
            f"Processing chunk {chunk_idx}/{len(chunks) - 1}: "
            f"frames {start_idx}-{end_idx} ({chunk_type})"
        )

        chunk_info = {
            "chunk_idx": chunk_idx,
            "frame_range": [start_idx, end_idx],
            "model_type": None,
            "prompt_points": None,
            "num_objects_tracked": 0,
            "source_frame_idx": None,
            "fallback_reason": None,
            "prescreen_used": False,
            "prescreen_source_frame_idx": None,
            "prescreen_num_objects": 0,
            "prescreen_fallback_reason": None,
        }

        use_tracker = False
        all_prompt_points = {}

        prescreen_cfg = cfg.get("text_prescreen", {})
        prescreen_enabled = prescreen_cfg.get("enabled", False)

        if prescreen_enabled:
            # --- Text prescreen drives EVERY chunk: prescreen → tracker ---
            prescreen_outputs = _run_text_prescreen(
                chunk_frames,
                global_chunk_start,
                prescreen_cfg.get("prescreen_frames", 25),
                cfg,
                device,
            )
            ps_frame_idx, ps_masks, ps_boxes, ps_ids = _find_best_prescreen_frame(
                prescreen_outputs,
                min_objects=prescreen_cfg.get(
                    "min_objects", cfg.min_objects_for_tracking
                ),
                method=prescreen_cfg.get("best_frame_method", "combined"),
            )

            if ps_frame_idx is not None:
                # Remap fresh IDs to previous-chunk IDs (if a previous chunk exists)
                if previous_chunk_outputs is not None and prescreen_cfg.get(
                    "id_matching", True
                ):
                    _, prev_masks_for_iou, _, prev_ids_for_iou = (
                        find_frame_with_enough_objects(
                            previous_chunk_outputs,
                            min_objects=1,
                            max_lookback=cfg.max_lookback_frames,
                        )
                    )
                    if prev_masks_for_iou:
                        id_map = _match_prescreen_ids_to_previous(
                            ps_masks,
                            ps_ids,
                            prev_masks_for_iou,
                            prev_ids_for_iou,
                            iou_threshold=prescreen_cfg.get(
                                "id_match_iou_threshold", 0.10
                            ),
                        )
                    else:
                        id_map = {int(pid): int(pid) for pid in ps_ids}
                else:
                    id_map = {int(pid): int(pid) for pid in ps_ids}

                ps_masks_array = np.stack(
                    [
                        m.squeeze(0) if m.ndim == 3 and m.shape[0] == 1 else m
                        for m in ps_masks
                    ]
                )
                sampled = extract_equidistant_points_from_masks(
                    ps_masks_array, num_points=3
                )
                prescreen_prompt_points = {}
                for i, ps_id in enumerate(ps_ids):
                    mapped_id = id_map.get(int(ps_id), int(ps_id))
                    pts = sampled[i]
                    if pts.size > 0:
                        prescreen_prompt_points[mapped_id] = pts.tolist()

                if prescreen_prompt_points:
                    all_prompt_points = prescreen_prompt_points
                    use_tracker = True
                    chunk_info["model_type"] = "Sam3TrackerVideoModel"
                    chunk_info["prompt_points"] = {
                        str(k): v for k, v in all_prompt_points.items()
                    }
                    chunk_info["num_objects_tracked"] = len(all_prompt_points)
                    chunk_info["prescreen_used"] = True
                    chunk_info["prescreen_source_frame_idx"] = int(ps_frame_idx)
                    chunk_info["prescreen_num_objects"] = len(prescreen_prompt_points)
                    logger.info(
                        f"  [prescreen] fresh text-grounded points from frame {ps_frame_idx} "
                        f"({len(prescreen_prompt_points)} objects)"
                    )
                else:
                    chunk_info["prescreen_fallback_reason"] = "no_points_extracted"
                    logger.warning("[prescreen] point extraction failed")

            else:
                # No qualifying frame in prescreen — try prev-chunk fallback
                chunk_info["prescreen_fallback_reason"] = "insufficient_objects"
                logger.warning("[prescreen] no qualifying frame found")

                if (
                    prescreen_cfg.get("fallback_to_prev_chunk", True)
                    and previous_chunk_outputs is not None
                ):
                    source_frame_idx, masks_list, boxes_list, object_ids_list = (
                        find_frame_with_enough_objects(
                            previous_chunk_outputs,
                            min_objects=cfg.min_objects_for_tracking,
                            max_lookback=cfg.max_lookback_frames,
                        )
                    )
                    if source_frame_idx is not None and len(masks_list) > 0:
                        masks_array = np.stack(
                            [
                                m.squeeze(0) if m.ndim == 3 and m.shape[0] == 1 else m
                                for m in masks_list
                            ]
                        )
                        point_method = cfg.get("point_extraction_method", "equidistant")
                        if point_method == "equidistant":
                            sampled = extract_equidistant_points_from_masks(
                                masks_array, num_points=3
                            )
                        else:
                            sampled = sample_points_from_masks(
                                masks_array, num_points=3
                            )
                        for i, obj_id in enumerate(object_ids_list):
                            pts = sampled[i]
                            if pts.size > 0:
                                all_prompt_points[obj_id] = pts.tolist()

                        if all_prompt_points:
                            use_tracker = True
                            chunk_info["model_type"] = "Sam3TrackerVideoModel"
                            chunk_info["prompt_points"] = {
                                str(k): v for k, v in all_prompt_points.items()
                            }
                            chunk_info["num_objects_tracked"] = len(all_prompt_points)
                            chunk_info["source_frame_idx"] = int(source_frame_idx)
                            logger.warning(
                                "[prescreen] using previous-chunk points as fallback"
                            )
                        else:
                            chunk_info["fallback_reason"] = "could_not_extract_points"
                    else:
                        chunk_info["fallback_reason"] = "no_objects_found"

            # Incrementally save prescreen outputs (mirrors tracking_outputs.parquet pattern)
            ps_chunk_df = process_tracking_outputs(prescreen_outputs)
            ps_chunk_df = ps_chunk_df.sort_index()
            if prescreen_results_path.exists():
                ps_existing_df = pd.read_parquet(prescreen_results_path)
                ps_chunk_df = pd.concat([ps_existing_df, ps_chunk_df]).sort_index()
                del ps_existing_df
            ps_chunk_df.to_parquet(prescreen_results_path)
            del ps_chunk_df
            logger.info(f"Prescreen outputs saved to {prescreen_results_path} (chunk {chunk_idx})")

            del prescreen_outputs
            free_gpu_memory()

        else:
            # --- Original logic (prescreen disabled) ---
            if chunk_type == "tracker" and previous_chunk_outputs is not None:
                source_frame_idx, masks_list, boxes_list, object_ids_list = (
                    find_frame_with_enough_objects(
                        previous_chunk_outputs,
                        min_objects=cfg.min_objects_for_tracking,
                        max_lookback=cfg.max_lookback_frames,
                    )
                )
                if source_frame_idx is not None and len(masks_list) > 0:
                    masks_array = np.stack(
                        [
                            m.squeeze(0) if m.ndim == 3 and m.shape[0] == 1 else m
                            for m in masks_list
                        ]
                    )
                    point_method = cfg.get("point_extraction_method", "equidistant")
                    if point_method == "equidistant":
                        sampled = extract_equidistant_points_from_masks(
                            masks_array, num_points=3
                        )
                    else:
                        sampled = sample_points_from_masks(masks_array, num_points=3)
                    for i, obj_id in enumerate(object_ids_list):
                        pts = sampled[i]
                        if pts.size > 0:
                            all_prompt_points[obj_id] = pts.tolist()

                    if all_prompt_points:
                        use_tracker = True
                        chunk_info["model_type"] = "Sam3TrackerVideoModel"
                        chunk_info["prompt_points"] = {
                            str(k): v for k, v in all_prompt_points.items()
                        }
                        chunk_info["num_objects_tracked"] = len(all_prompt_points)
                        chunk_info["source_frame_idx"] = int(source_frame_idx)
                    else:
                        chunk_info["fallback_reason"] = "could_not_extract_points"
                else:
                    chunk_info["fallback_reason"] = "no_objects_found"

        if not use_tracker:
            if chunk_info["fallback_reason"] is None:
                chunk_info["fallback_reason"] = (
                    "first_chunk" if chunk_idx == 0 else "prescreen_failed_no_fallback"
                )
            chunk_info["model_type"] = "Sam3VideoModel"

        # --- Process chunk with appropriate model ---
        # start_frame offset ensures global frame indices match original video positions
        global_start_idx = global_chunk_start
        if use_tracker:
            chunk_outputs = _process_tracker_chunk(
                chunk_frames, global_start_idx, all_prompt_points, cfg, device
            )
        else:
            chunk_outputs = _process_video_chunk(
                chunk_frames, global_start_idx, cfg, device
            )

        # Stop timer and calculate metrics
        elapsed_seconds = timer.end()
        avg_sec_per_frame = elapsed_seconds / max(1, num_frames)
        fps_achieved = 1.0 / avg_sec_per_frame if avg_sec_per_frame > 0 else 0.0

        # Add timing info to chunk metadata
        chunk_info["timing"] = {
            "elapsed_seconds": round(elapsed_seconds, 3),
            "avg_seconds_per_frame": round(avg_sec_per_frame, 4),
            "fps": round(fps_achieved, 2),
            "num_frames": num_frames,
        }

        # Stamp metadata on each frame
        for frame_idx in chunk_outputs:
            chunk_outputs[frame_idx]["_chunk_idx"] = chunk_idx
            chunk_outputs[frame_idx]["_model_type"] = chunk_info["model_type"]
            chunk_outputs[frame_idx]["_is_chunk_start"] = frame_idx == start_idx

        # Incrementally write chunk results to parquet (fault-tolerant)
        chunk_df = process_tracking_outputs(chunk_outputs)
        chunk_df = chunk_df.sort_index()
        if results_path.exists():
            existing_df = pd.read_parquet(results_path)
            chunk_df = pd.concat([existing_df, chunk_df]).sort_index()
            del existing_df
        chunk_df.to_parquet(results_path)
        del chunk_df
        logger.info(f"Tracking results saved to {results_path} (chunk {chunk_idx})")

        all_outputs_per_frame.update(chunk_outputs)
        previous_chunk_outputs = chunk_outputs
        chunk_info_list.append(chunk_info)

        # Free this chunk's frames from RAM and GPU memory
        del chunk_frames
        free_system_memory(label=f"chunk {chunk_idx}")
        free_gpu_memory(log_stats=True)

        # Log timing metrics
        logger.info(
            f"Chunk {chunk_idx} timing: "
            f"{elapsed_seconds:.2f}s total, "
            f"{avg_sec_per_frame:.3f}s/frame, "
            f"{fps_achieved:.2f} FPS"
        )
        logger.info(
            f"Chunk {chunk_idx} complete: {len(chunk_outputs)} frames processed"
        )

    logger.info(f"All chunks complete: {len(all_outputs_per_frame)} total frames")

    # Save chunk info JSON
    chunk_info_path = run_dir / "chunk_info.json"
    with open(chunk_info_path, "w") as f:
        json.dump({"chunks": chunk_info_list}, f, indent=2)
    logger.info(f"Chunk info saved to: {chunk_info_path}")

    # -----------------------------------------------------------------------
    # Post-processing (uses all_outputs_per_frame)
    # -----------------------------------------------------------------------

    # Create annotated video
    logger.info("Creating annotated video...")
    annotated_video_path = run_dir / "annotated_video.mp4"
    annotate_video_with_sam3_outputs(
        source_path=video_path,
        target_path=str(annotated_video_path),
        outputs_per_frame=all_outputs_per_frame,
    )
    logger.info(f"Annotated video saved to: {annotated_video_path}")

    # Load tracking results (already written incrementally per chunk)
    logger.info(f"Loading tracking results from {results_path}...")
    df_results = pd.read_parquet(results_path)

    # Compute per-frame metrics (mask-based spatial/overlap/quality)
    logger.info("Computing per-frame metrics...")
    per_frame = compute_per_frame_metrics(all_outputs_per_frame)
    per_frame_df = per_frame_metrics_to_df(per_frame)
    per_frame_path = metrics_dir / "per_frame_metrics.parquet"
    per_frame_df.to_parquet(per_frame_path)
    logger.info(f"Per-frame metrics saved to: {per_frame_path}")

    # Compute summary metrics (with occlusion-aware ID switch detection)
    logger.info("Computing summary metrics...")
    summary_metrics = compute_summary_metrics(
        all_outputs_per_frame, per_frame_metrics=per_frame
    )
    summary_metrics_df = summary_metrics_to_df(summary_metrics)
    summary_path = metrics_dir / "summary_metrics.parquet"
    summary_metrics_df.to_parquet(summary_path)
    logger.info(f"Summary metrics saved to: {summary_path}")
    # Log summary as key-value pairs (single-row DF is unreadable in logs)
    max_key_len = max(len(k) for k in summary_metrics)
    summary_lines = "\n".join(
        f"  {k:<{max_key_len}}  {v}" for k, v in summary_metrics.items()
    )
    logger.info(f"Summary metrics:\n{summary_lines}")

    # Compute per-run (per-ID lifecycle) metrics
    logger.info("Computing per-run metrics...")
    per_run = compute_per_run_metrics(
        all_outputs_per_frame, low_count_threshold=3, iou_thresh=0.5
    )
    per_run_df = per_run_metrics_to_multiindex_df(per_run)
    per_run_path = metrics_dir / "per_id_metrics.parquet"
    per_run_df.to_parquet(per_run_path)
    logger.info(f"Per-run metrics saved to: {per_run_path}")
    # Log without list columns (frames/low_count_frames) that blow up width
    log_cols = [
        c for c in per_run_df.columns if c not in ("frames", "low_count_frames")
    ]
    logger.info(f"Per-run metrics:\n{per_run_df[log_cols].to_string()}")
    logger.info(f"Metrics directory: {metrics_dir}")

    # Generate visualizations
    logger.info("Generating visualizations...")
    vis_dir = run_dir / "visualizations"
    generate_all_visualizations(
        tracking_df=df_results,
        per_frame_df=per_frame_df,
        output_dir=vis_dir,
        fps=fps,
        chunk_info={"chunks": chunk_info_list},
        video_path=video_path,
        yolo_prescan_df=yolo_prescan_metrics_df,
        yolo_occlusion_periods=yolo_occlusion_periods,
    )
    logger.info(f"Visualizations saved to: {vis_dir}")

    # Stop total timer and log overall statistics
    total_elapsed = total_timer.end()
    total_fps = total_frames / total_elapsed if total_elapsed > 0 else 0.0

    logger.info("=" * 60)
    logger.info(f"Results saved to: {run_dir}")
    logger.info(
        f"Pipeline complete in {total_elapsed:.2f}s ({total_timer.timestamp()})"
    )
    logger.info(f"Overall throughput: {total_fps:.2f} FPS ({total_frames} frames)")
    logger.info("=" * 60)
    logger.info("Run complete.")


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------


def _run_batch(cfg, video_dir: Path, job_type: str):
    """Process all video files in video_dir, each in its own subdirectory."""
    VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
    video_files = sorted(
        f
        for f in video_dir.iterdir()
        if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not video_files:
        raise ValueError(f"No video files found in {video_dir}")

    # Shared batch directory (one timestamp for the whole batch)
    batch_dir = create_run_directory(Path(cfg.output_dir), job_type)
    config_path = Path(_args.config)

    for video_file in video_files:
        sanitized = _sanitize_filename(video_file.stem)
        video_run_dir = batch_dir / sanitized
        video_run_dir.mkdir(parents=True, exist_ok=True)
        (video_run_dir / "metrics").mkdir(parents=True, exist_ok=True)

        # Build per-video config with updated video_path
        video_cfg = OmegaConf.create(
            {**OmegaConf.to_container(cfg, resolve=True), "video_path": str(video_file)}
        )

        try:
            _run_single_video(video_cfg, video_run_dir, config_path=config_path)
        except Exception as e:
            logger.error(f"Failed processing {video_file.name}: {e}")
            continue


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    cfg = _cfg
    video_path = cfg.get("video_path", None)
    video_dir = cfg.get("video_dir", None)

    if video_path and video_dir:
        raise ValueError("Specify exactly one of video_path or video_dir, not both.")
    if not video_path and not video_dir:
        raise ValueError("Must specify either video_path or video_dir.")

    job_type = "yolo_prescan" if cfg.get("prescan_only", False) else cfg.job_type

    if video_dir:
        _run_batch(cfg, Path(video_dir), job_type)
    else:
        run_dir = create_run_directory(Path(cfg.output_dir), job_type)
        _run_single_video(cfg, run_dir, config_path=Path(_args.config))


if __name__ == "__main__":
    main()
