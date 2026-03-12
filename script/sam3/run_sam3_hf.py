"""
SAM3 HuggingFace Video Tracking and Segmentation Pipeline

Processes video in chunks:
- Grounding phase: Initialise text-prompt based multi-object detection with Sam3VideoModel for first N frames (e.g. 125 frames = 5s at 25fps) of each chunk
- Transition phase: Sample points from masks in 'best' frame (i.e. frames with highest detection confidence score and/or lowest degree of occlusion), maintain object ID's between chunks
- Tracking phase: Use sampled points and object ID's from transition phase to initialise point-based multi-object tracking with Sam3TrackerVideoModel

The process is heavily inspired by the Grounded SAM 2 pipeline, the main difference being that both the grounding and tracking stages are run with a Sam3-based model from HuggingFace, rather two separate models (i.e. groundedDINO + SAM2).

The pipeline is designed to be modular and extensible, with support for manual chunking, adaptive chunking based on YOLO scan results.

Usage:
    python -m script.sam3.run_sam3_hf
"""

import argparse
import json
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

import torch

# Allow TF32 on Ampere+ GPUs — ~2x faster matmul with negligible precision loss
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from accelerate import Accelerator
from transformers import (
    Sam3TrackerVideoConfig,
    Sam3TrackerVideoModel,
    Sam3TrackerVideoProcessor,
    Sam3VideoConfig,
    Sam3VideoModel,
    Sam3VideoProcessor,
)

from src.grounding import (
    find_best_grounding_frame,
    match_grounding_ids_to_previous,
    run_grounding,
)
from src.metrics import (
    compute_max_pairwise_iou,
    compute_per_frame_metrics,
    compute_per_run_metrics,
    compute_summary_metrics,
    per_frame_metrics_to_df,
    per_run_metrics_to_multiindex_df,
    summary_metrics_to_df,
)
from src.processing import (
    extract_equidistant_points_from_masks,
    find_frame_with_enough_objects,
    process_tracking_outputs,
    reseed_tracker_memory,
)
from src.utils import (
    build_manual_chunks,
    create_run_directory,
    free_gpu_memory,
    free_system_memory,
    get_video_metadata,
    load_chunks_from_chunk_info,
    load_video_frames_torchcodec,
    sanitize_filename,
    setup_logger,
)
from src.viz import (
    annotate_video_with_sam3_outputs,
    generate_all_visualizations,
)
from src.yolo_scan import (
    chunk_video_frames_adaptive,
    run_yolo_scan,
    yolo_scan_to_df,
)

# ---------------------------------------------------------------------------
# Per-chunk processing helpers
# ---------------------------------------------------------------------------


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
                current_max_iou = compute_max_pairwise_iou(masks_np)
                iou_window.append(current_max_iou)
                frames_since_reseed += 1

                window_full = len(iou_window) == reseed_window
                window_stable = (
                    window_full and (sum(iou_window) / reseed_window) < reseed_max_iou
                )
                cooldown_ok = frames_since_reseed >= reseed_cooldown
                not_occluded = current_max_iou < reseed_max_iou

                if window_stable and cooldown_ok and not_occluded:
                    reseed_tracker_memory(
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

    # Read scan/chunking flags early so manual chunking can override them
    yolo_scan_only = cfg.get("yolo_scan_only", False)
    use_adaptive_chunking = cfg.get("use_adaptive_chunking", False)

    # Reuse chunk boundaries from a previous run's chunk_info.json
    reuse_chunk_info = cfg.get("reuse_chunk_info", False)
    reuse_run_dir = cfg.get("reuse_run_dir", None)
    reused_prompt_points: dict[int, dict[int, list]] = {}
    if reuse_chunk_info:
        if not reuse_run_dir:
            raise ValueError("reuse_chunk_info requires reuse_run_dir to be set")
        reuse_run_path = Path(reuse_run_dir)
        logger.info(f"Reusing chunk boundaries from: {reuse_run_path}")
        reused_frame_pairs, reused_prompt_points = load_chunks_from_chunk_info(reuse_run_path)
        logger.info(f"Loaded {len(reused_frame_pairs)} chunks from previous run")
        if reused_prompt_points:
            logger.info(f"  {len(reused_prompt_points)} chunk(s) have stored prompt_points")
        # Copy chunk_info.json to new run dir for provenance
        src_chunk_info = reuse_run_path / "chunk_info.json"
        dst_chunk_info = run_dir / "reused_chunk_info.json"
        shutil.copy2(src_chunk_info, dst_chunk_info)
        logger.info(f"Copied chunk_info.json to: {dst_chunk_info}")

    # Manual chunking override — supports both a flat list (all videos) and
    # a dict keyed by basename (per-video, e.g. {"video1.mp4": [[0,375],...], ...})
    if reuse_chunk_info:
        manual_chunk_frames = reused_frame_pairs
    else:
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
        if yolo_scan_only:
            logger.warning(
                "yolo_scan_only is invalid with manual_chunk_frames — ignoring"
            )
            yolo_scan_only = False
        if use_adaptive_chunking:
            logger.warning(
                "use_adaptive_chunking is invalid with manual_chunk_frames — ignoring"
            )
            use_adaptive_chunking = False
        if cfg.get("run_parameter_sensitivity", False):
            logger.warning(
                "run_parameter_sensitivity is invalid with manual_chunk_frames — ignoring"
            )
        tracker_overrides = set(reused_prompt_points.keys()) if reused_prompt_points else None
        chunks = build_manual_chunks(manual_chunk_frames, tracker_override_indices=tracker_overrides)
        logger.info(f"Manual chunking: {len(chunks)} chunks from config")

    # YOLO scan data (populated when scan or adaptive chunking is enabled)
    yolo_scan_metrics_df = None
    yolo_occlusion_periods = None
    yolo_separation_windows = []
    yolo_scan_results = None

    # -----------------------------------------------------------------------
    # Run YOLO scan if requested (yolo_scan_only or use_adaptive_chunking)
    # -----------------------------------------------------------------------
    if not use_manual_chunking and (yolo_scan_only or use_adaptive_chunking):
        if yolo_scan_only:
            logger.info("=" * 60)
            logger.info("YOLO-SCAN-ONLY MODE — running YOLO scan only")
            logger.info("=" * 60)
        else:
            logger.info("Running YOLO scan for adaptive chunking...")

        yolo_cfg = cfg.get("yolo_scan", {})
        yolo_result = run_yolo_scan(
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
            separation_min_objects=yolo_cfg.get(
                "separation_min_objects", cfg.get("min_objects_for_tracking", 3)
            ),
            separation_min_distance=yolo_cfg.get("separation_min_distance", 0.15),
            separation_min_window_seconds=yolo_cfg.get(
                "separation_min_window_seconds", 1.0
            ),
            separation_gap_tolerance_frames=yolo_cfg.get(
                "separation_gap_tolerance_frames", 5
            ),
            output_video_path=run_dir / "yolo_tracking.mp4",
        )

        # Save YOLO scan artifacts
        yolo_df = yolo_result["yolo_df"]
        yolo_scan_results = yolo_result["scan_results"]

        if not yolo_df.empty:
            yolo_tracking_path = run_dir / "yolo_tracking.parquet"
            yolo_df.to_parquet(yolo_tracking_path, index=False)
            logger.info(f"YOLO tracking saved to: {yolo_tracking_path}")

            yolo_scan_metrics_df = yolo_scan_to_df(
                yolo_scan_results["per_frame_metrics"]
            )
            yolo_scan_metrics_path = metrics_dir / "yolo_scan_metrics.parquet"
            yolo_scan_metrics_df.to_parquet(yolo_scan_metrics_path, index=False)
            logger.info(f"YOLO scan metrics saved to: {yolo_scan_metrics_path}")

            # Store for visualization and chunking
            yolo_occlusion_periods = yolo_scan_results["occlusion_periods"]
            yolo_separation_windows = yolo_scan_results.get("separation_windows", [])

        # Save summary as single-row Parquet
        yolo_scan_summary_df = pd.DataFrame(
            [
                {
                    "occlusion_periods": str(yolo_scan_results["occlusion_periods"]),
                    "transition_frames": str(
                        yolo_scan_results["transition_frames"].tolist()
                    ),
                    "num_occlusion_periods": len(
                        yolo_scan_results["occlusion_periods"]
                    ),
                    "num_transition_frames": len(
                        yolo_scan_results["transition_frames"]
                    ),
                    "total_frames": yolo_scan_results.get("total_frames", total_frames),
                    "video_duration_seconds": yolo_scan_results.get(
                        "video_duration_seconds", total_frames / fps
                    ),
                    "fps": fps,
                    "model_name": yolo_result["model_name"],
                    "conf_thresh": yolo_result["conf_thresh"],
                    "iou_thresh": yolo_result["iou_thresh"],
                }
            ]
        )
        yolo_scan_summary_path = metrics_dir / "yolo_scan_summary.parquet"
        yolo_scan_summary_df.to_parquet(yolo_scan_summary_path, index=False)
        logger.info(f"YOLO scan summary saved to: {yolo_scan_summary_path}")
        logger.info(f"YOLO scan output directory: {run_dir}")

    # Compute chunks (non-manual path): single call handles fixed, adaptive, and
    # separation-based refinement based on available scan data
    if not use_manual_chunking:
        chunks = chunk_video_frames_adaptive(
            total_frames,
            fps,
            cfg.get("chunk_seconds", 60),
            separation_windows=yolo_separation_windows or None,
            per_frame_metrics=yolo_scan_results.get("per_frame_metrics")
            if yolo_scan_results
            else None,
            occlusion_periods=yolo_scan_results.get("occlusion_periods")
            if yolo_scan_results
            else None,
            transition_frames=yolo_scan_results.get("transition_frames")
            if yolo_scan_results
            else None,
            search_window_seconds=cfg.get("adaptive_search_window_seconds", 10.0),
            max_chunk_seconds=cfg.get("adaptive_max_chunk_seconds", 150),
        )

    # Enforce max_frames_to_track: clip/drop chunks beyond total_frames
    # (total_frames was already capped by max_frames_to_track above, so this
    # ensures manual, reused, and adaptive chunks all respect the limit)
    clipped_chunks = []
    for s, e, mtype in chunks:
        if s >= total_frames:
            break
        clipped_chunks.append((s, min(e, total_frames), mtype))
    if len(clipped_chunks) < len(chunks):
        logger.info(
            f"max_frames_to_track: clipped chunks from {len(chunks)} to {len(clipped_chunks)} "
            f"(total_frames={total_frames})"
        )
    chunks = clipped_chunks

    # Exit early if yolo-scan-only mode
    if yolo_scan_only:
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

            chunk_info_path = run_dir / "chunk_info.json"
            with open(chunk_info_path, "w") as f:
                json.dump(chunk_info_for_viz, f, indent=2)
            logger.info(f"Chunk info saved to: {chunk_info_path}")

            viz_dir = run_dir / "visualizations"
            generate_all_visualizations(
                tracking_df=None,
                per_frame_df=None,
                output_dir=viz_dir,
                fps=fps,
                chunk_info=chunk_info_for_viz,
                video_path=video_path,
                yolo_scan_df=yolo_scan_metrics_df,
                yolo_occlusion_periods=yolo_occlusion_periods,
            )
            logger.info(f"Visualizations saved to: {viz_dir}")
        else:
            logger.warning("YOLO scan returned no detections!")

        free_gpu_memory()
        total_timer.end()
        logger.info(f"Total scan time: {total_timer.timestamp()}")
        logger.info("YOLO-scan-only mode complete. Exiting.")
        return

    # Continue with SAM3 pipeline (only reached if not yolo_scan_only)

    logger.info(f"Video: {total_frames} frames at {fps:.1f} FPS")
    _grounding_on = OmegaConf.to_container(
        cfg.get("text_grounding", {}), resolve=True
    ).get("enabled", False)
    if _grounding_on:
        logger.info(
            f"Chunks: {len(chunks)} (all via video model text-prompt grounding for {cfg.get('grounding_frames')} frames -> video tracker model)"
        )
    else:
        logger.info(
            f"Chunks: {len(chunks)} ({chunks[0][2]} + {len(chunks) - 1} tracker)"
        )
    for i, (s, e, mtype) in enumerate(chunks):
        display_type = "grounding→tracker" if _grounding_on else mtype
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
    grounding_results_path = run_dir / "grounding_outputs.parquet"

    for chunk_idx, (start_idx, end_idx, chunk_type) in enumerate(chunks):
        # Load only this chunk's frames from disk — avoids holding the full video in RAM
        global_chunk_start = start_frame + start_idx
        global_chunk_end = start_frame + end_idx
        chunk_frames = load_video_frames_torchcodec(
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
            "grounding_used": False,
            "grounding_source_frame_idx": None,
            "grounding_num_objects": 0,
            "grounding_fallback_reason": None,
        }

        use_tracker = False
        all_prompt_points = {}
        grounding_frame_offset = 0  # local offset into chunk_frames for tracker init

        grounding_cfg = cfg.get("text_grounding", {})
        grounding_enabled = grounding_cfg.get("enabled", False)

        if reused_prompt_points and chunk_idx in reused_prompt_points:
            # --- Manual prompt points from reused chunk_info.json ---
            all_prompt_points = reused_prompt_points[chunk_idx]
            use_tracker = True
            chunk_info["model_type"] = "Sam3TrackerVideoModel"
            chunk_info["prompt_points"] = {str(k): v for k, v in all_prompt_points.items()}
            chunk_info["num_objects_tracked"] = len(all_prompt_points)
            chunk_info["manual_prompt_points"] = True
            logger.info(f"  [manual] Using reused prompt_points ({len(all_prompt_points)} objects)")

        elif grounding_enabled:
            # --- Text grounding drives EVERY chunk: grounding → tracker ---
            grounding_outputs = run_grounding(
                chunk_frames,
                global_chunk_start,
                grounding_cfg.get("grounding_frames", 25),
                _process_video_chunk,
                cfg,
                device,
            )
            gr_out_frame_idx, gr_out_masks, gr_out_boxes, gr_out_ids = (
                find_best_grounding_frame(
                    grounding_outputs,
                    min_objects=grounding_cfg.get(
                        "min_objects", cfg.min_objects_for_tracking
                    ),
                    method=grounding_cfg.get("best_frame_method", "combined"),
                )
            )

            if gr_out_frame_idx is not None:
                # Remap fresh IDs to previous-chunk IDs (if a previous chunk exists)
                if previous_chunk_outputs is not None and grounding_cfg.get(
                    "id_matching", True
                ):
                    _, prev_masks_for_iou, _, prev_ids_for_iou = (
                        find_frame_with_enough_objects(
                            previous_chunk_outputs,
                            # min_objects=3,
                            min_objects=cfg.min_objects_for_tracking,
                            max_lookback=cfg.max_lookback_frames,
                        )
                    )
                    if prev_masks_for_iou:
                        id_map = match_grounding_ids_to_previous(
                            gr_out_masks,
                            gr_out_ids,
                            prev_masks_for_iou,
                            prev_ids_for_iou,
                            iou_threshold=grounding_cfg.get(
                                "id_match_iou_threshold", 0.10
                            ),
                        )
                    else:
                        id_map = {int(gid): int(gid) for gid in gr_out_ids}
                else:
                    id_map = {int(gid): int(gid) for gid in gr_out_ids}

                gr_out_masks_array = np.stack(
                    [
                        m.squeeze(0) if m.ndim == 3 and m.shape[0] == 1 else m
                        for m in gr_out_masks
                    ]
                )
                sampled = extract_equidistant_points_from_masks(
                    gr_out_masks_array, num_points=3
                )
                grounding_prompt_points = {}
                for i, gr_out_id in enumerate(gr_out_ids):
                    mapped_id = id_map.get(int(gr_out_id), int(gr_out_id))
                    pts = sampled[i]
                    if pts.size > 0:
                        grounding_prompt_points[mapped_id] = pts.tolist()

                if grounding_prompt_points:
                    all_prompt_points = grounding_prompt_points
                    use_tracker = True
                    grounding_frame_offset = int(gr_out_frame_idx) - global_chunk_start
                    chunk_info["model_type"] = "Sam3TrackerVideoModel"
                    chunk_info["prompt_points"] = {
                        str(k): v for k, v in all_prompt_points.items()
                    }
                    chunk_info["num_objects_tracked"] = len(all_prompt_points)
                    chunk_info["grounding_used"] = True
                    chunk_info["grounding_source_frame_idx"] = int(gr_out_frame_idx)
                    chunk_info["grounding_num_objects"] = len(grounding_prompt_points)
                    if grounding_frame_offset > 0:
                        logger.info(
                            f"  [grounding] trimming {grounding_frame_offset} frame(s) from chunk start "
                            f"to align tracker init with best grounding frame {gr_out_frame_idx}"
                        )
                    logger.info(
                        f"  [grounding] fresh text-grounded points from frame {gr_out_frame_idx} "
                        f"({len(grounding_prompt_points)} objects)"
                    )
                else:
                    chunk_info["grounding_fallback_reason"] = "no_points_extracted"
                    logger.warning("[grounding] point extraction failed")

            else:
                # No qualifying frame in grounding — try prev-chunk fallback
                chunk_info["grounding_fallback_reason"] = "insufficient_objects"
                logger.warning("[grounding] no qualifying frame found")

                if (
                    grounding_cfg.get("fallback_to_prev_chunk", True)
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
                        sampled = extract_equidistant_points_from_masks(
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
                                "[grounding] using previous-chunk points as fallback"
                            )
                        else:
                            chunk_info["fallback_reason"] = "could_not_extract_points"
                    else:
                        chunk_info["fallback_reason"] = "no_objects_found"

            # Incrementally save grounding outputs (mirrors tracking_outputs.parquet pattern)
            gs_chunk_df = process_tracking_outputs(grounding_outputs)
            gs_chunk_df = gs_chunk_df.sort_index()
            if grounding_results_path.exists():
                gs_existing_df = pd.read_parquet(grounding_results_path)
                gs_chunk_df = pd.concat([gs_existing_df, gs_chunk_df])
                gs_chunk_df = gs_chunk_df[~gs_chunk_df.index.duplicated(keep="last")].sort_index()
                del gs_existing_df
            gs_chunk_df.to_parquet(grounding_results_path)
            del gs_chunk_df
            logger.info(
                f"Grounding outputs saved to {grounding_results_path} (chunk {chunk_idx})"
            )

            del grounding_outputs
            free_gpu_memory()

        else:
            # --- Original logic (grounding disabled) ---
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
                    sampled = extract_equidistant_points_from_masks(
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
                    else:
                        chunk_info["fallback_reason"] = "could_not_extract_points"
                else:
                    chunk_info["fallback_reason"] = "no_objects_found"

        if not use_tracker:
            if chunk_info["fallback_reason"] is None:
                chunk_info["fallback_reason"] = (
                    "first_chunk" if chunk_idx == 0 else "grounding_failed_no_fallback"
                )
            chunk_info["model_type"] = "Sam3VideoModel"

        # --- Process chunk with appropriate model ---
        # start_frame offset ensures global frame indices match original video positions
        global_start_idx = global_chunk_start + grounding_frame_offset
        if use_tracker:
            chunk_outputs = _process_tracker_chunk(
                chunk_frames[grounding_frame_offset:],
                global_start_idx,
                all_prompt_points,
                cfg,
                device,
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
            chunk_df = pd.concat([existing_df, chunk_df])
            chunk_df = chunk_df[~chunk_df.index.duplicated(keep="last")].sort_index()
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
        yolo_scan_df=yolo_scan_metrics_df,
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
    VIDEO_EXTENSIONS = {
        ext for e in [".mp4", ".avi", ".mov", ".mkv", ".m4v"] for ext in (e, e.upper())
    }
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

    reuse_chunk_info = cfg.get("reuse_chunk_info", False)
    reuse_run_dir_base = Path(cfg.reuse_run_dir) if cfg.get("reuse_run_dir", None) else None

    for video_file in video_files:
        sanitized = sanitize_filename(video_file.stem)
        video_run_dir = batch_dir / sanitized
        video_run_dir.mkdir(parents=True, exist_ok=True)
        (video_run_dir / "metrics").mkdir(parents=True, exist_ok=True)

        # Build per-video config with updated video_path
        video_cfg = OmegaConf.create(
            {**OmegaConf.to_container(cfg, resolve=True), "video_path": str(video_file)}
        )

        # Resolve per-video reuse_run_dir in batch mode
        if reuse_chunk_info and reuse_run_dir_base:
            per_video_reuse = reuse_run_dir_base / sanitized
            if (per_video_reuse / "chunk_info.json").exists():
                video_cfg.reuse_run_dir = str(per_video_reuse)
            else:
                logger.warning(
                    f"No chunk_info.json for {video_file.name} at {per_video_reuse}, skipping reuse"
                )
                video_cfg.reuse_chunk_info = False

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

    job_type = "yolo_scan" if cfg.get("yolo_scan_only", False) else cfg.job_type

    if video_dir:
        _run_batch(cfg, Path(video_dir), job_type)
    else:
        run_dir = create_run_directory(Path(cfg.output_dir), job_type)
        _run_single_video(cfg, run_dir, config_path=Path(_args.config))


if __name__ == "__main__":
    main()
