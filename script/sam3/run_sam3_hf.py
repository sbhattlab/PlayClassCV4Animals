"""
SAM3 HuggingFace Video Tracking Script (Chunked Pipeline)

Processes video in chunks:
- Chunk 0: Sam3VideoModel with text prompt (short, e.g. 15s)
- Chunks 1+: Sam3TrackerVideoModel with point prompts (longer, e.g. 45s)

Usage:
    python -m script.sam3.run_sam3_hf
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from loguru import logger
from omegaconf import OmegaConf
from simpler_timer import SimplerTimer
from tqdm import tqdm


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
from accelerate import Accelerator  # noqa: E402
from transformers import (  # noqa: E402
    Sam3TrackerVideoConfig,
    Sam3TrackerVideoModel,
    Sam3TrackerVideoProcessor,
    Sam3VideoConfig,
    Sam3VideoModel,
    Sam3VideoProcessor,
)
from transformers.video_utils import load_video  # noqa: E402

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
    chunk_video_frames_dual,
    create_run_directory,
    extract_equidistant_points_from_masks,
    find_frame_with_enough_objects,
    free_gpu_memory,
    process_tracking_outputs,
    sample_points_from_masks,
    setup_logger,
)
from src.viz import generate_all_visualizations  # noqa: E402

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
    outputs_per_frame = {}
    with torch.inference_mode():
        for model_outputs in tqdm(
            model.propagate_in_video_iterator(
                inference_session=inference_session,
                max_frame_num_to_track=len(chunk_frames),
            ),
            total=len(chunk_frames),
            desc=f"Chunk (text, frames {start_idx}-{start_idx + len(chunk_frames)})",
            unit="frame",
            ncols=100,  # Fixed width to prevent wrapping
            ascii=True,  # Use ASCII chars for better compatibility
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

    if custom_resolution is not None:
        config = Sam3TrackerVideoConfig.from_pretrained("facebook/sam3")
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
        model = Sam3TrackerVideoModel.from_pretrained("facebook/sam3").to(
            device, dtype=torch.bfloat16
        )
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
    with torch.inference_mode():
        _ = model(
            inference_session=inference_session,
            frame_idx=ann_frame_idx,
        )

    # Process all frames
    outputs_per_frame = {}
    with torch.inference_mode():
        for tracker_output in model.propagate_in_video_iterator(
            inference_session, show_progress_bar=True
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
            # tracker_score (see modeling_sam3_video.py L1678-1679).
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

            global_frame_idx = start_idx + tracker_output.frame_idx
            outputs_per_frame[global_frame_idx] = {
                "masks": masks_np,
                "boxes": np.array(boxes),
                "object_ids": np.array(active_ids),
                "scores": active_scores,
                "obj_id_to_tracker_score": obj_id_to_tracker_score,
            }

    # Cleanup
    if hasattr(inference_session, "reset_inference_session"):
        inference_session.reset_inference_session()
    del inference_session, model, processor

    return outputs_per_frame


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = _args
    cfg = _cfg

    # Create timestamped run directory
    run_dir = create_run_directory(Path(cfg.output_dir), cfg.job_type)
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Setup logger (console + file in run dir)
    log_file = setup_logger(run_dir, job_type=cfg.job_type)

    logger.info("=" * 60)
    logger.info("SAM3 HuggingFace Video Tracking (Chunked Pipeline)")
    logger.info("=" * 60)
    # Copy config for reproducibility
    config_path = Path(args.config)
    shutil.copy(config_path, run_dir / config_path.name)
    logger.info(f"Config copied to {run_dir / config_path.name}")
    logger.info(f"Loaded config file: {args.config}")
    logger.info("CONFIGURATION")
    logger.info(f"\n{OmegaConf.to_yaml(cfg, resolve=True)}")
    logger.info("=" * 60)
    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Log file: {log_file}")

    # Start total pipeline timer
    total_timer = SimplerTimer()

    # Read config values
    video_path = cfg.video_path

    # Load video frames
    device = Accelerator().device
    logger.info(f"Using device: {device}")
    logger.info(f"Loading video: {video_path}")
    video_frames, video_metadata = load_video(video_path)
    fps = video_metadata.fps
    total_frames = len(video_frames)

    # Cap total frames if max_frames_to_track is set
    max_frames = cfg.get("max_frames_to_track", 0)
    if max_frames and max_frames > 0:
        total_frames = min(total_frames, max_frames)
        video_frames = video_frames[:total_frames]
        logger.info(
            f"Capped to {total_frames} frames (max_frames_to_track={max_frames})"
        )

    # Compute chunks
    chunks = chunk_video_frames_dual(
        total_frames,
        fps,
        cfg.video_model_chunk_seconds,
        cfg.tracker_chunk_seconds,
    )
    logger.info(f"Video: {total_frames} frames at {fps:.1f} FPS")
    logger.info(f"Chunks: {len(chunks)} ({chunks[0][2]} + {len(chunks) - 1} tracker)")
    for i, (s, e, mtype) in enumerate(chunks):
        logger.info(f"  Chunk {i}: frames {s}-{e} ({mtype}, {(e - s) / fps:.1f}s)")

    # -----------------------------------------------------------------------
    # Chunk processing loop
    # -----------------------------------------------------------------------
    all_outputs_per_frame = {}
    chunk_info_list = []
    previous_chunk_outputs = None

    for chunk_idx, (start_idx, end_idx, chunk_type) in enumerate(chunks):
        chunk_frames = video_frames[start_idx:end_idx]
        num_frames = len(chunk_frames)

        # Start timing this chunk
        timer = SimplerTimer()

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
        }

        use_tracker = False
        all_prompt_points = {}

        if chunk_type == "tracker" and previous_chunk_outputs is not None:
            # Try to extract point prompts from previous chunk
            source_frame_idx, masks_list, boxes_list, object_ids_list = (
                find_frame_with_enough_objects(
                    previous_chunk_outputs,
                    min_objects=cfg.min_objects_for_tracking,
                    max_lookback=cfg.max_lookback_frames,
                )
            )
            if source_frame_idx is not None and len(masks_list) > 0:
                # Stack masks to (N, H, W), squeeze (1, H, W) -> (H, W) if needed
                masks_array = np.stack(
                    [
                        m.squeeze(0) if m.ndim == 3 and m.shape[0] == 1 else m
                        for m in masks_list
                    ]
                )
                # Sample points from all masks -> (N, num_points, 2) in (x, y)
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
                chunk_info["fallback_reason"] = "first_chunk"
            chunk_info["model_type"] = "Sam3VideoModel"

        # --- Process chunk with appropriate model ---
        if use_tracker:
            chunk_outputs = _process_tracker_chunk(
                chunk_frames, start_idx, all_prompt_points, cfg, device
            )
        else:
            chunk_outputs = _process_video_chunk(chunk_frames, start_idx, cfg, device)

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

        all_outputs_per_frame.update(chunk_outputs)
        previous_chunk_outputs = chunk_outputs
        chunk_info_list.append(chunk_info)

        # Free GPU memory between chunks
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
    # Post-processing (same as before, using all_outputs_per_frame)
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

    # Save raw tracking results
    results_path = run_dir / "tracking_outputs.parquet"
    logger.info(f"Saving all per-frame outputs to {results_path}...")
    df_results = process_tracking_outputs(all_outputs_per_frame)
    df_results = df_results.sort_index()
    df_results.to_parquet(results_path)

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


if __name__ == "__main__":
    main()
