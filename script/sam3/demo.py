"""
SAM3 HuggingFace Video Tracking Script

Usage:
    python -m script.sam3.demo --config config/sam3_hf_config.yaml
"""

import argparse
import shutil
from pathlib import Path

from loguru import logger


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

    from script.sam3.utils import load_config, set_env_vars

    cfg = load_config(args.config)
    set_env_vars(cfg)
    return args, cfg


# Call early init BEFORE importing torch/transformers
_args, _cfg = _early_init()

import torch  # noqa: E402
from accelerate import Accelerator  # noqa: E402
from transformers import Sam3VideoConfig, Sam3VideoModel, Sam3VideoProcessor  # noqa: E402
from transformers.video_utils import load_video  # noqa: E402

from script.sam3.metrics import (  # noqa: E402
    compute_per_frame_metrics,
    compute_per_run_metrics,
    compute_summary_metrics,
    per_frame_metrics_to_df,
    per_run_metrics_to_multiindex_df,
    summary_metrics_to_df,
)
from script.sam3.utils import (  # noqa: E402
    annotate_video_with_sam3_outputs,
    create_run_directory,
    process_tracking_outputs,
    setup_logger,
)
from script.sam3.viz import generate_all_visualizations  # noqa: E402


def main():
    args = _args
    cfg = _cfg

    # Create timestamped run directory
    run_dir = create_run_directory(Path(cfg.output_dir), cfg.job_type)
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Setup logger (console + file in run dir)
    log_file = setup_logger(run_dir, job_type=cfg.job_type)

    # Copy config for reproducibility
    config_path = Path(args.config)
    shutil.copy(config_path, run_dir / config_path.name)
    logger.info(f"Config copied to {run_dir / config_path.name}")

    logger.info("=" * 60)
    logger.info("SAM3 HuggingFace Video Tracking")
    logger.info("=" * 60)
    logger.info(f"Config file: {args.config}")
    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Log file: {log_file}")

    # Read config values
    custom_resolution = cfg.custom_resolution
    max_frames_to_track = cfg.max_frames_to_track
    video_path = cfg.video_path
    text_prompt = cfg.text_prompt

    # Load model and processor
    device = Accelerator().device
    logger.info(f"Using device: {device}")

    logger.info(
        f"Loading model and processor (custom resolution: {custom_resolution}x{custom_resolution})..."
    )
    config = Sam3VideoConfig.from_pretrained("facebook/sam3")
    config.image_size = custom_resolution
    model = Sam3VideoModel.from_pretrained("facebook/sam3", config=config).to(
        device, dtype=torch.bfloat16
    )
    processor = Sam3VideoProcessor.from_pretrained(
        "facebook/sam3",
        size={"height": custom_resolution, "width": custom_resolution},
    )

    # Load video frames
    logger.info(f"Loading video: {video_path}")
    video_frames, fps = load_video(video_path)

    # Initialize video inference session
    logger.info("Initializing video inference session...")
    inference_session = processor.init_video_session(
        video=video_frames,
        inference_device=device,
        processing_device="cpu",
        video_storage_device="cpu",
        dtype=torch.bfloat16,
    )

    # Add text prompt to detect and track objects
    logger.info(f"Adding text prompt to inference session: {text_prompt}")
    inference_session = processor.add_text_prompt(
        inference_session=inference_session,
        text=text_prompt,
    )

    # Process all frames in the video
    outputs_per_frame = {}
    logger.info(f"Running propagation on {max_frames_to_track} frames...")

    for model_outputs in model.propagate_in_video_iterator(
        inference_session=inference_session,
        max_frame_num_to_track=max_frames_to_track,
    ):
        processed_outputs = processor.postprocess_outputs(
            inference_session, model_outputs
        )
        # Preserve raw tracking fields
        processed_outputs["obj_id_to_tracker_score"] = dict(
            model_outputs.obj_id_to_tracker_score
        )
        processed_outputs["removed_obj_ids"] = set(model_outputs.removed_obj_ids)
        processed_outputs["suppressed_obj_ids"] = set(model_outputs.suppressed_obj_ids)
        outputs_per_frame[model_outputs.frame_idx] = processed_outputs

    logger.info(f"Processed {len(outputs_per_frame)} frames")

    logger.info("Resetting inference session...")
    inference_session.reset_inference_session()

    # Create annotated video
    logger.info("Creating annotated video...")
    annotated_video_path = run_dir / "annotated_video.mp4"
    annotate_video_with_sam3_outputs(
        source_path=video_path,
        target_path=str(annotated_video_path),
        outputs_per_frame=outputs_per_frame,
    )
    logger.info(f"Annotated video saved to: {annotated_video_path}")

    # Save raw tracking results
    results_path = run_dir / "tracking_outputs.parquet"
    logger.info(f"Saving all per-frame outputs to {results_path}...")
    df_results = process_tracking_outputs(outputs_per_frame)
    df_results = df_results.sort_index()
    df_results.to_parquet(results_path)

    # Compute per-frame metrics (mask-based spatial/overlap/quality)
    logger.info("Computing per-frame metrics...")
    per_frame = compute_per_frame_metrics(outputs_per_frame)
    per_frame_df = per_frame_metrics_to_df(per_frame)
    per_frame_path = metrics_dir / "per_frame_metrics.parquet"
    per_frame_df.to_parquet(per_frame_path)
    logger.info(f"Per-frame metrics saved to: {per_frame_path}")

    # Compute summary metrics (with occlusion-aware ID switch detection)
    logger.info("Computing summary metrics...")
    summary_metrics = compute_summary_metrics(
        outputs_per_frame, per_frame_metrics=per_frame
    )
    summary_metrics_df = summary_metrics_to_df(summary_metrics)
    summary_path = metrics_dir / "summary_metrics.parquet"
    summary_metrics_df.to_parquet(summary_path)
    logger.info(f"Summary metrics saved to: {summary_path}")
    logger.info(f"Summary metrics:\n{summary_metrics_df}")

    # Compute per-run (per-ID lifecycle) metrics
    logger.info("Computing per-run metrics...")
    per_run = compute_per_run_metrics(
        outputs_per_frame, low_count_threshold=3, iou_thresh=0.5
    )
    per_run_df = per_run_metrics_to_multiindex_df(per_run)
    per_run_path = metrics_dir / "per_id_metrics.parquet"
    per_run_df.to_parquet(per_run_path)
    logger.info(f"Per-run metrics saved to: {per_run_path}")
    logger.info(f"Per-run metrics:\n{per_run_df}")

    # Generate visualizations
    logger.info("Generating visualizations...")
    vis_dir = run_dir / "visualizations"
    generate_all_visualizations(
        tracking_df=df_results,
        per_frame_df=per_frame_df,
        output_dir=vis_dir,
        fps=fps,
    )
    logger.info(f"Visualizations saved to: {vis_dir}")

    logger.info("Run complete.")


if __name__ == "__main__":
    main()
