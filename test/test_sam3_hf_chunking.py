"""
SAM3 HuggingFace Video Chunking Test Script

This is a shortened version for testing the multi-object tracking workflow.
Uses smaller chunks (34 seconds) for faster iteration.

Usage:
    CUDA_VISIBLE_DEVICES=1 python -m test.test_sam3_hf_chunking
    CUDA_VISIBLE_DEVICES=1 python -m test.test_sam3_hf_chunking --config config/my_custom_config.yaml
"""

import argparse
import os
from pathlib import Path

from loguru import logger

from src.sam3_hf_video_tracking import (
    autoselect_torch_device,
    chunk_video_frames,
    convert_results_for_json,
    load_config,
    load_video_frames,
    overlay_masks_on_frame,
    process_chunk,
)
from src.utils import save_results_json, setup_logger

# Default config path (test config with shorter chunks)
DEFAULT_CONFIG = "config/sam3_hf_config_test.yaml"


def main():
    """Main entry point for the test chunking script."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="SAM3 HuggingFace Video Chunking Test")
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG,
        help=f"Path to config file (default: {DEFAULT_CONFIG})",
    )
    args = parser.parse_args()

    # Load configuration
    cfg = load_config(args.config)

    # Setup logger with file output
    log_output_dir = Path(cfg.log_output_dir)
    log_file = setup_logger(log_output_dir, debug=False)

    logger.info("=" * 60)
    logger.info("SAM3 HuggingFace Video Chunking TEST Script")
    logger.info("=" * 60)
    logger.info(f"Config file: {args.config}")
    logger.info(f"Log file: {log_file}")
    logger.info(
        f"Tracking config: min_objects={cfg.min_objects_for_tracking}, "
        f"max_lookback={cfg.max_lookback_frames}"
    )

    # Setup device
    device = autoselect_torch_device()
    logger.info(f"Using device: {device}")
    if cuda_id := os.environ.get("CUDA_VISIBLE_DEVICES"):
        logger.info(f"CUDA_VISIBLE_DEVICES: {cuda_id}")

    # Load video
    video_frames, fps = load_video_frames(cfg.video_path)

    # Calculate chunks
    chunks = chunk_video_frames(video_frames, fps, cfg.chunk_duration_seconds)

    # Resolve output paths
    output_path = Path(cfg.output_path)
    vis_output_dir = Path(cfg.vis_output_dir)

    # Process each chunk
    all_results: dict[int, dict] = {}  # chunk_idx -> {frame_idx -> results}
    chunk_metadata: dict[int, dict] = {}  # chunk_idx -> metadata
    previous_outputs: dict[int, dict] | None = None

    for chunk_idx, chunk_range in enumerate(chunks):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"CHUNK {chunk_idx + 1}/{len(chunks)}")
        logger.info(f"{'=' * 60}")

        outputs_per_frame, chunk_info = process_chunk(
            chunk_idx=chunk_idx,
            video_frames=video_frames,
            chunk_range=chunk_range,
            device=device,
            text_prompt=cfg.text_prompt,
            previous_outputs=previous_outputs,
            min_objects_for_tracking=cfg.min_objects_for_tracking,
            max_lookback_frames=cfg.max_lookback_frames,
            # Tracking config tweaks
            init_trk_keep_alive=cfg.tracking.init_trk_keep_alive,
            max_trk_keep_alive=cfg.tracking.max_trk_keep_alive,
            min_trk_keep_alive=cfg.tracking.min_trk_keep_alive,
            trk_assoc_iou_thresh=cfg.tracking.trk_assoc_iou_thresh,
            hotstart_dup_thresh=cfg.tracking.hotstart_dup_thresh,
            suppress_overlapping_based_on_recent_occlusion_threshold=cfg.tracking.suppress_overlap_thresh,
            recondition_every_nth_frame=cfg.tracking.recondition_every_nth_frame,
        )

        # Convert and store results
        chunk_results = {}
        for frame_idx, results in outputs_per_frame.items():
            chunk_results[frame_idx] = convert_results_for_json(results)

        all_results[chunk_idx] = chunk_results

        # Store chunk metadata
        chunk_metadata[chunk_idx] = {
            "frame_range": list(chunk_range),
            **chunk_info,
        }

        # Pass ALL outputs to next chunk for multi-object tracking
        previous_outputs = outputs_per_frame

        # Build metadata for saving
        metadata = {
            "config_file": args.config,
            "video_path": cfg.video_path,
            "text_prompt": cfg.text_prompt,
            "chunk_duration_seconds": cfg.chunk_duration_seconds,
            "min_objects_for_tracking": cfg.min_objects_for_tracking,
            "max_lookback_frames": cfg.max_lookback_frames,
            "total_chunks_processed": len(all_results),
            "total_frames": len(video_frames),
            "fps": fps,
        }

        # INCREMENTAL SAVE: Save results after each chunk
        logger.info("Saving incremental results...")
        save_results_json(
            all_results=all_results,
            metadata=metadata,
            chunk_metadata=chunk_metadata,
            video_frames=video_frames,
            output_path=output_path,
            vis_output_dir=vis_output_dir,
            overlay_func=overlay_masks_on_frame,
            vis_stride=cfg.vis_frame_stride,
        )

    logger.info("\n" + "=" * 60)
    logger.info("PROCESSING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Results: {output_path}")
    logger.info(f"Visualizations: {vis_output_dir}")
    logger.info(f"Log file: {log_file}")


if __name__ == "__main__":
    main()
