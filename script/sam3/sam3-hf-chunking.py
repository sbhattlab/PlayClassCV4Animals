"""
SAM3 HuggingFace Video Chunking Script

This script processes long videos by splitting them into 1-minute chunks,
running segmentation on each chunk, and passing the last mask/bbox to the
next chunk as a prompt for tracking continuity.

Usage:
    CUDA_VISIBLE_DEVICES=1 python -m script.sam3.sam3-hf-chunking-test
"""

import os
from pathlib import Path

from loguru import logger

from src.sam3_hf_video_tracking import (
    autoselect_torch_device,
    chunk_video_frames,
    convert_results_for_json,
    load_video_frames,
    overlay_masks_on_frame,
    process_chunk,
)
from src.utils import save_results_json, setup_logger

# =============================================================================
# Configuration
# =============================================================================

# VIDEO_PATH = "/mnt/birds/rebecca2025/raw/video_1_full.mp4"
VIDEO_PATH = "/mnt/birds/rebecca2025/test/video_1_5min.mp4"
TEXT_PROMPT = "bird"
CHUNK_DURATION_SECONDS = 60  # 1 minute chunks
OUTPUT_PATH = Path("data/results/sam3-hf/results.json")
VIS_OUTPUT_DIR = Path("sandbox/test/sam3-hf-chunking/")
LOG_OUTPUT_DIR = Path("sandbox/logs/sam3-hf-chunking/")
VIS_FRAME_STRIDE = 25  # Visualize every Nth frame

# Multi-object tracking heuristics
MIN_OBJECTS_FOR_TRACKING = 3  # Minimum objects to look for in previous chunk
MAX_LOOKBACK_FRAMES = 15  # Max frames to look back (<1 second at 25fps)


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    """Main entry point for the chunking script."""
    # Setup logger with file output
    log_file = setup_logger(LOG_OUTPUT_DIR, debug=False)

    logger.info("=" * 60)
    logger.info("SAM3 HuggingFace Video Chunking Script")
    logger.info("=" * 60)
    logger.info(f"Log file: {log_file}")
    logger.info(
        f"Config: min_objects={MIN_OBJECTS_FOR_TRACKING}, max_lookback={MAX_LOOKBACK_FRAMES}"
    )

    # Setup device
    device = autoselect_torch_device()
    logger.info(f"Using device: {device}")
    if cuda_id := os.environ.get("CUDA_VISIBLE_DEVICES"):
        logger.info(f"CUDA_VISIBLE_DEVICES: {cuda_id}")

    # Load video
    video_frames, fps = load_video_frames(VIDEO_PATH)

    # Calculate chunks
    chunks = chunk_video_frames(video_frames, fps, CHUNK_DURATION_SECONDS)

    # Process each chunk
    all_results: dict[int, dict] = {}  # chunk_idx -> {frame_idx -> results}
    chunk_metadata: dict[int, dict] = {}  # chunk_idx -> metadata
    previous_outputs: dict[int, dict] | None = (
        None  # frame_idx -> results from previous chunk
    )

    for chunk_idx, chunk_range in enumerate(chunks):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"CHUNK {chunk_idx + 1}/{len(chunks)}")
        logger.info(f"{'=' * 60}")

        outputs_per_frame, chunk_info = process_chunk(
            chunk_idx=chunk_idx,
            video_frames=video_frames,
            chunk_range=chunk_range,
            device=device,
            text_prompt=TEXT_PROMPT,
            previous_outputs=previous_outputs,
            min_objects_for_tracking=MIN_OBJECTS_FOR_TRACKING,
            max_lookback_frames=MAX_LOOKBACK_FRAMES,
        )

        # Convert and store results
        chunk_results = {}
        for frame_idx, results in outputs_per_frame.items():
            chunk_results[frame_idx] = convert_results_for_json(results)

        all_results[chunk_idx] = chunk_results

        # Store chunk metadata
        chunk_metadata[chunk_idx] = {
            "frame_range": list(chunk_range),
            **chunk_info,  # Include model_type, prompt_points, num_objects_tracked, etc.
        }

        # Pass ALL outputs to next chunk for multi-object tracking
        previous_outputs = outputs_per_frame

        # Build metadata for saving
        metadata = {
            "video_path": VIDEO_PATH,
            "text_prompt": TEXT_PROMPT,
            "chunk_duration_seconds": CHUNK_DURATION_SECONDS,
            "min_objects_for_tracking": MIN_OBJECTS_FOR_TRACKING,
            "max_lookback_frames": MAX_LOOKBACK_FRAMES,
            "total_chunks_processed": len(all_results),
            "total_frames": len(video_frames),
            "fps": fps,
        }

        # INCREMENTAL SAVE: Save results after each chunk with chunk suffix
        logger.info("Saving incremental results...")
        incremental_path = OUTPUT_PATH.with_stem(f"{OUTPUT_PATH.stem}_{chunk_idx}")
        save_results_json(
            all_results=all_results,
            metadata=metadata,
            chunk_metadata=chunk_metadata,
            video_frames=video_frames,
            output_path=incremental_path,
            vis_output_dir=VIS_OUTPUT_DIR,
            overlay_func=overlay_masks_on_frame,
            vis_stride=VIS_FRAME_STRIDE,
        )

    # Build final metadata
    final_metadata = {
        "video_path": VIDEO_PATH,
        "text_prompt": TEXT_PROMPT,
        "chunk_duration_seconds": CHUNK_DURATION_SECONDS,
        "min_objects_for_tracking": MIN_OBJECTS_FOR_TRACKING,
        "max_lookback_frames": MAX_LOOKBACK_FRAMES,
        "total_chunks_processed": len(all_results),
        "total_frames": len(video_frames),
        "fps": fps,
    }

    # FINAL SAVE: Save consolidated results with all chunks
    logger.info("Saving final consolidated results...")
    save_results_json(
        all_results=all_results,
        metadata=final_metadata,
        chunk_metadata=chunk_metadata,
        video_frames=video_frames,
        output_path=OUTPUT_PATH,
        vis_output_dir=VIS_OUTPUT_DIR,
        overlay_func=overlay_masks_on_frame,
        vis_stride=VIS_FRAME_STRIDE,
    )

    logger.info("\n" + "=" * 60)
    logger.info("PROCESSING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Final results: {OUTPUT_PATH}")
    logger.info(f"Visualizations: {VIS_OUTPUT_DIR}")
    logger.info(f"Log file: {log_file}")


if __name__ == "__main__":
    main()
