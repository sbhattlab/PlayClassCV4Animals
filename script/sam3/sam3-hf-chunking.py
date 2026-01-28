"""
SAM3 HuggingFace Video Chunking Script

This script processes long videos by splitting them into 1-minute chunks,
running segmentation on each chunk, and passing the last mask/bbox to the
next chunk as a prompt for tracking continuity.

Usage:
    CUDA_VISIBLE_DEVICES=1 python -m script.sam3.sam3-hf-chunking-test
"""

import json
import os
from pathlib import Path

import numpy as np
import pycocotools.mask as mask_util
from loguru import logger

from src.sam3_hf_video_tracking import (
    autoselect_torch_device,
    chunk_video_frames,
    convert_results_for_json,
    load_video_frames,
    overlay_masks_on_frame,
    process_chunk,
)
from src.utils import setup_logger

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
# Save Functions
# =============================================================================


def save_incremental_results(
    all_results: dict[int, dict],
    chunk_metadata: dict[int, dict],
    video_frames: list,
    fps: float,
    output_path: Path,
    vis_output_dir: Path,
    vis_stride: int = 25,
):
    """
    Save results incrementally after each chunk.

    Args:
        all_results: Results accumulated so far {chunk_idx -> {frame_idx -> results}}
        chunk_metadata: Metadata for each chunk
        video_frames: All video frames
        fps: Video frames per second
        output_path: Path to save JSON results
        vis_output_dir: Directory to save visualizations
        vis_stride: Save every Nth frame for visualization
    """
    # Convert keys to strings for JSON serialization
    json_results = {
        str(chunk_idx): {
            str(frame_idx): frame_results
            for frame_idx, frame_results in chunk_data.items()
        }
        for chunk_idx, chunk_data in all_results.items()
    }

    # Add metadata
    json_output = {
        "metadata": {
            "video_path": VIDEO_PATH,
            "text_prompt": TEXT_PROMPT,
            "chunk_duration_seconds": CHUNK_DURATION_SECONDS,
            "min_objects_for_tracking": MIN_OBJECTS_FOR_TRACKING,
            "max_lookback_frames": MAX_LOOKBACK_FRAMES,
            "total_chunks_processed": len(all_results),
            "total_frames": len(video_frames),
            "fps": fps,
        },
        "chunk_metadata": {
            str(chunk_idx): meta for chunk_idx, meta in chunk_metadata.items()
        },
        "results": json_results,
    }

    # Save JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(json_output, f, indent=2)
    logger.info(f"Incremental results saved to {output_path}")

    # Save visualizations for new frames
    vis_output_dir.mkdir(parents=True, exist_ok=True)

    # Flatten results and save visualizations
    for chunk_idx, chunk_data in all_results.items():
        for frame_idx, frame_results in chunk_data.items():
            if int(frame_idx) % vis_stride != 0:
                continue

            output_img_path = vis_output_dir / f"frame_{int(frame_idx):06d}.png"
            if output_img_path.exists():
                continue  # Skip already saved frames

            # Decode RLE masks
            masks_rle = frame_results.get("masks_rle", [])
            if masks_rle:
                masks = np.stack([mask_util.decode(rle) for rle in masks_rle])
                frame = video_frames[int(frame_idx)]
                vis_image = overlay_masks_on_frame(frame, masks)
                vis_image.save(output_img_path)


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

        # INCREMENTAL SAVE: Save results after each chunk with chunk suffix
        logger.info("Saving incremental results...")
        incremental_path = OUTPUT_PATH.with_stem(f"{OUTPUT_PATH.stem}_{chunk_idx}")
        save_incremental_results(
            all_results=all_results,
            chunk_metadata=chunk_metadata,
            video_frames=video_frames,
            fps=fps,
            output_path=incremental_path,
            vis_output_dir=VIS_OUTPUT_DIR,
            vis_stride=VIS_FRAME_STRIDE,
        )

    logger.info("\n" + "=" * 60)
    logger.info("PROCESSING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Results: {OUTPUT_PATH}")
    logger.info(f"Visualizations: {VIS_OUTPUT_DIR}")
    logger.info(f"Log file: {log_file}")


if __name__ == "__main__":
    main()
