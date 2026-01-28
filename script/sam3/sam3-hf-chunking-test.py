"""
SAM3 HuggingFace Video Chunking Script

This script processes long videos by splitting them into 1-minute chunks,
running segmentation on each chunk, and passing the last mask/bbox to the
next chunk as a prompt for tracking continuity.

Usage:
    CUDA_VISIBLE_DEVICES=1 python -m script.sam3.sam3-hf-chunking-test
"""

import gc
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pycocotools.mask as mask_util
import torch
from loguru import logger
from PIL import Image
from transformers import (
    Sam3TrackerVideoModel,
    Sam3TrackerVideoProcessor,
    Sam3VideoModel,
    Sam3VideoProcessor,
)
from transformers.video_utils import load_video

from script.sam3.utils import autoselect_torch_device

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

# =============================================================================
# Logger Setup
# =============================================================================


def setup_logger(log_dir: Path, debug: bool = False) -> Path:
    """
    Configure loguru logger with both console and file output.

    Args:
        log_dir: Directory to store log files
        debug: Enable debug level logging

    Returns:
        Path to the log file
    """
    level = "DEBUG" if debug else "INFO"

    # Create log directory
    log_dir.mkdir(parents=True, exist_ok=True)

    # Generate unique log filename with datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = log_dir / f"sam3_hf_chunking_{timestamp}.log"

    # Remove default logger
    logger.remove()

    # Add console handler with colored output
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> "
        "[<level>{level}</level>] {message}",
        level=level,
    )

    # Add file handler
    logger.add(
        str(log_filename),
        format="{time:YYYY-MM-DD HH:mm:ss} [{level}] {message}",
        level=level,
        enqueue=True,
        backtrace=True,
        diagnose=True,
    )

    return log_filename


# =============================================================================
# Helper Functions
# =============================================================================


def single_mask_to_rle(mask: np.ndarray) -> dict:
    """
    Convert a single binary mask to COCO RLE format.

    Args:
        mask: HxW binary mask (uint8 or bool)

    Returns:
        dict with 'counts' (str) and 'size' ([H, W])
    """
    rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    if "size" in rle and isinstance(rle["size"], (list, tuple)):
        rle["size"] = [int(rle["size"][0]), int(rle["size"][1])]
    return rle


def convert_results_for_json(results: dict) -> dict:
    """
    Convert model outputs to JSON-serializable format.

    Args:
        results: dict with 'masks', 'boxes', 'object_ids', 'scores'

    Returns:
        dict with RLE-encoded masks and Python native types
    """
    # Convert masks to RLE
    masks = results.get("masks")
    if masks is not None:
        if isinstance(masks, torch.Tensor):
            masks = masks.cpu().numpy()
        masks_rle = [single_mask_to_rle(m) for m in masks]
    else:
        masks_rle = []

    # Convert boxes to list of floats
    boxes = results.get("boxes")
    if boxes is not None:
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.cpu().numpy()
        boxes_list = [[float(c) for c in box] for box in boxes]
    else:
        boxes_list = []

    # Convert object_ids to list of integers
    object_ids = results.get("object_ids")
    if object_ids is not None:
        if isinstance(object_ids, torch.Tensor):
            object_ids = object_ids.cpu().numpy()
        object_ids_list = [int(oid) for oid in object_ids]
    else:
        object_ids_list = []

    # Convert scores to list of floats
    scores = results.get("scores")
    if scores is not None:
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()
        scores_list = [float(s) for s in scores]
    else:
        scores_list = []

    return {
        "masks_rle": masks_rle,
        "boxes": boxes_list,
        "object_ids": object_ids_list,
        "scores": scores_list,
    }


def get_best_object_from_results(
    results: dict,
) -> tuple[np.ndarray | None, list | None]:
    """
    Extract the best (highest score) object's mask and bbox from results.

    Args:
        results: dict with 'masks', 'boxes', 'scores'

    Returns:
        Tuple of (mask, bbox) for the highest-scoring object, or (None, None)
    """
    scores = results.get("scores")
    if scores is None or len(scores) == 0:
        return None, None

    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()

    best_idx = int(np.argmax(scores))

    masks = results.get("masks")
    if masks is not None:
        if isinstance(masks, torch.Tensor):
            masks = masks.cpu().numpy()
        best_mask = masks[best_idx]
    else:
        best_mask = None

    boxes = results.get("boxes")
    if boxes is not None:
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.cpu().numpy()
        best_bbox = [float(c) for c in boxes[best_idx]]
    else:
        best_bbox = None

    return best_mask, best_bbox


def extract_equidistant_points_from_mask(
    mask: np.ndarray, num_points: int = 3
) -> list[list[int]] | None:
    """
    Extract equidistant points along the mask's positive region.

    This samples points along the skeleton/centerline of the mask by finding
    the centroid and points along the major axis.

    Args:
        mask: Binary mask (H, W) with 1s indicating the object
        num_points: Number of equidistant points to extract

    Returns:
        List of [x, y] points, or None if mask is empty
    """
    # Find all positive pixels
    y_coords, x_coords = np.where(mask > 0)
    if len(y_coords) == 0:
        return None

    # Get centroid
    center_x = int(np.mean(x_coords))
    center_y = int(np.mean(y_coords))

    if num_points == 1:
        return [[center_x, center_y]]

    # Sort points by x-coordinate to find extremes along horizontal axis
    sorted_indices = np.argsort(x_coords)
    n_pixels = len(sorted_indices)

    # Sample equidistant indices along the sorted pixels
    indices = np.linspace(0, n_pixels - 1, num_points, dtype=int)

    points = []
    for idx in indices:
        sorted_idx = sorted_indices[idx]
        x = int(x_coords[sorted_idx])
        y = int(y_coords[sorted_idx])
        points.append([x, y])

    logger.debug(f"Extracted {num_points} equidistant points from mask: {points}")
    return points


def free_gpu_memory():
    """
    Force garbage collection and clear GPU memory cache.
    Call this AFTER deleting heavy objects in the calling scope.
    """
    # Multiple rounds of garbage collection
    for _ in range(3):
        gc.collect()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()  # Collect IPC shared memory
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        # Log memory status
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info(
            f"GPU memory after cleanup - Allocated: {allocated:.2f} GiB, Reserved: {reserved:.2f} GiB"
        )


def overlay_masks_on_frame(
    frame: np.ndarray, masks: np.ndarray | torch.Tensor
) -> Image.Image:
    """
    Overlay segmentation masks on a video frame.

    Args:
        frame: RGB numpy array (H, W, 3)
        masks: Binary masks (N, H, W)

    Returns:
        PIL Image with masks overlaid
    """
    import matplotlib

    image = Image.fromarray(frame).convert("RGBA")

    if isinstance(masks, torch.Tensor):
        masks = masks.cpu().numpy()

    masks = 255 * masks.astype(np.uint8)

    n_masks = masks.shape[0]
    if n_masks == 0:
        return image.convert("RGB")

    cmap = matplotlib.colormaps.get_cmap("rainbow").resampled(n_masks)
    colors = [tuple(int(c * 255) for c in cmap(i)[:3]) for i in range(n_masks)]

    for mask, color in zip(masks, colors):
        mask_img = Image.fromarray(mask)
        # Resize mask to match image dimensions if needed
        if mask_img.size != image.size:
            mask_img = mask_img.resize(image.size, Image.Resampling.NEAREST)
        overlay = Image.new("RGBA", image.size, color + (0,))
        alpha = mask_img.point(lambda v: int(v * 0.5))
        overlay.putalpha(alpha)
        image = Image.alpha_composite(image, overlay)

    return image.convert("RGB")


# =============================================================================
# Main Processing Functions
# =============================================================================


def load_video_frames(video_path: str) -> tuple[list, float]:
    """
    Load all video frames and determine FPS.

    Args:
        video_path: Path to video file

    Returns:
        Tuple of (frames list, fps)
    """
    logger.info(f"Loading video from {video_path}...")
    video_frames, metadata = load_video(video_path)
    fps = metadata.get("fps", 25.0) if metadata else 25.0
    logger.info(f"Loaded {len(video_frames)} frames at {fps} FPS")
    return video_frames, fps


def chunk_video_frames(
    video_frames: list, fps: float, chunk_duration_seconds: int
) -> list[tuple[int, int]]:
    """
    Calculate frame ranges for each chunk.

    Args:
        video_frames: List of video frames
        fps: Frames per second
        chunk_duration_seconds: Duration of each chunk in seconds

    Returns:
        List of (start_idx, end_idx) tuples for each chunk
    """
    frames_per_chunk = int(fps * chunk_duration_seconds)
    total_frames = len(video_frames)

    chunks = []
    start_idx = 0
    while start_idx < total_frames:
        end_idx = min(start_idx + frames_per_chunk, total_frames)
        chunks.append((start_idx, end_idx))
        start_idx = end_idx

    logger.info(
        f"Video split into {len(chunks)} chunks of ~{chunk_duration_seconds}s each"
    )
    return chunks


def process_chunk(
    chunk_idx: int,
    video_frames: list,
    chunk_range: tuple[int, int],
    device: torch.device,
    text_prompt: str,
    previous_mask: np.ndarray | None = None,
) -> tuple[dict[int, dict], np.ndarray | None, list | None, list | None]:
    """
    Process a single video chunk with SAM3.

    For the first chunk (chunk_idx=0), uses Sam3VideoModel with text prompt.
    For subsequent chunks, uses Sam3TrackerVideoModel with point prompts
    extracted from the previous chunk's last mask.

    Args:
        chunk_idx: Index of this chunk
        video_frames: All video frames
        chunk_range: (start_idx, end_idx) for this chunk
        device: PyTorch device
        text_prompt: Text prompt for segmentation (used for first chunk)
        previous_mask: Mask from previous chunk for continuity (optional)

    Returns:
        Tuple of (outputs_per_frame, last_mask, last_bbox, prompt_points)
        prompt_points is the list of points used as prompt (None for first chunk)
    """
    start_idx, end_idx = chunk_range
    chunk_frames = video_frames[start_idx:end_idx]

    logger.info(
        f"Processing chunk {chunk_idx}: frames {start_idx}-{end_idx} "
        f"({len(chunk_frames)} frames)"
    )

    # Log GPU memory before loading model
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info(
            f"GPU memory before loading - Allocated: {allocated:.2f} GiB, Reserved: {reserved:.2f} GiB"
        )
        # If there's significant reserved memory, try to release it
        if reserved > 1.0:
            logger.warning(
                f"High reserved memory detected ({reserved:.2f} GiB), attempting additional cleanup..."
            )
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            # Log again after additional cleanup
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(
                f"GPU memory after additional cleanup - Allocated: {allocated:.2f} GiB, Reserved: {reserved:.2f} GiB"
            )

    # Track the prompt points used (for metadata)
    prompt_points = None

    if chunk_idx == 0 or previous_mask is None:
        # =====================================================================
        # FIRST CHUNK: Use Sam3VideoModel with text prompt (PCS)
        # =====================================================================
        logger.info("Loading Sam3VideoModel for text-based segmentation...")
        model = Sam3VideoModel.from_pretrained("facebook/sam3").to(
            device, dtype=torch.bfloat16
        )
        processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")

        # Initialize video inference session
        logger.info("Initializing video inference session...")
        inference_session = processor.init_video_session(
            video=chunk_frames,
            inference_device=device,
            processing_device="cpu",
            video_storage_device="cpu",
            dtype=torch.bfloat16,
        )

        # Add text prompt
        logger.info(f"Adding text prompt: '{text_prompt}'")
        inference_session = processor.add_text_prompt(
            inference_session=inference_session,
            text=text_prompt,
        )

        # Process all frames in the chunk
        logger.info("Running inference...")
        outputs_per_frame = {}
        for model_outputs in model.propagate_in_video_iterator(
            inference_session=inference_session,
            max_frame_num_to_track=len(chunk_frames),
        ):
            processed_outputs = processor.postprocess_outputs(
                inference_session, model_outputs
            )
            global_frame_idx = start_idx + model_outputs.frame_idx
            outputs_per_frame[global_frame_idx] = processed_outputs

    else:
        # =====================================================================
        # SUBSEQUENT CHUNKS: Use Sam3TrackerVideoModel with point prompts (PVS)
        # =====================================================================
        # Extract equidistant points from previous mask
        prompt_points = extract_equidistant_points_from_mask(
            previous_mask, num_points=3
        )
        if prompt_points is None:
            logger.warning(
                "Could not extract points from previous mask, falling back to text prompt"
            )
            # Fallback to text-based model
            model = Sam3VideoModel.from_pretrained("facebook/sam3").to(
                device, dtype=torch.bfloat16
            )
            processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")
            inference_session = processor.init_video_session(
                video=chunk_frames,
                inference_device=device,
                processing_device="cpu",
                video_storage_device="cpu",
                dtype=torch.bfloat16,
            )
            inference_session = processor.add_text_prompt(
                inference_session=inference_session,
                text=text_prompt,
            )
            outputs_per_frame = {}
            for model_outputs in model.propagate_in_video_iterator(
                inference_session=inference_session,
                max_frame_num_to_track=len(chunk_frames),
            ):
                processed_outputs = processor.postprocess_outputs(
                    inference_session, model_outputs
                )
                global_frame_idx = start_idx + model_outputs.frame_idx
                outputs_per_frame[global_frame_idx] = processed_outputs
        else:
            logger.info(
                f"Using Sam3TrackerVideoModel with {len(prompt_points)} point prompts: {prompt_points}"
            )

            model = Sam3TrackerVideoModel.from_pretrained("facebook/sam3").to(
                device, dtype=torch.bfloat16
            )
            processor = Sam3TrackerVideoProcessor.from_pretrained("facebook/sam3")

            # Initialize video inference session
            # IMPORTANT: Keep video frames on CPU to reduce GPU memory pressure
            logger.info("Initializing video inference session...")
            inference_session = processor.init_video_session(
                video=chunk_frames,
                inference_device=device,
                processing_device="cpu",  # Process on CPU to save GPU memory
                video_storage_device="cpu",  # Store video frames on CPU
                dtype=torch.bfloat16,
            )

            # Add point inputs on first frame of chunk
            # Format: [[[[x1, y1], [x2, y2], ...]]] - 4D: batch, obj, points, coords
            ann_frame_idx = 0
            ann_obj_id = 1
            points = [
                [[[p[0], p[1]] for p in prompt_points]]
            ]  # Wrap for batch/obj dims
            labels = [[[1] * len(prompt_points)]]  # All positive points

            logger.info(f"Adding {len(prompt_points)} point prompts on frame 0")
            processor.add_inputs_to_inference_session(
                inference_session=inference_session,
                frame_idx=ann_frame_idx,
                obj_ids=ann_obj_id,
                input_points=points,
                input_labels=labels,
            )

            # IMPORTANT: Must run inference on the first frame before propagating
            # This initializes the tracking state
            logger.info("Running initial inference on frame 0...")
            _ = model(
                inference_session=inference_session,
                frame_idx=ann_frame_idx,
            )

            # Process all frames in the chunk
            logger.info("Running inference...")
            outputs_per_frame = {}
            for tracker_output in model.propagate_in_video_iterator(inference_session):
                # Post-process masks
                video_res_masks = processor.post_process_masks(
                    [tracker_output.pred_masks],
                    original_sizes=[
                        [inference_session.video_height, inference_session.video_width]
                    ],
                    binarize=True,
                )[0]

                global_frame_idx = start_idx + tracker_output.frame_idx

                # Convert tracker output to same format as Sam3VideoModel output
                masks_np = (
                    video_res_masks.cpu().numpy()
                    if isinstance(video_res_masks, torch.Tensor)
                    else video_res_masks
                )
                # Masks shape: (num_objects, 1, H, W) -> squeeze to (num_objects, H, W)
                if masks_np.ndim == 4:
                    masks_np = masks_np.squeeze(1)

                # Calculate bounding boxes from masks
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

                outputs_per_frame[global_frame_idx] = {
                    "masks": masks_np,
                    "boxes": np.array(boxes),
                    "object_ids": np.array([ann_obj_id]),
                    "scores": np.array([1.0]),  # Tracker doesn't provide scores
                }

    logger.info(f"Processed {len(outputs_per_frame)} frames in chunk {chunk_idx}")

    # Get the last frame's best object for continuity
    last_frame_idx = max(outputs_per_frame.keys())
    last_results = outputs_per_frame[last_frame_idx]
    last_mask, last_bbox = get_best_object_from_results(last_results)

    logger.info(f"Last frame bbox for next chunk: {last_bbox}")

    # Free GPU memory - MUST delete objects explicitly in this scope
    logger.info("Cleaning up GPU memory...")

    # Reset inference session state if it has a reset method
    if hasattr(inference_session, "reset_inference_session"):
        inference_session.reset_inference_session()

    # Delete heavy objects explicitly
    del inference_session
    del model
    del processor

    # Force garbage collection and clear cache
    free_gpu_memory()

    return outputs_per_frame, last_mask, last_bbox, prompt_points


def save_visualizations(
    all_results: dict[int, dict],
    video_frames: list,
    output_dir: Path,
    stride: int = 25,
):
    """
    Save visualization images for sampled frames.

    Args:
        all_results: Results for all frames
        video_frames: All video frames
        output_dir: Directory to save images
        stride: Save every Nth frame
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving visualizations to {output_dir}...")

    frame_indices = sorted(all_results.keys())
    sampled_indices = frame_indices[::stride]

    for frame_idx in sampled_indices:
        results = all_results[frame_idx]
        frame = video_frames[frame_idx]

        masks = results.get("masks")
        if masks is not None:
            vis_image = overlay_masks_on_frame(frame, masks)
            output_path = output_dir / f"frame_{frame_idx:06d}.png"
            vis_image.save(output_path)

    logger.info(f"Saved {len(sampled_indices)} visualization images")


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
    chunk_metadata: dict[int, dict] = {}  # chunk_idx -> metadata (prompt_points, etc.)
    previous_mask = None

    for chunk_idx, chunk_range in enumerate(chunks):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"CHUNK {chunk_idx + 1}/{len(chunks)}")
        logger.info(f"{'=' * 60}")

        outputs_per_frame, last_mask, last_bbox, prompt_points = process_chunk(
            chunk_idx=chunk_idx,
            video_frames=video_frames,
            chunk_range=chunk_range,
            device=device,
            text_prompt=TEXT_PROMPT,
            previous_mask=previous_mask,
        )

        # Convert and store results
        chunk_results = {}
        for frame_idx, results in outputs_per_frame.items():
            chunk_results[frame_idx] = convert_results_for_json(results)

        all_results[chunk_idx] = chunk_results

        # Store chunk metadata including prompt points
        chunk_metadata[chunk_idx] = {
            "frame_range": list(chunk_range),
            "prompt_points": prompt_points,  # None for first chunk, list of [x, y] for subsequent
            "model_type": "Sam3VideoModel"
            if chunk_idx == 0 or previous_mask is None
            else "Sam3TrackerVideoModel",
        }

        # Pass mask to next chunk for point extraction
        previous_mask = last_mask

    # Save results to JSON
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving results to {OUTPUT_PATH}...")

    # Convert keys to strings for JSON serialization
    json_results = {
        str(chunk_idx): {
            str(frame_idx): frame_results
            for frame_idx, frame_results in chunk_data.items()
        }
        for chunk_idx, chunk_data in all_results.items()
    }

    # Add metadata including per-chunk prompt points
    json_output = {
        "metadata": {
            "video_path": VIDEO_PATH,
            "text_prompt": TEXT_PROMPT,
            "chunk_duration_seconds": CHUNK_DURATION_SECONDS,
            "total_chunks": len(chunks),
            "total_frames": len(video_frames),
            "fps": fps,
        },
        "chunk_metadata": {
            str(chunk_idx): meta for chunk_idx, meta in chunk_metadata.items()
        },
        "results": json_results,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(json_output, f, indent=2)

    logger.info(f"Results saved to {OUTPUT_PATH}")

    # Save visualizations (optional)
    logger.info("\nGenerating visualizations...")

    # Flatten all results for visualization
    flat_results = {}
    for chunk_idx, chunk_data in all_results.items():
        for frame_idx, frame_results in chunk_data.items():
            # Need to decode RLE back to masks for visualization
            masks_rle = frame_results.get("masks_rle", [])
            if masks_rle:
                masks = np.stack([mask_util.decode(rle) for rle in masks_rle])
                flat_results[frame_idx] = {"masks": masks}
            else:
                flat_results[frame_idx] = {"masks": None}

    save_visualizations(
        all_results=flat_results,
        video_frames=video_frames,
        output_dir=VIS_OUTPUT_DIR,
        stride=VIS_FRAME_STRIDE,
    )

    logger.info("\n" + "=" * 60)
    logger.info("PROCESSING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Results: {OUTPUT_PATH}")
    logger.info(f"Visualizations: {VIS_OUTPUT_DIR}")
    logger.info(f"Log file: {log_file}")


if __name__ == "__main__":
    main()
