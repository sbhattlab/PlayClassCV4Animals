"""
Test script for SAM3 multi-object chunking logic.

This is a slimmed-down version of sam3-hf-chunking-test.py for quick iteration.
Uses 10-second chunks on a short (~1 min) video.

Usage:
    # In sam3-hf environment shell:
    CUDA_VISIBLE_DEVICES=1 python test/test_sam3_chunking_multiobj.py
"""

import gc
import json
import os
from pathlib import Path

import numpy as np
import pycocotools.mask as mask_util
import torch
from loguru import logger
from PIL import Image
from tqdm import tqdm
from transformers import (
    Sam3TrackerVideoModel,
    Sam3TrackerVideoProcessor,
    Sam3VideoModel,
    Sam3VideoProcessor,
)
from transformers.video_utils import load_video

# =============================================================================
# Configuration - EDIT THESE
# =============================================================================

VIDEO_PATH = "/mnt/birds/rebecca2025/test/video_1_1min.mp4"
TEXT_PROMPT = "bird"
CHUNK_DURATION_SECONDS = 34  # Short chunks for testing

# Multi-object tracking settings
MIN_OBJECTS_FOR_TRACKING = 3
MAX_LOOKBACK_FRAMES = 5

# Output paths
OUTPUT_DIR = Path("sandbox/test/sam3-chunking-test/")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Helper Functions (copied from main script)
# =============================================================================


def autoselect_torch_device() -> torch.device:
    """Auto-select best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def single_mask_to_rle(mask: np.ndarray) -> dict:
    """Convert a single binary mask to COCO RLE format."""
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


def get_all_objects_from_results(
    results: dict,
) -> tuple[list[np.ndarray], list[list[float]], list[int]]:
    """Extract ALL objects' masks, bboxes, and object IDs from results."""
    masks = results.get("masks")
    boxes = results.get("boxes")
    object_ids = results.get("object_ids")

    if masks is None or len(masks) == 0:
        return [], [], []

    if isinstance(masks, torch.Tensor):
        masks = masks.cpu().numpy()
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().numpy()
    if isinstance(object_ids, torch.Tensor):
        object_ids = object_ids.cpu().numpy()

    masks_list = [masks[i] for i in range(len(masks))]
    boxes_list = [[float(c) for c in boxes[i]] for i in range(len(boxes))]
    object_ids_list = (
        [int(object_ids[i]) for i in range(len(object_ids))]
        if object_ids is not None
        else list(range(1, len(masks) + 1))
    )

    return masks_list, boxes_list, object_ids_list


def find_frame_with_enough_objects(
    outputs_per_frame: dict[int, dict],
    min_objects: int = 3,
    max_lookback: int = 25,
) -> tuple[int | None, list[np.ndarray], list[list[float]], list[int]]:
    """Search backwards through frame results to find a frame with enough objects."""
    frame_indices = sorted(outputs_per_frame.keys(), reverse=True)

    for i, frame_idx in enumerate(frame_indices):
        if i >= max_lookback:
            logger.warning(
                f"Could not find frame with {min_objects}+ objects within {max_lookback} frames"
            )
            return None, [], [], []

        results = outputs_per_frame[frame_idx]
        masks_list, boxes_list, object_ids_list = get_all_objects_from_results(results)

        if len(masks_list) >= min_objects:
            logger.info(
                f"Found frame {frame_idx} with {len(masks_list)} objects "
                f"({i} frames back from end)"
            )
            return frame_idx, masks_list, boxes_list, object_ids_list

    # Return last frame's objects anyway if we have any
    if frame_indices:
        last_frame = frame_indices[0]
        results = outputs_per_frame[last_frame]
        masks_list, boxes_list, object_ids_list = get_all_objects_from_results(results)
        if masks_list:
            logger.warning(
                f"No frame with {min_objects}+ objects, using last frame with {len(masks_list)}"
            )
            return last_frame, masks_list, boxes_list, object_ids_list

    return None, [], [], []


def extract_equidistant_points_from_mask(
    mask: np.ndarray, num_points: int = 3
) -> list[list[int]] | None:
    """Extract equidistant points along the mask's positive region."""
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


def free_gpu_memory():
    """Force garbage collection and clear GPU memory cache."""
    for _ in range(3):
        gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def overlay_masks_on_frame(frame: np.ndarray, masks: np.ndarray) -> Image.Image:
    """Overlay segmentation masks on a video frame."""
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
        if mask_img.size != image.size:
            mask_img = mask_img.resize(image.size, Image.Resampling.NEAREST)
        overlay = Image.new("RGBA", image.size, color + (0,))
        alpha = mask_img.point(lambda v: int(v * 0.5))
        overlay.putalpha(alpha)
        image = Image.alpha_composite(image, overlay)

    return image.convert("RGB")


# =============================================================================
# Main Processing
# =============================================================================


def process_chunk(
    chunk_idx: int,
    video_frames: list,
    chunk_range: tuple[int, int],
    device: torch.device,
    text_prompt: str,
    previous_outputs: dict[int, dict] | None = None,
) -> tuple[dict[int, dict], dict]:
    """Process a single video chunk."""
    start_idx, end_idx = chunk_range
    chunk_frames = video_frames[start_idx:end_idx]

    logger.info(
        f"Chunk {chunk_idx}: frames {start_idx}-{end_idx} ({len(chunk_frames)} frames)"
    )

    chunk_info = {
        "model_type": None,
        "num_objects_tracked": 0,
        "source_frame_idx": None,
        "fallback_reason": None,
    }

    use_tracker = False
    masks_list = []
    object_ids_list = []
    all_prompt_points = {}

    if chunk_idx == 0 or previous_outputs is None:
        chunk_info["model_type"] = "Sam3VideoModel"
        chunk_info["fallback_reason"] = "first_chunk"
    else:
        source_frame_idx, masks_list, _, object_ids_list = (
            find_frame_with_enough_objects(
                previous_outputs,
                min_objects=MIN_OBJECTS_FOR_TRACKING,
                max_lookback=MAX_LOOKBACK_FRAMES,
            )
        )

        if source_frame_idx is None or len(masks_list) == 0:
            chunk_info["model_type"] = "Sam3VideoModel"
            chunk_info["fallback_reason"] = "no_objects_found"
            logger.warning("No objects found, falling back to text model")
        else:
            for obj_id, mask in zip(object_ids_list, masks_list):
                points = extract_equidistant_points_from_mask(mask, num_points=3)
                if points is not None:
                    all_prompt_points[obj_id] = points

            if len(all_prompt_points) == 0:
                chunk_info["model_type"] = "Sam3VideoModel"
                chunk_info["fallback_reason"] = "could_not_extract_points"
            else:
                use_tracker = True
                chunk_info["model_type"] = "Sam3TrackerVideoModel"
                chunk_info["num_objects_tracked"] = len(all_prompt_points)
                chunk_info["source_frame_idx"] = int(source_frame_idx)
                logger.info(
                    f"Will track {len(all_prompt_points)} objects from frame {source_frame_idx}"
                )

    if not use_tracker:
        # =====================================================================
        # Sam3VideoModel with text prompt
        # =====================================================================
        logger.info("Loading Sam3VideoModel...")
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

        logger.info(f"Adding text prompt: '{text_prompt}'")
        inference_session = processor.add_text_prompt(
            inference_session=inference_session,
            text=text_prompt,
        )

        outputs_per_frame = {}
        with torch.inference_mode():
            for model_outputs in tqdm(
                model.propagate_in_video_iterator(
                    inference_session=inference_session,
                    max_frame_num_to_track=len(chunk_frames),
                ),
                total=len(chunk_frames),
                desc=f"Chunk {chunk_idx} (text)",
            ):
                processed_outputs = processor.postprocess_outputs(
                    inference_session, model_outputs
                )
                global_frame_idx = start_idx + model_outputs.frame_idx
                outputs_per_frame[global_frame_idx] = processed_outputs

    else:
        # =====================================================================
        # Sam3TrackerVideoModel with point prompts for ALL objects
        # =====================================================================
        logger.info(
            f"Loading Sam3TrackerVideoModel for {len(all_prompt_points)} objects..."
        )
        for obj_id, points in all_prompt_points.items():
            logger.info(f"  Object {obj_id}: {points}")

        model = Sam3TrackerVideoModel.from_pretrained("facebook/sam3").to(
            device, dtype=torch.bfloat16
        )
        processor = Sam3TrackerVideoProcessor.from_pretrained("facebook/sam3")

        inference_session = processor.init_video_session(
            video=chunk_frames,
            inference_device=device,
            processing_device="cpu",
            video_storage_device="cpu",
            dtype=torch.bfloat16,
        )

        ann_frame_idx = 0
        obj_ids = list(all_prompt_points.keys())
        input_points = [[all_prompt_points[oid] for oid in obj_ids]]
        input_labels = [[[1] * len(all_prompt_points[oid]) for oid in obj_ids]]

        logger.info(f"Adding prompts for {len(obj_ids)} objects on frame 0")
        processor.add_inputs_to_inference_session(
            inference_session=inference_session,
            frame_idx=ann_frame_idx,
            obj_ids=obj_ids,
            input_points=input_points,
            input_labels=input_labels,
        )

        # Initialize tracking
        with torch.inference_mode():
            _ = model(inference_session=inference_session, frame_idx=ann_frame_idx)

        outputs_per_frame = {}
        with torch.inference_mode():
            for tracker_output in tqdm(
                model.propagate_in_video_iterator(inference_session),
                total=len(chunk_frames),
                desc=f"Chunk {chunk_idx} (tracker, {len(obj_ids)} objs)",
            ):
                video_res_masks = processor.post_process_masks(
                    [tracker_output.pred_masks],
                    original_sizes=[
                        [inference_session.video_height, inference_session.video_width]
                    ],
                    binarize=True,
                )[0]

                global_frame_idx = start_idx + tracker_output.frame_idx
                masks_np = (
                    video_res_masks.cpu().numpy()
                    if isinstance(video_res_masks, torch.Tensor)
                    else video_res_masks
                )
                if masks_np.ndim == 4:
                    masks_np = masks_np.squeeze(1)

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

                tracked_obj_ids = (
                    list(inference_session.obj_ids)
                    if hasattr(inference_session, "obj_ids")
                    else obj_ids
                )

                outputs_per_frame[global_frame_idx] = {
                    "masks": masks_np,
                    "boxes": np.array(boxes),
                    "object_ids": np.array(tracked_obj_ids[: len(masks_np)]),
                    "scores": np.array([1.0] * len(masks_np)),
                }

    logger.info(f"Processed {len(outputs_per_frame)} frames")

    # Cleanup
    if hasattr(inference_session, "reset_inference_session"):
        inference_session.reset_inference_session()
    del inference_session, model, processor
    free_gpu_memory()

    return outputs_per_frame, chunk_info


def main():
    logger.remove()
    logger.add(
        lambda msg: print(msg, end=""),
        colorize=True,
        format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>",
    )

    logger.info("=" * 50)
    logger.info("SAM3 Multi-Object Chunking Test")
    logger.info("=" * 50)
    logger.info(f"Video: {VIDEO_PATH}")
    logger.info(f"Chunk duration: {CHUNK_DURATION_SECONDS}s")
    logger.info(
        f"Min objects: {MIN_OBJECTS_FOR_TRACKING}, Max lookback: {MAX_LOOKBACK_FRAMES}"
    )

    device = autoselect_torch_device()
    logger.info(f"Device: {device}")
    if cuda_id := os.environ.get("CUDA_VISIBLE_DEVICES"):
        logger.info(f"CUDA_VISIBLE_DEVICES: {cuda_id}")

    # Load video
    logger.info("Loading video...")
    video_frames, metadata = load_video(VIDEO_PATH)
    fps = metadata.get("fps", 25.0) if metadata else 25.0
    logger.info(f"Loaded {len(video_frames)} frames at {fps} FPS")

    # Calculate chunks
    frames_per_chunk = int(fps * CHUNK_DURATION_SECONDS)
    chunks = []
    start_idx = 0
    while start_idx < len(video_frames):
        end_idx = min(start_idx + frames_per_chunk, len(video_frames))
        chunks.append((start_idx, end_idx))
        start_idx = end_idx
    logger.info(f"Split into {len(chunks)} chunks of ~{CHUNK_DURATION_SECONDS}s")

    # Process chunks
    all_results = {}  # chunk_idx -> {frame_idx -> raw results with numpy arrays}
    all_results_json = {}  # chunk_idx -> {frame_idx -> JSON-serializable results}
    chunk_metadata = {}
    previous_outputs = None

    for chunk_idx, chunk_range in enumerate(chunks):
        logger.info(f"\n{'=' * 50}")
        logger.info(f"CHUNK {chunk_idx + 1}/{len(chunks)}")
        logger.info(f"{'=' * 50}")

        outputs_per_frame, chunk_info = process_chunk(
            chunk_idx=chunk_idx,
            video_frames=video_frames,
            chunk_range=chunk_range,
            device=device,
            text_prompt=TEXT_PROMPT,
            previous_outputs=previous_outputs,
        )

        # Store raw results for visualization and next chunk
        all_results[chunk_idx] = outputs_per_frame

        # Convert to JSON-serializable format
        chunk_results_json = {}
        for frame_idx, results in outputs_per_frame.items():
            chunk_results_json[frame_idx] = convert_results_for_json(results)
        all_results_json[chunk_idx] = chunk_results_json

        chunk_metadata[chunk_idx] = {"frame_range": list(chunk_range), **chunk_info}
        previous_outputs = outputs_per_frame

        # Log objects found
        last_frame_idx = max(outputs_per_frame.keys())
        last_results = outputs_per_frame[last_frame_idx]
        n_objects = len(last_results.get("masks", []))
        logger.info(f"Chunk {chunk_idx} complete: {n_objects} objects in last frame")

    # Save sample visualizations (first and last frame of each chunk)
    logger.info("\nSaving sample visualizations...")
    for chunk_idx, outputs in all_results.items():
        frame_indices = sorted(outputs.keys())
        for label, fidx in [("first", frame_indices[0]), ("last", frame_indices[-1])]:
            results = outputs[fidx]
            masks = results.get("masks")
            if masks is not None and len(masks) > 0:
                frame = video_frames[fidx]
                vis = overlay_masks_on_frame(frame, masks)
                out_path = OUTPUT_DIR / f"chunk{chunk_idx}_{label}_frame{fidx}.png"
                vis.save(out_path)
                logger.info(f"Saved: {out_path}")

    # Build and save full JSON results (matching main script format)
    logger.info("\nSaving full JSON results...")
    json_results = {
        str(chunk_idx): {
            str(frame_idx): frame_results
            for frame_idx, frame_results in chunk_data.items()
        }
        for chunk_idx, chunk_data in all_results_json.items()
    }

    json_output = {
        "metadata": {
            "video_path": VIDEO_PATH,
            "text_prompt": TEXT_PROMPT,
            "chunk_duration_seconds": CHUNK_DURATION_SECONDS,
            "min_objects_for_tracking": MIN_OBJECTS_FOR_TRACKING,
            "max_lookback_frames": MAX_LOOKBACK_FRAMES,
            "total_chunks": len(chunks),
            "total_frames": len(video_frames),
            "fps": fps,
        },
        "chunk_metadata": {
            str(chunk_idx): meta for chunk_idx, meta in chunk_metadata.items()
        },
        "results": json_results,
    }

    results_path = OUTPUT_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(json_output, f, indent=2)
    logger.info(f"Full results saved to: {results_path}")

    logger.info("\n" + "=" * 50)
    logger.info("TEST COMPLETE")
    logger.info("=" * 50)
    logger.info(f"Results JSON: {results_path}")
    logger.info(f"Visualizations: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
