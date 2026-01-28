"""
SAM3 HuggingFace Video Tracking - Core Module

This module provides shared functionality for SAM3-based video segmentation
and tracking with chunked processing for long videos.

Features:
- Text-based segmentation (Sam3VideoModel) for initial detection
- Point-based tracking (Sam3TrackerVideoModel) for object continuity
- Multi-object tracking across video chunks
- RLE mask encoding for efficient storage
- GPU memory management

Usage:
    from src.sam3_hf_video_tracking import (
        process_chunk,
        convert_results_for_json,
        load_video_frames,
        chunk_video_frames,
        # ... etc
    )
"""

import gc
from pathlib import Path

import numpy as np
import pycocotools.mask as mask_util
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from tqdm import tqdm
from transformers import (
    Sam3TrackerVideoModel,
    Sam3TrackerVideoProcessor,
    Sam3VideoConfig,
    Sam3VideoModel,
    Sam3VideoProcessor,
)
from transformers.video_utils import load_video

__all__ = [
    # Config loading
    "load_config",
    # Device selection
    "autoselect_torch_device",
    # Mask utilities
    "single_mask_to_rle",
    "convert_results_for_json",
    "overlay_masks_on_frame",
    # Object extraction
    "get_all_objects_from_results",
    "find_frame_with_enough_objects",
    "extract_equidistant_points_from_mask",
    # Memory management
    "free_gpu_memory",
    # Video loading/chunking
    "load_video_frames",
    "chunk_video_frames",
    # Core processing
    "process_chunk",
]


# =============================================================================
# Config Loading
# =============================================================================


def load_config(config_path: str | Path) -> DictConfig:
    """
    Load configuration from a YAML file using OmegaConf.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        DictConfig object with configuration values
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = OmegaConf.load(config_path)
    logger.info(f"Loaded config from {config_path}")
    return cfg


# =============================================================================
# Device Selection
# =============================================================================


def autoselect_torch_device() -> torch.device:
    """
    Automatically select the best available PyTorch device.

    Priority: CUDA > MPS > CPU

    Returns:
        torch.device: The selected device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# =============================================================================
# Mask Utilities
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
        dict with RLE-encoded masks and Python native types:
        - masks_rle: List of RLE-encoded masks
        - boxes: List of [x1, y1, x2, y2] bounding boxes
        - object_ids: List of object IDs
        - scores: List of confidence scores
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


def overlay_masks_on_frame(
    frame: np.ndarray, masks: np.ndarray | torch.Tensor
) -> Image.Image:
    """
    Overlay segmentation masks on a video frame.

    Args:
        frame: RGB numpy array (H, W, 3)
        masks: Binary masks (N, H, W)

    Returns:
        PIL Image with masks overlaid using rainbow colormap
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
# Object Extraction
# =============================================================================


def get_all_objects_from_results(
    results: dict,
) -> tuple[list[np.ndarray], list[list[float]], list[int]]:
    """
    Extract ALL objects' masks, bboxes, and object IDs from results.

    Args:
        results: dict with 'masks', 'boxes', 'object_ids', 'scores'

    Returns:
        Tuple of (masks_list, boxes_list, object_ids_list)
        Each list has one entry per object.
    """
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
    """
    Search backwards through frame results to find a frame with enough objects.

    Args:
        outputs_per_frame: Dict mapping frame_idx -> results
        min_objects: Minimum number of objects required
        max_lookback: Maximum number of frames to search backwards

    Returns:
        Tuple of (frame_idx, masks_list, boxes_list, object_ids_list)
        Returns (None, [], [], []) if no suitable frame found within lookback limit.
    """
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

    # If we get here, no frame had enough objects but we're within lookback
    # Return the last frame's objects anyway
    if frame_indices:
        last_frame = frame_indices[0]
        results = outputs_per_frame[last_frame]
        masks_list, boxes_list, object_ids_list = get_all_objects_from_results(results)
        if masks_list:
            logger.warning(
                f"No frame with {min_objects}+ objects found, using last frame with {len(masks_list)} objects"
            )
            return last_frame, masks_list, boxes_list, object_ids_list

    return None, [], [], []


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


# =============================================================================
# Memory Management
# =============================================================================


def free_gpu_memory(log_stats: bool = False):
    """
    Force garbage collection and clear GPU memory cache.
    Call this AFTER deleting heavy objects in the calling scope.

    Args:
        log_stats: If True, log memory statistics after cleanup
    """
    # Multiple rounds of garbage collection
    for _ in range(3):
        gc.collect()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        if log_stats:
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(
                f"GPU memory after cleanup - Allocated: {allocated:.2f} GiB, Reserved: {reserved:.2f} GiB"
            )


# =============================================================================
# Video Loading and Chunking
# =============================================================================


def load_video_frames(video_path: str | Path) -> tuple[list, float]:
    """
    Load all video frames and determine FPS.

    Args:
        video_path: Path to video file

    Returns:
        Tuple of (frames list, fps)
    """
    logger.info(f"Loading video from {video_path}...")
    video_frames, metadata = load_video(str(video_path))
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


# =============================================================================
# Core Processing
# =============================================================================


def process_chunk(
    chunk_idx: int,
    video_frames: list,
    chunk_range: tuple[int, int],
    device: torch.device,
    text_prompt: str,
    previous_outputs: dict[int, dict] | None = None,
    min_objects_for_tracking: int = 3,
    max_lookback_frames: int = 25,
    # Sam3VideoConfig tracking parameters
    init_trk_keep_alive: int | None = None,
    max_trk_keep_alive: int | None = None,
    min_trk_keep_alive: int | None = None,
    trk_assoc_iou_thresh: float | None = None,
    hotstart_dup_thresh: int | None = None,
    suppress_overlapping_based_on_recent_occlusion_threshold: float | None = None,
    recondition_every_nth_frame: int | None = None,
) -> tuple[dict[int, dict], dict]:
    """
    Process a single video chunk with SAM3.

    For the first chunk (chunk_idx=0), uses Sam3VideoModel with text prompt.
    For subsequent chunks, uses Sam3TrackerVideoModel with point prompts
    extracted from ALL objects in the previous chunk.

    Args:
        chunk_idx: Index of this chunk
        video_frames: All video frames
        chunk_range: (start_idx, end_idx) for this chunk
        device: PyTorch device
        text_prompt: Text prompt for segmentation (used for first chunk or fallback)
        previous_outputs: Outputs from previous chunk for continuity
        min_objects_for_tracking: Minimum objects needed to use tracker
        max_lookback_frames: Max frames to search back for enough objects

        # Tracking config tweaks (Sam3VideoConfig parameters):
        init_trk_keep_alive: Initial keep-alive counter for new tracks (default: 30)
        max_trk_keep_alive: Maximum keep-alive counter value (default: 30)
        min_trk_keep_alive: Minimum keep-alive counter value (default: -1)
        trk_assoc_iou_thresh: IoU threshold for detection-to-track matching (default: 0.5)
            Lower = more lenient matching, tracks persist longer
        hotstart_dup_thresh: Overlapping frames required to remove duplicate (default: 8)
            Higher = stricter duplicate detection
        suppress_overlapping_based_on_recent_occlusion_threshold: IoU threshold for
            suppressing overlapping objects (default: 0.7). Higher = stricter
        recondition_every_nth_frame: Frequency of mask reconditioning (default: 16)
            Lower = more frequent reconditioning, prevents drift

    Returns:
        Tuple of (outputs_per_frame, chunk_info_dict)
        - outputs_per_frame: dict mapping global frame_idx to results with
          'masks', 'boxes', 'object_ids', 'scores' (numpy arrays)
        - chunk_info_dict: metadata about the chunk processing including
          model_type, prompt_points, num_objects_tracked, etc.
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
                f"High reserved memory detected ({reserved:.2f} GiB), attempting cleanup..."
            )
            free_gpu_memory(log_stats=True)

    # Track the chunk info for metadata
    chunk_info = {
        "model_type": None,
        "prompt_points": None,
        "num_objects_tracked": 0,
        "source_frame_idx": None,
        "fallback_reason": None,
    }

    # Determine if we should use tracker or text model
    use_tracker = False
    masks_list = []
    _boxes_list = []  # Returned by find_frame_with_enough_objects but not used currently
    object_ids_list = []
    all_prompt_points = {}  # obj_id -> list of points

    if chunk_idx == 0 or previous_outputs is None:
        # First chunk: always use text model
        chunk_info["model_type"] = "Sam3VideoModel"
        chunk_info["fallback_reason"] = "first_chunk"
    else:
        # Try to find a frame with enough objects
        source_frame_idx, masks_list, _boxes_list, object_ids_list = (
            find_frame_with_enough_objects(
                previous_outputs,
                min_objects=min_objects_for_tracking,
                max_lookback=max_lookback_frames,
            )
        )

        if source_frame_idx is None or len(masks_list) == 0:
            # No suitable frame found, fall back to text model
            chunk_info["model_type"] = "Sam3VideoModel"
            chunk_info["fallback_reason"] = "no_objects_found"
            logger.warning(
                "No objects found in previous chunk, falling back to text model"
            )
        else:
            # Extract points from each object's mask
            for obj_id, mask in zip(object_ids_list, masks_list):
                points = extract_equidistant_points_from_mask(mask, num_points=3)
                if points is not None:
                    all_prompt_points[obj_id] = points

            if len(all_prompt_points) == 0:
                chunk_info["model_type"] = "Sam3VideoModel"
                chunk_info["fallback_reason"] = "could_not_extract_points"
                logger.warning(
                    "Could not extract points from any mask, falling back to text model"
                )
            else:
                use_tracker = True
                chunk_info["model_type"] = "Sam3TrackerVideoModel"
                # Convert to native Python types for JSON serialization
                chunk_info["prompt_points"] = {
                    int(k): [[int(c) for c in pt] for pt in v]
                    for k, v in all_prompt_points.items()
                }
                chunk_info["num_objects_tracked"] = len(all_prompt_points)
                chunk_info["source_frame_idx"] = int(source_frame_idx)
                logger.info(
                    f"Will track {len(all_prompt_points)} objects from frame {source_frame_idx}"
                )

    if not use_tracker:
        # =====================================================================
        # Use Sam3VideoModel with text prompt (PCS)
        # =====================================================================
        logger.info("Loading Sam3VideoModel for text-based segmentation...")

        # Build custom config if any tracking parameters are specified
        config = Sam3VideoConfig.from_pretrained("facebook/sam3")
        config_modified = False

        if init_trk_keep_alive is not None:
            config.init_trk_keep_alive = init_trk_keep_alive
            config_modified = True
        if max_trk_keep_alive is not None:
            config.max_trk_keep_alive = max_trk_keep_alive
            config_modified = True
        if min_trk_keep_alive is not None:
            config.min_trk_keep_alive = min_trk_keep_alive
            config_modified = True
        if trk_assoc_iou_thresh is not None:
            config.trk_assoc_iou_thresh = trk_assoc_iou_thresh
            config_modified = True
        if hotstart_dup_thresh is not None:
            config.hotstart_dup_thresh = hotstart_dup_thresh
            config_modified = True
        if suppress_overlapping_based_on_recent_occlusion_threshold is not None:
            config.suppress_overlapping_based_on_recent_occlusion_threshold = (
                suppress_overlapping_based_on_recent_occlusion_threshold
            )
            config_modified = True
        if recondition_every_nth_frame is not None:
            config.recondition_every_nth_frame = recondition_every_nth_frame
            config_modified = True

        if config_modified:
            logger.info(
                f"Using custom tracking config: "
                f"init_trk_keep_alive={config.init_trk_keep_alive}, "
                f"max_trk_keep_alive={config.max_trk_keep_alive}, "
                f"min_trk_keep_alive={config.min_trk_keep_alive}, "
                f"trk_assoc_iou_thresh={config.trk_assoc_iou_thresh}, "
                f"hotstart_dup_thresh={config.hotstart_dup_thresh}, "
                f"suppress_overlapping_thresh={config.suppress_overlapping_based_on_recent_occlusion_threshold}, "
                f"recondition_every_nth_frame={config.recondition_every_nth_frame}"
            )
            model = Sam3VideoModel.from_pretrained("facebook/sam3", config=config).to(
                device, dtype=torch.bfloat16
            )
        else:
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
        with torch.inference_mode():
            for model_outputs in tqdm(
                model.propagate_in_video_iterator(
                    inference_session=inference_session,
                    max_frame_num_to_track=len(chunk_frames),
                ),
                total=len(chunk_frames),
                desc=f"Chunk {chunk_idx} (text)",
                unit="frame",
            ):
                processed_outputs = processor.postprocess_outputs(
                    inference_session, model_outputs
                )
                global_frame_idx = start_idx + model_outputs.frame_idx
                outputs_per_frame[global_frame_idx] = processed_outputs

    else:
        # =====================================================================
        # Use Sam3TrackerVideoModel with point prompts for ALL objects (PVS)
        # =====================================================================
        logger.info(
            f"Using Sam3TrackerVideoModel to track {len(all_prompt_points)} objects"
        )
        for obj_id, points in all_prompt_points.items():
            logger.info(f"  Object {obj_id}: points {points}")

        model = Sam3TrackerVideoModel.from_pretrained("facebook/sam3").to(
            device, dtype=torch.bfloat16
        )
        processor = Sam3TrackerVideoProcessor.from_pretrained("facebook/sam3")

        # Initialize video inference session
        logger.info("Initializing video inference session...")
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

        # Build input_points and input_labels for multi-object tracking
        # Format: [[obj1_points, obj2_points, ...]] where each obj_points is [[x1,y1], [x2,y2], ...]
        input_points = [
            [all_prompt_points[oid] for oid in obj_ids]
        ]  # [batch[obj1_pts, obj2_pts, ...]]
        input_labels = [
            [[1] * len(all_prompt_points[oid]) for oid in obj_ids]
        ]  # [batch[obj1_labels, obj2_labels, ...]]

        logger.info(f"Adding prompts for {len(obj_ids)} objects on frame 0")
        processor.add_inputs_to_inference_session(
            inference_session=inference_session,
            frame_idx=ann_frame_idx,
            obj_ids=obj_ids,
            input_points=input_points,
            input_labels=input_labels,
        )

        # Run inference on the first frame to initialize tracking
        logger.info("Running initial inference on frame 0...")
        with torch.inference_mode():
            _ = model(
                inference_session=inference_session,
                frame_idx=ann_frame_idx,
            )

        # Process all frames in the chunk
        logger.info("Running inference...")
        outputs_per_frame = {}
        with torch.inference_mode():
            for tracker_output in tqdm(
                model.propagate_in_video_iterator(inference_session),
                total=len(chunk_frames),
                desc=f"Chunk {chunk_idx} (tracker, {len(obj_ids)} objs)",
                unit="frame",
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

                # Use the actual tracked object IDs
                tracked_obj_ids = (
                    list(inference_session.obj_ids)
                    if hasattr(inference_session, "obj_ids")
                    else obj_ids
                )

                outputs_per_frame[global_frame_idx] = {
                    "masks": masks_np,
                    "boxes": np.array(boxes),
                    "object_ids": np.array(tracked_obj_ids[: len(masks_np)]),
                    "scores": np.array(
                        [1.0] * len(masks_np)
                    ),  # Tracker doesn't provide scores
                }

    logger.info(f"Processed {len(outputs_per_frame)} frames in chunk {chunk_idx}")

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
    free_gpu_memory(log_stats=True)

    return outputs_per_frame, chunk_info
