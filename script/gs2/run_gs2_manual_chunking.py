"""
GS2 Manual Chunking Pipeline

Config-driven chunked video tracking with Grounded-SAM-2.
Mask-based carryover between chunks for identity preservation.

- Chunk 0: GroundingDINO text-prompted detection → SAM2 image predictor → mask seeds
- Chunks 1+: masks from last frame of previous chunk → SAM2 video predictor

Usage:
    pixi run -e gs2 gs2-manual-chunking
    pixi run -e gs2 python -m script.gs2.run_gs2_manual_chunking --config config/gs2_manual_chunking.yaml
"""

import argparse
import gc
import json
import shutil
import tempfile
import time
from pathlib import Path


def _early_init():
    """Parse config and set env vars BEFORE torch import."""
    parser = argparse.ArgumentParser(description="GS2 Manual Chunking Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="config/gs2_manual_chunking.yaml",
        help="Path to config file (default: config/gs2_manual_chunking.yaml)",
    )
    args, _ = parser.parse_known_args()

    from src.utils import load_config, set_env_vars

    cfg = load_config(args.config)
    set_env_vars(cfg)
    return args, cfg


# Set env vars (CUDA_VISIBLE_DEVICES, PYTORCH_ALLOC_CONF) before torch import
_args, _cfg = _early_init()

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from loguru import logger  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from PIL import Image  # noqa: E402
from sam2.build_sam import build_sam2, build_sam2_video_predictor  # noqa: E402
from sam2.sam2_image_predictor import SAM2ImagePredictor  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
)

import supervision as sv  # noqa: E402

from src.utils import (  # noqa: E402
    build_manual_chunks,
    create_run_directory,
    get_video_metadata,
    load_video_frames_range,
    process_tracking_outputs,
    setup_logger,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GS2_DIR = Path(__file__).resolve().parent.parent.parent / "Grounded-SAM-2-fork"
GDINO_MODEL_ID = "IDEA-Research/grounding-dino-tiny"

_SIZE_TO_CONFIG_CODE = {
    "large": "l",
    "base_plus": "b+",
    "small": "s",
    "tiny": "t",
}

# ---------------------------------------------------------------------------
# Model building helpers
# ---------------------------------------------------------------------------


def _build_sam2_models(device: str, model_size: str = "large"):
    """Build SAM2 video predictor and image predictor from local checkpoint."""
    size_code = _SIZE_TO_CONFIG_CODE.get(model_size, "l")
    checkpoint = str(GS2_DIR / f"checkpoints/sam2.1_hiera_{model_size}.pt")
    model_cfg = f"configs/sam2.1/sam2.1_hiera_{size_code}.yaml"

    logger.info(f"Building SAM2 video predictor ({model_size}) from {checkpoint}")
    video_predictor = build_sam2_video_predictor(model_cfg, checkpoint, device=device)

    logger.info("Building SAM2 image predictor")
    sam2_model = build_sam2(model_cfg, checkpoint, device=device)
    image_predictor = SAM2ImagePredictor(sam2_model)

    return video_predictor, image_predictor


def _build_grounding_dino(device: str):
    """Build GroundingDINO model and processor."""
    logger.info(f"Loading GroundingDINO from {GDINO_MODEL_ID}")
    processor = AutoProcessor.from_pretrained(GDINO_MODEL_ID, use_fast=True)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO_MODEL_ID).to(
        device
    )
    return processor, model


# ---------------------------------------------------------------------------
# Frame I/O
# ---------------------------------------------------------------------------


def _write_frames_to_dir(frames: list, out_dir: str) -> None:
    """Write a list of RGB numpy frames to a directory as numbered JPEGs.

    Files are named 00000.jpg, 00001.jpg, ... — the format SAM2 expects when
    loading from a JPEG directory (sorted by numeric stem).
    """
    for i, frame_rgb in enumerate(frames):
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(Path(out_dir) / f"{i:05d}.jpg"), frame_bgr)


# ---------------------------------------------------------------------------
# First-chunk detection (GroundingDINO + SAM2 image predictor)
# ---------------------------------------------------------------------------


def _filter_boxes_by_area(
    boxes: np.ndarray,
    labels: list,
    frame_h: int,
    frame_w: int,
    min_area_fraction: float,
    max_area_fraction: float,
) -> tuple[np.ndarray, list]:
    """Remove boxes that are too small (noise) or too large (scene-level false positives)."""
    if boxes.shape[0] == 0:
        return boxes, labels
    frame_area = frame_h * frame_w
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    fractions = areas / frame_area
    keep = (fractions >= min_area_fraction) & (fractions <= max_area_fraction)
    rejected = ~keep
    if rejected.any():
        for i, (frac, lbl) in enumerate(zip(fractions, labels)):
            if rejected[i]:
                reason = "too large" if fractions[i] > max_area_fraction else "too small"
                logger.info(
                    f"  Filtered box {i} ({lbl!r}): {100*frac:.1f}% of frame — {reason}"
                )
    return boxes[keep], [lbl for lbl, k in zip(labels, keep) if k]


def _detect_and_segment_first_frame(
    frame_rgb: np.ndarray,
    text_prompt: str,
    gdino_processor,
    gdino_model,
    image_predictor: SAM2ImagePredictor,
    box_threshold: float,
    text_threshold: float,
    device: str,
    min_objects: int = 1,
    min_box_threshold: float = 0.15,
    threshold_step: float = 0.05,
    min_box_area_fraction: float = 0.003,
    max_box_area_fraction: float = 0.40,
) -> tuple[dict, list[str]]:
    """Run GroundingDINO + SAM2 image predictor on the seed frame.

    GroundingDINO inference runs once; the threshold is applied during
    post-processing, so we can retry with progressively lower thresholds
    until at least `min_objects` *valid* detections are found — at no extra
    GPU cost. Boxes that cover too much or too little of the frame are
    filtered out before the count is checked.

    Args:
        box_threshold: Starting detection threshold (lowered each retry).
        min_objects: Keep lowering threshold until this many valid objects found.
        min_box_threshold: Never go below this threshold floor.
        threshold_step: Amount to lower box_threshold on each retry.
        min_box_area_fraction: Reject boxes covering less than this fraction of the frame.
        max_box_area_fraction: Reject boxes covering more than this fraction of the frame.

    Returns:
        obj_id_to_mask: dict mapping int obj_id -> (H, W) uint8 binary mask
        labels: list of string labels from GroundingDINO
    """
    frame_h, frame_w = frame_rgb.shape[:2]
    image_pil = Image.fromarray(frame_rgb)
    inputs = gdino_processor(
        images=image_pil, text=text_prompt, return_tensors="pt"
    ).to(device)

    # Run inference ONCE — the raw logits can be re-thresholded cheaply
    with torch.inference_mode():
        gdino_outputs = gdino_model(**inputs)

    # Progressively lower box_threshold until min_objects *valid* boxes are found
    current_threshold = box_threshold
    input_boxes = np.zeros((0, 4), dtype=np.float32)
    labels = []
    while current_threshold >= min_box_threshold:
        results = gdino_processor.post_process_grounded_object_detection(
            gdino_outputs,
            inputs.input_ids,
            threshold=current_threshold,
            text_threshold=text_threshold,
            target_sizes=[image_pil.size[::-1]],
        )
        raw_boxes = results[0]["boxes"].detach().cpu().numpy()
        raw_labels = results[0]["labels"]
        input_boxes, labels = _filter_boxes_by_area(
            raw_boxes, raw_labels, frame_h, frame_w,
            min_area_fraction=min_box_area_fraction,
            max_area_fraction=max_box_area_fraction,
        )
        n = len(input_boxes)
        if n >= min_objects:
            logger.info(
                f"GroundingDINO: {n} valid detections at box_threshold={current_threshold:.2f} "
                f"(raw={len(raw_boxes)}, after area filter={n})"
            )
            break
        logger.info(
            f"GroundingDINO: only {n}/{min_objects} valid detections at "
            f"box_threshold={current_threshold:.2f} (raw={len(raw_boxes)}) — lowering threshold"
        )
        current_threshold = round(current_threshold - threshold_step, 4)
    else:
        logger.warning(
            f"GroundingDINO: reached floor threshold ({min_box_threshold}) "
            f"with only {len(input_boxes)} valid detections (need {min_objects})"
        )

    if input_boxes.shape[0] == 0:
        logger.warning(f"GroundingDINO: no valid detections for '{text_prompt}' on first frame")
        return {}, []

    # Refine each box to a mask using SAM2 image predictor
    image_predictor.set_image(frame_rgb)
    obj_id_to_mask = {}
    for obj_id, box in enumerate(input_boxes, start=1):
        box_k = np.asarray(box, dtype=np.float32)[None]
        with torch.inference_mode():
            masks_k, scores_k, _ = image_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box_k,
                multimask_output=False,
            )
        # Squeeze to (H, W)
        m = masks_k
        if m.ndim == 4:
            m = m.squeeze(0).squeeze(0)
        elif m.ndim == 3:
            m = m.squeeze(0)
        obj_id_to_mask[int(obj_id)] = m.astype(np.uint8)

    logger.info(
        f"SAM2 image predictor: generated {len(obj_id_to_mask)} masks "
        f"(obj_ids: {list(obj_id_to_mask.keys())})"
    )
    return obj_id_to_mask, [str(lbl) for lbl in labels]


# ---------------------------------------------------------------------------
# Per-chunk SAM2 video processing
# ---------------------------------------------------------------------------


def _process_chunk(
    chunk_frames: list,
    global_start: int,
    chunk_idx: int,
    obj_id_to_mask: dict,
    video_predictor,
    offload_video_to_cpu: bool = True,
) -> dict:
    """Process one chunk with SAM2 video predictor using mask prompts on frame 0.

    Steps:
    1. Write frames to a temp JPEG directory
    2. init_state from that directory (loads all frames into memory)
    3. Delete temp directory (frames now in inference_state)
    4. add_new_mask() for each object on frame 0
    5. propagate_in_video() through all frames
    6. Return outputs_per_frame dict (global frame indices)

    Args:
        chunk_frames: List of RGB numpy arrays
        global_start: Global frame index of the first frame in this chunk
        chunk_idx: Chunk number (for metadata stamping)
        obj_id_to_mask: Dict of obj_id (int) -> (H, W) uint8 or bool mask
        video_predictor: SAM2VideoPredictor instance
        offload_video_to_cpu: If True, keep frames in CPU RAM (saves VRAM)

    Returns:
        outputs_per_frame: Dict mapping global_frame_idx -> output dict
    """
    num_frames = len(chunk_frames)
    model_type = "gs2_grounded" if chunk_idx == 0 else "gs2_tracker"

    # --- Write frames to temp dir, init SAM2 state, then delete temp dir ---
    temp_dir = tempfile.mkdtemp(prefix=f"gs2_chunk{chunk_idx}_")
    try:
        logger.info(f"  Writing {num_frames} frames to temp dir...")
        _write_frames_to_dir(chunk_frames, temp_dir)

        logger.info("  Initialising SAM2 inference state...")
        with torch.inference_mode():
            inference_state = video_predictor.init_state(
                video_path=temp_dir,
                offload_video_to_cpu=offload_video_to_cpu,
                offload_state_to_cpu=False,
                async_loading_frames=False,  # load all frames NOW so temp dir can be deleted
            )
    finally:
        # Frames are now in inference_state["images"] — safe to delete temp dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    # --- Seed frame 0 with object masks ---
    seeded = 0
    with torch.inference_mode():
        for obj_id, mask in obj_id_to_mask.items():
            mask_bool = mask.astype(bool)
            if mask_bool.sum() == 0:
                logger.warning(f"  Object {obj_id}: empty mask, skipping")
                continue
            video_predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=int(obj_id),
                mask=mask_bool,
            )
            seeded += 1

    if seeded == 0:
        logger.error("  No valid masks to seed — cannot propagate")
        del inference_state
        return {}

    logger.info(f"  Seeded {seeded} objects, propagating {num_frames} frames...")

    # --- Propagate ---
    outputs_per_frame = {}
    with torch.inference_mode():
        for frame_idx, obj_ids, video_res_masks in video_predictor.propagate_in_video(
            inference_state
        ):
            global_frame_idx = global_start + frame_idx

            # video_res_masks: (N, 1, H, W) logits; > 0.0 = foreground
            masks_np = (video_res_masks > 0.0).squeeze(1).cpu().numpy()  # (N, H, W)

            # Bounding boxes from masks
            boxes = []
            for m in masks_np:
                ys, xs = np.where(m)
                if len(ys) > 0:
                    boxes.append(
                        [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
                    )
                else:
                    boxes.append([0.0, 0.0, 0.0, 0.0])

            active_ids = list(obj_ids)[: len(masks_np)]

            outputs_per_frame[global_frame_idx] = {
                "object_ids": np.array(active_ids),
                "boxes": np.array(boxes),
                "masks": masks_np,
                "scores": np.ones(len(active_ids), dtype=np.float32),
                "obj_id_to_tracker_score": {},
                "_chunk_idx": chunk_idx,
                "_model_type": model_type,
                "_is_chunk_start": frame_idx == 0,
            }

            if frame_idx % 25 == 0 or frame_idx == num_frames - 1:
                logger.info(
                    f"  [chunk {chunk_idx}] frame {frame_idx + 1}/{num_frames} "
                    f"({100 * (frame_idx + 1) / num_frames:.0f}%)"
                )

    del inference_state
    return outputs_per_frame


# ---------------------------------------------------------------------------
# Carryover extraction
# ---------------------------------------------------------------------------


def _extract_carryover_masks(
    outputs_per_frame: dict,
    max_lookback: int = 10,
    min_objects: int = 1,
) -> dict:
    """Extract per-object masks from the last frame(s) of a processed chunk.

    Searches backwards from the last frame to find one with at least
    min_objects non-empty masks.

    Returns:
        Dict of obj_id (int) -> (H, W) uint8 mask
    """
    if not outputs_per_frame:
        return {}

    sorted_frames = sorted(outputs_per_frame.keys(), reverse=True)

    for frame_idx in sorted_frames[:max_lookback]:
        out = outputs_per_frame[frame_idx]
        masks = out["masks"]  # (N, H, W)
        obj_ids = out["object_ids"]

        carryover = {}
        for i, oid in enumerate(obj_ids):
            m = masks[i]
            if m.sum() > 0:
                carryover[int(oid)] = m.astype(np.uint8)

        if len(carryover) >= min_objects:
            logger.info(
                f"  Carryover: {len(carryover)} objects extracted from frame {frame_idx}"
            )
            return carryover

    logger.warning(
        f"  Could not extract carryover masks (all empty in last {max_lookback} frames)"
    )
    return {}


# ---------------------------------------------------------------------------
# Annotated video
# ---------------------------------------------------------------------------


def _create_annotated_video(
    video_path: str,
    all_outputs_per_frame: dict,
    output_path: str,
    fps: float,
) -> None:
    """Write an annotated video with masks, bounding boxes, and ID labels.

    Streams frames sequentially through ffmpeg (libx264/H.264) to avoid the
    seek-corruption and codec issues that come with cv2.VideoWriter + mp4v.
    """
    import subprocess

    frame_indices = sorted(all_outputs_per_frame.keys())
    if not frame_indices:
        logger.warning("No frames to annotate — skipping annotated video")
        return

    tracked_set = set(frame_indices)
    max_frame = max(frame_indices)

    cap = cv2.VideoCapture(str(video_path))
    ret, sample = cap.read()
    if not ret:
        logger.warning("Could not read frame for annotated video — skipping")
        cap.release()
        return
    h, w = sample.shape[:2]
    cap.release()

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{w}x{h}", "-pix_fmt", "bgr24", "-r", str(fps),
        "-i", "pipe:0",
        "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        str(output_path),
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    mask_annotator = sv.MaskAnnotator()

    cap = cv2.VideoCapture(str(video_path))
    current_frame = 0
    frames_written = 0

    while current_frame <= max_frame:
        ret, frame = cap.read()
        if not ret:
            logger.warning(f"Source video ended at frame {current_frame} — stopping annotated video")
            break

        if current_frame in tracked_set:
            out = all_outputs_per_frame[current_frame]
            obj_ids = out["object_ids"].tolist()
            masks = out["masks"]  # (N, H, W)

            if obj_ids:
                masks_uint8 = (masks > 0).astype(np.uint8)
                xyxy = sv.mask_to_xyxy(masks_uint8)
                detections = sv.Detections(
                    xyxy=xyxy,
                    mask=masks_uint8.astype(bool),
                    class_id=np.array(obj_ids, dtype=np.int32),
                )
                annotated = mask_annotator.annotate(scene=frame.copy(), detections=detections)
                annotated = box_annotator.annotate(scene=annotated, detections=detections)
                annotated = label_annotator.annotate(
                    annotated,
                    detections=detections,
                    labels=[str(oid) for oid in obj_ids],
                )
                proc.stdin.write(annotated.tobytes())
            else:
                proc.stdin.write(frame.tobytes())
        else:
            proc.stdin.write(frame.tobytes())

        frames_written += 1
        current_frame += 1

    cap.release()
    proc.stdin.close()
    proc.wait()

    if proc.returncode == 0:
        logger.info(f"Annotated video saved to: {output_path}")
    else:
        logger.error(f"ffmpeg exited with code {proc.returncode} — annotated video may be incomplete")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def _run_pipeline(cfg, run_dir: Path, config_path: Path | None = None) -> None:
    """Full GS2 manual chunking pipeline for a single video."""
    job_type = cfg.get("job_type", "gs2_manual")
    log_file = setup_logger(run_dir, job_type=job_type)

    logger.info("=" * 60)
    logger.info("GS2 Manual Chunking Pipeline")
    logger.info("=" * 60)

    if config_path is not None and config_path.exists():
        shutil.copy(config_path, run_dir / config_path.name)
        logger.info(f"Config copied to {run_dir / config_path.name}")
    if config_path is not None:
        logger.info(f"Loaded config: {config_path}")
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg, resolve=True)}")
    logger.info("=" * 60)
    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Log file: {log_file}")

    total_start = time.perf_counter()

    # --- Config values ---
    video_path = cfg.video_path
    text_prompt = cfg.text_prompt
    start_frame = cfg.get("start_frame", 0)
    max_frames = cfg.get("max_frames_to_track", 0)
    max_chunks = cfg.get("max_chunks", None)

    gs2_cfg = cfg.get("gs2", {})
    model_size = gs2_cfg.get("model_size", "large")
    box_threshold = gs2_cfg.get("box_threshold", 0.25)
    text_threshold = gs2_cfg.get("text_threshold", 0.3)
    offload_video_to_cpu = gs2_cfg.get("offload_video_to_cpu", True)

    # --- Video metadata ---
    fps, total_frames_raw = get_video_metadata(video_path)
    logger.info(f"Video: {video_path}")
    logger.info(f"  {total_frames_raw} frames at {fps:.2f} FPS")

    if start_frame > 0:
        if start_frame >= total_frames_raw:
            logger.error(
                f"start_frame ({start_frame}) >= total_frames ({total_frames_raw}), aborting"
            )
            return
        total_frames = total_frames_raw - start_frame
        logger.info(f"  Starting from frame {start_frame}: {total_frames} frames remaining")
    else:
        total_frames = total_frames_raw

    if max_frames and max_frames > 0:
        total_frames = min(total_frames, max_frames)
        logger.info(f"  Capped to {total_frames} frames (max_frames_to_track={max_frames})")

    # --- Build chunks ---
    raw_chunks = cfg.get("manual_chunk_frames", None)
    if raw_chunks is None:
        raise ValueError(
            "manual_chunk_frames is required in config for run_gs2_manual_chunking"
        )

    manual_chunk_frames = OmegaConf.to_container(raw_chunks, resolve=True)
    chunks = build_manual_chunks(manual_chunk_frames)

    if max_chunks is not None:
        chunks = chunks[:int(max_chunks)]
        logger.info(f"max_chunks={max_chunks}: processing only first {len(chunks)} chunks")

    logger.info(f"Chunks: {len(chunks)}")
    for i, (s, e, mtype) in enumerate(chunks):
        logger.info(f"  Chunk {i}: frames {s}-{e} ({mtype}, {(e - s) / fps:.1f}s)")

    # --- Device ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # Enable TF32 on Ampere+ GPUs (free matmul speedup)
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("TF32 enabled (Ampere+ GPU)")

    # Enter bfloat16 autocast for the rest of the pipeline.
    # This lets SAM2 use Flash/Memory-Efficient Attention (float32 disables it).
    # The context is intentionally left open for the entire pipeline run.
    torch.autocast(device_type=device, dtype=torch.bfloat16).__enter__()
    logger.info("bfloat16 autocast active — Flash Attention enabled")

    # --- Load models ---
    logger.info("Building SAM2 models...")
    video_predictor, image_predictor = _build_sam2_models(device, model_size)

    logger.info("Building GroundingDINO...")
    gdino_processor, gdino_model = _build_grounding_dino(device)

    # --- Chunk loop ---
    all_outputs_per_frame = {}
    chunk_info_list = []
    carryover_masks: dict = {}
    results_path = run_dir / "tracking_outputs.parquet"

    for chunk_idx, (start_idx, end_idx, chunk_type) in enumerate(chunks):
        global_chunk_start = start_frame + start_idx
        global_chunk_end = start_frame + end_idx

        chunk_start_time = time.perf_counter()

        if torch.cuda.is_available():
            alloc_mb = torch.cuda.memory_allocated() / 1024**2
            res_mb = torch.cuda.memory_reserved() / 1024**2
            logger.info(
                f"GPU memory before chunk {chunk_idx}: "
                f"{alloc_mb:.1f} MB allocated, {res_mb:.1f} MB reserved"
            )

        logger.info(
            f"Chunk {chunk_idx}/{len(chunks) - 1}: "
            f"frames {start_idx}-{end_idx} ({(end_idx - start_idx) / fps:.1f}s)"
        )

        # Load frames into memory
        logger.info(f"  Loading frames {global_chunk_start}–{global_chunk_end}...")
        chunk_frames = load_video_frames_range(
            video_path, global_chunk_start, global_chunk_end
        )
        num_frames = len(chunk_frames)
        logger.info(f"  Loaded {num_frames} frames")

        chunk_info = {
            "chunk_idx": chunk_idx,
            "frame_range": [start_idx, end_idx],
            "model_type": "gs2_grounded" if chunk_idx == 0 else "gs2_tracker",
            "prompt_type": None,
            "num_objects": 0,
        }

        # --- Determine prompts ---
        if chunk_idx == 0:
            # Detect objects in first frame
            logger.info("  Running GroundingDINO + SAM2 image predictor on first frame...")
            gs2_cfg = cfg.get("gs2", {})
            obj_id_to_mask, _labels = _detect_and_segment_first_frame(
                frame_rgb=chunk_frames[0],
                text_prompt=text_prompt,
                gdino_processor=gdino_processor,
                gdino_model=gdino_model,
                image_predictor=image_predictor,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                device=device,
                min_objects=cfg.get("min_objects_for_tracking", 1),
                min_box_area_fraction=gs2_cfg.get("min_box_area_fraction", 0.003),
                max_box_area_fraction=gs2_cfg.get("max_box_area_fraction", 0.40),
            )

            if not obj_id_to_mask:
                logger.error("No objects detected on first frame — aborting pipeline")
                return

            chunk_info["prompt_type"] = "grounding_dino"
            chunk_info["num_objects"] = len(obj_id_to_mask)

            # Free detection models — not needed for subsequent chunks
            logger.info(
                "  Freeing GroundingDINO + SAM2 image predictor to reclaim VRAM..."
            )
            del gdino_model, gdino_processor, image_predictor
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                alloc_mb = torch.cuda.memory_allocated() / 1024**2
                logger.info(f"  GPU after model free: {alloc_mb:.1f} MB allocated")

        else:
            # Use carryover masks from previous chunk
            if not carryover_masks:
                logger.error(
                    f"No carryover masks available for chunk {chunk_idx} — aborting"
                )
                return

            obj_id_to_mask = carryover_masks
            chunk_info["prompt_type"] = "mask_carryover"
            chunk_info["num_objects"] = len(obj_id_to_mask)
            logger.info(
                f"  Using carryover masks: {len(obj_id_to_mask)} objects "
                f"(IDs: {list(obj_id_to_mask.keys())})"
            )

        # --- Process chunk ---
        chunk_outputs = _process_chunk(
            chunk_frames=chunk_frames,
            global_start=global_chunk_start,
            chunk_idx=chunk_idx,
            obj_id_to_mask=obj_id_to_mask,
            video_predictor=video_predictor,
            offload_video_to_cpu=offload_video_to_cpu,
        )

        if not chunk_outputs:
            logger.error(f"Chunk {chunk_idx} produced no outputs — stopping")
            break

        # --- Extract carryover for next chunk ---
        carryover_masks = _extract_carryover_masks(chunk_outputs)

        # --- Timing ---
        elapsed = time.perf_counter() - chunk_start_time
        fps_achieved = num_frames / elapsed if elapsed > 0 else 0.0
        chunk_info["timing"] = {
            "elapsed_seconds": round(elapsed, 3),
            "fps": round(fps_achieved, 2),
            "num_frames": num_frames,
        }

        # --- Incremental parquet save ---
        chunk_df = process_tracking_outputs(chunk_outputs)
        chunk_df = chunk_df.sort_index()
        if results_path.exists():
            existing_df = pd.read_parquet(results_path)
            chunk_df = pd.concat([existing_df, chunk_df]).sort_index()
            del existing_df
        chunk_df.to_parquet(results_path)
        del chunk_df
        logger.info(f"Tracking results saved to {results_path} (after chunk {chunk_idx})")

        n_frames_chunk = len(chunk_outputs)
        all_outputs_per_frame.update(chunk_outputs)
        chunk_info_list.append(chunk_info)

        # --- Cleanup ---
        del chunk_frames, chunk_outputs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            alloc_mb = torch.cuda.memory_allocated() / 1024**2
            res_mb = torch.cuda.memory_reserved() / 1024**2
            logger.info(
                f"GPU memory after chunk {chunk_idx}: "
                f"{alloc_mb:.1f} MB allocated, {res_mb:.1f} MB reserved"
            )

        logger.info(
            f"Chunk {chunk_idx} complete: {n_frames_chunk} frames, "
            f"{elapsed:.2f}s, {fps_achieved:.2f} FPS"
        )

    logger.info(f"All chunks complete: {len(all_outputs_per_frame)} total frames processed")

    # --- Save chunk info ---
    chunk_info_path = run_dir / "chunk_info.json"
    with open(chunk_info_path, "w") as f:
        json.dump({"chunks": chunk_info_list}, f, indent=2)
    logger.info(f"Chunk info saved to: {chunk_info_path}")

    # --- Annotated video ---
    logger.info("Creating annotated video...")
    annotated_video_path = run_dir / "annotated_video.mp4"
    _create_annotated_video(
        video_path=video_path,
        all_outputs_per_frame=all_outputs_per_frame,
        output_path=str(annotated_video_path),
        fps=fps,
    )

    # --- Final summary ---
    total_elapsed = time.perf_counter() - total_start
    total_fps = (
        len(all_outputs_per_frame) / total_elapsed if total_elapsed > 0 else 0.0
    )

    logger.info("=" * 60)
    logger.info(f"Results saved to: {run_dir}")
    logger.info(
        f"Pipeline complete in {total_elapsed:.2f}s  "
        f"({len(all_outputs_per_frame)} frames, {total_fps:.2f} FPS overall)"
    )
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    cfg = _cfg
    job_type = cfg.get("job_type", "gs2_manual")
    run_dir = create_run_directory(Path(cfg.output_dir), job_type)
    _run_pipeline(cfg, run_dir, config_path=Path(_args.config))


if __name__ == "__main__":
    main()
