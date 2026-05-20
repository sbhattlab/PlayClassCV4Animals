"""
Grounded-SAM-2 (gs2) tracker pipeline — faithful IDEA-Research baseline.

A baseline that sits between YOLO+BoT-SORT and SAM3 in sophistication:
GroundingDINO detects birds on the seed frame, SAM2 image predictor refines
boxes to masks, and SAM2 video predictor propagates masks within each chunk.
Between chunks, the mask state of the last frame of chunk N is carried over as
the prompt for chunk N+1 — preserving identities without re-grounding.

The grounding step is **strict** by design — single GroundingDINO call on
frame 0 of chunk 0 at the user-specified `box_threshold`, no retry loop, no
area filter, accept whatever comes back. This matches the IDEA-Research
reference behavior (https://github.com/IDEA-Research/Grounded-SAM-2). The
*only* unavoidable adaptation is chunking-with-mask-carryover, because
SAM2's `init_state` cannot fit a 15-minute video in VRAM.
"""

from __future__ import annotations

import gc
import json
import shutil
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from loguru import logger
from omegaconf import OmegaConf
from PIL import Image
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
)

from src.config import (
    create_run_directory,
    create_video_run_directory,
    setup_logger,
)
from src.io import get_video_metadata, load_video_frames_sequential
from src.tracker.chunking import chunk_video_frames_adaptive
from src.tracker.masks import process_tracking_outputs

REPO_ROOT = Path(__file__).resolve().parents[2]
GS2_DIR = REPO_ROOT / "ext" / "Grounded-SAM-2"
# Larger Swin-B variant (~340M params). Defensible "leg up" for the gs2
# baseline — gives the canonical grounder the best HF-published weights.
DEFAULT_GDINO_MODEL_ID = "IDEA-Research/grounding-dino-base"

_SIZE_TO_CONFIG_CODE = {
    "large": "l",
    "base_plus": "b+",
    "small": "s",
    "tiny": "t",
}


def _build_sam2_models(device: str, model_size: str = "large"):
    size_code = _SIZE_TO_CONFIG_CODE.get(model_size, "l")
    checkpoint = str(GS2_DIR / f"checkpoints/sam2.1_hiera_{model_size}.pt")
    model_cfg = f"configs/sam2.1/sam2.1_hiera_{size_code}.yaml"

    logger.info(f"Building SAM2 video predictor ({model_size}) from {checkpoint}")
    video_predictor = build_sam2_video_predictor(model_cfg, checkpoint, device=device)

    logger.info("Building SAM2 image predictor")
    sam2_model = build_sam2(model_cfg, checkpoint, device=device)
    image_predictor = SAM2ImagePredictor(sam2_model)

    return video_predictor, image_predictor


def _build_grounding_dino(device: str, model_id: str = DEFAULT_GDINO_MODEL_ID):
    logger.info(f"Loading GroundingDINO from {model_id}")
    processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    return processor, model


def _write_frames_to_dir(frames: list[np.ndarray], out_dir: str) -> None:
    for i, frame_rgb in enumerate(frames):
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(Path(out_dir) / f"{i:05d}.jpg"), frame_bgr)


def _gdino_detect_boxes(
    frame_rgb: np.ndarray,
    text_prompt: str,
    gdino_processor,
    gdino_model,
    box_threshold: float,
    text_threshold: float,
    device: str,
) -> tuple[np.ndarray, list[str]]:
    """Single GroundingDINO call. Returns (boxes, labels) — no masking."""
    image_pil = Image.fromarray(frame_rgb)
    inputs = gdino_processor(
        images=image_pil, text=text_prompt, return_tensors="pt"
    ).to(device)
    with torch.inference_mode():
        gdino_outputs = gdino_model(**inputs)
    results = gdino_processor.post_process_grounded_object_detection(
        gdino_outputs,
        inputs.input_ids,
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[image_pil.size[::-1]],
    )
    boxes = results[0]["boxes"].detach().cpu().numpy()
    labels = [str(lbl) for lbl in results[0]["labels"]]
    return boxes, labels


def _refine_boxes_to_masks(
    frame_rgb: np.ndarray,
    boxes: np.ndarray,
    image_predictor: SAM2ImagePredictor,
    start_obj_id: int = 1,
) -> dict[int, np.ndarray]:
    """Refine GDINO boxes into SAM2 masks. obj_ids are assigned sequentially
    starting at start_obj_id (lets the caller continue numbering after a
    re-init so IDs don't collide with previously-tracked objects).
    """
    if boxes.shape[0] == 0:
        return {}
    image_predictor.set_image(frame_rgb)
    obj_id_to_mask: dict[int, np.ndarray] = {}
    for offset, box in enumerate(boxes):
        box_k = np.asarray(box, dtype=np.float32)[None]
        with torch.inference_mode():
            masks_k, _scores_k, _ = image_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box_k,
                multimask_output=False,
            )
        m = masks_k
        if m.ndim == 4:
            m = m.squeeze(0).squeeze(0)
        elif m.ndim == 3:
            m = m.squeeze(0)
        obj_id_to_mask[int(start_obj_id + offset)] = m.astype(np.uint8)
    return obj_id_to_mask


def _select_best_seed_frame(
    frames: list[np.ndarray],
    text_prompt: str,
    gdino_processor,
    gdino_model,
    box_threshold: float,
    text_threshold: float,
    device: str,
    scan_window: int = 125,
) -> tuple[int, np.ndarray, list[str]]:
    """Scan the first `scan_window` frames with GroundingDINO; pick the frame
    with the most detections at the given threshold. Tie-break: earliest.

    Returns (best_frame_idx, best_boxes, best_labels). If no frame yields
    any detections, returns (0, empty_array, []).
    """
    n_scan = min(scan_window, len(frames))
    logger.info(
        f"  Best-frame seed selection: scanning {n_scan} frames "
        f"(box_threshold={box_threshold:.2f})"
    )
    best_idx = 0
    best_boxes = np.zeros((0, 4), dtype=np.float32)
    best_labels: list[str] = []
    best_count = -1
    for i in range(n_scan):
        boxes, labels = _gdino_detect_boxes(
            frames[i],
            text_prompt,
            gdino_processor,
            gdino_model,
            box_threshold,
            text_threshold,
            device,
        )
        n = len(boxes)
        if n > best_count:
            best_count = n
            best_idx = i
            best_boxes = boxes
            best_labels = labels
    logger.info(
        f"  Best seed frame: idx={best_idx} with {best_count} detections "
        f"(scanned {n_scan} frames)"
    )
    return best_idx, best_boxes, best_labels


def _detect_and_segment_first_frame(
    frame_rgb: np.ndarray,
    text_prompt: str,
    gdino_processor,
    gdino_model,
    image_predictor: SAM2ImagePredictor,
    box_threshold: float,
    text_threshold: float,
    device: str,
) -> tuple[dict, list[str]]:
    """Strict IDEA-Research grounding: single GroundingDINO call on the given
    frame at the user-specified thresholds. No retry loop, no area filter.
    Returns ({}, []) if zero boxes survive the threshold.
    """
    boxes, labels = _gdino_detect_boxes(
        frame_rgb,
        text_prompt,
        gdino_processor,
        gdino_model,
        box_threshold,
        text_threshold,
        device,
    )
    logger.info(
        f"GroundingDINO: {len(boxes)} detections at "
        f"box_threshold={box_threshold:.2f}, text_threshold={text_threshold:.2f}"
    )
    if boxes.shape[0] == 0:
        logger.warning(
            f"GroundingDINO: no detections for '{text_prompt}' on seed frame"
        )
        return {}, []
    obj_id_to_mask = _refine_boxes_to_masks(frame_rgb, boxes, image_predictor)
    logger.info(
        f"SAM2 image predictor: generated {len(obj_id_to_mask)} masks "
        f"(obj_ids: {list(obj_id_to_mask.keys())})"
    )
    return obj_id_to_mask, labels


def _process_chunk(
    chunk_frames: list[np.ndarray],
    global_start: int,
    chunk_idx: int,
    obj_id_to_mask: dict,
    video_predictor,
    offload_video_to_cpu: bool = True,
    seed_frame_offset: int = 0,
) -> dict:
    """Run SAM2 video propagation over a chunk.

    seed_frame_offset: the chunk-local frame index at which the mask prompts
    apply. 0 = mask prompts are anchored to the first frame of the chunk
    (standard for carryover). >0 = mask prompts are anchored later in the
    chunk (best-frame selection chose that frame); propagation runs from
    there forward, so chunk frames before seed_frame_offset have no
    tracking outputs in this run.
    """
    num_frames = len(chunk_frames)
    model_type = "gs2_grounded" if chunk_idx == 0 else "gs2_tracker"

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
                async_loading_frames=False,
            )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    seeded = 0
    with torch.inference_mode():
        for obj_id, mask in obj_id_to_mask.items():
            mask_bool = mask.astype(bool)
            if mask_bool.sum() == 0:
                logger.warning(f"  Object {obj_id}: empty mask, skipping")
                continue
            video_predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=seed_frame_offset,
                obj_id=int(obj_id),
                mask=mask_bool,
            )
            seeded += 1

    if seeded == 0:
        logger.error("  No valid masks to seed — cannot propagate")
        del inference_state
        return {}

    logger.info(
        f"  Seeded {seeded} objects at chunk-local frame {seed_frame_offset}, "
        f"propagating {num_frames - seed_frame_offset} frames..."
    )

    outputs_per_frame: dict = {}
    with torch.inference_mode():
        for frame_idx, obj_ids, video_res_masks in video_predictor.propagate_in_video(
            inference_state
        ):
            global_frame_idx = global_start + frame_idx
            masks_np = (video_res_masks > 0.0).squeeze(1).cpu().numpy()

            boxes = []
            for m in masks_np:
                ys, xs = np.where(m)
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


def _extract_carryover_masks(
    outputs_per_frame: dict,
    max_lookback: int = 10,
    min_objects: int = 1,
) -> dict:
    if not outputs_per_frame:
        return {}

    sorted_frames = sorted(outputs_per_frame.keys(), reverse=True)
    for frame_idx in sorted_frames[:max_lookback]:
        out = outputs_per_frame[frame_idx]
        masks = out["masks"]
        obj_ids = out["object_ids"]

        carryover: dict[int, np.ndarray] = {}
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


def _run_single_video(cfg, run_dir: Path, config_path: Path | None = None) -> None:
    job_type = cfg.get("job_type", "gs2_fixed")
    log_file = setup_logger(run_dir, job_type=job_type)

    logger.info("=" * 60)
    logger.info("Grounded-SAM-2 Tracker (fixed chunking)")
    logger.info("=" * 60)

    if config_path is not None and Path(config_path).exists():
        shutil.copy(config_path, run_dir / Path(config_path).name)
        logger.info(f"Config copied to {run_dir / Path(config_path).name}")
    logger.info("CONFIGURATION")
    logger.info(f"\n{OmegaConf.to_yaml(cfg, resolve=True)}")
    logger.info("=" * 60)
    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Log file: {log_file}")

    total_start = time.perf_counter()

    video_path = cfg.video_path
    text_prompt = cfg.text_prompt
    start_frame = cfg.get("start_frame", 0)
    max_frames = cfg.get("max_frames_to_track", 0)

    gs2_cfg = cfg.get("gs2", {})
    model_size = gs2_cfg.get("model_size", "large")
    box_threshold = gs2_cfg.get("box_threshold", 0.25)
    text_threshold = gs2_cfg.get("text_threshold", 0.3)
    offload_video_to_cpu = gs2_cfg.get("offload_video_to_cpu", True)
    gdino_model_id = gs2_cfg.get("gdino_model_id", DEFAULT_GDINO_MODEL_ID)
    # Parity-recovery flags (default off = strict IDEA-Research reference).
    enable_recovery = bool(gs2_cfg.get("enable_recovery", False))
    seed_scan_window = int(gs2_cfg.get("seed_scan_window", 125))

    chunk_seconds = float(cfg.get("chunk_seconds", 60))
    max_lookback_frames = int(cfg.get("max_lookback_frames", 10))

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
        logger.info(
            f"  Starting from frame {start_frame}: {total_frames} frames remaining"
        )
    else:
        total_frames = total_frames_raw

    if max_frames and max_frames > 0:
        total_frames = min(total_frames, max_frames)
        logger.info(f"  Capped to {total_frames} frames (max_frames_to_track={max_frames})")

    chunks = chunk_video_frames_adaptive(
        total_frames,
        fps,
        chunk_seconds,
        per_frame_metrics=None,
    )

    logger.info(f"Chunks: {len(chunks)} ({chunk_seconds:.1f}s each, fixed)")
    for i, (s, e, _mtype) in enumerate(chunks):
        logger.info(f"  Chunk {i}: frames {s}-{e} ({(e - s) / fps:.1f}s)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("TF32 enabled (Ampere+ GPU)")

    # Enter bfloat16 autocast for the whole pipeline (Flash Attention requires it).
    torch.autocast(device_type=device, dtype=torch.bfloat16).__enter__()
    logger.info("bfloat16 autocast active — Flash Attention enabled")

    logger.info("Building SAM2 models...")
    video_predictor, image_predictor = _build_sam2_models(device, model_size)

    logger.info("Building GroundingDINO...")
    gdino_processor, gdino_model = _build_grounding_dino(device, gdino_model_id)

    chunk_info_list: list[dict] = []
    carryover_masks: dict = {}
    results_path = run_dir / "tracking_outputs.parquet"
    total_frames_written = 0
    next_obj_id = 1  # incremented across re-inits so fresh objects don't collide

    if enable_recovery:
        logger.info(
            f"Recovery enabled: best-frame seed selection (scan_window={seed_scan_window}); "
            "GDINO re-init on total carryover loss."
        )

    for chunk_idx, (start_idx, end_idx, _chunk_type) in enumerate(chunks):
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

        logger.info(f"  Loading frames {global_chunk_start}–{global_chunk_end}...")
        chunk_frames = load_video_frames_sequential(
            video_path, global_chunk_start, global_chunk_end
        )
        num_frames = len(chunk_frames)
        logger.info(f"  Loaded {num_frames} frames")

        chunk_info: dict = {
            "chunk_idx": chunk_idx,
            "frame_range": [start_idx, end_idx],
            "model_type": "gs2_grounded" if chunk_idx == 0 else "gs2_tracker",
            "prompt_type": None,
            "num_objects": 0,
        }

        # -------------------------------------------------------------
        # Determine prompts for this chunk
        # -------------------------------------------------------------
        # 4 cases:
        #   chunk 0, recovery off → GDINO on frame 0 (strict)
        #   chunk 0, recovery on  → GDINO best-frame search over scan_window
        #   chunk N>0, carryover OK → propagate carryover masks
        #   chunk N>0, carryover empty AND recovery on → GDINO best-frame
        #                                                 search; fresh IDs
        #   chunk N>0, carryover empty AND recovery off → abort
        if chunk_idx == 0:
            if enable_recovery:
                logger.info(
                    f"  Best-frame seed selection over first {seed_scan_window} frames..."
                )
                best_idx, best_boxes, _best_labels = _select_best_seed_frame(
                    frames=chunk_frames,
                    text_prompt=text_prompt,
                    gdino_processor=gdino_processor,
                    gdino_model=gdino_model,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                    device=device,
                    scan_window=seed_scan_window,
                )
                if best_boxes.shape[0] == 0:
                    logger.error(
                        f"GDINO returned 0 detections across all {seed_scan_window} "
                        "scanned frames — aborting pipeline"
                    )
                    return
                obj_id_to_mask = _refine_boxes_to_masks(
                    chunk_frames[best_idx],
                    best_boxes,
                    image_predictor,
                    start_obj_id=next_obj_id,
                )
                next_obj_id += len(obj_id_to_mask)
                chunk_info["prompt_type"] = "grounding_dino_best_frame"
                chunk_info["best_frame_idx"] = int(best_idx)
                chunk_info["num_objects"] = len(obj_id_to_mask)
                logger.info(
                    f"  Seed selected: chunk-local frame {best_idx} → "
                    f"{len(obj_id_to_mask)} masks (obj_ids: {list(obj_id_to_mask.keys())})"
                )
                # NOTE: when the best seed is not frame 0, masks are anchored
                # to that frame; the SAM2 video predictor will still be init'd
                # on the whole chunk, but add_new_mask is called at frame
                # best_idx, so frames 0..best_idx-1 of this chunk produce no
                # tracking output. Acceptable: matches SAM3's behavior on
                # frames before its chosen grounding frame.
                chunk_info["seed_frame_offset"] = int(best_idx)
            else:
                logger.info("  Running GroundingDINO + SAM2 image predictor on first frame...")
                obj_id_to_mask, _labels = _detect_and_segment_first_frame(
                    frame_rgb=chunk_frames[0],
                    text_prompt=text_prompt,
                    gdino_processor=gdino_processor,
                    gdino_model=gdino_model,
                    image_predictor=image_predictor,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                    device=device,
                )
                if not obj_id_to_mask:
                    logger.error("No objects detected on first frame — aborting pipeline")
                    return
                next_obj_id += len(obj_id_to_mask)
                chunk_info["prompt_type"] = "grounding_dino"
                chunk_info["num_objects"] = len(obj_id_to_mask)
                chunk_info["seed_frame_offset"] = 0

            if not enable_recovery:
                logger.info("  Freeing GroundingDINO + SAM2 image predictor to reclaim VRAM...")
                del gdino_model, gdino_processor, image_predictor
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    alloc_mb = torch.cuda.memory_allocated() / 1024**2
                    logger.info(f"  GPU after model free: {alloc_mb:.1f} MB allocated")
            else:
                # Keep GDINO + image_predictor resident for re-init recovery.
                logger.info(
                    "  Recovery on: keeping GDINO + SAM2 image predictor resident."
                )
        else:
            if carryover_masks:
                obj_id_to_mask = carryover_masks
                chunk_info["prompt_type"] = "mask_carryover"
                chunk_info["num_objects"] = len(obj_id_to_mask)
                chunk_info["seed_frame_offset"] = 0
                logger.info(
                    f"  Using carryover masks: {len(obj_id_to_mask)} objects "
                    f"(IDs: {list(obj_id_to_mask.keys())})"
                )
            elif enable_recovery:
                logger.warning(
                    f"  Carryover empty at chunk {chunk_idx} — attempting GDINO re-init..."
                )
                best_idx, best_boxes, _best_labels = _select_best_seed_frame(
                    frames=chunk_frames,
                    text_prompt=text_prompt,
                    gdino_processor=gdino_processor,
                    gdino_model=gdino_model,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                    device=device,
                    scan_window=seed_scan_window,
                )
                if best_boxes.shape[0] == 0:
                    logger.error(
                        f"Re-init failed at chunk {chunk_idx}: 0 detections across "
                        f"{seed_scan_window} scanned frames — aborting pipeline"
                    )
                    return
                obj_id_to_mask = _refine_boxes_to_masks(
                    chunk_frames[best_idx],
                    best_boxes,
                    image_predictor,
                    start_obj_id=next_obj_id,
                )
                next_obj_id += len(obj_id_to_mask)
                chunk_info["prompt_type"] = "grounding_dino_reinit"
                chunk_info["best_frame_idx"] = int(best_idx)
                chunk_info["seed_frame_offset"] = int(best_idx)
                chunk_info["num_objects"] = len(obj_id_to_mask)
                logger.info(
                    f"  Re-init successful: chunk-local frame {best_idx} → "
                    f"{len(obj_id_to_mask)} masks (fresh obj_ids: {list(obj_id_to_mask.keys())})"
                )
            else:
                logger.error(
                    f"No carryover masks available for chunk {chunk_idx} — aborting"
                )
                return

        chunk_outputs = _process_chunk(
            chunk_frames=chunk_frames,
            global_start=global_chunk_start,
            chunk_idx=chunk_idx,
            obj_id_to_mask=obj_id_to_mask,
            video_predictor=video_predictor,
            offload_video_to_cpu=offload_video_to_cpu,
            seed_frame_offset=int(chunk_info.get("seed_frame_offset", 0)),
        )

        if not chunk_outputs:
            logger.error(f"Chunk {chunk_idx} produced no outputs — stopping")
            break

        carryover_masks = _extract_carryover_masks(
            chunk_outputs, max_lookback=max_lookback_frames
        )

        elapsed = time.perf_counter() - chunk_start_time
        fps_achieved = num_frames / elapsed if elapsed > 0 else 0.0
        chunk_info["timing"] = {
            "elapsed_seconds": round(elapsed, 3),
            "fps": round(fps_achieved, 2),
            "num_frames": num_frames,
        }

        chunk_df = process_tracking_outputs(chunk_outputs)
        chunk_df = chunk_df.sort_index()
        if results_path.exists():
            existing_df = pd.read_parquet(results_path)
            chunk_df = pd.concat([existing_df, chunk_df]).sort_index()
            del existing_df
        chunk_df.to_parquet(results_path)
        del chunk_df
        logger.info(
            f"Tracking results saved to {results_path} (after chunk {chunk_idx})"
        )

        n_frames_chunk = len(chunk_outputs)
        total_frames_written += n_frames_chunk
        chunk_info_list.append(chunk_info)

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

    logger.info(
        f"All chunks complete: {total_frames_written} total frames processed"
    )

    chunk_info_path = run_dir / "chunk_info.json"
    with open(chunk_info_path, "w") as f:
        json.dump({"chunks": chunk_info_list}, f, indent=2)
    logger.info(f"Chunk info saved to: {chunk_info_path}")

    total_elapsed = time.perf_counter() - total_start
    total_fps = total_frames_written / total_elapsed if total_elapsed > 0 else 0.0

    logger.info("=" * 60)
    logger.info(f"Results saved to: {run_dir}")
    logger.info(
        f"Pipeline complete in {total_elapsed:.2f}s "
        f"({total_frames_written} frames, {total_fps:.2f} FPS overall)"
    )
    logger.info("=" * 60)


def _run_batch(cfg, batch_dir: Path, video_dir: Path, config_path: Path) -> None:
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

    for video_file in video_files:
        video_run_dir = create_video_run_directory(batch_dir, video_file.stem)

        video_cfg = OmegaConf.create(
            {**OmegaConf.to_container(cfg, resolve=True), "video_path": str(video_file)}
        )

        try:
            _run_single_video(video_cfg, video_run_dir, config_path=config_path)
        except Exception as e:
            logger.error(f"Failed processing {video_file.name}: {e}")
            continue


def run(cfg, config_path: str | Path) -> None:
    config_path = Path(config_path)
    video_path = cfg.get("video_path", None)
    video_dir = cfg.get("video_dir", None)

    if video_path and video_dir:
        raise ValueError("Specify exactly one of video_path or video_dir, not both.")
    if not video_path and not video_dir:
        raise ValueError("Must specify either video_path or video_dir.")

    job_type = cfg.get("job_type", "gs2_fixed")
    batch_dir = create_run_directory(Path(cfg.output_dir), job_type)

    if video_dir:
        _run_batch(cfg, batch_dir, Path(video_dir), config_path=config_path)
    else:
        video_stem = Path(video_path).stem
        run_dir = create_video_run_directory(batch_dir, video_stem)
        _run_single_video(cfg, run_dir, config_path=config_path)
