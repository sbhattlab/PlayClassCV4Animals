"""DINOv3 embedding extraction from tracked objects."""

from pathlib import Path

import numpy as np
import torch
from loguru import logger
from PIL import Image

from src.utils import load_video_frames_range
from .utils import _decode_rle_mask


def extract_embeddings(
    tracks,
    video_path: str | Path,
    model,
    processor,
    *,
    batch_size: int = 32,
    bg_attenuation: float = 0.9,
) -> dict[int, torch.Tensor]:
    """Extract DINOv3 CLS-token embeddings for all tracked objects across all frames.

    Parameters
    ----------
    tracks : pd.DataFrame
        Tracking outputs with MultiIndex ``["frame_idx", "object_id"]`` and columns
        ``bbox`` (list ``[x1, y1, x2, y2]``), ``counts``, ``size``.
    video_path : str | Path
        Path to the source video file.
    model : transformers.AutoModel
        DINOv3 model, already on device and in eval mode.
    processor : transformers.AutoImageProcessor
        DINOv3 image processor.
    batch_size : int
        Number of crops to process per forward pass.
    bg_attenuation : float
        Factor applied to background pixels (outside mask). 0.9 dims background to
        ~90% brightness, preserving context for the ViT.

    Returns
    -------
    dict[int, torch.Tensor]
        Mapping ``{object_id: Tensor(F_i, 768)}`` where ``F_i`` is the number of
        frames in which object_id appears.
    """
    frame_indices = tracks.index.get_level_values("frame_idx")
    min_frame = int(frame_indices.min())
    max_frame = int(frame_indices.max())

    logger.info(
        f"Loading video frames [{min_frame}, {max_frame}] from {Path(video_path).name}"
    )
    frames = load_video_frames_range(video_path, min_frame, max_frame + 1)
    logger.info(f"Loaded {len(frames)} frames")

    object_ids = sorted(tracks.index.get_level_values("object_id").unique())
    logger.info(f"Extracting crops for {len(object_ids)} objects")

    # Collect crops per object, maintaining order
    crops_per_object: dict[int, list[Image.Image]] = {oid: [] for oid in object_ids}
    all_crops: list[Image.Image] = []
    # Map from flat index in all_crops -> (object_id, per-object index)
    crop_mapping: list[tuple[int, int]] = []

    for oid in object_ids:
        obj_frames = tracks.xs(oid, level="object_id")
        for fidx in sorted(obj_frames.index):
            row = obj_frames.loc[fidx]

            # Decode mask
            mask = _decode_rle_mask(row["counts"], row["size"])

            # Get bbox [x1, y1, x2, y2]
            bbox = row["bbox"]
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

            # Clamp to frame bounds
            frame_np = frames[fidx - min_frame]
            h, w = frame_np.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            # Crop + attenuate background
            crop = frame_np[y1:y2, x1:x2].astype(np.float32)
            mask_crop = mask[y1:y2, x1:x2]
            crop[~mask_crop] *= bg_attenuation

            pil_crop = Image.fromarray(crop.astype(np.uint8))
            crops_per_object[oid].append(pil_crop)
            crop_mapping.append((oid, len(crops_per_object[oid]) - 1))
            all_crops.append(pil_crop)

    total_crops = len(all_crops)
    logger.info(f"Total crops: {total_crops}")

    if total_crops == 0:
        return {oid: torch.empty(0, 768) for oid in object_ids}

    # Determine model device/dtype from its parameters
    param = next(model.parameters())
    device = param.device
    model_dtype = param.dtype

    # Batch inference
    all_embeddings = []
    with torch.no_grad():
        for i in range(0, total_crops, batch_size):
            batch = all_crops[i : i + batch_size]
            inputs = processor(images=batch, return_tensors="pt").to(device)
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=model_dtype)
            outputs = model(**inputs)
            all_embeddings.append(outputs.pooler_output.float().cpu())

    flat_embeddings = torch.cat(all_embeddings, dim=0)  # (total_crops, 768)

    # Split back into per-object tensors
    result: dict[int, torch.Tensor] = {}
    offset = 0
    for oid in object_ids:
        n = len(crops_per_object[oid])
        if n > 0:
            result[oid] = flat_embeddings[offset : offset + n]
        else:
            result[oid] = torch.empty(0, 768)
        offset += n

    for oid, emb in result.items():
        logger.debug(f"  object {oid}: {emb.shape[0]} embeddings")

    return result
