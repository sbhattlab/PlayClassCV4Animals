"""DINOv3 embedding extraction from tracked objects."""

from pathlib import Path

import numpy as np
import pycocotools.mask as mask_util
import torch
from loguru import logger
from PIL import Image
from sklearn.decomposition import PCA as skPCA
from tqdm import tqdm

from src.dataset.crops import compute_union_origin, crop_frame
from src.io import load_video_frames_torchcodec as load_video_frames


def extract_embeddings(
    tracks,
    video_path: str | Path,
    model,
    processor,
    *,
    batch_size: int = 32,
    bbox_scale: float = 1.0,
    raw: bool = False,
    n_sample_frames: int | None = None,
    crop_mode: str = "bbox",
) -> dict[tuple, torch.Tensor]:
    """Extract DINOv3 embeddings for tracked objects, grouped by window.

    Parameters
    ----------
    tracks : pd.DataFrame
        Flat DataFrame with columns ``bird_id``, ``frame_idx``, ``window``,
        ``bbox`` (list ``[x1, y1, x2, y2]``).
        Must be pre-filtered to a single video.
    video_path : str | Path
        Path to the source video file.
    model : transformers.AutoModel
        DINOv3 model, already on device and in eval mode.
    processor : transformers.AutoImageProcessor
        DINOv3 image processor.
    batch_size : int
        Number of crops to process per forward pass.
    bbox_scale : float
        Scale factor for the bounding box crop. 1.0 uses the original bbox;
        1.25 adds 25% padding on each side (clamped to frame bounds).
    crop_mode : str
        One of ``'bbox'``, ``'plain256'``, ``'union512'``, ``'darken512'``,
        ``'roi512'``. See :mod:`src.dataset.crops` for details.

    Returns
    -------
    dict[tuple, torch.Tensor]
        ``{(video_id, bird_id, window): Tensor(F_w, D)}`` where ``F_w`` is the
        number of frames with valid crops in that window and ``D`` is the
        embedding dimension (768 for ViT-B, 1024 for ViT-L).
    """
    min_frame = int(tracks["frame_idx"].min())
    max_frame = int(tracks["frame_idx"].max())

    logger.info(
        f"Loading video frames [{min_frame}, {max_frame}] from {Path(video_path).name}"
    )
    frames = load_video_frames(video_path, min_frame, max_frame + 1)
    n_expected = max_frame - min_frame + 1
    n_loaded = len(frames)
    logger.info(f"Loaded {n_loaded} frames")
    if n_loaded < n_expected:
        n_skipped = n_expected - n_loaded
        last_valid = min_frame + n_loaded - 1
        logger.warning(
            f"Video decoder returned {n_loaded}/{n_expected} frames — "
            f"last {n_skipped} frames (>{last_valid}) will be skipped"
        )

    # Collect crops in (video_id, bird_id, window) order
    groups = sorted(tracks.groupby(["video_id", "bird_id", "window"]).groups.keys())
    logger.info(f"Extracting crops for {len(groups)} (video, bird, window) groups")

    crops_per_group: dict[tuple, int] = {}
    all_crops: list[Image.Image] = []

    for video_id, bird_id, window in groups:
        group_rows = tracks[
            (tracks["video_id"] == video_id)
            & (tracks["bird_id"] == bird_id)
            & (tracks["window"] == window)
        ].sort_values("frame_idx")
        n_before = len(all_crops)

        # Subsample frames (reduces storage for raw mode)
        if n_sample_frames is not None and len(group_rows) > n_sample_frames:
            indices = np.linspace(0, len(group_rows) - 1, n_sample_frames, dtype=int)
            group_rows = group_rows.iloc[indices]

        # Pre-compute union origin for union-based crop modes
        union_origin = None
        if crop_mode in ("union512", "darken512", "roi512"):
            first_local = int(group_rows.iloc[0]["frame_idx"]) - min_frame
            if first_local < len(frames):
                fh, fw = frames[first_local].shape[:2]
                all_bboxes = group_rows["bbox"].tolist()
                union_origin = compute_union_origin(all_bboxes, fh, fw)

        for _, row in group_rows.iterrows():
            frame_idx = int(row["frame_idx"])
            local_idx = frame_idx - min_frame
            if local_idx >= len(frames):
                continue

            crop, _ = crop_frame(
                frames[local_idx],
                row["bbox"],
                crop_mode,
                bbox_scale=bbox_scale,
                union_origin=union_origin,
            )
            if crop is None:
                continue
            all_crops.append(Image.fromarray(crop))

        crops_per_group[(video_id, bird_id, window)] = len(all_crops) - n_before

    n_crops = len(all_crops)
    assert (
        n_crops > 0
    ), "No valid crops extracted — check video loading and bbox validity"
    logger.info(f"Total crops: {n_crops}")

    # Determine model device/dtype from its parameters
    param = next(model.parameters())
    device = param.device
    model_dtype = param.dtype
    d_model = model.config.hidden_size

    # Batch inference
    all_embeddings = []
    n_patches = None  # set on first batch for raw mode
    with torch.inference_mode():
        for i in tqdm(
            range(0, n_crops, batch_size),
            total=(n_crops + batch_size - 1) // batch_size,
            desc="Extracting embeddings",
        ):
            batch = all_crops[i : i + batch_size]
            inputs = processor(images=batch, return_tensors="pt").to(device)
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=model_dtype)
            outputs = model(**inputs)
            if raw:
                # CLS + patch tokens: (B, 1+P, D) — token 0 is CLS
                tokens = outputs.last_hidden_state.float().cpu()
                if n_patches is None:
                    n_patches = tokens.shape[1]
                all_embeddings.append(tokens)
            else:
                all_embeddings.append(outputs.pooler_output.float().cpu())

    flat_embeddings = torch.cat(all_embeddings, dim=0)

    # Split back into per-(bird, window) tensors
    embeddings: dict[tuple, torch.Tensor] = {}
    offset = 0
    for key in groups:
        n = crops_per_group[key]
        if n > 0:
            if raw:
                # (F_w, P, D) — per-frame patch tokens
                embeddings[key] = flat_embeddings[offset : offset + n]
            else:
                # (F_w, D) — per-frame CLS tokens
                embeddings[key] = flat_embeddings[offset : offset + n]
        else:
            if raw:
                embeddings[key] = torch.empty(0, n_patches or 196, d_model)
            else:
                embeddings[key] = torch.empty(0, d_model)
        offset += n

    return embeddings


def _split_mask_thirds(mask_binary):
    """Split a binary mask into 3 regions along its major axis.

    Returns a label array (same shape as mask_binary) with values
    0=background, 1=tip_a, 2=center, 3=tip_b.
    Returns None if the mask has too few pixels.
    """
    ys, xs = np.where(mask_binary)
    if len(xs) < 10:
        return None
    coords = np.column_stack([xs, ys])
    pca = skPCA(n_components=min(2, coords.shape[0]))
    pca.fit(coords)
    axis = pca.components_[0]
    center = pca.mean_
    projections = (coords - center) @ axis
    p_min, p_max = projections.min(), projections.max()
    p_range = p_max - p_min
    if p_range < 1e-6:
        return None
    t1 = p_min + p_range / 3
    t2 = p_min + 2 * p_range / 3
    region_mask = np.zeros_like(mask_binary, dtype=np.uint8)
    for idx, (x, y) in enumerate(coords):
        if projections[idx] < t1:
            region_mask[y, x] = 1
        elif projections[idx] < t2:
            region_mask[y, x] = 2
        else:
            region_mask[y, x] = 3
    return region_mask


def extract_bodypart_embeddings(
    tracks,
    video_path: str | Path,
    model,
    processor,
    *,
    batch_size: int = 32,
    bbox_scale: float = 1.0,
    n_sample_frames: int | None = None,
) -> dict[tuple, torch.Tensor]:
    """Extract body-part-pooled DINOv3 embeddings (tip_a + center + tip_b).

    For each frame:
    1. Run DINOv3 on bbox crop → 14×14 patch tokens
    2. Decode SAM3 mask, find major axis via PCA, split into thirds
    3. Downsample mask thirds to 14×14 grid
    4. Mean-pool patch tokens per region → 3 × D

    Returns
    -------
    dict[tuple, torch.Tensor]
        ``{(video_id, bird_id, window): Tensor(F_w, 3*D)}``
    """
    min_frame = int(tracks["frame_idx"].min())
    max_frame = int(tracks["frame_idx"].max())

    logger.info(
        f"Loading video frames [{min_frame}, {max_frame}] from {Path(video_path).name}"
    )
    frames = load_video_frames(video_path, min_frame, max_frame + 1)
    n_loaded = len(frames)
    logger.info(f"Loaded {n_loaded} frames")

    groups = sorted(tracks.groupby(["video_id", "bird_id", "window"]).groups.keys())
    logger.info(f"Extracting body-part embeddings for {len(groups)} groups")

    # Determine model device/dtype
    param = next(model.parameters())
    device = param.device
    model_dtype = param.dtype
    n_register = getattr(model.config, "num_register_tokens", 0)
    d_model = model.config.hidden_size

    embeddings: dict[tuple, torch.Tensor] = {}

    for video_id, bird_id, window in tqdm(groups, desc="Body-part extraction"):
        group_rows = tracks[
            (tracks["video_id"] == video_id)
            & (tracks["bird_id"] == bird_id)
            & (tracks["window"] == window)
        ].sort_values("frame_idx")

        if n_sample_frames is not None and len(group_rows) > n_sample_frames:
            indices = np.linspace(0, len(group_rows) - 1, n_sample_frames, dtype=int)
            group_rows = group_rows.iloc[indices]

        # Collect crops and mask regions for this window
        crops = []
        mask_regions_14x14 = []  # per-crop: (14, 14) with values 0/1/2/3

        for _, row in group_rows.iterrows():
            frame_idx = int(row["frame_idx"])
            local_idx = frame_idx - min_frame
            if local_idx >= n_loaded:
                continue

            bbox = row["bbox"]
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

            if bbox_scale != 1.0:
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                bw, bh = (x2 - x1) * bbox_scale, (y2 - y1) * bbox_scale
                x1, y1 = int(cx - bw / 2), int(cy - bh / 2)
                x2, y2 = int(cx + bw / 2), int(cy + bh / 2)

            frame_np = frames[local_idx]
            h, w = frame_np.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            crop = frame_np[y1:y2, x1:x2]

            # Decode mask and split into body thirds
            counts = row["counts"]
            if isinstance(counts, str):
                counts = counts.encode("utf-8")
            rle = {"counts": counts, "size": row["size"]}
            full_mask = mask_util.decode(rle)
            mask_crop = full_mask[y1:y2, x1:x2]

            region_mask = _split_mask_thirds(mask_crop)
            if region_mask is None:
                continue

            # Downsample region mask to 14×14 (nearest-neighbor)
            region_14 = np.array(
                Image.fromarray(region_mask).resize((14, 14), Image.Resampling.NEAREST)
            )

            crops.append(Image.fromarray(crop))
            mask_regions_14x14.append(region_14)

        if not crops:
            embeddings[(video_id, bird_id, window)] = torch.empty(0, 3 * d_model)
            continue

        # Batch inference — get patch tokens
        n_crops = len(crops)
        all_tokens = []
        with torch.inference_mode():
            for i in range(0, n_crops, batch_size):
                batch = crops[i : i + batch_size]
                inputs = processor(images=batch, return_tensors="pt").to(device)
                inputs["pixel_values"] = inputs["pixel_values"].to(dtype=model_dtype)
                outputs = model(**inputs)
                # Skip CLS + register tokens, keep patch tokens only
                patch_start = 1 + n_register
                tokens = outputs.last_hidden_state[:, patch_start:, :].float().cpu()
                all_tokens.append(tokens)

        all_tokens = torch.cat(all_tokens, dim=0)  # (N, 196, D)

        # Pool per body region
        frame_embeddings = []
        for j in range(n_crops):
            tokens = all_tokens[j]  # (196, D)
            region_flat = mask_regions_14x14[j].flatten()  # (196,)

            parts = []
            for r in [1, 2, 3]:  # tip_a, center, tip_b
                mask_r = region_flat == r
                if mask_r.sum() > 0:
                    parts.append(tokens[mask_r].mean(dim=0))
                else:
                    # Fallback: use mean of all patches
                    parts.append(tokens.mean(dim=0))
            frame_embeddings.append(torch.cat(parts))  # (3*D,)

        embeddings[(video_id, bird_id, window)] = torch.stack(
            frame_embeddings
        )  # (F_w, 3*D)

    return embeddings
