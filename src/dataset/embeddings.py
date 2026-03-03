"""DINOv3 embedding extraction from tracked objects."""

from pathlib import Path

import torch
from loguru import logger
from PIL import Image
from tqdm import tqdm

from src.utils import load_video_frames_range


def extract_embeddings(
    tracks,
    video_path: str | Path,
    model,
    processor,
    *,
    batch_size: int = 32,
) -> dict[tuple, torch.Tensor]:
    """Extract DINOv3 CLS-token embeddings for tracked objects, grouped by window.

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
    frames = load_video_frames_range(video_path, min_frame, max_frame + 1)
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

        for _, row in group_rows.iterrows():
            frame_idx = int(row["frame_idx"])

            bbox = row["bbox"]
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

            # Skip frames that the decoder failed to load
            local_idx = frame_idx - min_frame
            if local_idx >= len(frames):
                continue

            # Clamp bbox to frame dimensions (SAM3 bboxes can slightly overshoot),
            # skip if it collapses to zero area
            frame_np = frames[local_idx]
            h, w = frame_np.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            all_crops.append(Image.fromarray(frame_np[y1:y2, x1:x2]))

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
            all_embeddings.append(outputs.pooler_output.float().cpu())

    flat_embeddings = torch.cat(all_embeddings, dim=0)  # (n_crops, d_model)

    # Split back into per-(bird, window) tensors
    embeddings: dict[tuple, torch.Tensor] = {}
    offset = 0
    for key in groups:
        n = crops_per_group[key]
        if n > 0:
            embeddings[key] = flat_embeddings[offset : offset + n]
        else:
            embeddings[key] = torch.empty(0, d_model)
        offset += n

    return embeddings
