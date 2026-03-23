"""VideoPrism embedding extraction from tracked objects."""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
from loguru import logger
from PIL import Image
from tqdm import tqdm

from src.dataset.crops import compute_union_bbox, compute_union_origin, crop_frame
from src.io import load_video_frames_sequential


def extract_videoprism_embeddings(
    tracks,
    video_path,
    forward_fn,
    num_frames=16,
    frame_size=288,
    crop_mode="bbox",
    raw=False,
    temporal=False,
):
    """Extract VideoPrism embeddings for all windows in a video.

    Returns:
        dict[(video_id, bird_id, window), np.ndarray]
    """
    min_frame = int(tracks["frame_idx"].min())
    max_frame = int(tracks["frame_idx"].max())

    logger.info(
        f"Loading frames [{min_frame}, {max_frame}] from {Path(video_path).name}"
    )
    frames = load_video_frames_sequential(video_path, min_frame, max_frame + 1)
    logger.info(f"Loaded {len(frames)} frames")

    groups = sorted(tracks.groupby(["video_id", "bird_id", "window"]).groups.keys())
    logger.info(f"Extracting embeddings for {len(groups)} windows")

    embeddings = {}

    for video_id, bird_id, window in tqdm(groups, desc="Windows"):
        group_rows = tracks[
            (tracks["video_id"] == video_id)
            & (tracks["bird_id"] == bird_id)
            & (tracks["window"] == window)
        ].sort_values("frame_idx")

        # Pre-compute per-window metadata for union-based modes
        all_bboxes = group_rows["bbox"].tolist()
        union_origin = None
        union_bbox = None
        if crop_mode in ("union512", "darken512", "roi512"):
            first_local = int(group_rows.iloc[0]["frame_idx"]) - min_frame
            if first_local < len(frames):
                fh, fw = frames[first_local].shape[:2]
                union_origin = compute_union_origin(all_bboxes, fh, fw)
        elif crop_mode == "union":
            union_bbox = compute_union_bbox(all_bboxes)

        # Process all frames via non-overlapping clips of num_frames
        n = len(group_rows)
        n_spatial_side = 16
        n_spatial = n_spatial_side * n_spatial_side
        all_clip_tokens = []

        for clip_start in range(0, n, num_frames):
            clip_rows = group_rows.iloc[clip_start : clip_start + num_frames]

            crops = []
            bbox_patches = []

            for _, row in clip_rows.iterrows():
                frame_idx = int(row["frame_idx"])
                local_idx = frame_idx - min_frame
                if local_idx >= len(frames):
                    continue

                frame_np = frames[local_idx]

                if crop_mode == "union":
                    crop_np, _ = crop_frame(frame_np, union_bbox, "bbox")
                else:
                    crop_np, extra = crop_frame(
                        frame_np,
                        row["bbox"],
                        crop_mode,
                        union_origin=union_origin,
                    )
                    if extra is not None and "patch_bounds" in extra:
                        bbox_patches.append(extra["patch_bounds"])

                if crop_np is None:
                    continue

                if crop_np.shape[0] != frame_size or crop_np.shape[1] != frame_size:
                    crop_np = np.array(
                        Image.fromarray(crop_np).resize(
                            (frame_size, frame_size),
                            Image.Resampling.BILINEAR,
                        )
                    )
                crops.append(crop_np)

            if not crops:
                continue

            n_valid = len(crops)
            while len(crops) < num_frames:
                crops.append(crops[-1])
                if crop_mode == "roi512" and bbox_patches:
                    bbox_patches.append(bbox_patches[-1])

            # Stack to (1, T, H, W, 3) float32 [0, 1]
            clip = np.stack(crops[:num_frames]).astype(np.float32) / 255.0
            clip = jnp.asarray(clip[None, ...])

            tokens, _ = forward_fn(clip)
            tokens = np.asarray(tokens[0])  # (T*S, D)

            # Trim to valid timesteps (no temporal compression in VideoPrism)
            reshaped = tokens.reshape(num_frames, n_spatial, -1)
            valid = reshaped[:n_valid]  # (n_valid, S, D)

            if raw:
                all_clip_tokens.append(valid.reshape(-1, tokens.shape[-1]))
            elif crop_mode == "roi512":
                valid_grid = valid.reshape(n_valid, n_spatial_side, n_spatial_side, -1)
                roi_tokens = []
                for t in range(min(n_valid, len(bbox_patches))):
                    py1, py2, px1, px2 = bbox_patches[t]
                    bird = valid_grid[t, py1:py2, px1:px2, :]
                    if bird.size > 0:
                        roi_tokens.append(bird.reshape(-1, bird.shape[-1]).mean(axis=0))
                    else:
                        roi_tokens.append(valid[t].mean(axis=0))
                all_clip_tokens.append(np.stack(roi_tokens))
            else:
                all_clip_tokens.append(valid.mean(axis=1))  # (n_valid, D)

        if not all_clip_tokens:
            continue

        combined = np.concatenate(all_clip_tokens, axis=0)
        if raw or temporal:
            embedding = combined
        else:
            embedding = combined.mean(axis=0)

        embeddings[(video_id, bird_id, window)] = embedding

    del frames
    return embeddings
