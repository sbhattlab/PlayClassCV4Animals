"""
Extract VideoPrism embeddings from tracked objects.

Runs in the `videoprism` pixi env (JAX). Outputs are saved as PyTorch .pt
files for compatibility with the existing classification pipeline.

Usage::

    pixi run -e videoprism extract_videoprism \
        --video-dir data/video/batch data/video/batch2 --device 0

    # Save raw patch tokens for trainable pooler
    pixi run -e videoprism extract_videoprism \
        --video-dir data/video/batch data/video/batch2 --device 0 --raw
"""

import os
import sys
from argparse import ArgumentParser
from pathlib import Path

# Ensure conda CUDA libs are on LD_LIBRARY_PATH before JAX init
_env_lib = str(Path(sys.prefix) / "lib")
os.environ["LD_LIBRARY_PATH"] = _env_lib + ":" + os.environ.get("LD_LIBRARY_PATH", "")

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from loguru import logger
from PIL import Image
from tqdm import tqdm
from videoprism import models as vp

# Lazy torch import — only needed for saving .pt files
import torch

# Block TensorFlow from grabbing GPU
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")
tf.config.set_visible_devices([], "TPU")

from src._config import DEFAULT_DATASET_DIR, DEFAULT_TRACKING_DIR, DEFAULT_VIDEO_DIR
from src.dataset.crops import (
    CROP_MODES,
    compute_union_bbox,
    compute_union_origin,
    crop_frame,
)
from src.dataset.utils import (
    assert_embedding_label_alignment,
    resolve_video_path,
)
from src.io import load_video_frames_sequential


def parse_args():
    parser = ArgumentParser(
        description="Extract VideoPrism embeddings from tracked objects."
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--tracking-dir", type=Path, default=DEFAULT_TRACKING_DIR)
    parser.add_argument("--video-dir", type=Path, nargs="+", default=None, help=f"Directory(ies) with .mp4 files (default: all subdirs of {DEFAULT_VIDEO_DIR}/)")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--model-name",
        type=str,
        default="videoprism_public_v1_base",
    )
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--frame-size", type=int, default=288)
    parser.add_argument(
        "--crop-mode",
        type=str,
        choices=list(CROP_MODES) + ["union"],
        default="bbox",
        help="Crop strategy: 'bbox'=per-frame bbox crop (default), "
        "'plain256'=fixed 256x256 around bbox centroid, "
        "'union512'=fixed 512x512 around union centroid, "
        "'union'=union of all bboxes (resized to frame-size), "
        "'darken512'=union512 with darkened background (untested), "
        "'roi512'=union512 + ROI-pool bird patches (untested)",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Save raw patch tokens (T*S, D) per window instead of pooled (D,)",
    )
    parser.add_argument(
        "--temporal",
        action="store_true",
        help="Save per-timestep spatial-mean-pooled tokens (T, D) per window",
    )
    parser.add_argument("--output-name", type=str, default=None)
    return parser.parse_args()


def _build_output_name(args):
    parts = ["embeddings", "videoprism"]
    if "large" in args.model_name:
        parts.append("large")
    if args.raw:
        parts.append("raw")
    elif args.temporal:
        parts.append("temporal")
    if args.crop_mode != "bbox":
        parts.append(args.crop_mode)
    return "_".join(parts) + ".pt"


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


def main():
    args = parse_args()

    # Set JAX device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

    # Resolve video dirs
    if args.video_dir is None:
        root = Path(DEFAULT_VIDEO_DIR)
        args.video_dir = sorted(p for p in root.iterdir() if p.is_dir())
        logger.info(f"Auto-discovered {len(args.video_dir)} video dir(s) under {root}")

    tracks_path = args.dataset_dir / "tracks.parquet"
    if not tracks_path.exists():
        logger.error(f"tracks.parquet not found in {args.dataset_dir}")
        sys.exit(1)

    load_cols = ["video_id", "bird_id", "frame_idx", "window", "bbox"]
    tracks = pd.read_parquet(tracks_path, columns=load_cols)
    video_ids = sorted(tracks["video_id"].unique())
    logger.info(f"Loaded {len(tracks)} track rows across {len(video_ids)} video(s)")

    # Load model
    logger.info(f"Loading model: {args.model_name}")
    flax_model = vp.get_model(args.model_name)
    loaded_state = vp.load_pretrained_weights(args.model_name)

    @jax.jit
    def forward_fn(inputs):
        return flax_model.apply(loaded_state, inputs, train=False)

    # Warm up JIT
    logger.info("Warming up JIT...")
    dummy = jnp.zeros((1, args.num_frames, args.frame_size, args.frame_size, 3))
    _ = forward_fn(dummy)
    logger.info(f"Model ready on {jax.devices()[0]}")

    # Extract per video
    all_embeddings = {}
    for video_id in video_ids:
        logger.info(f"--- Processing video: {video_id} ---")
        video_path = resolve_video_path(video_id, args.tracking_dir, args.video_dir)
        if video_path is None:
            logger.error(f"Skipping {video_id}: video file not found")
            continue

        video_tracks = tracks[tracks["video_id"] == video_id]
        emb_dict = extract_videoprism_embeddings(
            video_tracks,
            video_path,
            forward_fn,
            num_frames=args.num_frames,
            frame_size=args.frame_size,
            crop_mode=args.crop_mode,
            raw=args.raw,
            temporal=args.temporal,
        )
        all_embeddings.update(emb_dict)
        logger.info(f"  {len(emb_dict)} window embeddings extracted")

    if not all_embeddings:
        raise ValueError("No embeddings extracted.")

    # Check alignment with labels
    assert_embedding_label_alignment(set(all_embeddings.keys()), args.dataset_dir)

    # Convert numpy arrays to torch tensors and save
    torch_embeddings = {
        k: torch.from_numpy(v.astype(np.float32)) for k, v in all_embeddings.items()
    }

    output_name = args.output_name or _build_output_name(args)
    output_path = args.dataset_dir / output_name
    torch.save(torch_embeddings, output_path)

    sample_key = next(iter(torch_embeddings))
    sample_shape = torch_embeddings[sample_key].shape
    logger.info(f"Saved {len(torch_embeddings)} embeddings to {output_path}")
    logger.info(f"Embedding shape per window: {sample_shape}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
