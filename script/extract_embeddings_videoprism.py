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
from src.dataset.crops import CROP_MODES
from src.dataset.embeddings.videoprism import extract_videoprism_embeddings
from src.dataset.utils import (
    assert_embedding_label_alignment,
    resolve_video_path,
)


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
    parser.add_argument("--num-frames", type=int, default=None,
                        help="Frames per clip (default: 16 for base, 8 for large)")
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
    from src.dataset.utils import parse_model_size

    parts = ["embeddings", "videoprism"]
    size = parse_model_size(args.model_name)
    if size:
        parts.append(size)
    if args.raw:
        parts.append("raw")
    elif args.temporal:
        parts.append("temporal")
    if args.crop_mode != "bbox":
        parts.append(args.crop_mode)
    return "_".join(parts) + ".pt"


def main():
    args = parse_args()

    # Auto-set num_frames based on model variant
    if args.num_frames is None:
        args.num_frames = 8 if "large" in args.model_name else 16
        logger.info(f"Auto-set num_frames={args.num_frames} for {args.model_name}")

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
