"""
Extract DINOv3 CLS-token embeddings from tracked objects.

Separate from ``build_dataset.py`` because it requires GPU + model loading
(~300 MB), while the rest of the dataset pipeline is CPU-only.

Usage::

    pixi run -e sam3-hf extract-embeddings \
        --tracking-dir data/tracking/20260225_214929_sam3_hf \
        --video-dir video-data/batch
"""

import sys
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import torch
from loguru import logger
from transformers import AutoImageProcessor, AutoModel

from src.dataset.embeddings import extract_embeddings
from src.utils import free_gpu_memory


def parse_args():
    parser = ArgumentParser(
        description="Extract DINOv3 embeddings from tracked objects."
    )
    parser.add_argument(
        "--tracking-dir",
        type=Path,
        required=True,
        help="Batch tracking dir containing dataset_tracks.parquet",
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        required=True,
        help="Directory with .mp4 files matching tracking subdir names",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Crops per forward pass (default: 32)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="facebook/dinov3-vitl16-pretrain-lvd1689m",
        help='HuggingFace model ID (default: "facebook/dinov3-vitl16-pretrain-lvd1689m")',
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="CUDA device index (default: 0)",
    )
    return parser.parse_args()


def _resolve_video_path(
    video_id: str, tracking_dir: Path, video_dir: Path
) -> Path | None:
    """Find the video file for a given video_id.

    Tracking subdirs have full stems (e.g. ``C1G1_Test_1_day_28_...``).
    ``video_id`` is the short prefix (e.g. ``C1G1``). We find the matching
    subdir and look for ``{subdir_name}.mp4`` in ``video_dir``.
    """
    for subdir in sorted(tracking_dir.iterdir()):
        if subdir.is_dir() and subdir.name.startswith(video_id):
            video_path = video_dir / f"{subdir.name}.mp4"
            if video_path.exists():
                return video_path
            logger.warning(f"Expected video not found: {video_path}")
            return None
    logger.warning(f"No tracking subdir found starting with '{video_id}'")
    return None


def main():
    args = parse_args()

    tracks_path = args.tracking_dir / "dataset_tracks.parquet"
    if not tracks_path.exists():
        logger.error(f"dataset_tracks.parquet not found in {args.tracking_dir}")
        logger.error("Run build_dataset first to generate it.")
        sys.exit(1)

    # Only load columns needed for cropping (skip RLE masks to save time)
    tracks = pd.read_parquet(
        tracks_path,
        columns=["video_id", "bird_id", "frame_idx", "window", "bbox"],
    )
    video_ids = sorted(tracks["video_id"].unique())
    logger.info(f"Loaded {len(tracks)} track rows across {len(video_ids)} video(s)")

    # Load model + processor once
    logger.info(f"Loading model: {args.model_name}")
    processor = AutoImageProcessor.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name, dtype=torch.bfloat16)
    device = torch.device(f"cuda:{args.device}")
    model = model.to(device).eval()
    logger.info(f"Model loaded on {device} (bfloat16)")

    # Check that window column exists
    if "window" not in tracks.columns:
        raise ValueError(
            (
                "dataset_tracks.parquet missing 'window' column."
                "Run build_dataset first to assign windows."
            )
        )

    # Loop per video so only one video's frames are in memory at a time
    all_embeddings: dict[tuple, torch.Tensor] = {}

    for video_id in video_ids:
        logger.info(f"--- Processing video: {video_id} ---")

        video_path = _resolve_video_path(video_id, args.tracking_dir, args.video_dir)
        if video_path is None:
            logger.error(f"Skipping {video_id}: video file not found")
            continue

        video_tracks = tracks[tracks["video_id"] == video_id]
        logger.info(f"  {len(video_tracks)} track rows, video: {video_path.name}")

        # Structure
        # {(video_id, bird_id, window): Tensor(F_w, D)}
        emb_dict = extract_embeddings(
            video_tracks,
            video_path,
            model,
            processor,
            batch_size=args.batch_size,
        )

        all_embeddings.update(emb_dict)

        n_total = sum(e.shape[0] for e in emb_dict.values())
        logger.info(f"  Extracted {n_total} embeddings for {len(emb_dict)} groups")

    if not all_embeddings:
        raise ValueError("No embeddings extracted for any video.")

    # Check alignment with labels
    labels = pd.read_parquet(args.tracking_dir / "dataset_labels.parquet")
    _key_cols = ["video_id", "bird_id", "window"]
    embedding_keys = set(all_embeddings.keys())
    label_keys = set(labels[_key_cols].itertuples(index=False, name=None))
    assert embedding_keys == label_keys, (
        f"Window key mismatch with labels: {len(embedding_keys - label_keys)} in embeddings only, "
        f"{len(label_keys - embedding_keys)} in labels only"
    )

    # Save embeddings keyed by (video_id, bird_id, window)
    raw_path = args.tracking_dir / "dataset_embeddings.pt"
    torch.save(all_embeddings, raw_path)
    logger.info(f"Saved {len(all_embeddings)} embedding tensors: {raw_path}")

    # Cleanup
    del model, processor
    free_gpu_memory()
    logger.info("Done.")


if __name__ == "__main__":
    main()
