"""
Extract DINOv3 CLS-token embeddings from tracked objects.

Separate from ``build_dataset.py`` because it requires GPU + model loading
(~300 MB), while the rest of the dataset pipeline is CPU-only.

Usage::

    pixi run -e sam3-hf extract-embeddings \
        --video-dir data/video/batch data/video/batch2
"""

import sys
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import torch
from loguru import logger
from transformers import AutoImageProcessor, AutoModel

from src._config import DEFAULT_DATASET_DIR, DEFAULT_TRACKING_DIR
from src.dataset.embeddings import extract_bodypart_embeddings, extract_embeddings
from src.dataset.utils import assert_embedding_label_alignment, resolve_video_path
from src.memory import free_gpu_memory


def parse_args():
    parser = ArgumentParser(
        description="Extract DINOv3 embeddings from tracked objects."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Directory containing tracks.parquet and labels.parquet (default: %(default)s)",
    )
    parser.add_argument(
        "--tracking-dir",
        type=Path,
        default=DEFAULT_TRACKING_DIR,
        help="Root tracking dir for resolving video subdir names (default: %(default)s)",
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        nargs="+",
        required=True,
        help="Directory(ies) with .mp4 files matching tracking subdir names",
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
        type=str,
        default="cuda:0",
        help="Device (e.g. cuda:0, cuda:1, cpu)",
    )
    parser.add_argument(
        "--bbox-scale",
        type=float,
        default=1.0,
        help="Scale factor for bbox crops (1.0 = tight, 1.25 = 25%% padding, default: 1.0)",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="Override processor resize resolution (default: use processor config, typically 224). "
        "DINOv3 was trained at 256, adapted up to 768. Local crops used 112-336.",
    )
    parser.add_argument(
        "--lora-weights",
        type=Path,
        default=None,
        help="Path to LoRA adapter directory (from finetune_dino.py)",
    )
    parser.add_argument(
        "--crop-mode",
        type=str,
        choices=["bbox", "plain256", "union512", "darken512", "roi512"],
        default="bbox",
        help="Crop strategy: 'bbox'=per-frame bbox crop (default), "
        "'plain256'=fixed 256x256 around bbox centroid per frame, "
        "'union512'=fixed 512x512 around union centroid (only contained frames), "
        "'darken512'=union512 with darkened background (untested), "
        "'roi512'=union512 + ROI patch indices (untested)",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Save raw patch tokens (F_w, P, D) instead of CLS tokens (F_w, D)",
    )
    parser.add_argument(
        "--n-sample-frames",
        type=int,
        default=None,
        help="Subsample this many frames per window for raw mode (default: all frames). "
        "E.g. --n-sample-frames 12 gives (12, 196, 1024) per window.",
    )
    parser.add_argument(
        "--bodyparts",
        action="store_true",
        help="Extract body-part-pooled embeddings (tip_a + center + tip_b) using mask geometry. "
        "Output: (F_w, 3*D) per window.",
    )
    return parser.parse_args()


def _build_output_name(args) -> str:
    """Build output filename reflecting active transforms, e.g. embeddings_vitb_125_blur.pt."""
    parts = ["embeddings"]
    # Encode model variant (skip default vitl)
    model_lower = args.model_name.lower()
    if "vitb" in model_lower:
        parts.append("vitb")
    elif "vits" in model_lower:
        parts.append("vits")
    if args.resolution is not None:
        parts.append(f"r{args.resolution}")
    if args.bbox_scale != 1.0:
        parts.append(str(int(args.bbox_scale * 100)))
    if args.lora_weights is not None:
        parts.append("lora")
    if args.crop_mode != "bbox":
        parts.append(args.crop_mode)
    if args.raw:
        parts.append("raw")
    if args.bodyparts:
        parts.append("bodyparts")
    return "_".join(parts) + ".pt"


def main():
    args = parse_args()

    tracks_path = args.dataset_dir / "tracks.parquet"
    if not tracks_path.exists():
        logger.error(f"tracks.parquet not found in {args.dataset_dir}")
        logger.error("Run build_dataset first to generate it.")
        sys.exit(1)

    load_cols = ["video_id", "bird_id", "frame_idx", "window", "bbox"]
    if args.bodyparts:
        load_cols.extend(["counts", "size"])
    tracks = pd.read_parquet(tracks_path, columns=load_cols)
    video_ids = sorted(tracks["video_id"].unique())
    logger.info(f"Loaded {len(tracks)} track rows across {len(video_ids)} video(s)")

    # Load model + processor once
    logger.info(f"Loading model: {args.model_name}")
    proc_kwargs = {}
    if args.resolution is not None:
        proc_kwargs["size"] = {"height": args.resolution, "width": args.resolution}
        logger.info(
            f"Overriding processor resolution to {args.resolution}×{args.resolution}"
        )
    processor = AutoImageProcessor.from_pretrained(args.model_name, **proc_kwargs)
    model = AutoModel.from_pretrained(args.model_name, dtype=torch.bfloat16)
    if args.lora_weights is not None:
        from peft import PeftModel

        logger.info(f"Loading LoRA adapter from {args.lora_weights}")
        model = PeftModel.from_pretrained(model, str(args.lora_weights))
        model = (
            model.merge_and_unload()
        )  # merge LoRA into base weights for fast inference
        logger.info("LoRA adapter merged into base model")
    device = torch.device(args.device)
    model = model.to(device).eval()
    logger.info(f"Model loaded on {device} (bfloat16)")

    # Check that window column exists
    if "window" not in tracks.columns:
        raise ValueError(
            (
                "tracks.parquet missing 'window' column. "
                "Run build_dataset first to assign windows."
            )
        )

    # Loop per video so only one video's frames are in memory at a time
    all_embeddings: dict[tuple, torch.Tensor] = {}

    for video_id in video_ids:
        logger.info(f"--- Processing video: {video_id} ---")

        video_path = resolve_video_path(video_id, args.tracking_dir, args.video_dir)
        if video_path is None:
            logger.error(f"Skipping {video_id}: video file not found")
            continue

        video_tracks = tracks[tracks["video_id"] == video_id]
        logger.info(f"  {len(video_tracks)} track rows, video: {video_path.name}")

        if args.bodyparts:
            emb_dict = extract_bodypart_embeddings(
                video_tracks,
                video_path,
                model,
                processor,
                batch_size=args.batch_size,
                bbox_scale=args.bbox_scale,
                n_sample_frames=args.n_sample_frames,
            )
        else:
            emb_dict = extract_embeddings(
                video_tracks,
                video_path,
                model,
                processor,
                batch_size=args.batch_size,
                bbox_scale=args.bbox_scale,
                raw=args.raw,
                n_sample_frames=args.n_sample_frames,
                crop_mode=args.crop_mode,
            )

        all_embeddings.update(emb_dict)

        n_total = sum(e.shape[0] for e in emb_dict.values())
        logger.info(f"  Extracted {n_total} embeddings for {len(emb_dict)} groups")

    if not all_embeddings:
        raise ValueError("No embeddings extracted for any video.")

    # Check alignment with labels
    assert_embedding_label_alignment(set(all_embeddings.keys()), args.dataset_dir)

    # Save embeddings keyed by (video_id, bird_id, window)
    output_name = _build_output_name(args)
    raw_path = args.dataset_dir / output_name
    torch.save(all_embeddings, raw_path)
    logger.info(f"Saved {len(all_embeddings)} embedding tensors: {raw_path}")

    # Cleanup
    del model, processor
    free_gpu_memory()
    logger.info("Done.")


if __name__ == "__main__":
    main()
