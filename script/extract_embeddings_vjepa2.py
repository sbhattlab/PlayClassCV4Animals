"""
Extract video-model embeddings (V-JEPA 2 / 2.1) from tracked objects.

Unlike DINOv3 extraction (per-frame CLS tokens), this feeds K frames per
window as a single video clip through a spatiotemporal backbone and
mean-pools the output tokens into one embedding per window.

Usage::

    # V-JEPA 2 (HF)
    pixi run -e tracker python -m script.extract_embeddings_vjepa2 \
        --video-dir data/video/batch data/video/batch2 --device cuda:0

    # V-JEPA 2.1 (torch.hub — requires one-time setup, see below)
    pixi run -e tracker python -m script.extract_embeddings_vjepa2 \
        --video-dir data/video/batch data/video/batch2 --device cuda:0 \
        --model-name vjepa2_1_vit_large_384

V-JEPA 2.1 setup (one-time)::

    bash script/setup_vjepa21.sh              # default: vjepa2_1_vit_large_384
    bash script/setup_vjepa21.sh <model>      # other variants

This downloads the checkpoint and patches the torch.hub cache to rename
``src/`` to ``vjepa2/`` (avoids collision with this project's ``src/``).
"""

import sys
import warnings
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoVideoProcessor

from src._config import DEFAULT_DATASET_DIR, DEFAULT_TRACKING_DIR, DEFAULT_VIDEO_DIR
from src.dataset.crops import CROP_MODES, compute_union_origin, crop_frame
from src.dataset.utils import assert_embedding_label_alignment, resolve_video_path
from src.io import load_video_frames_torchcodec as load_video_frames
from src.memory import free_gpu_memory


def parse_args():
    parser = ArgumentParser(
        description="Extract video-model embeddings from tracked objects."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
    )
    parser.add_argument(
        "--tracking-dir",
        type=Path,
        default=DEFAULT_TRACKING_DIR,
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        nargs="+",
        default=None,
        help=f"Directory(ies) with .mp4 files (default: all subdirs of {DEFAULT_VIDEO_DIR}/)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="facebook/vjepa2-vitl-fpc64-256",
        help="HF model name, or torch.hub entry for V-JEPA 2.1 "
        "(e.g. 'vjepa2_1_vit_large_384')",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device (e.g. cuda:0, cuda:1, cpu)"
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=64,
        help="Frames per clip (must match model's pretrained frame count, default: 64)",
    )
    parser.add_argument(
        "--crop-mode",
        type=str,
        choices=list(CROP_MODES),
        default="bbox",
        help="Crop strategy: 'bbox'=per-frame bbox crop (default), "
        "'plain256'=fixed 256x256 around bbox centroid, "
        "'plain384'=fixed 384x384 around bbox centroid, "
        "'union512'=fixed 512x512 around union centroid, "
        "'darken512'=union512 with darkened background (untested), "
        "'roi512'=union512 + ROI patch indices (untested)",
    )
    parser.add_argument(
        "--temporal",
        action="store_true",
        help="Output per-timestep spatial-mean-pooled tokens (T, D) instead of a single (D,) vector",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Save raw patch tokens (T*S, D) per window instead of pooled",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Output filename (default: auto-generated from model name)",
    )
    return parser.parse_args()


def _build_output_name(args):
    """Auto-generate output filename from model config."""
    model_lower = args.model_name.lower()
    if "vjepa2_1" in model_lower or "vjepa2.1" in model_lower:
        parts = ["embeddings", "vjepa21"]
        if "vitl" in model_lower or "large" in model_lower:
            parts.append("vitl")
        elif "vitg" in model_lower or "giant" in model_lower:
            parts.append("vitg")
    elif "vjepa2" in model_lower:
        parts = ["embeddings", "vjepa2"]
        if "vitl" in model_lower:
            parts.append("vitl")
        elif "vitg" in model_lower:
            parts.append("vitg")
        elif "vith" in model_lower:
            parts.append("vith")
    else:
        parts = ["embeddings", "video"]
    if args.num_frames != 64:
        parts.append(f"f{args.num_frames}")
    if args.crop_mode != "bbox":
        parts.append(args.crop_mode)
    if args.raw:
        parts.append("raw")
    elif args.temporal:
        parts.append("temporal")
    return "_".join(parts) + ".pt"



# ---- V-JEPA 2.1 torch.hub support ----------------------------------------

_HUB_MODEL_NAMES = {
    "vjepa2_1_vit_large_384",
    "vjepa2_1_vit_giant",
}


def _is_hub_model(model_name: str) -> bool:
    return model_name in _HUB_MODEL_NAMES


class _VJEPA21Wrapper:
    """Wrap the torch.hub V-JEPA 2.1 encoder to match the HF API."""

    def __init__(self, encoder, device):
        self.encoder = encoder.to(device).eval()
        self.device = device

    def get_vision_features(self, pixel_values_videos, **kwargs):
        # HF processor outputs (B, T, C, H, W), hub encoder expects (B, C, T, H, W)
        x = pixel_values_videos.permute(0, 2, 1, 3, 4)
        with warnings.catch_warnings(), torch.autocast("cuda", dtype=torch.bfloat16):
            warnings.filterwarnings("ignore", message=".*sdp_kernel.*")
            return self.encoder(x)


# ---------------------------------------------------------------------------


def extract_video_embeddings(
    tracks,
    video_path,
    model,
    processor,
    device,
    num_frames=64,
    crop_mode="bbox",
    temporal=False,
    raw=False,
):
    """Extract embeddings per window using a video backbone.

    Returns:
        dict[(video_id, bird_id, window), Tensor(D,) or Tensor(T, D)]
    """
    min_frame = int(tracks["frame_idx"].min())
    max_frame = int(tracks["frame_idx"].max())

    logger.info(
        f"Loading frames [{min_frame}, {max_frame}] from {Path(video_path).name}"
    )
    frames = load_video_frames(video_path, min_frame, max_frame + 1)
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

        # Pre-compute union origin for union-based crop modes
        union_origin = None
        if crop_mode in ("union512", "darken512", "roi512"):
            first_local = int(group_rows.iloc[0]["frame_idx"]) - min_frame
            if first_local < len(frames):
                fh, fw = frames[first_local].shape[:2]
                all_bboxes = group_rows["bbox"].tolist()
                union_origin = compute_union_origin(all_bboxes, fh, fw)

        # Process all frames via non-overlapping clips of num_frames
        n = len(group_rows)
        all_clip_tokens = []

        for clip_start in range(0, n, num_frames):
            clip_rows = group_rows.iloc[clip_start : clip_start + num_frames]

            crops = []
            for _, row in clip_rows.iterrows():
                frame_idx = int(row["frame_idx"])
                local_idx = frame_idx - min_frame
                if local_idx >= len(frames):
                    continue
                crop_np, _ = crop_frame(
                    frames[local_idx],
                    row["bbox"],
                    crop_mode,
                    union_origin=union_origin,
                )
                if crop_np is None:
                    continue
                # Resize bbox crops to 256×256 (plain256/union512 already fixed-size)
                if crop_mode == "bbox":
                    crop_np = np.array(
                        Image.fromarray(crop_np).resize(
                            (256, 256),
                            Image.Resampling.BILINEAR,
                        )
                    )
                crops.append(crop_np)

            if not crops:
                continue

            n_valid = len(crops)
            while len(crops) < num_frames:
                crops.append(crops[-1])

            # Stack to (T, C, H, W) tensor and pass to processor
            clip = torch.from_numpy(np.stack(crops[:num_frames])).permute(0, 3, 1, 2)
            inputs = processor(clip, return_tensors="pt").to(device)

            with torch.inference_mode():
                tokens = model.get_vision_features(**inputs).squeeze(0)  # (T*S, D)

            # Trim padded timesteps (tubelet_size=2 halves temporal dim)
            n_temporal_total = num_frames // 2
            n_temporal_valid = max(1, (n_valid + 1) // 2)
            n_spatial = tokens.shape[0] // n_temporal_total
            reshaped = tokens.reshape(n_temporal_total, n_spatial, -1)
            valid = reshaped[:n_temporal_valid]  # (T_valid, S, D)

            if raw:
                all_clip_tokens.append(valid.reshape(-1, tokens.shape[-1]))
            else:
                all_clip_tokens.append(valid.mean(dim=1))  # (T_valid, D)

        if not all_clip_tokens:
            continue

        combined = torch.cat(all_clip_tokens, dim=0).float().cpu()
        if raw or temporal:
            embedding = combined
        else:
            embedding = combined.mean(dim=0)

        embeddings[(video_id, bird_id, window)] = embedding

    del frames
    return embeddings


def main():
    args = parse_args()

    tracks_path = args.dataset_dir / "tracks.parquet"
    if not tracks_path.exists():
        logger.error(f"tracks.parquet not found in {args.dataset_dir}")
        sys.exit(1)

    # Resolve video dirs
    if args.video_dir is None:
        root = Path(DEFAULT_VIDEO_DIR)
        args.video_dir = sorted(p for p in root.iterdir() if p.is_dir())
        logger.info(f"Auto-discovered {len(args.video_dir)} video dir(s) under {root}")

    load_cols = ["video_id", "bird_id", "frame_idx", "window", "bbox"]
    tracks = pd.read_parquet(tracks_path, columns=load_cols)
    video_ids = sorted(tracks["video_id"].unique())
    logger.info(f"Loaded {len(tracks)} track rows across {len(video_ids)} video(s)")

    # Load model + processor
    device = torch.device(args.device)
    logger.info(f"Loading model: {args.model_name}")

    if _is_hub_model(args.model_name):
        # V-JEPA 2.1 via torch.hub (no HF checkpoint yet).
        # Temporarily evict our src package from sys.modules so torch.hub
        # can import the hub repo's own src/ without collision.
        processor = AutoVideoProcessor.from_pretrained("facebook/vjepa2-vitl-fpc64-256")
        encoder, _predictor = torch.hub.load("facebookresearch/vjepa2", args.model_name)
        del _predictor
        model = _VJEPA21Wrapper(encoder, device)
        d_model = 1024  # ViT-L hidden size
    else:
        processor = AutoVideoProcessor.from_pretrained(args.model_name)
        model = AutoModel.from_pretrained(args.model_name, dtype=torch.bfloat16)
        model = model.to(device).eval()
        d_model = model.config.hidden_size

    logger.info(f"Model on {device}, hidden_size={d_model}")

    # Extract per video
    # For raw mode, save per-video to avoid OOM (8192 x 1024 x ~500 windows = ~4 GB/video)
    output_name = args.output_name or _build_output_name(args)
    output_path = (
        Path(output_name) if "/" in str(output_name) else args.dataset_dir / output_name
    )
    save_incremental = args.raw

    all_embeddings = {}
    for video_id in video_ids:
        logger.info(f"--- Processing video: {video_id} ---")
        video_path = resolve_video_path(video_id, args.tracking_dir, args.video_dir)
        if video_path is None:
            logger.error(f"Skipping {video_id}: video file not found")
            continue

        video_tracks = tracks[tracks["video_id"] == video_id]
        emb_dict = extract_video_embeddings(
            video_tracks,
            video_path,
            model,
            processor,
            device,
            num_frames=args.num_frames,
            crop_mode=args.crop_mode,
            temporal=args.temporal,
            raw=args.raw,
        )
        all_embeddings.update(emb_dict)
        logger.info(f"  {len(emb_dict)} window embeddings extracted")

        if save_incremental:
            # Save progress after each video to avoid OOM
            torch.save(all_embeddings, output_path)
            logger.info(f"  Saved {len(all_embeddings)} total embeddings (incremental)")

    if not all_embeddings:
        raise ValueError("No embeddings extracted.")

    # Check alignment with labels
    assert_embedding_label_alignment(set(all_embeddings.keys()), args.dataset_dir)

    # Save
    output_name = args.output_name or _build_output_name(args)
    output_path = (
        Path(output_name) if "/" in str(output_name) else args.dataset_dir / output_name
    )
    torch.save(all_embeddings, output_path)
    logger.info(f"Saved {len(all_embeddings)} embeddings to {output_path}")

    del model, processor
    free_gpu_memory()
    logger.info("Done.")


if __name__ == "__main__":
    main()
