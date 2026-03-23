"""V-JEPA 2/2.1 embedding extraction from tracked objects."""

import warnings
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from PIL import Image
from tqdm import tqdm

from src.dataset.crops import compute_union_origin, crop_frame
from src.io import load_video_frames_torchcodec as load_video_frames

_HUB_MODEL_NAMES = {
    "vjepa2_1_vit_base_384",
    "vjepa2_1_vit_large_384",
    "vjepa2_1_vit_giant",
}


def is_hub_model(model_name: str) -> bool:
    return model_name in _HUB_MODEL_NAMES


class VJEPA21Wrapper:
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
        _prefix = next(
            (p for p in ("union", "darken", "roi") if crop_mode.startswith(p)), None
        )
        if _prefix is not None:
            _crop_sz = int(crop_mode.removeprefix(_prefix))
            first_local = int(group_rows.iloc[0]["frame_idx"]) - min_frame
            if first_local < len(frames):
                fh, fw = frames[first_local].shape[:2]
                all_bboxes = group_rows["bbox"].tolist()
                union_origin = compute_union_origin(
                    all_bboxes, fh, fw, crop_size=_crop_sz
                )

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
                # Resize bbox crops to 256x256 (plain256/union512 already fixed-size)
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
