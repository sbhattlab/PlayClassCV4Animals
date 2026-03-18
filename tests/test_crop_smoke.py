"""Visual smoke test for crop modes and bodypart splitting.

Picks one window per behaviour class, saves side-by-side crops for all
crop modes + PCA body-part split visualization. Outputs to img/crop_smoke_test/.

Usage::

    pixi run -e sam3-hf python -m src.test.crop_smoke_test \
        --video-dir data/video/batch data/video/batch2
"""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import pycocotools.mask as mask_util
from loguru import logger
from PIL import Image, ImageDraw

from src._config import DEFAULT_DATASET_DIR, DEFAULT_TRACKING_DIR, LABEL_ORDER
from src.dataset.crops import CROP_MODES, compute_union_origin, crop_frame
from src.dataset.embeddings import _split_mask_thirds
from src.dataset.utils import resolve_video_path
from src.io import load_video_frames_torchcodec as load_video_frames


OUTPUT_DIR = Path("img/crop_smoke_test")
N_FRAMES = 4  # frames per window to visualize


def parse_args():
    parser = ArgumentParser(description="Visual smoke test for crop modes.")
    parser.add_argument(
        "--video-dir",
        type=Path,
        nargs="+",
        required=True,
        help="Directory(ies) with .mp4 files",
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
        "--exclude",
        type=str,
        default="social",
        help="Exclude a behaviour class (default: social)",
    )
    return parser.parse_args()


def draw_bbox_on_crop(crop_np, bbox, crop_origin):
    """Draw a red bbox rectangle on a crop, given the crop's origin in frame coords."""
    img = Image.fromarray(crop_np.copy())
    draw = ImageDraw.Draw(img)
    ox, oy = crop_origin
    x1, y1, x2, y2 = bbox
    draw.rectangle(
        [x1 - ox, y1 - oy, x2 - ox, y2 - oy],
        outline=(255, 0, 0),
        width=2,
    )
    return np.array(img)


def visualize_bodypart_split(frame_np, bbox, rle_mask):
    """Crop the bbox region and overlay PCA body-part split as colored regions."""
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    h, w = frame_np.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame_np[y1:y2, x1:x2].copy()

    # Decode mask and split
    full_mask = mask_util.decode(rle_mask)
    mask_crop = full_mask[y1:y2, x1:x2]
    regions = _split_mask_thirds(mask_crop)
    if regions is None:
        return None

    # Overlay colors: tip_a=red, center=green, tip_b=blue
    colors = {1: (255, 80, 80), 2: (80, 255, 80), 3: (80, 80, 255)}
    overlay = crop.copy()
    for region_id, color in colors.items():
        mask_r = regions == region_id
        overlay[mask_r] = (
            0.5 * crop[mask_r].astype(float) + 0.5 * np.array(color)
        ).astype(np.uint8)

    return overlay


def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tracks = pd.read_parquet(
        args.dataset_dir / "tracks.parquet",
        columns=[
            "video_id",
            "bird_id",
            "frame_idx",
            "window",
            "bbox",
            "counts",
            "size",
        ],
    )
    labels = pd.read_parquet(args.dataset_dir / "labels.parquet")

    # Filter excluded class
    label_order = [l for l in LABEL_ORDER if l != args.exclude]
    labels = labels[labels["behav_label"].isin(label_order)]

    logger.info(f"Crop modes: {CROP_MODES}")
    logger.info(f"Behaviours: {label_order}")

    for behav in label_order:
        behav_labels = labels[labels["behav_label"] == behav]
        if behav_labels.empty:
            logger.warning(f"No samples for {behav}, skipping")
            continue

        # Pick a window
        row = behav_labels.iloc[len(behav_labels) // 2]
        vid, bid, win = row["video_id"], row["bird_id"], row["window"]
        logger.info(f"--- {behav}: {vid} bird={bid} window={win} ---")

        # Resolve video
        video_path = resolve_video_path(vid, args.tracking_dir, args.video_dir)
        if video_path is None:
            logger.error(f"Video not found for {vid}, skipping")
            continue

        # Get window tracks
        window_tracks = tracks[
            (tracks["video_id"] == vid)
            & (tracks["bird_id"] == bid)
            & (tracks["window"] == win)
        ].sort_values("frame_idx")

        if window_tracks.empty:
            continue

        # Load frames
        min_frame = int(window_tracks["frame_idx"].min())
        max_frame = int(window_tracks["frame_idx"].max())
        frames = load_video_frames(video_path, min_frame, max_frame + 1)

        # Sample N_FRAMES evenly
        indices = np.linspace(0, len(window_tracks) - 1, N_FRAMES, dtype=int)
        sampled = window_tracks.iloc[indices]

        # Pre-compute union origin
        all_bboxes = window_tracks["bbox"].tolist()
        fh, fw = frames[0].shape[:2]
        union_origin = compute_union_origin(all_bboxes, fh, fw)

        # --- Crop mode comparison ---
        for fi, (_, trow) in enumerate(sampled.iterrows()):
            frame_idx = int(trow["frame_idx"])
            local_idx = frame_idx - min_frame
            if local_idx >= len(frames):
                continue
            frame_np = frames[local_idx]
            bbox = trow["bbox"]
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

            for mode in CROP_MODES:
                origin = (
                    union_origin
                    if mode in ("union512", "darken512", "roi512")
                    else None
                )
                crop_np, _ = crop_frame(
                    frame_np,
                    bbox,
                    mode,
                    union_origin=origin,
                )
                if crop_np is None:
                    continue

                # Annotate with bbox rectangle for context crops
                if mode in ("plain256", "union512", "darken512", "roi512"):
                    if mode == "plain256":
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        ox = max(0, min(cx - 128, fw - 256))
                        oy = max(0, min(cy - 128, fh - 256))
                    else:
                        ox, oy = union_origin
                    crop_np = draw_bbox_on_crop(crop_np, (x1, y1, x2, y2), (ox, oy))

                out_path = OUTPUT_DIR / f"{behav}_{mode}_f{fi}.png"
                Image.fromarray(crop_np).save(out_path)

            logger.info(
                f"  frame {fi} (idx={frame_idx}): saved {len(CROP_MODES)} crop modes"
            )

        # --- Bodypart split visualization (middle frame) ---
        mid_row = window_tracks.iloc[len(window_tracks) // 2]
        mid_local = int(mid_row["frame_idx"]) - min_frame
        if mid_local < len(frames):
            counts = mid_row["counts"]
            if isinstance(counts, str):
                counts = counts.encode("utf-8")
            rle = {"counts": counts, "size": mid_row["size"]}
            overlay = visualize_bodypart_split(frames[mid_local], mid_row["bbox"], rle)
            if overlay is not None:
                out_path = OUTPUT_DIR / f"{behav}_bodypart_split.png"
                Image.fromarray(overlay).save(out_path)
                logger.info(f"  bodypart split saved")

        del frames

    logger.info(f"All outputs saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
