import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pycocotools.mask as mask_util
import torch
from loguru import logger

# Try to import supervision, fall back gracefully if not available
try:
    import supervision as sv

    SUPERVISION_AVAILABLE = True
except ImportError:
    SUPERVISION_AVAILABLE = False


def setup_logger(log_dir: Path, debug: bool = False) -> Path:
    """
    Configure loguru logger with both console and file output.

    Args:
        log_dir: Directory to store log files
        debug: Enable debug level logging

    Returns:
        Path to the log file
    """
    level = "DEBUG" if debug else "INFO"

    # Create log directory
    log_dir.mkdir(parents=True, exist_ok=True)

    # Generate unique log filename with datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = log_dir / f"sam3_hf_chunking_{timestamp}.log"

    # Remove default logger
    logger.remove()

    # Add console handler with colored output
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> "
        "[<level>{level}</level>] {message}",
        level=level,
    )

    # Add file handler
    logger.add(
        str(log_filename),
        format="{time:YYYY-MM-DD HH:mm:ss} [{level}] {message}",
        level=level,
        enqueue=True,
        backtrace=True,
        diagnose=True,
    )

    return log_filename


def save_results_json(
    all_results: dict[int, dict],
    metadata: dict,
    chunk_metadata: dict[int, dict],
    video_frames: list,
    output_path: Path,
    vis_output_dir: Path,
    overlay_func: callable,
    vis_stride: int = 25,
) -> None:
    """
    Save segmentation results to JSON and optionally generate frame visualizations.

    Args:
        all_results: Results dictionary {chunk_idx -> {frame_idx -> results}}
        metadata: Dictionary of metadata to include in the output
        chunk_metadata: Metadata for each chunk {chunk_idx -> metadata_dict}
        video_frames: List of video frames (PIL Images or numpy arrays)
        output_path: Path to save JSON results
        vis_output_dir: Directory to save visualizations
        overlay_func: Function to overlay masks on frames (frame, masks) -> PIL Image
        vis_stride: Save every Nth frame for visualization
    """
    # Convert keys to strings for JSON serialization
    json_results = {
        str(chunk_idx): {
            str(frame_idx): frame_results
            for frame_idx, frame_results in chunk_data.items()
        }
        for chunk_idx, chunk_data in all_results.items()
    }

    # Build output structure
    json_output = {
        "metadata": metadata,
        "chunk_metadata": {
            str(chunk_idx): meta for chunk_idx, meta in chunk_metadata.items()
        },
        "results": json_results,
    }

    # Save JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(json_output, f, indent=2)
    logger.info(f"Results saved to {output_path}")

    # Save visualizations for frames
    vis_output_dir.mkdir(parents=True, exist_ok=True)

    for _chunk_idx, chunk_data in all_results.items():
        for frame_idx, frame_results in chunk_data.items():
            if int(frame_idx) % vis_stride != 0:
                continue

            output_img_path = vis_output_dir / f"frame_{int(frame_idx):06d}.png"
            if output_img_path.exists():
                continue  # Skip already saved frames

            # Decode RLE masks
            masks_rle = frame_results.get("masks_rle", [])
            if masks_rle:
                masks = np.stack([mask_util.decode(rle) for rle in masks_rle])
                frame = video_frames[int(frame_idx)]
                vis_image = overlay_func(frame, masks)
                vis_image.save(output_img_path)


def autoselect_torch_device():
    """
    Automatically selects the best available PyTorch device:
    - Prioritizes CUDA if available.
    - Fallback to MPS if CUDA is not available.
    - Default to CPU if neither CUDA nor MPS are available.

           torch.device: The selected device.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    return device


def plot_frame_from_df(
    df: pd.DataFrame,
    decoder,  # torchcodec.decoders.VideoDecoder
    frame_idx: int,
    *,
    ax: Optional[plt.Axes] = None,
    alpha: float = 0.40,  # mask opacity
    box_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),  # white (RGB 0..1)
    box_linewidth: float = 2.0,
    draw_labels: bool = True,
    fontsize: int = 9,
):
    """
    Matplotlib visualization for one frame using detections/segments from a MultiIndex DataFrame.

    DataFrame expectations:
      - Index: MultiIndex with level names ['frame', 'object_id']
      - Columns: ['size', 'counts', 'bbox', 'class', 'score']
          * 'bbox' is [x1, y1, x2, y2] in pixels or normalized (<= 2.0)
          * 'size' is [H, W] and 'counts' is a COCO RLE string (optional)
          * 'class' string or NaN; 'score' float or None

    Parameters
    ----------
    df : pd.DataFrame
        Per-object results. One row per object per frame.
    decoder : torchcodec.decoders.VideoDecoder
        Opened decoder; we read `frame_idx` from it.
    frame_idx : int
        Global frame index to draw.
    ax : Optional[plt.Axes]
        Provide an axes to draw into; if None, a new fig/axes is created.
    alpha : float
        Mask overlay transparency (0..1).
    box_color : (r,g,b) in 0..1
        Default edge color for boxes.
    box_linewidth : float
        Rectangle edge thickness.
    draw_labels : bool
        Show label with class • id • score near the box.
    fontsize : int
        Label font size.

    Returns
    -------
    fig, ax : (matplotlib.figure.Figure, matplotlib.axes.Axes)
        The figure and axes containing the rendered plot.
    """

    # ---- 1) Load the frame (RGB HWC) ----
    frame_chw = decoder[frame_idx]  # uint8 [C,H,W], device CPU/CUDA
    if isinstance(frame_chw, torch.Tensor):
        frame_rgb = frame_chw.permute(1, 2, 0).contiguous().cpu().numpy()
    else:
        # If NumPy is returned, ensure HWC:
        arr = np.asarray(frame_chw)
        frame_rgb = (
            arr if arr.ndim == 3 and arr.shape[2] == 3 else np.transpose(arr, (1, 2, 0))
        )

    H, W = frame_rgb.shape[:2]

    # ---- 2) Prepare axes ----
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(min(12, W / 80), min(8, H / 80)))
        created_fig = True
    else:
        fig = ax.figure

    ax.imshow(frame_rgb)
    ax.set_title(f"Frame {frame_idx}", fontsize=fontsize + 1)
    ax.axis("off")

    # If no 'frame' level, bail early
    if "frame" not in df.index.names:
        raise ValueError("DataFrame must have a MultiIndex with level 'frame'.")

    # ---- 3) Fetch rows for this frame (if any) ----
    try:
        rows = df.loc[frame_idx]
        if isinstance(rows, pd.Series):
            rows = rows.to_frame().T
    except KeyError:
        # no detections for this frame
        return fig, ax

    # Helpers
    def color_for_id(obj_id: int) -> Tuple[float, float, float]:
        # Stable pseudo-random color seeded by id (return RGB in 0..1)
        rng = np.random.default_rng(obj_id * 9767 + 1337)
        c = rng.integers(0, 255, size=3).astype(np.float32) / 255.0
        return float(c[0]), float(c[1]), float(c[2])

    def to_pixel_xyxy(xyxy: np.ndarray) -> np.ndarray:
        out = xyxy.astype(float).copy()
        if np.nanmax(out) <= 2.0:  # looks normalized -> scale
            out[[0, 2]] *= W
            out[[1, 3]] *= H
        # clip
        out[0::2] = np.clip(out[0::2], 0, W - 1)
        out[1::2] = np.clip(out[1::2], 0, H - 1)
        return out

    # ---- 4) Draw masks and boxes ----
    for obj_id, row in rows.iterrows():
        # (a) Mask via RGBA overlay
        rle_counts = row.get("counts", None)
        rle_size = row.get("size", None)
        if isinstance(rle_counts, str) and rle_counts:
            try:
                rle = {"size": rle_size, "counts": rle_counts}
                m = mask_util.decode(rle)
                if m.ndim == 3:  # (H,W,1)
                    m = m[..., 0]
                m = m.astype(bool)
                if m.any():
                    r, g, b = color_for_id(int(obj_id))
                    # Build per-object RGBA overlay and alpha where mask is True
                    overlay = np.zeros((H, W, 4), dtype=np.float32)
                    overlay[..., 0] = r
                    overlay[..., 1] = g
                    overlay[..., 2] = b
                    overlay[..., 3] = alpha * m.astype(np.float32)
                    ax.imshow(overlay, interpolation="nearest")
            except Exception:
                # If decoding fails, just skip mask
                pass

        # (b) Box + label
        bbox = row.get("bbox", None)
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            xyxy = to_pixel_xyxy(np.array(bbox, dtype=float))
            x1, y1, x2, y2 = xyxy
            w_box = max(0.0, x2 - x1)
            h_box = max(0.0, y2 - y1)
            rect = patches.Rectangle(
                (x1, y1),
                w_box,
                h_box,
                linewidth=box_linewidth,
                edgecolor=box_color,
                facecolor="none",
            )
            ax.add_patch(rect)

            if draw_labels:
                cls_name = row.get("class", None)
                if isinstance(cls_name, float) and math.isnan(cls_name):
                    cls_name = None
                score = row.get("score", None)
                label_bits = []
                if cls_name:
                    label_bits.append(str(cls_name))
                label_bits.append(f"id={int(obj_id)}")
                if score is not None:
                    try:
                        label_bits.append(f"{float(score):.2f}")
                    except Exception:
                        pass
                txt = " • ".join(label_bits)
                ax.text(
                    x1,
                    max(0, y1 - 4),
                    txt,
                    fontsize=fontsize,
                    color="w",
                    va="bottom",
                    ha="left",
                    bbox=dict(
                        boxstyle="round,pad=0.2", fc="black", ec="none", alpha=0.6
                    ),
                )

    if created_fig:
        fig.tight_layout()
    return fig, ax


def render_annotated_video(
    json_path: str | Path,
    video_path: str | Path,
    output_path: str | Path,
    *,
    mask_opacity: float = 0.4,
    box_thickness: int = 2,
    label_font_scale: float = 0.5,
    label_thickness: int = 1,
    fps: Optional[float] = None,
) -> None:
    """
    Render an annotated video with bounding boxes, object IDs, and masks.

    Uses the Supervision library if available, otherwise falls back to pure OpenCV.

    Args:
        json_path: Path to the JSON results file from sam3-hf-chunking-test.py
        video_path: Path to the original input video
        output_path: Path to save the output annotated .mp4 file
        mask_opacity: Opacity of mask overlays (0.0-1.0)
        box_thickness: Thickness of bounding box lines
        label_font_scale: Font scale for labels
        label_thickness: Thickness of label text
        fps: Output video FPS. If None, uses the source video FPS.

    Returns:
        None. Writes the annotated video to output_path.
    """
    json_path = Path(json_path)
    video_path = Path(video_path)
    output_path = Path(output_path)

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load JSON results
    with open(json_path, "r") as f:
        data = json.load(f)

    results = data.get("results", {})

    # Flatten results: chunk_idx -> frame_idx -> results
    # into: frame_idx -> results
    flat_results: dict[int, dict] = {}
    for chunk_idx, chunk_data in results.items():
        for frame_idx, frame_results in chunk_data.items():
            flat_results[int(frame_idx)] = frame_results

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    output_fps = fps if fps is not None else source_fps

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, output_fps, (width, height))

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create video writer: {output_path}")

    print(f"Rendering annotated video: {total_frames} frames at {output_fps:.2f} FPS")
    print(f"Output: {output_path}")

    # Color generator for consistent object colors
    def get_color_for_id(obj_id: int) -> tuple[int, int, int]:
        """Generate consistent BGR color for an object ID."""
        rng = np.random.default_rng(obj_id * 9767 + 1337)
        return tuple(int(c) for c in rng.integers(50, 255, size=3))

    if SUPERVISION_AVAILABLE:
        # Use Supervision for annotation
        # Use TRACK color lookup since we have tracker_id, not class_id
        box_annotator = sv.BoxAnnotator(
            thickness=box_thickness,
            color_lookup=sv.ColorLookup.TRACK,
        )
        label_annotator = sv.LabelAnnotator(
            text_scale=label_font_scale,
            text_thickness=label_thickness,
            text_position=sv.Position.TOP_LEFT,
            color_lookup=sv.ColorLookup.TRACK,
        )
        mask_annotator = sv.MaskAnnotator(
            opacity=mask_opacity,
            color_lookup=sv.ColorLookup.TRACK,
        )

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Get results for this frame
        frame_results = flat_results.get(frame_idx)

        if frame_results is not None:
            masks_rle = frame_results.get("masks_rle", [])
            boxes = frame_results.get("boxes", [])
            object_ids = frame_results.get("object_ids", [])
            scores = frame_results.get("scores", [])

            if SUPERVISION_AVAILABLE and len(boxes) > 0:
                # Decode masks from RLE
                masks = []
                for rle in masks_rle:
                    if rle and "counts" in rle and "size" in rle:
                        try:
                            mask = mask_util.decode(rle)
                            if mask.ndim == 3:
                                mask = mask[..., 0]
                            masks.append(mask.astype(bool))
                        except Exception:
                            masks.append(np.zeros((height, width), dtype=bool))
                    else:
                        masks.append(np.zeros((height, width), dtype=bool))

                # Convert to numpy arrays
                xyxy = np.array(boxes, dtype=np.float32)
                masks_np = np.array(masks, dtype=bool) if masks else None
                tracker_ids = np.array(object_ids, dtype=int)

                # Create Supervision Detections
                detections = sv.Detections(
                    xyxy=xyxy,
                    mask=masks_np,
                    tracker_id=tracker_ids,
                    confidence=np.array(scores, dtype=np.float32) if scores else None,
                )

                # Create labels
                labels = [
                    f"id:{oid} ({score:.2f})" if score else f"id:{oid}"
                    for oid, score in zip(
                        object_ids, scores or [None] * len(object_ids)
                    )
                ]

                # Annotate frame
                frame = mask_annotator.annotate(scene=frame, detections=detections)
                frame = box_annotator.annotate(scene=frame, detections=detections)
                frame = label_annotator.annotate(
                    scene=frame, detections=detections, labels=labels
                )

            elif len(boxes) > 0:
                # Fallback to pure OpenCV
                frame = _annotate_frame_opencv(
                    frame=frame,
                    masks_rle=masks_rle,
                    boxes=boxes,
                    object_ids=object_ids,
                    scores=scores,
                    mask_opacity=mask_opacity,
                    box_thickness=box_thickness,
                    label_font_scale=label_font_scale,
                    label_thickness=label_thickness,
                    get_color_for_id=get_color_for_id,
                )

        writer.write(frame)
        frame_idx += 1

        # Progress indicator
        if frame_idx % 100 == 0:
            print(f"  Processed {frame_idx}/{total_frames} frames...")

    cap.release()
    writer.release()
    print(f"Done! Output saved to: {output_path}")


def _annotate_frame_opencv(
    frame: np.ndarray,
    masks_rle: list[dict],
    boxes: list[list[float]],
    object_ids: list[int],
    scores: list[float],
    mask_opacity: float,
    box_thickness: int,
    label_font_scale: float,
    label_thickness: int,
    get_color_for_id: callable,
) -> np.ndarray:
    """
    Annotate a frame using pure OpenCV (fallback when Supervision is not available).

    Args:
        frame: BGR image (H, W, 3)
        masks_rle: List of RLE-encoded masks
        boxes: List of [x1, y1, x2, y2] bounding boxes
        object_ids: List of object IDs
        scores: List of confidence scores
        mask_opacity: Opacity for mask overlay
        box_thickness: Line thickness for boxes
        label_font_scale: Font scale for labels
        label_thickness: Text thickness for labels
        get_color_for_id: Function to get consistent color for object ID

    Returns:
        Annotated frame
    """
    height, width = frame.shape[:2]
    overlay = frame.copy()

    for i, (rle, box, obj_id) in enumerate(zip(masks_rle, boxes, object_ids)):
        color = get_color_for_id(obj_id)
        score = scores[i] if i < len(scores) else None

        # Draw mask
        if rle and "counts" in rle and "size" in rle:
            try:
                mask = mask_util.decode(rle)
                if mask.ndim == 3:
                    mask = mask[..., 0]
                mask = mask.astype(bool)

                # Create colored overlay where mask is True
                overlay[mask] = [
                    int(overlay[mask, c] * (1 - mask_opacity) + color[c] * mask_opacity)
                    for c in range(3)
                ]
                # Simpler approach: blend colors
                colored = np.zeros_like(frame)
                colored[mask] = color
                overlay = cv2.addWeighted(
                    overlay,
                    1 - mask_opacity * mask.any(),
                    colored,
                    mask_opacity,
                    0,
                )
            except Exception:
                pass

        # Draw bounding box
        x1, y1, x2, y2 = [int(c) for c in box]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, box_thickness)

        # Draw label
        label = f"id:{obj_id}"
        if score is not None:
            label += f" ({score:.2f})"

        # Get text size for background
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, label_font_scale, label_thickness
        )

        # Draw label background
        cv2.rectangle(
            overlay,
            (x1, y1 - text_h - baseline - 4),
            (x1 + text_w + 4, y1),
            color,
            -1,
        )

        # Draw label text
        cv2.putText(
            overlay,
            label,
            (x1 + 2, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            label_font_scale,
            (255, 255, 255),
            label_thickness,
            cv2.LINE_AA,
        )

    # Blend overlay with original for mask effect
    frame = cv2.addWeighted(frame, 1 - mask_opacity, overlay, mask_opacity, 0)

    return overlay
