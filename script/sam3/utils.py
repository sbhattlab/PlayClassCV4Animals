import math
from typing import Optional, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pycocotools.mask as mask_util
import torch


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
