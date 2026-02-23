import os

from src.debug import load_inputs, load_outputs

cfg, video_info = load_inputs("config/sam3_hf_manual_chunking.yaml")
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.get("CUDA_VISIBLE_DEVICES", "1")


from typing import Iterable, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from accelerate import Accelerator
from matplotlib import cm
from matplotlib.axes import Axes

from script.sam3.run_sam3_hf import _process_video_chunk
from src.grounding import find_best_grounding_frame, run_grounding
from src.processing import extract_equidistant_points_from_masks
from src.utils import load_video_frames_range


def overlay_masks(frame, masks, colors=None, alpha=0.5):
    """
    Overlay multiple binary masks on an RGB frame.

    Parameters:
        frame:  (H, W, 3) numpy array
        masks:  list of (H, W) binary numpy arrays
        colors: list of (R, G, B) tuples (same length as masks)
        alpha:  blending factor
    """
    out = frame.copy().astype(float)

    if colors is None:
        colors = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
        ][: len(masks)]

    for mask, color in zip(masks, colors):
        mask = mask.astype(bool)
        color_arr = np.zeros_like(out)
        color_arr[mask] = color
        out[mask] = (1 - alpha) * out[mask] + alpha * color_arr[mask]

    return out.astype(frame.dtype)


def draw_points_on_axes(
    ax: Axes,
    points: Union[np.ndarray, Iterable],
    colors: Union[None, Iterable[Tuple[float, float, float]]] = None,
    marker: str = "o",
    marker_size: int = 60,
    marker_edgecolor: str = "white",
    line: bool = True,
    line_width: float = 1.0,
    line_alpha: float = 0.9,
    labels: bool = True,
    label_offset: Tuple[float, float] = (4, -4),
) -> dict:
    """
    Draw points on an image axis.

    - points: array-like shape (N_objects, N_points_per_object, 2) in (x, y) coords.
    - ax: matplotlib Axes with the image already shown (e.g. plt.gca()).
    - Returns a dict with 'scatters' and 'lines' lists of artists.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 3 or pts.shape[2] != 2:
        raise ValueError("points must have shape (N, M, 2)")

    n = pts.shape[0]
    if colors is None:
        cmap = cm.get_cmap("tab10")
        colors = [cmap(i % cmap.N) for i in range(n)]
    else:
        colors = list(colors)

    scatters: List = []
    lines: List = []
    texts: List = []

    for i in range(n):
        obj_pts = pts[i]
        if obj_pts.size == 0:
            continue

        x = obj_pts[:, 0]
        y = obj_pts[:, 1]

        if line:
            (ln,) = ax.plot(
                x,
                y,
                "-",
                color=colors[i],
                linewidth=line_width,
                alpha=line_alpha,
                zorder=9,
            )
            lines.append(ln)

        sc = ax.scatter(
            x,
            y,
            s=marker_size,
            c=[colors[i]] * len(x),
            marker=marker,
            edgecolors=marker_edgecolor,
            linewidths=0.6,
            zorder=10,
        )
        scatters.append(sc)

        if labels:
            for j, (xx, yy) in enumerate(zip(x, y)):
                t = ax.text(
                    xx + label_offset[0],
                    yy + label_offset[1],
                    f"{i}:{j}",
                    color="white",
                    fontsize=8,
                    ha="left",
                    va="center",
                    zorder=11,
                    bbox=dict(
                        boxstyle="round,pad=0.1", fc=colors[i], ec="none", alpha=0.6
                    ),
                )
                texts.append(t)

    return {"scatters": scatters, "lines": lines, "texts": texts}


N_GROUNDING_FRAMES = 75


device = Accelerator().device

chunks = {i: c for i, c in enumerate(cfg.get("manual_chunk_frames"))}
start, end = list(chunks.values())[-1]
frames = load_video_frames_range(cfg.get("video_path"), start, end)
grounding_outputs = run_grounding(
    frames[0:N_GROUNDING_FRAMES],
    0,
    N_GROUNDING_FRAMES,
    _process_video_chunk,
    cfg,
    device,
)

gr_out_frame_idx, gr_out_masks, gr_out_boxes, gr_out_ids = find_best_grounding_frame(
    grounding_outputs,
    min_objects=cfg.get("min_objects", cfg.min_objects_for_tracking),
    method=cfg.get("best_frame_method", "combined"),
)

overlay = overlay_masks(frames[gr_out_frame_idx], gr_out_masks)
gr_out_masks_np = np.array(gr_out_masks)
points = extract_equidistant_points_from_masks(gr_out_masks_np)

plt.figure(figsize=(8, 8))
plt.imshow(overlay)
ax = plt.gca()
artists = draw_points_on_axes(ax, points)
plt.axis("off")
plt.show()
