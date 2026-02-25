import os

from src.debug.debug import load_inputs

cfg, video_info = load_inputs("config/sam3_hf_manual_chunking_c5g2.yaml")
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.get("CUDA_VISIBLE_DEVICES", "1")


import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from accelerate import Accelerator

from script.sam3.run_sam3_hf import _process_video_chunk
from src.grounding import find_best_grounding_frame, run_grounding
from src.processing import extract_equidistant_points_from_masks
from src.utils import load_video_frames_range
from src.viz import draw_points_on_axes

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

N_GROUNDING_FRAMES = cfg.get("text_grounding").get("grounding_frames", 125)
logger.info(f"Number grounding frames: {N_GROUNDING_FRAMES}")


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


device = Accelerator().device

chunks = {i: c for i, c in enumerate(cfg.get("manual_chunk_frames"))}
start, end = list(chunks.values())[-1]
frames = load_video_frames_range(
    cfg.get("video_path"), start, start + N_GROUNDING_FRAMES + 1
)
grounding_outputs = run_grounding(
    frames,
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

fig = plt.figure(figsize=(8, 8))
plt.imshow(overlay)
ax = plt.gca()
artists = draw_points_on_axes(ax, points)
plt.axis("off")
plt.show()

out_dir = Path("sandbox/debug/grounding_debug/")
out_dir.mkdir(exist_ok=True, parents=True)
png_path = out_dir / "overlay_with_points.png"
fig.savefig(png_path, dpi=300, bbox_inches="tight")
logger.info
plt.close(fig)
