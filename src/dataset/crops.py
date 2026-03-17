"""Shared crop modes for embedding extraction.

All extraction scripts (DINOv3, V-JEPA 2, VideoPrism) use these functions
to crop tracked objects from video frames. Crops are returned as raw numpy
arrays; model-specific resizing is handled by the caller.
"""

import numpy as np

CROP_MODES = ("bbox", "plain256", "union512", "darken512", "roi512")


def compute_union_bbox(bboxes, bbox_scale=1.0):
    """Compute the union bounding box across all frames in a window.

    Parameters
    ----------
    bboxes : array-like, shape (N, 4)
        Bounding boxes ``[x1, y1, x2, y2]``.
    bbox_scale : float
        Optional scale factor applied to the union bbox.

    Returns
    -------
    tuple[int, int, int, int]
        ``(x1, y1, x2, y2)`` of the union bbox.
    """
    bboxes = np.asarray(bboxes)
    x1, y1 = bboxes[:, 0].min(), bboxes[:, 1].min()
    x2, y2 = bboxes[:, 2].max(), bboxes[:, 3].max()
    if bbox_scale != 1.0:
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = (x2 - x1) * bbox_scale, (y2 - y1) * bbox_scale
        x1, y1 = cx - w / 2, cy - h / 2
        x2, y2 = cx + w / 2, cy + h / 2
    return int(x1), int(y1), int(x2), int(y2)


def compute_union_origin(bboxes, frame_h, frame_w, crop_size=512):
    """Compute the fixed crop origin for union-centroid modes.

    Centers a ``crop_size x crop_size`` crop on the union-bbox centroid,
    clamped to frame bounds.

    Returns
    -------
    tuple[int, int]
        ``(origin_x, origin_y)`` — top-left corner of the crop.
    """
    bboxes = np.asarray(bboxes)
    x1, y1 = bboxes[:, 0].min(), bboxes[:, 1].min()
    x2, y2 = bboxes[:, 2].max(), bboxes[:, 3].max()
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    half = crop_size // 2
    ox = max(0, min(cx - half, frame_w - crop_size))
    oy = max(0, min(cy - half, frame_h - crop_size))
    return ox, oy


def crop_frame(frame_np, bbox, crop_mode, *, bbox_scale=1.0, union_origin=None):
    """Crop a single frame according to the specified crop mode.

    Parameters
    ----------
    frame_np : ndarray, shape (H, W, 3)
        RGB video frame.
    bbox : sequence of 4 ints
        ``[x1, y1, x2, y2]`` bounding box for the object in this frame.
    crop_mode : str
        One of :data:`CROP_MODES`.
    bbox_scale : float
        Scale factor for ``bbox`` mode (1.0 = tight crop).
    union_origin : tuple[int, int] or None
        ``(ox, oy)`` from :func:`compute_union_origin`. Required for
        ``union512``, ``darken512``, and ``roi512``.

    Returns
    -------
    crop : ndarray or None
        Cropped region, or ``None`` if the crop is invalid (empty bbox or
        bird not fully contained in the fixed crop).
    extra : dict or None
        Extra metadata. For ``roi512``, contains
        ``'patch_bounds': (py1, py2, px1, px2)`` mapping the bird bbox
        to a 16x16 patch grid.
    """
    h, w = frame_np.shape[:2]
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

    if crop_mode == "plain256":
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        half = 128
        ox = max(0, min(cx - half, w - 256))
        oy = max(0, min(cy - half, h - 256))
        return frame_np[oy:oy + 256, ox:ox + 256], None

    if crop_mode in ("union512", "darken512", "roi512"):
        assert union_origin is not None, f"{crop_mode} requires union_origin"
        ox, oy = union_origin
        crop_size = 512
        # Skip frames where bird is not fully contained in the fixed crop
        if x1 < ox or y1 < oy or x2 > ox + crop_size or y2 > oy + crop_size:
            return None, None
        crop = frame_np[oy:oy + crop_size, ox:ox + crop_size]

        extra = None
        if crop_mode == "darken512":
            # Darken to 40%, restore bird bbox to full brightness
            bright = crop.copy()
            crop = (crop.astype(np.float32) * 0.4).astype(np.uint8)
            rx1 = max(0, x1 - ox)
            ry1 = max(0, y1 - oy)
            rx2 = min(crop_size, x2 - ox)
            ry2 = min(crop_size, y2 - oy)
            if rx2 > rx1 and ry2 > ry1:
                crop[ry1:ry2, rx1:rx2] = bright[ry1:ry2, rx1:rx2]
        elif crop_mode == "roi512":
            # Map bird bbox to 16x16 patch grid within the 512 crop
            rx1 = max(0, x1 - ox)
            ry1 = max(0, y1 - oy)
            rx2 = min(crop_size, x2 - ox)
            ry2 = min(crop_size, y2 - oy)
            patch_scale = 16 / crop_size
            extra = {
                "patch_bounds": (
                    int(ry1 * patch_scale),
                    min(16, int(ry2 * patch_scale) + 1),
                    int(rx1 * patch_scale),
                    min(16, int(rx2 * patch_scale) + 1),
                )
            }

        return crop, extra

    # Default: bbox mode
    if bbox_scale != 1.0:
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        bw, bh = (x2 - x1) * bbox_scale, (y2 - y1) * bbox_scale
        x1, y1 = int(cx - bw / 2), int(cy - bh / 2)
        x2, y2 = int(cx + bw / 2), int(cy + bh / 2)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None, None
    return frame_np[y1:y2, x1:x2], None
