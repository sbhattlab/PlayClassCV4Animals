import pycocotools.mask as mask_util


def fmt_time(frame_idx, fps=25.0):
    """Frame index → MM:SS.f timestamp string."""
    t = frame_idx / fps
    m, s = divmod(t, 60)
    return f"{int(m):02d}:{s:05.2f}"


def _decode_rle_mask(counts, size):
    """Decode an RLE-encoded mask to a binary numpy array."""
    if isinstance(counts, str):
        counts = counts.encode("utf-8")
    rle = {"counts": counts, "size": size}
    return mask_util.decode(rle).astype(bool)
