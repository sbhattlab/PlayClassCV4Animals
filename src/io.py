"""
IO utilities for loading video metadata and frames.
"""

from pathlib import Path

import cv2
from loguru import logger


def get_video_metadata(video_path: str | Path) -> tuple[float, int]:
    """Return (fps, total_frames) for a video without loading all frames into RAM."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, total_frames


def load_video_frames_torchcodec(
    video_path: str | Path,
    start_frame: int,
    end_frame: int,
    device: str = "cpu",
) -> list:
    """Load frames [start_frame, end_frame) as a list of RGB numpy arrays.

    Uses torchcodec for frame-accurate decoding (seek_mode="exact").
    The decoder is cached per video path so the index scan (~18 s on first
    access) only happens once across all chunks of the same video.

    When *device* is ``"cuda"``, decoding is offloaded to NVDEC hardware.
    """
    from torchcodec.decoders import VideoDecoder

    key = str(video_path)
    if key not in _torchcodec_decoder_cache:
        logger.info(
            "Building torchcodec index for '{}' (device={}).",
            Path(video_path).name,
            device,
        )
        _torchcodec_decoder_cache[key] = VideoDecoder(key, device=device)

    decoder = _torchcodec_decoder_cache[key]
    batch = decoder.get_frames_in_range(start=start_frame, stop=end_frame)
    # batch.data shape: (N, C, H, W) uint8 tensor
    return [frame.permute(1, 2, 0).cpu().numpy() for frame in batch.data]


def load_video_frames_sequential(video_path, start_frame, end_frame):
    """Load frames [start_frame, end_frame) sequentially via cv2.

    Slower than seek-based loading but frame-accurate for all codecs
    (including H.264). Use this when torchcodec is not available (e.g.
    JAX environments).
    """
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    for i in range(end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        if i >= start_frame:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def load_video_frames_range(
    video_path: str | Path, start_frame: int, end_frame: int
) -> list:
    """Load frames [start_frame, end_frame) from a video file as a list of RGB numpy arrays.

    .. deprecated::
        Uses cv2.CAP_PROP_POS_FRAMES seeking, which is unreliable for
        H.264-encoded videos. Use :func:`load_video_frames` instead.
    """
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    for _ in range(end_frame - start_frame):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


_torchcodec_decoder_cache: dict[str, object] = {}
