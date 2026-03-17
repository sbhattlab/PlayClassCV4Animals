"""
Helper functions for chunking logic, including building manual chunks from user-supplied frame pairs and loading chunk info (frame ranges and prompt points) from a previous run's chunk_info.json
"""

from pathlib import Path

import numpy as np
from loguru import logger


def build_manual_chunks(
    frame_pairs: list[list[int]],
    tracker_override_indices: set[int] | None = None,
) -> list[tuple[int, int, str]]:
    """
    Build chunk list from user-supplied (start_frame, end_frame) pairs.

    If tracker_override_indices is provided, those chunk indices are forced to
    "tracker" type regardless of position (e.g. chunk 0 with manual prompts).
    """
    chunks = []
    for i, (start, end) in enumerate(frame_pairs):
        if tracker_override_indices and i in tracker_override_indices:
            model_type = "tracker"
        else:
            model_type = "video" if i == 0 else "tracker"
        chunks.append((int(start), int(end), model_type))
    return chunks


def load_chunks_from_chunk_info(
    run_dir: Path,
) -> tuple[list[list[int]], dict[int, dict[int, list]]]:
    """
    Load chunk boundaries and optional prompt points from a previous run's
    chunk_info.json.

    Returns:
        frame_pairs: list of [start_frame, end_frame] pairs
        prompt_points: dict mapping chunk_idx → {obj_id: [[x,y], ...]} for
            chunks that have non-empty prompt_points
    """
    import json

    chunk_info_path = Path(run_dir) / "chunk_info.json"
    if not chunk_info_path.exists():
        raise FileNotFoundError(f"chunk_info.json not found in: {run_dir}")

    with open(chunk_info_path) as f:
        chunk_info = json.load(f)

    frame_pairs = [chunk["frame_range"] for chunk in chunk_info["chunks"]]

    prompt_points: dict[int, dict[int, list]] = {}
    for i, chunk in enumerate(chunk_info["chunks"]):
        pp = chunk.get("prompt_points")
        if pp:
            # JSON serializes dict keys as strings; convert to int
            prompt_points[i] = {int(k): v for k, v in pp.items()}

    return frame_pairs, prompt_points


def chunk_video_frames_adaptive(
    total_frames: int,
    fps: float,
    chunk_seconds: float = 60.0,
    per_frame_metrics: list[dict] | None = None,
    search_window_seconds: float = 10.0,
    max_chunk_seconds: float = 150.0,
    # Legacy / unused parameters kept for call-site compatibility
    separation_windows: list[tuple[int, int]] | None = None,
    occlusion_periods: list[tuple[int, int]] | None = None,
    transition_frames: np.ndarray | None = None,
    min_chunk_seconds: float = 0.0,
    margin_seconds: float = 3.0,
) -> list[tuple[int, int, str]]:
    """
    Split video into chunks, with tracker-chunk boundaries placed at the frame
    of highest subject separation within a search window around each nominal cut.

    Generates initial fixed chunks (chunk 0 → video model, remainder → tracker),
    then for each tracker-chunk boundary searches ±``search_window_seconds`` for
    the frame with the highest ``separation_score``.  A frame's separation score
    is > 0 only when all hard gates pass (≥ min_objects detected, no overlapping
    bbox pairs above the IoU threshold, min centroid distance ≥ threshold); the
    score itself is ``min(min_centroid_distance, 1) × (1 − clustering_coeff)``.

    If no frame in the window has a positive separation score the nominal
    boundary is kept unchanged and a debug message is logged.

    Small trailing remainders (< 10 % of chunk size) are absorbed into
    the preceding chunk.

    Args:
        total_frames: Total number of frames in the video.
        fps: Video frame rate.
        chunk_seconds: Duration of each chunk in seconds.
        per_frame_metrics: Per-frame metric dicts from ``compute_yolo_per_frame_metrics``.
            Must contain ``frame_idx`` and ``separation_score``.  When ``None``
            the nominal fixed-chunk boundaries are returned unchanged.
        search_window_seconds: Search radius (±seconds) around each nominal boundary.
        max_chunk_seconds: Hard upper limit on chunk duration (VRAM constraint).
        separation_windows, occlusion_periods, transition_frames,
        min_chunk_seconds, margin_seconds: Unused legacy parameters kept for
            call-site compatibility.

    Returns:
        List of (start_frame, end_frame, model_type) tuples where model_type is
        ``"video"`` (first chunk) or ``"tracker"`` (all others).
    """
    chunk_frames_count = int(fps * chunk_seconds)
    search_window_frames = int(search_window_seconds * fps)
    max_frames = int(max_chunk_seconds * fps)

    # Step 1: Generate initial fixed chunks
    fixed_chunks: list[tuple[int, int, str]] = []
    end = min(chunk_frames_count, total_frames)
    fixed_chunks.append((0, end, "video"))

    start = end
    while start < total_frames:
        end = min(start + chunk_frames_count, total_frames)
        remaining_after = total_frames - end
        if 0 < remaining_after < chunk_frames_count * 0.1:
            end = total_frames
        fixed_chunks.append((start, end, "tracker"))
        start = end

    if len(fixed_chunks) <= 1 or per_frame_metrics is None:
        return fixed_chunks

    # Step 2: Build per-frame lookups
    # Primary: separation score (> 0 only when all hard gates pass)
    sep_lookup: dict[int, float] = {
        m["frame_idx"]: m.get("separation_score", 0.0) for m in per_frame_metrics
    }
    # Fallback: for frames that fail the sep gates (e.g. only 2 of 3 birds detected),
    # rank by (zero_iou, min_centroid_distance) so we still avoid high-overlap frames.
    # Requires at least 2 objects for pairwise metrics to exist.
    fallback_lookup: dict[int, tuple[float, float]] = {
        m["frame_idx"]: (
            -m.get("max_pairwise_bbox_iou", 1.0),
            m.get("min_centroid_distance", 0.0)
            if m.get("min_centroid_distance", float("inf")) != float("inf")
            else 0.0,
        )
        for m in per_frame_metrics
        if m.get("num_objects", 0) >= 2
    }

    boundaries = [c[0] for c in fixed_chunks]
    total_end = fixed_chunks[-1][1]
    adjusted_boundaries = list(boundaries)

    def _validate(candidate: int, idx: int) -> bool:
        prev_start = adjusted_boundaries[idx - 1]
        next_end = (
            adjusted_boundaries[idx + 1]
            if idx + 1 < len(adjusted_boundaries)
            else total_end
        )
        return (candidate - prev_start) <= max_frames and (
            next_end - candidate
        ) <= max_frames

    # Step 3: Refine each tracker-chunk boundary (index 1+)
    for i in range(1, len(adjusted_boundaries)):
        original = adjusted_boundaries[i]
        search_start = max(0, original - search_window_frames)
        search_end = min(total_end, original + search_window_frames)

        # Find the frame with the highest separation score in the window.
        best_frame: int | None = None
        best_score: float = 0.0
        for frame, score in sep_lookup.items():
            if search_start <= frame <= search_end and score > best_score:
                if _validate(frame, i):
                    best_score = score
                    best_frame = frame

        if best_frame is not None and best_frame != original:
            shift = best_frame - original
            adjusted_boundaries[i] = best_frame
            logger.info(
                f"Boundary {i}: frame {original} → {best_frame} "
                f"(shifted {shift:+d} frames, separation_score={best_score:.3f})"
            )
        else:
            # No frame with positive separation score found — fall back to the
            # least-bad frame: among n_obj>=2 candidates, minimise IoU then
            # maximise centroid distance.
            fb_frame: int | None = None
            fb_score: tuple[float, float] = (-float("inf"), -float("inf"))
            for frame, score in fallback_lookup.items():
                if search_start <= frame <= search_end and score > fb_score:
                    if _validate(frame, i):
                        fb_score = score
                        fb_frame = frame

            if fb_frame is not None and fb_frame != original:
                shift = fb_frame - original
                adjusted_boundaries[i] = fb_frame
                logger.warning(
                    f"Boundary {i}: frame {original} → {fb_frame} "
                    f"(shifted {shift:+d} frames, fallback: "
                    f"iou={-fb_score[0]:.3f}, dist={fb_score[1]:.3f})"
                )
            else:
                logger.debug(
                    f"Boundary {i}: frame {original} kept — no better frame found "
                    f"(best sep_score in window = {best_score:.3f})"
                )

    # Step 4: Rebuild chunks from adjusted boundaries
    adjusted_chunks: list[tuple[int, int, str]] = []
    for i in range(len(adjusted_boundaries)):
        s = adjusted_boundaries[i]
        e = (
            adjusted_boundaries[i + 1]
            if i + 1 < len(adjusted_boundaries)
            else total_end
        )
        adjusted_chunks.append((s, e, fixed_chunks[i][2]))

    # Step 5: Absorb small trailing chunks (< 10% of tracker chunk size)
    if len(adjusted_chunks) > 1:
        last_start, last_end, _ = adjusted_chunks[-1]
        if last_end - last_start < chunk_frames_count * 0.1:
            prev_start, _, prev_type = adjusted_chunks[-2]
            logger.info(
                f"Absorbing small trailing chunk "
                f"({last_end - last_start} frames, {(last_end - last_start) / fps:.1f}s) "
                f"into previous chunk"
            )
            adjusted_chunks[-2] = (prev_start, last_end, prev_type)
            adjusted_chunks.pop()

    return adjusted_chunks
