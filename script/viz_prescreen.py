"""
Render prescreen_outputs.parquet onto the original video.

Each chunk's prescreen window is saved as a separate annotated clip inside
<run_dir>/visualizations/prescreen_clips/.  A title card at the top of each
frame shows the chunk index and frame number.

Usage (from project root):
    pixi run -e sam3-hf python -m script.viz_prescreen \
        --run-dir ext-data/output/results/sam3-hf/20260222_221702_sam3_hf

Optional flags:
    --chunks 9 10 11 12 13   # only render these chunk indices (default: all)
    --video-path <path>      # override video path from saved config
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pycocotools.mask as mask_util


# ---------------------------------------------------------------------------
# Colour palette — one per object ID (cycles if >8 IDs)
# ---------------------------------------------------------------------------
_PALETTE = [
    (50, 205, 50),    # green
    (30, 100, 255),   # blue
    (0, 165, 255),    # orange
    (200, 50, 200),   # purple
    (255, 200, 0),    # cyan-ish
    (50, 50, 220),    # red
    (0, 215, 255),    # gold
    (180, 105, 255),  # hot-pink
]


def _color_for_id(obj_id: int):
    return _PALETTE[int(obj_id) % len(_PALETTE)]


def _decode_mask(counts, size) -> np.ndarray:
    if isinstance(counts, str):
        counts = counts.encode("utf-8")
    return mask_util.decode({"counts": counts, "size": list(size)}).astype(np.uint8)


def _draw_frame(frame: np.ndarray, frame_rows: pd.DataFrame, chunk_idx: int) -> np.ndarray:
    """Overlay masks, boxes, labels onto *frame* (BGR uint8 in-place copy)."""
    out = frame.copy()
    overlay = out.copy()

    for obj_id, row in frame_rows.iterrows():
        color = _color_for_id(obj_id)
        mask = _decode_mask(row["counts"], row["size"])
        # Semi-transparent mask fill
        overlay[mask > 0] = color
        # Bounding box
        x1, y1, x2, y2 = [int(v) for v in row["bbox"]]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        # Label
        score = row.get("scores", float("nan"))
        label = f"#{obj_id}  {score:.2f}" if not np.isnan(float(score)) else f"#{obj_id}"
        cv2.putText(out, label, (x1, max(y1 - 5, 10)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, color, 2, cv2.LINE_AA)

    # Blend mask overlay
    cv2.addWeighted(overlay, 0.35, out, 0.65, 0, out)
    return out


def _add_title(frame: np.ndarray, text: str) -> np.ndarray:
    """Burn a title bar onto the top of the frame."""
    h, w = frame.shape[:2]
    bar_h = 28
    out = np.zeros((h + bar_h, w, 3), dtype=np.uint8)
    out[bar_h:] = frame
    cv2.putText(out, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out


def render_prescreen_clips(
    run_dir: Path,
    video_path: Path,
    chunks_to_render: list[int] | None = None,
):
    # ---- Load artefacts ---------------------------------------------------
    df = pd.read_parquet(run_dir / "prescreen_outputs.parquet")
    with open(run_dir / "chunk_info.json") as f:
        chunk_info_raw = json.load(f)
    chunk_info = chunk_info_raw["chunks"]  # list of dicts

    if chunks_to_render is not None:
        chunk_info = [c for c in chunk_info if c["chunk_idx"] in chunks_to_render]
        if not chunk_info:
            print("No matching chunks found — check --chunks values.")
            return

    # Group parquet by chunk using prescreen frame ranges
    # Each prescreen window: [chunk_start, chunk_start + prescreen_length)
    # We reconstruct the window from the parquet frame indices per chunk.

    out_dir = run_dir / "visualizations" / "prescreen_clips"
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {W}×{H} @ {fps:.1f} fps, {total_video_frames} frames")

    # Frame index → parquet rows (only prescreen frames)
    all_prescreen_frames = set(df.index.get_level_values("frame_idx").unique())

    for ci in chunk_info:
        cidx = ci["chunk_idx"]
        chunk_start, chunk_end = ci["frame_range"]

        # Identify which frames in this chunk have prescreen data
        chunk_ps_frames = sorted(
            f for f in all_prescreen_frames if chunk_start <= f < chunk_end
        )
        if not chunk_ps_frames:
            print(f"  Chunk {cidx}: no prescreen frames found in parquet, skipping.")
            continue

        ps_start = chunk_ps_frames[0]
        ps_end = chunk_ps_frames[-1]
        n_frames = len(chunk_ps_frames)
        ps_source = ci.get("prescreen_source_frame_idx")
        n_objects = ci.get("prescreen_num_objects", "?")

        print(f"  Chunk {cidx}: prescreen frames {ps_start}–{ps_end} "
              f"({n_frames} frames, {n_objects} objects selected from frame {ps_source})")

        out_path = out_dir / f"prescreen_chunk{cidx:02d}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H + 28))

        # Seek to start of this prescreen window
        cap.set(cv2.CAP_PROP_POS_FRAMES, ps_start)
        current = ps_start

        for fi in chunk_ps_frames:
            # Advance to the correct frame (should be sequential, but handle gaps)
            if fi > current:
                cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                current = fi

            ret, frame = cap.read()
            if not ret:
                print(f"    Warning: could not read frame {fi}")
                current += 1
                continue
            current += 1

            # Get prescreen detections for this frame
            if fi in df.index.get_level_values("frame_idx"):
                frame_rows = df.loc[fi]
                frame = _draw_frame(frame, frame_rows, cidx)

            selected_marker = " ★" if fi == ps_source else ""
            title = (f"Chunk {cidx}  |  frame {fi}  |  "
                     f"chunk_start={chunk_start}{selected_marker}")
            frame = _add_title(frame, title)
            writer.write(frame)

        writer.release()
        print(f"    Saved → {out_path}")

    cap.release()
    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Render prescreen clips for a SAM3 run.")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--video-path", type=Path, default=None,
                        help="Override video path (default: read from saved config)")
    parser.add_argument("--chunks", nargs="+", type=int, default=None,
                        help="Chunk indices to render (default: all)")
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.exists():
        print(f"Run directory not found: {run_dir}", file=sys.stderr)
        sys.exit(1)

    # Resolve video path from saved config if not overridden
    video_path = args.video_path
    if video_path is None:
        config_files = list(run_dir.glob("*.yaml"))
        if not config_files:
            print("No YAML config found in run dir; use --video-path", file=sys.stderr)
            sys.exit(1)
        import yaml
        with open(config_files[0]) as f:
            cfg = yaml.safe_load(f)
        video_path = Path(cfg["video_path"])

    if not video_path.is_absolute():
        video_path = Path.cwd() / video_path
    if not video_path.exists():
        print(f"Video not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    render_prescreen_clips(run_dir, video_path, chunks_to_render=args.chunks)


if __name__ == "__main__":
    main()
