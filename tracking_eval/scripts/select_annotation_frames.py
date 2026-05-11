"""Select annotation frames for the tracking-eval ground-truth pass.

Plan reference: Phase 1.2 of the CVPR-workshop tracking-evaluation plan.

For each selected video in ``tracking_eval/video_manifest.csv`` (selected=True),
this script:

1. Loads cached ``yolo_tracking.parquet`` from the YOLO scan run directory
   pointed to by the manifest's ``scan_dir`` column.
2. Recomputes per-frame metrics using ``config/tracker.yaml`` defaults (the
   same parameters used to produce the manuscript's 30-video tracks).
3. Computes default-parameter adaptive chunk boundaries via
   ``chunk_video_frames_adaptive`` (chunk_seconds=60, max=120, search=10).
4. Loads occlusion periods from ``yolo_scan_summary.parquet`` and picks the
   ``K`` longest per video.
5. Builds the annotation frame list from three sources:
   - ``chunk_guided``:          for each internal chunk boundary B, sample B-5, B, B+5.
   - ``occlusion_bracketing``:  for each of the top-K longest occlusion periods
                                (start, end), sample start-3, start, mid, end, end+3.
                                Constrains CVAT linear interpolation through
                                occlusions where it is otherwise unreliable.
   - ``uniform``:               one frame every ``UNIFORM_INTERVAL_SECONDS``.
   Frames are clamped to ``[0, total_frames-1]`` and deduplicated by source
   priority: chunk_guided > occlusion_bracketing > uniform.

Output: ``tracking_eval/annotation_frames.csv`` with columns
``video_id, frame_idx, source``.

Run from project root::

    pixi run -e tracker python -m tracking_eval.scripts.select_annotation_frames
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger
from omegaconf import OmegaConf

from src.metrics import compute_yolo_per_frame_metrics
from src.tracker.chunking import chunk_video_frames_adaptive
from src.tracker.scan import identify_occlusion_periods

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "tracking_eval" / "video_manifest.csv"
OUTPUT_PATH = REPO_ROOT / "tracking_eval" / "annotation_frames.csv"
TRACKER_CFG = REPO_ROOT / "config" / "tracker.yaml"

UNIFORM_INTERVAL_SECONDS = 30.0
BOUNDARY_OFFSETS = (-5, 0, 5)
OCCLUSION_TOP_K = 3
OCCLUSION_OFFSETS_END = (-3, +3)  # additional pre/post brackets around (start, end)


def select_frames_for_video(
    scan_dir: Path,
    cfg,
) -> tuple[list[tuple[int, str]], dict]:
    """Returns (list of (frame_idx, source), info_dict)."""
    yolo_df = pd.read_parquet(scan_dir / "yolo_tracking.parquet")
    summary = pd.read_parquet(scan_dir / "metrics" / "yolo_scan_summary.parquet")
    s = summary.iloc[:, 0] if summary.shape[1] == 1 else summary.squeeze()
    fps = float(s["fps"])
    total_frames = int(s["total_frames"])

    yolo_scan_cfg = cfg.get("yolo_scan", {})
    per_frame_metrics = compute_yolo_per_frame_metrics(
        yolo_df,
        occlusion_iou_threshold=float(yolo_scan_cfg.get("occlusion_iou_threshold", 0.07)),
        clustering_distance_threshold=float(yolo_scan_cfg.get("clustering_distance_threshold", 0.15)),
        use_normalized_coords=True,
        separation_min_objects=int(yolo_scan_cfg.get("separation_min_objects", 3)),
        separation_min_distance=float(yolo_scan_cfg.get("separation_min_distance", 0.10)),
    )

    chunks = chunk_video_frames_adaptive(
        total_frames,
        fps,
        chunk_seconds=float(cfg.get("chunk_seconds", 60)),
        per_frame_metrics=per_frame_metrics,
        search_window_seconds=float(cfg.get("adaptive_search_window_seconds", 10.0)),
        max_chunk_seconds=float(cfg.get("adaptive_max_chunk_seconds", 120)),
    )

    # Internal boundaries: start of every chunk except the first (which is 0).
    internal_boundaries = [c[0] for c in chunks[1:]]

    # Top-K longest occlusion periods, recomputed from per_frame_metrics
    # (avoids parsing the serialized list in yolo_scan_summary.parquet).
    occlusion_periods = identify_occlusion_periods(per_frame_metrics)
    occlusion_periods = sorted(occlusion_periods, key=lambda p: p[1] - p[0], reverse=True)
    top_occlusions = occlusion_periods[:OCCLUSION_TOP_K]

    # Priority order via setdefault: chunk_guided > occlusion_bracketing > uniform.
    selected: dict[int, str] = {}

    # 1) chunk_guided
    for b in internal_boundaries:
        for off in BOUNDARY_OFFSETS:
            f = b + off
            if 0 <= f < total_frames:
                selected.setdefault(f, "chunk_guided")

    # 2) occlusion_bracketing: start-3, start, mid, end, end+3
    for (a, b) in top_occlusions:
        mid = (a + b) // 2
        candidates = [a + OCCLUSION_OFFSETS_END[0], a, mid, b, b + OCCLUSION_OFFSETS_END[1]]
        for f in candidates:
            if 0 <= f < total_frames:
                selected.setdefault(f, "occlusion_bracketing")

    # 3) uniform backbone
    interval_frames = int(round(UNIFORM_INTERVAL_SECONDS * fps))
    for f in range(0, total_frames, interval_frames):
        selected.setdefault(f, "uniform")

    rows = sorted(selected.items())
    info = {
        "fps": fps,
        "total_frames": total_frames,
        "n_chunks": len(chunks),
        "n_internal_boundaries": len(internal_boundaries),
        "n_chunk_guided": sum(1 for _, src in rows if src == "chunk_guided"),
        "n_occlusion_bracketing": sum(1 for _, src in rows if src == "occlusion_bracketing"),
        "n_uniform": sum(1 for _, src in rows if src == "uniform"),
        "n_total": len(rows),
        "chunk_boundaries": internal_boundaries,
        "top_occlusions": top_occlusions,
    }
    return rows, info


def main():
    cfg = OmegaConf.load(TRACKER_CFG)
    manifest = pd.read_csv(MANIFEST_PATH)
    selected_rows = manifest[manifest["selected"]].copy()
    logger.info(f"Loaded manifest: {len(selected_rows)} selected videos")

    all_rows = []
    summary_rows = []
    for _, row in selected_rows.iterrows():
        video_id = row["video_id"]
        scan_dir = REPO_ROOT / row["scan_dir"]
        if not scan_dir.exists():
            raise FileNotFoundError(f"scan_dir from manifest not found: {scan_dir}")
        logger.info(f"  {video_id}  scan={scan_dir.name}")
        frame_rows, info = select_frames_for_video(scan_dir, cfg)
        for f, src in frame_rows:
            all_rows.append({"video_id": video_id, "frame_idx": int(f), "source": src})
        summary_rows.append({
            "video_id": video_id,
            **{k: v for k, v in info.items() if k not in ("chunk_boundaries", "top_occlusions")},
            "chunk_boundaries": ",".join(str(b) for b in info["chunk_boundaries"]),
            "top_occlusions": ";".join(f"{a}-{b}" for a, b in info["top_occlusions"]),
        })
        logger.info(
            f"    chunks={info['n_chunks']} internal_bounds={info['n_internal_boundaries']} "
            f"chunk_guided={info['n_chunk_guided']} occlusion={info['n_occlusion_bracketing']} "
            f"uniform={info['n_uniform']} total={info['n_total']}"
        )

    df = pd.DataFrame(all_rows).sort_values(["video_id", "frame_idx"])
    df.to_csv(OUTPUT_PATH, index=False)
    logger.info(f"Wrote {len(df)} rows to {OUTPUT_PATH}")

    sdf = pd.DataFrame(summary_rows)
    summary_path = OUTPUT_PATH.with_name("annotation_frames_summary.csv")
    sdf.to_csv(summary_path, index=False)
    logger.info(f"Wrote per-video summary to {summary_path}")

    print("\n=== Per-video summary ===")
    cols = ["video_id", "n_chunks", "n_internal_boundaries",
            "n_chunk_guided", "n_occlusion_bracketing", "n_uniform", "n_total"]
    with pd.option_context("display.width", 200, "display.max_colwidth", 60):
        print(sdf[cols].to_string(index=False))
    print(f"\nGrand total frames to annotate: {len(df)} (target ≈ 400)")
    counts = df["source"].value_counts(normalize=True) * 100.0
    print(
        "Source split: " + ", ".join(f"{k}={v:.1f}%" for k, v in counts.items())
    )


if __name__ == "__main__":
    main()
