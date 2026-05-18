"""Single source of truth for all paths used by the tracking-eval pipeline.

Repo-side paths host small, version-controlled artefacts (manifests,
keyframe schedules, results CSVs). Ext-data-side paths host heavy
artefacts (CVAT Backup, tracker run parquets, source MP4 clips, MOT
files) on the mounted `/mnt/birds/rebecca2025/` drive (symlinked at
`ext-data/` on the Linux box; a regular directory on local dev).

Each subcommand's CLI accepts `--manifest` / `--out` / `--predictions-root`
overrides; these constants supply the defaults.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXT_DATA = ROOT / "ext-data"

# Repo-side (version-controlled, small)
PKG_DIR = ROOT / "tracking_eval"
MANIFEST_CSV = PKG_DIR / "video_manifest.csv"
ANNOTATION_FRAMES = PKG_DIR / "annotation_frames.csv"
ANNOTATION_SUMMARY = PKG_DIR / "annotation_frames_summary.csv"
RESULTS_DIR = PKG_DIR / "results"
TRACKER_CONFIG = ROOT / "config" / "tracker.yaml"
TRACKEVAL_DIR = ROOT / "ext" / "TrackEval"

# Ext-data side (heavy, on /mnt/birds via symlink in production)
BENCHMARK_DIR = EXT_DATA / "output" / "results" / "tracker_benchmark"
CVAT_BACKUP_DIR = BENCHMARK_DIR / "cvat_backup" / "playclass-tracking-eval"
TRACKER_RUNS_ADAPTIVE = BENCHMARK_DIR / "tracker_outputs_adaptive"
TRACKER_RUNS_FIXED = BENCHMARK_DIR / "tracker_outputs_fixed"
SOURCE_VIDEOS_DIR = BENCHMARK_DIR / "source_videos"
GROUND_TRUTH_DIR = BENCHMARK_DIR / "ground_truth"
PREDICTIONS_MOT_DIR = BENCHMARK_DIR / "predictions_mot"

# Scan inputs (already on ext-data)
SCAN_RUNS_ROOT = EXT_DATA / "output" / "results" / "sam3-hf"
RAW_VIDEO_ROOT = EXT_DATA / "raw"
