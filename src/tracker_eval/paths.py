"""Single source of truth for all paths used by the tracker-eval pipeline.

Data-side paths host small, version-controlled artefacts (manifests,
keyframe schedules, results CSVs). Ext-data-side paths host heavy
artefacts (CVAT Backup, tracker run parquets, source MP4 clips, MOT
files) on the mounted `/mnt/birds/rebecca2025/` drive (symlinked at
`ext-data/` on the Linux box; a regular directory on local dev).

Each subcommand's CLI accepts `--manifest` / `--out` / `--predictions-root`
overrides; these constants supply the defaults.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXT_DATA = ROOT / "ext-data"

# Data-side (version-controlled, small)
DATA_DIR = ROOT / "data" / "tracker_eval"
MANIFEST_CSV = DATA_DIR / "video_manifest.csv"
ANNOTATION_FRAMES = DATA_DIR / "annotation_frames.csv"
ANNOTATION_SUMMARY = DATA_DIR / "annotation_frames_summary.csv"
RESULTS_DIR = DATA_DIR / "results"
TRACKER_CONFIG = ROOT / "config" / "tracker.yaml"
TRACKEVAL_DIR = ROOT / "ext" / "TrackEval"

# Ext-data side (heavy, on /mnt/birds via symlink in production)
BENCHMARK_DIR = EXT_DATA / "tracker_benchmark"
CVAT_BACKUP_DIR = BENCHMARK_DIR / "cvat_backup" / "playclass-tracker-eval"
TRACKER_RUNS_ADAPTIVE = BENCHMARK_DIR / "tracker_outputs_adaptive"
TRACKER_RUNS_FIXED = BENCHMARK_DIR / "tracker_outputs_fixed"
TRACKER_RUNS_GS2 = BENCHMARK_DIR / "tracker_outputs_gs2"
TRACKER_RUNS_GS2_STRICT = BENCHMARK_DIR / "tracker_outputs_gs2_strict"
TRACKER_RUNS_FRAME_ZERO = BENCHMARK_DIR / "tracker_outputs_sam3_frame_zero"
SOURCE_VIDEOS_DIR = BENCHMARK_DIR / "source_videos"
GROUND_TRUTH_DIR = BENCHMARK_DIR / "ground_truth"
PREDICTIONS_MOT_DIR = BENCHMARK_DIR / "predictions_mot"

# Scan inputs (already on ext-data)
SCAN_RUNS_ROOT = EXT_DATA / "output" / "results" / "sam3-hf"
RAW_VIDEO_ROOT = EXT_DATA / "raw"
