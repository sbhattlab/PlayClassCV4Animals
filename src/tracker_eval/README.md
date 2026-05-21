# Tracker evaluation & ablation

PlayClass tracker-evaluation pipeline. Produces sparse-keyframe MOTChallenge ground truth from a CVAT project backup, converts tracker predictions for the 6-way ablation, and scores them with TrackEval (HOTA family) and py-motmetrics (IDF1, MOTA, IDsw, etc.).

The pipeline brackets an offline CVAT annotation checkpoint: everything before is the **prepare** stage (manifest + keyframe schedule), everything after is the **score** stage (MOT export + prediction conversion + evaluation).

## Layout

CLI lives in this package (`src/tracker_eval/`):

| Module | Subcommand | Purpose |
|---|---|---|
| `manifest.py` | `build-manifest` | Rank candidate videos by YOLO-scan difficulty; mark the 5 eval picks. |
| `frame_selection.py` | `select-frames` | Emit the per-video keyframe schedule (chunk-guided + occlusion-bracketing + uniform). |
| `cvat_to_mot.py` | `cvat-to-mot` | Convert the CVAT project backup to one sparse MOTChallenge 1.1 file per video. |
| `predictions.py` | `convert-preds` | Convert each variant's tracker output parquet to MOTChallenge .txt. |
| `evaluate.py` | `evaluate` | Score all variants with TrackEval + py-motmetrics. |
| `__main__.py` | `prepare`, `score` | Umbrella entry points. |

Path defaults are centralised in `paths.py`. Data-side (small, version-controlled) artefacts live under `data/tracker_eval/`; heavy artefacts (CVAT backup, tracker run parquets, source MP4s, MOT files) live under `ext-data/tracker_benchmark/`.

## Evaluation subset

Five 15-min videos from days 28 and 29, one per cage, with the hardest group per cage selected from the YOLO scan via a composite difficulty score (high-occlusion fraction, overlap count, mean pairwise IoU, object-count churn, occlusion periods per minute, inverse mean centroid distance, inverse mean separation score). Recorded in `data/tracker_eval/video_manifest.csv` (`selected=True` rows).

## Sparse ground truth

Ground truth is sparse by design: 87–88 human-verified frames per video (462 total). At each keyframe every visible bird has a true human-drawn bbox; CVAT track-mode linear interpolation is used only as a navigation/identity-tagging convenience and is **dropped** at export. Standard MOT metrics (HOTA, IDF1, MOTA) are defined over GT-present frames and impose no density assumption; this matches TAO, multi-animal DLC, SLEAP, and idtracker.ai conventions.

The schedule combines three sources, deduplicated in priority order (chunk-guided > occlusion-bracketing > uniform):

1. **Chunk-guided** — for every internal adaptive-chunk boundary `B` from the production config (`chunk_seconds=60`, `max_chunk_seconds=120`, `search_window_seconds=10`), sample `{B-5, B, B+5}`.
2. **Occlusion-bracketing** — for each of the three longest occlusion periods `(a, b)` from the YOLO scan, sample `{a-3, a, floor((a+b)/2), b, b+3}` to densify GT where interpolation is least reliable.
3. **Uniform** — one frame every 30 s as a temporal backbone.

Schedules live in `data/tracker_eval/annotation_frames.csv`; per-video totals in `annotation_frames_summary.csv`.

## Tracker variants (6-way ablation)

| Variant | Slot | Family | Recovery |
|---|---|---|---|
| `A_yolo_botsort` | A | YOLO + BoT-SORT | — |
| `B_gs2_strict` | B-strict | Grounded-SAM-2 | none (frame-0 GDINO; abort on failure) |
| `B_gs2_fixed` | B-parity | Grounded-SAM-2 | best-frame seed + GDINO reinit on total loss |
| `C_sam3_frame_zero` | C-strict | SAM 3 | none (frame-0 grounding; both fallbacks disabled) |
| `D_sam3_fixed` | D | SAM 3 | adaptive grounding (scan + ranking + fallbacks), fixed 60 s chunks |
| `E_sam3_adaptive` | E | SAM 3 | adaptive grounding + adaptive (occlusion-aware) chunking — **full method** |

A supplementary `F_sam3_adaptive_strict` (full method with both fallbacks disabled) isolates the failure-compensation contribution.

Each variant's run parquets are picked up from a dedicated root under `ext-data/tracker_benchmark/` (see `paths.py`). A missing root produces an empty MOT file for that variant so it is scored as a zero-prediction penalty rather than silently dropped.

## Pre-CVAT (prepare)

Runs the manifest build + keyframe selection. Inputs: YOLO scan outputs under `ext-data/output/results/sam3-hf/`. Outputs: `data/tracker_eval/{video_manifest.csv,annotation_frames.csv,annotation_frames_summary.csv}`.

```sh
pixi run -e tracker python -m src.tracker_eval prepare
```

Individual stages:

```sh
python -m src.tracker_eval build-manifest [--days 28 29 ...]
python -m src.tracker_eval select-frames
```

## CVAT handoff

Annotation runs offline on a separate workstation. Hand the annotator:

| File | Purpose |
|---|---|
| `data/tracker_eval/annotation_frames.csv` | The 462 keyframes to annotate (`video_id, frame_idx, source`). |
| `data/tracker_eval/annotation_frames_summary.csv` | Per-video keyframe counts and chunk/occlusion metadata. |
| `data/tracker_eval/video_manifest.csv` | Selection metadata including the source MP4 paths. |
| The 5 `.mp4` files | Listed in `video_manifest.csv` under `path` for rows with `selected=True`. |

**CVAT setup** (local Docker, v2.64.0+):

```sh
git clone https://github.com/cvat-ai/cvat.git && cd cvat
docker compose up -d
docker exec -it cvat_server bash -ic 'python manage.py createsuperuser'
# http://localhost:8080
```

**Project & task setup.** Create a project `playclass-tracker-eval` with a single label `bird` (no attributes — identity is captured by Track membership). Create one **Task per video** (5 total), task name = `video_id` from the manifest. Upload the MP4 directly; do **not** convert to image set — that disables interpolation.

**Workflow.** For each task, switch to Track mode. For each of the three birds, jump to the first listed keyframe, draw a tight bbox (creates a Track), then visit every subsequent listed keyframe and drag/resize the bbox to fit. Only visit frames listed in `annotation_frames.csv` — interpolated frames between keyframes are dropped at export, so corrective in-between keyframes are wasted effort.

**Conventions.** Partial occlusion: annotate the full estimated bbox, including the occluded portion (MOT convention). Total occlusion / bird out of frame: mark the Track `outside=true` (shortcut **O**); resume on the same Track when the bird reappears — do not create a new Track. Identity ambiguity after long occlusion: re-watch at reduced speed; if genuinely unrecoverable, flag rather than guess (a wrong guess creates a false ID switch the tracker is penalised for). IoU threshold at scoring is 0.5; pixel-perfect bboxes are not required.

## Basis of the backup

The handoff back from the annotation workstation is a **CVAT project Backup** (Project page → Actions → Backup project). Extract under `ext-data/tracker_benchmark/cvat_backup/playclass-tracker-eval/` (the path in `paths.CVAT_BACKUP_DIR`). The converter reads the per-task `annotations.json` inside the backup directly.

The backup is preferred over CVAT's MOT/COCO/CVAT-for-video exports because:

- It is a complete, self-contained snapshot of the project state (labels, task metadata, raw shapes) — re-importable into another CVAT instance for re-annotation or QA.
- The converter walks the native `annotations.json` and keeps only **human-drawn, non-`outside` rectangles**. CVAT's MOT export emits interpolated frames as if they were GT; the backup lets us identify and drop them explicitly.
- Track IDs are assigned in track-declaration order at conversion time. Their absolute values are irrelevant — HOTA/IDF1 are permutation-invariant — but the deterministic ordering keeps re-runs byte-identical.

The result is one sparse MOTChallenge 1.1 file per video under `ext-data/tracker_benchmark/ground_truth/`, containing only the 87–88 keyframe rows per bird — i.e. true sparse GT, no interpolation pseudo-GT.

## Post-CVAT (score)

Runs the conversion + evaluation. Inputs: the extracted CVAT backup and the per-variant tracker run dirs under `ext-data/tracker_benchmark/`. Outputs: MOT files under `ext-data/tracker_benchmark/{ground_truth,predictions_mot}/` and results CSVs under `data/tracker_eval/results/`.

```sh
pixi run -e tracker-evaluation python -m src.tracker_eval score
```

Individual stages:

```sh
python -m src.tracker_eval cvat-to-mot     # backup → ground_truth/*.txt
python -m src.tracker_eval convert-preds   # per-variant parquets → predictions_mot/<variant>/*.txt
python -m src.tracker_eval evaluate        # → data/tracker_eval/results/*.csv
```

## Results

`evaluate` writes three CSVs to `data/tracker_eval/results/`:

| File | Granularity |
|---|---|
| `metrics_per_video.csv` | One row per (variant, video). |
| `metrics_per_cage.csv` | Aggregated per (variant, cage). |
| `metrics_aggregate.csv` | One row per variant (whole eval subset). |

Reported metrics:

- **TrackEval (HOTA family)** — HOTA, DetA, AssA, LocA, DetRe, DetPr, AssRe, AssPr, OWTA. HOTA averaged across the standard α ∈ {0.05, 0.10, …, 0.95} sweep.
- **py-motmetrics 1.4.0** — IDF1, IDP, IDR, MOTA, MOTP, precision, recall, ID switches, mostly-tracked, mostly-lost, fragmentations.

Matching uses IoU 0.5. Variants with missing predictions are scored on empty MOT files (zero-prediction penalty), so partial result regenerations are safe.

## Running tracker inference

**SAM 3 variants (C-strict, D, E)** are launched via the normal pipeline workflow — `script/run_tracker.py` with the appropriate eval config from `data/tracker_eval/config/`. See `.claude/rules/tracker.md` for the main tracker workflow.

**Grounded-SAM-2 variants (B-strict, B-parity)** are launched via a dedicated entry point at `src/tracker_eval/run_gs2_tracker.py` (pixi task: `gs2_tracker`). gs2 is included only as an ablation baseline for this evaluation — it is not part of the main PlayClass pipeline, so its launcher lives inside the `tracker_eval` package rather than under `script/`.

```sh
pixi run -e gs2 python -m src.tracker_eval.run_gs2_tracker \
    --config data/tracker_eval/config/gs2_parity_day_28.yaml
```

## Configs

Per-variant tracker configs live under `data/tracker_eval/config/` (production configs stay in `config/`):

| Variant | Configs |
|---|---|
| A, E | `config/tracker.yaml` (production adaptive run; supplies both YOLO scan and SAM 3 adaptive outputs) |
| B-strict | `data/tracker_eval/config/gs2_fixed_day_{28,29}.yaml` (with `gs2.enable_recovery: false`) |
| B-parity | `data/tracker_eval/config/gs2_parity_day_{28,29}.yaml` |
| C-strict | `data/tracker_eval/config/tracker_frame_zero_day_{28,29}.yaml` |
| D | `data/tracker_eval/config/tracker_rerun_fixed_day_{28,29}.yaml` |

Per-video `grounding_frames` is held constant across the C→D ablation (125 for `C1G2`, `C2G2`, `C3G2`, `C4G2`; 375 for `C5G3`, whose earliest viable grounding frame is past the 125-frame default).