# Refined Plan: Tracking Evaluation for CVPR 2026 Workshop Revision

## Current status (2026-05-18)

**At Phase 5 (manuscript integration).** Phases 1–4 are complete: the 5-video held-out set has been selected, annotated in CVAT, converted to sparse MOTChallenge GT, and scored against Variants A and C. Per-video / per-cage / aggregate CSVs are already written to `tracking_eval/results/`.

**Pipeline consolidation (2026-05-18).** The previous flat `tracking_eval/scripts/` directory has been replaced by a Python package (`tracking_eval/__init__.py`, `__main__.py`, `paths.py`) exposing a single CLI with subcommands. The canonical entry points are now the two umbrella commands `pixi run -e tracker python -m tracking_eval prepare` (build-manifest + select-frames) and `pixi run -e tracker-evaluation python -m tracking_eval score` (cvat-to-mot + convert-preds + evaluate), bracketing the offline CVAT annotation checkpoint. Individual stages remain callable for ad-hoc re-runs (e.g. `python -m tracking_eval evaluate`). All runtime paths previously living under `tmp/tracker_benchmark/` (CVAT Backup, tracker run parquets, the 5 source MP4 clips) have been migrated to `ext-data/output/results/tracker_benchmark/{cvat_backup,tracker_outputs_adaptive,source_videos}/` so the artefact set is co-located with the existing `ground_truth/`, `predictions_mot/`, and (eventually) `tracker_outputs_fixed/` subdirs. Path defaults live in `tracking_eval/paths.py`.

**Remaining work before write-up:**

1. **Run Variant B (SAM 3 with fixed 60 s chunking, no YOLO scan).** Added retrospectively to turn the head-to-head into a three-way ablation that isolates the contribution of the YOLO-driven adaptive chunking step. Without it, Variant C's gain over Variant A could be attributed to SAM 3 alone rather than to the synergy with the occlusion-informed chunker — which is the actual method claim. See the revised Phase 2 table below.
2. **Re-run `convert_predictions.py` and `evaluate_tracking.py`** once Variant B parquets exist. The conversion script already accepts an optional `--predictions-root-fixed`; the evaluator gracefully skips a variant whose MOT files are not yet on disk, so partial re-runs are safe.
3. **Manuscript integration (Phase 5)** under the 3-way framing.

**Retrospective renaming (2026-05-18).** The previous "Variant B: SAM3 default" is now **Variant C: SAM3 adaptive** (since the production config uses adaptive chunking), and the new fixed-chunking arm slots in as **Variant B**. Output subdirectories and the `VARIANTS` tuple in the eval scripts have been updated accordingly.

## Context

PlayClass was accepted at CV4Animals (CVPR 2026 workshop) as a poster, with revisions. This plan addresses **Reviewer 1's tracking-related concerns only**:

- "Lack of standard tracking evaluation metrics (e.g., HOTA, IDF1) to validate identity preservation"
- "Dependence on manual post-processing in the tracking pipeline, reducing automation and reproducibility"
- "Limited analysis of robustness to tracking errors, occlusions, and domain variation"
- "Limited novelty, with the method primarily combining existing tracking, feature extraction, and classification components"

**Reviewer 2's concerns** (ethogram description, "end-to-end" terminology, precision/recall + class-level metrics, dataset release plan, fine-grained granularity exploration) are **out of scope** for this plan and will be handled separately.

**Strategy** (revised 2026-05-11; further revised 2026-05-12 and 2026-05-18): Use **in-distribution videos from days 28 and 29** as the evaluation set, one per cage, selecting the hardest group per cage via YOLO-scan-based difficulty ranking across both days jointly. At least 2 of the 5 picks are required to come from day 28 to ensure cross-day representativeness of the manuscript's training distribution. The day-37 external-validation approach was dropped after visual inspection showed day-37 birds look markedly different (older plumage) and move significantly less than the days 28–29 birds the manuscript's classification results are built on — benchmarking against day 37 would inflate tracker metrics relative to the actual training/eval distribution. The 5 LOCO folds in the manuscript already produce *disjoint* test partitions (each fold tests one full cage); drawing one video per cage from these natural per-fold test sets gives a 5-video subset that is in-distribution, cross-environment, and genuinely held out for the classifier. Run three **fully automated (no manual post-processing)** tracker configurations on the 5 bbox-annotated videos and report standard MOT metrics in a three-way ablation:

- **Variant A — YOLO + BoT-SORT only.** Generic tracker baseline. Isolates the scan stage and counters Reviewer 1's "limited novelty" framing.
- **Variant B — SAM 3 with fixed 60 s chunking.** SAM 3 propagation with no YOLO scan and no boundary refinement. Isolates the segmentation-propagation stage when run blind to occlusion structure.
- **Variant C — SAM 3 with adaptive, occlusion-informed chunking.** The full method. Boundaries are shifted within a ±10 s window toward frames maximising bird separation and avoiding occlusion.

The hypothesis is that Variant C strictly beats $\max(\text{A}, \text{B})$ and that the gain is concentrated in association-side metrics (AssA, IDF1, ID switches) at the chunk-boundary and occlusion-bracketed keyframes — i.e. that the lift comes from the *synergy* between YOLO scan and SAM 3, not from either component alone. This addresses Reviewer 1's three substantive concerns (missing HOTA/IDF1, manual post-processing, limited novelty) in a single ablation table.

**Sparse-keyframe evaluation (2026-05-12).** Tracker metrics are computed on the **88 human-verified keyframes per video** (~440 total across the 5 videos), not on dense interpolated ground truth. CVAT's Track-mode interpolation is treated as an annotation convenience for navigation, not a source of GT; interpolated frames are filtered out before scoring. Rationale: linearly interpolated bboxes drift off non-linearly-moving birds, producing fake GT that penalises correct trackers and requires expensive per-interval QA. The standard MOT metrics (MOTA, IDF1, HOTA) are defined over GT-present frames and impose no density assumption — sparse evaluation matches the conventions of TAO (1 FPS GT on 30 FPS video) and the multi-animal pose-tracking literature (DLC, SLEAP, idtracker.ai). See the appendix at the end of this document for the verified citation chain and the manuscript-ready paragraph.

**Scope reduction under deadline (2026-05-12).** CV4Animals revision is due **2026-05-21**. Within that window, running a sweep-tuned SAM3 variant on top of the planned configurations is not feasible. **The Phase 3 chunking hyperparameter sweep is dropped**, and what was originally labelled "Variant C (SAM3 tuned)" is dropped along with it. The deferred sweep is acknowledged in the limitations section.

**Three-way ablation added (2026-05-18).** The comparison was previously framed as **A (YOLO+BoT-SORT) vs B (SAM3 default)**, where "SAM3 default" meant SAM 3 with the production adaptive-chunking config. That framing conflates two ingredients of the proposed method (SAM 3 propagation + occlusion-informed chunking) into one arm, so a win for B doesn't tell us *which* ingredient drove it. To make the ablation crisp, SAM 3 with fixed 60 s chunking is added as a new arm and the variant labels are renamed accordingly: **A (YOLO+BoT-SORT) — B (SAM 3 + fixed chunking) — C (SAM 3 + adaptive chunking)**. Variant B costs one extra ~2.5 GPU-hour run; the rest of the pipeline (annotation, GT export, evaluation) is unchanged.

Day-37 scan results are retained on disk (`ext-data/output/results/sam3-hf/`) and the old candidate manifest is archived at `tracking_eval/video_manifest_day_37_superseded.csv` for possible future use, but day 37 is **out of scope** for this evaluation.

## Refinements vs. previous draft (`plan.md`)

1. **Drop the "boundary precision" annotation step.** Chunking quality is now evaluated by its downstream effect on HOTA/IDF1, using the same ground truth as the main tracking eval. Removes a non-standard metric, removes a second annotation pass, and aligns the entire study with the metrics Reviewer 1 asked for.
2. **Add a YOLO+BoT-SORT baseline tracker.** The existing `yolo_scan.py` already emits a `yolo_tracking.parquet` with `frame, track_id, bbox, confidence` (`src/tracker/scan.py:442–454`). Evaluating it as a baseline is essentially free and directly counters "just integrating existing components".
3. **Sweep is cheaper than the previous draft assumed.** `script/compute_chunk_boundaries.py` re-derives chunk boundaries from cached YOLO output without re-running YOLO. Each sweep config only re-runs SAM3 tracking, not the full pipeline. *(Note 2026-05-12: the sweep is deferred under the revision deadline — see refinement 6.)*
4. **Phase ordering**: annotation moves to Phase 1 (the bottleneck and foundation); chunk-guided frame selection still uses default-parameter chunking, but the chunking *evaluation* moves after annotation, since it now needs the GT.
5. **Switch eval set from day 37 to in-distribution day 29 (2026-05-11).** Day 37 birds are visually distinct (older plumage) and less active; using them would understate the tracker's difficulty on the data the classifier was actually trained and evaluated on. Day-29 selection draws one video per cage (hardest group per cage by YOLO-scan-based difficulty ranking), which aligns with the LOCO splitter's natural per-fold test partitions.
6. **Sparse-keyframe evaluation (2026-05-12).** Dropped the dense-interpolated-GT assumption. Metrics are computed only at the ~88 human-verified keyframes per video. Removes a QA-scrubbing pass (~3–4 hours per video) and removes measurement noise from linearly-interpolated bboxes on non-linearly-moving birds. Matches TAO and multi-animal pose-tracking conventions; metric definitions are unchanged. See Phase 1.3 / 1.4 / 4 below.
7. **Drop Phase 3 sweep (2026-05-12).** Revision deadline 2026-05-21 makes the 7-config sweep infeasible. Manuscript notes the deferred sweep in limitations.
8. **Add SAM 3 fixed-chunking arm and re-letter variants (2026-05-18).** The earlier two-way A-vs-B comparison made a method-vs-baseline point but left the source of the lift ambiguous — SAM 3 alone might do the work, with the occlusion-informed chunker contributing nothing. Adding a SAM-3-with-fixed-chunking arm turns the experiment into a proper ablation. Variant labels are re-lettered so that A → B → C corresponds to a monotonic increase in method components (YOLO only → +SAM 3 → +adaptive chunking). Script-level rename of `B_sam3_default` → `C_sam3_adaptive` and addition of `B_sam3_fixed` are reflected in `convert_predictions.py` and `evaluate_tracking.py`.

## Timeline

**Hard deadline:** 2026-05-21 (CV4Animals revision). Effort estimate after 2026-05-12 scope cuts: ~11 person-hours + ~2.5 GPU-hours. See "Effort estimate" near the end of this document.

**Pipeline version freeze:** Variant C uses the identical `config/tracker.yaml` defaults shipped with the manuscript's days-28–29 runs — no re-tuning. Variant B uses the same config with `use_adaptive_chunking: false` to disable YOLO-scan boundary refinement; everything else (chunk size, SAM 3 weights, prompt strategy) is unchanged.

**Annotation labour:** single annotator. Inter-annotator agreement is out of scope for this revision.

---

## Phase 1: Video selection & ground-truth annotation

Annotation is the bottleneck. Everything else depends on it.

### 1.1 Select videos
**5 videos**, one per cage (C1–C5), choosing the hardest group per cage by YOLO-scan-based difficulty ranking computed **jointly across days 28 and 29** so day-28 and day-29 difficulty scores are directly comparable. The ranking sums per-video ranks (1–30) across 7 difficulty proxies derived from `yolo_scan_metrics.parquet`:

- `frac_high_occlusion` (↑ harder)
- `mean_overlapping_pairs` (↑ harder)
- `mean_pairwise_bbox_iou` (↑ harder)
- `frac_object_count_change` (↑ harder)
- `num_occlusion_periods_per_min` (↑ harder)
- `mean_centroid_distance` (↓ harder; ranked inverted)
- `mean_separation_score` (↓ harder; ranked inverted)

Composite range: 7–210 across 30 candidates. Ranking script: `tracking_eval/manifest.py` (invoked as `python -m tracking_eval build-manifest`; promoted from the original `tmp/rank_videos_both_days.py` one-shot for reproducibility; discovers scan dirs under `ext-data/output/results/sam3-hf/`, computes the 7 proxies from each video's `metrics/yolo_scan_{metrics,summary}.parquet`, and writes the manifest deterministically). At least 2 of the 5 picks are required to come from day 28; the joint ranking naturally produces 2 day-28 picks without forcing (the script asserts this via `--min-day-28`).

**Selected (manifest: `tracking_eval/video_manifest.csv`):**

| Cage | Group | Day | Video stem | Rank-sum | Notes |
|------|-------|-----|------------|----------|-------|
| C1 | G2 | 29 | `C1G2_Test_2_day_29_2_Camera_5_2025_02_05_11_06_07_2` | 160 | hardest in C1 |
| C2 | G2 | 28 | `C2G2_Test_1_day_28_1_Camera_5_2025_02_04_09_43_25_2` | **186** | overall hardest, hardest in C2 |
| C3 | G2 | 29 | `C3G2_Test_2_day_29_2_Camera_5_2025_02_05_09_22_32_2` | 183 | hardest in C3 |
| C4 | G2 | 29 | `C4G2_Test_2_day_29_2_Camera_5_2025_02_05_10_31_53_2` | 158 | hardest in C4 |
| C5 | G3 | 28 | `C5G3_Test_1_day_28_1_Camera_8_2025_02_04_11_35_00_3` | 176 | hardest in C5 |

Source scan dirs (resolved per video in `video_manifest.csv` → `scan_dir`): four scan runs cover the 30 candidates — `20260317_162056_sam3_hf` (14 of the day-29 videos), `20260319_152425_sam3_hf` (C2G2 day 29), `20260308_230835_sam3_hf` and `20260309_230105_sam3_hf` (day-28 videos; latest scan run is used when duplicates exist).

**LOCO alignment**: Each selected video belongs to its respective cage's LOCO test fold (e.g., C1G2 is in fold 0's test set when C1 is held out). This means tracking metrics are reported on data that was *truly held out* for the classifier in the paper's LOCO results. Cross-day spread (2 day-28 + 3 day-29) matches the manuscript's training distribution.

### 1.2 Chunk-guided frame selection
Run **default-parameter chunking only** (no SAM3) on the 5 selected videos using existing infrastructure. For each video, build the annotation frame list:

- **Chunk-guided** (≈60%): for each detected boundary, sample frames at `boundary - 5`, `boundary`, `boundary + 5`.
- **Uniform backbone** (≈40%): one frame every 15 s. Guards against blind spots in the chunking heuristic.

**Target**: ~80 frames/video × 5 videos = ~400 frames, ~1200 bboxes (3 birds × 400 frames).

Script: `tracking_eval/frame_selection.py` (invoked as `python -m tracking_eval select-frames`). Output: `tracking_eval/annotation_frames.csv` with columns `video_id, frame_idx, source` (chunk_guided | occlusion_bracketing | uniform).

### 1.3 Annotation tool & protocol
**CVAT** (Docker, local). Project name: `playclass-tracking-eval`. Single label `bird`; bird identity is captured by Track membership (3 Tracks per video), not by label.

**Track mode is used** but its linear interpolation is treated as a navigation aid only — interpolated frames are **not** treated as ground truth (filtered out at export; see Phase 1.4). The annotator visits only the frames listed in `annotation_frames.csv` for that video (88 frames × 3 birds), draws / adjusts the bbox at each listed frame following the conventions in `annotation_guidelines.md` (full estimated bbox on partial occlusion, Outside flag on total occlusion, single Track per bird), and does **not** add corrective keyframes to fix interpolation drift between listed frames.

Bbox-only annotation (not masks): MOTChallenge metrics need bboxes only, ~5× faster than masks. Predicted masks from SAM3 will be converted to bboxes for evaluation.

QA is per-keyframe-only: at each of the 88 listed keyframes per video, confirm the bbox is on the correct bird with a tight fit. Drift between keyframes is not a QA concern.

Documentation: `tracking_eval/annotation_guidelines.md` (already in repo; revised 2026-05-12 alongside this plan).

### 1.4 Export to sparse GT
1. From CVAT, take a **project Backup** export (not the per-task MOT 1.1 export). The Backup includes each task's `annotations.json` in CVAT's native schema, which stores only human-drawn keyframes — no temporal interpolation is materialised.
2. Run `python -m tracking_eval cvat-to-mot` (module: `tracking_eval/cvat_to_mot.py`) against the backup root at `ext-data/output/results/tracker_benchmark/cvat_backup/playclass-tracking-eval/`. For each task it reads `annotations.json`, assigns track ids 1–3 in CVAT track-declaration order, drops shapes with `outside=true`, and writes `<video_id>.txt` in MOTChallenge 1.1 format. Row format: `frame, id, bb_left, bb_top, bb_width, bb_height, conf, -1, -1, -1` with `conf = 1` for GT. The `--keyframes-csv tracking_eval/annotation_frames.csv` default additionally reports per-bird scheduled / drawn / missing / extra counts.
3. Output: `ext-data/output/results/tracker_benchmark/ground_truth/<video_id>.txt`.

This replaces the originally-planned `filter_mot_to_keyframes.py` step (which would have filtered an MOT export against the scheduled-keyframe set) — going through the Backup avoids materialising CVAT's linear interpolation in the first place, so there's nothing to filter out.

---

## Phase 2: Tracker variants

Three trackers are run on the 5 annotated videos; all three share the same sparse ground truth.

| Variant | Description | Source | Status |
|---------|-------------|--------|--------|
| **A: YOLO + BoT-SORT** | Generic detector + tracker baseline. Isolates the scan stage. | Already produced by `src/tracker/scan.py:442` as `yolo_tracking.parquet`. No re-run needed. | **done** |
| **B: SAM 3 + fixed chunking** | SAM 3 propagation, uniform 60 s chunks, no YOLO scan, no boundary refinement. Isolates the segmentation stage. | Run `script/run_tracker.py` with `config/tracker.yaml` overridden to set `use_adaptive_chunking: false`. | **TODO** (added 2026-05-18) |
| **C: SAM 3 + adaptive chunking** | Full method. Boundaries shifted within a ±10 s window toward high-separation, low-occlusion frames using the YOLO-scan signal. | Run `script/run_tracker.py` with `config/tracker.yaml` defaults. | **done** |

**Critical**: all three variants run **fully automated**, no manual ID correction. This is the headline reproducibility claim.

Outputs: `ext-data/output/results/tracker_benchmark/predictions/{A_yolo_botsort,B_sam3_fixed,C_sam3_adaptive}/<video_id>.parquet`.

> **Tuned-SAM3 variant (Phase 3 sweep) is dropped (2026-05-12).** Original plan included a 7-config chunking hyperparameter sweep feeding a tuned variant. Under the 2026-05-21 deadline this is infeasible. The deferred sweep is noted in the manuscript's limitations section.

---

## Phase 4: Tracking evaluation

### 4.1 Convert predictions → MOTChallenge
Module: `tracking_eval/predictions.py` (invoked as `python -m tracking_eval convert-preds`). Reads each Parquet, extracts bbox per (frame, track_id), writes `<video_id>.txt` in MOTChallenge format. Run for **Variants A, B, and C**. The script takes `--predictions-root` (default `ext-data/output/results/tracker_benchmark/tracker_outputs/`; supplies A and C from the adaptive run dir) and `--predictions-root-fixed` (default `ext-data/output/results/tracker_benchmark/tracker_outputs_fixed/`; supplies B; silently skipped if dir is missing or empty).

Predictions stay **dense per-frame** (predicted for every frame). Do not filter predictions to match GT keyframes — `motmetrics` aligns automatically by only updating its accumulator on frames where GT exists. Pre-filtering predictions would hide false positives on non-keyframe frames and distort detection metrics.

### 4.2 Compute metrics
**Libraries**: [`py-motmetrics 1.4.0`](https://github.com/cheind/py-motmetrics) for IDF1/MOTA/MOTP/ID-switches/precision/recall, and **TrackEval** (vendored at `ext/TrackEval`, commit `12c8791`; canonical HOTA implementation of Luiten et al. 2021) for HOTA / DetA / AssA / LocA averaged over the standard $\alpha \in \{0.05, 0.10, \ldots, 0.95\}$ sweep. Using both side by side avoids the known IDF1/HOTA-implementation drift between metric libraries.

Metrics (priority order):
1. **HOTA** (Reviewer 1 explicitly asked) and its components **DetA**, **AssA**
2. **IDF1** (Reviewer 1 explicitly asked)
3. **MOTA** (community standard)
4. **ID switches** (raw count)

Bbox match threshold: IoU ≥ 0.5 (MOTChallenge default).

Module: `tracking_eval/evaluate.py` (invoked as `python -m tracking_eval evaluate`). The `VARIANTS` tuple is `("A_yolo_botsort", "B_sam3_fixed", "C_sam3_adaptive")`; variants whose prediction MOT files are not yet on disk are skipped with a warning, so partial / incremental re-runs are safe while Variant B is being produced.

Pseudocode:

```
For each tracker variant {A_yolo_botsort, B_sam3_fixed, C_sam3_adaptive}:
    For each video_id:
        gt = load_sparse_gt(video_id)           # ~88 frames × 3 birds
        pred = load_dense_predictions(variant, video_id)
        acc = motmetrics.MOTAccumulator()
        te_data = build_trackeval_sequence(gt, pred)   # GT-present frames only
        for frame in sorted(set(gt.frame)):
            match(gt[frame], pred[frame], iou_threshold=0.5)
            acc.update(...)
        compute IDF1 / MOTA / IDsw on acc; compute HOTA / DetA / AssA on te_data
    Aggregate per-cage and overall (combine_sequences for HOTA; compute_many for motmetrics)
```

### 4.3 Stratified reporting
- Per-video (variance)
- Per-cage (cross-environment consistency, mirrors LOCO design)
- Aggregate (frame-weighted)
- **All three variants side-by-side**

Outputs: `tracking_eval/results/metrics_{per_video,per_cage,aggregate}.csv`. (Already written for A and C; will be regenerated to include B once that variant runs.)

---

## Phase 5: Manuscript integration

### 5.1 New tables / figures
- **Table — Tracking metrics (three-way ablation)**: HOTA, IDF1, MOTA, IDsw, DetA, AssA × {Variant A: YOLO + BoT-SORT, Variant B: SAM 3 + fixed chunking, Variant C: SAM 3 + adaptive chunking} × {aggregate, per-cage}.
- No chunking-sweep table or figure (Phase 3 deferred).

### 5.2 Discussion / framing changes in §3.2 and §5
- Drop the "manual post-processing required" claim from §3.2 — replace with "automated chunking strategy quantified on a 5-video held-out evaluation subset (one video per cage, hardest group per cage); the original 30 videos used the same automated tracker, with manual identity correction restricted to chunk-boundary ID switches and protocol-ID assignment, and were not re-tracked for this evaluation".
- Add the sparse-keyframe-evaluation paragraph (see appendix at end of this document) and the three-way ablation framing to §3.2 where the tracker evaluation is first introduced. The full method-section draft for this is at `tmp/tracker_evaluation_methods.md`.
- Add to the limitations section:
  - the one-sentence blind-spot caveat about inter-keyframe self-correcting ID switches not being observable under sparse evaluation;
  - the deferred chunking hyperparameter sweep — reported SAM 3 numbers use the manuscript's existing default chunking parameters and are not an upper bound for this architecture.
- Address Reviewer 1 directly:
  - "Lack of HOTA/IDF1" → reported in the new three-way table.
  - "Manual post-processing" → the eval subset is tracked without it; all three variants are scored end-to-end from raw model output.
  - "Robustness to occlusion / domain variation" → per-cage stratification (5 cages) + per-cage hardest-group selection deliberately stresses occlusion-heavy conditions; the keyframe schedule densifies sampling at the three longest occlusion periods per video.
  - "Limited novelty / just combining components" → the ablation directly tests whether the proposed combination (YOLO scan + SAM 3 + occlusion-informed chunking) outperforms each constituent component run alone. Variants A and B isolate the two ingredients; Variant C is the full method. If $\text{HOTA}(C) > \max(\text{HOTA}(A), \text{HOTA}(B))$, the lift is attributable to the synergy, not to either component.

---

## Directory structure

Heavy outputs (tracking parquets, MOT files, ground truth) live under `ext-data/output/results/tracker_benchmark/`, written directly through the `ext-data → /mnt/birds/rebecca2025/` symlink (no DVC). Scripts, manifests, and small result CSVs stay in the repo. The repo branch `tracker_benchmark` is already checked out.

```
ext-data/output/results/tracker_benchmark/        # On the mounted drive, not version-controlled
├── cvat_backup/
│   ├── playclass-tracking-eval/                  # Extracted CVAT project Backup (project.json + task_*/)
│   └── playclass-tracking-eval.zip               # Kept for re-import to CVAT
├── source_videos/                                # 5 MP4 clips fed to CVAT (provenance)
├── tracker_outputs_adaptive/                     # SAM 3 adaptive-chunking + YOLO scan run dirs per video (Variant C)
├── tracker_outputs_fixed/                        # SAM 3 fixed-chunking run dirs per video (Variant B) — TODO
├── ground_truth/                                 # MOTChallenge .txt per video (sparse, ~88 frames × 3 birds)
└── predictions_mot/                              # A_yolo_botsort / B_sam3_fixed / C_sam3_adaptive .txt files

tracking_eval/                                    # In repo, version-controlled — Python package
├── __init__.py                                   # empty package marker
├── __main__.py                                   # CLI dispatcher (subcommands + prepare/score umbrellas)
├── paths.py                                      # single source of truth for all paths
├── manifest.py                                   # YOLO-scan-driven hardest-group-per-cage ranking
├── frame_selection.py                            # keyframe scheduler for CVAT
├── cvat_to_mot.py                                # CVAT Backup → sparse MOT GT
├── predictions.py                                # tracker parquets → MOT predictions (A/B/C)
├── evaluate.py                                   # sparse-keyframe MOT metrics (motmetrics + TrackEval HOTA)
├── PLAN.md / annotation_guidelines.md
├── video_manifest.csv
├── annotation_frames.csv / annotation_frames_summary.csv
└── results/
    ├── metrics_per_video.csv
    ├── metrics_per_cage.csv
    └── metrics_aggregate.csv
```

*(Removed under 2026-05-12 scope cuts: `predictions/C_sam3_tuned/`, `chunking_sweep/`, `tracking_eval/chunking/`, `tracking_eval/scripts/sweep_chunking.py`. Renamed under 2026-05-18 ablation refactor: `predictions/B_sam3_default/` → `predictions/C_sam3_adaptive/`. Superseded by the CVAT-Backup path: `filter_mot_to_keyframes.py` was never needed since the Backup's `annotations.json` already contains only human-drawn keyframes. Consolidated under 2026-05-18 pipeline refactor: the five standalone scripts in `tracking_eval/scripts/` were renamed and moved up one level, exposed via a single `python -m tracking_eval` CLI; runtime data migrated from `tmp/tracker_benchmark/` to `ext-data/output/results/tracker_benchmark/`.)*

---

## Critical existing code to reuse

| File | Purpose |
|------|---------|
| [src/tracker/chunking.py:64](src/tracker/chunking.py#L64) | `chunk_video_frames_adaptive()` |
| [src/tracker/scan.py:442](src/tracker/scan.py#L442) | YOLO+BoT-SORT scan → `yolo_tracking.parquet` |
| [src/metrics.py:373](src/metrics.py#L373) | `compute_separation_score` (parameter semantics) |
| [script/compute_chunk_boundaries.py:145](script/compute_chunk_boundaries.py#L145) | Sweep entry point — recomputes boundaries without re-running YOLO |
| [config/tracker_scan.yaml:81-89](config/tracker_scan.yaml#L81-L89) | Default chunking parameter values |
| [config/tracker.yaml](config/tracker.yaml) | Default tracking config (use unchanged for Variant B) |

---

## Effort estimate

Revised 2026-05-12 after sparse-eval and Variant-C scope cuts.

| Task | Hours | Notes |
|------|-------|-------|
| Video selection + path resolution | done | `video_manifest.csv` |
| Chunk-guided frame selection script | done | `annotation_frames.csv` |
| CVAT setup + annotation guideline | done | `annotation_guidelines.md` |
| Ground truth annotation (sparse, 88 keyframes × 3 birds × 5 videos ≈ 1,320 manual bboxes) | done | |
| `cvat_backup_to_mot.py` (CVAT Backup → sparse MOT GT) | done | Replaces the originally-planned `filter_mot_to_keyframes.py`. |
| Run Variant A (already cached from chunking pipeline) | done | `yolo_tracking.parquet` from existing scan runs |
| Run Variant C — SAM 3 + adaptive chunking (5 videos × ~30 min GPU) | done | 2.5 GPU-h spent |
| **Run Variant B — SAM 3 + fixed chunking** (5 videos × ~30 min GPU) | **TODO** | ~2.5 GPU-h. Added 2026-05-18 for the three-way ablation. |
| `convert_predictions.py` (incl. `--predictions-root-fixed` for Variant B) | done | |
| `evaluate_tracking.py` + metric computation (motmetrics + TrackEval HOTA) | done for A and C | Re-run once Variant B parquets exist (script skips missing variants). |
| Manuscript integration | ~3 | Three-way ablation framing, paragraph drop-in (`tmp/tracker_evaluation_methods.md`). |
| ~~Sweep on 2 videos × 7 configs~~ | ~~7 GPU-h~~ | Dropped (Phase 3 deferred). |
| ~~Run tuned-SAM3 variant on 5 videos~~ | ~~2.5 GPU-h~~ | Dropped. |
| **Remaining (as of 2026-05-18)** | **~3 person-hours + ~2.5 GPU-hours** | Variant B run + eval re-run + manuscript write-up. |

---

## Verification (end-to-end smoke test)

1. **Filter correctness**: after `filter_mot_to_keyframes.py` runs on one hand-checked video, confirm row count = `(n_keyframes × 3) − (Outside-flagged count)`. Visually overlay the filtered GT on the video for one frame; confirm bbox alignment and ID consistency.
2. **Conversion correctness**: pick one (video, frame, track_id) from a Parquet prediction, manually compute bbox from the row, compare to the corresponding line in the converted MOT .txt.
3. **Degenerate metric check**: feed identical sparse GT as both GT and prediction into `evaluate_tracking.py` → all metrics report perfect scores.
4. **Shift check**: shift predictions by +N frames → IDF1 / HOTA degrade predictably. With sparse GT, gaps between GT-present frames are large, so this is a *stronger* identity test than the dense-GT equivalent.
5. **Plausibility check**: aggregate HOTA for proposed pipeline should land in the 50–80 range typical of difficult multi-animal MOT; if it's >90 the GT and predictions may be miscompared, if <40 something is wrong with detection-side conversion.

---

## Appendix: Sparse-evaluation paragraph and verified citations

Drop this paragraph into §3.2 (methods) or §5 (limitations/methodology) of the manuscript revision, where the tracker evaluation is first introduced. Citation keys assume the manuscript's existing BibTeX convention — substitute as needed.

> Tracker evaluation follows the methodological convention established for long-form multi-object tracking and multi-animal behaviour analysis, in which the cost of dense per-frame ground truth is prohibitive. We evaluate at sparse human-verified keyframes (88 frames per video, biased toward chunk-boundary intervals that bracket the three longest occlusion periods identified by a preliminary YOLO scan; see Phase 1.2 of the evaluation plan). Standard MOT metrics — MOTA [Bernardin and Stiefelhagen, 2008], IDF1 [Ristani et al., 2016], and HOTA with its DetA / AssA decomposition [Luiten et al., 2021] — are defined over GT-present frames and impose no assumption of dense annotation. This matches the sparse-keyframe annotation strategy of the TAO long-tail tracking benchmark [Dave et al., 2020], which annotates at 1 FPS on 30 FPS video, and is the de facto convention in multi-animal pose-tracking studies including multi-animal DeepLabCut [Lauer et al., 2022], SLEAP [Pereira et al., 2022], and idtracker.ai [Romero-Ferrero et al., 2019]. We acknowledge one limitation: identity switches that occur and self-correct entirely within a single inter-keyframe interval are not observable under sparse evaluation. The chunk-boundary keyframe schedule, which densifies sampling around the three longest occlusion periods per video, reduces but does not eliminate this blind spot.

**Verified citations (DOIs / arXiv IDs):**

| Cite | Reference | DOI / arXiv |
|---|---|---|
| Bernardin & Stiefelhagen 2008 | CLEAR MOT metrics, EURASIP J. Image Video Process. 2008:246309 | 10.1155/2008/246309 |
| Ristani et al. 2016 | Performance Measures and a Data Set for Multi-target Multi-camera Tracking, ECCV 2016 Workshops | arXiv:1609.01775 |
| Luiten et al. 2021 | HOTA, IJCV 129:548–578 | 10.1007/s11263-020-01375-2 |
| Dave et al. 2020 | TAO benchmark, ECCV 2020 | arXiv:2005.10356 |
| Lauer et al. 2022 | Multi-animal DLC, Nat. Methods 19:496–504 | 10.1038/s41592-022-01443-0 |
| Pereira et al. 2022 | SLEAP, Nat. Methods 19:486–495 | 10.1038/s41592-022-01426-1 |
| Romero-Ferrero et al. 2019 | idtracker.ai, Nat. Methods | 10.1038/s41592-018-0295-5 |
