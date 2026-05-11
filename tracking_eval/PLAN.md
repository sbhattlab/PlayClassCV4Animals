# Refined Plan: Tracking Evaluation for CVPR 2026 Workshop Revision

## Context

PlayClass was accepted at CV4Animals (CVPR 2026 workshop) as a poster, with revisions. This plan addresses **Reviewer 1's tracking-related concerns only**:

- "Lack of standard tracking evaluation metrics (e.g., HOTA, IDF1) to validate identity preservation"
- "Dependence on manual post-processing in the tracking pipeline, reducing automation and reproducibility"
- "Limited analysis of robustness to tracking errors, occlusions, and domain variation"
- "Limited novelty, with the method primarily combining existing tracking, feature extraction, and classification components"

**Reviewer 2's concerns** (ethogram description, "end-to-end" terminology, precision/recall + class-level metrics, dataset release plan, fine-grained granularity exploration) are **out of scope** for this plan and will be handled separately.

**Strategy** (revised 2026-05-11): Use **in-distribution videos from days 28 and 29** as the evaluation set, one per cage, selecting the hardest group per cage via YOLO-scan-based difficulty ranking across both days jointly. At least 2 of the 5 picks are required to come from day 28 to ensure cross-day representativeness of the manuscript's training distribution. The day-37 external-validation approach was dropped after visual inspection showed day-37 birds look markedly different (older plumage) and move significantly less than the days 28–29 birds the manuscript's classification results are built on — benchmarking against day 37 would inflate tracker metrics relative to the actual training/eval distribution. The 5 LOCO folds in the manuscript already produce *disjoint* test partitions (each fold tests one full cage); drawing one video per cage from these natural per-fold test sets gives a 5-video subset that is in-distribution, cross-environment, and genuinely held out for the classifier. Run the SAM3 + adaptive chunking pipeline **fully automated (no manual post-processing)**, alongside the YOLO+BoT-SORT reference tracker, on these 5 bbox-annotated videos. Report standard MOT metrics. This addresses missing metrics, the automation concern, and — by including a generic tracker baseline — the "limited novelty" framing.

Day-37 scan results are retained on disk (`ext-data/output/results/sam3-hf/`) and the old candidate manifest is archived at `tracking_eval/video_manifest_day_37_superseded.csv` for possible future use, but day 37 is **out of scope** for this evaluation.

## Refinements vs. previous draft (`plan.md`)

1. **Drop the "boundary precision" annotation step.** Chunking quality is now evaluated by its downstream effect on HOTA/IDF1, using the same ground truth as the main tracking eval. Removes a non-standard metric, removes a second annotation pass, and aligns the entire study with the metrics Reviewer 1 asked for.
2. **Add a YOLO+BoT-SORT baseline tracker.** The existing `yolo_scan.py` already emits a `yolo_tracking.parquet` with `frame, track_id, bbox, confidence` (`src/tracker/scan.py:442–454`). Evaluating it as a baseline is essentially free and directly counters "just integrating existing components".
3. **Sweep is cheaper than the previous draft assumed.** `script/compute_chunk_boundaries.py` re-derives chunk boundaries from cached YOLO output without re-running YOLO. Each sweep config only re-runs SAM3 tracking, not the full pipeline.
4. **Phase ordering**: annotation moves to Phase 1 (the bottleneck and foundation); chunk-guided frame selection still uses default-parameter chunking, but the chunking *evaluation* moves after annotation, since it now needs the GT.
5. **Switch eval set from day 37 to in-distribution day 29 (2026-05-11).** Day 37 birds are visually distinct (older plumage) and less active; using them would understate the tracker's difficulty on the data the classifier was actually trained and evaluated on. Day-29 selection draws one video per cage (hardest group per cage by YOLO-scan-based difficulty ranking), which aligns with the LOCO splitter's natural per-fold test partitions.

## Open items (user input needed before kickoff)

| # | Item | Why it matters |
|---|------|----------------|
| 1 | **GPU/time budget** | Determines sweep granularity (3 vs 5 values per param) |
| 2 | **Pipeline version freeze** | Recommend: identical config to days 28–29 paper, for a fair comparison |
| 3 | **Annotation labour** | Single annotator vs. second-pass agreement on a 30-frame subset |
| 4 | **No remaining caveat** | After joint-day ranking the 5 picks have rank-sums 158–186 (all in the top ~15% of 30 candidates). The earlier C5 weak-link issue is resolved by swapping to C5G3 day-28. |

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

Composite range: 7–210 across 30 candidates. Ranking script: `tmp/rank_videos_both_days.py` (one-shot, can be promoted to `tracking_eval/scripts/` if needed for reproducibility). At least 2 of the 5 picks are required to come from day 28; the joint ranking naturally produces 2 day-28 picks without forcing.

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

Script to write: `tracking_eval/scripts/select_annotation_frames.py`. Output: `tracking_eval/annotation_frames.csv` with columns `video_id, frame_idx, source` (chunk_guided | uniform).

### 1.3 Annotation tool & protocol
**CVAT** (Docker, local) with **linear interpolation** between keyframes. Project name: `playclass-tracking-eval`. Labels are bird identity integers (consistent within a video).

Bbox-only annotation (not masks): MOTChallenge metrics need bboxes only, ~5× faster than masks. Predicted masks from SAM3 will be converted to bboxes for evaluation.

Document a one-page guideline: partial-occlusion handling (annotate full estimated bbox, MOTChallenge convention), minimum visibility, ID re-entry policy. Save as `tracking_eval/annotation_guidelines.md`.

### 1.4 Export
Export from CVAT in **MOTChallenge format** to `ext-data/output/results/tracker_benchmark/ground_truth/<video_id>.txt`.

---

## Phase 2: Tracker variants

Three trackers are run on the 5 annotated videos; all share the same ground truth.

| Variant | Description | Source |
|---------|-------------|--------|
| **A: YOLO+BoT-SORT** | Generic detector + tracker baseline | Already produced by `src/tracker/scan.py:442` as `yolo_tracking.parquet`. No re-run needed |
| **B: SAM3 (default chunking)** | Proposed pipeline at manuscript defaults | Run `script/run_tracker.py` once per video with `config/tracker.yaml` |
| **C: SAM3 (tuned chunking)** | Proposed pipeline at sweep-best params | Determined by Phase 3 |

**Critical**: all three variants run **fully automated**, no manual ID correction. This is the headline reproducibility claim.

Outputs: `ext-data/output/results/tracker_benchmark/predictions/{A_yolo_botsort,B_sam3_default,C_sam3_tuned}/<video_id>.parquet`.

---

## Phase 3: Chunking hyperparameter sweep

The defaults verified from `config/tracker_scan.yaml` (lines 81/85/89):

| Parameter | Default | Sweep values |
|-----------|---------|--------------|
| `occlusion_iou_threshold` | 0.07 | {0.04, 0.07, 0.10} |
| `separation_min_distance` | 0.10 | {0.07, 0.10, 0.13} |
| `clustering_distance_threshold` | 0.15 | {0.10, 0.15, 0.20} |

One-at-a-time sweep: 3 params × 3 values − 2 (default counted once) = **7 configs**. The separation score formula is in `src/metrics.py:373–406`; hard gates on overlap and minimum centroid distance dominate, so wide ranges are unlikely to be informative.

### 3.1 Sweep workflow (cheap)
For each config:
1. Reuse cached `yolo_tracking.parquet` from Phase 2 Variant A.
2. Re-run `script/compute_chunk_boundaries.py` with the new params → new boundary list. *(YOLO is not re-run.)*
3. Re-run SAM3 tracking with the new chunk schedule.
4. Convert to MOTChallenge format.
5. Score against ground truth.

Script to write: `tracking_eval/scripts/sweep_chunking.py` orchestrates 1–4.

To bound GPU cost: run the sweep on **2 of the 5 videos** (one easy, one hard cage), pick the best config, then re-run Phase 2 Variant C on the full 5 with the chosen config. This validates the choice without 7× the tracking cost.

### 3.2 Outcome and framing
- **Outcome A** (defaults within ≤2 HOTA points of best): present as **sensitivity analysis**. Variant C = Variant B; the tracking table reports two columns (A baseline, B proposed).
- **Outcome B** (tuned config >2 HOTA points better): present as **hyperparameter study**. Variant C is the tuned config; the tracking table reports three columns (A, B, C). The main paper's classification results are **not** retroactively rerun; they used default parameters and the classifier was trained on those tracks. The tuned setting is presented as prospective improvement validated on the 5-video evaluation subset.

Record the outcome in `tracking_eval/chunking/README.md`.

---

## Phase 4: Tracking evaluation

### 4.1 Convert predictions → MOTChallenge
Script: `tracking_eval/scripts/convert_predictions.py`. Reads each Parquet, extracts bbox per (frame, track_id), writes `<video_id>.txt` in MOTChallenge format. Run for variants A, B, and C.

### 4.2 Compute metrics
**Library**: [`motmetrics`](https://github.com/cheind/py-motmetrics) (lighter than TrackEval; pure-Python; well-suited to small evals). TrackEval is the canonical alternative if HOTA fidelity matters more.

Metrics (priority order):
1. **HOTA** (Reviewer 1 explicitly asked) and its components **DetA**, **AssA**
2. **IDF1** (Reviewer 1 explicitly asked)
3. **MOTA** (community standard)
4. **ID switches** (raw count)

Bbox match threshold: IoU ≥ 0.5 (MOTChallenge default).

Script: `tracking_eval/scripts/evaluate_tracking.py`.

### 4.3 Stratified reporting
- Per-video (variance)
- Per-cage (cross-environment consistency, mirrors LOCO design)
- Aggregate (frame-weighted)
- All three trackers side-by-side

Outputs: `tracking_eval/results/metrics_{per_video,per_cage,aggregate}.csv`.

---

## Phase 5: Manuscript integration

### 5.1 New tables / figures
- **Table — Tracking metrics**: HOTA, IDF1, MOTA, IDsw, DetA, AssA × {YOLO+BoT-SORT, SAM3 default, SAM3 tuned (if Outcome B)} × {aggregate, per-cage}.
- **Table or small figure — Chunking sensitivity** (Outcome A) or **chunking hyperparameter study** (Outcome B).

### 5.2 Discussion / framing changes in §3.2 and §5
- Drop the "manual post-processing required" claim from §3.2 — replace with "automated chunking strategy quantified on a 5-video held-out evaluation subset (one video per cage, hardest group per cage); the original 30 videos used the same automated tracker, with manual identity correction restricted to chunk-boundary ID switches and protocol-ID assignment, and were not re-tracked for this evaluation".
- Add: held-out evaluation subset, fully automated re-run with no manual post-processing, two- or three-way tracker comparison.
- Address Reviewer 1 directly:
  - "Lack of HOTA/IDF1" → reported in new table.
  - "Manual post-processing" → the eval subset is tracked without it.
  - "Robustness to occlusion / domain variation" → per-cage stratification (5 cages) + per-cage hardest-group selection deliberately stresses occlusion-heavy conditions.
  - "Limited novelty / just combining components" → quantified lift over YOLO+BoT-SORT baseline justifies the SAM3 + adaptive-chunking design.

---

## Directory structure

Heavy outputs (tracking parquets, MOT files, ground truth) live under `ext-data/output/results/tracker_benchmark/`, written directly through the `ext-data → /mnt/birds/rebecca2025/` symlink (no DVC). Scripts, manifests, and small result CSVs stay in the repo. The repo branch `tracker_benchmark` is already checked out.

```
ext-data/output/results/tracker_benchmark/        # On the mounted drive, not version-controlled
├── ground_truth/                                 # MOTChallenge .txt per video
├── predictions/
│   ├── A_yolo_botsort/                           # Parquets (or symlinks to existing yolo_scan run)
│   ├── B_sam3_default/                           # SAM3 run dirs (timestamped)
│   └── C_sam3_tuned/                             # SAM3 run dirs (timestamped, only if Outcome B)
├── predictions_mot/                              # Same A/B/C split, .txt format
└── chunking_sweep/                               # Per-config chunk schedules + SAM3 outputs

tracking_eval/                                    # In repo, version-controlled
├── video_manifest.csv
├── annotation_guidelines.md
├── annotation_frames.csv
├── chunking/
│   ├── README.md                                 # Outcome A or B + selected params
│   └── sweep_results.csv                         # HOTA/IDF1 per config × video
├── scripts/
│   ├── select_annotation_frames.py
│   ├── sweep_chunking.py
│   ├── convert_predictions.py
│   └── evaluate_tracking.py
└── results/
    ├── metrics_per_video.csv
    ├── metrics_per_cage.csv
    └── metrics_aggregate.csv
```

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

| Task | Hours |
|------|-------|
| Video selection + path resolution | 1 |
| Chunk-guided frame selection script | 2 |
| CVAT setup + annotation guideline | 2 |
| Ground truth annotation (≈400 frames × 3 birds w/ interpolation) | 6–10 |
| Run Variant A (already cached from chunking pipeline) | 0 |
| Run Variant B SAM3 default (5 videos × ~30 min GPU) | 2.5 GPU-h |
| Sweep on 2 videos × 7 configs (~30 min/run) | 7 GPU-h |
| Run Variant C on 5 videos (if Outcome B) | 2.5 GPU-h |
| Prediction conversion script | 2 |
| Evaluation script + metric computation | 3 |
| Manuscript integration | 4 |
| **Total** | **~25 person-hours + ~12 GPU-hours** |

---

## Verification (end-to-end smoke test)

1. **Annotation sanity**: open one MOTChallenge file in CVAT-viewer or `motmetrics` and visually overlay on the video for one frame; confirm bbox alignment and ID consistency.
2. **Conversion correctness**: pick one (video, frame, track_id) from a Parquet prediction, manually compute bbox from the row, compare to the corresponding line in the converted MOT .txt.
3. **Metric sanity**: run `motmetrics` on a degenerate case (predictions = ground truth) → all metrics should report perfect scores. Then on a known shifted-by-N-frames case → IDF1 should degrade predictably.
4. **Plausibility check**: aggregate HOTA for proposed pipeline should land in the 50–80 range typical of difficult multi-animal MOT; if it's >90 the GT and predictions may be miscompared, if <40 something is wrong with detection-side conversion.
