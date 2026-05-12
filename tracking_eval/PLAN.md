# Refined Plan: Tracking Evaluation for CVPR 2026 Workshop Revision

## Context

PlayClass was accepted at CV4Animals (CVPR 2026 workshop) as a poster, with revisions. This plan addresses **Reviewer 1's tracking-related concerns only**:

- "Lack of standard tracking evaluation metrics (e.g., HOTA, IDF1) to validate identity preservation"
- "Dependence on manual post-processing in the tracking pipeline, reducing automation and reproducibility"
- "Limited analysis of robustness to tracking errors, occlusions, and domain variation"
- "Limited novelty, with the method primarily combining existing tracking, feature extraction, and classification components"

**Reviewer 2's concerns** (ethogram description, "end-to-end" terminology, precision/recall + class-level metrics, dataset release plan, fine-grained granularity exploration) are **out of scope** for this plan and will be handled separately.

**Strategy** (revised 2026-05-11; further revised 2026-05-12): Use **in-distribution videos from days 28 and 29** as the evaluation set, one per cage, selecting the hardest group per cage via YOLO-scan-based difficulty ranking across both days jointly. At least 2 of the 5 picks are required to come from day 28 to ensure cross-day representativeness of the manuscript's training distribution. The day-37 external-validation approach was dropped after visual inspection showed day-37 birds look markedly different (older plumage) and move significantly less than the days 28–29 birds the manuscript's classification results are built on — benchmarking against day 37 would inflate tracker metrics relative to the actual training/eval distribution. The 5 LOCO folds in the manuscript already produce *disjoint* test partitions (each fold tests one full cage); drawing one video per cage from these natural per-fold test sets gives a 5-video subset that is in-distribution, cross-environment, and genuinely held out for the classifier. Run the SAM3 + adaptive chunking pipeline **fully automated (no manual post-processing)**, alongside the YOLO+BoT-SORT reference tracker, on these 5 bbox-annotated videos. Report standard MOT metrics. This addresses missing metrics, the automation concern, and — by including a generic tracker baseline — the "limited novelty" framing.

**Sparse-keyframe evaluation (2026-05-12).** Tracker metrics are computed on the **88 human-verified keyframes per video** (~440 total across the 5 videos), not on dense interpolated ground truth. CVAT's Track-mode interpolation is treated as an annotation convenience for navigation, not a source of GT; interpolated frames are filtered out before scoring. Rationale: linearly interpolated bboxes drift off non-linearly-moving birds, producing fake GT that penalises correct trackers and requires expensive per-interval QA. The standard MOT metrics (MOTA, IDF1, HOTA) are defined over GT-present frames and impose no density assumption — sparse evaluation matches the conventions of TAO (1 FPS GT on 30 FPS video) and the multi-animal pose-tracking literature (DLC, SLEAP, idtracker.ai). See the appendix at the end of this document for the verified citation chain and the manuscript-ready paragraph.

**Scope reduction under deadline (2026-05-12).** CV4Animals revision is due **2026-05-21**. Within that window, running both SAM3 default *and* a sweep-tuned SAM3 variant is not feasible. **Variant C (SAM3 tuned) and the Phase 3 chunking hyperparameter sweep are dropped.** The head-to-head comparison reduces to **Variant A (YOLO+BoT-SORT, already cached) vs Variant B (SAM3 default)**. Reviewer 1's three substantive concerns (missing HOTA/IDF1, manual post-processing, limited novelty vs a baseline tracker) are still addressed; only the sensitivity-analysis / hyperparameter-study framing is lost. The deferred sweep is acknowledged in the limitations section.

Day-37 scan results are retained on disk (`ext-data/output/results/sam3-hf/`) and the old candidate manifest is archived at `tracking_eval/video_manifest_day_37_superseded.csv` for possible future use, but day 37 is **out of scope** for this evaluation.

## Refinements vs. previous draft (`plan.md`)

1. **Drop the "boundary precision" annotation step.** Chunking quality is now evaluated by its downstream effect on HOTA/IDF1, using the same ground truth as the main tracking eval. Removes a non-standard metric, removes a second annotation pass, and aligns the entire study with the metrics Reviewer 1 asked for.
2. **Add a YOLO+BoT-SORT baseline tracker.** The existing `yolo_scan.py` already emits a `yolo_tracking.parquet` with `frame, track_id, bbox, confidence` (`src/tracker/scan.py:442–454`). Evaluating it as a baseline is essentially free and directly counters "just integrating existing components".
3. **Sweep is cheaper than the previous draft assumed.** `script/compute_chunk_boundaries.py` re-derives chunk boundaries from cached YOLO output without re-running YOLO. Each sweep config only re-runs SAM3 tracking, not the full pipeline. *(Note 2026-05-12: the sweep is deferred under the revision deadline — see refinement 6.)*
4. **Phase ordering**: annotation moves to Phase 1 (the bottleneck and foundation); chunk-guided frame selection still uses default-parameter chunking, but the chunking *evaluation* moves after annotation, since it now needs the GT.
5. **Switch eval set from day 37 to in-distribution day 29 (2026-05-11).** Day 37 birds are visually distinct (older plumage) and less active; using them would understate the tracker's difficulty on the data the classifier was actually trained and evaluated on. Day-29 selection draws one video per cage (hardest group per cage by YOLO-scan-based difficulty ranking), which aligns with the LOCO splitter's natural per-fold test partitions.
6. **Sparse-keyframe evaluation (2026-05-12).** Dropped the dense-interpolated-GT assumption. Metrics are computed only at the ~88 human-verified keyframes per video. Removes a QA-scrubbing pass (~3–4 hours per video) and removes measurement noise from linearly-interpolated bboxes on non-linearly-moving birds. Matches TAO and multi-animal pose-tracking conventions; metric definitions are unchanged. See Phase 1.3 / 1.4 / 4 below.
7. **Drop Variant C and Phase 3 sweep (2026-05-12).** Revision deadline 2026-05-21 makes running both Variants B and C, plus the 7-config sweep, infeasible. Comparison is now A vs B. Manuscript notes the deferred sweep in limitations.

## Timeline

**Hard deadline:** 2026-05-21 (CV4Animals revision). Effort estimate after 2026-05-12 scope cuts: ~11 person-hours + ~2.5 GPU-hours. See "Effort estimate" near the end of this document.

**Pipeline version freeze:** Variant B uses the identical `config/tracker.yaml` defaults shipped with the manuscript's days-28–29 runs — no re-tuning.

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
**CVAT** (Docker, local). Project name: `playclass-tracking-eval`. Single label `bird`; bird identity is captured by Track membership (3 Tracks per video), not by label.

**Track mode is used** but its linear interpolation is treated as a navigation aid only — interpolated frames are **not** treated as ground truth (filtered out at export; see Phase 1.4). The annotator visits only the frames listed in `annotation_frames.csv` for that video (88 frames × 3 birds), draws / adjusts the bbox at each listed frame following the conventions in `annotation_guidelines.md` (full estimated bbox on partial occlusion, Outside flag on total occlusion, single Track per bird), and does **not** add corrective keyframes to fix interpolation drift between listed frames.

Bbox-only annotation (not masks): MOTChallenge metrics need bboxes only, ~5× faster than masks. Predicted masks from SAM3 will be converted to bboxes for evaluation.

QA is per-keyframe-only: at each of the 88 listed keyframes per video, confirm the bbox is on the correct bird with a tight fit. Drift between keyframes is not a QA concern.

Documentation: `tracking_eval/annotation_guidelines.md` (already in repo; revised 2026-05-12 alongside this plan).

### 1.4 Export & filter to sparse GT
1. From CVAT, export each task in **MOT 1.1** format. Output includes all frames per Track (interpolated + keyframed).
2. Run `tracking_eval/scripts/filter_mot_to_keyframes.py`: for each video, intersect the exported MOT rows with the (video_id, frame_idx) set in `annotation_frames.csv`. Drop everything else.
3. Write filtered files to `ext-data/output/results/tracker_benchmark/ground_truth/<video_id>.txt`. Row format unchanged: `frame, id, bb_left, bb_top, bb_width, bb_height, conf, -1, -1, -1` with `conf = 1` for GT.

Sanity check inside the script: row count per file = `(n_keyframes × 3) − (Outside-flagged count)`.

---

## Phase 2: Tracker variants

Two trackers are run on the 5 annotated videos; both share the same sparse ground truth.

| Variant | Description | Source |
|---------|-------------|--------|
| **A: YOLO+BoT-SORT** | Generic detector + tracker baseline | Already produced by `src/tracker/scan.py:442` as `yolo_tracking.parquet`. No re-run needed |
| **B: SAM3 (default chunking)** | Proposed pipeline at manuscript defaults | Run `script/run_tracker.py` once per video with `config/tracker.yaml` |

**Critical**: both variants run **fully automated**, no manual ID correction. This is the headline reproducibility claim.

Outputs: `ext-data/output/results/tracker_benchmark/predictions/{A_yolo_botsort,B_sam3_default}/<video_id>.parquet`.

> **Variant C (SAM3 tuned) is dropped (2026-05-12).** Original plan included a 7-config chunking hyperparameter sweep (Phase 3) feeding a tuned Variant C. Under the 2026-05-21 deadline this is infeasible. The deferred sweep is noted in the manuscript's limitations section; the comparison reduces to A vs B.

---

## Phase 4: Tracking evaluation

### 4.1 Convert predictions → MOTChallenge
Script: `tracking_eval/scripts/convert_predictions.py`. Reads each Parquet, extracts bbox per (frame, track_id), writes `<video_id>.txt` in MOTChallenge format. Run for **Variants A and B**.

Predictions stay **dense per-frame** (predicted for every frame). Do not filter predictions to match GT keyframes — `motmetrics` aligns automatically by only updating its accumulator on frames where GT exists. Pre-filtering predictions would hide false positives on non-keyframe frames and distort detection metrics.

### 4.2 Compute metrics
**Library**: [`motmetrics`](https://github.com/cheind/py-motmetrics) (lighter than TrackEval; pure-Python; well-suited to small evals). TrackEval is the canonical alternative if HOTA fidelity matters more.

Metrics (priority order):
1. **HOTA** (Reviewer 1 explicitly asked) and its components **DetA**, **AssA**
2. **IDF1** (Reviewer 1 explicitly asked)
3. **MOTA** (community standard)
4. **ID switches** (raw count)

Bbox match threshold: IoU ≥ 0.5 (MOTChallenge default).

Script: `tracking_eval/scripts/evaluate_tracking.py`. Pseudocode:

```
For each tracker variant {A_yolo_botsort, B_sam3_default}:
    For each video_id:
        gt = load_sparse_gt(video_id)           # ~88 frames × 3 birds
        pred = load_dense_predictions(variant, video_id)
        acc = motmetrics.MOTAccumulator()
        for frame in sorted(set(gt.frame)):     # only GT-present frames
            match(gt[frame], pred[frame], iou_threshold=0.5)
            acc.update(...)
        compute HOTA / IDF1 / MOTA / IDsw / DetA / AssA on acc
    Aggregate per-cage and overall (frame-weighted by n_keyframes_per_video)
```

### 4.3 Stratified reporting
- Per-video (variance)
- Per-cage (cross-environment consistency, mirrors LOCO design)
- Aggregate (frame-weighted)
- **Both trackers side-by-side**

Outputs: `tracking_eval/results/metrics_{per_video,per_cage,aggregate}.csv`.

---

## Phase 5: Manuscript integration

### 5.1 New tables / figures
- **Table — Tracking metrics (two-way)**: HOTA, IDF1, MOTA, IDsw, DetA, AssA × {Variant A: YOLO+BoT-SORT, Variant B: SAM3 default} × {aggregate, per-cage}.
- No chunking-sweep table or figure (Phase 3 deferred).

### 5.2 Discussion / framing changes in §3.2 and §5
- Drop the "manual post-processing required" claim from §3.2 — replace with "automated chunking strategy quantified on a 5-video held-out evaluation subset (one video per cage, hardest group per cage); the original 30 videos used the same automated tracker, with manual identity correction restricted to chunk-boundary ID switches and protocol-ID assignment, and were not re-tracked for this evaluation".
- Add the sparse-keyframe-evaluation paragraph (see appendix at end of this document) to §3.2 where the tracker evaluation is first introduced.
- Add to the limitations section:
  - the one-sentence blind-spot caveat about inter-keyframe self-correcting ID switches not being observable under sparse evaluation;
  - the deferred chunking hyperparameter sweep — reported SAM3 numbers use the manuscript's existing default chunking parameters and are not an upper bound for this architecture.
- Address Reviewer 1 directly:
  - "Lack of HOTA/IDF1" → reported in the new two-way table.
  - "Manual post-processing" → the eval subset is tracked without it.
  - "Robustness to occlusion / domain variation" → per-cage stratification (5 cages) + per-cage hardest-group selection deliberately stresses occlusion-heavy conditions.
  - "Limited novelty / just combining components" → quantified lift of SAM3 + adaptive chunking over a generic YOLO+BoT-SORT baseline on the held-out 5-video evaluation subset. (Two-way comparison; sensitivity-analysis framing is dropped.)

---

## Directory structure

Heavy outputs (tracking parquets, MOT files, ground truth) live under `ext-data/output/results/tracker_benchmark/`, written directly through the `ext-data → /mnt/birds/rebecca2025/` symlink (no DVC). Scripts, manifests, and small result CSVs stay in the repo. The repo branch `tracker_benchmark` is already checked out.

```
ext-data/output/results/tracker_benchmark/        # On the mounted drive, not version-controlled
├── ground_truth/                                 # MOTChallenge .txt per video (sparse, ~88 frames × 3 birds)
├── predictions/
│   ├── A_yolo_botsort/                           # Parquets (or symlinks to existing yolo_scan run)
│   └── B_sam3_default/                           # SAM3 run dirs (timestamped)
└── predictions_mot/                              # Same A/B split, .txt format

tracking_eval/                                    # In repo, version-controlled
├── video_manifest.csv
├── annotation_guidelines.md
├── annotation_frames.csv
├── annotation_frames_summary.csv
├── scripts/
│   ├── select_annotation_frames.py
│   ├── filter_mot_to_keyframes.py
│   ├── convert_predictions.py
│   └── evaluate_tracking.py
└── results/
    ├── metrics_per_video.csv
    ├── metrics_per_cage.csv
    └── metrics_aggregate.csv
```

*(Removed under 2026-05-12 scope cuts: `predictions/C_sam3_tuned/`, `chunking_sweep/`, `tracking_eval/chunking/`, `tracking_eval/scripts/sweep_chunking.py`.)*

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
| CVAT setup + annotation guideline | done | `annotation_guidelines.md` (revised) |
| Ground truth annotation (sparse, 88 keyframes × 3 birds × 5 videos = 1,320 manual bboxes, no interpolation QA) | 3–5 | Reduced from 6–10 by dropping interpolation-drift scrubbing. |
| Run Variant A (already cached from chunking pipeline) | 0 | |
| Run Variant B SAM3 default (5 videos × ~30 min GPU) | 2.5 GPU-h | |
| ~~Sweep on 2 videos × 7 configs~~ | ~~7 GPU-h~~ | Dropped (Phase 3 deferred). |
| ~~Run Variant C on 5 videos~~ | ~~2.5 GPU-h~~ | Dropped. |
| `filter_mot_to_keyframes.py` + `convert_predictions.py` | 2 | |
| `evaluate_tracking.py` + metric computation | 3 | |
| Manuscript integration | 3 | Two-way framing, paragraph drop-in. |
| **Total (remaining)** | **~11 person-hours + ~2.5 GPU-hours** | Down from ~25 + ~12 GPU-h. Fits inside the 2026-05-21 deadline. |

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
