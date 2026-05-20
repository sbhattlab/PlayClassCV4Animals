# Refined Plan: Tracking Evaluation for CVPR 2026 Workshop Revision

## Current status (2026-05-20, evening)

**At Phase 5 (manuscript integration), with the ablation expanded from 4 variants to 6 (plus one supplementary).** The previous 4-way ablation (A `YOLO+BoT-SORT` → B `gs2_fixed` → C `sam3_fixed` → D `sam3_adaptive`) made a monotone-capability point but folded three orthogonal mechanism contributions (adaptive grounding, adaptive chunking, failure compensation) into two table steps, making attribution ambiguous. The revised 6-variant ablation isolates each mechanism.

The new layout:

- A `A_yolo_botsort` — detection-only baseline (unchanged)
- B-strict `B_gs2_strict` — gs2 with no recovery (promoted from supplementary)
- B-parity `B_gs2_fixed` — gs2 with structural-parity recovery (unchanged)
- C-strict `C_sam3_frame_zero` — **new**: SAM 3, frame-0 grounding, no scan, both fallbacks disabled, fixed chunking
- D `D_sam3_fixed` — SAM 3 adaptive grounding + fixed chunking (was `C_sam3_fixed`)
- E `E_sam3_adaptive` — SAM 3 adaptive grounding + adaptive chunking (was `D_sam3_adaptive`)
- F `F_sam3_adaptive_strict` — supplementary: full method with both fallbacks disabled (optional)

This isolates: B-strict → C-strict = SAM 2 → SAM 3 backbone swap at strict-recovery parity; C-strict → D = adaptive-grounding contribution; D → E = adaptive-chunking contribution; E → F = failure-compensation contribution. Cross-family comparison B-parity → D additionally gives the SAM 2 → SAM 3 swap at full-recovery parity.

The remaining work for the revision is summarised below in "Variant C-strict — implementation plan (2026-05-20)".

## Previous status (2026-05-19, afternoon — superseded by 2026-05-20 above)

**At Phase 5 (manuscript integration), with the ablation expanded from 3-way to 4-way.** Phases 1–4 are complete for A, C, D. Variant B has been added (Grounded-SAM-2 with fixed 60 s chunking and parity recovery mechanisms) and inference is in progress as of this update. The previous Variants B and C have been renamed to C and D respectively; the new B slot is `B_gs2_fixed`.

**Headline aggregate from the prior 3-way run (5 videos, sparse GT, 462 keyframes; A/C/D numbers, B pending):**

| Variant | HOTA | DetA | AssA | IDF1 | MOTA | ID switches |
|---|---:|---:|---:|---:|---:|---:|
| A_yolo_botsort | 0.0646 | 0.187 | 0.023 | 0.059 | 0.064 | 173 |
| **B_gs2_fixed** | **(pending)** | **(pending)** | **(pending)** | **(pending)** | **(pending)** | **(pending)** |
| C_sam3_fixed (was B_sam3_fixed) | 0.5443 | 0.629 | 0.471 | 0.671 | 0.664 | 49 |
| **D_sam3_adaptive** (was C_sam3_adaptive) | **0.5560** | **0.631** | **0.491** | **0.701** | **0.702** | **35** |

The 3-way claim `HOTA(D) > max(HOTA(A), HOTA(C))` is established, with monotone gains across association-sensitive metrics (AssA, IDF1) and a 29 % reduction in ID switches between C and D. Per-video, C and D swap on individual clips (C narrowly wins on C5G3_day_28 and C1G2_day_29; D wins clearly on C2G2_day_28 where C's chunk 0 fell back to `Sam3VideoModel` because grounding could not find ≥ 3 birds in the first 125 frames). The cross-video variance is worth a sentence in the §5 discussion.

The expected B placement is between A and C — see "Variant B (Grounded-SAM-2) — added 2026-05-19" below for design rationale.

**Pipeline consolidation (2026-05-18).** The previous flat `tracker_eval/scripts/` directory has been replaced by a Python package under `src/tracker_eval/` exposing a single CLI with subcommands. The canonical entry points are now the two umbrella commands `pixi run -e tracker python -m src.tracker_eval prepare` (build-manifest + select-frames) and `pixi run -e tracker-evaluation python -m src.tracker_eval score` (cvat-to-mot + convert-preds + evaluate), bracketing the offline CVAT annotation checkpoint. Individual stages remain callable for ad-hoc re-runs (e.g. `python -m src.tracker_eval evaluate`). All runtime artefacts (CVAT Backup, tracker run parquets, the 5 source MP4 clips, ground-truth and prediction MOT files) live at `ext-data/tracker_benchmark/{cvat_backup,tracker_outputs_adaptive,tracker_outputs_fixed,source_videos,ground_truth,predictions_mot}/`. Path defaults live in `src/tracker_eval/paths.py`.

**Remaining work before write-up:**

1. **Manuscript integration (Phase 5)** under the 3-way framing.

**Retrospective renaming (2026-05-18).** The previous "Variant B: SAM3 default" is now **Variant C: SAM3 adaptive** (since the production config uses adaptive chunking), and the new fixed-chunking arm slots in as **Variant B**. Output subdirectories and the `VARIANTS` tuple in the eval scripts have been updated accordingly.

## Context

PlayClass was accepted at CV4Animals (CVPR 2026 workshop) as a poster, with revisions. This plan addresses **Reviewer 1's tracker-related concerns only**:

- "Lack of standard tracker evaluation metrics (e.g., HOTA, IDF1) to validate identity preservation"
- "Dependence on manual post-processing in the tracker pipeline, reducing automation and reproducibility"
- "Limited analysis of robustness to tracker errors, occlusions, and domain variation"
- "Limited novelty, with the method primarily combining an existing tracker, feature extraction, and classification components"

**Reviewer 2's concerns** (ethogram description, "end-to-end" terminology, precision/recall + class-level metrics, dataset release plan, fine-grained granularity exploration) are **out of scope** for this plan and will be handled separately.

**Strategy** (revised 2026-05-11; further revised 2026-05-12 and 2026-05-18): Use **in-distribution videos from days 28 and 29** as the evaluation set, one per cage, selecting the hardest group per cage via YOLO-scan-based difficulty ranking across both days jointly. At least 2 of the 5 picks are required to come from day 28 to ensure cross-day representativeness of the manuscript's training distribution. The day-37 external-validation approach was dropped after visual inspection showed day-37 birds look markedly different (older plumage) and move significantly less than the days 28–29 birds the manuscript's classification results are built on — benchmarking against day 37 would inflate tracker metrics relative to the actual training/eval distribution. The 5 LOCO folds in the manuscript already produce *disjoint* test partitions (each fold tests one full cage); drawing one video per cage from these natural per-fold test sets gives a 5-video subset that is in-distribution, cross-environment, and genuinely held out for the classifier. Run three **fully automated (no manual post-processing)** tracker configurations on the 5 bbox-annotated videos and report standard MOT metrics in a three-way ablation:

- **Variant A — YOLO + BoT-SORT only.** Generic tracker baseline. Isolates the scan stage and counters Reviewer 1's "limited novelty" framing.
- **Variant B — SAM 3 with fixed 60 s chunking.** SAM 3 propagation with no YOLO scan and no boundary refinement. Isolates the segmentation-propagation stage when run blind to occlusion structure.
- **Variant C — SAM 3 with adaptive, occlusion-informed chunking.** The full method. Boundaries are shifted within a ±10 s window toward frames maximising bird separation and avoiding occlusion.

The hypothesis is that Variant C strictly beats $\max(\text{A}, \text{B})$ and that the gain is concentrated in association-side metrics (AssA, IDF1, ID switches) at the chunk-boundary and occlusion-bracketed keyframes — i.e. that the lift comes from the *synergy* between YOLO scan and SAM 3, not from either component alone. This addresses Reviewer 1's three substantive concerns (missing HOTA/IDF1, manual post-processing, limited novelty) in a single ablation table.

**Sparse-keyframe evaluation (2026-05-12).** Tracker metrics are computed on the **88 human-verified keyframes per video** (~440 total across the 5 videos), not on dense interpolated ground truth. CVAT's Track-mode interpolation is treated as an annotation convenience for navigation, not a source of GT; interpolated frames are filtered out before scoring. Rationale: linearly interpolated bboxes drift off non-linearly-moving birds, producing fake GT that penalises correct trackers and requires expensive per-interval QA. The standard MOT metrics (MOTA, IDF1, HOTA) are defined over GT-present frames and impose no density assumption; sparse evaluation matches the conventions of TAO, a long-tail tracker benchmark with 1 FPS GT on 30 FPS video, and related multi-animal pose-estimation and tracker studies (DLC, SLEAP, idtracker.ai). See the appendix at the end of this document for the verified citation chain and the manuscript-ready paragraph.

**Scope reduction under deadline (2026-05-12).** CV4Animals revision is due **2026-05-21**. Within that window, running a sweep-tuned SAM3 variant on top of the planned configurations is not feasible. **The Phase 3 chunking hyperparameter sweep is dropped**, and what was originally labelled "Variant C (SAM3 tuned)" is dropped along with it. The deferred sweep is acknowledged in the limitations section.

**Three-way ablation added (2026-05-18).** The comparison was previously framed as **A (YOLO+BoT-SORT) vs B (SAM3 default)**, where "SAM3 default" meant SAM 3 with the production adaptive-chunking config. That framing conflates two ingredients of the proposed method (SAM 3 propagation + occlusion-informed chunking) into one arm, so a win for B doesn't tell us *which* ingredient drove it. To make the ablation crisp, SAM 3 with fixed 60 s chunking is added as a new arm and the variant labels are renamed accordingly: **A (YOLO+BoT-SORT) — B (SAM 3 + fixed chunking) — C (SAM 3 + adaptive chunking)**. Variant B costs one extra ~2.5 GPU-hour run; the rest of the pipeline (annotation, GT export, evaluation) is unchanged.

**Four-way ablation added (2026-05-19).** The previous 3-way A/B/C ablation isolated YOLO-only tracking from SAM 3 (B/C), but conflated the contribution of *SAM 3's video pretraining and multi-frame grounding* with the contribution of *any mask-based propagation tracker*. Reviewers could have argued that the A→B/C gap reflects "any mask propagator beats detection-only," not specifically what SAM 3 brings. To make the ablation crisp, Grounded-SAM-2 (gs2) is added as a new B between A and the two SAM 3 arms, and the existing SAM 3 arms are re-lettered to C and D. Variant labels are now **A (YOLO+BoT-SORT) — B (Grounded-SAM-2 + fixed chunking) — C (SAM 3 + fixed chunking) — D (SAM 3 + adaptive chunking)**, monotone in sophistication. gs2 is the immediate technical predecessor of SAM 3: a frozen GroundingDINO detector grounds the seed frame, SAM 2 image predictor refines boxes into masks, SAM 2 video predictor propagates them — i.e. the off-the-shelf mask-propagation baseline a practitioner would reach for before SAM 3 existed. Implementation in `src/tracker/grounded_sam_2.py`; submodule at `ext/Grounded-SAM-2/` (IDEA-Research upstream, registered 2026-05-19).

### Variant B (Grounded-SAM-2) — added 2026-05-19

#### Design history (paths considered and rejected)

The gs2 baseline was initially run using the user's customised fork (`prince-ravi-leow/Grounded-SAM-2`, the same fork referenced on `main` at commit `53e7a1d`) with three dataset-tuned behavioural enhancements on top of the IDEA-Research reference: (i) a threshold-lowering retry loop dropping `box_threshold` from 0.25 to a floor of 0.15 until ≥ `min_objects_for_tracking=3` valid detections were found, (ii) an area filter rejecting boxes >40 % or <0.3 % of frame area, and (iii) a `min_objects_for_tracking=3` knob driving the retry. These tweaks encoded the domain prior "there are exactly 3 birds per pen" into the baseline. Keeping them would have invited the reviewer pushback "your gs2 baseline isn't really gs2 — you re-engineered it to suit your dataset." That initial run was killed mid-flight.

The replacement (Path C — strict reference) stripped all three tweaks and ran a single GroundingDINO call at threshold 0.25 with no retry, no area filter, accept-whatever-comes-back, abort on 0 detections. Results were honest but exposed two catastrophic failure modes that the asymmetric recovery scaffolding between gs2 and SAM 3 was responsible for: **C3G2** returned 0 detections on frame 0 of chunk 0 and aborted the whole pipeline; **C4G2** seeded only 1 bird at chunk 0 and that bird's mask became fully empty mid-video, so the pipeline aborted at chunk 5. The strict run is preserved as `B_gs2_strict` in the supplementary because it motivates the recovery-mechanism asymmetry.

The headline `B_gs2_fixed` (parity-recovery) is the IDEA-Research reference pipeline **plus** structural analogues of SAM 3's two recovery mechanisms, implemented with gs2-native components (no SAM 3 dependency). Two enhancements:

1. **Best-frame seed selection** over the first 125 frames at chunk 0 (mirrors SAM 3's `text_grounding.grounding_frames=125`). GroundingDINO is run on every candidate frame; the frame with the most detections at threshold 0.25 is chosen as the seed. Ties broken by earliest frame.
2. **GroundingDINO re-init on total carryover loss.** If `_extract_carryover_masks` returns an empty dict (no surviving masks in the last `max_lookback_frames` frames of chunk N), chunk N+1 runs the same 125-frame best-frame search to rediscover objects. New objects receive fresh integer IDs continuing from the highest ID seen so far, so the discontinuity is explicit in the parquet.

Per-chunk-boundary re-grounding with IoU-based ID matching (the *additional* recovery layer SAM 3 has) is **not** implemented for gs2. The remaining gap between B and C therefore attributes to (a) the quality of the grounder — image-only GroundingDINO vs SAM 3's text-grounded video-pretrained model — and (b) the additional per-chunk re-init layer. Both are SAM 3 contributions worth attributing.

Other deliberate choices: the larger Swin-B GroundingDINO variant (`IDEA-Research/grounding-dino-base`, ~340M params) is used rather than the tiny variant, matching the convention of using the larger published HF weights where available and giving the baseline the strongest grounder shipped by the same authors. SAM 2.1 (not SAM-HQ2) is used to avoid creating an asymmetric "stronger SAM" advantage that has no SAM 3 counterpart. The text prompt is `.bird.` matching SAM 3.

The two staggered runs scheduled 2026-05-19 evening: parity day 29 on CUDA 1 starts immediately (day 29 strict was already complete); parity day 28 on CUDA 0 starts as soon as strict day 28 finishes (~16:00). Combined inference ETA ~17:45.

Day-37 scan results are retained on disk (`ext-data/output/results/sam3-hf/`) and the old candidate manifest is archived at `data/tracker_eval/video_manifest_day_37_superseded.csv` for possible future use, but day 37 is **out of scope** for this evaluation.

## Refinements vs. previous draft (`plan.md`)

1. **Drop the "boundary precision" annotation step.** Chunking quality is now evaluated by its downstream effect on HOTA/IDF1, using the same ground truth as the main tracker eval. Removes a non-standard metric, removes a second annotation pass, and aligns the entire study with the metrics Reviewer 1 asked for.
2. **Add a YOLO+BoT-SORT baseline tracker.** The existing `yolo_scan.py` already emits a `yolo_tracking.parquet` with `frame, track_id, bbox, confidence` (`src/tracker/scan.py:442–454`). Evaluating it as a baseline is essentially free and directly counters "just integrating existing components".
3. **Sweep is cheaper than the previous draft assumed.** `script/compute_chunk_boundaries.py` re-derives chunk boundaries from cached YOLO output without re-running YOLO. Each sweep config only re-runs the SAM3 tracker, not the full pipeline. *(Note 2026-05-12: the sweep is deferred under the revision deadline — see refinement 6.)*
4. **Phase ordering**: annotation moves to Phase 1 (the bottleneck and foundation); chunk-guided frame selection still uses default-parameter chunking, but the chunking *evaluation* moves after annotation, since it now needs the GT.
5. **Switch eval set from day 37 to in-distribution day 29 (2026-05-11).** Day 37 birds are visually distinct (older plumage) and less active; using them would understate the tracker's difficulty on the data the classifier was actually trained and evaluated on. Day-29 selection draws one video per cage (hardest group per cage by YOLO-scan-based difficulty ranking), which aligns with the LOCO splitter's natural per-fold test partitions.
6. **Sparse-keyframe evaluation (2026-05-12).** Dropped the dense-interpolated-GT assumption. Metrics are computed only at the ~88 human-verified keyframes per video. Removes a QA-scrubbing pass (~3–4 hours per video) and removes measurement noise from linearly-interpolated bboxes on non-linearly-moving birds. Matches TAO and multi-animal pose-estimation/tracker conventions; metric definitions are unchanged. See Phase 1.3 / 1.4 / 4 below.
7. **Drop Phase 3 sweep (2026-05-12).** Revision deadline 2026-05-21 makes the 7-config sweep infeasible. Manuscript notes the deferred sweep in limitations.
8. **Add SAM 3 fixed-chunking arm and re-letter variants (2026-05-18).** The earlier two-way A-vs-B comparison made a method-vs-baseline point but left the source of the lift ambiguous — SAM 3 alone might do the work, with the occlusion-informed chunker contributing nothing. Adding a SAM-3-with-fixed-chunking arm turns the experiment into a proper ablation. Variant labels are re-lettered so that A → B → C corresponds to a monotonic increase in method components (YOLO only → +SAM 3 → +adaptive chunking). Script-level rename of `B_sam3_default` → `C_sam3_adaptive` and addition of `B_sam3_fixed` are reflected in `convert_predictions.py` and `evaluate_tracker.py`.
9. **Add Grounded-SAM-2 arm and expand to four-way ablation (2026-05-19).** The 3-way A/B/C ablation conflated "SAM 3 video pretraining + multi-frame grounding" with "any mask-propagation tracker," giving reviewers room to dismiss the SAM 3 contribution. Adding gs2 as a fourth arm between A and the SAM 3 variants (with parity recovery mechanisms scaled to match SAM 3's recovery scaffolding) makes the comparison crisp: A→B isolates mask-based identity preservation over detection-only tracking; B→C isolates SAM 3's video-pretrained text-grounded grounder + per-chunk re-grounding over GroundingDINO image-only grounding; C→D isolates occlusion-aware adaptive chunking. Variant labels re-lettered again: A (YOLO+BoT-SORT) → B (gs2 + fixed) → C (SAM 3 + fixed) → D (SAM 3 + adaptive). `src/tracker_eval/{paths,predictions,evaluate}.py` updated. Old `B_sam3_fixed` / `C_sam3_adaptive` directories under `predictions_mot/` will be regenerated under the new C/D names.

10. **Decompose the contribution into three mechanisms and expand to six-way ablation (2026-05-20).** The 4-way A/B/C/D ablation isolates the YOLO-only → mask-propagation jump (A → B) and the gs2 → SAM 3 family jump (B → C), but the C → D step still folds two of the proposed method's contributions (adaptive grounding + adaptive chunking) into a single gap, and the SAM 3 family's failure-compensation fallbacks are invisible to the table. The contribution claim was re-decomposed into three orthogonal mechanisms — adaptive grounding (scan + best-frame ranking), adaptive chunking (occlusion-informed boundary placement), and failure compensation (`Sam3VideoModel` whole-chunk fallback + prev-chunk-mask fallback) — and the ablation expanded to isolate each.

    Empirical analyses run during this session ruled out an earlier null candidate: comparing `earliest`-from-pool against the production `combined` ranking across all 75 chunks of the 5 eval videos showed they pick the same frame on **74 / 75 chunks (98.7 %)**, so an earliest-vs-combined ablation would have produced near-zero signal. The chosen C-variant null instead removes the scan entirely: ground at frame 0 of each chunk, no candidate filtering, and disable both fallbacks. Variant labels re-lettered: A (YOLO+BoT-SORT) → B-strict (gs2 no recovery) → B-parity (gs2 with parity recovery) → C-strict (SAM 3 frame-0 no fallback) → D (SAM 3 adaptive grounding + fixed chunking, was `C_sam3_fixed`) → E (SAM 3 adaptive grounding + adaptive chunking, was `D_sam3_adaptive`), with optional supplementary F (full method, both fallbacks disabled). `B_gs2_strict` is promoted from supplementary to main table; `src/tracker_eval/{paths,predictions,evaluate}.py` and the prediction-MOT directory layout need updating accordingly.

    The non-monotone dip at C-strict relative to B-parity is anticipated and intentional: it would demonstrate that the SAM 2 → SAM 3 backbone swap alone does not explain the tracking-quality gains, and that the proposed method's contribution is concentrated in the adaptive-grounding scaffolding around SAM 3. Reverting B to gs2-strict to maintain a monotone column ordering was explicitly considered and rejected — the parity recovery was added in the previous session for a documented, defensible reason, and rolling it back would invite reviewer suspicion.

## Variant C-strict — implementation plan (2026-05-20, finalised)

This section describes the concrete implementation for the new
`C_sam3_frame_zero` variant. Scope: one selection-method branch plus one
new config flag, two YAML configs (one per day), two GPU runs in
parallel, and eval-wiring updates. Supplementary Variant F (full method
with both fallbacks disabled) is **deferred** under the 2026-05-21
deadline.

### Design summary

C-strict is SAM 3 with the adaptive grounding mechanism removed:

- **Selection**: `text_grounding.best_frame_method = "frame_zero"`. The
  scan still runs over the `grounding_frames` window (so `Sam3VideoModel`
  receives normal multi-frame video context and produces honest
  segmentations), but `find_best_grounding_frame` is modified to **look
  up the chunk's first frame specifically** rather than picking the best
  candidate. If SAM 3 emitted any output at the chunk's start frame, the
  tracker is initialised from that frame's masks; otherwise selection
  returns `None`.
- **Chunk-0 exception** (`text_grounding.chunk_zero_init_offset_frames =
  125`): for chunk 0 only, the grounding scan is shifted by 125 frames
  (≈ 5 s at 25 FPS) before looking up the chunk's "first" frame. This
  mirrors the scan window adaptive grounding (variant D) would have had
  anyway and gives the strict baseline a fair shot past
  per-recording start-of-video failure modes (lighting cliff, camera
  adjustment). Chunks 1+ are unaffected — they still ground at their own
  first frame. Output frame indices remain global; the first 125 frames
  of each video simply have no predictions for variant C. The motivation
  is reviewer-defence: under a literal frame-0 null, C-strict has no
  chance on chunk 0 because of recording artefacts rather than a
  fundamental method weakness, which a reviewer could rightly object to.
  Shifting by 125 lets the chunk-0 measurement reflect grounding-quality
  difference, not start-of-recording physics.
- **Fallbacks disabled**: `text_grounding.allow_sam3_videomodel_fallback
  = false` (new flag, default `true` for backward compatibility) and
  `text_grounding.fallback_to_prev_chunk = false` (existing flag). When
  grounding produces no usable seed at the chunk's first frame, the
  chunk emits empty predictions instead of escalating to a whole-chunk
  `Sam3VideoModel` pass or copying masks from the previous chunk.
- **Chunking**: fixed 60 s chunks (`use_adaptive_chunking: false`).
- **Scan window**: `text_grounding.grounding_frames = 125` uniformly
  across all 5 videos. This eliminates the per-video 125 / 375
  inconsistency from the existing D-fixed runs because C-strict ignores
  everything past frame 0 anyway — the scan-window length is
  effectively cosmetic but kept at 125 to give SAM 3 enough video
  context for honest segmentation quality at frame 0.

The contribution being measured by C → D is the value of running the
adaptive grounding scan + best-frame selection + failure-compensation
fallbacks together, on top of the same underlying SAM 3 propagation. The
lighting cliff at chunk 0 of every video is part of the failure mode the
scan exists to handle, and the C → D gap therefore includes the lighting
cliff's contribution to grounding-pipeline value. This trade-off is
acknowledged in `tracker_eval/README.md` under "SAM 3 frame-zero baseline".

### Code changes

#### `src/tracker/grounding.py`

Extend `find_best_grounding_frame` with a `chunk_start_frame_idx`
parameter (default `None` for backward compatibility) and a `frame_zero`
branch that looks up exactly that frame:

```python
def find_best_grounding_frame(
    grounding_outputs: dict,
    min_objects: int = 3,
    method: str = "combined",
    chunk_start_frame_idx: int | None = None,
) -> tuple[int | None, list, list, list]:
    if method == "frame_zero":
        if chunk_start_frame_idx is None or chunk_start_frame_idx not in grounding_outputs:
            return None, [], [], []
        results = grounding_outputs[chunk_start_frame_idx]
        masks_list, boxes_list, object_ids_list = get_all_objects_from_results(results)
        return int(chunk_start_frame_idx), masks_list, boxes_list, object_ids_list
    # ... existing branches unchanged ...
```

The `frame_zero` branch deliberately does *not* apply the `min_objects`
filter — the goal is to feed the tracker whatever SAM 3 returned at the
chunk's first frame, even if that's 0 or 1 objects, and let the
resulting under-initialised tracker demonstrate the failure mode.

#### `src/tracker/tracker.py`

Three edits at the existing grounding call site:

```python
grounding_frames_count = grounding_cfg.get("grounding_frames", 25)
chunk_zero_offset = (
    grounding_cfg.get("chunk_zero_init_offset_frames", 0)
    if chunk_idx == 0
    else 0
)
if chunk_zero_offset > 0 and chunk_zero_offset < len(chunk_frames):
    scan_frames = chunk_frames[chunk_zero_offset:]
    scan_start_idx = global_chunk_start + chunk_zero_offset
else:
    scan_frames = chunk_frames
    scan_start_idx = global_chunk_start

grounding_outputs = run_grounding(
    scan_frames, scan_start_idx, grounding_frames_count, _process_video_chunk, cfg, device,
)
gr_out_frame_idx, gr_out_masks, gr_out_boxes, gr_out_ids = (
    find_best_grounding_frame(
        grounding_outputs,
        min_objects=grounding_cfg.get("min_objects", cfg.min_objects_for_tracking),
        method=grounding_cfg.get("best_frame_method", "combined"),
        chunk_start_frame_idx=scan_start_idx,   # NEW: shifts for chunk-0 exception
    )
)
```

Gate the `Sam3VideoModel` whole-chunk fallback behind a new config flag,
defaulting to `True` for backward compatibility. At the existing
`if not use_tracker:` block:

```python
allow_s3vm_fallback = cfg.get("text_grounding", {}).get(
    "allow_sam3_videomodel_fallback", True
)

if not use_tracker:
    if not allow_s3vm_fallback:
        chunk_info["model_type"] = "EmptyChunk"
        chunk_info["fallback_reason"] = "grounding_failed_no_fallback_allowed"
        chunk_outputs = {}
    else:
        if chunk_info["fallback_reason"] is None:
            chunk_info["fallback_reason"] = (
                "first_chunk" if chunk_idx == 0 else "grounding_failed_no_fallback"
            )
        chunk_info["model_type"] = "Sam3VideoModel"
```

The `if use_tracker / else _process_video_chunk` block below also needs
to short-circuit when `chunk_outputs` is already `{}` (empty chunk).

The prev-chunk fallback is already gated by the existing
`text_grounding.fallback_to_prev_chunk` flag — no code change needed
there.

### New configs

Two configs in `config/`, mirroring the existing
`tracker_rerun_fixed_day_{28,29}.yaml` layout:

#### `config/tracker_frame_zero_day_28.yaml`

```yaml
CUDA_VISIBLE_DEVICES: "0"
PYTORCH_ALLOC_CONF: "expandable_segments:True,garbage_collection_threshold:0.6"

job_type: "sam3_hf_frame_zero"
video_dir: "ext-data/raw/rerun_fixed_day_28"   # contains C2G2, C5G3
text_prompt: "bird"
output_dir: "ext-data/output/results/sam3-hf"
start_frame: 0
max_frames_to_track: 0

reuse_chunk_info: false
reuse_run_dir: null
yolo_scan_only: false
use_adaptive_chunking: false
chunk_seconds: 60

min_objects_for_tracking: 3
max_lookback_frames: 10

text_grounding:
  enabled: true
  grounding_frames: 125
  min_objects: 3
  best_frame_method: "frame_zero"
  fallback_to_prev_chunk: false
  allow_sam3_videomodel_fallback: false
  id_matching: true
  id_match_iou_threshold: 0.10

tracking:
  init_trk_keep_alive: 60
  max_trk_keep_alive: 60
  min_trk_keep_alive: 0
  trk_assoc_iou_thresh: 0.3
  hotstart_dup_thresh: 12
  suppress_overlap_thresh: 0.8
  recondition_every_nth_frame: 8

metrics:
  occlusion_iou_threshold: 0.15
  clustering_distance_threshold: 50.0
```

#### `config/tracker_frame_zero_day_29.yaml`

Identical to day 28 except `CUDA_VISIBLE_DEVICES: "1"` and
`video_dir: "ext-data/raw/rerun_fixed_day_29"` (contains C1G2, C3G2,
C4G2).

### Run plan

Two GPUs in parallel; total wall-clock dominated by CUDA 1 (the
3-video day).

| Run | Video(s) | Config | GPU | ETA |
| --- | --- | --- | --- | --- |
| C-strict day 28 | `C2G2`, `C5G3` | `tracker_frame_zero_day_28.yaml` | CUDA 0 | ~5 h (2 × 2.5 h) |
| C-strict day 29 | `C1G2`, `C3G2`, `C4G2` | `tracker_frame_zero_day_29.yaml` | CUDA 1 | ~7.5 h (3 × 2.5 h) |

Outputs land at `ext-data/output/results/sam3-hf/<timestamp>_sam3_hf_frame_zero/day_{28,29}/<stem>/`. Symlink the per-video output dirs into `ext-data/tracker_benchmark/tracker_outputs_sam3_frame_zero/<stem>/` for the eval pipeline.

### Fallback rule if C-strict collapses

Pre-committed contingency: if the aggregate C-strict HOTA falls below
the A baseline (≈ 0.065), the variant carries no useful signal — the
table would just be reporting "the pipeline produced little output."
In that case re-run C with `allow_sam3_videomodel_fallback: true` (but
`fallback_to_prev_chunk: false`, and `best_frame_method: "frame_zero"`
preserved), document the swap as a measurement-floor adjustment, and
label the row `C_sam3_frame_zero_with_s3vm` in the final table.

### Eval wiring updates (after inference)

`src/tracker_eval/paths.py`:

```python
TRACKER_RUNS_FRAME_ZERO = BENCHMARK_DIR / "tracker_outputs_sam3_frame_zero"
TRACKER_RUNS_GS2_STRICT = BENCHMARK_DIR / "tracker_outputs_gs2_strict"  # promoted
```

`src/tracker_eval/predictions.py`:

- Add `--predictions-root-frame-zero` (default `TRACKER_RUNS_FRAME_ZERO`).
- Promote `--predictions-root-gs2-strict` from supplementary handling.
- Bucket layout: `A_yolo_botsort`, `B_gs2_strict`, `B_gs2_fixed`,
  `C_sam3_frame_zero`, `D_sam3_fixed`, `E_sam3_adaptive`.

`src/tracker_eval/evaluate.py`:

```python
VARIANTS = (
    "A_yolo_botsort",
    "B_gs2_strict",
    "B_gs2_fixed",
    "C_sam3_frame_zero",
    "D_sam3_fixed",
    "E_sam3_adaptive",
)
```

On disk: rename `predictions_mot/C_sam3_fixed/` → `D_sam3_fixed/` and
`predictions_mot/D_sam3_adaptive/` → `E_sam3_adaptive/`. Stage
`B_gs2_strict/` into `predictions_mot/`.

### Figure regeneration

`tmp/viz_tracker_eval.py`:

- Update `VARIANT_ORDER`, `VARIANT_LABELS`, `VARIANT_COLORS` for 6 rows.
- `BAR_WIDTH = 0.13`; recentre `_offset(i)` for 6 bars per group.
- Y-axis range may need extension downward to accommodate B-strict's
  negative MOTA and any C-strict negative MOTA values.
- Per-cage HOTA figure: 6 bars per cage. Consider grouping by family
  (yellow YOLO; green pair gs2-strict / gs2-parity; blue triple
  C-strict / D / E) for readability.
- DetA-vs-AssA scatter: 6 dots, colour-coded by family.

Drop the existing 4-way figures as superseded; regenerate all from
the new aggregate / per-video CSVs.

### Manuscript integration

The §3.2 framing changes from "best initiation point" to the
three-mechanism decomposition described in `tracker_eval/README.md`.
The supplementary evidence that `earliest` and `combined` ranking agree
on 74 / 75 chunks goes into supplementary as a paragraph justifying
why the production ranking criterion is reported as `combined` even
though `earliest` would produce the same predictions on this dataset.

### Effort estimate

| Task | Hours | Notes |
| --- | --- | --- |
| Code changes (`grounding.py`, `tracker.py`) | ~1 | One new method, one new config gate. |
| New configs (2 YAMLs) | ~0.25 | One per day. |
| Inference (C-strict, 5 videos) | ~7.5 GPU-h wall-clock | 2 GPUs in parallel. |
| Symlink + stage outputs | ~0.25 | Mirror existing layout. |
| Eval wiring updates | ~0.5 | `paths.py`, `predictions.py`, `evaluate.py` + bucket renames. |
| Re-score | ~0.25 | `pixi run -e tracker-evaluation score-tracker-eval`. |
| Figure regen | ~1 | `tmp/viz_tracker_eval.py` 6-row update. |
| Manuscript revisions | ~3 | §3.2 framing + supplementary. |
| **Total** | **~12 person-hours + ~7.5 GPU-hours** | One-run budget; multi-seed runs out of scope. |

**Supplementary F deferred.** Variant F (full method, strict
no-fallback) is not in scope under the 2026-05-21 deadline. The
failure-compensation contribution stays implicit in the D / E numbers
for this revision; F can be added in a follow-up if reviewers ask.

---

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

Composite range: 7–210 across 30 candidates. Ranking script: `src/tracker_eval/manifest.py` (invoked as `python -m src.tracker_eval build-manifest`; promoted from the original `tmp/rank_videos_both_days.py` one-shot for reproducibility; discovers scan dirs under `ext-data/output/results/sam3-hf/`, computes the 7 proxies from each video's `metrics/yolo_scan_{metrics,summary}.parquet`, and writes the manifest deterministically). At least 2 of the 5 picks are required to come from day 28; the joint ranking naturally produces 2 day-28 picks without forcing (the script asserts this via `--min-day-28`).

**Selected (manifest: `data/tracker_eval/video_manifest.csv`):**

| Cage | Group | Day | Video stem | Rank-sum | Notes |
|------|-------|-----|------------|----------|-------|
| C1 | G2 | 29 | `C1G2_Test_2_day_29_2_Camera_5_2025_02_05_11_06_07_2` | 160 | hardest in C1 |
| C2 | G2 | 28 | `C2G2_Test_1_day_28_1_Camera_5_2025_02_04_09_43_25_2` | **186** | overall hardest, hardest in C2 |
| C3 | G2 | 29 | `C3G2_Test_2_day_29_2_Camera_5_2025_02_05_09_22_32_2` | 183 | hardest in C3 |
| C4 | G2 | 29 | `C4G2_Test_2_day_29_2_Camera_5_2025_02_05_10_31_53_2` | 158 | hardest in C4 |
| C5 | G3 | 28 | `C5G3_Test_1_day_28_1_Camera_8_2025_02_04_11_35_00_3` | 176 | hardest in C5 |

Source scan dirs (resolved per video in `video_manifest.csv` → `scan_dir`): four scan runs cover the 30 candidates — `20260317_162056_sam3_hf` (14 of the day-29 videos), `20260319_152425_sam3_hf` (C2G2 day 29), `20260308_230835_sam3_hf` and `20260309_230105_sam3_hf` (day-28 videos; latest scan run is used when duplicates exist).

**LOCO alignment**: Each selected video belongs to its respective cage's LOCO test fold (e.g., C1G2 is in fold 0's test set when C1 is held out). This means tracker metrics are reported on data that was *truly held out* for the classifier in the paper's LOCO results. Cross-day spread (2 day-28 + 3 day-29) matches the manuscript's training distribution.

### 1.2 Chunk-guided frame selection
Run **default-parameter chunking only** (no SAM3) on the 5 selected videos using existing infrastructure. For each video, build the annotation frame list:

- **Chunk-guided** (≈60%): for each detected boundary, sample frames at `boundary - 5`, `boundary`, `boundary + 5`.
- **Uniform backbone** (≈40%): one frame every 15 s. Guards against blind spots in the chunking heuristic.

**Target**: ~80 frames/video × 5 videos = ~400 frames, ~1200 bboxes (3 birds × 400 frames).

Script: `src/tracker_eval/frame_selection.py` (invoked as `python -m src.tracker_eval select-frames`). Output: `data/tracker_eval/annotation_frames.csv` with columns `video_id, frame_idx, source` (chunk_guided | occlusion_bracketing | uniform).

### 1.3 Annotation tool & protocol
**CVAT** (Docker, local). Project name: `playclass-tracker-eval`. Single label `bird`; bird identity is captured by Track membership (3 Tracks per video), not by label.

**Track mode is used** but its linear interpolation is treated as a navigation aid only — interpolated frames are **not** treated as ground truth (filtered out at export; see Phase 1.4). The annotator visits only the frames listed in `annotation_frames.csv` for that video (88 frames × 3 birds), draws / adjusts the bbox at each listed frame following the conventions in `annotation_guidelines.md` (full estimated bbox on partial occlusion, Outside flag on total occlusion, single Track per bird), and does **not** add corrective keyframes to fix interpolation drift between listed frames.

Bbox-only annotation (not masks): MOTChallenge metrics need bboxes only, ~5× faster than masks. Predicted masks from SAM3 will be converted to bboxes for evaluation.

QA is per-keyframe-only: at each of the 88 listed keyframes per video, confirm the bbox is on the correct bird with a tight fit. Drift between keyframes is not a QA concern.

Documentation: `tracker_eval/annotation_guidelines.md` (already in repo; revised 2026-05-12 alongside this plan).

### 1.4 Export to sparse GT
1. From CVAT, take a **project Backup** export (not the per-task MOT 1.1 export). The Backup includes each task's `annotations.json` in CVAT's native schema, which stores only human-drawn keyframes — no temporal interpolation is materialised.
2. Run `python -m src.tracker_eval cvat-to-mot` (module: `src/tracker_eval/cvat_to_mot.py`) against the backup root at `ext-data/tracker_benchmark/cvat_backup/playclass-tracker-eval/`. For each task it reads `annotations.json`, assigns track ids 1–3 in CVAT track-declaration order, drops shapes with `outside=true`, and writes `<video_id>.txt` in MOTChallenge 1.1 format. Row format: `frame, id, bb_left, bb_top, bb_width, bb_height, conf, -1, -1, -1` with `conf = 1` for GT. The `--keyframes-csv data/tracker_eval/annotation_frames.csv` default additionally reports per-bird scheduled / drawn / missing / extra counts.
3. Output: `ext-data/tracker_benchmark/ground_truth/<video_id>.txt`.

This replaces the originally-planned `filter_mot_to_keyframes.py` step (which would have filtered an MOT export against the scheduled-keyframe set) — going through the Backup avoids materialising CVAT's linear interpolation in the first place, so there's nothing to filter out.

---

## Phase 2: Tracker variants

Three trackers are run on the 5 annotated videos; all three share the same sparse ground truth.

| Variant | Description | Source | Status |
|---------|-------------|--------|--------|
| **A: YOLO + BoT-SORT** | Generic detector + tracker baseline. Isolates the scan stage. | Already produced by `src/tracker/scan.py:442` as `yolo_tracking.parquet`. No re-run needed. | **done** |
| **B: SAM 3 + fixed chunking** | SAM 3 propagation, uniform 60 s chunks, no YOLO scan, no boundary refinement. Isolates the segmentation stage. | Run `script/run_tracker.py` with `config/tracker.yaml` overridden to set `use_adaptive_chunking: false`. Configs used: `config/tracker_rerun_fixed_day_{28,29}.yaml`. | **done** (2026-05-18) |
| **C: SAM 3 + adaptive chunking** | Full method. Boundaries shifted within a ±10 s window toward high-separation, low-occlusion frames using the YOLO-scan signal. | Run `script/run_tracker.py` with `config/tracker.yaml` defaults. | **done** |

**Critical**: all three variants run **fully automated**, no manual ID correction. This is the headline reproducibility claim.

Outputs: per-variant tracker run directories at `ext-data/tracker_benchmark/tracker_outputs_adaptive/<stem>/` (`yolo_tracking.parquet` + `tracking_outputs.parquet` → supplies A and C) and `ext-data/tracker_benchmark/tracker_outputs_fixed/<stem>/` (`tracking_outputs.parquet` → supplies B). MOT-converted forms land in `ext-data/tracker_benchmark/predictions_mot/{A_yolo_botsort,B_sam3_fixed,C_sam3_adaptive}/<video_id>.txt`.

> **Tuned-SAM3 variant (Phase 3 sweep) is dropped (2026-05-12).** Original plan included a 7-config chunking hyperparameter sweep feeding a tuned variant. Under the 2026-05-21 deadline this is infeasible. The deferred sweep is noted in the manuscript's limitations section.

---

## Phase 4: Tracker evaluation

### 4.1 Convert predictions → MOTChallenge
Module: `src/tracker_eval/predictions.py` (invoked as `python -m src.tracker_eval convert-preds`). Reads each Parquet, extracts bbox per (frame, track_id), writes `<video_id>.txt` in MOTChallenge format. Run for **Variants A, B, and C**. The script takes `--predictions-root` (default `ext-data/tracker_benchmark/tracker_outputs_adaptive/`; supplies A and C from the adaptive run dir) and `--predictions-root-fixed` (default `ext-data/tracker_benchmark/tracker_outputs_fixed/`; supplies B; silently skipped if dir is missing or empty).

Predictions stay **dense per-frame** (predicted for every frame). Do not filter predictions to match GT keyframes — `motmetrics` aligns automatically by only updating its accumulator on frames where GT exists. Pre-filtering predictions would hide false positives on non-keyframe frames and distort detection metrics.

### 4.2 Compute metrics
**Libraries**: [`py-motmetrics 1.4.0`](https://github.com/cheind/py-motmetrics) for IDF1/MOTA/MOTP/ID-switches/precision/recall, and **TrackEval** (vendored at `ext/TrackEval`, commit `12c8791`; canonical HOTA implementation of Luiten et al. 2021) for HOTA / DetA / AssA / LocA averaged over the standard $\alpha \in \{0.05, 0.10, \ldots, 0.95\}$ sweep. Using both side by side avoids the known IDF1/HOTA-implementation drift between metric libraries.

Metrics (priority order):
1. **HOTA** (Reviewer 1 explicitly asked) and its components **DetA**, **AssA**
2. **IDF1** (Reviewer 1 explicitly asked)
3. **MOTA** (community standard)
4. **ID switches** (raw count)

Bbox match threshold: IoU ≥ 0.5 (MOTChallenge default).

Module: `src/tracker_eval/evaluate.py` (invoked as `python -m src.tracker_eval evaluate`). The `VARIANTS` tuple is `("A_yolo_botsort", "B_sam3_fixed", "C_sam3_adaptive")`; variants whose prediction MOT files are not yet on disk are skipped with a warning, so partial / incremental re-runs are safe while Variant B is being produced.

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

Outputs: `data/tracker_eval/results/metrics_{per_video,per_cage,aggregate}.csv`. Written for all three variants (A, B, C) as of 2026-05-18 evening.

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

Heavy outputs (tracker parquets, MOT files, ground truth) live under `ext-data/output/results/tracker_benchmark/`, written directly through the `ext-data → /mnt/birds/rebecca2025/` symlink (no DVC). Scripts, manifests, and small result CSVs stay in the repo. The repo branch `tracker_benchmark` is already checked out.

```
ext-data/tracker_benchmark/                       # On the mounted drive, not version-controlled
├── cvat_backup/
│   ├── playclass-tracker-eval/                  # Extracted CVAT project Backup (project.json + task_*/)
│   └── playclass-tracker-eval.zip               # Kept for re-import to CVAT
├── source_videos/                                # 5 MP4 clips fed to CVAT (provenance)
├── tracker_outputs_adaptive/                     # SAM 3 adaptive-chunking + YOLO scan run dirs per video (Variants A and C)
├── tracker_outputs_fixed/                        # SAM 3 fixed-chunking run dirs per video (Variant B)
├── ground_truth/                                 # MOTChallenge .txt per video (sparse, ~88 frames × 3 birds)
└── predictions_mot/                              # A_yolo_botsort / B_sam3_fixed / C_sam3_adaptive .txt files

src/tracker_eval/                                # In repo, version-controlled — Python package
├── __init__.py                                   # empty package marker
├── __main__.py                                   # CLI dispatcher (subcommands + prepare/score umbrellas)
├── paths.py                                      # single source of truth for all paths
├── manifest.py                                   # YOLO-scan-driven hardest-group-per-cage ranking
├── frame_selection.py                            # keyframe scheduler for CVAT
├── cvat_to_mot.py                                # CVAT Backup → sparse MOT GT
├── predictions.py                                # tracker parquets → MOT predictions (A/B/C)
└── evaluate.py                                   # sparse-keyframe MOT metrics (motmetrics + TrackEval HOTA)

tracker_eval/                                    # Protocol documentation
├── README.md
├── PLAN.md
└── annotation_guidelines.md

data/tracker_eval/                               # In repo, version-controlled — CSV artefacts
├── video_manifest.csv
├── annotation_frames.csv / annotation_frames_summary.csv
└── results/
    ├── metrics_per_video.csv
    ├── metrics_per_cage.csv
    └── metrics_aggregate.csv
```

*(Removed under 2026-05-12 scope cuts: `predictions/C_sam3_tuned/`, `chunking_sweep/`, `tracker_eval/chunking/`, `tracker_eval/scripts/sweep_chunking.py`. Renamed under 2026-05-18 ablation refactor: `predictions/B_sam3_default/` → `predictions/C_sam3_adaptive/`. Superseded by the CVAT-Backup path: `filter_mot_to_keyframes.py` was never needed since the Backup's `annotations.json` already contains only human-drawn keyframes. Consolidated under 2026-05-18 pipeline refactor: the five standalone scripts in `tracker_eval/scripts/` were renamed and moved up one level, exposed via a single `python -m src.tracker_eval` CLI; runtime data lives at `ext-data/tracker_benchmark/`.)*

---

## Critical existing code to reuse

| File | Purpose |
|------|---------|
| [src/tracker/chunking.py:64](src/tracker/chunking.py#L64) | `chunk_video_frames_adaptive()` |
| [src/tracker/scan.py:442](src/tracker/scan.py#L442) | YOLO+BoT-SORT scan → `yolo_tracking.parquet` |
| [src/metrics.py:373](src/metrics.py#L373) | `compute_separation_score` (parameter semantics) |
| [script/compute_chunk_boundaries.py:145](script/compute_chunk_boundaries.py#L145) | Sweep entry point — recomputes boundaries without re-running YOLO |
| [config/tracker_scan.yaml:81-89](config/tracker_scan.yaml#L81-L89) | Default chunking parameter values |
| [config/tracker.yaml](config/tracker.yaml) | Default tracker config (use unchanged for Variant B) |

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
| Run Variant B — SAM 3 + fixed chunking (5 videos) | done | ~13 GPU-h total wall-clock (≈ 2.5 h / video; significantly longer than the original ~30 min/video estimate due to shared-server CPU contention and chunk-0 grounding fallbacks on Day-28 clips — see `tmp/tracker_optimization_findings.md`). Added 2026-05-18 for the three-way ablation. |
| `convert_predictions.py` (incl. `--predictions-root-fixed` for Variant B) | done | |
| `evaluate_tracker.py` + metric computation (motmetrics + TrackEval HOTA) | done for A, B, and C | All three variants in `data/tracker_eval/results/`. |
| Manuscript integration | ~3 | Three-way ablation framing, paragraph drop-in (`tmp/tracker_evaluation_methods.md`). |
| ~~Sweep on 2 videos × 7 configs~~ | ~~7 GPU-h~~ | Dropped (Phase 3 deferred). |
| ~~Run tuned-SAM3 variant on 5 videos~~ | ~~2.5 GPU-h~~ | Dropped. |
| **Remaining (as of 2026-05-18 evening)** | **~3 person-hours** | Manuscript write-up only. |

---

## Verification (end-to-end smoke test)

1. **Filter correctness**: after `filter_mot_to_keyframes.py` runs on one hand-checked video, confirm row count = `(n_keyframes × 3) − (Outside-flagged count)`. Visually overlay the filtered GT on the video for one frame; confirm bbox alignment and ID consistency.
2. **Conversion correctness**: pick one (video, frame, track_id) from a Parquet prediction, manually compute bbox from the row, compare to the corresponding line in the converted MOT .txt.
3. **Degenerate metric check**: feed identical sparse GT as both GT and prediction into `evaluate_tracker.py` → all metrics report perfect scores.
4. **Shift check**: shift predictions by +N frames → IDF1 / HOTA degrade predictably. With sparse GT, gaps between GT-present frames are large, so this is a *stronger* identity test than the dense-GT equivalent.
5. **Plausibility check**: aggregate HOTA for proposed pipeline should land in the 50–80 range typical of difficult multi-animal MOT; if it's >90 the GT and predictions may be miscompared, if <40 something is wrong with detection-side conversion.

---

## Appendix: Sparse-evaluation paragraph and verified citations

Drop this paragraph into §3.2 (methods) or §5 (limitations/methodology) of the manuscript revision, where the tracker evaluation is first introduced. Citation keys assume the manuscript's existing BibTeX convention — substitute as needed.

> Tracker evaluation follows the methodological convention established for long-form multi-object tracker studies and multi-animal behaviour analysis, in which the cost of dense per-frame ground truth is prohibitive. We evaluate at sparse human-verified keyframes (88 frames per video, biased toward chunk-boundary intervals that bracket the three longest occlusion periods identified by a preliminary YOLO scan; see Phase 1.2 of the evaluation plan). Standard MOT metrics — MOTA [Bernardin and Stiefelhagen, 2008], IDF1 [Ristani et al., 2016], and HOTA with its DetA / AssA decomposition [Luiten et al., 2021] — are defined over GT-present frames and impose no assumption of dense annotation. This matches the sparse-keyframe annotation strategy of TAO, a long-tail tracker benchmark [Dave et al., 2020] with annotations at 1 FPS on 30 FPS video, and is the de facto convention in multi-animal pose-estimation and tracker studies including multi-animal DeepLabCut [Lauer et al., 2022], SLEAP [Pereira et al., 2022], and idtracker.ai [Romero-Ferrero et al., 2019]. We acknowledge one limitation: identity switches that occur and self-correct entirely within a single inter-keyframe interval are not observable under sparse evaluation. The chunk-boundary keyframe schedule, which densifies sampling around the three longest occlusion periods per video, reduces but does not eliminate this blind spot.

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
