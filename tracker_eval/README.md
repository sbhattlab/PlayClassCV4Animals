# Repository documentation: tracker evaluation protocol

This document is the long-form repository protocol for the PlayClass tracker evaluation. The manuscript should contain only the compact summary; this file records the implementation details needed to reproduce and audit the evaluation.

## Purpose

The evaluation addresses the tracker-related reviewer concerns:

- reporting standard MOT metrics, including HOTA and IDF1;
- separating fully automated tracker performance from the manual correction used for the final classification dataset;
- testing robustness under occlusion-heavy, long-video conditions;
- framing the proposed tracker as a method, not only a collection of existing components.

## Evaluation subset

The 5-video sparse-keyframe evaluation subset contains five 15 min videos, one per cage. Videos were selected from days 28 and 29, matching the manuscript's classification distribution. Within each cage, the hardest group was selected using a composite difficulty score derived from the YOLO scan:

- fraction of high-occlusion frames;
- mean number of overlapping pairs;
- mean pairwise bounding-box IoU;
- fraction of object-count changes;
- occlusion periods per minute;
- mean centroid distance, ranked inversely;
- mean separation score, ranked inversely.

The selected videos are recorded in `data/tracker_eval/video_manifest.csv`. This design keeps the subset aligned with leave-one-cage-out testing while deliberately stressing the conditions most likely to induce identity errors.

## Sparse ground truth

Ground truth is sparse by design. The evaluated videos are filtered by `data/tracker_eval/video_manifest.csv` (`selected=True`). Each selected video has 87--88 human-verified frames, for 462 scored frames across the five-video subset. These frames are true ground-truth boxes at the annotated frames, not interpolated boxes.

The frame schedule combines three sources, deduplicated in priority order (chunk-guided > occlusion-bracketing > uniform):

1. **Chunk-guided**: for every internal adaptive-chunk boundary `B` produced by the production tracker configuration (`chunk_seconds=60`, `max_chunk_seconds=120`, `search_window_seconds=10`), sample `{B-5, B, B+5}`.
2. **Occlusion-bracketing**: for each of the three longest occlusion periods `(a, b)` reported by the YOLO scan, sample `{a-3, a, floor((a+b)/2), b, b+3}` to densify ground truth where linear interpolation between keyframes is least reliable.
3. **Uniform**: one frame every 30 s as a temporal backbone covering stable tracking periods.

The exact schedules are in `data/tracker_eval/annotation_frames.csv` and summarised in `data/tracker_eval/annotation_frames_summary.csv`.

## Why sparse keyframes are used

Dense per-frame annotation of long, multi-animal videos is expensive. We therefore annotate and score only the selected keyframes rather than constructing dense tracks. No CVAT interpolation is used for scoring.

The CVAT project backup is converted from native `annotations.json`, and the MOT ground truth contains only human-drawn, non-`outside` rectangles. This avoids using linearly interpolated boxes as pseudo-ground truth for non-linearly moving, frequently overlapping birds.

Standard MOT metrics such as HOTA, IDF1, and MOTA are computed over frames where ground truth exists; they do not require dense annotation. This matches the sparse-keyframe annotation strategy of the TAO long-tail tracking benchmark (Dave et al., 2020), which annotates at 1 FPS on 30 FPS video, and is the de facto convention in multi-animal tracking studies including multi-animal DeepLabCut (Lauer et al., 2022), SLEAP (Pereira et al., 2022), and idtracker.ai (Romero-Ferrero et al., 2019).

The limitation is that an ID switch that starts and self-corrects entirely within a single inter-keyframe interval is not observable. The chunk-boundary and occlusion-bracketed components of the schedule reduce but do not eliminate this blind spot.

## Tracker variants

Six fully automated variants are evaluated in a structural ablation that
isolates three orthogonal mechanisms of the proposed method: (1) adaptive
grounding (the scan-and-select stage that finds a usable seed frame past
chunk-start failure modes), (2) adaptive chunking (occlusion-informed
boundary placement), and (3) failure compensation (whole-chunk and prev-chunk
fallbacks). None uses manual identity correction.

| Variant | Name | Family | Recovery | Purpose |
| --- | --- | --- | --- | --- |
| A | `A_yolo_botsort` | YOLO + BoT-SORT | — | Detection-only baseline; no mask propagation. |
| B-strict | `B_gs2_strict` | gs2 | none (frame-0 GDINO, abort on failure) | Strict mask-propagation baseline. Exposes the failure modes that motivate gs2-parity. |
| B-parity | `B_gs2_fixed` | gs2 | best-frame seed + GDINO reinit on total loss | gs2 with parity-recovery mechanisms scaled to match SAM 3's. Mask-propagation baseline using SAM 3's immediate predecessor. |
| C-strict | `C_sam3_frame_zero` | SAM 3 | none (frame-0 grounding, no scan, both fallbacks disabled) | SAM 3 propagation without the adaptive grounding scaffolding. Isolates the contribution of the scan-and-select stage from the SAM 2 → SAM 3 backbone swap. |
| D | `D_sam3_fixed` | SAM 3 | adaptive grounding (scan + ranking + fallbacks) | SAM 3 with adaptive grounding, fixed 60 s chunking. |
| E | `E_sam3_adaptive` | SAM 3 | adaptive grounding + adaptive chunking | Full method. |

A supplementary seventh variant, `F_sam3_adaptive_strict` (full method with both
fallback mechanisms disabled), is reported in supplementary materials to
quantify the failure-compensation contribution independently.

Variant E is the proposed tracker. Each pairwise step in the table now isolates
one mechanism rather than folding several into a single gap:

- **A → B-strict / B-parity**: adds mask-based identity preservation over
  detection-only tracking; gs2-strict → gs2-parity quantifies what gs2-style
  recovery scaffolding is worth on its own.
- **B-strict → C-strict**: isolates the SAM 2 → SAM 3 backbone swap **at strict
  recovery parity** (no scan, no reinit, no fallback on either side).
- **B-parity → D**: same backbone swap at **full recovery parity**, where each
  family runs with the recovery mechanisms structurally appropriate to it.
- **C-strict → D**: isolates the **adaptive grounding** contribution alone
  (scan + best-frame ranking + fallbacks). The lighting cliff at chunk 0 of
  every video is part of what the scan exists to handle and is therefore part
  of what this gap measures.
- **D → E**: isolates the **adaptive chunking** contribution (occlusion-aware
  boundary placement, with grounding held constant).
- **E → F** *(supplementary)*: isolates the **failure-compensation** contribution
  (Sam3VideoModel and prev-chunk fallbacks combined). 47 GT keyframes lie
  inside chunks that trigger the prev-chunk fallback in the existing D / E
  runs; 4 lie inside the single chunk that triggers the Sam3VideoModel
  whole-chunk fallback.

The intended manuscript comparison is:

```text
HOTA(E) > HOTA(D) > HOTA(C-strict)
```

with the cross-family contrast `B-parity → D` carrying the model-family
attribution. A non-monotone dip at C-strict relative to B-parity is empirically
expected — it would demonstrate that the model-family swap alone does not
explain the gains, and that the contribution is concentrated in the adaptive
grounding scaffolding around SAM 3.

### Grounded-SAM-2 baselines (Variants B-strict and B-parity)

The gs2 pipeline (`src/tracker/grounded_sam_2.py`) is the IDEA-Research Grounded-SAM-2 reference. A GroundingDINO call at a fixed confidence threshold (0.25) grounds the seed frame with the text prompt "`.bird.`"; every surviving detection is refined into a mask by the SAM 2 image predictor and propagated by the SAM 2 video predictor. No retry loop, no area filtering, and no dataset-tuned knobs are used. The Swin-B GroundingDINO variant (`IDEA-Research/grounding-dino-base`) is used rather than the tiny variant, matching the convention of using the larger published weights where available. SAM 2.1 (not SAM-HQ2) is used to avoid an asymmetric "stronger SAM" advantage with no SAM 3 counterpart.

Two configurations of this pipeline are reported as main-table variants:

#### B-strict (`B_gs2_strict`)

The reference pipeline with no recovery scaffolding (`gs2.enable_recovery: false`): single GroundingDINO call on frame 0 of each chunk, abort on 0 detections, abort on total mask loss. Of the five eval videos, one (`C3G2`) returned zero detections on frame 0 and aborted at chunk 0; another (`C4G2`) had its single tracked bird's mask drop to empty in chunk 4 and the pipeline aborted at chunk 5. Both failures are visible in the per-video metrics. Its role in the main table is as the strict-recovery anchor for the cross-family `B-strict → C-strict` comparison, where neither family receives recovery scaffolding.

#### B-parity (`B_gs2_fixed`)

The reference pipeline with two structural recovery mechanisms — exact analogues of those in SAM 3's reference behaviour — enabled (`gs2.enable_recovery: true`):

1. **Best-frame seed selection** over the first `seed_scan_window = 125` frames at chunk 0. GroundingDINO is run on every candidate; the frame with the most detections at threshold 0.25 is chosen as the seed (ties broken by earliest frame). This mirrors SAM 3's `text_grounding.grounding_frames = 125`.
2. **GroundingDINO re-init on total carryover loss.** If `_extract_carryover_masks` finds zero non-empty masks in the last `max_lookback_frames` frames of chunk N (all tracked objects have died), chunk N+1 runs the same 125-frame best-frame search to rediscover objects from scratch. New objects are assigned fresh integer IDs continuing from the highest ID seen so far, so the discontinuity is explicit in the parquet and visible to MOT scoring.

The departure from the IDEA-Research reference is chunked processing with mask carryover: SAM 2's `init_state` cannot fit a 15-minute video in VRAM, so each video is split into 60 s chunks (matching Variant D). At each chunk boundary the masks of the last frame of chunk N seed chunk N+1, preserving object IDs without re-grounding. **Per-boundary re-grounding with IoU-based ID matching is not done** — that is the additional recovery layer SAM 3 has and gs2 (parity) does not. The remaining gap between `B-parity` and `D` therefore attributes to (a) the quality of the grounder (image-only GroundingDINO vs SAM 3's text-grounded video model), (b) per-chunk re-grounding, and (c) the prev-chunk and Sam3VideoModel fallbacks that gs2 does not have.

Predicted failure modes that remain under parity recovery (and motivate the SAM 3 contribution): brittle single-frame grounding even with best-frame search when no frame in the first 125 has clear views of all birds; ID-discontinuity penalties whenever re-init fires (fresh IDs cannot be matched to pre-loss IDs because gs2 has no IoU-matching infrastructure); fixed chunk boundaries that do not avoid occlusion events.

### SAM 3 frame-zero baseline (Variant C-strict)

The C-strict variant (`C_sam3_frame_zero`) ablates the adaptive grounding stage of the SAM 3 pipeline. Configured with `text_grounding.best_frame_method: "frame_zero"`, `text_grounding.allow_sam3_videomodel_fallback: false`, `text_grounding.fallback_to_prev_chunk: false`, and `use_adaptive_chunking: false`, it (i) grounds at a single fixed frame per chunk regardless of detection quality (no candidate selection), (ii) disables both failure-compensation fallbacks so that a chunk producing no usable grounding output produces empty predictions for the chunk, and (iii) uses fixed 60 s chunking. This is the structural analogue of `B_gs2_strict` for the SAM 3 family.

**Chunk-0 exception** (`text_grounding.chunk_zero_init_offset_frames: 125`): for chunk 0 only, the grounding scan window is shifted forward by 125 frames (≈ 5 s at 25 FPS) before the frame-zero lookup is performed. The motivation is that *literal* frame 0 of each recording is dominated by per-recording start-of-video artefacts (lighting transition still resolving, camera auto-adjustment, occasional initial-frame motion blur) that have nothing to do with the grounding-pipeline contribution being measured. Under a strict frame-0 null with no offset, C-strict has zero chance of producing useful predictions on chunk 0 of any video, and a reviewer can reasonably object that this measures dataset physics rather than the proposed method. Shifting by 125 frames lets chunk 0 init at the first frame past where adaptive grounding (variant D) would have finished its scan anyway — the chunk-0 measurement then reflects grounding-quality difference, not start-of-recording artefacts. Chunks 1+ are unaffected and ground at their own first frame. Output frame indices remain global; the first 125 frames of each video simply have no predictions for variant C.

The lighting transition at the start of each recording — lights are still adjusting in approximately the first few seconds, depending on the video — means frame 0 of chunk 0 reliably fails to find 3 separated birds. This is the failure mode the adaptive grounding scan was designed to handle. Empirical inspection of the existing D / E grounding outputs confirms that 4 of 5 eval videos require the scan to find a usable seed past the lighting cliff (earliest viable frames in the production runs: 72, 78, 46, 286); the fifth (`C2G2`) has no viable frame in the scan window at all and triggers the `Sam3VideoModel` whole-chunk fallback. Variant C-strict disables that fallback, so chunks where grounding cannot find any usable seed produce no predictions.

The `C_sam3_frame_zero` configs use the same per-video `grounding_frames` values as the corresponding D-fixed runs (125 for `C1G2`, `C2G2`, `C3G2`, `C4G2`; 375 for `C5G3`) to keep the scan-window length constant across the C → D ablation. See the current-status section below for the existing inconsistency in `grounding_frames` across the benchmark; the new C runs preserve the per-video value used by D rather than introducing a third configuration.

## Annotation and conversion

Annotation was performed in CVAT v2.64.0 using track mode with one `bird` label and three tracks per video. Bird identity is represented by track membership. Fully hidden birds are marked `outside=true` and excluded from scoring. A single trained annotator labelled all five videos to ensure identity consistency across the evaluation subset.

The CVAT project backup is converted to sparse MOTChallenge files with:

```bash
python -m src.tracker_eval cvat-to-mot
```

The converter writes one MOTChallenge 1.1 file per video under the configured ground-truth directory. Track IDs are assigned in CVAT track-declaration order, and `outside=true` shapes are dropped.

Predictions are converted with:

```bash
python -m src.tracker_eval convert-preds
```

Predictions remain dense. Sparsity is applied at evaluation time by scoring only frames where ground truth is present.

## Metrics

Evaluation uses:

- TrackEval (commit `12c8791`; canonical HOTA implementation of Luiten et al., 2021) for HOTA, DetA, AssA, LocA, DetRe, DetPr, AssRe, AssPr, and OWTA. HOTA is averaged across the standard α ∈ {0.05, 0.10, …, 0.95} sweep.
- py-motmetrics 1.4.0 for IDF1, MOTA, MOTP, ID switches, precision, recall, mostly-tracked, mostly-lost, and fragmentations.

Matching uses the MOTChallenge IoU threshold of 0.5. Results are written to:

- `data/tracker_eval/results/metrics_per_video.csv`;
- `data/tracker_eval/results/metrics_per_cage.csv`;
- `data/tracker_eval/results/metrics_aggregate.csv`.

Missing variants are skipped by the evaluator so partial results can be regenerated independently.

## Reproducible entry points

The two umbrella commands are:

```bash
pixi run -e tracker python -m src.tracker_eval prepare
```

then, after CVAT annotation:

```bash
pixi run -e tracker-evaluation python -m src.tracker_eval score
```

Individual stages can also be run:

```bash
python -m src.tracker_eval build-manifest
python -m src.tracker_eval select-frames
python -m src.tracker_eval cvat-to-mot
python -m src.tracker_eval convert-preds
python -m src.tracker_eval evaluate
```

Path defaults are centralised in `src/tracker_eval/paths.py`.

## Current status

As of 2026-05-20, the ablation has been expanded again — from 4 variants (A / B / C / D, mapped to {YOLO+BoT-SORT, gs2-parity, SAM 3 fixed, SAM 3 adaptive}) to a 6-variant structural ablation that explicitly isolates three mechanisms: adaptive grounding, adaptive chunking, and failure compensation.

The relabelling is:

| Old slot | New slot | Notes |
| --- | --- | --- |
| `A_yolo_botsort` | `A_yolo_botsort` | unchanged |
| *(supplementary)* `B_gs2_strict` | **`B_gs2_strict`** *(promoted to main table)* | Strict-recovery anchor for the cross-family comparison. |
| `B_gs2_fixed` | `B_gs2_fixed` *(now denoted "B-parity")* | unchanged; pipeline and recovery mechanisms identical. |
| *(new)* | **`C_sam3_frame_zero`** *(new run)* | SAM 3 frame-0 grounding, no scan, both fallbacks disabled, fixed chunking. |
| `C_sam3_fixed` | `D_sam3_fixed` | unchanged — re-lettered. |
| `D_sam3_adaptive` | `E_sam3_adaptive` | unchanged — re-lettered. |
| — | `F_sam3_adaptive_strict` *(supplementary, optional)* | Full method with both fallbacks disabled. |

| Variant | Inference | Predictions converted | Scored |
| --- | --- | --- | --- |
| A `A_yolo_botsort` | ✓ done | ✓ done | ✓ done |
| B-strict `B_gs2_strict` | ✓ done | ⏳ to do (promote from supplementary) | ⏳ pending |
| B-parity `B_gs2_fixed` | ✓ done | ✓ done | ✓ done |
| C-strict `C_sam3_frame_zero` | ⏳ to run (2 configs, day 28 + day 29) | ⏳ pending | ⏳ pending |
| D `D_sam3_fixed` (was `C_sam3_fixed`) | ✓ done | ⏳ rename | ⏳ pending |
| E `E_sam3_adaptive` (was `D_sam3_adaptive`) | ✓ done | ⏳ rename | ⏳ pending |
| F `F_sam3_adaptive_strict` *(supplementary)* | ⏳ optional | ⏳ pending | ⏳ pending |

`src/tracker_eval/{paths,predictions,evaluate}.py` need updating: the `VARIANTS` tuple expands to 6 entries (plus optional F), and a new `TRACKER_RUNS_FRAME_ZERO` path is added for the staged frame-zero outputs.

### Per-video `grounding_frames` inconsistency

A pre-existing inconsistency in the benchmark, surfaced 2026-05-20: across the 18 existing run dirs in `ext-data/output/results/sam3-hf/tracker_day_28_29/`, 11 use `text_grounding.grounding_frames: 125` and 7 use `375`. The 5 eval videos specifically are 4 × 125 (`C1G2`, `C2G2`, `C3G2`, `C4G2`) + 1 × 375 (`C5G3` only). The `C5G3` run was given the wider window because its earliest viable grounding frame is at frame 286, outside the 125-frame default. This is documented here so the C-strict and F runs can preserve the per-video value used by the existing D / E runs and the comparison stays apples-to-apples.

A uniform-window re-run of D and E (e.g. all five videos at `grounding_frames: 375`) is a defensible follow-up if reviewer pushback warrants it, but is out of scope for the current revision.
