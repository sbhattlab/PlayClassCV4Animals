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

Four fully automated variants are evaluated in a monotone-sophistication ablation. None uses manual identity correction.

| Variant | Name | Purpose |
| --- | --- | --- |
| A | `A_yolo_botsort` | YOLO + BoT-SORT detector-tracker baseline. Detection-only tracking; no mask propagation. |
| B | `B_gs2_fixed` | Grounded-SAM-2 with fixed 60 s chunks **and parity recovery mechanisms** (best-frame seed selection + GroundingDINO re-init on total mask loss). Mask-propagation baseline using the immediate predecessor to SAM 3, with the recovery scaffolding scaled to match SAM 3's, so the remaining gap attributes cleanly to the quality of the grounder (GroundingDINO image-only vs SAM 3 text-grounded video). |
| C | `C_sam3_fixed` | SAM 3 propagation with fixed 60 s chunks. Isolates SAM 3 without YOLO-guided boundary refinement. |
| D | `D_sam3_adaptive` | Full method: SAM 3 with adaptive, occlusion-informed chunking. |

Variant D is the proposed tracker. Variants A–C are ablations of its constituent parts. The intended manuscript comparison is:

```text
HOTA(D) > max(HOTA(A), HOTA(B), HOTA(C))
```

with corresponding gains expected in association-sensitive metrics such as AssA, IDF1, and ID switches. Each successive variant addresses one specific weakness of the prior:

- **A → B**: adds mask-based identity preservation over detection-only tracking.
- **B → C**: adds video-pretrained text grounding, multi-frame seed selection, and mid-video re-initialisation (SAM 3 vs SAM 2).
- **C → D**: adds occlusion-aware adaptive chunk boundary placement.

### Grounded-SAM-2 baseline (Variant B)

The gs2 pipeline (`src/tracker/grounded_sam_2.py`) is the IDEA-Research Grounded-SAM-2 reference with **parity recovery mechanisms** added to scale gs2's recovery scaffolding to match SAM 3's. A GroundingDINO call at a fixed confidence threshold (0.25) grounds the seed frame with the text prompt "`.bird.`"; every surviving detection is refined into a mask by the SAM 2 image predictor and propagated by the SAM 2 video predictor. No retry loop, no area filtering, and no dataset-tuned knobs are used. The Swin-B GroundingDINO variant (`IDEA-Research/grounding-dino-base`) is used rather than the tiny variant, matching the convention of using the larger published weights where available.

Two structural recovery mechanisms — exact analogues of those in SAM 3's reference behaviour — are enabled (`gs2.enable_recovery: true`):

1. **Best-frame seed selection** over the first `seed_scan_window = 125` frames at chunk 0. GroundingDINO is run on every candidate; the frame with the most detections at threshold 0.25 is chosen as the seed (ties broken by earliest frame). This mirrors SAM 3's `text_grounding.grounding_frames = 125`.
2. **GroundingDINO re-init on total carryover loss.** If `_extract_carryover_masks` finds zero non-empty masks in the last `max_lookback_frames` frames of chunk N (all tracked objects have died), chunk N+1 runs the same 125-frame best-frame search to rediscover objects from scratch. New objects are assigned fresh integer IDs continuing from the highest ID seen so far, so the discontinuity is explicit in the parquet and visible to MOT scoring.

The remaining departure from the IDEA-Research reference is chunked processing with mask carryover: SAM 2's `init_state` cannot fit a 15-minute video in VRAM, so each video is split into 60 s chunks (matching Variant C). At each chunk boundary the masks of the last frame of chunk N seed chunk N+1, preserving object IDs without re-grounding. **Per-boundary re-grounding with IoU-based ID matching is not done** — that is the additional recovery layer SAM 3 has and gs2 (parity) does not. The remaining gap between B and C therefore attributes to (a) the quality of the grounder (image-only GroundingDINO vs SAM 3's text-grounded video model) and (b) per-chunk re-grounding, separable from the recovery-mechanism asymmetry that strict gs2 would have introduced.

The strict, no-recovery variant of the gs2 pipeline (best-frame selection and re-init both disabled, single GroundingDINO call on frame 0, abort on 0 detections or total mask loss) was also run, and is reported in the supplementary as `B_gs2_strict`. Inspection of the strict run motivates the recovery mechanisms: of five videos, one (`C3G2`) returned zero detections on frame 0 and aborted at chunk 0; another (`C4G2`) had its single tracked bird's mask drop to empty in chunk 4 and the pipeline aborted at chunk 5. Both failures are recovered by the parity-recovery variant.

Predicted failure modes that remain under parity recovery (and motivate the SAM 3 contribution): brittle single-frame grounding even with best-frame search when no frame in the first 125 has clear views of all birds; ID-discontinuity penalties whenever re-init fires (fresh IDs cannot be matched to pre-loss IDs because gs2 has no IoU-matching infrastructure); fixed chunk boundaries that do not avoid occlusion events.

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

As of 2026-05-19, the ablation has been expanded from three variants (A/B/C) to four (A/B/C/D). The rename is: the previous B (`B_sam3_fixed`) and C (`C_sam3_adaptive`) are now C and D; the new B slot is occupied by `B_gs2_fixed` (Grounded-SAM-2 with fixed chunking and parity recovery).

| Variant | Inference | Predictions converted | Scored |
| --- | --- | --- | --- |
| A `A_yolo_botsort` | ✓ done | ✓ done | ✓ done |
| B `B_gs2_fixed` (parity-recovery, headline) | ⏳ in progress (day 29 on CUDA 1; day 28 queued behind strict gs2 on CUDA 0) | ⏳ pending | ⏳ pending |
| B `B_gs2_strict` (supplementary, motivates recovery) | ✓ day 29 done (3 videos: 1 full, 1 chunk-0 abort, 1 chunk-5 abort), ⏳ day 28 in progress (C5G3 chunk 5/14) | ⏳ pending | ⏳ pending |
| C `C_sam3_fixed` | ✓ done | ✓ done | ✓ done |
| D `D_sam3_adaptive` | ✓ done | ✓ done | ✓ done |

`src/tracker_eval/{paths,predictions,evaluate}.py` already reflect the four-variant layout. The scorer will skip missing variants, so A/C/D results can be regenerated at any time. The manuscript should only make final comparative claims after Variant B (parity) has completed and `data/tracker_eval/results/*.csv` has been regenerated with all four variants.

The supplementary `B_gs2_strict` run uses the same module (`src/tracker/grounded_sam_2.py`) with `gs2.enable_recovery: false`, single-frame grounding on chunk-0 frame 0 with no scan window, and no GroundingDINO re-init on total carryover loss. Its outputs land under `ext-data/output/results/grounded-sam-2/<timestamp>_gs2_fixed/` and will be staged separately (not into `tracker_outputs_gs2/`) so the headline scorer sees only the parity variant.

### Inference ETA at last check (15:59)

| | GPU | Status | ETA |
| --- | --- | --- | --- |
| Strict gs2 day 28 (C2G2 ✓, C5G3 in progress) | CUDA 0 | C5G3 on chunk 5 of 14 | ~16:00 |
| Strict gs2 day 29 (C1G2 ✓, C3G2 aborted ch0, C4G2 aborted ch5) | CUDA 1 | done | — |
| Parity gs2 day 29 (3 videos) | CUDA 1 | C1G2 in chunk 0 best-frame scan | ~18:30 |
| Parity gs2 day 28 (2 videos) | CUDA 0 | queued behind strict | start ~16:00, finish ~17:45 |
