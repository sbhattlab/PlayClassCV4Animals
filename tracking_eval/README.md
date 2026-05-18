# Repository documentation: tracking evaluation protocol

This document is the long-form repository protocol for the PlayClass tracker evaluation. The manuscript should contain only the compact summary; this file records the implementation details needed to reproduce and audit the evaluation.

## Purpose

The evaluation addresses the tracking-related reviewer concerns:

- reporting standard MOT metrics, including HOTA and IDF1;
- separating fully automated tracker performance from the manual correction used for the final classification dataset;
- testing robustness under occlusion-heavy, long-video conditions;
- framing the proposed tracker as a method, not only a collection of existing components.

## Evaluation subset

The held-out subset contains five 15 min videos, one per cage. Videos were selected from days 28 and 29, matching the manuscript's classification distribution. Within each cage, the hardest group was selected using a composite difficulty score derived from the YOLO scan:

- fraction of high-occlusion frames;
- mean number of overlapping pairs;
- mean pairwise bounding-box IoU;
- fraction of object-count changes;
- occlusion periods per minute;
- mean centroid distance, ranked inversely;
- mean separation score, ranked inversely.

The selected videos are recorded in `tracking_eval/video_manifest.csv`. This design keeps the subset aligned with leave-one-cage-out testing while deliberately stressing the conditions most likely to induce identity errors.

## Sparse ground truth

Ground truth is sparse by design. Each video has 87--88 human-verified frames, for 462 scored frames across the five-video subset. These frames are true ground-truth boxes at the annotated frames, not interpolated boxes.

The frame schedule combines three sources:

- chunk-guided frames: `B - 5`, `B`, and `B + 5` around each internal adaptive-chunk boundary `B`;
- occlusion-bracketing frames: five frames around each of the three longest YOLO-detected occlusion periods;
- a uniform temporal backbone covering stable periods.

The exact schedules are in `tracking_eval/annotation_frames.csv` and summarised in `tracking_eval/annotation_frames_summary.csv`.

## Why sparse keyframes are used

Dense per-frame annotation of long, multi-animal videos is expensive, and CVAT's linear track interpolation is not an adequate substitute for dense ground truth. Chicks move non-linearly, change pose quickly, and frequently overlap. Interpolated boxes can drift away from the animal between human-verified frames, creating pseudo-ground truth that can penalise a correct tracker.

For that reason, CVAT track mode is used only as an annotation interface. The CVAT project backup is converted from native `annotations.json`, which stores human-drawn keyframes, and interpolated frames are not materialised for scoring.

Standard MOT metrics such as HOTA, IDF1, and MOTA are computed over frames where ground truth exists; they do not require dense annotation. The limitation is that an ID switch that starts and self-corrects entirely between two keyframes is unobservable. The frame schedule reduces this blind spot by oversampling chunk boundaries and occlusion intervals, where such errors are most likely.

## Tracker variants

Three fully automated variants are evaluated. None uses manual identity correction.

| Variant | Name | Purpose |
| --- | --- | --- |
| A | `A_yolo_botsort` | YOLO26x + BoT-SORT detector-tracker baseline. Isolates the scan stage. |
| B | `B_sam3_fixed` | SAM 3 propagation with fixed 60 s chunks. Isolates SAM 3 without YOLO-guided boundary refinement. |
| C | `C_sam3_adaptive` | Full method: SAM 3 with adaptive, occlusion-informed chunking. |

Variant C is the proposed tracker. Variants A and B are ablations of its constituent parts. The intended manuscript comparison is:

```text
HOTA(C) > max(HOTA(A), HOTA(B))
```

with corresponding gains expected in association-sensitive metrics such as AssA, IDF1, and ID switches. This comparison tests whether the proposed adaptive SAM 3 tracker benefits from the interaction between the YOLO/BoT-SORT scan and SAM 3 propagation, rather than from either component alone.

## Annotation and conversion

Annotation was performed in CVAT v2.64.0 using track mode with one `bird` label and three tracks per video. Bird identity is represented by track membership. Fully hidden birds are marked `outside=true` and excluded from scoring.

The CVAT project backup is converted to sparse MOTChallenge files with:

```bash
python -m tracking_eval cvat-to-mot
```

The converter writes one MOTChallenge 1.1 file per video under the configured ground-truth directory. Track IDs are assigned in CVAT track-declaration order, and `outside=true` shapes are dropped.

Predictions are converted with:

```bash
python -m tracking_eval convert-preds
```

Predictions remain dense. Sparsity is applied at evaluation time by scoring only frames where ground truth is present.

## Metrics

Evaluation uses:

- TrackEval for HOTA, DetA, AssA, LocA, DetRe, DetPr, AssRe, AssPr, and OWTA;
- py-motmetrics 1.4.0 for IDF1, MOTA, MOTP, ID switches, precision, recall, mostly-tracked, mostly-lost, and fragmentations.

Matching uses the MOTChallenge IoU threshold of 0.5. Results are written to:

- `tracking_eval/results/metrics_per_video.csv`;
- `tracking_eval/results/metrics_per_cage.csv`;
- `tracking_eval/results/metrics_aggregate.csv`.

Variant B may be absent until the fixed-chunking SAM 3 run has completed; the evaluator skips missing variants so A/C results can be regenerated independently.

## Reproducible entry points

The two umbrella commands are:

```bash
pixi run -e tracker python -m tracking_eval prepare
```

then, after CVAT annotation:

```bash
pixi run -e tracker-evaluation python -m tracking_eval score
```

Individual stages can also be run:

```bash
python -m tracking_eval build-manifest
python -m tracking_eval select-frames
python -m tracking_eval cvat-to-mot
python -m tracking_eval convert-preds
python -m tracking_eval evaluate
```

Path defaults are centralised in `tracking_eval/paths.py`.

## Current status

As of 2026-05-18, the sparse ground truth and Variants A/C have been converted and scored. Variant B, SAM 3 with fixed chunking, is still pending. The manuscript should only make final comparative claims after Variant B has completed and `tracking_eval/results/*.csv` has been regenerated.
