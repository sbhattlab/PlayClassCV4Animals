# Tracking Evaluation — Annotation Guidelines

CVAT-based bbox annotation of 5 videos (2 from day 28, 3 from day 29) to produce MOTChallenge ground truth for HOTA / IDF1 / MOTA / IDsw evaluation. Background and the full evaluation plan are in `tracking_eval/PLAN.md`.

## Handoff package (what to copy to the annotation machine)

| File | Purpose |
|------|---------|
| `tracking_eval/PLAN.md` | Full evaluation plan (context only). |
| `tracking_eval/annotation_guidelines.md` | This file. |
| `tracking_eval/annotation_frames.csv` | The 438 keyframes to annotate (3 columns: `video_id, frame_idx, source`). |
| `tracking_eval/annotation_frames_summary.csv` | Per-video keyframe counts and chunk/occlusion metadata. |
| `tracking_eval/video_manifest.csv` | Selection metadata, including the source paths of the 5 `.mp4` files. |
| The 5 `.mp4` files | Listed in `video_manifest.csv` under `path` (rows with `selected=True`). |

Once annotation is complete, ship back the 5 **raw CVAT MOT 1.1 exports** (one per task). The keyframe-filtering step (`filter_mot_to_keyframes.py`) runs on the original machine and produces the final sparse-GT files under `ext-data/output/results/tracker_benchmark/ground_truth/`. See **Export** below for the export procedure and **`tracking_eval/PLAN.md` §1.4** for the filtering rationale.



| Video | Keyframes | Birds | Bboxes |
|-------|---:|---:|---:|
| C1G2_day_29 | 88 | 3 | 264 |
| C2G2_day_28 | 88 | 3 | 264 |
| C3G2_day_29 | 87 | 3 | 261 |
| C4G2_day_29 | 87 | 3 | 261 |
| C5G3_day_28 | 88 | 3 | 264 |
| **Total** | **438** | | **1314** |

Authoritative keyframe list: `tracking_eval/annotation_frames.csv`. **The 438 keyframes themselves are the ground truth** — there is no dense interpolated GT. Tracker metrics are computed only at these frames (see `tracking_eval/PLAN.md` Phase 1.4 / Phase 4 for the rationale and citations). CVAT's Track-mode linear interpolation is used as a navigation/identity-tagging convenience only; interpolated frames are filtered out at export. Estimated annotation effort: **3–5 person-hours** (reduced from the previous 6–10 by dropping interpolation-drift QA).

## Tooling: CVAT (local Docker)

### Setup (one-time)

```sh
git clone https://github.com/cvat-ai/cvat.git
cd cvat
docker compose up -d
docker exec -it cvat_server bash -ic 'python manage.py createsuperuser'
# Visit http://localhost:8080
```

### Project & task setup

1. Create a project named `playclass-tracking-eval`.
2. Define a single label `bird` (no attributes needed — track identity is captured by Track membership, not by label).
3. Create one **Task per video** (5 tasks total). Upload the `.mp4` directly. Use default video-frame ingestion — **do not** convert to image set; that disables interpolation.
4. Use task name = `video_id` from the manifest (`C1G2_day_29`, etc.) so the exported zip names match downstream expectations.

## Annotation workflow

For each task:

1. Open the task. Switch to **Track mode** (used as a navigation / identity-tagging convenience; the interpolated frames it produces are filtered out at export — they are not GT).
2. For **bird 1**:
   - Jump to the first listed keyframe (CVAT toolbar → "Go to frame" → enter `frame_idx` from the CSV).
   - Draw a tight bbox around the bird with the rectangle tool — CVAT creates a new Track automatically and marks this frame as a keyframe.
   - Jump to the next listed keyframe. Drag/resize the bbox to match. CVAT auto-marks the frame as a keyframe on modification.
   - Repeat until the last listed frame for that video.
3. Repeat step 2 for **bird 2** and **bird 3**, each creating a separate Track.
4. Save (Ctrl+S) frequently.

**Visit only the frames listed in `annotation_frames.csv`. Do not add corrective keyframes on frames between listed ones — interpolation drift between keyframes is not a QA concern because interpolated frames are dropped before scoring.**

Three Tracks per video. CVAT assigns track IDs at export — the absolute values are irrelevant; HOTA/IDF1 are permutation-invariant.

## Annotation conventions

### Partial occlusion (one bird in front of another)
Annotate the **full estimated bbox** of each bird, including the occluded portion. MOTChallenge convention. Use neighbouring frames to infer the hidden body parts.

### Total occlusion (bird fully hidden, e.g., behind feeder)
Mark the bbox as **Outside** for those keyframes (CVAT shortcut **O**, or right-click → set `outside=true`). Resume the Track at the next listed keyframe where the bird is visible. Do **not** create a new Track on re-emergence — keep the same Track to preserve identity.

### Bird leaves frame
Mark the Track Outside. If it re-enters, resume on the same Track. (Rare in our recordings.)

### Identity ambiguity
If you cannot determine which bird is which after a long occlusion, re-watch the surrounding frames at reduced playback speed. If identity is genuinely unrecoverable, flag it to the project lead before guessing — guessing wrong creates a false ID switch that the tracker will be penalised for.

### Bbox quality
- Tight fit: full body (head, body, tail, feet) but no surrounding background.
- Consistent across keyframes — define "this bird's bbox" the same way each time so linear interpolation between keyframes stays on the bird.
- Pixel-perfect precision is not required; the evaluation IoU threshold is 0.5.

### Minimum visibility
There is no visibility threshold below which a bird is dropped. Partial visibility still gets a full estimated bbox. Only fully occluded / outside-frame cases use the Outside flag.

## Quality assurance

QA is **per-keyframe only**. Before exporting, walk through each of the 88 listed keyframes per video and confirm each bbox is on the correct bird with a tight fit. The full-video scrub-at-2×-for-interpolation-drift pass from earlier drafts of this document is **no longer required** under the sparse-GT scheme — interpolated frames are filtered out before scoring, so drift between keyframes has no effect on the reported metrics.

The keyframe schedule already brackets the top-3 longest occlusion periods per video, so the identity-ambiguous moments are already densified in the listed keyframes themselves; QA at those frames is where attention should land.

## Export

When all 5 tasks are annotated and QA-passed:

1. Task page → **Actions** → **Export task dataset**.
2. Format: **MOT 1.1**.
3. Download the zip; extract `gt/gt.txt`.
4. Rename to `<video_id>_cvat.txt` and place it somewhere convenient (e.g. `ext-data/output/results/tracker_benchmark/cvat_raw_export/`).
5. Run `tracking_eval/scripts/filter_mot_to_keyframes.py` to filter out CVAT's interpolated frames and keep only the rows whose `frame` is in `annotation_frames.csv` for the matching `video_id`. The filtered file is written to:
   `ext-data/output/results/tracker_benchmark/ground_truth/<video_id>.txt`

Final layout (after filter):

```
ext-data/output/results/tracker_benchmark/ground_truth/
├── C1G2_day_29.txt
├── C2G2_day_28.txt
├── C3G2_day_29.txt
├── C4G2_day_29.txt
└── C5G3_day_28.txt
```

MOTChallenge row format (unchanged):
```
frame, id, bb_left, bb_top, bb_width, bb_height, conf, -1, -1, -1
```
For ground truth, `conf = 1`. After filtering, sanity-check each `<video_id>.txt`:

- Exactly 3 unique IDs.
- Row count ≈ `n_keyframes × 3` minus any Outside-flagged entries (per `annotation_frames_summary.csv`).
- Frame indices form a subset of the `frame_idx` values listed for that `video_id` in `annotation_frames.csv` — no rows on intermediate frames.
