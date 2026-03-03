# Dataset construction pipeline

## Workflow

```
    [file]  function()  <column>

      [tracking_outputs.parquet]      [Registration protocols (.xlsx)]
                  |                                |
                  v                                v
         detect_tracking_issues()           process_labels()
                  |                           |          |
                  v                           v          v
         [tracking_issues.json]            labels    bird_info
                                              |
                                              v
                                     resolve_dual_groups()    <behav_label>
                  |
                  v
         prefill_postprocessing()
                  |
                  v
      [tracking_postprocessing.json]  <--- USER REVIEWS & EDITS
                  |
                  v
            process_tracks()
                  |
               trim()                       <tracking_id>
                  |
            merge_id_on_switch()            <tracking_id>
                  |
            match_bird_ids()          <tracking_id> → <bird_id>
                  |
                  v
            align_labels()                    <bird_id>
                  |
                  v
            assign_windows()             <bird_id> + <window>
                  |
                  v
         filter_incomplete_windows()    drop windows below coverage threshold
                  |
                  v
         extract_mask_features()        spatial + temporal + pairwise
                  |
                  v
         summarize_features_by_window()    per (video_id, bird_id, window)
                  |
                  v
         [dataset_tracks.parquet] + [dataset_labels.parquet]
         [all_features.parquet]  + [dataset_features.parquet]

                  |  (GPU step, separate script)
                  v
         extract_embeddings()              DINOv3 CLS-token per frame
                  |
                  v
         [all_embeddings.pt]
```

### Column naming

The pipeline uses two ID columns:

| Column | Used by | Meaning |
|--------|---------|---------|
| `tracking_id` | `trim`, `merge_id_on_switch`, `match_bird_ids` | Tracker-assigned integer ID |
| `bird_id` | `align_labels`, `assign_windows`, dataset parquets | Protocol ID (real bird identity) |

`match_bird_ids()` is the boundary: it remaps `tracking_id` values to protocol
IDs and renames the column to `bird_id`.

### Step 1: Detect issues (automatic)

`tracking_issues.py` scans `tracking_outputs.parquet` and flags three issue types:

| Issue type   | What it detects                                   |
|--------------|---------------------------------------------------|
| `id_switch`  | Frames where the set of tracked object IDs changes |
| `overlap`    | Frame ranges where two IDs share overlapping masks |
| `low_score`  | Per-ID stretches where `tracker_score` stays low   |

Output: `tracking_issues.json` (read-only diagnostic, not consumed by postprocessing).

### Step 2: Prefill actions (automatic)

`prefill_postprocessing()` converts each detected issue into a postprocessing entry
in `tracking_postprocessing.json`:

| Issue             | Prefilled action                                              |
|-------------------|---------------------------------------------------------------|
| `overlap`         | `{"type": "trim", "cause": "overlap", "from": F, "to": F, "id": N}` — `id` is the one with the lower mean `tracker_score` in the overlap range |
| `low_score`       | `{"type": "trim", "cause": "low_score", "from": F, "to": F, "id": N}` — `id` is the one flagged with low score |
| `id_switch`       | `{"type": "id_switch", "frame": F, "from": N, "to": null}` — user fills in `to` |
| (per bird)        | `{"type": "id_match", "protocol_id": "NNNN", "description": "...", "tracking_id": null, "frame": null}` — user fills in `tracking_id` and `frame` |

### Step 3: User review (manual)

Edit `tracking_postprocessing.json`:

- Fill in `"to"` values for `id_switch` entries.
  Semantics: "at `frame`, the chicken that **was** `from` gets reassigned to `to`."
  Rows before the switch frame are renamed; rows at or after keep their ID
  (a different chicken may now occupy the `from` slot). Remaps at the same
  frame are applied simultaneously (swaps work correctly).
- Fill in `"tracking_id"` and `"frame"` for `id_match` entries.
  `tracking_id` must be the **post-merge surviving ID** (after all `id_switch`
  remaps), not the original tracker-assigned ID. `frame` is a reference frame
  where that ID is visible, used for verification.
- Review `"id"` in overlap trims (swap if the wrong ID was picked)
- Remove trim entries for issues that don't need fixing
- Add global trims, e.g. `{"type": "trim", "from": 0, "to": 1592}` to remove the first 63 seconds

### Step 4: Postprocess (automatic)

`process_tracks()` reads the JSON and applies three operations in order:

1. **`trim()`** — drops track rows within each trim's `[from, to]` frame range.
   If `id` is present, only rows for that tracking ID are dropped; otherwise all
   rows in the range are removed. Labels whose time falls in the range are always
   dropped. The `cause` field is logged but does not affect behaviour.

2. **`merge_id_on_switch()`** — for each `id_switch` entry, renames rows with
   `tracking_id == from` **before** the switch `frame` to `to`. Remaps at the
   same frame are grouped and applied simultaneously (masks computed before
   renames) so that swaps work correctly. Across frames, groups are applied
   earliest-first.

3. **`match_bird_ids()`** — for each `id_match` entry, renames all rows with the
   post-merge `tracking_id` to the `protocol_id`. Verifies the ID exists at the
   reference `frame` before applying. Renames the column from `tracking_id` to
   `bird_id`.

### Step 5: Align labels, assign windows & extract features (automatic)

After postprocessing, the build script aligns labels with tracks, assigns
temporal windows, and extracts features:

1. **`resolve_dual_groups()`** — resolves multi-valued `behav_group` entries
   (e.g. `"locomotor, worm"`) to a single class in a new `behav_label` column.
   Picks the rarest component class, since rare behaviours are more informative
   as classification targets.

2. **`align_labels()`** — computes the min/max frame per `(video_id, bird_id)` in
   the tracks, converts to time, and drops any label rows that fall outside that
   range or have no matching bird. This removes labels for time periods where no
   tracking data exists.

3. **`assign_windows()`** — each label defines a temporal window `(prev_time, time]`.
   Track frames whose time falls in that interval share the same `window` index.
   Both tracks and labels receive a `window` column, numbered per
   `(video_id, bird_id)` starting from 0.

4. **`filter_incomplete_windows()`** — computes the actual vs expected frame count
   per `(video_id, bird_id, window)` and drops windows below a coverage fraction
   (default 50%). Expected frames are derived from the median label spacing and
   per-video FPS. This prevents meaningless feature statistics from windows left
   with very few frames after trimming. Controlled by `--min-window-coverage`.

5. **`extract_mask_features()`** — extracts spatial (mask area, bbox, centroid),
   temporal (velocity, area change), and pairwise (nearest-neighbor distance)
   features per frame. Saved as `all_features.parquet`.

6. **`summarize_features_by_window()`** — aggregates per-frame features into
   per-window summary statistics (mean, std, min, max, median) plus frame count.
   Saved as `dataset_features.parquet` (one row per video/bird/window).

## Known issues

**FPS mismatch**: `_time` hints in `tracking_postprocessing.json` (and all
`fmt_time` output) assume the FPS from `yolo_scan_summary.parquet` (fallback:
25.0). Some videos have an actual FPS of ~24 fps, causing timestamps to drift
(~4%, e.g. ~14 s off at the 6-minute mark). Always cross-reference frame
indices, not `_time` strings, when matching events to the source video. To
check the true FPS, probe the video with OpenCV:

```python
import cv2
cap = cv2.VideoCapture(str(video_path))
print(cap.get(cv2.CAP_PROP_FPS))
cap.release()
```

## Running

```sh
# First run: generates tracking_issues.json + tracking_postprocessing.json
pixi run -e sam3-hf build_dataset \
    --tracking-dir data/tracking/20260225_214929_sam3_hf \
    --label-dir data/labels

# User fills in "to" values and reviews trim entries

# Second run: applies postprocessing, builds dataset
pixi run -e sam3-hf build_dataset \
    --tracking-dir data/tracking/20260225_214929_sam3_hf \
    --label-dir data/labels
```

Output: `dataset_tracks.parquet`, `dataset_labels.parquet`, `all_features.parquet`, and `dataset_features.parquet` saved to the tracking dir.

### Embedding extraction (GPU)

After the main pipeline completes, run the embedding extraction script separately
(requires GPU + DINOv3 model):

```sh
pixi run -e sam3-hf extract-embeddings \
    --tracking-dir data/tracking/20260225_214929_sam3_hf \
    --video-dir video-data/batch
```

Output: `all_embeddings.pt` (per-frame, per-bird embeddings) saved to the tracking dir.

## Module overview

| Module                       | Role                                                    |
|------------------------------|---------------------------------------------------------|
| `tracking_issues.py`        | Detection: ID switches, mask overlaps, low-score periods |
| `tracking_postprocessing.py` | Remediation: prefill, trim, ID remap, match; label alignment & windowing |
| `labels.py`                  | Parse behaviour labels + bird info from Excel; resolve dual groups |
| `features.py`                | Handcrafted mask features (spatial, temporal, pairwise)  |
| `embeddings.py`              | DINOv3 CLS-token embedding extraction + window summarization |
| `utils.py`                   | Shared helpers: `fmt_time`, `get_video_fps`, `_decode_rle_mask` |

## Example `tracking_postprocessing.json`

```json
[
    {"type": "trim", "cause": "overlap", "from": 106, "to": 1178, "id": 4, "_time": "00:04.24-00:47.12"},
    {"type": "trim", "cause": "low_score", "from": 6657, "to": 6831, "id": 0, "_time": "04:26.28-04:33.24"},
    {"type": "id_switch", "frame": 1592, "from": 7, "to": 0, "_time": "01:03.67"},
    {"type": "id_switch", "frame": 1592, "from": 5, "to": 4, "_time": "01:03.67"},
    {"type": "id_match", "protocol_id": "2664", "description": "White, blue head", "tracking_id": 0, "frame": 5000},
    {"type": "id_match", "protocol_id": "2670", "description": "Black", "tracking_id": 4, "frame": 5000}
]
```

Fields starting with `_` (e.g. `_time`) are human-readable hints ignored by the code.
