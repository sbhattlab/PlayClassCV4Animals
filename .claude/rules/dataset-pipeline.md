# Dataset Pipeline

Three-step pipeline. Steps 2 and 3 are independent and can run in parallel (different resources).

```sh
# 1. Labels, postprocessing, windows (fast, ~seconds)
pixi run -e dataset build_dataset

# 2. Mask features (slow, ~2 min, CPU-only)
pixi run -e dataset extract_features

# 3. DINOv3 embeddings (slow, GPU required)
pixi run -e embeddings extract_embeddings \
    --video-dir data/video/batch
```

All scripts default to `data/postprocessing/` (input) and `data/dataset/` (output). Override with `--tracking-dir` / `--dataset-dir` / `--output-dir`.

## Step 1: `build_dataset`

`script/build_dataset.py` discovers all `tracking_outputs.parquet` files recursively under `data/postprocessing/` (across day directories), parses labels from Excel, detects issues, applies postprocessing, assigns windows, and filters incomplete windows. On first run it generates `tracking_postprocessing.json` templates; manually fill in `to` (id_switch) and `tracking_id` (id_match) fields, then rerun.

Per-video JSON files (`tracking_postprocessing.json`) support three entry types:

- **`trim`** -- Remove track rows in a frame range, optionally scoped to one ID. Causes: `overlap`, `low_score`, `merged_object`.
- **`id_switch`** -- Remap `from` ID to `to` ID for all rows before `frame` (merges split tracks).
- **`id_match`** -- Rename tracker `tracking_id` to real `protocol_id` (bird identity).

Output: `tracks.parquet` + `labels.parquet` in `data/dataset/`.

## Step 2: `extract_features`

`script/extract_features.py` reads `tracks.parquet` from `data/dataset/`, decodes RLE masks, computes per-frame spatial/temporal/pairwise features, then summarizes per window. Also builds temporal feature tensors (per-frame features binned per window, same key format as embeddings).

Feature set (19 features): `mask_area`, `aspect_ratio`, `velocity`, `acceleration`, `velocity_autocorr`, `area_change_rate`, `min_dist_to_other`, `mean_dist_to_other`, plus additional spatial/temporal derivatives.

Output: `features_all.parquet` (per-frame) + `features_windowed.parquet` (per-window summaries) + `features_binned.pt` (temporal feature tensors keyed by `(video_id, bird_id, window)`).

## Step 3: `extract_embeddings`

`script/extract_embeddings.py` reads `tracks.parquet` from `data/dataset/`, crops bbox regions from video frames, and extracts DINOv3 CLS-token embeddings per (video_id, bird_id, window).

```sh
pixi run -e embeddings python -m script.extract_embeddings \
    --video-dir data/video --device 0

# Alternative backbone (ViT-B)
pixi run -e embeddings python -m script.extract_embeddings \
    --video-dir data/video --device 0 \
    --model-name facebook/dinov3-vitb16-pretrain-lvd1689m

# Custom resolution
pixi run -e embeddings python -m script.extract_embeddings \
    --video-dir data/video --device 0 \
    --resolution 256
```

Output filename encodes variant: `embeddings.pt` (default), `embeddings_vitb.pt`, `embeddings_r256.pt`. Dict keyed by `(video_id, bird_id, window)` with `Tensor(F_w, D)` values.

## Step 3b: Video embeddings (`extract_embeddings_vjepa2` / `extract_embeddings_videoprism`)

Two scripts for video-model embedding extraction. Both use shared `CROP_MODES` from `src/dataset/crops.py` (`--crop-mode`: bbox, plain256, union512, darken512, roi512).

- **`script/extract_embeddings_vjepa2.py`** (embeddings env) -- V-JEPA 2 embeddings. Feeds K frames as a video clip through the backbone. Supports `--temporal` (per-timestep spatial-mean-pool).
- **`script/extract_embeddings_videoprism.py`** (videoprism env, JAX) -- VideoPrism embeddings. Supports `--temporal`, `--raw` (full patch tokens).

```sh
# V-JEPA 2 temporal (embeddings env)
pixi run -e embeddings python -m script.extract_embeddings_vjepa2 \
    --video-dir data/video/batch data/video/batch2 --device 0 --temporal

# VideoPrism temporal (videoprism env)
pixi run -e videoprism extract_videoprism \
    --video-dir data/video/batch data/video/batch2 --device 0 --temporal

# VideoPrism raw tokens for trainable pooler (~88 GB)
pixi run -e videoprism extract_videoprism \
    --video-dir data/video/batch data/video/batch2 --device 0 --raw
```

Output: `embeddings_vjepa2_vitl.pt`, `embeddings_videoprism_temporal.pt`, `embeddings_videoprism_raw.pt`, etc.

## Dataset output schemas

- `tracks.parquet` -- cleaned tracks with window column
- `labels.parquet` -- aligned behaviour labels
- `features_all.parquet` -- per-frame mask features
- `features_windowed.parquet` -- per-window feature summaries
- `embeddings.pt` -- DINOv3 embeddings keyed by `(video_id, bird_id, window)`
