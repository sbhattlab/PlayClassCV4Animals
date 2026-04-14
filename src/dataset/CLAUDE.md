# Dataset package (`src/dataset/`)

## Features module (`src/dataset/features.py`)

Extracts handcrafted features from tracking masks. All functions are vectorized (no row-by-row iteration):

- **`extract_spatial_features`**: Batch `pycocotools.mask.area()` for mask areas, numpy array math for bbox metrics, batch `mask_util.decode()` + `scipy.ndimage.center_of_mass` for centroids. Masks grouped by `size` (pycocotools requirement) and chunked at 256 to cap memory.
- **`extract_temporal_features`**: `pandas.groupby().diff()` / `.shift()` for frame-to-frame deltas.
- **`extract_pairwise_features`**: Numpy broadcasting for `(M, M)` distance matrices per frame.
- **`summarize_features_by_window`**: Single `groupby().agg()` call with `[mean, std, min, max, median]`.

Feature set (`_FEATURE_COLS`): `mask_area`, `aspect_ratio`, `velocity`, `area_change_rate`, `min_dist_to_other`, `mean_dist_to_other`. Additionally `bbox_area` is output by spatial features as a debugging column (not in `_FEATURE_COLS`) for detecting saltatory bbox spikes.

## Embeddings package (`src/dataset/embeddings/`)

Embedding extraction from tracked objects. Three backends:

- **`dinov3.py`**: DINOv3 CLS-token extraction. Collects all bbox crops across windows, batch-processes through the model. Uses bfloat16.
- **`vjepa2.py`**: V-JEPA 2/2.1 video embeddings. Processes non-overlapping clips of `num_frames`, spatially mean-pools per timestep. Includes `VJEPA21Wrapper` for torch.hub models.
- **`videoprism.py`**: VideoPrism (JAX/Flax). Same clip-based approach as V-JEPA but uses JAX arrays.

All backends output `{(video_id, bird_id, window): Tensor(F_w, D)}`. Embedding filenames follow `embeddings_{backbone}_{size}[_{variant}].pt` (e.g. `embeddings_dinov3_vitl.pt`, `embeddings_vjepa21_vitb_temporal.pt`). Model size is inferred via `parse_model_size()` in `src/dataset/utils.py`.
