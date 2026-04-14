# Classification package (`src/classification/`)

Behaviour classification from dataset outputs. Uses PyTorch Lightning (pixi env `classifier`).

- **`models.py`**: Pure `nn.Module` backbones. `SimpleLinear` (linear probe), `SimpleMLP` (features MLP), `TemporalMLP` (gated attention pool), `TemporalCNNv2` (GELU bottleneck + conv). All temporal models expect `(B, F, D)` input. `MODEL_REGISTRY` maps model names to `(backbone_cls, temporal_flag)`.
- **`model_selection.py`**: Sklearn-style cross-validation splitters. `LOCO` (leave-one-cage-out, default) and `LOVO` (leave-one-video-out, deprecated). Both yield `(test_id, val_id)` pairs with deterministic circular rotation for val selection.
- **`datamodule.py`**: `BehaviourDataModule` loads data once via `prepare_data()`, then re-splits per fold via `set_split_groups(test_group, val_group)` + `setup()`. Exposes `video_ids`, `class_weights` (inverse-sqrt from train split), `n_classes`, `data_dim`, `label_encoder`. Features are median-imputed (2 rows with NaN pairwise distances from solo-bird windows) then z-score normalized at load time. Embeddings are mean-pooled by default; when `temporal=True`, segment-pooled via adaptive average pooling.
- **`trainer.py`**: `BehaviourClassifier(LightningModule)` wraps any backbone. Weighted CE loss with label smoothing (0.1), AdamW. Uses `torchmetrics.MetricCollection` with `MulticlassF1Score` (macro) + `MulticlassConfusionMatrix`, cloned with prefixes for train/val/test stages. `evaluate_fold` in `train.py` loads the best checkpoint manually and re-evaluates all splits (Lightning resets metrics after fit/test).
- **`stats.py`**: Aggregation. `aggregate_metrics()` dispatches to `aggregate_scalars()` (summary CSV with MEAN/STD/POOLED rows) and `aggregate_confusion_matrices()` (summed CMs as .txt + .npy for train/val/test, returns pooled F1 per split).

`LABEL_ORDER` and `DEFAULT_FPS` are defined in `src/_config.py`. `--exclude` removes a class. Training script: `script/train.py` (`pixi run -e classifier train`).
