# Classification Training

Runs in the `classifier` pixi environment. Requires dataset pipeline outputs (steps 1-3).

Default is LOCO (Leave-One-Cage-Out) cross-validation via `src/classification/model_selection.LOCO`. LOVO (Leave-One-Video-Out) is available via `--cv lovo` but deprecated. Val cage is auto-selected from the next cage in sorted circular order (cage-aware rotation to avoid environment leakage). A fresh model is created per fold.

Key classes: `LOCO` and `LOVO` in `src/classification/model_selection.py` (sklearn-style splitters), `BehaviourDataModule` in `src/classification/datamodule.py`.

```sh
# Features only (MLP, LOCO default)
pixi run -e classifier train --model mlp --input features --exclude social

# Mean-pooled embeddings (MLP)
pixi run -e classifier train --model mlp --input embeddings_250 --exclude social

# Features + embeddings combined
pixi run -e classifier train --model mlp --input features+embeddings --exclude social

# Temporal model on embeddings
pixi run -e classifier train --model temporal_mlp --input embeddings --exclude social

# Temporal hybrid mode (temporal embeddings + flat windowed features)
pixi run -e classifier train --model temporal_cnn2 --input features+embeddings --exclude social

# LOVO (deprecated)
pixi run -e classifier train --model mlp --input features --cv lovo

# Dry run (first fold, 1 batch, no checkpoints)
pixi run -e classifier train --model mlp --input features --dry-run
```

## LOCO output structure

```
data/eval/20260304_174946_mlp/
├── cfg.json                          # CLI args for reproducibility
├── fold_0_C1/                        # per-fold checkpoints + CSVLogger
│   ├── checkpoints/                  # ModelCheckpoint (best val_loss)
│   └── lightning_logs/version_0/     # CSVLogger (metrics.csv per epoch)
├── fold_1_C2/
├── ...
├── loco_summary.csv                  # scalar metrics per fold + MEAN/STD
├── loco_train_confusion_matrix.txt   # summed train CM
├── loco_val_confusion_matrix.txt     # summed val CM
├── loco_test_confusion_matrix.txt    # summed test CM (disjoint test sets)
└── loco_test_confusion_matrix.npy    # summed test CM as numpy array
```

## Model table

| `--model` | Backbone | Temporal |
|-----------|----------|----------|
| `linear` | `SimpleLinear` | No (mean-pool / windowed stats) |
| `mlp` | `SimpleMLP` | No (mean-pool / windowed stats) |
| `temporal_mlp` | `TemporalMLP` | Yes (segment-pooled, gated attention) |
| `temporal_cnn2` | `TemporalCNNv2` | Yes (GELU bottleneck + single conv) |

## Input specification

Input is specified separately via `--input`: `features`, `embeddings`, `embeddings_250`, `features+embeddings`, etc. Any `embeddings[_variant]` maps to `{name}.pt` in the dataset dir. Multiple embedding files can be concatenated: `features+embeddings+embeddings_union512` loads both and concatenates along the feature dimension (multi-scale). Non-temporal models mean-pool embeddings and use windowed feature stats; temporal models segment-pool embeddings and use binned feature tensors. Temporal models also support **hybrid mode**: e.g. `--model temporal_mlp --input features+embeddings` passes temporal embeddings through the backbone and concatenates flat windowed features before the classification head.

Additional training args: `--dropout` (override model dropout), `--d-hidden` (override hidden dim), `--label-smoothing` (CE label smoothing), `--feature-dropout` (dropout on flat features only), `--n-segments` (temporal bins, default 12).

## Classification results

For latest classification results and ablation summaries, see the v0.2.0 release:

```sh
gh release view v0.2.0 --repo prince-ravi-leow/chicken-behaviour-classifier
```

Detailed ablation tables are in `notes/ablation_v2.md`.
