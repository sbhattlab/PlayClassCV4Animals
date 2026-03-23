"""Train a behaviour classifier on tracked chicken data.

Supports LOCO (leave-one-cage-out, default) and LOVO (leave-one-video-out,
deprecated) cross-validation via ``--cv``.

Usage::

    # Features only (MLP, LOCO default)
    pixi run -e classifier train --model mlp --input features

    # LOVO (v0.1.0)
    pixi run -e classifier train --model mlp --input features --cv lovo

    # Mean-pooled embeddings (MLP)
    pixi run -e classifier train --model mlp --input embeddings

    # Features + embeddings combined
    pixi run -e classifier train --model mlp --input features+embeddings

    # Temporal model on embeddings
    pixi run -e classifier train --model temporal_mlp --input embeddings

    # Dry run — first fold, 1 batch, no checkpoints
    pixi run -e classifier train --model mlp --input features --dry-run
"""

import gc
import json
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import logging

import lightning as L
import torch

# Suppress noisy Lightning logs (GPU info, tips, LOCAL_RANK)
for _name in (
    "lightning.pytorch.utilities.rank_zero",
    "lightning.pytorch.accelerators.cuda",
    "lightning.fabric.utilities.distributed",
):
    logging.getLogger(_name).setLevel(logging.WARNING)
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from loguru import logger

from src._config import DEFAULT_CHECKPOINT_DIR, DEFAULT_DATASET_DIR
from src.classification.datamodule import BehaviourDataModule
from src.classification.model_selection import LOCO, LOVO
from src.classification.models import MODEL_REGISTRY
from src.classification.stats import aggregate_metrics
from src.classification.trainer import BehaviourClassifier

torch.set_float32_matmul_precision("high")


def parse_args():
    parser = ArgumentParser(description="Train behaviour classifier.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Directory containing dataset files (default: %(default)s)",
    )
    parser.add_argument("--model", type=str, choices=MODEL_REGISTRY, required=True)
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help=(
            "Input data: 'features', 'embeddings', 'embeddings_250', "
            "'features+embeddings', etc."
        ),
    )
    parser.add_argument(
        "--exclude", type=str, default=None, help="Exclude a class (e.g. social)"
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device (e.g. cuda:0, cuda:1, cpu)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run 1 train + 1 val batch, first fold only",
    )
    parser.add_argument(
        "--n-segments",
        type=int,
        default=12,
        help="Number of temporal bins for segment pooling (default: 12)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="Override model dropout (default: use model default, typically 0.3)",
    )
    parser.add_argument(
        "--d-hidden",
        type=int,
        default=None,
        help="Override model hidden dimension (default: use model default, typically 128)",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.1,
        help="Label smoothing factor (default: 0.1, 0=none)",
    )
    parser.add_argument(
        "--feature-dropout",
        type=float,
        default=0.0,
        help="Dropout on flat features before classification (default: 0.0)",
    )
    parser.add_argument(
        "--class-weights",
        type=str,
        default="inv-sqrt",
        choices=["inv-sqrt", "inv", "none"],
        help="Class weighting scheme (default: inv-sqrt)",
    )
    parser.add_argument(
        "--cv",
        type=str,
        default="loco",
        choices=["lovo", "loco"],
        help="Cross-validation: loco (leave-one-cage-out, default) or lovo (leave-one-video-out, deprecated)",
    )
    return parser.parse_args()


def parse_input(input_str):
    """Parse --input string into data loading flags.

    Returns (use_features, use_embeddings, embeddings_files).

    Examples::

        "features"                          → (True,  False, [])
        "embeddings"                        → (False, True,  ["embeddings.pt"])
        "embeddings_250"                    → (False, True,  ["embeddings_250.pt"])
        "features+embeddings"               → (True,  True,  ["embeddings.pt"])
        "features+embeddings+embeddings_union512" → (True, True, ["embeddings.pt", "embeddings_union512.pt"])
    """
    parts = input_str.split("+")
    use_features = False
    use_embeddings = False
    embeddings_files = []

    for part in parts:
        if part == "features":
            use_features = True
        elif part.startswith("embeddings"):
            use_embeddings = True
            embeddings_files.append(f"{part}.pt")
        else:
            raise ValueError(
                f"Unknown input component: '{part}'. "
                "Expected 'features' or 'embeddings[_variant]'."
            )

    if not use_features and not use_embeddings:
        raise ValueError(
            "--input must include 'features' and/or 'embeddings[_variant]'"
        )

    return use_features, use_embeddings, embeddings_files


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_device(device: str) -> tuple:
    """Parse device string into Lightning accelerator/devices args."""
    if "cuda" in device:
        return "gpu", [int(device.split(":")[-1])]
    return "cpu", "auto"


# ---------------------------------------------------------------------------
# Train / evaluate a single fold
# ---------------------------------------------------------------------------


def train_fold(
    *,
    dm: BehaviourDataModule,
    model: BehaviourClassifier,
    pl_logger,
    fold_dir: Path,
    fold_idx: int = 0,
    **training_args,
) -> tuple[L.Trainer, BehaviourClassifier]:
    """Fit a model on the current DM split.

    Expects ``dm.setup()`` to have been called already.
    Returns ``(trainer, model)`` for downstream evaluation.
    """
    fold_dir.mkdir(parents=True, exist_ok=True)

    trainer = L.Trainer(
        accelerator=training_args["accelerator"],
        devices=training_args["devices"],
        max_epochs=training_args["epochs"],
        logger=pl_logger,
        enable_model_summary=(fold_idx == 0),
        callbacks=[
            ModelCheckpoint(
                dirpath=fold_dir / "checkpoints",
                monitor="val_loss",
                mode="min",
                save_top_k=1,
            ),
        ],
        fast_dev_run=training_args["dry_run"],
    )

    trainer.fit(model, dm)
    return trainer, model


def _collect_metrics(result, collection):
    """Extract scalars and arrays from a MetricCollection into result dict."""
    for k, v in collection.compute().items():
        if v.ndim == 0:
            result[k] = v.item()
        else:
            result[k] = v.cpu().numpy().astype(int)


def _eval_dataloader(model, dataloader, metrics):
    """Run model on a dataloader and update metrics (no grad)."""
    metrics.reset()
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            flat = batch.get("flat")
            if flat is not None:
                logits = model.backbone(batch["data"], flat=flat)
            else:
                logits = model.backbone(batch["data"])
            metrics.update(logits, batch["label"])


def evaluate_fold(
    trainer: L.Trainer,
    model: BehaviourClassifier,
    dm: BehaviourDataModule,
) -> dict:
    """Collect all metrics for the fold (train, val, test).

    Lightning resets epoch metrics after fit() and test(), so we
    load the best checkpoint manually and re-evaluate all splits.
    """
    result = {"test_id": dm.test_id, "val_id": dm.val_id}

    # Load best checkpoint (trainer.test resets metrics, so we do it manually)
    best_path = trainer.checkpoint_callback.best_model_path
    if best_path:
        ckpt = torch.load(best_path, weights_only=True)
        model.load_state_dict(ckpt["state_dict"])

    # Evaluate all splits with best model
    _eval_dataloader(model, dm.test_dataloader(), model.test_metrics)
    _collect_metrics(result, model.test_metrics)

    _eval_dataloader(model, dm.train_dataloader(), model.train_metrics)
    _collect_metrics(result, model.train_metrics)

    _eval_dataloader(model, dm.val_dataloader(), model.val_metrics)
    _collect_metrics(result, model.val_metrics)

    return result


def cleanup_fold(trainer, model):
    """Free GPU memory and finalize wandb between folds."""
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# CV orchestrator
# ---------------------------------------------------------------------------


def run_cv(dm, backbone_cls, run_dir, dry_run=False, **training_args) -> list[dict]:
    """Run cross-validation loop (LOCO or LOVO). Returns fold results."""
    folds = list(dm.splitter.split(dm.video_ids))
    logger.info(f"{dm.splitter.__class__.__name__}: {len(folds)} folds")

    fold_results = []

    for fold_idx, (test_id, val_id) in enumerate(folds):
        logger.info(f"Fold {fold_idx}: test={test_id}, val={val_id}")

        dm.set_fold(test_id=test_id, val_id=val_id)
        dm.setup()

        backbone_kwargs = {}
        if dm.flat_dim > 0:
            backbone_kwargs["d_flat"] = dm.flat_dim
        if training_args.get("dropout") is not None:
            backbone_kwargs["dropout"] = training_args["dropout"]
        if training_args.get("d_hidden") is not None:
            backbone_kwargs["d_hidden"] = training_args["d_hidden"]
        model = BehaviourClassifier(
            backbone=backbone_cls(dm.data_dim, dm.n_classes, **backbone_kwargs),
            n_classes=dm.n_classes,
            lr=training_args["lr"],
            class_weights=dm.class_weights,
            label_smoothing=training_args.get("label_smoothing", 0.1),
            feature_dropout=training_args.get("feature_dropout", 0.0),
        )

        fold_dir = run_dir / f"fold_{fold_idx}_{test_id}"
        fold_logger = CSVLogger(save_dir=fold_dir)

        trainer, model = train_fold(
            dm=dm,
            model=model,
            pl_logger=fold_logger,
            fold_dir=fold_dir,
            dry_run=dry_run,
            **training_args,
        )

        if dry_run:
            logger.info("Dry run: stopping after first fold")
            break

        result = evaluate_fold(trainer, model, dm)
        cleanup_fold(trainer, model)
        fold_results.append(result)

    return fold_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()

    # Parse input specification
    use_features, use_embeddings, embeddings_files = parse_input(args.input)
    backbone_cls, temporal = MODEL_REGISTRY[args.model]

    # Timestamped run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(DEFAULT_CHECKPOINT_DIR) / f"{timestamp}_{args.model}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save CLI args for reproducibility
    cfg = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    with open(run_dir / "cfg.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    # Load dataset
    splitter = LOCO() if args.cv == "loco" else LOVO()
    dm = BehaviourDataModule(
        dataset_dir=args.dataset_dir,
        splitter=splitter,
        batch_size=args.batch_size,
        exclude=args.exclude,
        use_features=use_features,
        use_embeddings=use_embeddings,
        temporal=temporal,
        embeddings_files=embeddings_files or ["embeddings.pt"],
        n_segments=args.n_segments,
    )
    dm.class_weight_scheme = args.class_weights
    dm.prepare_data()

    # Training args
    accelerator, devices = parse_device(args.device)
    training_args = {
        "epochs": args.epochs,
        "lr": args.lr,
        "dropout": args.dropout,
        "d_hidden": args.d_hidden,
        "label_smoothing": args.label_smoothing,
        "feature_dropout": args.feature_dropout,
        "accelerator": accelerator,
        "devices": devices,
    }

    fold_results = run_cv(
        dm, backbone_cls, run_dir, dry_run=args.dry_run, **training_args
    )
    if fold_results:
        labels = list(dm.label_encoder.lab2ind.keys())
        aggregate_metrics(fold_results, run_dir, labels, prefix=args.cv)


if __name__ == "__main__":
    main()
