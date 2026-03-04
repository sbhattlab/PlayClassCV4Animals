"""Train a behaviour classifier on tracked chicken data.

Always runs full LOVO (Leave-One-Video-Out) cross-validation.

Usage::

    # Full LOVO cross-validation
    pixi run -e classifier train \\
        --tracking-dir data/tracking/20260225_214929_sam3_hf \\
        --model mlp

    # Dry run — first fold, 1 batch, no checkpoints
    pixi run -e classifier train \\
        --tracking-dir data/tracking/20260225_214929_sam3_hf \\
        --model mlp --dry-run
"""

import gc
import json
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from loguru import logger

from src._config import DEFAULT_CHECKPOINT_DIR
from src.classification.datamodule import BehaviourDataModule
from src.classification.models import MODEL_REGISTRY
from src.classification.stats import aggregate_metrics
from src.classification.trainer import BehaviourClassifier


def parse_args():
    parser = ArgumentParser(description="Train behaviour classifier.")
    parser.add_argument("--tracking-dir", type=Path, required=True)
    parser.add_argument("--model", type=str, choices=MODEL_REGISTRY, default="mlp")
    parser.add_argument(
        "--exclude", type=str, default=None, help="Exclude videos with this substring"
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--patience", type=int, default=5, help="EarlyStopping patience"
    )
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
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def select_val_video(test_video: str, all_videos: list[str]) -> str:
    """Pick val video from the next cage (cage-aware rotation).

    Extracts cage prefix (first 2 chars), rotates to next cage in sorted order,
    and picks the first video from that cage.
    """
    cage_to_videos: dict[str, list[str]] = {}
    for v in sorted(all_videos):
        cage_to_videos.setdefault(v[:2], []).append(v)
    cages = sorted(cage_to_videos)
    test_cage = test_video[:2]
    next_cage = cages[(cages.index(test_cage) + 1) % len(cages)]
    return cage_to_videos[next_cage][0]


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
        callbacks=[
            EarlyStopping(
                monitor="val_loss", patience=training_args["patience"], mode="min"
            ),
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


def evaluate_fold(
    trainer: L.Trainer,
    model: BehaviourClassifier,
    dm: BehaviourDataModule,
) -> dict:
    """Collect all metrics for the fold (train, val, test)."""
    result = {"test_video": dm.test_video, "val_video": dm.val_video}

    # Train/val metrics (accumulated during fit)
    for collection in (model.train_metrics, model.val_metrics):
        for k, v in collection.compute().items():
            if v.ndim == 0:
                result[k] = v.item()
            else:
                result[k] = v.cpu().numpy().astype(int)

    # Test metrics
    trainer.test(model, dm, ckpt_path="best")
    for k, v in model.test_metrics.compute().items():
        if v.ndim == 0:
            result[k] = v.item()
        else:
            result[k] = v.cpu().numpy().astype(int)

    return result


def cleanup_fold(trainer, model):
    """Free GPU memory and finalize wandb between folds."""
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# LOVO orchestrator
# ---------------------------------------------------------------------------


def run_lovo(dm, backbone_cls, run_dir, dry_run=False, **training_args) -> list[dict]:
    """Run full LOVO loop across all videos. Returns fold results."""
    all_videos = dm.video_ids
    logger.info(f"LOVO: {len(all_videos)} videos — {all_videos}")

    fold_results = []

    for fold_idx, test_video in enumerate(all_videos):
        # Set test and val videos for this fold
        val_video = select_val_video(test_video, all_videos)
        logger.info(f"Fold {fold_idx}: test={test_video}, val={val_video}")

        # Update data splits
        dm.set_split_groups(test_video=test_video, val_video=val_video)
        dm.setup()

        # Update model
        model = BehaviourClassifier(
            backbone=backbone_cls(dm.data_dim, dm.n_classes),
            n_classes=dm.n_classes,
            lr=training_args["lr"],
            class_weights=dm.class_weights,
        )

        # Logger and checkpoint dir for this fold
        fold_dir = run_dir / f"fold_{fold_idx}_{test_video}"
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

    # Timestamped run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(DEFAULT_CHECKPOINT_DIR) / f"{timestamp}_{args.model}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save CLI args for reproducibility
    cfg = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    with open(run_dir / "cfg.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    # Load model class and arguments
    backbone_cls, backbone_args = MODEL_REGISTRY[args.model]

    # Load dataset
    dm = BehaviourDataModule(
        tracking_dir=args.tracking_dir,
        batch_size=args.batch_size,
        exclude=args.exclude,
        **backbone_args,
    )
    dm.prepare_data()

    # Training args
    accelerator, devices = parse_device(args.device)
    training_args = {
        "epochs": args.epochs,
        "patience": args.patience,
        "lr": args.lr,
        "accelerator": accelerator,
        "devices": devices,
    }

    fold_results = run_lovo(
        dm, backbone_cls, run_dir, dry_run=args.dry_run, **training_args
    )
    if fold_results:
        labels = list(dm.label_encoder.lab2ind.keys())
        aggregate_metrics(fold_results, run_dir, labels)


if __name__ == "__main__":
    main()
