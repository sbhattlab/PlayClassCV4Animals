"""LightningModule wrapper for behaviour classification backbones."""

import lightning as L
import torch
import torch.nn.functional as F
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassConfusionMatrix, MulticlassF1Score


def _build_metrics(n_classes: int) -> MetricCollection:
    """Metric set used for each stage. Add new metrics here."""
    return MetricCollection({
        "macro_f1": MulticlassF1Score(num_classes=n_classes, average="macro"),
        "confusion_matrix": MulticlassConfusionMatrix(num_classes=n_classes),
    })


# Keys that can't be logged as scalars via log_dict
_NON_SCALAR_METRICS = {"confusion_matrix"}


class BehaviourClassifier(L.LightningModule):
    """Wraps any backbone with weighted CE loss, AdamW, and torchmetrics.

    Parameters
    ----------
    backbone : nn.Module
        Any model from ``src.classification.models``.
    n_classes : int
        Number of output classes.
    lr : float
        Learning rate for AdamW.
    class_weights : Tensor | None
        Per-class weights for cross-entropy loss (length ``n_classes``).
    """

    def __init__(self, backbone, n_classes, lr=1e-3, class_weights=None):
        super().__init__()
        self.save_hyperparameters(ignore=["backbone", "class_weights"])
        self.backbone = backbone
        self.lr = lr

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

        metrics = _build_metrics(n_classes)
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

    def _log_scalars(self, metrics: MetricCollection, **kwargs):
        """Log only scalar metrics from a collection."""
        self.log_dict(
            {k: v for k, v in metrics.items()
             if k.split("_", 1)[-1] not in _NON_SCALAR_METRICS},
            **kwargs,
        )

    def _step(self, batch):
        logits = self.backbone(batch["data"])
        loss = F.cross_entropy(logits, batch["label"], weight=self.class_weights)
        return logits, loss

    def training_step(self, batch, batch_idx):
        logits, loss = self._step(batch)
        self.train_metrics(logits, batch["label"])
        self.log("train_loss", loss, prog_bar=True)
        self._log_scalars(self.train_metrics, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits, loss = self._step(batch)
        self.val_metrics(logits, batch["label"])
        self.log("val_loss", loss, prog_bar=True)
        self._log_scalars(self.val_metrics, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        logits, loss = self._step(batch)
        self.test_metrics(logits, batch["label"])
        self.log("test_loss", loss)
        self._log_scalars(self.test_metrics, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
