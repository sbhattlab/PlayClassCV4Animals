"""BehaviourDataset and BehaviourDataModule for classification training."""

import warnings
from pathlib import Path

import lightning as L

# Data is fully in-memory tensors; num_workers won't help
warnings.filterwarnings("ignore", ".*does not have many workers.*")
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from loguru import logger
from torch.utils.data import DataLoader, Dataset, Subset

from src._config import LABEL_ORDER
from src.classification.model_selection import LOO

_KEY_COLS = {"video_id", "bird_id", "window"}
_SKIP_COLS = _KEY_COLS | {"n_frames"}


class LabelEncoder:
    """Encode categorical labels as integers with a fixed order.

    Parameters
    ----------
    classes : list[str]
        Ordered class names.  Index 0 -> first element, etc.
    """

    def __init__(self, classes):
        self.lab2ind = {l: i for i, l in enumerate(classes)}
        self.ind2lab = {i: l for i, l in enumerate(classes)}

    def encode(self, labels):
        """Encode label string(s) -> integer index(es)."""
        if isinstance(labels, str):
            return self.lab2ind[labels]
        return [self.lab2ind[l] for l in labels]

    def decode(self, indices):
        """Decode integer index(es) -> label string(s)."""
        if isinstance(indices, int):
            return self.ind2lab[indices]
        return [self.ind2lab[i] for i in indices]


def adaptive_segment_pool1d(x, n_segments):
    """Reduce (T, D) -> (K, D) via adaptive average pooling."""
    # adaptive_avg_pool1d expects (B, C, L) — add batch dim, treat D as channels
    x = x.unsqueeze(0).transpose(1, 2)  # (1, D, T)
    # pylint: disable=not-callable
    x = F.adaptive_avg_pool1d(x, n_segments)  # (1, D, K)
    # pylint: enable=not-callable
    return x.transpose(1, 2).squeeze(0)  # (K, D)


class BehaviourDataset(Dataset):
    """Behaviour classification dataset.

    Builds a single ``data`` tensor from the requested combination of features
    and embeddings.  ``__getitem__`` returns ``{"data": Tensor, "label": int}``.

    Parameters
    ----------
    dataset_dir : str | Path
        Directory containing the dataset files.
    use_features : bool
        Load handcrafted features from ``features_windowed.parquet``.
    use_embeddings : bool
        Load embeddings from ``embeddings.pt``.
    temporal : bool
        If True, keep embeddings as sequences for temporal models.
        If False, mean-pool to ``(D,)``.
    n_segments : int
        Number of segments for adaptive average pooling when ``temporal=True``.
        Each segment represents ``1/n_segments`` of the window duration.
    exclude : str | list[str] | None
        Label(s) to exclude (e.g. ``"social"``).
    """

    def __init__(
        self,
        dataset_dir,
        use_features=True,
        use_embeddings=False,
        temporal=False,
        n_segments=12,
        exclude=None,
        embeddings_file="embeddings.pt",
        embeddings_files=None,
    ):
        dataset_dir = Path(dataset_dir)

        self.n_segments = n_segments
        # Support both single file (legacy) and list of files
        if embeddings_files is not None:
            self.embeddings_files = embeddings_files
        else:
            self.embeddings_files = [embeddings_file]

        # 1. Load data
        labels_df = pd.read_parquet(dataset_dir / "labels.parquet")
        self.data, self.flat = self.load_data(
            dataset_dir, labels_df, use_features, use_embeddings, temporal
        )

        # 2. Exclude data from unwanted labels (if specified)
        label_order = LABEL_ORDER.copy()
        if exclude is not None:
            exclude_list = [exclude] if isinstance(exclude, str) else exclude
            label_order = [l for l in LABEL_ORDER if l not in exclude_list]
            mask = ~labels_df["behav_label"].isin(exclude_list)
            labels_df = labels_df[mask].reset_index(drop=True)
            self.data = self.data[mask.values]
            if self.flat is not None:
                self.flat = self.flat[mask.values]

        # 3. Encode labels
        self.label_encoder = LabelEncoder(label_order)
        self.n_classes = len(label_order)

        # 4. Build label tensor
        self.labels = torch.tensor(
            self.label_encoder.encode(labels_df["behav_label"].tolist()),
            dtype=torch.long,
        )
        self.video_ids = labels_df["video_id"].tolist()
        self.data_dim = self.data.shape[-1]
        self.flat_dim = self.flat.shape[-1] if self.flat is not None else 0

    def load_data(self, dataset_dir, labels_df, use_features, use_embeddings, temporal):
        """Load data and return (data, flat).

        ``flat`` is non-None only in hybrid mode: temporal model with
        windowed features alongside temporal embeddings.

        Returns
        -------
        data : Tensor
            Main input — flat (N, D) or temporal (N, K, D).
        flat : Tensor | None
            Windowed features (N, F) for hybrid mode, else None.
        """
        assert (
            use_features or use_embeddings
        ), "At least one of use_features or use_embeddings must be True"

        if temporal:
            # Temporal embeddings as main data
            temporal_parts = []
            if use_embeddings:
                temporal_parts.append(
                    self._load_embeddings(dataset_dir, labels_df, temporal)
                )
            if use_features and not use_embeddings:
                # Features-only temporal: use binned features as main data
                return self._load_temporal_features(dataset_dir, labels_df), None
            if use_features and use_embeddings:
                # Hybrid: temporal embeddings + flat windowed features
                data = temporal_parts[0]
                flat = self._load_windowed_features(dataset_dir)
                return data, flat
            return temporal_parts[0], None

        # Non-temporal: everything is flat
        parts = []
        if use_embeddings:
            parts.append(self._load_embeddings(dataset_dir, labels_df, temporal))
        if use_features:
            parts.append(self._load_windowed_features(dataset_dir))

        data = torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]
        return data, None

    def _load_windowed_features(self, dataset_dir):
        """Load windowed feature summaries as a flat (N, F) tensor."""
        features_df = pd.read_parquet(dataset_dir / "features_windowed.parquet").drop(
            columns=_SKIP_COLS, errors="ignore"
        )
        features_df = features_df.fillna(features_df.median())
        feat = torch.tensor(features_df.values, dtype=torch.float32)
        self._feat_mean = feat.mean(dim=0)
        self._feat_std = feat.std(dim=0).clamp(min=1e-8)
        return (feat - self._feat_mean) / self._feat_std

    def _load_temporal_features(self, dataset_dir, labels_df):
        """Load per-frame features as segment-pooled temporal sequences.

        Z-score normalizes across all frames before pooling, matching
        the normalization in the windowed features path.
        """
        feat_path = dataset_dir / "features_binned.pt"
        feat_dict = torch.load(feat_path, weights_only=False)

        keys = list(
            zip(labels_df["video_id"], labels_df["bird_id"], labels_df["window"])
        )

        # Compute global mean/std across all frames for normalization
        all_frames = torch.cat([feat_dict[(v, b, w)].float() for v, b, w in keys])
        feat_mean = all_frames.mean(dim=0)
        feat_std = all_frames.std(dim=0).clamp(min=1e-8)

        return torch.stack(
            [
                adaptive_segment_pool1d(
                    (feat_dict[(v, b, w)].float() - feat_mean) / feat_std,
                    self.n_segments,
                )
                for v, b, w in keys
            ]
        )

    def _load_embeddings(self, dataset_dir, labels_df, temporal):
        keys = list(
            zip(labels_df["video_id"], labels_df["bird_id"], labels_df["window"])
        )

        all_emb_parts = []
        for emb_file in self.embeddings_files:
            emb_path = dataset_dir / emb_file
            embeddings_dict = torch.load(emb_path, weights_only=False)

            if temporal:
                part = torch.stack(
                    [
                        adaptive_segment_pool1d(
                            embeddings_dict[(v, b, w)].float(), self.n_segments
                        )
                        for v, b, w in keys
                    ]
                )
            else:

                def _pool(t):
                    t = t.float()
                    return t if t.ndim == 1 else t.mean(dim=0)

                part = torch.stack(
                    [_pool(embeddings_dict[(v, b, w)]) for v, b, w in keys]
                )

            all_emb_parts.append(part)
            del embeddings_dict

        # Concatenate along feature dim: (N, D1) + (N, D2) -> (N, D1+D2)
        # or (N, K, D1) + (N, K, D2) -> (N, K, D1+D2)
        return torch.cat(all_emb_parts, dim=-1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {"data": self.data[idx], "label": self.labels[idx]}
        if self.flat is not None:
            item["flat"] = self.flat[idx]
        return item


# ----------------------------------------------------------------------
# DataModule
# ----------------------------------------------------------------------


def _compute_class_weights(dataset, indices, n_classes, scheme="inv-sqrt"):
    """Class weights from a subset's labels.

    Parameters
    ----------
    scheme : str
        ``"inv-sqrt"`` (default), ``"inv"``, or ``"none"``.
    """
    if scheme == "none":
        return None
    labels = dataset.labels[indices]
    counts = torch.bincount(labels, minlength=n_classes).float().clamp(min=1)
    if scheme == "inv":
        weights = 1.0 / counts
    else:  # inv-sqrt
        weights = 1.0 / counts.sqrt()
    return weights / weights.sum() * n_classes


class BehaviourDataModule(L.LightningDataModule):
    """Data module for behaviour classification with cross-validation.

    Accepts a ``LOO`` splitter (LOCO or LOVO) that controls fold iteration.
    Call ``set_fold(test_id, val_id)`` then ``setup()`` for each fold.

    After ``setup()``, exposes:

    - ``class_weights`` — inverse-sqrt weights from the train split
    - ``n_classes``, ``data_dim``
    - ``label_encoder`` — maps class names <-> indices
    """

    def __init__(
        self,
        dataset_dir,
        splitter: LOO,
        batch_size=32,
        use_features=True,
        use_embeddings=False,
        temporal=False,
        n_segments=12,
        exclude=None,
        embeddings_file="embeddings.pt",
        embeddings_files=None,
    ):
        super().__init__()
        self.dataset_dir = Path(dataset_dir)
        self.splitter = splitter
        self.test_id = None
        self.val_id = None
        self.batch_size = batch_size
        self.use_features = use_features
        self.use_embeddings = use_embeddings
        self.temporal = temporal
        self.n_segments = n_segments
        self.exclude = exclude
        self.embeddings_files = embeddings_files or [embeddings_file]
        self.class_weight_scheme = "inv-sqrt"

    @property
    def video_ids(self) -> list[str]:
        """Unique video IDs in the dataset. Available after ``prepare_data()``."""
        return sorted(set(self._video_ids))

    def set_fold(self, test_id: str, val_id: str):
        """Set test/val IDs for the next ``setup()`` call."""
        self.test_id = test_id
        self.val_id = val_id

    def prepare_data(self):
        # NOTE: assigns self._dataset here, which Lightning docs warn against for
        # multi-GPU/distributed (state won't broadcast to other ranks). Fine for
        # our single-GPU setup; revisit if we ever go multi-GPU.
        # pylint: disable=attribute-defined-outside-init
        self._dataset = BehaviourDataset(
            self.dataset_dir,
            use_features=self.use_features,
            use_embeddings=self.use_embeddings,
            temporal=self.temporal,
            n_segments=self.n_segments,
            exclude=self.exclude,
            embeddings_files=self.embeddings_files,
        )
        self._video_ids = list(self._dataset.video_ids)
        # pylint: enable=attribute-defined-outside-init

    def setup(self, stage=None):
        # pylint: disable=attribute-defined-outside-init
        if self.test_id is None or self.val_id is None:
            raise ValueError("Call set_fold(test_id, val_id) before setup()")
        data = self._dataset

        # Build per-sample group labels from the splitter
        groups = np.array(self.splitter.get_groups(data.video_ids))
        train_idx = np.where(~np.isin(groups, [self.test_id, self.val_id]))[0].tolist()
        val_idx = np.where(groups == self.val_id)[0].tolist()
        test_idx = np.where(groups == self.test_id)[0].tolist()
        train_ds = Subset(data, train_idx)
        val_ds = Subset(data, val_idx)
        test_ds = Subset(data, test_idx)

        # Subset doesn't forward dataset attributes, so we expose them here
        # for model construction (data_dim, n_classes) and confusion matrix labels
        self.label_encoder = data.label_encoder
        self.n_classes = data.n_classes
        self.data_dim = data.data_dim
        self.flat_dim = data.flat_dim

        if stage == "fit" or stage is None:
            self.train_ds = train_ds
            self.val_ds = val_ds
            self.class_weights = _compute_class_weights(
                data,
                train_ds.indices,
                self.n_classes,
                scheme=self.class_weight_scheme,
            )

            logger.info(
                f"Fit: train={len(self.train_ds)}  val={len(self.val_ds)}  "
                f"classes={list(self.label_encoder.lab2ind.keys())}  weights={self.class_weights}"
            )

        if stage == "test" or stage is None:
            self.test_ds = test_ds
            logger.info(f"Test: {len(self.test_ds)} samples")

        # pylint: enable=attribute-defined-outside-init

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size)
