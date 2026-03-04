"""BehaviourDataset and BehaviourDataModule for classification training."""

from pathlib import Path

import lightning as L
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from loguru import logger
from torch.utils.data import DataLoader, Dataset, Subset

LABEL_ORDER = ["none", "worm", "locomotor", "social"]

_KEY_COLS = {"video_id", "bird_id", "window"}
_SKIP_COLS = _KEY_COLS | {"n_frames"}


class LabelEncoder:
    """Encode categorical labels as integers with a fixed order.

    Parameters
    ----------
    classes : list[str]
        Ordered class names.  Index 0 → first element, etc.
    """

    def __init__(self, classes):
        self.lab2ind = {l: i for i, l in enumerate(classes)}
        self.ind2lab = {i: l for i, l in enumerate(classes)}

    def encode(self, labels):
        """Encode label string(s) → integer index(es)."""
        if isinstance(labels, str):
            return self.lab2ind[labels]
        return [self.lab2ind[l] for l in labels]

    def decode(self, indices):
        """Decode integer index(es) → label string(s)."""
        if isinstance(indices, int):
            return self.ind2lab[indices]
        return [self.ind2lab[i] for i in indices]


def adaptive_segment_pool1d(x, n_segments):
    """Reduce (T, D) → (K, D) via adaptive average pooling."""
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
    tracking_dir : str | Path
        Directory containing the dataset files.
    use_features : bool
        Load handcrafted features from ``dataset_features.parquet``.
    use_embeddings : bool
        Load embeddings from ``dataset_embeddings.pt``.
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
        tracking_dir,
        use_features=True,
        use_embeddings=False,
        temporal=False,
        n_segments=12,
        exclude=None,
    ):
        tracking_dir = Path(tracking_dir)

        self.n_segments = n_segments

        # 1. Load data
        labels_df = pd.read_parquet(tracking_dir / "dataset_labels.parquet")
        self.data = self.load_data(
            tracking_dir, labels_df, use_features, use_embeddings, temporal
        )

        # 2. Exclude data from unwanted labels (if specified)
        label_order = LABEL_ORDER.copy()
        if exclude is not None:
            exclude_list = [exclude] if isinstance(exclude, str) else exclude
            label_order = [l for l in LABEL_ORDER if l not in exclude_list]
            mask = ~labels_df["behav_label"].isin(exclude_list)
            labels_df = labels_df[mask].reset_index(drop=True)
            self.data = self.data[mask.values]

        # 3. Encode labels
        self.label_encoder = LabelEncoder(label_order)
        self.n_classes = len(label_order)

        # 4. Build label tensor
        self.labels = torch.tensor(
            self.label_encoder.encode(labels_df["behav_label"].tolist()),
            dtype=torch.long,
        )
        self.groups = labels_df["video_id"].tolist()  # for LOVO splitting
        self.data_dim = self.data.shape[-1]

    def load_data(
        self, tracking_dir, labels_df, use_features, use_embeddings, temporal
    ):
        if temporal:
            assert use_embeddings, "Temporal mode requires embeddings"
            assert not use_features, "Temporal mode does not use features"
            return self._load_embeddings(tracking_dir, labels_df, temporal)

        parts = []
        if use_embeddings:
            parts.append(self._load_embeddings(tracking_dir, labels_df, temporal))
        if use_features:
            features_df = pd.read_parquet(
                tracking_dir / "dataset_features.parquet"
            ).drop(columns=_SKIP_COLS, errors="ignore")
            parts.append(torch.tensor(features_df.values, dtype=torch.float32))

        assert parts, "At least one of use_features or use_embeddings must be True"
        return torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]

    def _load_embeddings(self, tracking_dir, labels_df, temporal):
        emb_path = tracking_dir / "dataset_embeddings.pt"
        embeddings_dict = torch.load(emb_path, weights_only=False)

        keys = list(
            zip(labels_df["video_id"], labels_df["bird_id"], labels_df["window"])
        )
        if temporal:
            # Adaptive segment pool: (F_w, D) → (K, D), then stack → (N, K, D)
            return torch.stack(
                [
                    adaptive_segment_pool1d(
                        embeddings_dict[(v, b, w)].float(), self.n_segments
                    )
                    for v, b, w in keys
                ]
            )
        # Not temporal → mean-pool over windows: (F_w, D) → (D,)
        return torch.stack(
            [embeddings_dict[(v, b, w)].float().mean(dim=0) for v, b, w in keys]
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {"data": self.data[idx], "label": self.labels[idx]}


# ----------------------------------------------------------------------
# Splitting
# ----------------------------------------------------------------------


def leave_one_group_out(dataset, test_group, val_group):
    """Split a dataset by group membership (LOVO).

    Parameters
    ----------
    dataset : BehaviourDataset
        Must have a ``groups`` attribute (array of group labels).
    test_group, val_group : str
        Group values to hold out.

    Returns
    -------
    train, val, test : Subset
    """
    groups = np.array(dataset.groups)
    train_idx = np.where(~np.isin(groups, [test_group, val_group]))[0].tolist()
    val_idx = np.where(groups == val_group)[0].tolist()
    test_idx = np.where(groups == test_group)[0].tolist()
    return (
        Subset(dataset, train_idx),
        Subset(dataset, val_idx),
        Subset(dataset, test_idx),
    )


# ----------------------------------------------------------------------
# DataModule
# ----------------------------------------------------------------------


def _compute_class_weights(dataset, indices, n_classes):
    """Inverse-sqrt class weights from a subset's labels."""
    labels = dataset.labels[indices]
    counts = torch.bincount(labels, minlength=n_classes).float().clamp(min=1)
    weights = 1.0 / counts.sqrt()
    return weights / weights.sum() * n_classes


class BehaviourDataModule(L.LightningDataModule):
    """LOVO (Leave-One-Video-Out) data module.

    After ``setup()``, exposes:

    - ``class_weights`` — inverse-sqrt weights from the train split
    - ``n_classes``, ``data_dim``
    - ``label_encoder`` — maps class names ↔ indices
    """

    def __init__(
        self,
        tracking_dir,
        test_video=None,
        val_video=None,
        batch_size=32,
        use_features=True,
        use_embeddings=False,
        temporal=False,
        n_segments=12,
        exclude=None,
    ):
        super().__init__()
        self.tracking_dir = Path(tracking_dir)
        self.test_video = test_video
        self.val_video = val_video
        self.batch_size = batch_size
        self.use_features = use_features
        self.use_embeddings = use_embeddings
        self.temporal = temporal
        self.n_segments = n_segments
        self.exclude = exclude

    @property
    def video_ids(self) -> list[str]:
        """Unique video IDs in the dataset. Available after ``prepare_data()``."""
        return sorted(set(self._dataset.groups))

    def set_split_groups(self, test_video: str, val_video: str):
        """Set test/val video IDs for the next ``setup()`` call."""
        self.test_video = test_video
        self.val_video = val_video

    def prepare_data(self):
        # NOTE: assigns self._dataset here, which Lightning docs warn against for
        # multi-GPU/distributed (state won't broadcast to other ranks). Fine for
        # our single-GPU setup; revisit if we ever go multi-GPU.
        # pylint: disable=attribute-defined-outside-init
        self._dataset = BehaviourDataset(
            self.tracking_dir,
            use_features=self.use_features,
            use_embeddings=self.use_embeddings,
            temporal=self.temporal,
            n_segments=self.n_segments,
            exclude=self.exclude,
        )
        # pylint: enable=attribute-defined-outside-init

    def setup(self, stage=None):
        # pylint: disable=attribute-defined-outside-init
        data = self._dataset

        # LOVO split
        train_ds, val_ds, test_ds = leave_one_group_out(
            data, self.test_video, self.val_video
        )

        # Subset (LOVO) doesn't forward dataset attributes, so we expose them here
        # for model construction (data_dim, n_classes) and confusion matrix labels
        self.label_encoder = data.label_encoder
        self.n_classes = data.n_classes
        self.data_dim = data.data_dim

        if stage == "fit" or stage is None:
            self.train_ds = train_ds
            self.val_ds = val_ds
            self.class_weights = _compute_class_weights(
                data, train_ds.indices, self.n_classes
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
