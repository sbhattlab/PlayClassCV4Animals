import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedAttentionPool(nn.Module):
    """Learnable gated pooling over sequence dimension: (B, T, D) → (B, D)."""

    def __init__(self, d_input):
        super().__init__()
        self.gate_fc = nn.Linear(d_input, 1)

    def forward(self, x):
        # x: (B, T, D)
        gates = torch.sigmoid(self.gate_fc(x))  # (B, T, 1)
        return (gates * x).sum(dim=1) / gates.sum(dim=1).clamp(min=1e-8)  # (B, D)


class SimpleLinear(nn.Module):
    def __init__(self, d_input, n_classes):
        super().__init__()
        self.fc = nn.Linear(d_input, n_classes)

    def forward(self, x):
        # x: (B, D) -> (B, n_classes)
        return self.fc(x)


class SimpleMLP(nn.Module):
    def __init__(self, d_input, n_classes, d_hidden=64, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_input, d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, n_classes),
        )

    def forward(self, x):
        # x: (B, d_input)
        return self.net(x)


class TemporalMLP(nn.Module):
    """Gated pooling + linear head on raw temporal features (no conv/GRU)."""

    def __init__(self, d_input, n_classes, d_hidden=128, dropout=0.3):
        super().__init__()
        self.proj = nn.Linear(d_input, d_hidden)
        self.pool = GatedAttentionPool(d_hidden)
        self.head = nn.Linear(d_hidden, n_classes)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, D)
        x = self.proj(x)  # (B, T, d_hidden)
        x = self.pool(x)  # (B, d_hidden)
        return self.head(self.drop(x))  # (B, n_classes)


class TemporalCNN(nn.Module):
    def __init__(
        self,
        d_input,
        n_classes,
        d_hidden=128,
        dropout=0.3,
        pool="gated",
    ):
        super().__init__()
        self.proj = nn.Linear(d_input, d_hidden)
        self.conv1 = nn.Conv1d(d_hidden, d_hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(d_hidden)
        self.conv2 = nn.Conv1d(d_hidden, d_hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(d_hidden)
        self.drop = nn.Dropout(dropout)
        self.post_pool = GatedAttentionPool(d_hidden) if "gated" in pool else None
        self.head = nn.Linear(d_hidden, n_classes)

    def forward(self, x):
        # x: (B, K, D)
        x = self.proj(x)  # (B, K, d_hidden)
        x = x.transpose(1, 2)  # (B, d_hidden, K)
        x = self.drop(F.relu(self.bn1(self.conv1(x))))  # Conv block 1
        x = self.drop(F.relu(self.bn2(self.conv2(x))))  # Conv block 2
        if self.post_pool is not None:
            x = x.transpose(1, 2)  # (B, K, d_hidden)
            x = self.post_pool(x)  # (B, d_hidden)
        else:
            x = x.mean(dim=2)  # (B, d_hidden)
        return self.head(x)  # (B, n_classes)


class TemporalGRU(nn.Module):
    def __init__(
        self,
        d_input,
        n_classes,
        d_hidden=128,
        dropout=0.3,
        pool="gated",
    ):
        super().__init__()
        self.proj = nn.Linear(d_input, d_hidden)
        self.gru = nn.GRU(d_hidden, d_hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.post_pool = GatedAttentionPool(d_hidden) if "gated" in pool else None
        self.head = nn.Linear(d_hidden, n_classes)

    def forward(self, x):
        # x: (B, K, D)
        x = self.proj(x)  # (B, K, d_hidden)
        out, h = self.gru(x)  # out: (B, K, d_hidden), h: (1, B, d_hidden)
        if self.post_pool is not None:
            x = self.post_pool(out)  # (B, d_hidden)
        else:
            x = h.squeeze(0)  # (B, d_hidden)
        return self.head(self.drop(x))  # (B, n_classes)


MODEL_REGISTRY = {
    "linear": (
        SimpleLinear,
        {"use_features": False, "use_embeddings": True, "temporal": False},
    ),
    "mlp": (
        SimpleMLP,
        {"use_features": True, "use_embeddings": False, "temporal": False},
    ),
    "mlp_embed": (
        SimpleMLP,
        {"use_features": True, "use_embeddings": True, "temporal": False},
    ),
    "temporal_mlp": (
        TemporalMLP,
        {"use_features": False, "use_embeddings": True, "temporal": True},
    ),
    "temporal_cnn": (
        TemporalCNN,
        {"use_features": False, "use_embeddings": True, "temporal": True},
    ),
    "temporal_gru": (
        TemporalGRU,
        {"use_features": False, "use_embeddings": True, "temporal": True},
    ),
}
