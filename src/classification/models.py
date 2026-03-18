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
    def __init__(self, d_input, n_classes, d_hidden=64, dropout=0.0):
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

    def __init__(self, d_input, n_classes, d_hidden=128, dropout=0.0, d_flat=0):
        super().__init__()
        self.proj = nn.Linear(d_input, d_hidden)
        self.pool = GatedAttentionPool(d_hidden)
        self.head = nn.Linear(d_hidden + d_flat, n_classes)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, flat=None):
        # x: (B, T, D)
        x = self.proj(x)  # (B, T, d_hidden)
        x = self.pool(x)  # (B, d_hidden)
        if flat is not None:
            x = torch.cat([x, flat], dim=-1)
        return self.head(self.drop(x))  # (B, n_classes)


class TemporalCNNv2(nn.Module):
    """Nonlinear projection + single 1D conv + gated pooling."""

    def __init__(
        self, d_input, n_classes, d_hidden=128, dropout=0.0, d_flat=0, d_bottleneck=256
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_input, d_bottleneck),
            nn.GELU(),
        )
        self.conv = nn.Conv1d(d_bottleneck, d_hidden, kernel_size=3, padding=1)
        self.pool = GatedAttentionPool(d_hidden)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d_hidden + d_flat, n_classes)

    def forward(self, x, flat=None):
        x = self.proj(x)  # (B, K, 256)
        x = x.transpose(1, 2)  # (B, 256, K)
        x = F.gelu(self.conv(x))  # (B, d_hidden, K)
        x = x.transpose(1, 2)  # (B, K, d_hidden)
        x = self.pool(x)  # (B, d_hidden)
        if flat is not None:
            x = torch.cat([x, flat], dim=-1)
        return self.head(self.drop(x))


MODEL_REGISTRY = {
    "linear": (SimpleLinear, False),
    "mlp": (SimpleMLP, False),
    "temporal_mlp": (TemporalMLP, True),
    "temporal_cnn2": (TemporalCNNv2, True),
}
