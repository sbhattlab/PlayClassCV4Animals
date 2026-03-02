import torch.nn as nn
import torch.nn.functional as F


class LinearBaseline(nn.Module):
    def __init__(self, d_model, n_classes):
        super().__init__()
        self.fc = nn.Linear(d_model, n_classes)

    def forward(self, x):
        # x: (B, F, D) -> mean pool over time -> (B, D) -> (B, n_classes)
        return self.fc(x.mean(dim=1))


class TemporalCNN(nn.Module):
    def __init__(self, d_model, n_classes, d_hidden=128, dropout=0.3):
        super().__init__()
        self.proj = nn.Linear(d_model, d_hidden)
        self.conv1 = nn.Conv1d(d_hidden, d_hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(d_hidden)
        self.conv2 = nn.Conv1d(d_hidden, d_hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(d_hidden)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d_hidden, n_classes)

    def forward(self, x):
        # x: (B, F, D)
        x = self.proj(x)  # (B, F, d_hidden)
        x = x.transpose(1, 2)  # (B, d_hidden, F)
        x = self.drop(F.relu(self.bn1(self.conv1(x))))  # Conv block 1
        x = self.drop(F.relu(self.bn2(self.conv2(x))))  # Conv block 2
        x = x.mean(dim=2)  # (B, d_hidden)
        return self.head(x)  # (B, n_classes)
