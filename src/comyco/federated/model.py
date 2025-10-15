"""Lightweight neural network used for the Comyco simulations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import torch
from torch import nn


class StreamingQualityModel(nn.Module):
    """Simple feed-forward network predicting QoE quality."""

    def __init__(self, input_dim: int = 6, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(features)


@dataclass
class TrainConfig:
    epochs: int = 1
    lr: float = 1e-3
    weight_decay: float = 0.0


def train_one_epoch(
    model: nn.Module,
    loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    for features, targets in loader:
        features = features.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * features.size(0)
        total_samples += features.size(0)
    return total_loss / max(total_samples, 1)


def evaluate(
    model: nn.Module,
    loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    *,
    device: torch.device,
    criterion: nn.Module,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)
            logits = model(features)
            loss = criterion(logits, targets)
            predictions = logits.argmax(dim=1)
            total_loss += loss.item() * features.size(0)
            total_correct += (predictions == targets).sum().item()
            total_samples += features.size(0)
    return {
        "loss": total_loss / max(total_samples, 1),
        "accuracy": total_correct / max(total_samples, 1),
    }


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
