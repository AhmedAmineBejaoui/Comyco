"""Federated learning client logic."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import torch
from torch import nn

from .dataset import ClientDataset
from .model import TrainConfig, evaluate, train_one_epoch


@dataclass
class FederatedClient:
    """A client capable of performing local training."""

    client_id: str
    dataset: ClientDataset
    config: TrainConfig
    device: torch.device
    model_factory: callable
    history: Dict[str, list] = field(default_factory=lambda: {"train_loss": [], "val_accuracy": []})

    def init_model(self) -> nn.Module:
        model = self.model_factory()
        return model.to(self.device)

    def train_local(self, initial_weights: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], int]:
        model = self.init_model()
        model.load_state_dict(initial_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        criterion = nn.CrossEntropyLoss()

        for _ in range(self.config.epochs):
            loss = train_one_epoch(
                model,
                self.dataset.train_loader,
                device=self.device,
                optimizer=optimizer,
                criterion=criterion,
            )
            self.history["train_loss"].append(loss)

        metrics = evaluate(model, self.dataset.test_loader, device=self.device, criterion=criterion)
        self.history["val_accuracy"].append(metrics["accuracy"])
        return model.state_dict(), len(self.dataset.train_loader.dataset)
