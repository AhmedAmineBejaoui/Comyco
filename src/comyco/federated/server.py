"""Federated server orchestrating the learning process."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple

import torch
from torch import nn

from .client import FederatedClient
from .model import evaluate


@dataclass
class AggregationResult:
    round_idx: int
    global_accuracy: float
    global_loss: float


@dataclass
class FederatedServer:
    clients: List[FederatedClient]
    device: torch.device
    aggregation_history: List[AggregationResult] = field(default_factory=list)

    def initialise_weights(self) -> Dict[str, torch.Tensor]:
        if not self.clients:
            raise ValueError("At least one client is required")
        model = self.clients[0].init_model()
        return model.state_dict()

    def aggregate(self, updates: Iterable[Tuple[Dict[str, torch.Tensor], int]]) -> Dict[str, torch.Tensor]:
        total_samples = 0
        avg_state: Dict[str, torch.Tensor] = {}
        for state_dict, num_samples in updates:
            total_samples += num_samples
            for key, tensor in state_dict.items():
                tensor = tensor.to(self.device)
                if key not in avg_state:
                    avg_state[key] = tensor * num_samples
                else:
                    avg_state[key] += tensor * num_samples
        for key in avg_state:
            avg_state[key] /= max(total_samples, 1)
        return avg_state

    def evaluate_global_model(self, global_weights: Dict[str, torch.Tensor]) -> Tuple[float, float]:
        accuracies: List[float] = []
        losses: List[float] = []
        criterion = nn.CrossEntropyLoss()
        for client in self.clients:
            model = client.init_model()
            model.load_state_dict(global_weights)
            metrics = evaluate(model, client.dataset.test_loader, device=self.device, criterion=criterion)
            accuracies.append(metrics["accuracy"])
            losses.append(metrics["loss"])
        return float(sum(accuracies) / len(accuracies)), float(sum(losses) / len(losses))

    def federated_round(
        self, round_idx: int, client_states: Iterable[Tuple[Dict[str, torch.Tensor], int]]
    ) -> Dict[str, torch.Tensor]:
        new_global = self.aggregate(client_states)
        accuracy, loss = self.evaluate_global_model(new_global)
        self.aggregation_history.append(
            AggregationResult(round_idx=round_idx, global_accuracy=accuracy, global_loss=loss)
        )
        return new_global
