"""Baseline experiment for centralised Comyco training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader

from ..data.scenarios import default_scenarios, expand_scenarios
from ..federated.dataset import build_datasets_for_clients, dataframe_to_tensors
from ..federated.model import StreamingQualityModel, evaluate, get_device, train_one_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clients", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--length", type=int, default=600)
    parser.add_argument("--output", type=Path, default=Path("centralised_run.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()
    scenarios = expand_scenarios(default_scenarios(), total_clients=args.clients, base_seed=2024)
    datasets = build_datasets_for_clients(scenarios, length=args.length, batch_size=args.batch_size)

    tensors = []
    for dataset in datasets.values():
        features, targets = dataframe_to_tensors(dataset.raw_dataframe)
        tensors.append(torch.utils.data.TensorDataset(features, targets))

    combined_dataset = ConcatDataset(tensors)
    loader = DataLoader(combined_dataset, batch_size=args.batch_size, shuffle=True)

    model = StreamingQualityModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    history = []
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, loader, device=device, optimizer=optimizer, criterion=criterion)
        metrics = evaluate(model, loader, device=device, criterion=criterion)
        history.append({"epoch": epoch, "loss": loss, "accuracy": metrics["accuracy"]})

    args.output.write_text(json.dumps({"history": history}, indent=2))
    print(f"Saved centralised history to {args.output.resolve()}")


if __name__ == "__main__":
    main()
