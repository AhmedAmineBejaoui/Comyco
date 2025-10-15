"""Command line entry point to simulate Comyco with Federated Learning."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch

from ..data.scenarios import default_scenarios, expand_scenarios
from ..federated.client import FederatedClient
from ..federated.dataset import build_datasets_for_clients
from ..federated.model import StreamingQualityModel, TrainConfig, get_device
from ..federated.server import AggregationResult, FederatedServer
from ..reporting.summaries import build_run_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clients", type=int, default=3, help="Number of simulated clients")
    parser.add_argument("--rounds", type=int, default=5, help="Number of federated rounds")
    parser.add_argument("--local-epochs", type=int, default=1, help="Local epochs per client")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--length", type=int, default=600, help="Number of samples per client")
    parser.add_argument(
        "--update-frequency",
        type=int,
        default=1,
        help="Number of local rounds before synchronisation (>=1)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("federated_run.json"),
        help="Where to store the summary metrics",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()

    scenarios = expand_scenarios(default_scenarios(), total_clients=args.clients, base_seed=2024)
    datasets = build_datasets_for_clients(
        scenarios,
        length=args.length,
        batch_size=args.batch_size,
    )

    train_config = TrainConfig(epochs=args.local_epochs)
    clients: List[FederatedClient] = []
    for client_id, dataset in datasets.items():
        client = FederatedClient(
            client_id=client_id,
            dataset=dataset,
            config=train_config,
            device=device,
            model_factory=StreamingQualityModel,
        )
        clients.append(client)

    server = FederatedServer(clients=clients, device=device)
    global_weights = server.initialise_weights()

    def clone_state(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {key: tensor.clone().detach() for key, tensor in state.items()}

    local_states: Dict[str, Dict[str, torch.Tensor]] = {
        client.client_id: clone_state(global_weights) for client in clients
    }
    sample_counts: Dict[str, int] = {
        client.client_id: len(client.dataset.train_loader.dataset) for client in clients
    }

    for round_idx in range(1, args.rounds + 1):
        for client in clients:
            state, num_samples = client.train_local(local_states[client.client_id])
            local_states[client.client_id] = clone_state(state)
            sample_counts[client.client_id] = num_samples

        if round_idx % args.update_frequency == 0 or round_idx == args.rounds:
            aggregated = server.aggregate(
                (local_states[client.client_id], sample_counts[client.client_id])
                for client in clients
            )
            global_weights = clone_state(aggregated)
            for client in clients:
                local_states[client.client_id] = clone_state(global_weights)
            accuracy, loss = server.evaluate_global_model(global_weights)
            server.aggregation_history.append(
                AggregationResult(round_idx=round_idx, global_accuracy=accuracy, global_loss=loss)
            )

    summary = build_run_summary(server, datasets)
    args.output.write_text(json.dumps(summary, indent=2))
    print(f"Saved summary to {args.output.resolve()}")


if __name__ == "__main__":
    main()
