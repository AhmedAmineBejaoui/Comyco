"""Helper utilities to compare centralised and federated experiments."""
from __future__ import annotations

from dataclasses import asdict
from typing import Dict

from ..federated.dataset import ClientDataset
from ..federated.server import FederatedServer


def build_run_summary(server: FederatedServer, datasets: Dict[str, ClientDataset]) -> Dict[str, object]:
    scenario_overview = {
        client_id: {
            "scenario": dataset.scenario.name,
            "base_throughput": dataset.scenario.base_kbps,
            "variability": dataset.scenario.variability,
            "burst_chance": dataset.scenario.burst_chance,
            "burst_scale": dataset.scenario.burst_scale,
        }
        for client_id, dataset in datasets.items()
    }

    history = [asdict(result) for result in server.aggregation_history]

    per_client_stats = {}
    for client_id, dataset in datasets.items():
        df = dataset.raw_dataframe.copy()
        corr = df.drop(columns=["timestamp", "experience_ok"]).corr()
        # Remove the noisy ``wait_ms`` metric mentioned in the advisor feedback.
        if "wait_ms" in corr.columns:
            corr = corr.drop(index="wait_ms", columns="wait_ms")
        per_client_stats[client_id] = {
            "scenario": dataset.scenario.name,
            "samples": len(df),
            "experience_rate": float(df["experience_ok"].mean()),
            "interruptions_mean": float(df["interruptions"].mean()),
            "throughput_mean": float(df["throughput_kbps"].mean()),
            "correlation_matrix": corr.round(3).to_dict(),
        }

    return {
        "aggregations": history,
        "clients": per_client_stats,
        "scenarios": scenario_overview,
    }
