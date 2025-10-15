"""Utilities to build synthetic video streaming datasets for FDL experiments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch

from ..data.scenarios import ThroughputScenario, rolling_window_average


@dataclass
class ClientDataset:
    """Container for the train/test loaders of a client."""

    scenario: ThroughputScenario
    train_loader: DataLoader
    test_loader: DataLoader
    raw_dataframe: pd.DataFrame


def _simulate_interruptions(throughput: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Simulate buffering interruptions based on throughput."""

    baseline = np.maximum(0, 2_000 - throughput)
    spikes = rng.poisson(lam=baseline / 800.0)
    jitter = rng.binomial(n=1, p=np.clip(throughput / 10_000.0, 0.05, 0.95))
    return spikes + (1 - jitter)


def _simulate_buffer_ratio(throughput: np.ndarray, interruptions: np.ndarray) -> np.ndarray:
    ratio = np.clip((interruptions * 3) / (throughput / 500.0 + 1), 0, 1)
    return ratio


def _simulate_quality(throughput: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    quality = 144 + throughput / 20
    quality += rng.normal(0, 30, size=throughput.shape[0])
    return np.clip(quality, 144, 2160)


def create_client_dataframe(
    scenario: ThroughputScenario,
    length: int,
    seed: int,
    smoothing_window: int = 5,
) -> pd.DataFrame:
    """Generate a dataframe describing the streaming session of one client."""

    throughput = scenario.generate(length=length, seed=seed)
    rng = np.random.default_rng(seed + scenario.seed_offset)
    interruptions = _simulate_interruptions(throughput, rng)
    buffer_ratio = _simulate_buffer_ratio(throughput, interruptions)
    video_quality = _simulate_quality(throughput, rng)
    latency_ms = np.clip(4000 / np.sqrt(throughput), 20, 250)
    stability_index = rolling_window_average(throughput, window=smoothing_window)

    df = pd.DataFrame(
        {
            "timestamp": np.arange(length),
            "throughput_kbps": throughput,
            "interruptions": interruptions,
            "buffer_ratio": buffer_ratio,
            "video_quality": video_quality,
            "latency_ms": latency_ms,
            "stability_index": stability_index[:length],
        }
    )

    # Binary target: 1 if the experience is smooth, 0 otherwise.
    df["experience_ok"] = (
        (df["throughput_kbps"] > 2_500)
        & (df["buffer_ratio"] < 0.35)
        & (df["interruptions"] <= 1)
        & (df["latency_ms"] < 120)
    ).astype(int)

    return df


def dataframe_to_tensors(df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
    features = df[
        [
            "throughput_kbps",
            "interruptions",
            "buffer_ratio",
            "video_quality",
            "latency_ms",
            "stability_index",
        ]
    ].to_numpy(dtype=np.float32)
    targets = df["experience_ok"].to_numpy(dtype=np.int64)
    return torch.from_numpy(features), torch.from_numpy(targets)


def build_client_dataset(
    scenario: ThroughputScenario,
    *,
    length: int = 600,
    batch_size: int = 32,
    seed: int = 123,
    test_size: float = 0.2,
) -> ClientDataset:
    """Create the dataloaders for a client."""

    df = create_client_dataframe(scenario, length=length, seed=seed)
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        shuffle=True,
        stratify=df["experience_ok"],
        random_state=seed,
    )

    train_features, train_targets = dataframe_to_tensors(train_df)
    test_features, test_targets = dataframe_to_tensors(test_df)

    train_loader = DataLoader(
        TensorDataset(train_features, train_targets), batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(test_features, test_targets), batch_size=batch_size, shuffle=False
    )

    return ClientDataset(
        scenario=scenario,
        train_loader=train_loader,
        test_loader=test_loader,
        raw_dataframe=df,
    )


def build_datasets_for_clients(
    scenarios: Iterable[ThroughputScenario],
    *,
    length: int = 600,
    batch_size: int = 32,
    base_seed: int = 1234,
) -> Dict[str, ClientDataset]:
    datasets: Dict[str, ClientDataset] = {}
    for idx, scenario in enumerate(scenarios):
        client_id = f"client_{idx+1}_{scenario.name}"
        datasets[client_id] = build_client_dataset(
            scenario,
            length=length,
            batch_size=batch_size,
            seed=base_seed + idx * 41,
        )
    return datasets
