"""Network throughput scenario simulation utilities for Comyco.

This module exposes helpers to generate synthetic throughput traces for
multiple clients.  Each scenario is parameterised so that every client can
have distinct behaviour, as requested in the project specifications and
review comments.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np


@dataclass(frozen=True)
class ThroughputScenario:
    """Description of a throughput scenario.

    Attributes
    ----------
    name:
        Human readable name for the scenario (``"maison"``, ``"campagne"`` …).
    base_kbps:
        Typical throughput value in kilobits per second.
    variability:
        Standard deviation applied to the gaussian noise used to perturb the
        throughput trace.
    burst_chance:
        Probability that a given timestep experiences a burst event with sharp
        increase or drop in throughput.
    burst_scale:
        Magnitude of the burst when it happens.  Positive values represent
        peaks, negative values represent drops.
    seed_offset:
        Offset added to the random seed to guarantee that two clients using
        the same scenario still obtain distinct traces.  The offset is added to
        the base seed provided to :func:`ThroughputScenario.generate`.
    """

    name: str
    base_kbps: float
    variability: float
    burst_chance: float
    burst_scale: float
    seed_offset: int = 0

    def generate(self, length: int, seed: int | None = None) -> np.ndarray:
        """Generate a throughput trace.

        Parameters
        ----------
        length:
            Number of timesteps to simulate.
        seed:
            Optional base seed used to make experiments reproducible.  The
            scenario's :attr:`seed_offset` is added to this value so that two
            clients configured with the same scenario never obtain identical
            traces, satisfying the requirement highlighted in the advisor's
            feedback.
        """

        rng = np.random.default_rng(None if seed is None else seed + self.seed_offset)
        noise = rng.normal(loc=0.0, scale=self.variability, size=length)
        throughput = np.full(length, self.base_kbps, dtype=float) + noise

        # Introduce bursty behaviour to mimic real-world networks.  Bursts can
        # be either positive (better bandwidth) or negative (temporary drop).
        burst_mask = rng.random(length) < self.burst_chance
        burst_directions = rng.choice([-1.0, 1.0], size=length)
        throughput += burst_mask * burst_directions * self.burst_scale * rng.random(length)

        # Ensure throughput never becomes negative.
        np.maximum(throughput, 1.0, out=throughput)
        return throughput


def default_scenarios(seed: int = 42) -> Sequence[ThroughputScenario]:
    """Return the three canonical scenarios described in the roadmap."""

    return (
        ThroughputScenario(
            name="maison",
            base_kbps=3_500,
            variability=150,
            burst_chance=0.05,
            burst_scale=600,
            seed_offset=seed,
        ),
        ThroughputScenario(
            name="campagne",
            base_kbps=1_200,
            variability=400,
            burst_chance=0.20,
            burst_scale=900,
            seed_offset=seed + 101,
        ),
        ThroughputScenario(
            name="voiture",
            base_kbps=2_400,
            variability=700,
            burst_chance=0.35,
            burst_scale=1_300,
            seed_offset=seed + 251,
        ),
    )


def expand_scenarios(
    scenarios: Sequence[ThroughputScenario],
    total_clients: int,
    base_seed: int,
) -> List[ThroughputScenario]:
    """Create a scenario list large enough for ``total_clients`` clients.

    The function cycles through the provided ``scenarios`` while ensuring that
    each returned scenario receives an increasing :attr:`seed_offset`.  This
    guarantees all clients experience subtly different throughput behaviour
    even when they share the same physical environment (``maison``/``campagne``/``voiture``).
    """

    expanded: List[ThroughputScenario] = []
    for idx in range(total_clients):
        base = scenarios[idx % len(scenarios)]
        expanded.append(
            ThroughputScenario(
                name=base.name,
                base_kbps=base.base_kbps,
                variability=base.variability,
                burst_chance=base.burst_chance,
                burst_scale=base.burst_scale,
                seed_offset=base_seed + idx * 997,
            )
        )
    return expanded


def rolling_window_average(values: Iterable[float], window: int) -> np.ndarray:
    """Compute a rolling average with reflection padding for edges."""

    arr = np.asarray(list(values), dtype=float)
    if window <= 1:
        return arr
    padded = np.pad(arr, pad_width=window // 2, mode="reflect")
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode="valid")
