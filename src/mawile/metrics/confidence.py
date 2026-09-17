from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

from mawile.schemas import ItemRisk


def bootstrap_ci(
    values: list[float],
    n_samples: int,
    rng: random.Random,
    alpha: float = 0.05,
) -> dict[str, float] | None:
    """Percentile bootstrap CI for the mean of ``values``.

    Returns ``None`` for empty input. With very few items the interval is wide
    by design -- that width is the honest signal that the estimate is noisy.
    """

    if not values:
        return None
    n = len(values)
    means = []
    for _ in range(n_samples):
        resample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(resample) / n)
    means.sort()
    lo_index = int((alpha / 2) * n_samples)
    hi_index = min(n_samples - 1, int((1 - alpha / 2) * n_samples))
    return {
        "mean": sum(values) / n,
        "lo": means[lo_index],
        "hi": means[hi_index],
        "n": n,
    }


def compute_confidence_intervals(
    item_risks: list[ItemRisk],
    n_samples: int,
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    flip_values = [risk.flip_risk for risk in item_risks if risk.flip_risk is not None]
    noise_values = [risk.noise_risk for risk in item_risks if risk.noise_risk is not None]

    family_values: dict[str, list[float]] = defaultdict(list)
    for risk in item_risks:
        for family, value in risk.family_risks.items():
            family_values[family].append(value)

    return {
        "mean_flip_risk": bootstrap_ci(flip_values, n_samples, rng),
        "mean_noise_risk": bootstrap_ci(noise_values, n_samples, rng),
        "family_flip_rate": {
            family: bootstrap_ci(values, n_samples, rng)
            for family, values in sorted(family_values.items())
        },
    }
