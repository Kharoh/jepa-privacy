"""Statistical metrics helpers for the MNIST experiment."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np


def bootstrap_ci(values: Iterable[float], num_samples: int = 2000, alpha: float = 0.05) -> Tuple[float, float]:
    vals = np.array(list(values), dtype=float)
    if len(vals) == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(42)
    means = []
    for _ in range(num_samples):
        sample = rng.choice(vals, size=len(vals), replace=True)
        means.append(sample.mean())
    lower = np.percentile(means, 100 * (alpha / 2))
    upper = np.percentile(means, 100 * (1 - alpha / 2))
    return float(lower), float(upper)


def permutation_test(a: Iterable[float], b: Iterable[float], num_samples: int = 5000) -> float:
    """Two-sided permutation test for difference in means."""
    a_vals = np.array(list(a), dtype=float)
    b_vals = np.array(list(b), dtype=float)
    if len(a_vals) == 0 or len(b_vals) == 0:
        return 1.0
    observed = np.abs(a_vals.mean() - b_vals.mean())
    pooled = np.concatenate([a_vals, b_vals])
    rng = np.random.default_rng(42)
    count = 0
    for _ in range(num_samples):
        rng.shuffle(pooled)
        new_a = pooled[: len(a_vals)]
        new_b = pooled[len(a_vals) :]
        diff = np.abs(new_a.mean() - new_b.mean())
        if diff >= observed:
            count += 1
    return float((count + 1) / (num_samples + 1))


def summarize_metric(values: Iterable[float]) -> Dict[str, float]:
    vals = np.array(list(values), dtype=float)
    if len(vals) == 0:
        return {"mean": 0.0, "std": 0.0}
    return {"mean": float(vals.mean()), "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0}
