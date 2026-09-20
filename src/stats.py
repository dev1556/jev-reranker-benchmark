"""Pure statistics. Imports numpy only.

Every headline number in this project carries a CI from here, and every arm
comparison a p-value. Quarantined and human-reviewed for the same reason as
metrics.py — a bug here is invisible downstream.
"""

from collections.abc import Mapping, Sequence

import numpy as np


def bootstrap_ci(
    values: Sequence[float], n: int = 10_000, alpha: float = 0.05, seed: int = 42
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean, resampling over queries.

    Efron (1979). Resampling is over the per-query metric values, so the
    interval reflects query-to-query variation — the thing that actually
    limits what this benchmark can claim.
    """
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n, values.size))
    means = values[idx].mean(axis=1)
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def paired_randomisation_test(
    a: Sequence[float], b: Sequence[float], n: int = 10_000, seed: int = 42
) -> float:
    """Two-sided paired permutation test on the mean difference.

    Under the null, the sign of each per-query difference is exchangeable, so we
    flip signs at random and count how often |mean| is at least the observed
    value. Paired because both arms rerank the identical candidate pool for the
    identical queries — an unpaired test would throw that away and lose power.

    Returns a p-value with the standard (hits + 1) / (n + 1) correction, which
    never returns exactly 0.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"paired test needs equal shapes, got {a.shape} and {b.shape}")
    diffs = a - b
    observed = abs(float(diffs.mean()))
    if observed == 0.0:
        return 1.0
    rng = np.random.default_rng(seed)
    signs = rng.choice((-1.0, 1.0), size=(n, diffs.size))
    null = np.abs((signs * diffs).mean(axis=1))
    return float((np.sum(null >= observed) + 1) / (n + 1))


def holm(pvalues: Mapping[str, float]) -> dict[str, float]:
    """Holm-Bonferroni step-down correction (Holm, 1979).

    Applied across the family of arm-pair comparisons. Uniformly more powerful
    than plain Bonferroni at the same family-wise error rate, and with ten arm
    pairs the difference is not cosmetic.
    """
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted: dict[str, float] = {}
    running = 0.0
    for i, (key, p) in enumerate(items):
        running = max(running, min(1.0, p * (m - i)))  # enforce monotonicity
        adjusted[key] = running
    return adjusted
