"""Pure metric functions. Imports numpy and src.types only — never an arm, never a client.

Quarantined on purpose: these are the numbers a public benchmark lives or dies by,
and they must be auditable without reading a line of API code.
"""

from collections.abc import Mapping, Sequence

import numpy as np


def _gains(ranked_ids: Sequence[str], relevant: Mapping[str, int], k: int) -> np.ndarray:
    """Exponential gain 2**grade - 1 for the top-k, zero for unjudged (BEIR convention)."""
    return np.array([2 ** relevant.get(d, 0) - 1 for d in ranked_ids[:k]], dtype=float)


def _dcg(gains: np.ndarray) -> float:
    """DCG = sum_i gain_i / log2(i + 2), i zero-indexed."""
    if gains.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, gains.size + 2))
    return float(np.sum(gains / discounts))


def ndcg_at_k(ranked_ids: Sequence[str], relevant: Mapping[str, int], k: int) -> float:
    """Normalised Discounted Cumulative Gain at k (Jarvelin & Kekalainen, 2002).

    Returns 0.0 when no relevant documents exist, rather than NaN, so that
    unanswerable queries contribute a defined value to the mean.
    """
    ideal_grades = sorted(relevant.values(), reverse=True)[:k]
    idcg = _dcg(np.array([2**g - 1 for g in ideal_grades], dtype=float))
    if idcg == 0.0:
        return 0.0
    return _dcg(_gains(ranked_ids, relevant, k)) / idcg


def recall_at_k(ranked_ids: Sequence[str], relevant: Mapping[str, int], k: int) -> float:
    """Fraction of all relevant documents retrieved within the top-k."""
    positives = {d for d, g in relevant.items() if g > 0}
    if not positives:
        return 0.0
    return len(positives & set(ranked_ids[:k])) / len(positives)


def precision_at_k(ranked_ids: Sequence[str], relevant: Mapping[str, int], k: int) -> float:
    """Fraction of the top-k that is relevant. Divides by k, not by len(ranked_ids)."""
    if k <= 0:
        return 0.0
    positives = {d for d, g in relevant.items() if g > 0}
    return len(positives & set(ranked_ids[:k])) / k


def mrr_at_k(ranked_ids: Sequence[str], relevant: Mapping[str, int], k: int) -> float:
    """Reciprocal rank of the first relevant document within the top-k; 0.0 if none."""
    positives = {d for d, g in relevant.items() if g > 0}
    for i, doc_id in enumerate(ranked_ids[:k], start=1):
        if doc_id in positives:
            return 1.0 / i
    return 0.0
