"""Pure metric functions. Imports numpy and stdlib typing only — never an arm, never a client.

Quarantined on purpose: these are the numbers a public benchmark lives or dies by,
and they must be auditable without reading a line of API code.

Each public function takes the judgments for a single query — one value of
src.types.Qrels (i.e. Qrels[query_id]) — as `relevant: Mapping[str, int]`. It does
not import Qrels itself: Qrels types the whole corpus (dict[str, dict[str, int]]),
not a single query's relevance dict, so importing it here would mislabel the
per-query argument these functions actually take.
"""

from collections.abc import Mapping, Sequence

import numpy as np


def _dedupe(ranked_ids: Sequence[str]) -> list[str]:
    """First-occurrence-preserving dedupe.

    Applied uniformly before every metric computes anything, so a ranker that
    repeats a doc_id cannot inflate a list-based metric (nDCG, MRR) relative to
    the set-based ones (Recall, Precision) — a repeated document degrades to
    "shown once", not "shown twice for free".
    """
    seen: set[str] = set()
    out: list[str] = []
    for doc_id in ranked_ids:
        if doc_id not in seen:
            seen.add(doc_id)
            out.append(doc_id)
    return out


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
    unanswerable queries contribute a defined value to the mean. Returns 0.0
    for k <= 0 rather than relying on Python's negative-slice semantics.
    """
    if k <= 0:
        return 0.0
    ranked_ids = _dedupe(ranked_ids)
    ideal_grades = sorted(relevant.values(), reverse=True)[:k]
    idcg = _dcg(np.array([2**g - 1 for g in ideal_grades], dtype=float))
    if idcg == 0.0:
        return 0.0
    return _dcg(_gains(ranked_ids, relevant, k)) / idcg


def recall_at_k(ranked_ids: Sequence[str], relevant: Mapping[str, int], k: int) -> float:
    """Recall@k = |relevant docs retrieved in top-k| / |all relevant docs|.

    Standard IR recall (Manning, Raghavan & Schuetze, *Introduction to
    Information Retrieval*, 2008, ch. 8). Returns 0.0 for k <= 0.
    """
    if k <= 0:
        return 0.0
    ranked_ids = _dedupe(ranked_ids)
    positives = {d for d, g in relevant.items() if g > 0}
    if not positives:
        return 0.0
    return len(positives & set(ranked_ids[:k])) / len(positives)


def precision_at_k(ranked_ids: Sequence[str], relevant: Mapping[str, int], k: int) -> float:
    """Precision@k = |relevant docs in top-k| / k.

    Standard IR precision-at-k (Manning, Raghavan & Schuetze, *Introduction to
    Information Retrieval*, 2008, ch. 8). Divides by k, not by len(ranked_ids),
    and returns 0.0 for k <= 0.
    """
    if k <= 0:
        return 0.0
    ranked_ids = _dedupe(ranked_ids)
    positives = {d for d, g in relevant.items() if g > 0}
    return len(positives & set(ranked_ids[:k])) / k


def mrr_at_k(ranked_ids: Sequence[str], relevant: Mapping[str, int], k: int) -> float:
    """MRR@k = 1 / rank of the first relevant document within the top-k, else 0.0.

    Mean Reciprocal Rank as defined for TREC-8 Question Answering (Voorhees,
    1999). Returns 0.0 for k <= 0.
    """
    if k <= 0:
        return 0.0
    ranked_ids = _dedupe(ranked_ids)
    positives = {d for d, g in relevant.items() if g > 0}
    for i, doc_id in enumerate(ranked_ids[:k], start=1):
        if doc_id in positives:
            return 1.0 / i
    return 0.0
