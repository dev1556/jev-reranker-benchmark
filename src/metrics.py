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
from dataclasses import dataclass

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


@dataclass(frozen=True)
class Bin:
    lo: float
    hi: float
    mean_pred: float
    frac_pos: float
    count: int


@dataclass(frozen=True)
class BrierDecomposition:
    brier: float
    reliability: float
    resolution: float
    uncertainty: float


def _equal_mass_edges(probs: np.ndarray, n_bins: int) -> np.ndarray:
    """Quantile bin edges. Equal-mass rather than equal-width because reranker
    probabilities pile up near 0 — equal-width bins would leave most bins empty
    and understate ECE."""
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(probs, qs)
    edges[0], edges[-1] = -np.inf, np.inf
    return np.unique(edges)


def reliability_bins(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> list[Bin]:
    """Partition predictions into equal-mass bins of (mean predicted, observed frequency)."""
    probs, labels = np.asarray(probs, float), np.asarray(labels, int)
    if probs.size == 0:
        return []
    edges = _equal_mass_edges(probs, n_bins)
    idx = np.clip(np.digitize(probs, edges[1:-1], right=False), 0, len(edges) - 2)
    out: list[Bin] = []
    for b in range(len(edges) - 1):
        mask = idx == b
        if not mask.any():
            continue
        out.append(
            Bin(
                lo=float(edges[b]),
                hi=float(edges[b + 1]),
                mean_pred=float(probs[mask].mean()),
                frac_pos=float(labels[mask].mean()),
                count=int(mask.sum()),
            )
        )
    return out


def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error: sum_b (n_b/N) * |mean_pred_b - frac_pos_b|.

    Naeini et al. (2015), equal-mass binning.
    """
    bins = reliability_bins(probs, labels, n_bins)
    n = int(np.asarray(probs).size)
    if n == 0 or not bins:
        return float("nan")
    return float(sum(b.count / n * abs(b.mean_pred - b.frac_pos) for b in bins))


def brier_decomposition(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> BrierDecomposition:
    """Brier score with Murphy's (1973) reliability-resolution-uncertainty split."""
    probs, labels = np.asarray(probs, float), np.asarray(labels, int)
    brier = float(np.mean((probs - labels) ** 2))
    base = float(labels.mean()) if labels.size else float("nan")
    bins = reliability_bins(probs, labels, n_bins)
    n = probs.size
    rel = sum(b.count / n * (b.mean_pred - b.frac_pos) ** 2 for b in bins)
    res = sum(b.count / n * (b.frac_pos - base) ** 2 for b in bins)
    return BrierDecomposition(brier, float(rel), float(res), base * (1 - base))


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC curve via the rank (Mann-Whitney U) identity.

    Returns NaN when only one class is present — undefined, and saying so is
    safer than returning a plausible-looking 0.5.
    """
    scores, labels = np.asarray(scores, float), np.asarray(labels, int)
    n_pos, n_neg = int((labels == 1).sum()), int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, scores.size + 1)
    # average ranks within ties, so tied scores cannot inflate the statistic
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.zeros(counts.size)
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
