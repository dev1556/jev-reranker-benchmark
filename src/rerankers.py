"""The five rerankers, behind one protocol.

score orders the ranking; p_relevant is a probability claim the calibration
metrics hold the arm to. They are separate fields on purpose - see types.py.
"""

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from src.types import Doc, Query, Scored


class Reranker(Protocol):
    name: str

    def rerank(self, query: Query, candidates: Sequence[Doc]) -> list[Scored]: ...


class CalibratorNotFittedError(RuntimeError):
    """Raised when arm E is used before `.fit()` on the dev split."""


def minmax(values: np.ndarray) -> np.ndarray:
    """Per-query min-max scaling to [0, 1].

    Constant input maps to 0.5: no spread means no information, and it avoids
    a divide-by-zero that would otherwise produce NaN probabilities.
    """
    values = np.asarray(values, dtype=float)
    lo, hi = values.min(), values.max()
    if hi - lo < 1e-12:
        return np.full_like(values, 0.5)
    return (values - lo) / (hi - lo)


class CosineReranker:
    """Arm A. Identity reorder of the pool; p_relevant is per-query min-max.

    Deliberately naive. This arm exists to be the floor, and quantifying how
    fictional its p_relevant is constitutes hypothesis H2.
    """

    name = "cosine"

    def __init__(self, sims: dict[str, dict[str, float]]) -> None:
        self.sims = sims

    def rerank(self, query: Query, candidates: Sequence[Doc]) -> list[Scored]:
        """score = raw cosine similarity; p_relevant = that similarity min-maxed
        to [0, 1] within this query's pool (not a calibrated probability)."""
        raw = np.array([self.sims[query.query_id][d.doc_id] for d in candidates])
        probs = minmax(raw)
        scored = [
            Scored(doc_id=d.doc_id, score=float(r), p_relevant=float(p), confidence=None)
            for d, r, p in zip(candidates, raw, probs, strict=True)
        ]
        return sorted(scored, key=lambda s: (-s.score, s.doc_id))


class PlattReranker:
    """Arm E. Logistic recalibration of arm A, fitted on dev and applied frozen.

    Exists to pre-empt the obvious rebuttal to any calibration win by arm D:
    that a trivial recalibration of cosine would have done the same.
    """

    name = "platt"

    def __init__(self, base: CosineReranker) -> None:
        self.base = base
        self._model = None

    def fit(self, scores: np.ndarray, labels: np.ndarray, split: str = "dev") -> None:
        """Fit the logistic (Platt) calibrator on dev-split (score, label) pairs.

        Refuses any split other than 'dev' - CLAUDE.md non-negotiable #1: never
        tune on test.
        """
        if split != "dev":
            raise ValueError(
                f"calibrator must be fitted on dev, got split={split!r} "
                f"(CLAUDE.md non-negotiable #1)"
            )
        from sklearn.linear_model import LogisticRegression

        self._model = LogisticRegression().fit(
            np.asarray(scores, dtype=float).reshape(-1, 1), np.asarray(labels, dtype=int)
        )

    def transform(self, scores: np.ndarray) -> np.ndarray:
        """Map raw cosine scores to calibrated P(relevant) via the fitted model."""
        if self._model is None:
            raise CalibratorNotFittedError("PlattReranker.fit() must run on dev first")
        return self._model.predict_proba(np.asarray(scores, dtype=float).reshape(-1, 1))[:, 1]

    def rerank(self, query: Query, candidates: Sequence[Doc]) -> list[Scored]:
        """Rank exactly like the base cosine arm (Platt is monotone); p_relevant
        is the calibrated probability in place of arm A's raw min-max."""
        base = self.base.rerank(query, candidates)
        probs = self.transform(np.array([s.score for s in base]))
        return [
            Scored(doc_id=s.doc_id, score=s.score, p_relevant=float(p), confidence=None)
            for s, p in zip(base, probs, strict=True)
        ]
