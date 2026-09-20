"""The five rerankers, behind one protocol.

score orders the ranking; p_relevant is a probability claim the calibration
metrics hold the arm to. They are separate fields on purpose - see types.py.
"""

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from src.config import Config
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


class CrossEncoderModel(Protocol):
    """Duck-typed contract for whatever `CrossEncoderReranker.model` holds:
    the real `sentence_transformers.CrossEncoder` in production, a fake in
    tests. Both only need `.predict`."""

    def predict(self, pairs: list[list[str]]) -> np.ndarray: ...


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic, P(relevant) = 1 / (1 + exp(-x)).

    Naive exp(-x) overflows for very negative x and exp(x) overflows for very
    positive x; both branches below only ever exponentiate a non-positive
    number, so this can't produce inf/NaN or a probability outside [0, 1].
    """
    x = np.clip(x, -500, 500)
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


class CrossEncoderReranker:
    """Arm B. BAAI/bge-reranker-base - the strongest honest opponent.

    `model` is injectable so tests and CI never download torch or the model
    weights. Given policy 4 (equal effort for every arm), pair ordering
    matters here specifically: bge is trained on [query, passage] pairs, and
    reversing them would quietly handicap the main rival.
    """

    name = "cross_encoder"

    def __init__(self, cfg: Config, model: CrossEncoderModel | None = None) -> None:
        self.cfg = cfg
        if model is None:
            import torch
            from sentence_transformers import CrossEncoder

            # activation_fn=Identity is load-bearing, not tidiness. bge's
            # CrossEncoder applies Sigmoid inside predict() by default, so
            # leaving it alone would hand this class an already-squashed
            # probability, and rerank() would sigmoid it a second time:
            # a true logit of 2.85 (P=0.945) would be published as 0.72.
            # Verified against the real model on 2026-09-20 —
            # default predict() -> [0.945, 0.00048],
            # activation_fn=Identity -> [2.848, -7.643].
            model = CrossEncoder(
                cfg.CROSS_ENCODER, max_length=512, activation_fn=torch.nn.Identity()
            )
        self.model = model

    def rerank(self, query: Query, candidates: Sequence[Doc]) -> list[Scored]:
        """score = raw cross-encoder logit; p_relevant = sigmoid(logit).

        confidence is always None: bge emits no separate certainty signal
        distinct from the logit itself, and coercing it to 1.0 would hand
        this arm free narrowing under the gating policy it did not earn.
        """
        pairs = [[query.text, d.text] for d in candidates]
        logits = np.asarray(self.model.predict(pairs), dtype=float)
        probs = _sigmoid(logits)
        scored = [
            Scored(
                doc_id=d.doc_id,
                score=float(logit),
                p_relevant=float(np.clip(p, 0.0, 1.0)),
                confidence=None,
                meta={"logit": float(logit)},
            )
            for d, logit, p in zip(candidates, logits, probs, strict=True)
        ]
        return sorted(scored, key=lambda s: (-s.score, s.doc_id))
