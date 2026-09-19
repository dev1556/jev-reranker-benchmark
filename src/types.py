"""Shared data contracts. No logic, no project imports — everything depends on this."""

from dataclasses import dataclass, field

Qrels = dict[str, dict[str, int]]


@dataclass(frozen=True)
class Doc:
    doc_id: str
    text: str
    title: str = ""


@dataclass(frozen=True)
class Query:
    query_id: str
    text: str


@dataclass(frozen=True)
class Scored:
    """One reranked candidate.

    score:      arm-native ordering value. NOT comparable across arms.
    p_relevant: claimed P(document is relevant) in [0,1]. This is what the
                calibration metrics (ECE, Brier) hold the arm to.
    confidence: the arm's own certainty, or None where it has no such notion.
    """

    doc_id: str
    score: float
    p_relevant: float
    confidence: float | None = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.p_relevant <= 1.0:
            raise ValueError(f"p_relevant must be in [0,1], got {self.p_relevant}")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0,1] or None, got {self.confidence}")


@dataclass(frozen=True)
class CandidatePool:
    query: Query
    docs: tuple[Doc, ...]


@dataclass(frozen=True)
class Corpus:
    name: str
    docs: dict[str, Doc]
    queries: dict[str, Query]
    qrels: Qrels
