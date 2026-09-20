import numpy as np
import pytest

from src.config import CONFIG
from src.embed import FakeEmbedder, assert_pools_identical, build_pools, cosine_top_k
from src.types import CandidatePool, Corpus, Doc, Query


def test_cosine_top_k_orders_by_similarity() -> None:
    q = np.array([1.0, 0.0])
    docs = np.array([[0.0, 1.0], [1.0, 0.0], [0.7, 0.7]])
    assert cosine_top_k(q, docs, ["orthogonal", "identical", "between"], 3) == [
        "identical",
        "between",
        "orthogonal",
    ]


def test_cosine_is_magnitude_invariant() -> None:
    """Cosine must ignore vector length — a longer document vector is not a
    more relevant one. Equal-direction docs tie, so the doc_id tiebreak decides."""
    q = np.array([1.0, 0.0])
    docs = np.array([[5.0, 0.0], [1.0, 0.0]])
    assert cosine_top_k(q, docs, ["long", "short"], 2) == ["long", "short"]


def test_cosine_top_k_truncates_to_k() -> None:
    q = np.array([1.0, 0.0])
    docs = np.random.default_rng(0).normal(size=(50, 2))
    assert len(cosine_top_k(q, docs, [f"d{i}" for i in range(50)], 10)) == 10


def test_cosine_handles_zero_vector_without_nan() -> None:
    q = np.array([1.0, 0.0])
    docs = np.array([[0.0, 0.0], [1.0, 0.0]])
    assert cosine_top_k(q, docs, ["zero", "match"], 2)[0] == "match"


def _fake_corpus(n: int = 80) -> Corpus:
    return Corpus(
        "fake",
        {f"d{i}": Doc(f"d{i}", f"text {i}") for i in range(n)},
        {"q1": Query("q1", "query")},
        {"q1": {"d1": 1}},
    )


def test_build_pools_returns_pool_size_candidates() -> None:
    corpus = _fake_corpus()
    pools = build_pools(corpus, [corpus.queries["q1"]], FakeEmbedder(), CONFIG)
    assert len(pools["q1"].docs) == CONFIG.POOL_SIZE


def test_build_pools_is_deterministic() -> None:
    corpus = _fake_corpus()
    a = build_pools(corpus, [corpus.queries["q1"]], FakeEmbedder(), CONFIG)
    b = build_pools(corpus, [corpus.queries["q1"]], FakeEmbedder(), CONFIG)
    assert [d.doc_id for d in a["q1"].docs] == [d.doc_id for d in b["q1"].docs]


def test_build_pools_caps_at_corpus_size() -> None:
    """A corpus smaller than POOL_SIZE must not crash or pad — arms simply see
    fewer candidates, and the pool stays identical across arms."""
    corpus = _fake_corpus(n=7)
    pools = build_pools(corpus, [corpus.queries["q1"]], FakeEmbedder(), CONFIG)
    assert len(pools["q1"].docs) == 7


def test_fake_embedder_is_stable_across_instances() -> None:
    """FakeEmbedder seeds from a stable digest, not Python's per-process
    randomised hash(), so fixtures do not shift between runs."""
    assert np.array_equal(FakeEmbedder().embed(["a", "b"]), FakeEmbedder().embed(["a", "b"]))


def test_assert_pools_identical_passes_for_equal_pools() -> None:
    pool = CandidatePool(Query("q1", "q"), (Doc("d1", "t"),))
    assert_pools_identical({"cosine": {"q1": pool}, "jev": {"q1": pool}})


def test_assert_pools_identical_raises_on_divergence() -> None:
    """FR-2: every arm must rerank the identical pool. A divergence silently
    turns the benchmark into a retrieval comparison instead of a reranking one."""
    a = CandidatePool(Query("q1", "q"), (Doc("d1", "t"),))
    b = CandidatePool(Query("q1", "q"), (Doc("d2", "t"),))
    with pytest.raises(AssertionError, match="identical"):
        assert_pools_identical({"cosine": {"q1": a}, "jev": {"q1": b}})
