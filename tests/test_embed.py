from dataclasses import replace

import numpy as np
import pytest

from src.config import CONFIG
from src.embed import (
    FakeEmbedder,
    OpenAIEmbedder,
    assert_pools_identical,
    build_pools,
    build_pools_and_sims,
    cosine_top_k,
)
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


def test_build_pools_and_sims_pool_matches_build_pools() -> None:
    """build_pools is now a thin wrapper — same pool, same order, either way."""
    corpus = _fake_corpus()
    queries = [corpus.queries["q1"]]
    pools_only = build_pools(corpus, queries, FakeEmbedder(), CONFIG)
    pools, _sims = build_pools_and_sims(corpus, queries, FakeEmbedder(), CONFIG)
    assert [d.doc_id for d in pools["q1"].docs] == [d.doc_id for d in pools_only["q1"].docs]


def test_sims_cover_every_pooled_doc_for_every_query() -> None:
    corpus = _fake_corpus()
    queries = [corpus.queries["q1"]]
    pools, sims = build_pools_and_sims(corpus, queries, FakeEmbedder(), CONFIG)
    assert set(sims["q1"]) == {d.doc_id for d in pools["q1"].docs}


def test_sims_are_the_values_that_selected_the_top_k() -> None:
    """The returned similarities must be the same numbers used to rank the
    pool, not a fresh recomputation: re-sorting the pool by (-sim, doc_id)
    using the returned sims must reproduce the pool's own order exactly."""
    corpus = _fake_corpus(n=80)
    queries = [corpus.queries["q1"]]
    pools, sims = build_pools_and_sims(corpus, queries, FakeEmbedder(), CONFIG)
    pooled_ids = [d.doc_id for d in pools["q1"].docs]
    order = sorted(pooled_ids, key=lambda d: (-sims["q1"][d], d))
    assert order == pooled_ids


def test_sims_pooled_docs_beat_every_unpooled_doc() -> None:
    corpus = _fake_corpus(n=80)
    queries = [corpus.queries["q1"]]
    pools, sims = build_pools_and_sims(corpus, queries, FakeEmbedder(), CONFIG)
    pooled_ids = {d.doc_id for d in pools["q1"].docs}
    assert pooled_ids < set(corpus.docs)  # sanity: pool is a strict subset here
    min_pooled_sim = min(sims["q1"].values())
    doc_matrix = FakeEmbedder().embed([corpus.docs[d].text for d in sorted(corpus.docs)])
    query_vec = FakeEmbedder().embed([queries[0].text])[0]
    doc_ids = sorted(corpus.docs)
    all_sims = dict(
        zip(
            doc_ids,
            (doc_matrix / np.linalg.norm(doc_matrix, axis=-1, keepdims=True))
            @ (query_vec / np.linalg.norm(query_vec)),
            strict=True,
        )
    )
    unpooled_sims = [v for k, v in all_sims.items() if k not in pooled_ids]
    assert all(v <= min_pooled_sim for v in unpooled_sims)


def _counting_openai_client(dim: int = 4):
    class _Client:
        def __init__(self) -> None:
            self.calls = 0
            self.embeddings = self

        def create(self, model: str, input: list[str]):  # noqa: A002 - matches openai's kwarg
            self.calls += 1
            return type(
                "R",
                (),
                {"data": [type("D", (), {"embedding": [0.1] * dim})() for _ in input]},
            )()

    return _Client()


class _ExplodingClient:
    """Any call is a test failure — used to prove a cache hit makes zero API calls."""

    def __init__(self) -> None:
        self.embeddings = self

    def create(self, model: str, input: list[str]):  # noqa: A002
        raise AssertionError("cache hit must not call the API")


def test_embedding_cache_hit_makes_zero_api_calls(tmp_path) -> None:
    cfg = replace(CONFIG, DATA_DIR=tmp_path)
    first = _counting_openai_client()
    warm = OpenAIEmbedder(cfg, client=first).embed(["alpha", "beta"])
    assert first.calls == 1

    cold = OpenAIEmbedder(cfg, client=_ExplodingClient()).embed(["alpha", "beta"])
    assert np.array_equal(cold, warm)


def test_embedding_cache_misses_on_text_change(tmp_path) -> None:
    cfg = replace(CONFIG, DATA_DIR=tmp_path)
    client = _counting_openai_client()
    OpenAIEmbedder(cfg, client=client).embed(["alpha", "beta"])
    OpenAIEmbedder(cfg, client=client).embed(["alpha", "gamma"])
    assert client.calls == 2


def test_embedding_cache_misses_on_reorder(tmp_path) -> None:
    """A reordering must not silently reuse a mismatched matrix."""
    cfg = replace(CONFIG, DATA_DIR=tmp_path)
    client = _counting_openai_client()
    OpenAIEmbedder(cfg, client=client).embed(["alpha", "beta"])
    OpenAIEmbedder(cfg, client=client).embed(["beta", "alpha"])
    assert client.calls == 2


def test_embedding_cache_misses_on_model_change(tmp_path) -> None:
    cfg_a = replace(CONFIG, DATA_DIR=tmp_path, EMBED_MODEL="model-a")
    cfg_b = replace(CONFIG, DATA_DIR=tmp_path, EMBED_MODEL="model-b")
    client = _counting_openai_client()
    OpenAIEmbedder(cfg_a, client=client).embed(["alpha", "beta"])
    OpenAIEmbedder(cfg_b, client=client).embed(["alpha", "beta"])
    assert client.calls == 2


def test_embedding_cache_write_is_atomic_no_tmp_left_behind(tmp_path) -> None:
    cfg = replace(CONFIG, DATA_DIR=tmp_path)
    OpenAIEmbedder(cfg, client=_counting_openai_client()).embed(["alpha", "beta"])
    leftovers = list((tmp_path / "embeddings").glob("*.tmp"))
    assert leftovers == []
    npz_files = list((tmp_path / "embeddings").glob("*.npz"))
    assert len(npz_files) == 1
