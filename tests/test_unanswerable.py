import pytest

from src.config import CONFIG
from src.embed import FakeEmbedder
from src.gating import Selection
from src.types import Corpus, Doc, Query
from src.unanswerable import abstention_report, build_cross_domain, build_gold_removed


def _corpus(name: str = "fake", n: int = 80) -> Corpus:
    return Corpus(
        name,
        {f"d{i}": Doc(f"d{i}", f"text {i}") for i in range(n)},
        {f"q{i}": Query(f"q{i}", f"query {i}") for i in range(3)},
        {f"q{i}": {f"d{i}": 1} for i in range(3)},
    )


def test_gold_removed_pool_contains_no_relevant_documents() -> None:
    """Ground truth by construction — this is what makes H5 labelling-free."""
    corpus = _corpus()
    queries = list(corpus.queries.values())
    pools = build_gold_removed(corpus, queries, FakeEmbedder(), CONFIG)
    for q in queries:
        gold = set(corpus.qrels[q.query_id])
        assert not gold & {d.doc_id for d in pools[q.query_id].docs}


def test_gold_removal_does_not_mutate_the_source_corpus() -> None:
    """A bug here would silently corrupt the main benchmark, which shares the
    corpus object."""
    corpus = _corpus()
    before = set(corpus.docs)
    build_gold_removed(corpus, list(corpus.queries.values()), FakeEmbedder(), CONFIG)
    assert set(corpus.docs) == before


def test_gold_removed_pool_is_still_full_size() -> None:
    corpus = _corpus()
    pools = build_gold_removed(corpus, list(corpus.queries.values()), FakeEmbedder(), CONFIG)
    assert len(pools["q0"].docs) == CONFIG.POOL_SIZE


def test_cross_domain_pool_draws_only_from_the_other_corpus() -> None:
    a, b = (
        _corpus("a"),
        Corpus("b", {f"z{i}": Doc(f"z{i}", f"other {i}") for i in range(80)}, {}, {}),
    )
    pools = build_cross_domain(b, list(a.queries.values()), FakeEmbedder(), CONFIG)
    assert all(d.doc_id.startswith("z") for d in pools["q0"].docs)


def test_gold_removed_embeds_the_corpus_once_for_all_queries() -> None:
    """Cost guard. The obvious implementation strips the corpus and calls
    build_pools per query, which re-embeds every passage each time: 300 FiQA
    queries over 57,638 passages is ~17M embedding calls instead of 57,638, and
    OpenAIEmbedder holds no cache. Two embed calls total (documents, then
    queries) regardless of query count."""

    class CountingEmbedder(FakeEmbedder):
        def __init__(self) -> None:
            super().__init__()
            self.texts_embedded = 0
            self.calls = 0

        def embed(self, texts: list[str]):
            self.calls += 1
            self.texts_embedded += len(texts)
            return super().embed(texts)

    corpus = _corpus("a")
    queries = list(corpus.queries.values())
    assert len(queries) > 1, "the guard is meaningless with a single query"
    embedder = CountingEmbedder()
    build_gold_removed(corpus, queries, embedder, CONFIG)
    assert embedder.calls == 2
    assert embedder.texts_embedded == len(corpus.docs) + len(queries)


def test_abstention_report_perfect_discriminator() -> None:
    sel = {
        "a1": Selection(["d1"], False, "confidence_gated"),
        "u1": Selection([], True, "confidence_gated"),
    }
    r = abstention_report(sel, answerable_ids=["a1"], unanswerable_ids=["u1"])
    assert r.rate_unanswerable == 1.0 and r.false_abstention_rate == 0.0


def test_abstention_report_flags_an_over_refusing_arm() -> None:
    """A reranker that abstains constantly scores a perfect refusal rate and is
    useless. Both numbers must always be reported together."""
    sel = {"a1": Selection([], True, "x"), "u1": Selection([], True, "x")}
    r = abstention_report(sel, ["a1"], ["u1"])
    assert r.rate_unanswerable == 1.0 and r.false_abstention_rate == 1.0


def test_abstention_report_requires_both_groups() -> None:
    with pytest.raises(ValueError, match="both"):
        abstention_report({}, [], ["u1"])
