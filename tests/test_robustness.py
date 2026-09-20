import pytest

from src.config import CONFIG, Config
from src.robustness import INJECTION_STYLES, inject, rank_inflation, run_probe
from src.types import CandidatePool, Doc, Query, Scored

DOC = Doc("d1", "An unrelated paragraph about gardening.")


def test_all_three_styles_are_registered() -> None:
    assert set(INJECTION_STYLES) == {
        "naive_imperative",
        "keyword_stuffing",
        "framed_instruction",
    }


@pytest.mark.parametrize("style", ["naive_imperative", "keyword_stuffing", "framed_instruction"])
def test_injection_preserves_the_original_text(style: str) -> None:
    """The probe measures what injected content ADDS. Replacing the original
    text would confound the measurement with plain topic change."""
    out = inject(DOC, "what causes rain", style)
    assert DOC.text in out.text


@pytest.mark.parametrize("style", ["naive_imperative", "keyword_stuffing", "framed_instruction"])
def test_injection_changes_the_text(style: str) -> None:
    assert inject(DOC, "what causes rain", style).text != DOC.text


def test_injection_preserves_doc_id() -> None:
    """Rank inflation is tracked by id; a changed id would break the join."""
    assert inject(DOC, "q", "naive_imperative").doc_id == "d1"


def test_keyword_stuffing_repeats_the_query() -> None:
    assert inject(DOC, "what causes rain", "keyword_stuffing").text.count("what causes rain") >= 2


def test_rank_inflation_counts_positions_gained() -> None:
    base = ["a", "b", "c", "d"]
    injected = ["c", "a", "b", "d"]  # c moved from index 2 to 0 -> +2
    assert rank_inflation(base, injected, ["c"]) == pytest.approx(2.0)


def test_rank_inflation_is_negative_when_a_chunk_drops() -> None:
    assert rank_inflation(["a", "b"], ["b", "a"], ["a"]) == pytest.approx(-1.0)


def test_rank_inflation_averages_over_injected_docs() -> None:
    base = ["a", "b", "c", "d"]
    injected = ["c", "d", "a", "b"]  # c +2, d +2
    assert rank_inflation(base, injected, ["c", "d"]) == pytest.approx(2.0)


def test_rank_inflation_ignores_missing_ids() -> None:
    assert rank_inflation(["a", "b"], ["a", "b"], ["zzz"]) == pytest.approx(0.0)


def test_unknown_style_raises() -> None:
    with pytest.raises(KeyError):
        inject(DOC, "q", "mind_control")


def test_rank_inflation_cannot_report_positive_gain_for_an_already_top_ranked_doc() -> None:
    """A doc at rank 0 has nowhere higher to go. This isn't a property we bolt
    on with a special case: base_index - injected_index is <= 0 whenever
    base_index is 0, so the metric can't report a spurious improvement here."""
    base = ["a", "b", "c"]
    injected = ["a", "c", "b"]  # a was already first and stays first
    assert rank_inflation(base, injected, ["a"]) <= 0.0


class _LexicalReranker:
    """Test double standing in for a real arm: ranks candidates by how many
    times the query text appears in their text (case-insensitive), which is
    exactly the crude signal keyword-stuffing is designed to exploit. Ties
    (usually score 0) break on doc_id ascending, same as the real arms."""

    name = "lexical_fake"

    def rerank(self, query: Query, candidates: list[Doc]) -> list[Scored]:
        scored = [
            Scored(
                doc_id=d.doc_id,
                score=float(d.text.lower().count(query.text.lower())),
                p_relevant=0.5,
            )
            for d in candidates
        ]
        return sorted(scored, key=lambda s: (-s.score, s.doc_id))


class _AlwaysExcludedReranker:
    """Test double simulating every call failing, matching how arms C/D mark
    a failed API call: score=-inf, p_relevant=0.0, meta['excluded']=True."""

    name = "always_excluded_fake"

    def rerank(self, query: Query, candidates: list[Doc]) -> list[Scored]:
        return [
            Scored(doc_id=d.doc_id, score=float("-inf"), p_relevant=0.0, meta={"excluded": True})
            for d in candidates
        ]


def test_run_probe_is_deterministic_given_the_same_seed() -> None:
    """Which document gets injected is picked by a seeded RNG. The same seed
    over the same pools must pick the same documents and so produce a
    bit-identical result."""
    pool = CandidatePool(
        query=Query("q1", "what causes rain"),
        docs=(
            Doc("d1", "The water cycle moves moisture through the atmosphere."),
            Doc("d2", "A recipe for chocolate cake."),
            Doc("d3", "Notes on 19th century railway timetables."),
        ),
    )
    qrels = {"q1": {"d1": 1}}
    cfg = Config(SEED=7)
    first = run_probe(_LexicalReranker(), [pool], qrels, "keyword_stuffing", cfg)
    second = run_probe(_LexicalReranker(), [pool], qrels, "keyword_stuffing", cfg)
    assert first == second


def test_run_probe_shows_no_inflation_when_the_only_candidate_is_injected() -> None:
    """With one candidate there is nowhere to rise from or fall to: injecting
    it must not manufacture inflation out of an empty ranking race."""
    pool = CandidatePool(
        query=Query("q1", "what causes rain"),
        docs=(Doc("d1", "The water cycle moves moisture through the atmosphere."),),
    )
    # d1 must be irrelevant to be an injection target at all (BRD §8), so the
    # qrels row names a document that is not in the pool.
    result = run_probe(
        _LexicalReranker(), [pool], {"q1": {"d99": 1}}, "keyword_stuffing", Config(SEED=1)
    )
    assert result.n_injected == 1
    assert result.mean_inflation == pytest.approx(0.0)


def test_run_probe_excludes_failed_calls_from_n_injected() -> None:
    """An excluded row (meta['excluded'] is True, an errored API call) is not
    a document that 'failed to rise' - it was never scored, and must not be
    counted as measured."""
    pool = CandidatePool(
        query=Query("q1", "what causes rain"),
        docs=(Doc("d1", "The water cycle moves moisture through the atmosphere."),),
    )
    result = run_probe(
        _AlwaysExcludedReranker(), [pool], {"q1": {"d1": 1}}, "keyword_stuffing", Config(SEED=1)
    )
    assert result.n_injected == 0
    assert result.mean_inflation == pytest.approx(0.0)
    assert result.ndcg_delta == pytest.approx(0.0)


def test_run_probe_reports_style_and_arm_name() -> None:
    pool = CandidatePool(
        query=Query("q1", "what causes rain"),
        docs=(Doc("d1", "The water cycle moves moisture through the atmosphere."),),
    )
    result = run_probe(
        _LexicalReranker(), [pool], {"q1": {"d1": 1}}, "framed_instruction", Config(SEED=1)
    )
    assert result.style == "framed_instruction"
    assert result.arm == "lexical_fake"


def test_run_probe_mean_inflation_is_never_negative_of_positions_that_dont_exist() -> None:
    """Sanity bound: on a manipulable arm, injecting keyword stuffing into a
    non-top candidate can only move it up or leave it - never produces a
    mean_inflation more negative than the pool allows."""
    pool = CandidatePool(
        query=Query("q1", "what causes rain"),
        docs=(
            Doc("d1", "The water cycle moves moisture through the atmosphere."),
            Doc("d2", "A recipe for chocolate cake."),
        ),
    )
    result = run_probe(
        _LexicalReranker(), [pool], {"q1": {"d1": 1}}, "keyword_stuffing", Config(SEED=7)
    )
    assert result.n_injected == 1
    assert -1.0 <= result.mean_inflation <= 1.0


def test_probe_only_injects_documents_that_are_irrelevant() -> None:
    """BRD §8 says the injected chunks are genuinely irrelevant ones. Injecting a
    document that is already relevant measures nothing: the attack modelled is an
    irrelevant document climbing, and a relevant one already sits near the top
    with little room to rise, which would dilute the mean toward zero and read as
    robustness the arm never earned."""
    docs = tuple(Doc(f"d{i}", f"passage about rain {i}") for i in range(6))
    pool = CandidatePool(query=Query("q1", "what causes rain"), docs=docs)
    # every document relevant -> nothing is a legitimate target
    all_relevant = {"q1": {d.doc_id: 1 for d in docs}}
    result = run_probe(_LexicalReranker(), [pool], all_relevant, "keyword_stuffing", CONFIG)
    assert result.n_injected == 0


def test_probe_injects_up_to_the_configured_count_per_query() -> None:
    """cfg.N_INJECTED_PER_QUERY documents per query, capped by how many
    irrelevant candidates the pool actually holds."""
    docs = tuple(Doc(f"d{i}", f"passage about rain {i}") for i in range(20))
    pool = CandidatePool(query=Query("q1", "what causes rain"), docs=docs)
    qrels = {"q1": {"d0": 1}}  # 19 irrelevant candidates available
    result = run_probe(_LexicalReranker(), [pool], qrels, "keyword_stuffing", CONFIG)
    assert result.n_injected == CONFIG.N_INJECTED_PER_QUERY

    small = CandidatePool(query=Query("q2", "what causes rain"), docs=docs[:3])
    capped = run_probe(_LexicalReranker(), [small], {"q2": {"d0": 1}}, "keyword_stuffing", CONFIG)
    assert capped.n_injected == 2  # only d1, d2 are irrelevant here
