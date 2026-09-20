"""Arm D (Jev) and its cookbook variant.

Key API-contract facts this file must exercise (see task-12 brief deltas,
verified against typesafe-sdk 0.7.0 on 2026-09-20):

- `probabilities`/`legend` are `dict[int, ...]` in the SDK but string-keyed
  over the wire and after a JSON cache round-trip. A cache MISS hands
  `compose_relevance` int keys; a cache HIT hands it string keys. Both must
  produce the same answer, or a second run of the same query silently
  changes p_relevant.
- `response.answers[key]` holds pydantic models in production
  (`ScoreAnswer`/`NoulAnswer`); `JevReranker` calls `.model_dump()` at the
  client boundary so everything downstream (cache, compose_relevance, these
  tests) only ever sees plain dicts.
- a `noul` is a probability in [0, 1], never a boolean.
"""

import pytest

from src.cache import ResponseCache
from src.config import CONFIG
from src.rerankers import JevCookbookReranker, JevReranker, compose_relevance
from src.types import Doc, Query

QUERY = Query("q1", "a claim")
DOCS = [Doc("d1", "one"), Doc("d2", "two")]


def _answers(
    topical: float,
    answers_q: float,
    conf: float,
    probs: dict | None = None,
    cookbook: float = 0.5,
    contradictory: float = 0.1,
) -> dict:
    """A full four-question payload, shaped like `ScoreAnswer.model_dump()` /
    `NoulAnswer.model_dump()` — i.e. int-keyed `probabilities`, matching a
    fresh (cache-miss) response. `test_compose_handles_a_json_cache_round_trip`
    covers the string-keyed (cache-hit) shape separately."""
    return {
        "topical_overlap": {
            "type": "score",
            "score": topical,
            "confidence": 0.9,
            "legend": {0: "a", 1: "b", 2: "c", 3: "d"},
            "probabilities": probs.get("topical")
            if probs and "topical" in probs
            else {0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4},
        },
        "answers_query": {
            "type": "score",
            "score": answers_q,
            "confidence": conf,
            "legend": {0: "a", 1: "b", 2: "c"},
            "probabilities": (probs.get("answers") if probs and "answers" in probs else probs)
            or {0: 0.2, 1: 0.3, 2: 0.5},
        },
        "is_contradictory": {"type": "noul", "noul": contradictory},
        "cookbook_relevant": {"type": "noul", "noul": cookbook},
    }


class FakeJev:
    """Mimics `client.system_one(...) -> response.answers[key]`.

    Payload values are already plain dicts (as `.model_dump()` would return
    them), not real pydantic objects — `JevReranker` must accept both via its
    `_dump` boundary helper, since a live `ScoreAnswer`/`NoulAnswer` has
    `.model_dump()` and these test doubles do not.
    """

    def __init__(self, payloads: list[dict]) -> None:
        self.payloads, self.calls = payloads, 0

    def system_one(self, state: str, questions: dict, model: str = ""):
        payload = self.payloads[self.calls % len(self.payloads)]
        self.calls += 1
        return type("R", (), {"answers": payload})()


# ---- compose_relevance ------------------------------------------------


def test_compose_uses_configured_weights() -> None:
    # topical 3/3 = 1.0, answers 2/2 = 1.0 -> score 1.0
    score, _, _ = compose_relevance(_answers(3.0, 2.0, 0.8), CONFIG)
    assert score == pytest.approx(CONFIG.W_TOPICAL + CONFIG.W_ANSWERS)


def test_compose_normalises_each_score_by_its_own_max_level() -> None:
    """topical has 4 levels (max 3), answers_query has 3 (max 2). Dividing both
    by the same constant would silently weight them wrong."""
    score, _, _ = compose_relevance(_answers(1.5, 1.0, 0.8), CONFIG)
    assert score == pytest.approx(CONFIG.W_TOPICAL * 0.5 + CONFIG.W_ANSWERS * 0.5)


def test_p_relevant_is_mass_on_top_two_answer_levels() -> None:
    probs = {"answers": {0: 0.2, 1: 0.3, 2: 0.5}}
    _, p, _ = compose_relevance(_answers(2.0, 1.0, 0.8, probs), CONFIG)
    assert p == pytest.approx(0.8)  # 0.3 + 0.5


def test_confidence_comes_from_answers_query() -> None:
    _, _, c = compose_relevance(_answers(2.0, 1.0, 0.42), CONFIG)
    assert c == pytest.approx(0.42)


def test_compose_output_is_a_valid_probability() -> None:
    probs = {"answers": {0: 0.0, 1: 0.0, 2: 1.0}}
    _, p, _ = compose_relevance(_answers(3.0, 2.0, 0.9, probs), CONFIG)
    assert 0.0 <= p <= 1.0


def test_compose_handles_a_json_cache_round_trip() -> None:
    """A cache HIT loads probabilities through `json.loads`, which always
    produces string keys, even though the SDK's own `ScoreAnswer.probabilities`
    is `dict[int, float]`. Both key shapes must compose to the identical
    answer, or the second (cached) run of a query silently reports a
    different p_relevant than the first (live) run — the exact regression the
    API-contract deltas warn about."""
    int_keyed = _answers(2.0, 1.0, 0.8, {"answers": {0: 0.2, 1: 0.3, 2: 0.5}})
    str_keyed = _answers(2.0, 1.0, 0.8, {"answers": {"0": 0.2, "1": 0.3, "2": 0.5}})
    assert compose_relevance(int_keyed, CONFIG) == compose_relevance(str_keyed, CONFIG)


def test_compose_ignores_cookbook_relevant() -> None:
    """ORCHESTRATOR ADDITION: the pre-registered composed arm must use only
    the three original answers. Changing the fourth (cookbook) question must
    not move the composed score, p_relevant, or confidence at all."""
    low_cookbook = compose_relevance(_answers(2.0, 1.0, 0.8, cookbook=0.01), CONFIG)
    high_cookbook = compose_relevance(_answers(2.0, 1.0, 0.8, cookbook=0.99), CONFIG)
    assert low_cookbook == high_cookbook


# ---- JevReranker (arm D) ------------------------------------------------


def test_rerank_orders_by_composed_score(tmp_path) -> None:
    client = FakeJev(
        [
            _answers(0.0, 0.0, 0.9, {"answers": {0: 1.0, 1: 0.0, 2: 0.0}}),
            _answers(3.0, 2.0, 0.9, {"answers": {0: 0.0, 1: 0.0, 2: 1.0}}),
        ]
    )
    arm = JevReranker(CONFIG, ResponseCache(tmp_path), client=client)
    assert [s.doc_id for s in arm.rerank(QUERY, DOCS)] == ["d2", "d1"]


def test_per_level_probabilities_are_retained_in_meta(tmp_path) -> None:
    """The full distribution is the point of the project; reducing it to a
    scalar at this stage would discard what H2 measures."""
    arm = JevReranker(CONFIG, ResponseCache(tmp_path), client=FakeJev([_answers(2.0, 1.0, 0.8)]))
    assert "probabilities" in arm.rerank(QUERY, DOCS)[0].meta["answers_query"]


def test_second_run_hits_the_cache(tmp_path) -> None:
    client = FakeJev([_answers(2.0, 1.0, 0.8)])
    cache = ResponseCache(tmp_path)
    JevReranker(CONFIG, cache, client=client).rerank(QUERY, DOCS)
    first = client.calls
    JevReranker(CONFIG, cache, client=client).rerank(QUERY, DOCS)
    assert client.calls == first


def test_cache_hit_produces_the_same_score_as_the_original_call(tmp_path) -> None:
    """Regression test for the arm/cache-shard bug described in the deltas:
    `.get()` and `.put()` must be called with the SAME `arm=` value, or a hit
    silently reads the wrong (empty) shard and this arm looks like it never
    caches at all, or worse, recomposes from a different (string-keyed)
    payload than the live call produced."""
    client = FakeJev([_answers(2.0, 1.0, 0.8, {"answers": {0: 0.2, 1: 0.3, 2: 0.5}})])
    cache = ResponseCache(tmp_path)
    first = JevReranker(CONFIG, cache, client=client).rerank(QUERY, DOCS)
    second = JevReranker(CONFIG, cache, client=client).rerank(QUERY, DOCS)
    assert [(s.doc_id, s.score, s.p_relevant) for s in first] == [
        (s.doc_id, s.score, s.p_relevant) for s in second
    ]


def test_state_contains_only_the_one_pair(tmp_path) -> None:
    """BRD 4.4: Jev's documented weakness #5 is that accuracy falls as the
    state fills with unrelated content. One pair per state is the whole
    design; the cookbook confirms it independently: 'one request per
    candidate, no request sees another'."""
    seen: list[str] = []

    class Spy(FakeJev):
        def system_one(self, state, questions, model=""):
            seen.append(state)
            return super().system_one(state, questions, model)

    JevReranker(CONFIG, ResponseCache(tmp_path), client=Spy([_answers(2.0, 1.0, 0.8)])).rerank(
        QUERY, DOCS
    )
    assert "one" in seen[0] and "two" not in seen[0]


def test_api_failure_is_recorded_not_scored_zero(tmp_path) -> None:
    class Broken(FakeJev):
        def system_one(self, state, questions, model=""):
            raise RuntimeError("529")

    out = JevReranker(CONFIG, ResponseCache(tmp_path), client=Broken([])).rerank(QUERY, DOCS)
    assert all(s.meta.get("excluded") is True for s in out)


def test_api_failure_is_cached_so_a_warm_run_never_calls_live(tmp_path) -> None:
    """The published reproduce path is `git lfs pull && make report` with no
    API keys; an uncached failure would make a warm-cache run attempt a live
    call and die."""
    client_calls = {"n": 0}

    class Broken(FakeJev):
        def system_one(self, state, questions, model=""):
            client_calls["n"] += 1
            raise RuntimeError("529")

    cache = ResponseCache(tmp_path)
    client = Broken([])
    JevReranker(CONFIG, cache, client=client).rerank(QUERY, DOCS)
    JevReranker(CONFIG, cache, client=client).rerank(QUERY, DOCS)
    assert client_calls["n"] == len(DOCS)  # second run served entirely from cache


# ---- jev_cookbook (the fourth-question arm) -----------------------------


def test_cookbook_arm_scores_only_the_cookbook_noul(tmp_path) -> None:
    client = FakeJev([_answers(2.0, 1.0, 0.8, cookbook=0.77)])
    jev = JevReranker(CONFIG, ResponseCache(tmp_path), client=client)
    cookbook = JevCookbookReranker(jev)
    out = cookbook.rerank(QUERY, [DOCS[0]])
    assert out[0].score == pytest.approx(0.77)
    assert out[0].p_relevant == pytest.approx(0.77)
    assert out[0].confidence is None


def test_cookbook_arm_reuses_the_composed_arms_cached_call(tmp_path) -> None:
    """The fourth question rides the SAME call/state as the three
    pre-registered ones (independent questions over one state run in
    parallel) — it must never trigger a second request."""
    client = FakeJev([_answers(2.0, 1.0, 0.8)])
    cache = ResponseCache(tmp_path)
    jev = JevReranker(CONFIG, cache, client=client)
    jev.rerank(QUERY, DOCS)
    calls_after_composed = client.calls
    JevCookbookReranker(jev).rerank(QUERY, DOCS)
    assert client.calls == calls_after_composed  # served from the shared cache


def test_cookbook_and_composed_arms_can_disagree_on_ordering(tmp_path) -> None:
    """The whole point of running both from one call: they are measured, and
    can rank, independently. Doc d1 wins on the composed rubric but loses on
    the cookbook Noul, and vice versa for d2."""
    client = FakeJev(
        [
            _answers(3.0, 2.0, 0.9, {"answers": {0: 0.0, 1: 0.0, 2: 1.0}}, cookbook=0.1),
            _answers(0.0, 0.0, 0.9, {"answers": {0: 1.0, 1: 0.0, 2: 0.0}}, cookbook=0.9),
        ]
    )
    cache = ResponseCache(tmp_path)
    jev = JevReranker(CONFIG, cache, client=client)
    composed_order = [s.doc_id for s in jev.rerank(QUERY, DOCS)]
    cookbook_order = [s.doc_id for s in JevCookbookReranker(jev).rerank(QUERY, DOCS)]
    assert composed_order == ["d1", "d2"]
    assert cookbook_order == ["d2", "d1"]
    assert composed_order != cookbook_order
