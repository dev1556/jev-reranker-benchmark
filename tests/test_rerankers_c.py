import threading
import time
from dataclasses import replace

import pytest

from src.cache import ResponseCache
from src.config import CONFIG
from src.rerankers import LLMReranker, parse_llm_score
from src.types import Doc, Query

QUERY = Query("q1", "a query")
DOCS = [Doc("d1", "one"), Doc("d2", "two")]


def _text_response(text: str, stop_reason: str = "end_turn"):
    return type(
        "R",
        (),
        {
            "content": [type("C", (), {"type": "text", "text": text})()],
            "stop_reason": stop_reason,
            "usage": type("U", (), {"input_tokens": 100, "output_tokens": 5})(),
        },
    )()


class FakeAnthropic:
    """Mimics client.messages.create(...). Unlike the brief's original fixture,
    content blocks carry .type (arm C must select the text block, not assume
    content[0] is text - ORCHESTRATOR CORRECTIONS #1) and the response carries
    .stop_reason (arm C must treat a truncated reply as a failure - #4)."""

    def __init__(self, replies: list[str], stop_reason: str = "end_turn") -> None:
        self.replies, self.calls = replies, 0
        self.stop_reason = stop_reason
        self.messages = self

    def create(self, **kwargs):
        reply = self.replies[self.calls % len(self.replies)]
        self.calls += 1
        return type(
            "R",
            (),
            {
                "content": [type("C", (), {"type": "text", "text": reply})()],
                "stop_reason": self.stop_reason,
                "usage": type("U", (), {"input_tokens": 100, "output_tokens": 5})(),
            },
        )()


class NoTextBlockClient(FakeAnthropic):
    """Simulates a reply with no text block at all (e.g. only a non-text
    block) - the StopIteration from `next(... if b.type == "text")` must be
    caught, not raised out of rerank()."""

    def create(self, **kwargs):
        self.calls += 1
        return type(
            "R",
            (),
            {
                "content": [type("C", (), {"type": "thinking", "thinking": "hmm"})()],
                "stop_reason": "end_turn",
                "usage": type("U", (), {"input_tokens": 100, "output_tokens": 5})(),
            },
        )()


class TruncatedClient(FakeAnthropic):
    """A reply that looks like a valid grade but was cut off mid-generation."""

    def __init__(self, replies: list[str]) -> None:
        super().__init__(replies, stop_reason="max_tokens")


@pytest.mark.parametrize(
    "text,expected",
    [
        ("3", 3),
        ("Relevance: 2", 2),
        ("  0  ", 0),
        ("I would say 1 out of 3", 1),
    ],
)
def test_parse_llm_score_extracts_the_grade(text: str, expected: int) -> None:
    assert parse_llm_score(text) == expected


def test_parse_llm_score_rejects_out_of_range() -> None:
    with pytest.raises(ValueError, match="0-3"):
        parse_llm_score("9")


def test_parse_llm_score_rejects_unparseable() -> None:
    """Must raise, not default to 0. A silent 0 is an invisible wrong number."""
    with pytest.raises(ValueError):
        parse_llm_score("I cannot determine relevance")


def test_p_relevant_is_grade_over_three(tmp_path) -> None:
    arm = LLMReranker(CONFIG, ResponseCache(tmp_path), client=FakeAnthropic(["3", "0"]))
    out = {s.doc_id: s.p_relevant for s in arm.rerank(QUERY, DOCS)}
    assert out["d1"] == pytest.approx(1.0) and out["d2"] == pytest.approx(0.0)


def test_has_no_confidence(tmp_path) -> None:
    """Anthropic exposes no logprobs. Arm C genuinely has no confidence signal,
    and pretending otherwise would misrepresent the comparison."""
    arm = LLMReranker(CONFIG, ResponseCache(tmp_path), client=FakeAnthropic(["2"]))
    assert all(s.confidence is None for s in arm.rerank(QUERY, DOCS))


def test_second_run_hits_the_cache(tmp_path) -> None:
    client = FakeAnthropic(["2"])
    cache = ResponseCache(tmp_path)
    LLMReranker(CONFIG, cache, client=client).rerank(QUERY, DOCS)
    calls_after_first = client.calls
    LLMReranker(CONFIG, cache, client=client).rerank(QUERY, DOCS)
    assert client.calls == calls_after_first


def test_api_failure_is_recorded_not_scored_zero(tmp_path) -> None:
    """Non-negotiable: an errored call must never become 0.0. That would depress
    the arm's metrics in a way nobody would spot in a CSV."""

    class Broken(FakeAnthropic):
        def create(self, **kwargs):
            raise RuntimeError("503")

    out = LLMReranker(CONFIG, ResponseCache(tmp_path), client=Broken([])).rerank(QUERY, DOCS)
    assert all("error" in s.meta for s in out)
    assert all(s.meta.get("excluded") is True for s in out)


def test_no_text_block_is_recorded_not_scored(tmp_path) -> None:
    """ORCHESTRATOR CORRECTION #1: response.content[0] is not safe to assume is
    text. When no block in the reply is text at all, the resulting
    StopIteration must land as a recorded, excluded row - never a grade, and
    never an uncaught exception out of rerank()."""
    out = LLMReranker(CONFIG, ResponseCache(tmp_path), client=NoTextBlockClient([])).rerank(
        QUERY, DOCS
    )
    assert all("error" in s.meta for s in out)
    assert all(s.meta.get("excluded") is True for s in out)


def test_truncated_reply_is_recorded_not_parsed(tmp_path) -> None:
    """ORCHESTRATOR CORRECTION #4: stop_reason == "max_tokens" means the grade
    may be truncated. It must be treated as a failure even when the partial
    text happens to parse as a valid-looking digit - parsing it anyway would
    silently accept a reply the model never finished producing."""
    out = LLMReranker(CONFIG, ResponseCache(tmp_path), client=TruncatedClient(["2"])).rerank(
        QUERY, DOCS
    )
    assert all("error" in s.meta for s in out)
    assert all(s.meta.get("excluded") is True for s in out)


def test_a_cached_failure_does_not_call_the_api_again(tmp_path) -> None:
    """The published reproduce path is `git lfs pull && make report` with no API
    keys. If failures were not cached, a warm-cache run would attempt a live
    call for every historically-errored row, and this class raises at
    construction without a key — one old 503 would break the whole
    reproduction. Retrying a transient fault is a deliberate human action
    (delete the entry), not something a re-run does silently."""

    class Broken(FakeAnthropic):
        def create(self, **kwargs):
            self.calls += 1  # counted BEFORE raising, or the assertion below is vacuous
            raise RuntimeError("503")

    cache = ResponseCache(tmp_path)
    broken = Broken([])
    first = LLMReranker(CONFIG, cache, client=broken).rerank(QUERY, DOCS)
    assert broken.calls == len(DOCS), "the first run must actually attempt the calls"
    assert all(s.meta.get("excluded") is True for s in first)

    replay = Broken([])
    second = LLMReranker(CONFIG, cache, client=replay).rerank(QUERY, DOCS)
    assert replay.calls == 0, "a cached failure must be served from cache"
    assert all(s.meta.get("excluded") is True for s in second)
    assert [s.meta["error"] for s in second] == [s.meta["error"] for s in first]


def test_concurrency_is_bounded_by_semaphore(tmp_path) -> None:
    """arm C must run concurrently (SEMAPHORE workers), but never more than
    that many calls in flight at once, no matter how many candidates there
    are — proven with a fake client that tracks its own max concurrency."""

    class ConcurrencyTrackingClient:
        def __init__(self) -> None:
            self.lock = threading.Lock()
            self.current = 0
            self.max_seen = 0
            self.calls = 0
            self.messages = self

        def create(self, **kwargs):
            with self.lock:
                self.current += 1
                self.max_seen = max(self.max_seen, self.current)
                self.calls += 1
            time.sleep(0.02)  # long enough that overlap is measurable
            try:
                return _text_response("2")
            finally:
                with self.lock:
                    self.current -= 1

    cfg = replace(CONFIG, SEMAPHORE=3)
    client = ConcurrencyTrackingClient()
    docs = [Doc(f"d{i:02d}", f"text{i}") for i in range(20)]
    LLMReranker(cfg, ResponseCache(tmp_path), client=client).rerank(Query("q1", "q"), docs)
    assert client.calls == 20
    assert client.max_seen <= cfg.SEMAPHORE
    assert client.max_seen > 1, "never actually ran concurrently"


class TextKeyedGradingClient:
    """Grades keyed by exact document text rather than call order.

    A call-order-keyed fake (like FakeAnthropic) is unsafe for comparing
    concurrent vs. sequential runs: concurrent calls complete in a different
    order than they were submitted, so a reply list indexed by call count
    would scramble which grade lands on which doc - a fake-client artefact,
    not a production bug. Keying by content sidesteps that entirely.
    """

    def __init__(self, grades_by_text: dict[str, str]) -> None:
        self.grades_by_text = grades_by_text
        self.calls = 0
        self.messages = self

    def create(self, **kwargs):
        self.calls += 1
        prompt = kwargs["messages"][0]["content"]
        for text, grade in self.grades_by_text.items():
            if text in prompt:
                return _text_response(grade)
        raise AssertionError(f"no grade configured for prompt: {prompt!r}")


def test_concurrent_output_matches_sequential_result(tmp_path) -> None:
    """Concurrency must not change the output: same inputs, same
    sorted-by-(-score, doc_id) order, whether SEMAPHORE forces effectively
    sequential execution or lets many calls overlap."""
    query = Query("q1", "q")
    # Zero-padded so no doc's text is a substring of another's (e.g. "text-1"
    # would otherwise match inside "text-10" and "text-11").
    docs = [Doc(f"d{i:02d}", f"unique-text-{i:02d}") for i in range(12)]
    grades_by_text = {doc.text: str(i % 4) for i, doc in enumerate(docs)}

    seq_cfg = replace(CONFIG, SEMAPHORE=1)
    conc_cfg = replace(CONFIG, SEMAPHORE=8)

    seq_out = LLMReranker(
        seq_cfg, ResponseCache(tmp_path / "seq"), client=TextKeyedGradingClient(grades_by_text)
    ).rerank(query, docs)
    conc_out = LLMReranker(
        conc_cfg, ResponseCache(tmp_path / "conc"), client=TextKeyedGradingClient(grades_by_text)
    ).rerank(query, docs)

    assert [(s.doc_id, s.score, s.p_relevant) for s in seq_out] == [
        (s.doc_id, s.score, s.p_relevant) for s in conc_out
    ]
    # and matches the expected (-score, doc_id) order computed independently
    expected_order = sorted(
        ((int(grades_by_text[d.text]), d.doc_id) for d in docs), key=lambda t: (-t[0], t[1])
    )
    assert [s.doc_id for s in conc_out] == [doc_id for _, doc_id in expected_order]


def test_one_failure_does_not_abort_siblings(tmp_path) -> None:
    """One raising call must not abort the others — every sibling still gets
    its recorded-and-excluded row, or its correct grade if it succeeded."""

    class SelectivelyBrokenClient:
        def __init__(self) -> None:
            self.calls = 0
            self.messages = self

        def create(self, **kwargs):
            self.calls += 1
            prompt = kwargs["messages"][0]["content"]
            if "FAIL_MARKER" in prompt:
                raise RuntimeError("intermittent 503")
            return _text_response("2")

    docs = [Doc(f"d{i:02d}", "FAIL_MARKER" if i == 3 else f"text{i}") for i in range(8)]
    cfg = replace(CONFIG, SEMAPHORE=4)
    out = LLMReranker(cfg, ResponseCache(tmp_path), client=SelectivelyBrokenClient()).rerank(
        QUERY, docs
    )
    by_id = {s.doc_id: s for s in out}
    assert len(by_id) == 8

    failed = by_id["d03"]
    assert failed.meta.get("excluded") is True
    assert "error" in failed.meta
    assert failed.p_relevant == 0.0

    for i in range(8):
        if i == 3:
            continue
        ok = by_id[f"d{i:02d}"]
        assert ok.meta.get("excluded") is not True
        assert ok.p_relevant == pytest.approx(2 / 3)
