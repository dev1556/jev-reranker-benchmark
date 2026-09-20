import numpy as np
import pytest

from src.config import CONFIG
from src.rerankers import CrossEncoderReranker
from src.types import Doc, Query

QUERY = Query("q1", "what causes rain")
DOCS = [Doc("d1", "irrelevant"), Doc("d2", "rain forms when vapour condenses")]


class FakeCrossEncoder:
    """Stands in for bge-reranker-base, returning raw logits.

    The real model only returns logits because we construct it with
    `activation_fn=Identity`; its default `predict()` applies Sigmoid. See
    `test_real_model_is_built_without_the_default_sigmoid`.
    """

    def __init__(self, logits: list[float]) -> None:
        self.logits = logits

    def predict(self, pairs: list[list[str]]) -> np.ndarray:
        return np.array(self.logits[: len(pairs)])


def test_orders_by_logit_descending() -> None:
    arm = CrossEncoderReranker(CONFIG, model=FakeCrossEncoder([-2.0, 3.0]))
    assert [s.doc_id for s in arm.rerank(QUERY, DOCS)] == ["d2", "d1"]


def test_p_relevant_is_sigmoid_of_logit() -> None:
    arm = CrossEncoderReranker(CONFIG, model=FakeCrossEncoder([0.0, 2.0]))
    out = {s.doc_id: s.p_relevant for s in arm.rerank(QUERY, DOCS)}
    assert out["d1"] == pytest.approx(0.5)
    assert out["d2"] == pytest.approx(1 / (1 + np.exp(-2.0)))


def test_extreme_logits_stay_in_range() -> None:
    """Overflow in a naive sigmoid would produce p_relevant outside [0,1] and
    Scored would reject it - clamp rather than crash mid-run."""
    arm = CrossEncoderReranker(CONFIG, model=FakeCrossEncoder([-900.0, 900.0]))
    assert all(0.0 <= s.p_relevant <= 1.0 for s in arm.rerank(QUERY, DOCS))


def test_has_no_confidence() -> None:
    arm = CrossEncoderReranker(CONFIG, model=FakeCrossEncoder([1.0, 2.0]))
    assert all(s.confidence is None for s in arm.rerank(QUERY, DOCS))


def test_pairs_are_query_then_document() -> None:
    """bge expects [query, passage]. Reversed pairs silently degrade the
    strongest rival arm, which would quietly flatter arm D."""
    seen: list[list[str]] = []

    class Spy(FakeCrossEncoder):
        def predict(self, pairs):
            seen.extend(pairs)
            return super().predict(pairs)

    CrossEncoderReranker(CONFIG, model=Spy([1.0, 2.0])).rerank(QUERY, DOCS)
    assert seen[0][0] == QUERY.text and seen[0][1].startswith("irrelevant")


def test_real_model_is_built_without_the_default_sigmoid(monkeypatch) -> None:
    """Regression guard for a bug that would have shipped: bge's predict()
    applies Sigmoid by default, so without activation_fn=Identity rerank()
    sigmoids an already-squashed probability and arm B's p_relevant comes out
    wrong (a logit of 2.85 -> 0.72 instead of 0.945). Arm B is the strongest
    rival; silently miscalibrating it would flatter the arm under test.

    Stubs the import so the test never downloads 2GB of weights.
    """
    import sys
    import types

    captured: dict[str, object] = {}

    class StubIdentity:
        pass

    def stub_cross_encoder(name: str, **kwargs: object) -> FakeCrossEncoder:
        captured.update(kwargs, name=name)
        return FakeCrossEncoder([1.0])

    st = types.ModuleType("sentence_transformers")
    st.CrossEncoder = stub_cross_encoder
    torch_stub = types.ModuleType("torch")
    torch_stub.nn = types.SimpleNamespace(Identity=StubIdentity)
    monkeypatch.setitem(sys.modules, "sentence_transformers", st)
    monkeypatch.setitem(sys.modules, "torch", torch_stub)

    CrossEncoderReranker(CONFIG)

    assert captured["name"] == CONFIG.CROSS_ENCODER
    assert isinstance(captured["activation_fn"], StubIdentity)
