import numpy as np
import pytest

from src.metrics import ece
from src.rerankers import (
    CalibratorNotFittedError,
    CosineReranker,
    PlattReranker,
    minmax,
)
from src.types import Doc, Query

QUERY = Query("q1", "a query")
DOCS = [Doc("d1", "one"), Doc("d2", "two"), Doc("d3", "three")]
SIMS = {"q1": {"d1": 0.9, "d2": 0.5, "d3": 0.1}}


def test_minmax_maps_to_unit_interval() -> None:
    out = minmax(np.array([0.1, 0.5, 0.9]))
    assert out.min() == pytest.approx(0.0) and out.max() == pytest.approx(1.0)


def test_minmax_of_constant_array_is_one_half() -> None:
    """No spread means no information; 0.5 is the honest answer, and it avoids
    a divide-by-zero that would produce NaN probabilities."""
    assert np.allclose(minmax(np.array([0.4, 0.4, 0.4])), 0.5)


def test_cosine_arm_preserves_similarity_order() -> None:
    out = CosineReranker(SIMS).rerank(QUERY, DOCS)
    assert [s.doc_id for s in out] == ["d1", "d2", "d3"]


def test_cosine_arm_p_relevant_is_in_range() -> None:
    assert all(0.0 <= s.p_relevant <= 1.0 for s in CosineReranker(SIMS).rerank(QUERY, DOCS))


def test_cosine_arm_has_no_confidence() -> None:
    """Arm A has no notion of confidence. It must report None so policy 4
    cannot mistake silence for certainty."""
    assert all(s.confidence is None for s in CosineReranker(SIMS).rerank(QUERY, DOCS))


def test_platt_requires_fitting_before_use() -> None:
    with pytest.raises(CalibratorNotFittedError):
        PlattReranker(CosineReranker(SIMS)).rerank(QUERY, DOCS)


def test_platt_preserves_the_ranking_of_its_base_arm() -> None:
    """Platt is monotone, so arm E must rank exactly like arm A. Any difference
    between them in the results is calibration, never ordering."""
    arm = PlattReranker(CosineReranker(SIMS))
    rng = np.random.default_rng(0)
    s = rng.uniform(size=400)
    arm.fit(s, (rng.uniform(size=400) < s).astype(int))
    assert [x.doc_id for x in arm.rerank(QUERY, DOCS)] == ["d1", "d2", "d3"]


def test_platt_improves_calibration_of_a_skewed_signal() -> None:
    """The whole point of arm E: if 20 lines of logistic regression fixes
    cosine's calibration, H2 needs to beat that, not just beat raw cosine."""
    rng = np.random.default_rng(1)
    raw = rng.uniform(size=4000)
    labels = (rng.uniform(size=4000) < raw**3).astype(int)  # true P is raw**3
    arm = PlattReranker(CosineReranker(SIMS))
    arm.fit(raw, labels)
    assert ece(arm.transform(raw), labels) < ece(raw, labels)


def test_platt_refuses_a_calibrator_fitted_on_test() -> None:
    """Guard test for non-negotiable #1."""
    arm = PlattReranker(CosineReranker(SIMS))
    rng = np.random.default_rng(2)
    s = rng.uniform(size=100)
    with pytest.raises(ValueError, match="dev"):
        arm.fit(s, (s > 0.5).astype(int), split="test")
