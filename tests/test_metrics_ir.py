import pytest

from src.metrics import mrr_at_k, ndcg_at_k, precision_at_k, recall_at_k

# Ranking: d1(rel=1) d2(rel=0) d3(rel=1) d4(rel=0) d5(rel=0); one more relevant (d9) missed.
RANKED = ["d1", "d2", "d3", "d4", "d5"]
REL = {"d1": 1, "d3": 1, "d9": 1}


def test_ndcg_at_5_hand_computed() -> None:
    # DCG  = 1/log2(2) + 1/log2(4)           = 1.0 + 0.5           = 1.5
    # IDCG = 1/log2(2) + 1/log2(3) + 1/log2(4) = 1.0 + 0.63093 + 0.5 = 2.13093
    # nDCG = 1.5 / 2.13093 = 0.70392
    assert ndcg_at_k(RANKED, REL, 5) == pytest.approx(0.70392, abs=1e-5)


def test_ndcg_perfect_ranking_is_one() -> None:
    assert ndcg_at_k(["d1", "d3", "d9"], REL, 3) == pytest.approx(1.0)


def test_ndcg_with_no_relevant_docs_is_zero() -> None:
    """Not NaN — an unanswerable query must contribute 0, not poison the mean."""
    assert ndcg_at_k(RANKED, {}, 5) == 0.0


def test_recall_at_5() -> None:
    assert recall_at_k(RANKED, REL, 5) == pytest.approx(2 / 3)


def test_precision_at_5() -> None:
    assert precision_at_k(RANKED, REL, 5) == pytest.approx(2 / 5)


def test_precision_at_k_divides_by_k_not_by_list_length() -> None:
    """Truncating at k=10 with only 5 results must still divide by 10."""
    assert precision_at_k(RANKED, REL, 10) == pytest.approx(2 / 10)


def test_mrr_first_relevant_at_rank_1() -> None:
    assert mrr_at_k(RANKED, REL, 10) == pytest.approx(1.0)


def test_mrr_first_relevant_at_rank_3() -> None:
    assert mrr_at_k(["x", "y", "d1"], REL, 10) == pytest.approx(1 / 3)


def test_mrr_no_hit_within_k_is_zero() -> None:
    assert mrr_at_k(["x", "y", "d1"], REL, 2) == 0.0


def test_graded_relevance_uses_gain() -> None:
    # grade 2 -> gain 2**2-1 = 3 at rank 1; IDCG identical -> 1.0
    assert ndcg_at_k(["a"], {"a": 2}, 1) == pytest.approx(1.0)
