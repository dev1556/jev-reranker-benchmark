import numpy as np
import pytest

from src.metrics import auroc, brier_decomposition, ece, reliability_bins


def test_perfectly_calibrated_scores_near_zero_ece() -> None:
    """The anchor test for H2. A generator whose stated probability equals the
    observed frequency must score ECE ~ 0; sampling noise is the only error."""
    rng = np.random.default_rng(42)
    probs = rng.uniform(0, 1, size=20_000)
    labels = (rng.uniform(0, 1, size=20_000) < probs).astype(int)
    assert ece(probs, labels, n_bins=15) < 0.02


def test_maximally_miscalibrated_scores_near_one() -> None:
    """Confidently wrong every time -> ECE ~ 1.0."""
    probs = np.ones(1000)
    labels = np.zeros(1000, dtype=int)
    assert ece(probs, labels, n_bins=15) == pytest.approx(1.0, abs=1e-6)


def test_ece_hand_computed_two_bins() -> None:
    # 4 items. probs .1 .1 .9 .9 ; labels 0 0 1 0
    # equal-mass split -> bin A {.1,.1} mean_pred .1, frac_pos 0.0, |diff| .1
    #                     bin B {.9,.9} mean_pred .9, frac_pos 0.5, |diff| .4
    # ECE = 0.5*0.1 + 0.5*0.4 = 0.25
    probs = np.array([0.1, 0.1, 0.9, 0.9])
    labels = np.array([0, 0, 1, 0])
    assert ece(probs, labels, n_bins=2) == pytest.approx(0.25)


def test_reliability_bins_counts_sum_to_n() -> None:
    rng = np.random.default_rng(0)
    probs, labels = rng.uniform(size=500), rng.integers(0, 2, size=500)
    assert sum(b.count for b in reliability_bins(probs, labels, n_bins=15)) == 500


def test_brier_matches_mean_squared_error() -> None:
    probs = np.array([0.2, 0.8, 0.6])
    labels = np.array([0, 1, 1])
    expected = np.mean((probs - labels) ** 2)
    assert brier_decomposition(probs, labels).brier == pytest.approx(expected)


def test_brier_decomposition_identity_holds() -> None:
    """Murphy (1973): brier ~= reliability - resolution + uncertainty."""
    rng = np.random.default_rng(7)
    probs = rng.uniform(size=5000)
    labels = (rng.uniform(size=5000) < probs).astype(int)
    d = brier_decomposition(probs, labels, n_bins=15)
    assert d.reliability - d.resolution + d.uncertainty == pytest.approx(d.brier, abs=1e-3)


def test_auroc_perfect_separation_is_one() -> None:
    assert auroc(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1])) == pytest.approx(1.0)


def test_auroc_random_is_half() -> None:
    # positives {0.2, 0.8} vs negatives {0.1, 0.9}: pairs 0.2>0.1 OK, 0.2<0.9 no,
    # 0.8>0.1 OK, 0.8<0.9 no -> 2/4 = 0.5
    assert auroc(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 1, 1, 0])) == pytest.approx(0.5)


def test_auroc_partial_separation() -> None:
    # positives {0.2, 0.9} vs negatives {0.1, 0.8}: pairs 0.2>0.1 OK, 0.2<0.8 no,
    # 0.9>0.1 OK, 0.9>0.8 OK -> 3/4
    assert auroc(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 1, 0, 1])) == pytest.approx(0.75)


def test_auroc_single_class_returns_nan() -> None:
    """Undefined, and must say so rather than return a misleading 0.5."""
    assert np.isnan(auroc(np.array([0.1, 0.9]), np.array([1, 1])))


def test_auroc_matches_sklearn_including_ties() -> None:
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(3)
    scores = rng.integers(0, 5, size=300).astype(float)  # heavy ties on purpose
    labels = rng.integers(0, 2, size=300)
    assert auroc(scores, labels) == pytest.approx(roc_auc_score(labels, scores))
