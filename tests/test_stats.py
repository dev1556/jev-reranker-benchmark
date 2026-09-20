import numpy as np
import pytest

from src.stats import bootstrap_ci, holm, paired_randomisation_test


def test_bootstrap_ci_brackets_the_mean() -> None:
    rng = np.random.default_rng(1)
    values = rng.normal(0.5, 0.1, size=500)
    lo, hi = bootstrap_ci(values, n=2000, seed=42)
    assert lo < values.mean() < hi


def test_bootstrap_ci_is_seed_reproducible() -> None:
    """NFR-5: warm cache + seed must give byte-identical output."""
    v = np.linspace(0, 1, 100)
    assert bootstrap_ci(v, n=1000, seed=42) == bootstrap_ci(v, n=1000, seed=42)


def test_bootstrap_ci_narrows_with_more_data() -> None:
    rng = np.random.default_rng(2)
    small = bootstrap_ci(rng.normal(0, 1, 50), n=2000, seed=42)
    large = bootstrap_ci(rng.normal(0, 1, 5000), n=2000, seed=42)
    assert (large[1] - large[0]) < (small[1] - small[0])


def test_bootstrap_ci_of_constant_is_degenerate() -> None:
    lo, hi = bootstrap_ci(np.full(100, 0.7), n=500, seed=42)
    assert lo == pytest.approx(0.7) and hi == pytest.approx(0.7)


def test_randomisation_identical_arms_is_not_significant() -> None:
    v = np.random.default_rng(4).uniform(size=200)
    assert paired_randomisation_test(v, v.copy(), n=2000, seed=42) == pytest.approx(1.0)


def test_randomisation_large_consistent_difference_is_significant() -> None:
    a = np.random.default_rng(5).uniform(size=200)
    assert paired_randomisation_test(a + 0.3, a, n=2000, seed=42) < 0.01


def test_randomisation_is_paired_not_unpaired() -> None:
    """A tiny but perfectly consistent per-query gain must be detected even when
    the between-query variance dwarfs it. This is why the test is paired."""
    rng = np.random.default_rng(6)
    base = rng.uniform(0, 1, size=300)  # large spread across queries
    assert paired_randomisation_test(base + 0.02, base, n=5000, seed=42) < 0.01


def test_holm_leaves_a_single_pvalue_unchanged() -> None:
    assert holm({"a": 0.03}) == pytest.approx({"a": 0.03})


def test_holm_hand_computed() -> None:
    # sorted .01 .02 .03 with m=3 -> .01*3=.03 ; .02*2=.04 ; .03*1=.03 -> monotone -> .04
    out = holm({"a": 0.01, "b": 0.02, "c": 0.03})
    assert out["a"] == pytest.approx(0.03)
    assert out["b"] == pytest.approx(0.04)
    assert out["c"] == pytest.approx(0.04)


def test_holm_never_exceeds_one() -> None:
    assert all(v <= 1.0 for v in holm({"a": 0.6, "b": 0.7, "c": 0.9}).values())
