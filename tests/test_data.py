import pytest

from src.config import CONFIG
from src.data import (
    TestSplitAccessError,
    get_split,
    sample_query_ids,
    split_query_ids,
    tuning_context,
)
from src.types import Corpus, Doc, Query


def _corpus(n: int = 100) -> Corpus:
    return Corpus(
        name="fake",
        docs={f"d{i}": Doc(f"d{i}", f"text {i}") for i in range(n)},
        queries={f"q{i}": Query(f"q{i}", f"query {i}") for i in range(n)},
        qrels={f"q{i}": {f"d{i}": 1} for i in range(n)},
    )


def test_sampling_is_seed_reproducible() -> None:
    ids = [f"q{i}" for i in range(500)]
    assert sample_query_ids(ids, 300, seed=42) == sample_query_ids(ids, 300, seed=42)


def test_sampling_is_order_independent() -> None:
    """Sorting first means a reshuffled input corpus cannot change the sample."""
    ids = [f"q{i}" for i in range(500)]
    assert sample_query_ids(ids, 50, 42) == sample_query_ids(list(reversed(ids)), 50, 42)


def test_sampling_more_than_available_returns_all() -> None:
    assert len(sample_query_ids([f"q{i}" for i in range(10)], 50, 42)) == 10


def test_split_is_disjoint_and_exhaustive() -> None:
    dev, test = split_query_ids([f"q{i}" for i in range(100)], seed=42)
    assert not set(dev) & set(test)
    assert len(dev) + len(test) == 100


def test_split_is_seed_reproducible() -> None:
    ids = [f"q{i}" for i in range(100)]
    assert split_query_ids(ids, 42) == split_query_ids(ids, 42)


def test_dev_split_is_freely_accessible() -> None:
    assert len(get_split(_corpus(), "dev", CONFIG)) > 0


def test_test_split_access_inside_tuning_context_raises() -> None:
    """The single most important guard in the repo. Non-negotiable #1 says the
    test split is never used for tuning; this makes the violation impossible
    rather than merely discouraged."""
    with pytest.raises(TestSplitAccessError, match="tuning"), tuning_context():
        get_split(_corpus(), "test", CONFIG)


def test_test_split_access_outside_tuning_context_is_allowed() -> None:
    assert len(get_split(_corpus(), "test", CONFIG)) > 0


def test_tuning_context_resets_on_exception() -> None:
    """A raised error inside tuning must not leave the guard latched on."""
    with pytest.raises(RuntimeError), tuning_context():
        raise RuntimeError("boom")
    assert len(get_split(_corpus(), "test", CONFIG)) > 0
