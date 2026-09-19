import dataclasses

import pytest

from src.config import CONFIG
from src.types import Scored


def test_config_is_frozen() -> None:
    """Tuned constants must not be mutated at runtime — a silently changed
    TAU mid-run would make results unreproducible."""
    with pytest.raises(dataclasses.FrozenInstanceError):
        CONFIG.TAU = 0.9  # type: ignore[misc]


def test_config_defaults_match_spec() -> None:
    assert CONFIG.POOL_SIZE == 50
    assert CONFIG.SEED == 42
    assert CONFIG.SEMAPHORE == 16
    assert CONFIG.PROMPT_VERSION == "v1"
    assert pytest.approx(1.0) == CONFIG.W_TOPICAL + CONFIG.W_ANSWERS


def test_scored_rejects_out_of_range_probability() -> None:
    """p_relevant is a probability claim the calibration metrics hold arms to.
    A value outside [0,1] means an arm is broken; fail loudly, not silently."""
    with pytest.raises(ValueError, match="p_relevant"):
        Scored(doc_id="d1", score=1.0, p_relevant=1.3)


def test_scored_allows_none_confidence() -> None:
    """Arms A, B and E have no confidence notion. None must be representable —
    coercing it to 1.0 would hand them an undeserved win under policy 4."""
    s = Scored(doc_id="d1", score=0.5, p_relevant=0.5)
    assert s.confidence is None
