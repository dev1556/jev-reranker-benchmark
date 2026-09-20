import pytest

from src.config import CONFIG
from src.gating import POLICIES, select
from src.types import Scored


def _scored(probs: list[float], conf: float | None = None) -> list[Scored]:
    return [Scored(f"d{i}", score=p, p_relevant=p, confidence=conf) for i, p in enumerate(probs)]


def test_fixed_policy_takes_exactly_k() -> None:
    assert len(select(_scored([0.9] * 20), "fixed", CONFIG).doc_ids) == CONFIG.FIXED_K


def test_fixed_policy_never_abstains() -> None:
    """The baseline everyone ships has no abstention concept — that's the point."""
    assert select(_scored([0.01] * 20), "fixed", CONFIG).abstained is False


def test_threshold_policy_keeps_only_above_tau() -> None:
    cfg = CONFIG.tuned(TAU=0.5)
    assert select(_scored([0.9, 0.8, 0.4, 0.1]), "threshold", cfg).doc_ids == ["d0", "d1"]


def test_threshold_policy_abstains_when_nothing_qualifies() -> None:
    cfg = CONFIG.tuned(TAU=0.5)
    out = select(_scored([0.1, 0.05]), "threshold", cfg)
    assert out.doc_ids == [] and out.abstained is True


def test_mass_policy_stops_once_target_reached() -> None:
    cfg = CONFIG.tuned(MASS_TARGET=0.8)
    assert select(_scored([0.5, 0.3, 0.2, 0.1]), "mass", cfg).doc_ids == ["d0", "d1"]


def test_mass_policy_adapts_to_distribution_sharpness() -> None:
    """A sharp distribution should yield fewer chunks than a flat one — this is
    the token saving H3 claims."""
    cfg = CONFIG.tuned(MASS_TARGET=0.8)
    sharp = select(_scored([0.95, 0.02, 0.02, 0.01]), "mass", cfg)
    flat = select(_scored([0.25, 0.25, 0.25, 0.25]), "mass", cfg)
    assert len(sharp.doc_ids) < len(flat.doc_ids)


def test_confidence_gated_widens_when_confidence_is_low() -> None:
    cfg = CONFIG.tuned(TAU=0.5, C_LOW=0.5)
    low = select(_scored([0.9, 0.8, 0.45, 0.4], conf=0.2), "confidence_gated", cfg)
    high = select(_scored([0.9, 0.8, 0.45, 0.4], conf=0.95), "confidence_gated", cfg)
    assert len(low.doc_ids) > len(high.doc_ids)


def test_confidence_gated_abstains_when_nothing_clears_and_confidence_is_low() -> None:
    cfg = CONFIG.tuned(TAU=0.5, C_LOW=0.5)
    assert select(_scored([0.1, 0.05], conf=0.2), "confidence_gated", cfg).abstained is True


def test_confidence_gated_with_none_confidence_does_not_assume_certainty() -> None:
    """Arms A, B, E report confidence=None. Treating None as 1.0 would hand them
    the narrowing behaviour for free and flatter them against arm D."""
    cfg = CONFIG.tuned(TAU=0.5, C_LOW=0.5)
    none_conf = select(_scored([0.9, 0.8, 0.45], conf=None), "confidence_gated", cfg)
    assert none_conf.doc_ids == ["d0", "d1"]  # plain threshold behaviour, no widening


def test_confidence_gated_narrows_when_confident_and_top_p_is_high() -> None:
    """BRD §6's second half: confident *and* a high top probability raises the
    bar to TAU_NARROW, which is where H3's token saving comes from."""
    cfg = CONFIG.tuned(TAU=0.5, C_LOW=0.5, TAU_WIDE=0.3, TAU_NARROW=0.7)
    confident = select(_scored([0.9, 0.6, 0.55], conf=0.95), "confidence_gated", cfg)
    assert confident.doc_ids == ["d0"]  # 0.6 and 0.55 clear TAU but not TAU_NARROW


def test_confidence_gated_does_not_narrow_when_top_p_is_low() -> None:
    """Confident about a weak top pick is not grounds for narrowing — both
    halves of the BRD §6 condition have to hold."""
    cfg = CONFIG.tuned(TAU=0.5, C_LOW=0.5, TAU_WIDE=0.3, TAU_NARROW=0.7)
    assert select(_scored([0.45, 0.4], conf=0.95), "confidence_gated", cfg).abstained is True


def test_gated_bars_are_ordered() -> None:
    """TAU_WIDE < TAU < TAU_NARROW. If a dev tuning pass ever breaks this
    ordering, 'widen' and 'narrow' would swap meanings silently."""
    assert CONFIG.TAU_WIDE < CONFIG.TAU < CONFIG.TAU_NARROW


def test_excluded_candidates_are_never_selected() -> None:
    """An errored API call must not be silently forwarded as context."""
    items = [
        Scored("bad", float("-inf"), 0.0, None, {"excluded": True}),
        Scored("good", 0.9, 0.9, None),
    ]
    assert "bad" not in select(items, "fixed", CONFIG).doc_ids


def test_all_policies_are_registered() -> None:
    assert set(POLICIES) == {"fixed", "threshold", "mass", "confidence_gated"}


def test_unknown_policy_raises() -> None:
    with pytest.raises(ValueError, match="unknown policy"):
        select(_scored([0.5]), "telepathy", CONFIG)
