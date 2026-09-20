"""The four chunk-selection policies.

Turns a ranked list of `Scored` candidates into a `Selection`: the doc_ids to
keep, plus an `abstained` flag meaning "do not answer this query at all".

Policies 2-4 only work if p_relevant transfers across queries, which is
exactly what H2 tests — an uncalibrated arm should visibly degrade here.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from src.config import Config
from src.types import Scored

POLICIES: tuple[str, ...] = ("fixed", "threshold", "mass", "confidence_gated")


@dataclass(frozen=True)
class Selection:
    doc_ids: list[str]
    abstained: bool
    policy: str


def _usable(scored: Sequence[Scored]) -> list[Scored]:
    """Drop candidates whose API call failed. They are excluded from metrics
    and must never be forwarded as context."""
    return [s for s in scored if not s.meta.get("excluded")]


def _by_score(items: list[Scored]) -> list[Scored]:
    """Arm-native rank order, used by every policy.

    A policy decides *how many* chunks to keep, never what order they come in
    (BRD §6). Re-sorting policies 2-4 by `p_relevant` would change nDCG for
    reasons unrelated to selection, which would confound H1 with H3 — and the
    two orders differ only for an arm whose score and p_relevant are not
    monotone in each other, which is arm D exactly.

    Ties broken by doc_id, never by input/dict order.
    """
    return sorted(items, key=lambda s: (-s.score, s.doc_id))


def select(scored: Sequence[Scored], policy: str, cfg: Config) -> Selection:
    """Turn a ranked candidate list into a kept set, or an abstention.

    - "fixed": keeps the top `cfg.FIXED_K` candidates by the arm's own score.
      This is the baseline every pipeline already ships without any
      calibration awareness. It never abstains — there is no probability
      signal in play that could trigger one.
    - "threshold": keeps every candidate with `p_relevant >= cfg.TAU`.
      Abstains (empty selection, `abstained=True`) when nothing clears the
      bar: the calibrated read is "nothing here is relevant enough".
    - "mass": walks candidates in the arm's own rank order and keeps the
      smallest prefix whose *raw* cumulative `p_relevant` reaches
      `cfg.MASS_TARGET` (no renormalisation — a candidate's own claimed
      probability is what has to add up). A sharp probability distribution
      needs fewer chunks than a flat one; that is the token saving H3
      claims. Abstains only when there are no usable candidates at all.
    - "confidence_gated": "threshold" with a bar that moves on the arm's own
      confidence, per BRD §6. Low confidence (`< cfg.C_LOW`) drops the bar to
      `cfg.TAU_WIDE` — the arm is unsure, so buy more context rather than
      narrowing onto a possibly-wrong top pick. Confidence at or above
      `cfg.C_LOW` *and* a top `p_relevant` at or above `cfg.TAU` raises it to
      `cfg.TAU_NARROW`, which is where the token saving in H3 comes from.
      Anything else uses plain `cfg.TAU`. All three bars are named constants in
      `config.py`, so there is no tuned value hiding in this module.
      `confidence=None` (arms with no confidence signal at all) is never
      treated as high confidence — it gets plain `cfg.TAU`, never the narrowed
      bar. Abstains when nothing clears whichever bar applies.

    Excluded candidates (`meta["excluded"]`) are dropped before any policy
    runs and can never appear in a selection.
    """
    usable = _usable(scored)

    if policy == "fixed":
        ranked = _by_score(usable)
        return Selection([s.doc_id for s in ranked[: cfg.FIXED_K]], False, policy)

    if policy == "threshold":
        ranked = _by_score(usable)
        kept = [s.doc_id for s in ranked if s.p_relevant >= cfg.TAU]
        return Selection(kept, not kept, policy)

    if policy == "mass":
        ranked = _by_score(usable)
        kept: list[str] = []
        running = 0.0
        for s in ranked:
            kept.append(s.doc_id)
            running += s.p_relevant
            if running >= cfg.MASS_TARGET:
                break
        return Selection(kept, not kept, policy)

    if policy == "confidence_gated":
        ranked = _by_score(usable)
        top = ranked[0] if ranked else None
        # confidence=None means "this arm has no confidence signal" — NOT
        # "certain". Coercing it to 1.0 would give arms A/B/E free narrowing
        # here and flatter them against the arm under test.
        tau = cfg.TAU
        if top is not None and top.confidence is not None:
            if top.confidence < cfg.C_LOW:
                tau = cfg.TAU_WIDE
            elif top.p_relevant >= cfg.TAU:
                tau = cfg.TAU_NARROW
        kept = [s.doc_id for s in ranked if s.p_relevant >= tau]
        return Selection(kept, not kept, policy)

    raise ValueError(f"unknown policy: {policy!r}; expected one of {POLICIES}")
