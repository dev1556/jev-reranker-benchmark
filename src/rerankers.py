"""The five rerankers, behind one protocol.

score orders the ranking; p_relevant is a probability claim the calibration
metrics hold the arm to. They are separate fields on purpose - see types.py.
"""

import asyncio
import os
import re
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol

import numpy as np

from src.cache import ResponseCache, cache_key
from src.config import Config
from src.prompts import JEV_QUESTIONS_V1, JEV_STATE, LLM_RERANK_PROMPT
from src.types import Doc, Query, Scored


class Reranker(Protocol):
    name: str

    def rerank(self, query: Query, candidates: Sequence[Doc]) -> list[Scored]: ...


class CalibratorNotFittedError(RuntimeError):
    """Raised when arm E is used before `.fit()` on the dev split."""


def minmax(values: np.ndarray) -> np.ndarray:
    """Per-query min-max scaling to [0, 1].

    Constant input maps to 0.5: no spread means no information, and it avoids
    a divide-by-zero that would otherwise produce NaN probabilities.
    """
    values = np.asarray(values, dtype=float)
    lo, hi = values.min(), values.max()
    if hi - lo < 1e-12:
        return np.full_like(values, 0.5)
    return (values - lo) / (hi - lo)


class CosineReranker:
    """Arm A. Identity reorder of the pool; p_relevant is per-query min-max.

    Deliberately naive. This arm exists to be the floor, and quantifying how
    fictional its p_relevant is constitutes hypothesis H2.
    """

    name = "cosine"

    def __init__(self, sims: dict[str, dict[str, float]]) -> None:
        self.sims = sims

    def rerank(self, query: Query, candidates: Sequence[Doc]) -> list[Scored]:
        """score = raw cosine similarity; p_relevant = that similarity min-maxed
        to [0, 1] within this query's pool (not a calibrated probability)."""
        raw = np.array([self.sims[query.query_id][d.doc_id] for d in candidates])
        probs = minmax(raw)
        scored = [
            Scored(doc_id=d.doc_id, score=float(r), p_relevant=float(p), confidence=None)
            for d, r, p in zip(candidates, raw, probs, strict=True)
        ]
        return sorted(scored, key=lambda s: (-s.score, s.doc_id))


class PlattReranker:
    """Arm E. Logistic recalibration of arm A, fitted on dev and applied frozen.

    Exists to pre-empt the obvious rebuttal to any calibration win by arm D:
    that a trivial recalibration of cosine would have done the same.
    """

    name = "platt"

    def __init__(self, base: CosineReranker) -> None:
        self.base = base
        self._model = None

    def fit(self, scores: np.ndarray, labels: np.ndarray, split: str = "dev") -> None:
        """Fit the logistic (Platt) calibrator on dev-split (score, label) pairs.

        Refuses any split other than 'dev' - CLAUDE.md non-negotiable #1: never
        tune on test.
        """
        if split != "dev":
            raise ValueError(
                f"calibrator must be fitted on dev, got split={split!r} "
                f"(CLAUDE.md non-negotiable #1)"
            )
        from sklearn.linear_model import LogisticRegression

        self._model = LogisticRegression().fit(
            np.asarray(scores, dtype=float).reshape(-1, 1), np.asarray(labels, dtype=int)
        )

    def transform(self, scores: np.ndarray) -> np.ndarray:
        """Map raw cosine scores to calibrated P(relevant) via the fitted model."""
        if self._model is None:
            raise CalibratorNotFittedError("PlattReranker.fit() must run on dev first")
        return self._model.predict_proba(np.asarray(scores, dtype=float).reshape(-1, 1))[:, 1]

    def rerank(self, query: Query, candidates: Sequence[Doc]) -> list[Scored]:
        """Rank exactly like the base cosine arm (Platt is monotone); p_relevant
        is the calibrated probability in place of arm A's raw min-max."""
        base = self.base.rerank(query, candidates)
        probs = self.transform(np.array([s.score for s in base]))
        return [
            Scored(doc_id=s.doc_id, score=s.score, p_relevant=float(p), confidence=None)
            for s, p in zip(base, probs, strict=True)
        ]


class CrossEncoderModel(Protocol):
    """Duck-typed contract for whatever `CrossEncoderReranker.model` holds:
    the real `sentence_transformers.CrossEncoder` in production, a fake in
    tests. Both only need `.predict`."""

    def predict(self, pairs: list[list[str]]) -> np.ndarray: ...


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic, P(relevant) = 1 / (1 + exp(-x)).

    Naive exp(-x) overflows for very negative x and exp(x) overflows for very
    positive x; both branches below only ever exponentiate a non-positive
    number, so this can't produce inf/NaN or a probability outside [0, 1].
    """
    x = np.clip(x, -500, 500)
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


class CrossEncoderReranker:
    """Arm B. BAAI/bge-reranker-base - the strongest honest opponent.

    `model` is injectable so tests and CI never download torch or the model
    weights. Given policy 4 (equal effort for every arm), pair ordering
    matters here specifically: bge is trained on [query, passage] pairs, and
    reversing them would quietly handicap the main rival.
    """

    name = "cross_encoder"

    def __init__(self, cfg: Config, model: CrossEncoderModel | None = None) -> None:
        self.cfg = cfg
        if model is None:
            import torch
            from sentence_transformers import CrossEncoder

            # activation_fn=Identity is load-bearing, not tidiness. bge's
            # CrossEncoder applies Sigmoid inside predict() by default, so
            # leaving it alone would hand this class an already-squashed
            # probability, and rerank() would sigmoid it a second time:
            # a true logit of 2.85 (P=0.945) would be published as 0.72.
            # Verified against the real model on 2026-09-20 —
            # default predict() -> [0.945, 0.00048],
            # activation_fn=Identity -> [2.848, -7.643].
            model = CrossEncoder(
                cfg.CROSS_ENCODER, max_length=512, activation_fn=torch.nn.Identity()
            )
        self.model = model

    def rerank(self, query: Query, candidates: Sequence[Doc]) -> list[Scored]:
        """score = raw cross-encoder logit; p_relevant = sigmoid(logit).

        confidence is always None: bge emits no separate certainty signal
        distinct from the logit itself, and coercing it to 1.0 would hand
        this arm free narrowing under the gating policy it did not earn.
        """
        pairs = [[query.text, d.text] for d in candidates]
        logits = np.asarray(self.model.predict(pairs), dtype=float)
        probs = _sigmoid(logits)
        scored = [
            Scored(
                doc_id=d.doc_id,
                score=float(logit),
                p_relevant=float(np.clip(p, 0.0, 1.0)),
                confidence=None,
                meta={"logit": float(logit)},
            )
            for d, logit, p in zip(candidates, logits, probs, strict=True)
        ]
        return sorted(scored, key=lambda s: (-s.score, s.doc_id))


_GRADE_RE = re.compile(r"\b([0-9]+)\b")


def parse_llm_score(text: str) -> int:
    """Extract the 0-3 grade from a raw LLM reply.

    Raises on anything unparseable or out of range - never defaults. A
    silently-defaulted grade is an invisible wrong number in a public chart
    (CLAUDE.md non-negotiable #3: every reported number must be trustworthy).
    """
    match = _GRADE_RE.search(text)
    if match is None:
        raise ValueError(f"no grade found in LLM reply: {text!r}")
    grade = int(match.group(1))
    if not 0 <= grade <= 3:
        raise ValueError(f"grade {grade} outside 0-3 in reply: {text!r}")
    return grade


class AnthropicMessages(Protocol):
    def create(self, **kwargs: object) -> object: ...


class AnthropicClient(Protocol):
    """Duck-typed contract for whatever `LLMReranker.client` holds: the real
    `anthropic.Anthropic()` in production, a fake in tests."""

    messages: AnthropicMessages


class LLMReranker:
    """Arm C. Pointwise 0-3 grading by cfg.LLM_MODEL (Claude Haiku), one call
    per (query, doc) pair.

    p_relevant = grade / 3. Anthropic exposes no logprobs, so this
    pseudo-probability is exactly the uncalibrated-artefact shape hypothesis
    H2 critiques - it is a rescaled ordinal judgment, not a probability the
    model actually estimated. confidence is None because the arm genuinely
    has no such signal; faking one would misrepresent the comparison.

    Equal-effort rule: this rubric (src/prompts.py) gets the same care as
    arm D's - a lazy rival prompt would make any win by the arm under test
    meaningless.

    Every failure path - API exception, no text block in the reply, or a
    reply truncated at max_tokens - is recorded as data (meta["error"]) and
    excluded (meta["excluded"] = True), never scored 0.0. A silent zero would
    depress this arm's metrics in a way nobody would spot in a CSV.
    """

    name = "llm"

    def __init__(
        self,
        cfg: Config,
        cache: ResponseCache,
        client: AnthropicClient | None = None,
        dataset: str = "",
    ) -> None:
        self.cfg = cfg
        self.cache = cache
        self.dataset = dataset
        if client is None:
            import os

            import anthropic

            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise RuntimeError("ANTHROPIC_API_KEY is not set; arm C needs it")
            client = anthropic.Anthropic()
        self.client = client

    def _grade(self, query: Query, doc: Doc) -> dict:
        """Run (or fetch from cache) one grading call. Returns either
        {"grade": int, ...} on success or {"error": str, ...} on any failure -
        callers must branch on "error" in the result, never assume "grade"."""
        key = cache_key(
            self.name,
            self.cfg.LLM_MODEL,
            self.dataset,
            query.query_id,
            doc.doc_id,
            self.cfg.PROMPT_VERSION,
        )
        cached = self.cache.get(key, arm=self.name)
        if cached is not None:
            return cached
        prompt = LLM_RERANK_PROMPT.format(query=query.text, document=doc.text[:4000])
        started = time.perf_counter()
        try:
            # No `thinking` param: Haiku 4.5 only thinks if explicitly told to,
            # and this call wants a bare digit, not reasoning tokens.
            resp = self.client.messages.create(
                model=self.cfg.LLM_MODEL,
                max_tokens=16,
                messages=[{"role": "user", "content": prompt}],
            )
            if resp.stop_reason == "max_tokens":
                # The grade may be truncated mid-token; treat as unusable even
                # if the partial text happens to look like a valid digit.
                raise ValueError(f"reply truncated: stop_reason={resp.stop_reason!r}")
            # content is a list of typed blocks (TextBlock, ThinkingBlock, ...);
            # content[0] is not guaranteed to be text. A reply with no text
            # block at all raises StopIteration, caught below like any other
            # failure.
            text = next(b.text for b in resp.content if b.type == "text")
            payload = {
                "grade": parse_llm_score(text),
                "latency_s": time.perf_counter() - started,
                "input_tokens": resp.usage.input_tokens,
                "output_tokens": resp.usage.output_tokens,
            }
        except Exception as exc:  # recorded, never silently zeroed
            payload = {
                "error": f"{type(exc).__name__}: {exc}",
                "latency_s": time.perf_counter() - started,
            }
        # Failures are cached too, deliberately. The published reproduce path is
        # `git lfs pull && make report` with no API keys; an uncached failure
        # would make a warm-cache run attempt a live call, and without a key
        # this class raises at construction, so the whole reproduction dies on
        # one historical 503. A cached failure keeps the run offline and
        # byte-identical (NFR-5) and keeps the error itself auditable.
        # Retrying a transient fault is therefore an explicit human action:
        # delete those cache entries and re-run the arm.
        self.cache.put(key, payload, arm=self.name)
        return payload

    def rerank(self, query: Query, candidates: Sequence[Doc]) -> list[Scored]:
        """Grade every candidate concurrently, bounded by cfg.SEMAPHORE.

        `ThreadPoolExecutor.map` preserves input order in its output even
        though the calls run out of order, so `zip(candidates, results)`
        below still pairs each doc with its own grade regardless of which
        call finished first. `_grade` never raises - every failure path is
        caught inside it and returned as an {"error": ...} payload - so one
        slow or failing call cannot abort its siblings or the pool.
        """
        with ThreadPoolExecutor(max_workers=self.cfg.SEMAPHORE) as executor:
            results = list(executor.map(lambda doc: self._grade(query, doc), candidates))
        scored: list[Scored] = []
        for doc, r in zip(candidates, results, strict=True):
            if "error" in r:
                scored.append(
                    Scored(
                        doc_id=doc.doc_id,
                        score=float("-inf"),
                        p_relevant=0.0,
                        confidence=None,
                        meta={**r, "excluded": True},
                    )
                )
            else:
                grade = r["grade"]
                scored.append(
                    Scored(
                        doc_id=doc.doc_id,
                        score=float(grade),
                        p_relevant=grade / 3.0,
                        confidence=None,
                        meta=r,
                    )
                )
        return sorted(scored, key=lambda s: (-s.score, s.doc_id))


class JevAnswer(Protocol):
    """Duck-typed contract for one value of `SystemOneResponse.answers`: the
    real `ScoreAnswer`/`NoulAnswer` pydantic models in production, a plain
    dict already shaped like their `.model_dump()` in tests."""

    def model_dump(self) -> dict: ...


class JevClient(Protocol):
    """Duck-typed contract for whatever `JevReranker.client` holds: the real
    `typesafe_sdk.TypeSafeClient` in production, a fake in tests."""

    def system_one(self, state: str, questions: dict, model: str = "") -> object: ...


def _dump(answer: object) -> dict:
    """Boundary between the SDK's typed `ScoreAnswer`/`NoulAnswer` objects and
    the plain JSON-serialisable dicts everything downstream (the cache,
    `compose_relevance`, the fakes in tests) works with.

    A real answer has `.model_dump()`; a test double that already hands back
    a plain dict does not, and passes through unchanged (task-12 delta #2).
    """
    if hasattr(answer, "model_dump"):
        return answer.model_dump()
    return dict(answer)


def _int_keyed(probabilities: dict) -> dict[int, float]:
    """Normalise a Score's `probabilities` to int keys.

    The SDK types this `dict[int, float]`, but the wire format and a JSON
    cache round-trip both force string keys - so a cache MISS (fresh
    `.model_dump()`) and a cache HIT (`json.loads`) hand this function
    different key types for what must be the same distribution (task-12
    delta #1). Not normalising here would make p_relevant depend on whether
    this is the first or second time a query has been scored.
    """
    return {int(k): float(v) for k, v in probabilities.items()}


def compose_relevance(answers: dict, cfg: Config) -> tuple[float, float, float]:
    """Combine Jev's three pre-registered atomic answers into
    (score, p_relevant, confidence), per BRD §4.4.

    Composition happens here, in code, rather than by asking Jev one broad
    question - TypeSafe's guidance is to decompose and compose, and their
    documented "indirection" weakness is why.

    Each Score is normalised by its OWN maximum level: topical_overlap has
    four levels (max 3), answers_query has three (max 2). Dividing both by
    the same constant would silently mis-weight them.

    Only `topical_overlap` and `answers_query` are read. `is_contradictory`
    and the cookbook-recipe `cookbook_relevant` (orchestrator addition) are
    carried through to `Scored.meta` for the separate `jev_cookbook` arm and
    for analysis, but must never move this arm's score - a document that
    contradicts the query's claim is directly relevant to deciding it, and
    answers_query's own top level already says so ("supporting it or
    contradicting it").
    """
    topical = answers["topical_overlap"]
    answers_q = answers["answers_query"]

    topical_probs = _int_keyed(topical["probabilities"])
    answers_probs = _int_keyed(answers_q["probabilities"])
    n_topical = max(topical_probs)
    n_answers = max(answers_probs)

    score = cfg.W_TOPICAL * (topical["score"] / n_topical) + cfg.W_ANSWERS * (
        answers_q["score"] / n_answers
    )

    # P(relevant) = mass on the top two levels of answers_query: "partial or
    # indirect evidence" and "settles the claim". A refuting document lands
    # in the top level, which is correct - it IS relevant.
    top_two = sorted(answers_probs)[-2:]
    p_relevant = float(min(1.0, max(0.0, sum(answers_probs[k] for k in top_two))))

    return float(score), p_relevant, float(answers_q["confidence"])


class JevReranker:
    """Arm D. TypeSafe Jev `Score`/`Noul`, one call per (query, chunk) pair.

    **This is the subject of the benchmark.** Parallelism comes from
    ~cfg.SEMAPHORE concurrent calls, never from stuffing all candidates into
    one state: Jev's documented weakness #5 is that accuracy falls as the
    state fills with content unrelated to the decision, and their own
    reranking cookbook says the same thing independently ("one request per
    candidate - no request sees another").

    Each call asks four questions over the one-pair state: the three
    pre-registered ones `compose_relevance` combines, plus `cookbook_relevant`
    for the separate `jev_cookbook` arm - independent questions over one
    state run in parallel, so this costs tokens but not an extra request.

    Failures (API exception, or any answer this arm's contract does not
    expect) are recorded as data (`meta["error"]`) and excluded
    (`meta["excluded"] = True`), never scored 0.0 - matching arm C. Failures
    are cached too: the published reproduce path is `git lfs pull && make
    report` with no API keys, so an uncached failure would make a warm-cache
    run attempt a live call and die.
    """

    name = "jev"

    def __init__(
        self,
        cfg: Config,
        cache: ResponseCache,
        client: JevClient | None = None,
        dataset: str = "",
    ) -> None:
        self.cfg = cfg
        self.cache = cache
        self.dataset = dataset
        self._injected_client = client
        if client is None:
            if not os.environ.get("TYPESAFE_API_KEY"):
                raise RuntimeError("TYPESAFE_API_KEY is not set; arm D needs it")
            from typesafe_sdk import TypeSafeClient

            client = TypeSafeClient(api_key=os.environ["TYPESAFE_API_KEY"])
        self.client = client

    def _cache_key(self, query: Query, doc: Doc) -> str:
        return cache_key(
            self.name,
            self.cfg.JEV_MODEL,
            self.dataset,
            query.query_id,
            doc.doc_id,
            self.cfg.PROMPT_VERSION,
        )

    def _judge(self, query: Query, doc: Doc) -> dict:
        """Run (or fetch from cache) one four-question call. Returns either
        the four dumped answers plus `latency_s` on success, or
        {"error": str, "latency_s": float} on any failure - callers must
        branch on "error" in the result, never assume the question keys."""
        key = self._cache_key(query, doc)
        # `arm=self.name` on BOTH get and put - passing different values
        # (or the `misc` default) reads and writes different shard files,
        # which would make every "cache hit" silently a miss.
        cached = self.cache.get(key, arm=self.name)
        if cached is not None:
            return cached
        state = JEV_STATE.format(query=query.text, document=doc.text[:4000])
        started = time.perf_counter()
        try:
            resp = self.client.system_one(
                state=state, questions=JEV_QUESTIONS_V1, model=self.cfg.JEV_MODEL
            )
            payload = {q: _dump(a) for q, a in resp.answers.items()}
            payload["latency_s"] = time.perf_counter() - started
        except Exception as exc:  # recorded, never silently zeroed
            payload = {
                "error": f"{type(exc).__name__}: {exc}",
                "latency_s": time.perf_counter() - started,
            }
        self.cache.put(key, payload, arm=self.name)
        return payload

    async def _judge_all(self, query: Query, candidates: Sequence[Doc]) -> list[dict]:
        from typesafe_sdk import AsyncTypeSafeClient

        sem = asyncio.Semaphore(self.cfg.SEMAPHORE)
        async with AsyncTypeSafeClient(api_key=os.environ["TYPESAFE_API_KEY"]) as client:

            async def one(doc: Doc) -> dict:
                key = self._cache_key(query, doc)
                cached = self.cache.get(key, arm=self.name)
                if cached is not None:
                    return cached
                state = JEV_STATE.format(query=query.text, document=doc.text[:4000])
                started = time.perf_counter()
                async with sem:
                    try:
                        resp = await client.system_one(
                            state=state, questions=JEV_QUESTIONS_V1, model=self.cfg.JEV_MODEL
                        )
                        payload = {q: _dump(a) for q, a in resp.answers.items()}
                        payload["latency_s"] = time.perf_counter() - started
                    except Exception as exc:
                        payload = {
                            "error": f"{type(exc).__name__}: {exc}",
                            "latency_s": time.perf_counter() - started,
                        }
                self.cache.put(key, payload, arm=self.name)
                return payload

            return list(await asyncio.gather(*(one(d) for d in candidates)))

    def _judge_batch(self, query: Query, candidates: Sequence[Doc]) -> list[dict]:
        """Raw per-doc four-question results, shared by this arm and
        `JevCookbookReranker` so the fourth question rides the same cached
        call instead of doubling API spend. Concurrency is an implementation
        detail: `rerank` stays synchronous so latency is measured identically
        across all five arms (ARCHITECTURE.md §3)."""
        if self._injected_client is None:
            return asyncio.run(self._judge_all(query, candidates))
        return [self._judge(query, d) for d in candidates]

    def rerank(self, query: Query, candidates: Sequence[Doc]) -> list[Scored]:
        results = self._judge_batch(query, candidates)
        scored: list[Scored] = []
        for doc, r in zip(candidates, results, strict=True):
            if "error" in r:
                scored.append(
                    Scored(
                        doc_id=doc.doc_id,
                        score=float("-inf"),
                        p_relevant=0.0,
                        confidence=None,
                        meta={**r, "excluded": True},
                    )
                )
                continue
            score, p, conf = compose_relevance(r, self.cfg)
            scored.append(
                Scored(doc_id=doc.doc_id, score=score, p_relevant=p, confidence=conf, meta=dict(r))
            )
        return sorted(scored, key=lambda s: (-s.score, s.doc_id))


class JevCookbookReranker:
    """TypeSafe's own reranking-cookbook design (docs.typesafe.ai/cookbooks/
    rerank_typesafe.md, orchestrator addition approved 2026-09-20): one Noul
    per (query, chunk), sorted descending, no composition, no weights.

    Reuses `JevReranker`'s judge calls and cache verbatim - the fourth
    question (`cookbook_relevant`) runs inside the SAME state/call as the
    three pre-registered ones, so this arm never issues a second request. It
    reads only `cookbook_relevant`; `compose_relevance` reads only the other
    three. The two are measured, and can rank, independently - that is the
    whole point of running both from one call: if the composed arm loses, the
    rebuttal is "you didn't follow their recipe"; if it wins, this arm is
    there to ask whether the tuned weights did the work.

    `confidence` is always None: a Noul has no separate confidence field
    distinct from the probability itself (see the primitives doc).
    """

    name = "jev_cookbook"

    def __init__(self, jev: JevReranker) -> None:
        self.jev = jev

    def rerank(self, query: Query, candidates: Sequence[Doc]) -> list[Scored]:
        results = self.jev._judge_batch(query, candidates)
        scored: list[Scored] = []
        for doc, r in zip(candidates, results, strict=True):
            if "error" in r:
                scored.append(
                    Scored(
                        doc_id=doc.doc_id,
                        score=float("-inf"),
                        p_relevant=0.0,
                        confidence=None,
                        meta={**r, "excluded": True},
                    )
                )
                continue
            noul = float(r["cookbook_relevant"]["noul"])
            scored.append(
                Scored(
                    doc_id=doc.doc_id, score=noul, p_relevant=noul, confidence=None, meta=dict(r)
                )
            )
        return sorted(scored, key=lambda s: (-s.score, s.doc_id))
