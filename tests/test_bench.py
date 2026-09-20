"""Tests for src.bench — the benchmark orchestration CLI.

bench.py wires stages together and contains no scoring logic of its own:
every metric value comes from src.metrics, every CI from src.stats. These
tests fake every network-touching stage (dataset loading, embeddings, the
LLM/Jev clients) so the suite spends no money and touches no network, per
CLAUDE.md's cost discipline.

`build_pools_and_sims` is being added to src/embed.py by a parallel worker
and may not exist on disk yet. Production code in bench.py imports it lazily
(inside the functions that need it) for exactly this reason; tests here
patch it onto the src.embed module with `raising=False` so they pass whether
or not it has landed yet.
"""

import math

import pytest

import src.bench as bench
import src.embed as embed
from src.cache import ResponseCache
from src.config import CONFIG
from src.rerankers import CosineReranker, PlattReranker
from src.types import CandidatePool, Corpus, Doc, Query, Scored

# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------


class FakeArm:
    """A reranker double: returns a canned, already-sorted ranking per query."""

    def __init__(self, name: str, ranking: dict[str, list[Scored]]) -> None:
        self.name = name
        self._ranking = ranking

    def rerank(self, query: Query, candidates) -> list[Scored]:
        return self._ranking[query.query_id]


def _doc(doc_id: str, n_words: int) -> Doc:
    return Doc(doc_id, " ".join(["word"] * n_words))


def _pool(query_id: str, docs: list[Doc]) -> CandidatePool:
    return CandidatePool(Query(query_id, "some query text"), tuple(docs))


def _scored(doc_id: str, score: float, p_relevant: float, meta: dict | None = None) -> Scored:
    return Scored(doc_id=doc_id, score=score, p_relevant=p_relevant, meta=meta or {})


# ---------------------------------------------------------------------------
# parse_args (draft tests)
# ---------------------------------------------------------------------------


def test_parse_args_defaults_to_all_arms_and_policies() -> None:
    args = bench.parse_args([])
    assert args.arms == "all" and args.policies == "all"


def test_parse_args_accepts_an_arm_subset() -> None:
    """Arm B must stay skippable — torch is the likeliest install failure."""
    assert bench.parse_args(["--arms", "cosine,llm,jev,platt"]).arms == "cosine,llm,jev,platt"


def test_parse_args_rejects_an_unknown_arm() -> None:
    with pytest.raises(SystemExit):
        bench.parse_args(["--arms", "telepathy"])


def test_parse_args_defaults_split_to_dev() -> None:
    """Running test by accident burns the once-per-prompt_version budget."""
    assert bench.parse_args([]).split == "dev"


# ---------------------------------------------------------------------------
# ResultRow (draft test)
# ---------------------------------------------------------------------------


def test_result_row_carries_every_stamp_field() -> None:
    fields = set(bench.ResultRow.__dataclass_fields__)
    assert {
        "dataset",
        "split",
        "arm",
        "policy",
        "metric",
        "value",
        "ci_low",
        "ci_high",
        "n_queries",
        "n_excluded",
        "prompt_version",
        "model",
        "seed",
        "git_sha",
    } <= fields


# ---------------------------------------------------------------------------
# build_arms
# ---------------------------------------------------------------------------


def test_build_arms_skips_cross_encoder_when_not_requested() -> None:
    arms = bench.build_arms(["cosine"], CONFIG, cache=None, sims={"q1": {"d1": 0.5}}, dataset="d")
    assert [a.name for a in arms] == ["cosine"]


def test_jev_cookbook_is_a_known_arm_and_constructible(monkeypatch, tmp_path) -> None:
    """Correction #5: jev_cookbook is the sixth arm and must be buildable."""
    monkeypatch.setenv("TYPESAFE_API_KEY", "fake-key")
    assert "jev_cookbook" in bench.ARM_NAMES
    cache = ResponseCache(tmp_path)
    arms = bench.build_arms(["jev_cookbook"], CONFIG, cache, sims={}, dataset="d")
    assert [a.name for a in arms] == ["jev_cookbook"]


def test_build_arms_jev_cookbook_shares_the_composed_jev_arm(monkeypatch, tmp_path) -> None:
    """jev_cookbook must reuse jev's client/cache so the fourth question rides
    the same call — it must never trigger a second API call per pair."""
    monkeypatch.setenv("TYPESAFE_API_KEY", "fake-key")
    cache = ResponseCache(tmp_path)
    jev_arm, cookbook_arm = bench.build_arms(
        ["jev", "jev_cookbook"], CONFIG, cache, sims={}, dataset="d"
    )
    assert cookbook_arm.jev is jev_arm


def test_build_arms_platt_wraps_the_same_cosine_instance(monkeypatch) -> None:
    """Arm E recalibrates arm A's own scores; it must share the cosine arm
    built for this run, not construct a second, independent one."""
    sims = {"q1": {"d1": 0.5}}
    cosine_arm, platt_arm = bench.build_arms(
        ["cosine", "platt"], CONFIG, cache=None, sims=sims, dataset="d"
    )
    assert platt_arm.base is cosine_arm


def test_unknown_arm_name_in_build_arms_raises() -> None:
    with pytest.raises(KeyError):
        bench.build_arms(["telepathy"], CONFIG, cache=None, sims={}, dataset="d")


# ---------------------------------------------------------------------------
# git_sha
# ---------------------------------------------------------------------------


def test_git_sha_returns_a_short_hash_in_this_repo() -> None:
    sha = bench.git_sha()
    assert sha != "unknown"
    assert len(sha) >= 7


def test_git_sha_falls_back_to_unknown_when_git_is_unavailable(monkeypatch) -> None:
    def boom(*args, **kwargs):
        raise FileNotFoundError("no git binary")

    monkeypatch.setattr(bench.subprocess, "check_output", boom)
    assert bench.git_sha() == "unknown"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def test_main_calls_run_with_parsed_args(monkeypatch) -> None:
    sentinel = object()
    calls = []
    monkeypatch.setattr(bench, "parse_args", lambda: sentinel)
    monkeypatch.setattr(bench, "run", lambda args: calls.append(args))
    bench.main()
    assert calls == [sentinel]


# ---------------------------------------------------------------------------
# _assert_sims_vary (correction #1)
# ---------------------------------------------------------------------------


def test_assert_sims_vary_raises_on_placeholder_identical_scores() -> None:
    """A silent regression to placeholder zeros must fail loudly, not produce
    a plausible-looking baseline row."""
    with pytest.raises(AssertionError):
        bench._assert_sims_vary({"q1": {"d1": 0.0, "d2": 0.0, "d3": 0.0}})


def test_assert_sims_vary_passes_when_scores_differ() -> None:
    bench._assert_sims_vary({"q1": {"d1": 0.1, "d2": 0.9}})


def test_assert_sims_vary_tolerates_a_single_candidate() -> None:
    """A query with exactly one pooled candidate has nothing to differ from —
    that is not a placeholder regression."""
    bench._assert_sims_vary({"q1": {"d1": 0.42}})


# ---------------------------------------------------------------------------
# fit_platt (correction #2)
# ---------------------------------------------------------------------------


def test_fit_platt_requests_the_dev_split_regardless_of_the_runs_own_split(
    monkeypatch,
) -> None:
    """CLAUDE.md non-negotiable #1: the calibrator is fitted on dev only.
    fit_platt takes no split argument at all, so it is structurally
    impossible for a --split test run to feed it test data."""
    seen_splits: list[str] = []

    def fake_get_split(corpus, split, cfg):
        seen_splits.append(split)
        return [Query("qd1", "dev query")]

    def fake_build_pools_and_sims(corpus, queries, embedder, cfg):
        pool = _pool("qd1", [_doc("d1", 3), _doc("d2", 4)])
        sims = {"qd1": {"d1": 0.9, "d2": 0.1}}
        return {"qd1": pool}, sims

    monkeypatch.setattr(bench, "get_split", fake_get_split)
    monkeypatch.setattr(embed, "build_pools_and_sims", fake_build_pools_and_sims, raising=False)

    corpus = Corpus("d", {}, {}, {"qd1": {"d1": 1, "d2": 0}})
    # The base cosine arm's own sims are for the RUN split, entirely separate
    # from the dev sims fit_platt builds internally just to fit the
    # calibrator — mirror that separation here rather than reusing one dict.
    run_sims = {"qd1": {"d1": 0.9, "d2": 0.1}}
    arm = PlattReranker(CosineReranker(run_sims))

    bench.fit_platt(arm, corpus, embedder=object(), cfg=CONFIG)

    assert seen_splits == ["dev"]
    # fitted: rerank no longer raises CalibratorNotFittedError
    out = arm.rerank(Query("qd1", "dev query"), [_doc("d1", 3), _doc("d2", 4)])
    assert {s.doc_id for s in out} == {"d1", "d2"}


def test_fit_platt_labels_from_qrels_grade_greater_than_zero(monkeypatch) -> None:
    """Label is 1 iff the qrels grade for that (query, doc) pair is > 0."""
    captured: dict = {}

    def fake_get_split(corpus, split, cfg):
        return [Query("qd1", "dev query")]

    def fake_build_pools_and_sims(corpus, queries, embedder, cfg):
        pool = _pool("qd1", [_doc("d1", 1), _doc("d2", 1), _doc("d3", 1)])
        sims = {"qd1": {"d1": 0.9, "d2": 0.5, "d3": 0.1}}
        return {"qd1": pool}, sims

    monkeypatch.setattr(bench, "get_split", fake_get_split)
    monkeypatch.setattr(embed, "build_pools_and_sims", fake_build_pools_and_sims, raising=False)

    real_fit = PlattReranker.fit

    def spy_fit(self, scores, labels, split="dev"):
        captured["scores"] = list(scores)
        captured["labels"] = list(labels)
        return real_fit(self, scores, labels, split=split)

    monkeypatch.setattr(PlattReranker, "fit", spy_fit)

    # d1 grade 2 (>0 -> relevant), d2 grade 0 (not relevant), d3 unjudged (not relevant)
    corpus = Corpus("d", {}, {}, {"qd1": {"d1": 2, "d2": 0}})
    arm = PlattReranker(CosineReranker({}))
    bench.fit_platt(arm, corpus, embedder=object(), cfg=CONFIG)

    paired = dict(zip(captured["scores"], captured["labels"], strict=True))
    assert paired[0.9] == 1
    assert paired[0.5] == 0
    assert paired[0.1] == 0


# ---------------------------------------------------------------------------
# build_result_rows: ranking metrics not duplicated per policy (correction #3)
# ---------------------------------------------------------------------------


def _two_query_pools() -> dict[str, CandidatePool]:
    return {
        "q1": _pool("q1", [_doc("d1", 2), _doc("d2", 3)]),
        "q2": _pool("q2", [_doc("d3", 1), _doc("d4", 4)]),
    }


def _cosine_like_ranking() -> dict[str, list[Scored]]:
    return {
        "q1": [_scored("d1", 0.9, 0.9), _scored("d2", 0.1, 0.1)],
        "q2": [_scored("d3", 0.8, 0.8), _scored("d4", 0.2, 0.2)],
    }


def test_ranking_metrics_appear_exactly_once_per_dataset_arm_with_policy_none() -> None:
    pools = _two_query_pools()
    queries = [pools["q1"].query, pools["q2"].query]
    qrels = {"q1": {"d1": 1}, "q2": {"d3": 1}}
    arm = FakeArm("cosine", _cosine_like_ranking())

    rows = bench.build_result_rows(
        dataset="d",
        split="dev",
        queries=queries,
        pools=pools,
        qrels=qrels,
        arms=[arm],
        policies=["fixed", "threshold"],
        cfg=CONFIG,
        seed=42,
        sha="abc1234",
    )

    ranking_rows = [r for r in rows if r.policy == "none"]
    metric_names = [r.metric for r in ranking_rows]
    # one row per ranking metric, no duplication across the two policies
    assert len(metric_names) == len(set(metric_names))
    assert len(ranking_rows) == len(bench._metric_fns(CONFIG))


def test_selection_metrics_appear_per_dataset_arm_and_policy() -> None:
    pools = _two_query_pools()
    queries = [pools["q1"].query, pools["q2"].query]
    qrels = {"q1": {"d1": 1}, "q2": {"d3": 1}}
    arm = FakeArm("cosine", _cosine_like_ranking())

    rows = bench.build_result_rows(
        dataset="d",
        split="dev",
        queries=queries,
        pools=pools,
        qrels=qrels,
        arms=[arm],
        policies=["fixed", "threshold"],
        cfg=CONFIG,
        seed=42,
        sha="abc1234",
    )

    selection_metrics = {"mean_kept", "abstention_rate", "mean_words_forwarded"}
    for policy in ("fixed", "threshold"):
        policy_rows = {r.metric for r in rows if r.policy == policy}
        assert selection_metrics <= policy_rows

    # fixed (TAU irrelevant, keeps up to FIXED_K) and threshold (TAU=0.5) must
    # be genuinely independent rows, not one row duplicated under two labels
    fixed_kept = next(r.value for r in rows if r.policy == "fixed" and r.metric == "mean_kept")
    threshold_kept = next(
        r.value for r in rows if r.policy == "threshold" and r.metric == "mean_kept"
    )
    assert fixed_kept != threshold_kept


def test_mean_words_forwarded_counts_whitespace_tokens_of_kept_docs() -> None:
    """BRD calls this 'tokens forwarded', but a real tokenizer is a new
    dependency for a number only ever compared between arms over identical
    text — so this is a whitespace-word proxy, named for what it measures."""
    pools = {"q1": _pool("q1", [_doc("d1", 2), _doc("d2", 3)])}
    queries = [pools["q1"].query]
    qrels = {"q1": {"d1": 1}}
    # both candidates clear TAU=0.5 under "threshold" -> both kept -> 2+3 words
    ranking = {"q1": [_scored("d1", 0.9, 0.9), _scored("d2", 0.8, 0.8)]}
    arm = FakeArm("cosine", ranking)

    rows = bench.build_result_rows(
        dataset="d",
        split="dev",
        queries=queries,
        pools=pools,
        qrels=qrels,
        arms=[arm],
        policies=["threshold"],
        cfg=CONFIG,
        seed=42,
        sha="abc1234",
    )
    row = next(r for r in rows if r.metric == "mean_words_forwarded")
    assert row.value == pytest.approx(5.0)


def test_abstention_rate_is_the_fraction_of_queries_that_abstained() -> None:
    pools = {
        "q1": _pool("q1", [_doc("d1", 1)]),
        "q2": _pool("q2", [_doc("d2", 1)]),
    }
    queries = [pools["q1"].query, pools["q2"].query]
    qrels = {"q1": {"d1": 1}, "q2": {"d2": 1}}
    # q1's only candidate clears TAU=0.5, q2's does not -> one abstention of two
    ranking = {"q1": [_scored("d1", 0.9, 0.9)], "q2": [_scored("d2", 0.1, 0.1)]}
    arm = FakeArm("cosine", ranking)

    rows = bench.build_result_rows(
        dataset="d",
        split="dev",
        queries=queries,
        pools=pools,
        qrels=qrels,
        arms=[arm],
        policies=["threshold"],
        cfg=CONFIG,
        seed=42,
        sha="abc1234",
    )
    row = next(r for r in rows if r.metric == "abstention_rate")
    assert row.value == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# model stamping (correction #4)
# ---------------------------------------------------------------------------


def test_model_for_arm_maps_each_arm_to_its_own_model() -> None:
    assert bench.model_for_arm("cosine", CONFIG) == CONFIG.EMBED_MODEL
    assert bench.model_for_arm("platt", CONFIG) == CONFIG.EMBED_MODEL
    assert bench.model_for_arm("cross_encoder", CONFIG) == CONFIG.CROSS_ENCODER
    assert bench.model_for_arm("llm", CONFIG) == CONFIG.LLM_MODEL
    assert bench.model_for_arm("jev", CONFIG) == CONFIG.JEV_MODEL
    assert bench.model_for_arm("jev_cookbook", CONFIG) == CONFIG.JEV_MODEL


def test_each_rows_model_is_its_own_arms_model_never_jevs_for_a_non_jev_arm() -> None:
    pools = {"q1": _pool("q1", [_doc("d1", 1)])}
    queries = [pools["q1"].query]
    qrels = {"q1": {"d1": 1}}
    cosine_arm = FakeArm("cosine", {"q1": [_scored("d1", 0.9, 0.9)]})
    jev_arm = FakeArm("jev", {"q1": [_scored("d1", 0.7, 0.7)]})

    rows = bench.build_result_rows(
        dataset="d",
        split="dev",
        queries=queries,
        pools=pools,
        qrels=qrels,
        arms=[cosine_arm, jev_arm],
        policies=["fixed"],
        cfg=CONFIG,
        seed=42,
        sha="abc1234",
    )

    cosine_models = {r.model for r in rows if r.arm == "cosine"}
    jev_models = {r.model for r in rows if r.arm == "jev"}
    assert cosine_models == {CONFIG.EMBED_MODEL}
    assert jev_models == {CONFIG.JEV_MODEL}
    assert CONFIG.EMBED_MODEL != CONFIG.JEV_MODEL


# ---------------------------------------------------------------------------
# exclusion handling
# ---------------------------------------------------------------------------


def test_excluded_rows_are_dropped_from_rankings_before_metrics() -> None:
    """An excluded (failed-call) doc must never be scored as if it were the
    top-ranked relevant result, and must be counted in n_excluded."""
    pools = {"q1": _pool("q1", [_doc("d1", 1), _doc("d2", 1)])}
    queries = [pools["q1"].query]
    qrels = {"q1": {"d2": 1}}
    ranking = {
        "q1": [
            _scored("d1", float("-inf"), 0.0, meta={"excluded": True}),
            _scored("d2", 0.5, 0.5),
        ]
    }
    arm = FakeArm("jev", ranking)

    rows = bench.build_result_rows(
        dataset="d",
        split="dev",
        queries=queries,
        pools=pools,
        qrels=qrels,
        arms=[arm],
        policies=["fixed"],
        cfg=CONFIG,
        seed=42,
        sha="abc1234",
    )
    ndcg_row = next(r for r in rows if r.metric == "ndcg@10")
    # d2 (the only non-excluded, relevant doc) still ranks first once d1 is
    # dropped, so ndcg@10 must be a perfect 1.0, not degraded by d1 sitting
    # ahead of it in the raw (unfiltered) ranking.
    assert ndcg_row.value == pytest.approx(1.0)
    assert all(r.n_excluded == 1 for r in rows)


def test_n_excluded_counts_per_dataset_arm_not_cumulatively_across_arms() -> None:
    pools = {"q1": _pool("q1", [_doc("d1", 1)])}
    queries = [pools["q1"].query]
    qrels = {"q1": {"d1": 1}}
    clean_arm = FakeArm("cosine", {"q1": [_scored("d1", 0.9, 0.9)]})
    broken_arm = FakeArm(
        "jev", {"q1": [_scored("d1", float("-inf"), 0.0, meta={"excluded": True})]}
    )

    rows = bench.build_result_rows(
        dataset="d",
        split="dev",
        queries=queries,
        pools=pools,
        qrels=qrels,
        arms=[clean_arm, broken_arm],
        policies=["fixed"],
        cfg=CONFIG,
        seed=42,
        sha="abc1234",
    )
    assert all(r.n_excluded == 0 for r in rows if r.arm == "cosine")
    assert all(r.n_excluded == 1 for r in rows if r.arm == "jev")


# ---------------------------------------------------------------------------
# cost/latency rows
# ---------------------------------------------------------------------------


def test_cost_and_latency_rows_are_emitted_from_scored_meta() -> None:
    pools = {"q1": _pool("q1", [_doc("d1", 1), _doc("d2", 1)])}
    queries = [pools["q1"].query]
    qrels = {"q1": {"d1": 1}}
    meta1 = {"latency_s": 0.1, "input_tokens": 10, "output_tokens": 2}
    meta2 = {"latency_s": 0.2, "input_tokens": 12, "output_tokens": 3}
    ranking = {"q1": [_scored("d1", 3.0, 1.0, meta=meta1), _scored("d2", 1.0, 0.3, meta=meta2)]}
    arm = FakeArm("jev", ranking)

    rows = bench.build_result_rows(
        dataset="d",
        split="dev",
        queries=queries,
        pools=pools,
        qrels=qrels,
        arms=[arm],
        policies=["fixed"],
        cfg=CONFIG,
        seed=42,
        sha="abc1234",
    )
    cost_prefixes = ("latency", "n_calls", "input", "output")
    by_metric = {r.metric: r.value for r in rows if r.metric.startswith(cost_prefixes)}
    assert by_metric["n_calls"] == 2
    assert by_metric["input_tokens_total"] == 22
    assert by_metric["output_tokens_total"] == 5
    assert "latency_p50" in by_metric and "latency_p95" in by_metric


def test_no_cost_rows_for_arms_with_no_latency_meta() -> None:
    """Arms A, B*, E have no per-call latency/token meta — no fabricated rows."""
    pools = {"q1": _pool("q1", [_doc("d1", 1)])}
    queries = [pools["q1"].query]
    qrels = {"q1": {"d1": 1}}
    arm = FakeArm("cosine", {"q1": [_scored("d1", 0.9, 0.9)]})

    rows = bench.build_result_rows(
        dataset="d",
        split="dev",
        queries=queries,
        pools=pools,
        qrels=qrels,
        arms=[arm],
        policies=["fixed"],
        cfg=CONFIG,
        seed=42,
        sha="abc1234",
    )
    assert not any(r.metric in {"latency_p50", "latency_p95", "n_calls"} for r in rows)


def test_does_not_convert_tokens_to_dollars() -> None:
    """CLAUDE.md: Task 17 does the money conversion from a documented price
    table. bench.py must never invent a price."""
    import inspect

    source = inspect.getsource(bench)
    for banned in ("dollar", "usd", "price", "cost_usd", "$"):
        assert banned not in source.lower()


# ---------------------------------------------------------------------------
# stamped fields, generally
# ---------------------------------------------------------------------------


def test_every_row_carries_seed_prompt_version_and_git_sha() -> None:
    pools = {"q1": _pool("q1", [_doc("d1", 1)])}
    queries = [pools["q1"].query]
    qrels = {"q1": {"d1": 1}}
    arm = FakeArm("cosine", {"q1": [_scored("d1", 0.9, 0.9)]})

    rows = bench.build_result_rows(
        dataset="d",
        split="dev",
        queries=queries,
        pools=pools,
        qrels=qrels,
        arms=[arm],
        policies=["fixed"],
        cfg=CONFIG,
        seed=99,
        sha="deadbee",
    )
    assert all(r.seed == 99 for r in rows)
    assert all(r.git_sha == "deadbee" for r in rows)
    assert all(r.prompt_version == CONFIG.PROMPT_VERSION for r in rows)
    assert all(r.dataset == "d" and r.split == "dev" for r in rows)


# ---------------------------------------------------------------------------
# run(): end-to-end wiring with everything faked
# ---------------------------------------------------------------------------


def test_run_end_to_end_with_fakes_fits_platt_on_dev_during_a_test_split_run(
    monkeypatch, tmp_path
) -> None:
    docs = {f"d{i}": Doc(f"d{i}", " ".join(["word"] * (i + 1))) for i in range(3)}
    corpus = Corpus(
        "fake",
        docs,
        {"t1": Query("t1", "test query"), "v1": Query("v1", "dev query")},
        {"t1": {"d0": 1}, "v1": {"d0": 1}},
    )
    split_calls: list[str] = []

    def fake_get_split(c, split, cfg):
        split_calls.append(split)
        return [c.queries["t1"]] if split == "test" else [c.queries["v1"]]

    def fake_load_corpus(name, cfg):
        return corpus

    def fake_build_pools_and_sims(c, queries, embedder, cfg):
        pools = {q.query_id: CandidatePool(q, tuple(c.docs.values())) for q in queries}
        sims = {q.query_id: {"d0": 0.9, "d1": 0.5, "d2": 0.1} for q in queries}
        return pools, sims

    monkeypatch.setattr(bench, "load_corpus", fake_load_corpus)
    monkeypatch.setattr(bench, "get_split", fake_get_split)
    monkeypatch.setattr(bench, "write_manifest", lambda *a, **k: None)
    monkeypatch.setattr(bench, "OpenAIEmbedder", lambda cfg: object())
    monkeypatch.setattr(embed, "build_pools_and_sims", fake_build_pools_and_sims, raising=False)

    cfg = CONFIG.tuned(RESULTS_DIR=tmp_path, CACHE_DIR=tmp_path / "cache")
    args = bench.parse_args(
        ["--arms", "cosine,platt", "--policies", "fixed", "--split", "test", "--datasets", "fake"]
    )
    frame = bench.run(args, cfg=cfg)

    assert "dev" in split_calls  # platt was fitted on dev despite --split test
    assert "test" in split_calls
    assert (tmp_path / "metrics-test.csv").exists()
    assert set(frame["arm"]) == {"cosine", "platt"}
    assert "ndcg@10" in set(frame.loc[frame["policy"] == "none", "metric"])
    assert (frame["split"] == "test").all()


def test_run_writes_csv_named_for_the_split(monkeypatch, tmp_path) -> None:
    corpus = Corpus(
        "fake",
        {"d0": Doc("d0", "one two")},
        {"v1": Query("v1", "dev query")},
        {"v1": {"d0": 1}},
    )

    def fake_get_split(c, split, cfg):
        return [c.queries["v1"]]

    def fake_build_pools_and_sims(c, queries, embedder, cfg):
        pools = {q.query_id: CandidatePool(q, tuple(c.docs.values())) for q in queries}
        sims = {q.query_id: {"d0": 0.7} for q in queries}
        return pools, sims

    monkeypatch.setattr(bench, "load_corpus", lambda name, cfg: corpus)
    monkeypatch.setattr(bench, "get_split", fake_get_split)
    monkeypatch.setattr(bench, "write_manifest", lambda *a, **k: None)
    monkeypatch.setattr(bench, "OpenAIEmbedder", lambda cfg: object())
    monkeypatch.setattr(embed, "build_pools_and_sims", fake_build_pools_and_sims, raising=False)

    cfg = CONFIG.tuned(RESULTS_DIR=tmp_path, CACHE_DIR=tmp_path / "cache")
    args = bench.parse_args(["--arms", "cosine", "--split", "dev", "--datasets", "fake"])
    bench.run(args, cfg=cfg)

    assert (tmp_path / "metrics-dev.csv").exists()


def test_compared_cost_numbers_carry_a_ci_and_census_counts_do_not() -> None:
    """CLAUDE.md non-negotiable #3: a number that gets compared across arms
    carries a CI. The means do (bootstrap over the per-call population); the
    percentiles and totals deliberately do not, because they are census counts
    of what this run did rather than estimates, and bootstrap_ci estimates a
    mean, not a percentile. A fabricated interval would be worse than none."""
    scored = {
        f"q{i}": [
            Scored(
                doc_id="d1",
                score=1.0,
                p_relevant=0.5,
                confidence=None,
                meta={"latency_s": 0.1 * (i + 1), "input_tokens": 100 + i, "output_tokens": 5},
            )
        ]
        for i in range(12)
    }
    common = {
        "dataset": "scifact",
        "split": "dev",
        "arm": "llm",
        "n_queries": 12,
        "n_excluded": 0,
        "prompt_version": "v1",
        "model": "claude-haiku-4-5",
        "seed": 42,
        "git_sha": "abc1234",
    }
    rows = {r.metric: r for r in bench._cost_rows(scored, common)}

    for metric in ("mean_latency_s", "mean_input_tokens", "mean_output_tokens"):
        row = rows[metric]
        assert not math.isnan(row.ci_low) and not math.isnan(row.ci_high), metric
        assert row.ci_low <= row.value <= row.ci_high, metric

    for metric in ("latency_p50", "latency_p95", "n_calls", "input_tokens_total"):
        assert math.isnan(rows[metric].ci_low), metric
        assert math.isnan(rows[metric].ci_high), metric
