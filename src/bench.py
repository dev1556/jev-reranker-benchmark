"""Benchmark orchestration CLI.

Wires stages together — dataset load, pooling, reranking, selection — and
contains no scoring logic of its own: every metric value comes from
`src.metrics`, every confidence interval from `src.stats`. If a number in the
output CSV did not come from one of those two modules (or a plain mean of
their per-query outputs), it does not belong here.

`build_pools_and_sims` (src.embed) is owned by a parallel worker and may not
exist on disk yet; it is imported lazily inside the functions that need it so
importing this module never fails on that account.
"""

import argparse
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.cache import ResponseCache
from src.config import CONFIG, Config
from src.data import get_split, load_corpus, write_manifest
from src.embed import Embedder, OpenAIEmbedder
from src.gating import POLICIES, select
from src.metrics import mrr_at_k, ndcg_at_k, precision_at_k, recall_at_k
from src.rerankers import PlattReranker, Reranker
from src.stats import bootstrap_ci
from src.types import CandidatePool, Corpus, Qrels, Query, Scored

ARM_NAMES: tuple[str, ...] = ("cosine", "cross_encoder", "llm", "jev", "jev_cookbook", "platt")


@dataclass(frozen=True)
class ResultRow:
    """One (dataset, arm, policy, metric) result. Every field here is a stamp
    non-negotiable #5 requires — an unstamped number cannot be defended later."""

    dataset: str
    split: str
    arm: str
    policy: str
    metric: str
    value: float
    ci_low: float
    ci_high: float
    n_queries: int
    n_excluded: int
    prompt_version: str
    model: str
    seed: int
    git_sha: str


def git_sha() -> str:
    """Short commit hash of this checkout, or "unknown" if git is unavailable.

    Broad except is deliberate and harmless here: this stamps *environment*
    metadata, never a measurement value, so a missing git binary degrades to
    a visible "unknown" stamp rather than crashing a run.
    """
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            cwd=Path(__file__).resolve().parent.parent,
        ).strip()
    except Exception:
        return "unknown"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="src.bench")
    p.add_argument("--datasets", default="scifact,fiqa")
    p.add_argument("--arms", default="all")
    p.add_argument("--policies", default="all")
    p.add_argument("--queries", type=int, default=None)
    p.add_argument("--split", choices=["dev", "test"], default="dev")
    p.add_argument("--seed", type=int, default=CONFIG.SEED)
    p.add_argument("--no-cache", action="store_true")
    args = p.parse_args(argv)
    if args.arms != "all":
        unknown = set(args.arms.split(",")) - set(ARM_NAMES)
        if unknown:
            p.error(f"unknown arm(s): {sorted(unknown)}; expected {ARM_NAMES}")
    return args


def build_arms(
    names: Sequence[str],
    cfg: Config,
    cache: ResponseCache | None,
    sims: dict[str, dict[str, float]],
    dataset: str,
) -> list[Reranker]:
    """Construct only the requested arms.

    Arm B (cross_encoder) is skipped unless requested, so a torch install
    failure cannot block the rest of the benchmark. `jev_cookbook` reuses the
    composed `jev` arm's client and cache (`JevReranker._judge_batch`) so its
    fourth question rides the same call instead of doubling API spend.
    """
    from src.rerankers import (
        CosineReranker,
        CrossEncoderReranker,
        JevCookbookReranker,
        JevReranker,
        LLMReranker,
    )

    cosine = CosineReranker(sims)
    jev_holder: dict[str, JevReranker] = {}

    def get_jev() -> JevReranker:
        if "jev" not in jev_holder:
            jev_holder["jev"] = JevReranker(cfg, cache, dataset=dataset)
        return jev_holder["jev"]

    factory: dict[str, Callable[[], Reranker]] = {
        "cosine": lambda: cosine,
        "cross_encoder": lambda: CrossEncoderReranker(cfg),
        "llm": lambda: LLMReranker(cfg, cache, dataset=dataset),
        "jev": get_jev,
        "jev_cookbook": lambda: JevCookbookReranker(get_jev()),
        "platt": lambda: PlattReranker(cosine),
    }
    return [factory[n]() for n in names]


def model_for_arm(arm_name: str, cfg: Config) -> str:
    """The model stamp that actually produced an arm's numbers.

    Never falls back to a default — a wrong or borrowed stamp here would
    misattribute every downstream chart (correction #4: cosine/platt use the
    embedding model, not cfg.JEV_MODEL).
    """
    try:
        return {
            "cosine": cfg.EMBED_MODEL,
            "platt": cfg.EMBED_MODEL,
            "cross_encoder": cfg.CROSS_ENCODER,
            "llm": cfg.LLM_MODEL,
            "jev": cfg.JEV_MODEL,
            "jev_cookbook": cfg.JEV_MODEL,
        }[arm_name]
    except KeyError:
        raise ValueError(f"no model stamp registered for arm {arm_name!r}") from None


def _metric_fns(cfg: Config) -> dict[str, Callable[[list[str], Mapping[str, int]], float]]:
    """Ranking metrics, each a thin wrapper around src.metrics — no formula
    lives here."""
    return {
        "ndcg@10": lambda r, g: ndcg_at_k(r, g, cfg.NDCG_K),
        "recall@10": lambda r, g: recall_at_k(r, g, cfg.NDCG_K),
        "precision@5": lambda r, g: precision_at_k(r, g, 5),
        "mrr@10": lambda r, g: mrr_at_k(r, g, cfg.NDCG_K),
        "recall@50": lambda r, g: recall_at_k(r, g, 50),
    }


def _assert_sims_vary(sims: dict[str, dict[str, float]]) -> None:
    """Guard against a silent regression to placeholder identical similarities
    (e.g. all zeros): every document would tie, the ranking would collapse to
    doc_id order, and arm A's p_relevant would be a meaningless constant 0.5
    everywhere while still being reported as the cosine baseline. Fail loudly
    instead of producing a plausible-looking row.
    """
    for query_id, doc_scores in sims.items():
        values = list(doc_scores.values())
        if len(values) > 1 and len(set(values)) == 1:
            raise AssertionError(
                f"sims for query {query_id!r} are all identical ({values[0]!r}); "
                f"refusing to score what would be a placeholder cosine baseline."
            )


def fit_platt(arm: PlattReranker, corpus: Corpus, embedder: Embedder, cfg: Config) -> None:
    """Fit arm E's calibrator on the DEV split of this dataset.

    Takes no `split` argument from the run at all — that is the point: a
    `--split test` run has no way to feed this function anything but dev
    data (CLAUDE.md non-negotiable #1). Labels are qrels grade > 0.
    """
    from src.embed import build_pools_and_sims

    dev_queries = get_split(corpus, "dev", cfg)
    _, dev_sims = build_pools_and_sims(corpus, dev_queries, embedder, cfg)

    scores: list[float] = []
    labels: list[int] = []
    for q in dev_queries:
        gold = corpus.qrels.get(q.query_id, {})
        for doc_id, sim in dev_sims[q.query_id].items():
            scores.append(sim)
            labels.append(1 if gold.get(doc_id, 0) > 0 else 0)
    arm.fit(np.asarray(scores, dtype=float), np.asarray(labels, dtype=int), split="dev")


def _cost_rows(
    scored_by_query: dict[str, list[Scored]], common: dict[str, object]
) -> list[ResultRow]:
    """Cost/latency diagnostics for the Pareto chart, aggregated straight from
    `Scored.meta` (arms C and D populate `latency_s`, `input_tokens`,
    `output_tokens`).

    Two shapes of number live here, and the difference matters:

    - **Means carry a bootstrap CI**, resampled over the per-call population.
      The Pareto chart compares arms on latency and token consumption, and a
      compared number without a CI is not publishable (CLAUDE.md
      non-negotiable #3). `mean_latency_s`, `mean_input_tokens` and
      `mean_output_tokens` are the rows a chart or a claim should use.
    - **Percentiles and totals carry NaN CIs**, deliberately. `latency_p50`,
      `latency_p95`, `n_calls`, `input_tokens_total` and `output_tokens_total`
      are descriptive census counts of what this run actually did, not
      estimates of a population parameter, and `bootstrap_ci` estimates a mean
      rather than a percentile. Reporting a fabricated interval on them would
      be worse than reporting none.

    Stays in tokens/seconds throughout: a monetary conversion needs a
    documented per-token rate table this repo does not own (Task 17).
    """
    all_scored = [s for scored in scored_by_query.values() for s in scored]
    latencies = [s.meta["latency_s"] for s in all_scored if "latency_s" in s.meta]
    if not latencies:
        return []
    input_tokens = [s.meta["input_tokens"] for s in all_scored if "input_tokens" in s.meta]
    output_tokens = [s.meta["output_tokens"] for s in all_scored if "output_tokens" in s.meta]

    nan = float("nan")
    seed = int(common["seed"])  # type: ignore[call-overload]
    lat_lo, lat_hi = bootstrap_ci(latencies, seed=seed)
    rows = [
        ResultRow(
            policy="none",
            metric="mean_latency_s",
            value=float(np.mean(latencies)),
            ci_low=lat_lo,
            ci_high=lat_hi,
            **common,
        ),
        ResultRow(
            policy="none",
            metric="latency_p50",
            value=float(np.percentile(latencies, 50)),
            ci_low=nan,
            ci_high=nan,
            **common,
        ),
        ResultRow(
            policy="none",
            metric="latency_p95",
            value=float(np.percentile(latencies, 95)),
            ci_low=nan,
            ci_high=nan,
            **common,
        ),
        ResultRow(
            policy="none",
            metric="n_calls",
            value=float(len(latencies)),
            ci_low=nan,
            ci_high=nan,
            **common,
        ),
    ]
    if input_tokens:
        in_lo, in_hi = bootstrap_ci(input_tokens, seed=seed)
        rows.append(
            ResultRow(
                policy="none",
                metric="mean_input_tokens",
                value=float(np.mean(input_tokens)),
                ci_low=in_lo,
                ci_high=in_hi,
                **common,
            )
        )
        rows.append(
            ResultRow(
                policy="none",
                metric="input_tokens_total",
                value=float(sum(input_tokens)),
                ci_low=nan,
                ci_high=nan,
                **common,
            )
        )
    if output_tokens:
        out_lo, out_hi = bootstrap_ci(output_tokens, seed=seed)
        rows.append(
            ResultRow(
                policy="none",
                metric="mean_output_tokens",
                value=float(np.mean(output_tokens)),
                ci_low=out_lo,
                ci_high=out_hi,
                **common,
            )
        )
        rows.append(
            ResultRow(
                policy="none",
                metric="output_tokens_total",
                value=float(sum(output_tokens)),
                ci_low=nan,
                ci_high=nan,
                **common,
            )
        )
    return rows


def build_result_rows(
    dataset: str,
    split: str,
    queries: Sequence[Query],
    pools: dict[str, CandidatePool],
    qrels: Qrels,
    arms: Sequence[Reranker],
    policies: Sequence[str],
    cfg: Config,
    seed: int,
    sha: str,
) -> list[ResultRow]:
    """Score every arm against the given queries/pools/qrels and stamp every
    row. This is the one place ranking, selection, and cost rows are
    assembled — but every metric value in it still comes from
    src.metrics/src.stats; this function only loops, filters excluded
    candidates, and takes means.
    """
    rows: list[ResultRow] = []
    n_queries = len(queries)
    metric_fns = _metric_fns(cfg)

    for arm in arms:
        scored_by_query: dict[str, list[Scored]] = {
            q.query_id: arm.rerank(q, pools[q.query_id].docs) for q in queries
        }
        n_excluded = sum(
            1 for scored in scored_by_query.values() for s in scored if s.meta.get("excluded")
        )
        common = dict(
            dataset=dataset,
            split=split,
            arm=arm.name,
            n_queries=n_queries,
            n_excluded=n_excluded,
            prompt_version=cfg.PROMPT_VERSION,
            model=model_for_arm(arm.name, cfg),
            seed=seed,
            git_sha=sha,
        )

        # Ranking metrics: computed once per arm, independent of any
        # selection policy. policy="none" so a reader never mistakes these
        # for a policy effect (correction #3).
        for metric_name, fn in metric_fns.items():
            values = [
                fn(
                    [s.doc_id for s in scored_by_query[q.query_id] if not s.meta.get("excluded")],
                    qrels.get(q.query_id, {}),
                )
                for q in queries
            ]
            lo, hi = bootstrap_ci(values, n=cfg.N_BOOTSTRAP, seed=seed)
            rows.append(
                ResultRow(
                    policy="none",
                    metric=metric_name,
                    value=float(np.mean(values)),
                    ci_low=lo,
                    ci_high=hi,
                    **common,
                )
            )

        # Selection metrics: depend on the policy, so one row set per policy.
        for policy in policies:
            kept_counts: list[float] = []
            abstentions: list[float] = []
            words_forwarded: list[float] = []
            for q in queries:
                selection = select(scored_by_query[q.query_id], policy, cfg)
                kept_counts.append(float(len(selection.doc_ids)))
                abstentions.append(1.0 if selection.abstained else 0.0)
                text_by_id = {d.doc_id: d.text for d in pools[q.query_id].docs}
                words_forwarded.append(
                    float(sum(len(text_by_id[doc_id].split()) for doc_id in selection.doc_ids))
                )
            for metric_name, values in (
                ("mean_kept", kept_counts),
                ("abstention_rate", abstentions),
                ("mean_words_forwarded", words_forwarded),
            ):
                lo, hi = bootstrap_ci(values, n=cfg.N_BOOTSTRAP, seed=seed)
                rows.append(
                    ResultRow(
                        policy=policy,
                        metric=metric_name,
                        value=float(np.mean(values)),
                        ci_low=lo,
                        ci_high=hi,
                        **common,
                    )
                )

        rows.extend(_cost_rows(scored_by_query, common))

    return rows


def run(args: argparse.Namespace, cfg: Config = CONFIG) -> pd.DataFrame:
    cache = ResponseCache(cfg.CACHE_DIR, enabled=not args.no_cache)
    sha = git_sha()
    rows: list[ResultRow] = []
    arm_names = list(ARM_NAMES) if args.arms == "all" else args.arms.split(",")
    policies = list(POLICIES) if args.policies == "all" else args.policies.split(",")

    for dataset in args.datasets.split(","):
        corpus = load_corpus(dataset, cfg)
        queries = get_split(corpus, args.split, cfg)
        if args.queries:
            queries = queries[: args.queries]
        write_manifest(dataset, args.split, [q.query_id for q in queries], cfg)

        embedder = OpenAIEmbedder(cfg)
        from src.embed import build_pools_and_sims

        pools, sims = build_pools_and_sims(corpus, queries, embedder, cfg)
        _assert_sims_vary(sims)

        arms = build_arms(arm_names, cfg, cache, sims, dataset)
        for arm in arms:
            if arm.name == "platt":
                fit_platt(arm, corpus, embedder, cfg)

        rows.extend(
            build_result_rows(
                dataset,
                args.split,
                queries,
                pools,
                corpus.qrels,
                arms,
                policies,
                cfg,
                args.seed,
                sha,
            )
        )

    frame = pd.DataFrame([r.__dict__ for r in rows])
    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cfg.RESULTS_DIR / f"metrics-{args.split}.csv", index=False)
    return frame


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
