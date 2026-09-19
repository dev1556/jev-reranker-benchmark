# TECH_REQUIREMENTS.md

Concrete build targets. The *why* is in `BRD.md`; the *how it's shaped* is in `ARCHITECTURE.md`.
Every requirement below is testable — if it can't be checked, it isn't a requirement.

---

## 1. Platform

| | |
|---|---|
| Python | **3.12** (`>=3.12,<3.13`). Hard pin: no torch wheels for 3.13/3.14. `uv` provisions it. |
| Package manager | **uv** only. `uv.lock` committed and authoritative. |
| OS target | Windows 11 primary (dev machine); CI runs ubuntu-latest. No OS-specific paths — `pathlib` everywhere. |
| Hardware | CPU only. Cross-encoder must not require CUDA. |
| Repo | `github.com/dev1556/<repo>`, public. Git LFS for `cache/**`. |

## 2. Dependencies

Pinned to minor in `pyproject.toml`; `uv.lock` fixes exact versions.

**Runtime**

| Package | Purpose | Notes |
|---|---|---|
| `typesafe-sdk` | arm D | `AsyncTypeSafeClient`, `Score`, `Noul`, model `jev-latest` |
| `anthropic` | arm C | Haiku 4.5, pointwise |
| `openai` | embeddings | `text-embedding-3-small`, 1536-d |
| `sentence-transformers` + `torch` (CPU) | arm B | `BAAI/bge-reranker-base`; heaviest dep, arm must be skippable |
| `datasets` | BEIR loading | `BeIR/scifact`, `BeIR/fiqa` + qrels |
| `numpy`, `scipy` | vectors, stats | |
| `scikit-learn` | arm E Platt, AUROC | `LogisticRegression`, `roc_auc_score` |
| `pandas` | result frames, CSV | |
| `matplotlib` | charts | no seaborn; one plotting dep is enough |
| `streamlit` | explorer | |
| `python-dotenv` | key loading | |
| `tenacity` | retry beyond SDK defaults | only where an SDK lacks it |

**Dev:** `pytest`, `pytest-asyncio`, `ruff`.

New dependencies require human approval (`CLAUDE.md`). A few lines of numpy beats a new package.

## 3. Configuration

All tuned constants live in **one** frozen dataclass in `src/config.py`. Nothing tuned may be
hardcoded elsewhere.

| Constant | Default | Tuned on |
|---|---|---|
| `POOL_SIZE` | 50 | fixed, never tuned |
| `W_TOPICAL` / `W_ANSWERS` | 0.35 / 0.65 | dev, one pass, then frozen |
| `TAU` (P(relevant) threshold) | tuned | dev |
| `C_LOW` (confidence gate) | tuned | dev |
| `MASS_TARGET` | tuned | dev |
| `SEMAPHORE` | 16 | not a tuned value; perf only |
| `PROMPT_VERSION` | `"v1"` | bumped on any rubric change |
| `SEED` | 42 | fixed |

**Secrets** come only from `.env` (gitignored) or environment: `TYPESAFE_API_KEY`,
`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`. `.env.example` documents all three. A missing key must fail
at startup with a message naming the key and which arm needs it — never mid-run after $8 of calls.

## 4. Functional requirements

### FR-1 Data
- Load SciFact (all 300 test queries) and FiQA (300 sampled, `seed=42`) with qrels.
- Sampled query IDs written to `data/manifests/` and **committed** — the sample must be reproducible.
- Dev/test split materialised explicitly; a single function is the only way to obtain either split,
  and requesting `test` logs the access. Tuning code requesting `test` is a bug.

### FR-2 Retrieval
- Embed corpus + queries via `text-embedding-3-small`, batched, cached to `.npz`.
- Cosine top-50 per query. **Identical pool handed to all five arms** — asserted at runtime, not
  assumed.
- Report Recall@50 per dataset as the shared ceiling.

### FR-3 Rerankers — all five implement `Reranker`
- **A cosine** — identity order; `p_relevant` = per-query min-max.
- **B cross-encoder** — `bge-reranker-base`; `p_relevant = sigmoid(logit)`; skippable via `--arms`.
- **C LLM** — Haiku 4.5, pointwise 0–3 rubric; `p_relevant = score/3`; prompt written with the same
  care as arm D (`CLAUDE.md` non-negotiable #4).
- **D Jev** — one `system_one` call per (query, chunk); three questions per `BRD.md` §4.4;
  ≤`SEMAPHORE` concurrent; `p_relevant` = mass on top two levels of `answers_query`; full per-level
  probabilities retained in `meta`.
- **E Platt** — `LogisticRegression` on arm A scores, **fitted on dev**, applied frozen to test.
  Loading a calibrator fitted on test must raise.

### FR-4 Gating — four policies, every arm
`fixed` (top-5) · `threshold` (P ≥ τ) · `mass` (cumulative expected relevance) ·
`confidence_gated` (τ + confidence widening/narrowing + abstain).
Arms with `confidence is None` must be handled explicitly; treating `None` as high confidence is a
defect, not a default.

### FR-5 Metrics
- **IR:** nDCG@10, Recall@10, Precision@5, MRR@10, Recall@50.
- **Calibration:** ECE (15 equal-mass bins), Brier + reliability/resolution/uncertainty
  decomposition, reliability diagram data, AUROC of confidence vs. correctness.
- Unjudged documents treated as irrelevant (standard BEIR); the count is reported, not hidden.

### FR-6 Statistics
- Bootstrap 95% CI, 10,000 resamples over queries, seeded.
- Paired two-sided randomisation test, 10,000 permutations, for every arm pair on nDCG@10.
- Holm correction across the arm-pair family. Both raw and corrected p reported.

### FR-7 Abstention (H5)
- Gold-removed sets: 50 queries per dataset, relevant docs removed from an **index copy**, re-retrieved.
- Cross-domain sets: 50 queries scored against the other dataset's corpus.
- Report per arm: abstention rate on each family, **false-abstention rate on answerable queries**,
  and AUROC over the pooled set. Refusal rate without false-refusal rate must not be emitted.

### FR-8 Robustness (H6)
- Three injection styles (naive imperative / keyword stuffing / framed instruction) applied to 5
  irrelevant chunks across 50 queries.
- Injection is a transform on candidate text reusing the normal scoring path — no separate pipeline.
- Report mean rank inflation and nDCG@10 delta, per arm per style.
- Injected corpora generated on demand under `data/adversarial/`, gitignored, never committed.

### FR-9 Ablations
- Rubric sensitivity: three variants on **dev**; report nDCG and ECE spread as a band.
- Determinism: 200 pairs × 5 repeats; report `score` stddev and level-flip rate.
- Structural invariant: `is_contradictory` asked in both polarities; report `|p + p' − 1|`
  distribution (tests TypeSafe's documented weakness #8).

### FR-10 Outputs
- `results/*.csv` — one row per (dataset, arm, policy, metric) with value, CI bounds, and the stamp
  fields from `CLAUDE.md` non-negotiable #5.
- `results/*.png` — reliability grid, Pareto (nDCG vs cost vs latency), robustness chart.
- `results/report.md` — table per hypothesis, explicit **ACCEPT / REJECT / INCONCLUSIVE** verdict on
  each of H1–H7 with numbers and p-values, plus a limitations section.

### FR-11 CLI
```
uv run python -m src.bench [--datasets scifact,fiqa] [--arms all|a,b,c,d,e]
                           [--policies all|...] [--queries N] [--no-cache]
                           [--split dev|test] [--seed 42]
```
`make smoke` = 25 queries, both datasets, all arms (~$0.50). `make all` = full run.
`make report` = regenerate outputs from cache with zero API calls.

### FR-12 Streamlit
Query input → all five rankings side by side → per-chunk `p_relevant` bar + confidence → which chunks
each policy keeps → injection toggle showing adversarial chunks moving → gold labels on demand.
Reads the same result files the CLI writes; reimplements no scoring.

## 5. Non-functional requirements

| | Target |
|---|---|
| **NFR-1** Cost | Full cold run ≤ $25. Smoke ≤ $1. Cost printed before a run starts and after it ends. |
| **NFR-2** Runtime | Full cold ≤ 90 min. Warm rerun from cache ≤ 2 min. |
| **NFR-3** Arm D latency | p50 < 1s to rerank 50 candidates (the H4 claim; measured `--no-cache`). |
| **NFR-4** Resumability | A `Ctrl-C` at any point loses no completed API call. |
| **NFR-5** Reproducibility | Warm cache + seed ⇒ byte-identical CSVs. Verified by a test. |
| **NFR-6** Offline tests | Full `pytest` passes with no network and no API keys. |
| **NFR-7** Repo size | Working tree ex-LFS < 25MB. Cache in LFS ≤ ~500MB. |
| **NFR-8** Failure visibility | Any excluded/errored call is counted and surfaced in the report. |

## 6. Testing

- **Metric correctness against hand-computed values** — a known nDCG, a known Precision@5, a
  synthetic perfectly-calibrated set scoring ECE ≈ 0, a deliberately miscalibrated set scoring high,
  a fixed-seed bootstrap CI. These are the tests that matter most; `metrics.py` and `stats.py` bugs
  are invisible downstream.
- **Contract tests** — all five arms satisfy `Reranker`; every `Scored.p_relevant ∈ [0,1]`; pools are
  identical across arms; `confidence is None` handled by all four policies.
- **Fixture tests** — arms C and D parse real cached responses, not hand-written mocks.
- **Guard tests** — fitting a calibrator on `test` raises; an error response never becomes `0.0`;
  requesting `test` during a tuning routine is detectable.
- No mocking of `metrics.py` or `stats.py`. Ever. They are the thing being trusted.

## 7. CI

GitHub Actions on every PR to `main`: `uv sync` → `ruff check` → `ruff format --check` → `pytest`.
Ubuntu, Python 3.12, no secrets, no network. Green CI is a precondition for the self-merge rule in
`CLAUDE.md` — that rule is only safe because this exists.

## 8. Acceptance

Ship when all of `BRD.md` §12 holds, plus:

1. `uv sync && make smoke` works from a clean clone on a machine with three keys.
2. `make report` reproduces every published number from the LFS cache with **zero** API calls.
3. CI green on `main`.
4. `results/report.md` carries a verdict on H1–H7 — including any that failed.
5. `README.md` leads with the headline chart and the numbers, and links the reproduction steps.
