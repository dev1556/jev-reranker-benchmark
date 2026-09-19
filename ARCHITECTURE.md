# ARCHITECTURE.md

How the code is laid out and why. Spec lives in `BRD.md`; build targets in `TECH_REQUIREMENTS.md`.

---

## 1. The shape of the problem

This is a **measurement harness**, not an application. That drives every structural decision:

- The five rerankers must be **swappable behind one interface**, or the comparison isn't fair — any
  difference in how arms are invoked becomes a confound in the results.
- Everything expensive is **cached at the edge**, so the pipeline can be re-run freely while
  iterating on metrics without re-spending money.
- Measurement code is **separated from the things being measured**, so a bug in a reranker can't
  silently alter a metric and vice versa.
- The whole run is **deterministic given a seed and a warm cache** — a reader must be able to
  reproduce the numbers exactly.

## 2. Layers

```
            ┌─────────────────────────────────────────────┐
  stage 0   │  data.py      datasets, splits, sampling     │
            └──────────────────────┬──────────────────────┘
                                   │ Query, Doc, Qrels
            ┌──────────────────────▼──────────────────────┐
  stage 1   │  embed.py     OpenAI vectors → cosine top-50 │
            └──────────────────────┬──────────────────────┘
                                   │ CandidatePool (identical for every arm)
            ┌──────────────────────▼──────────────────────┐
  stage 2   │  rerankers.py   A cosine   B cross-encoder   │
            │                 C llm      D jev    E platt  │
            └──────────────────────┬──────────────────────┘
                                   │ list[Scored]  (score, p_relevant, confidence)
            ┌──────────────────────▼──────────────────────┐
  stage 3   │  gating.py    fixed │ threshold │ mass │     │
            │               confidence_gated (+ abstain)   │
            └──────────────────────┬──────────────────────┘
                                   │ Selection
            ┌──────────────────────▼──────────────────────┐
  stage 4   │  metrics.py  IR + calibration                │
            │  stats.py    bootstrap CI, randomisation     │
            └──────────────────────┬──────────────────────┘
                                   │ ResultRow
            ┌──────────────────────▼──────────────────────┐
  stage 5   │  bench.py → results/*.csv, *.png, report.md  │
            │  app.py   → Streamlit explorer               │
            └─────────────────────────────────────────────┘

  cross-cutting:  cache.py  ·  config.py  ·  unanswerable.py  ·  robustness.py
```

Data flows one way. No stage reaches backwards. Each stage's output is a plain dataclass that can be
serialised, cached, and inspected in isolation — which is also what makes `app.py` cheap to build:
it re-renders stage 2–3 output for a single query rather than reimplementing anything.

## 3. The central abstraction

Everything hinges on one protocol. Five arms, one shape:

```python
class Reranker(Protocol):
    name: str
    def rerank(self, query: Query, candidates: list[Doc]) -> list[Scored]: ...

@dataclass(frozen=True)
class Scored:
    doc_id: str
    score: float             # arm-native, sorts the ranking. NOT comparable across arms.
    p_relevant: float        # calibrated-ish P(relevant) in [0,1]. THIS is what §5 measures.
    confidence: float | None # arm's own certainty. None where the arm has no such notion.
    meta: dict               # per-level probabilities, raw response, token counts, latency
```

**`score` vs `p_relevant` is the most important distinction in the codebase.** `score` orders the
ranking. `p_relevant` claims a probability, and the calibration metrics hold it to that claim. Cosine
has a `score` but its `p_relevant` is a fiction — measuring exactly how much of a fiction is the
point of hypothesis H2. Conflating the two would erase the entire finding, so they are separate
fields with separate docstrings.

`confidence` is `None` for arms A, B, E. The gating policies must handle `None` explicitly rather
than defaulting it — silently treating "no confidence signal" as "confidence 1.0" would hand those
arms an undeserved advantage in policy 4.

### Why arms are sync at the interface but async underneath

`rerank()` is synchronous, because the benchmark loop is simpler that way and four of five arms are
naturally sequential. Arm D's 50 concurrent Jev calls happen *inside* its implementation via
`asyncio.run()` over `AsyncTypeSafeClient`. The concurrency is an implementation detail of one arm,
not a contract imposed on all five.

Latency is therefore measured identically for every arm — wall-clock around `rerank()` — which is the
honest comparison. Arm D's parallelism shows up as a genuinely lower number rather than as a
different measurement method.

## 4. Caching

One decorator wraps every paid call. Key:

```
sha256(arm | model | dataset | query_id | doc_id | prompt_version)
```

Stored as sharded JSON under `cache/<arm>/<shard>/<key>.json`, tracked in **Git LFS**.

Consequences, all of which the project depends on:

- Iterating on metrics costs nothing. The expensive stage runs once.
- A killed run resumes exactly where it stopped — important across ~35k calls.
- `pytest` runs fully offline against the same files, so tests exercise real response shapes rather
  than hand-written mocks that drift from the API.
- **A reader clones, `git lfs pull`, and reproduces every published number with no API key.** This is
  the difference between claiming reproducibility and having it.

`prompt_version` in the key means a rubric change invalidates cleanly and *visibly* — old and new
responses coexist rather than one silently overwriting the other. That matters for the §9 rubric
ablation, which needs all three variants' responses simultaneously.

`--no-cache` bypasses reads (still writes) and exists solely for honest latency measurement.

## 5. Measurement is quarantined

`metrics.py` and `stats.py` import **nothing** from `rerankers.py`, `gating.py`, or any API client.
They take arrays and return numbers. This is deliberate:

- They are pure functions, so they can be tested against hand-computed values — a known nDCG, a
  synthetic perfectly-calibrated set that must score ECE ≈ 0, a fixed-seed bootstrap CI.
- No reranker bug can reach into a metric.
- A skeptical reader can audit the scoring logic without reading a line of API code. For a public
  benchmark this is the file people will actually scrutinise, and it should reward the scrutiny.

`CLAUDE.md` requires human review on every change to these two files for the same reason.

## 6. Concurrency and failure

Arm D: `asyncio.gather` over candidates, bounded by `asyncio.Semaphore(16)`. Arm C: same shape, a
lower bound. Retries use each SDK's built-in exponential backoff for 429/529 rather than a
hand-rolled loop.

**A failed call becomes recorded data, never a silent zero.** `Scored.meta["error"]` carries the
failure and the row is excluded from metrics with the exclusion counted and reported. A call that
errors and quietly scores 0.0 would depress that arm's numbers in a way nobody would ever spot in a
CSV — the single most dangerous bug class in this repo, and the reason `CLAUDE.md` forbids bare
`except: pass` in measurement paths.

## 7. Where the awkward parts live

**`unanswerable.py`** constructs the H5 evaluation sets: the gold-removed variant (delete a query's
relevant docs from the corpus, re-retrieve — ground truth exact by construction) and the cross-domain
variant. It writes derived pools to `data/unanswerable/` and never mutates the source corpus in
place; the deletion is applied to an index copy, so a bug here cannot corrupt the main benchmark.

**`robustness.py`** generates injected corpora for H6 under `data/adversarial/` — generated on
demand, `.gitignore`d, never committed. It reuses the ordinary pipeline for scoring; injection is a
transform on candidate text, not a separate code path, so the measured inflation is exactly what the
real pipeline would see.

**`config.py`** holds every tuned constant — composition weights, `τ`, `c_low`, semaphore size,
`prompt_version` — in one frozen dataclass. One file to audit, one file to diff when a number moves,
and one place for the dev/test discipline to be visibly enforced.

## 8. Layout

```
src/
  config.py         tuned constants, one frozen dataclass
  data.py           BEIR load, splits, seeded sampling
  embed.py          OpenAI embeddings + cache, cosine top-k
  cache.py          the hashing/storage decorator
  rerankers.py      Reranker protocol + arms A-E
  gating.py         the four selection policies
  unanswerable.py   H5 set construction
  robustness.py     H6 injection + rank-inflation
  metrics.py        IR + calibration      (pure; human review required)
  stats.py          bootstrap, randomisation test, Holm  (pure; human review required)
  report.py         CSV → plots → report.md
  bench.py          CLI orchestration
app.py              Streamlit explorer
tests/              offline, fixture-backed
cache/              LFS-tracked API responses
data/               datasets, embeddings, derived sets  (gitignored except manifests)
results/            CSVs, plots, report.md  (committed — they're the deliverable)
```

`rerankers.py` is the file most likely to outgrow itself. When it does, split it per-arm into
`src/arms/` rather than letting it become the place everything hides.

## 9. What a full run does

```
bench.py --datasets scifact,fiqa --arms all --policies all
  │
  ├─ load dataset, seeded query sample            (data.py)
  ├─ embed corpus + queries, cached               (embed.py)
  ├─ cosine top-50 → the shared candidate pool    (embed.py)
  │
  ├─ for each arm:  rerank every query            (rerankers.py, cached)
  ├─ for each (arm, policy): select chunks        (gating.py)
  │
  ├─ IR + calibration metrics per (arm, policy)   (metrics.py)
  ├─ bootstrap CIs; paired tests + Holm           (stats.py)
  ├─ H5 abstention, H6 robustness sweeps          (unanswerable.py, robustness.py)
  │
  └─ CSVs → plots → report.md with H1-H7 verdicts (report.py)
```

Warm cache: seconds. Cold: ~60–90 minutes, ~$20, resumable at any point.

## 10. Deliberately not built

- **No vector database.** A numpy dot product over ≤60k vectors takes milliseconds. Introducing FAISS
  or a server would add ops surface and change nothing measured.
- **No generation stage.** We measure which documents surface, not what is written from them. Answer
  quality needs a judge model and a second benchmark — out of scope per `BRD.md` §3.
- **No plugin system for rerankers.** Five arms, one file, one protocol. A registry for five known
  implementations is indirection with no payoff.
- **No web service.** Streamlit reads the same result files the CLI writes. Nothing is served.
