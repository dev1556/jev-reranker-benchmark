# jev-reranker-benchmark

<!-- Headline results — filled in with real numbers by Task 19. -->
**Results: TBD.**

## What this is

A benchmark measuring whether [TypeSafe Jev](https://typesafe.ai)'s `Score` primitive produces
**calibrated** relevance probabilities when used as a reranker over a fixed first-stage retrieval
pool (BEIR datasets, OpenAI `text-embedding-3-small` embeddings, cosine top-50). Every arm reranks
the identical candidate set — the comparison is purely about reordering and thresholding, not
recall.

## The five arms

| # | Arm | Mechanism | Role |
|---|-----|-----------|------|
| A | Cosine-only | identity reorder on embedding similarity | the floor / strawman |
| B | Cross-encoder | `BAAI/bge-reranker-base`, local CPU | the honest opponent — what teams actually ship |
| C | LLM reranker | Claude Haiku 4.5, pointwise 0-3 rubric | the expensive rival; its pseudo-probability is the artifact being critiqued |
| D | **Jev** | `Score` primitive, probability mass over rubric levels | the subject of this benchmark |
| E | Cosine + Platt | logistic recalibration of A, fitted on dev | the control that makes the calibration hypothesis falsifiable |

## Quickstart

```bash
uv sync
cp .env.example .env   # fill in TYPESAFE_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY
make smoke              # small dev-split run across all arms and policies
```

Arm B (cross-encoder) requires `torch` + `sentence-transformers`; it is independently skippable if
that install fails on your platform — the other four arms do not depend on it.

## Reproduce from cache (no API keys needed)

All arm responses are cached and tracked with Git LFS, so the full result set can be reproduced
without calling any paid API:

```bash
git lfs pull
make report
```

## Development

```bash
make test   # uv run pytest
make lint   # ruff check + format --check
make fmt    # ruff format .
```
