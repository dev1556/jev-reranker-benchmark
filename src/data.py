"""BEIR dataset loading, seeded sampling, and the dev/test split guard.

The guard is the mechanism behind non-negotiable #1: code running inside
`tuning_context()` cannot read the test split at all. Discipline that depends
on remembering is discipline that eventually fails.

Scope of the guard, established by execution rather than assumption:
- It PROPAGATES into `asyncio` tasks created inside the context, because
  asyncio copies the current `contextvars.Context` into new tasks — this is
  the case that matters here, since the benchmark fires ~50 concurrent async
  calls per query.
- It does NOT propagate into a raw `threading.Thread` started inside the
  context, because Python does not copy ContextVar state across OS threads;
  code in such a thread would see the default and be allowed through. No
  thread is used in this project today — this is recorded so a future change
  to threads doesn't silently disable the protection.
- It only protects callers that go through `get_split`. `Corpus.queries` is a
  plain public dict, so any code holding a `Corpus` can read test-split
  queries directly and bypass the guard entirely. It's a tripwire for honest
  mistakes, not a security boundary.
"""

import json
import random
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Literal

from src.config import Config
from src.types import Corpus, Doc, Qrels, Query

_TUNING: ContextVar[bool] = ContextVar("tuning", default=False)


class TestSplitAccessError(RuntimeError):
    """Raised when tuning code reaches for the test split."""


@contextmanager
def tuning_context() -> Iterator[None]:
    """Mark a region as tuning. Test-split access inside it raises."""
    token = _TUNING.set(True)
    try:
        yield
    finally:
        _TUNING.reset(token)


_HF_NAMES = {"scifact": "BeIR/scifact", "fiqa": "BeIR/fiqa"}


def load_corpus(name: Literal["scifact", "fiqa"], cfg: Config) -> Corpus:
    """Load a BEIR dataset from HuggingFace, cached under cfg.DATA_DIR."""
    from datasets import load_dataset

    hf = _HF_NAMES[name]
    cache = str(cfg.DATA_DIR / "hf")
    corpus_ds = load_dataset(hf, "corpus", cache_dir=cache)["corpus"]
    queries_ds = load_dataset(hf, "queries", cache_dir=cache)["queries"]
    qrels_ds = load_dataset(f"{hf}-qrels", cache_dir=cache)["test"]

    docs = {r["_id"]: Doc(r["_id"], r["text"], r.get("title", "")) for r in corpus_ds}
    queries = {r["_id"]: Query(r["_id"], r["text"]) for r in queries_ds}
    qrels: Qrels = {}
    for r in qrels_ds:
        qrels.setdefault(str(r["query-id"]), {})[str(r["corpus-id"])] = int(r["score"])
    # keep only queries that have judgements — an unjudged query is unscoreable
    queries = {qid: q for qid, q in queries.items() if qid in qrels}
    return Corpus(name=name, docs=docs, queries=queries, qrels=qrels)


def sample_query_ids(query_ids: Sequence[str], n: int, seed: int) -> list[str]:
    """Deterministic sample. Sorts first so input ordering cannot affect the result."""
    pool = sorted(query_ids)
    if n >= len(pool):
        return pool
    return sorted(random.Random(seed).sample(pool, n))


def split_query_ids(
    query_ids: Sequence[str], seed: int, dev_frac: float = 0.3
) -> tuple[list[str], list[str]]:
    """Disjoint, exhaustive, reproducible dev/test partition."""
    pool = sorted(query_ids)
    rng = random.Random(seed)
    shuffled = pool[:]
    rng.shuffle(shuffled)
    cut = int(len(shuffled) * dev_frac)
    return sorted(shuffled[:cut]), sorted(shuffled[cut:])


def get_split(corpus: Corpus, split: Literal["dev", "test"], cfg: Config) -> list[Query]:
    """The only supported way to obtain queries. Guards test-split access."""
    if split == "test" and _TUNING.get():
        raise TestSplitAccessError(
            "test split requested inside tuning_context(). Tune on dev only "
            "(CLAUDE.md non-negotiable #1)."
        )
    dev_ids, test_ids = split_query_ids(list(corpus.queries), cfg.SEED)
    ids = dev_ids if split == "dev" else test_ids
    return [corpus.queries[qid] for qid in ids]


def write_manifest(name: str, split: str, ids: Sequence[str], cfg: Config) -> None:
    """Commit the exact query IDs used, so the sample is reproducible by a reader."""
    path = cfg.DATA_DIR / "manifests" / f"{name}-{split}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sorted(ids), indent=2), encoding="utf-8")
