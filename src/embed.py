"""First-stage retrieval: OpenAI embeddings and the shared cosine top-k pool.

Every arm reranks the pool this module produces. If the pools diverge, the
benchmark stops measuring reranking and starts measuring retrieval, so
`assert_pools_identical` is called at runtime rather than trusted.
"""

import hashlib
import os
from collections.abc import Sequence
from typing import Protocol

import numpy as np

from src.config import Config
from src.types import CandidatePool, Corpus, Query


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...


class FakeEmbedder:
    """Deterministic hash-based vectors. Test-only; no network, no key.

    Seeded from a blake2b digest rather than `hash()`, which Python randomises
    per process — that would make every fixture pool shift between runs.
    """

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.empty((len(texts), self.dim))
        for i, t in enumerate(texts):
            digest = hashlib.blake2b(t.encode(), digest_size=4).digest()
            seed = int.from_bytes(digest, "big")
            out[i] = np.random.default_rng(seed).normal(size=self.dim)
        return out


class OpenAIEmbedder:
    """text-embedding-3-small, batched."""

    def __init__(self, cfg: Config, batch_size: int = 256) -> None:
        from openai import OpenAI

        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. It is required for first-stage "
                "retrieval, which every arm depends on."
            )
        self.client = OpenAI(api_key=key)
        self.cfg = cfg
        self.batch_size = batch_size

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            resp = self.client.embeddings.create(model=self.cfg.EMBED_MODEL, input=batch)
            vectors.extend(d.embedding for d in resp.data)
        return np.array(vectors, dtype=np.float32)


def _normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return matrix / np.where(norms == 0, 1.0, norms)  # zero vectors -> zero sim, not NaN


def cosine_top_k(
    query_vec: np.ndarray, doc_matrix: np.ndarray, doc_ids: Sequence[str], k: int
) -> list[str]:
    """Top-k document ids by cosine similarity, cos(a,b) = a·b / (|a||b|).

    Ties are broken by doc_id so the pool is reproducible; see NFR-5.
    """
    sims = _normalise(doc_matrix) @ _normalise(query_vec)
    order = sorted(range(len(doc_ids)), key=lambda i: (-sims[i], doc_ids[i]))
    return [doc_ids[i] for i in order[:k]]


def build_pools(
    corpus: Corpus, queries: Sequence[Query], embedder: Embedder, cfg: Config
) -> dict[str, CandidatePool]:
    """Embed corpus and queries, then build one shared top-k pool per query."""
    doc_ids = sorted(corpus.docs)
    doc_matrix = embedder.embed([corpus.docs[d].text for d in doc_ids])
    query_matrix = embedder.embed([q.text for q in queries])
    pools: dict[str, CandidatePool] = {}
    for q, qvec in zip(queries, query_matrix, strict=True):
        top = cosine_top_k(qvec, doc_matrix, doc_ids, cfg.POOL_SIZE)
        pools[q.query_id] = CandidatePool(q, tuple(corpus.docs[d] for d in top))
    return pools


def assert_pools_identical(pools_by_arm: dict[str, dict[str, CandidatePool]]) -> None:
    """FR-2 runtime check: all arms must see the same candidates, same order."""
    arms = list(pools_by_arm)
    if len(arms) < 2:
        return
    reference = {q: [d.doc_id for d in p.docs] for q, p in pools_by_arm[arms[0]].items()}
    for arm in arms[1:]:
        other = {q: [d.doc_id for d in p.docs] for q, p in pools_by_arm[arm].items()}
        if other != reference:
            raise AssertionError(
                f"candidate pools for '{arm}' are not identical to '{arms[0]}'. "
                f"Every arm must rerank the same candidates or the comparison "
                f"measures retrieval, not reranking."
            )
