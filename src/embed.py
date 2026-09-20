"""First-stage retrieval: OpenAI embeddings and the shared cosine top-k pool.

Every arm reranks the pool this module produces. If the pools diverge, the
benchmark stops measuring reranking and starts measuring retrieval, so
`assert_pools_identical` is called at runtime rather than trusted.
"""

import hashlib
import os
from collections.abc import Sequence
from pathlib import Path
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


class EmbeddingsAPI(Protocol):
    def create(self, model: str, input: list[str]) -> object: ...


class OpenAIClient(Protocol):
    """Duck-typed contract for whatever `OpenAIEmbedder.client` holds: the
    real `openai.OpenAI()` in production, a counting fake in tests."""

    embeddings: EmbeddingsAPI


def _embedding_cache_key(model: str, texts: Sequence[str]) -> str:
    """SHA-256 over the model id and every text, in order.

    Any change to the model or the corpus must miss the cache; a reordering
    must not silently reuse a mismatched matrix, so position is hashed in via
    a separator byte that cannot appear inside a UTF-8 string.
    """
    digest = hashlib.sha256()
    digest.update(model.encode("utf-8"))
    for text in texts:
        digest.update(b"\x00")
        digest.update(text.encode("utf-8"))
    return digest.hexdigest()


def _load_embedding_cache(cache_dir: Path, key: str) -> np.ndarray | None:
    path = cache_dir / f"{key}.npz"
    if not path.exists():
        return None
    with np.load(path) as data:
        return np.asarray(data["vectors"])


def _save_embedding_cache(cache_dir: Path, key: str, vectors: np.ndarray) -> None:
    """Write atomically, matching `src/cache.py`: a temp file in the same
    directory, then an atomic replace, so a killed run never leaves a
    truncated `.npz` a later run trusts."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{key}.npz"
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as f:
        np.savez(f, vectors=vectors)
    tmp.replace(path)


class OpenAIEmbedder:
    """text-embedding-3-small, batched, cached on disk by (model, texts)."""

    def __init__(
        self, cfg: Config, batch_size: int = 256, client: OpenAIClient | None = None
    ) -> None:
        self.cfg = cfg
        self.batch_size = batch_size
        self.cache_dir = cfg.DATA_DIR / "embeddings"
        if client is None:
            from openai import OpenAI

            key = os.environ.get("OPENAI_API_KEY")
            if not key:
                raise RuntimeError(
                    "OPENAI_API_KEY is not set. It is required for first-stage "
                    "retrieval, which every arm depends on."
                )
            client = OpenAI(api_key=key)
        self.client = client

    def embed(self, texts: list[str]) -> np.ndarray:
        key = _embedding_cache_key(self.cfg.EMBED_MODEL, texts)
        cached = _load_embedding_cache(self.cache_dir, key)
        if cached is not None:
            return cached
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            resp = self.client.embeddings.create(model=self.cfg.EMBED_MODEL, input=batch)
            vectors.extend(d.embedding for d in resp.data)
        result = np.array(vectors, dtype=np.float32)
        _save_embedding_cache(self.cache_dir, key, result)
        return result


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


def build_pools_and_sims(
    corpus: Corpus, queries: Sequence[Query], embedder: Embedder, cfg: Config
) -> tuple[dict[str, CandidatePool], dict[str, dict[str, float]]]:
    """Embed corpus and queries, build one shared top-k pool per query, and
    return the cosine similarity of every pooled (query, doc) pair.

    These are the same similarity values used to pick the top-k, not a
    recomputation — arm A (`CosineReranker`) needs the honest number rather
    than a placeholder.
    """
    doc_ids = sorted(corpus.docs)
    doc_matrix = embedder.embed([corpus.docs[d].text for d in doc_ids])
    query_matrix = embedder.embed([q.text for q in queries])
    norm_docs = _normalise(doc_matrix)
    pools: dict[str, CandidatePool] = {}
    sims: dict[str, dict[str, float]] = {}
    for q, qvec in zip(queries, query_matrix, strict=True):
        query_sims = norm_docs @ _normalise(qvec)
        order = sorted(range(len(doc_ids)), key=lambda i: (-query_sims[i], doc_ids[i]))
        top = order[: cfg.POOL_SIZE]
        pools[q.query_id] = CandidatePool(q, tuple(corpus.docs[doc_ids[i]] for i in top))
        sims[q.query_id] = {doc_ids[i]: float(query_sims[i]) for i in top}
    return pools, sims


def build_pools(
    corpus: Corpus, queries: Sequence[Query], embedder: Embedder, cfg: Config
) -> dict[str, CandidatePool]:
    """Embed corpus and queries, then build one shared top-k pool per query.

    Thin wrapper over `build_pools_and_sims` for callers that only need the
    pools.
    """
    pools, _sims = build_pools_and_sims(corpus, queries, embedder, cfg)
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
