"""H5 evaluation sets, built with zero manual labelling.

Gold-removed is the realistic failure: the retriever returns topically adjacent
near-misses and the generator confabulates from them. Cross-domain is the easy
case — an arm that fails it fails badly.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from src.config import Config
from src.embed import Embedder, build_pools, cosine_top_k
from src.gating import Selection
from src.metrics import auroc
from src.types import CandidatePool, Corpus, Query


@dataclass(frozen=True)
class AbstentionReport:
    rate_unanswerable: float
    false_abstention_rate: float
    auroc: float


def build_gold_removed(
    corpus: Corpus, queries: Sequence[Query], embedder: Embedder, cfg: Config
) -> dict[str, CandidatePool]:
    """Delete each query's relevant documents from an index COPY, then re-retrieve.

    Every document judged relevant for a query (its full qrels row, not just the
    top-graded one) is removed before retrieval, so the resulting pool is
    unanswerable by construction — a pool that still holds one relevant document
    is answerable, and H5's numbers would no longer mean what they claim to.

    The corpus is embedded ONCE and each query's gold documents are dropped by
    index before the top-k. The obvious version — strip the corpus, call
    `build_pools` for one query, repeat — re-embeds every passage per query:
    300 FiQA queries over 57,638 passages is ~17M embedding calls instead of
    57,638, and `OpenAIEmbedder` holds no cache. Same cosine, same doc_id
    tiebreak, same pools; ~300x less spend.

    Nothing is mutated: the corpus is only read, and the pool is built from
    filtered ids.
    """
    doc_ids = sorted(corpus.docs)  # sorted, so tie-breaking matches build_pools
    doc_matrix = embedder.embed([corpus.docs[d].text for d in doc_ids])
    query_matrix = embedder.embed([q.text for q in queries])
    position = {doc_id: i for i, doc_id in enumerate(doc_ids)}

    pools: dict[str, CandidatePool] = {}
    for query, query_vec in zip(queries, query_matrix, strict=True):
        gold = set(corpus.qrels.get(query.query_id, {}))
        kept = [d for d in doc_ids if d not in gold]
        if not kept:
            pools[query.query_id] = CandidatePool(query, ())
            continue
        top = cosine_top_k(query_vec, doc_matrix[[position[d] for d in kept]], kept, cfg.POOL_SIZE)
        pools[query.query_id] = CandidatePool(query, tuple(corpus.docs[d] for d in top))
    return pools


def build_cross_domain(
    doc_corpus: Corpus,
    queries: Sequence[Query],
    embedder: Embedder,
    cfg: Config,
) -> dict[str, CandidatePool]:
    """Score one dataset's queries against the other's corpus. Nothing is relevant.

    `queries` come from one dataset and `doc_corpus` from the other. The source
    corpus is not a parameter because the queries are already passed in and
    nothing here reads the rest of it.

    `doc_corpus` is whatever pool of documents they are scored against — the
    full corpus or a sampled subset (e.g. FiQA's 300-query sample) both work,
    since retrieval here only ever looks at `doc_corpus.docs`.
    """
    return build_pools(doc_corpus, queries, embedder, cfg)


def abstention_report(
    selections: dict[str, Selection],
    answerable_ids: Sequence[str],
    unanswerable_ids: Sequence[str],
) -> AbstentionReport:
    """Refusal rate, false-refusal rate, and the AUROC over the pooled set.

    rate_unanswerable = mean(abstained) over the UNANSWERABLE population — the
    fraction of genuinely unanswerable queries the arm correctly refused.

    false_abstention_rate = mean(abstained) over the ANSWERABLE population — the
    fraction of queries with a real answer that the arm wrongly refused.

    Refusal rate alone is meaningless — an arm that always abstains scores a
    perfect 1.0 on it — so the two are always computed and returned together.
    auroc treats `abstained` as the score and "is unanswerable" as the positive
    label, via the rank-based estimator in `src.metrics.auroc`.

    Either population missing from `selections` entirely yields NaN for its
    rate (and for auroc) rather than raising or dividing by zero; only an empty
    *requested* population (no unanswerable/answerable ids at all) is an error,
    since that means the caller built the wrong query lists.
    """
    if not answerable_ids or not unanswerable_ids:
        raise ValueError("abstention_report needs both answerable and unanswerable queries")
    un = [selections[q].abstained for q in unanswerable_ids if q in selections]
    an = [selections[q].abstained for q in answerable_ids if q in selections]
    ids = list(unanswerable_ids) + list(answerable_ids)
    scores = np.array([float(selections[q].abstained) for q in ids if q in selections])
    labels = np.array([1] * len(un) + [0] * len(an))
    return AbstentionReport(
        rate_unanswerable=float(np.mean(un)) if un else float("nan"),
        false_abstention_rate=float(np.mean(an)) if an else float("nan"),
        auroc=auroc(scores, labels),
    )
