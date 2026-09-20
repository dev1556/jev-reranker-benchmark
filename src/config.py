"""Every tuned constant in the project. One file to audit, one file to diff."""

from dataclasses import dataclass, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Config:
    # --- fixed, never tuned ---
    POOL_SIZE: int = 50
    SEED: int = 42
    # FiQA has 648 test queries; BRD §4.1 samples 300 of them, seeded, and commits
    # the sampled ids. Confirmed by the user 2026-09-20. SciFact uses all 300 of
    # its test queries, so both datasets carry the same query budget and neither
    # dominates a pooled result. Not a tuned value — a scope decision.
    FIQA_N_QUERIES: int = 300
    # BRD §8: per query, 5 genuinely irrelevant chunks are injected, and the
    # damage is read as the nDCG@10 drop — the same k as the headline table, so
    # the robustness cost is comparable to the accuracy numbers. Fixed by the
    # spec, not tuned.
    N_INJECTED_PER_QUERY: int = 5
    NDCG_K: int = 10
    SEMAPHORE: int = 16
    N_BOOTSTRAP: int = 10_000
    N_PERMUTATIONS: int = 10_000
    N_BINS: int = 15

    # --- tuned on dev, then frozen (never on test) ---
    W_TOPICAL: float = 0.35
    W_ANSWERS: float = 0.65
    TAU: float = 0.5
    C_LOW: float = 0.5
    TAU_WIDE: float = 0.3  # bar when the arm is unsure: buy more context
    TAU_NARROW: float = 0.7  # bar when it is confident AND the top P is high
    MASS_TARGET: float = 0.8
    FIXED_K: int = 5

    # --- stamps ---
    PROMPT_VERSION: str = "v1"
    EMBED_MODEL: str = "text-embedding-3-small"
    LLM_MODEL: str = "claude-haiku-4-5"
    JEV_MODEL: str = "jev-latest"
    CROSS_ENCODER: str = "BAAI/bge-reranker-base"

    # --- paths ---
    DATA_DIR: Path = ROOT / "data"
    CACHE_DIR: Path = ROOT / "cache"
    RESULTS_DIR: Path = ROOT / "results"

    def tuned(self, **overrides: float) -> "Config":
        """Return a copy with tuned values overridden. Used by dev-split tuning only."""
        return replace(self, **overrides)


CONFIG = Config()
