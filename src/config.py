"""Every tuned constant in the project. One file to audit, one file to diff."""

from dataclasses import dataclass, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Config:
    # --- fixed, never tuned ---
    POOL_SIZE: int = 50
    SEED: int = 42
    SEMAPHORE: int = 16
    N_BOOTSTRAP: int = 10_000
    N_PERMUTATIONS: int = 10_000
    N_BINS: int = 15

    # --- tuned on dev, then frozen (never on test) ---
    W_TOPICAL: float = 0.35
    W_ANSWERS: float = 0.65
    TAU: float = 0.5
    C_LOW: float = 0.5
    MASS_TARGET: float = 0.8
    FIXED_K: int = 5

    # --- stamps ---
    PROMPT_VERSION: str = "v1"
    EMBED_MODEL: str = "text-embedding-3-small"
    LLM_MODEL: str = "claude-haiku-4-5-20251001"
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
