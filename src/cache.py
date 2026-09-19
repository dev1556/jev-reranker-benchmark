"""Content-addressed on-disk cache for paid API responses.

This is what makes the project's reproducibility claim literally true: a reader
clones, runs `git lfs pull`, and regenerates every published number with no API
key and no spend.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    writes: int = 0


def cache_key(
    arm: str, model: str, dataset: str, query_id: str, doc_id: str, prompt_version: str
) -> str:
    """SHA-256 over every field that could change a response.

    prompt_version is included so a rubric edit invalidates visibly rather than
    silently reusing answers produced by different wording.
    """
    raw = "|".join((arm, model, dataset, query_id, doc_id, prompt_version))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class ResponseCache:
    """Sharded JSON store. `enabled=False` disables reads but keeps writes, so
    a --no-cache latency run still banks the responses it paid for."""

    def __init__(self, root: Path, enabled: bool = True) -> None:
        self.root = Path(root)
        self.enabled = enabled
        self._stats = CacheStats()

    def _path(self, key: str, arm: str = "misc") -> Path:
        return self.root / arm / key[:2] / f"{key}.json"

    def _find(self, key: str) -> Path | None:
        hits = list(self.root.rglob(f"{key}.json"))
        return hits[0] if hits else None

    def get(self, key: str) -> dict | None:
        if not self.enabled:
            return None
        path = self._find(key)
        if path is None:
            self._stats.misses += 1
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"corrupt cache entry at {path} — if this is an LFS pointer, "
                f"run `git lfs pull`. Treating it as a miss would trigger an "
                f"expensive cold rerun."
            ) from exc
        self._stats.hits += 1
        return value

    def put(self, key: str, value: dict, arm: str = "misc") -> None:
        path = self._path(key, arm)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(value), encoding="utf-8")
        tmp.replace(path)  # atomic: a killed run never leaves a partial file
        self._stats.writes += 1

    def stats(self) -> CacheStats:
        return self._stats
