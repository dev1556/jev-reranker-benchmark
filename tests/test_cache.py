from src.cache import ResponseCache, cache_key


def test_key_is_stable_across_calls() -> None:
    args = ("jev", "jev-latest", "scifact", "q1", "d1", "v1")
    assert cache_key(*args) == cache_key(*args)


def test_prompt_version_changes_the_key() -> None:
    """A rubric change must invalidate visibly — old and new responses coexist,
    which the Task 18 rubric ablation depends on."""
    base = ("jev", "jev-latest", "scifact", "q1", "d1")
    assert cache_key(*base, "v1") != cache_key(*base, "v2")


def test_every_field_participates_in_the_key() -> None:
    base = ["jev", "jev-latest", "scifact", "q1", "d1", "v1"]
    keys = {cache_key(*base)}
    for i in range(len(base)):
        mutated = list(base)
        mutated[i] = "CHANGED"
        keys.add(cache_key(*mutated))
    assert len(keys) == len(base) + 1


def test_roundtrip(tmp_path) -> None:
    c = ResponseCache(tmp_path)
    c.put("abc123", {"score": 0.7, "probabilities": {"0": 0.3, "1": 0.7}})
    assert c.get("abc123")["score"] == 0.7


def test_miss_returns_none(tmp_path) -> None:
    assert ResponseCache(tmp_path).get("nope") is None


def test_survives_a_new_instance(tmp_path) -> None:
    ResponseCache(tmp_path).put("k", {"v": 1})
    assert ResponseCache(tmp_path).get("k") == {"v": 1}


def test_disabled_cache_still_writes_but_never_reads(tmp_path) -> None:
    """--no-cache exists for honest latency measurement. It must not throw away
    responses that were already paid for."""
    c = ResponseCache(tmp_path, enabled=False)
    c.put("k", {"v": 1})
    assert c.get("k") is None
    assert ResponseCache(tmp_path, enabled=True).get("k") == {"v": 1}


def test_sharding_limits_files_per_directory(tmp_path) -> None:
    c = ResponseCache(tmp_path)
    for i in range(200):
        c.put(cache_key("jev", "m", "scifact", f"q{i}", "d1", "v1"), {"i": i}, arm="jev")
    dirs = [p for p in (tmp_path / "jev").iterdir() if p.is_dir()]
    assert len(dirs) > 1


def test_stats_track_hits_and_misses(tmp_path) -> None:
    c = ResponseCache(tmp_path)
    c.put("k", {"v": 1})
    c.get("k")
    c.get("missing")
    s = c.stats()
    assert (s.hits, s.misses, s.writes) == (1, 1, 1)


def test_corrupt_entry_raises_rather_than_returning_none(tmp_path) -> None:
    """A truncated LFS pointer or partial write must not look like a cache miss —
    that would silently trigger a $12 cold rerun."""
    import pytest

    c = ResponseCache(tmp_path)
    c.put("k", {"v": 1})
    path = next(tmp_path.rglob("k.json"))
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="corrupt"):
        c.get("k")
