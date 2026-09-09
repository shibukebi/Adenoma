from pathlib import Path

import pytest

from challenge_review.wsi import ISyntaxWorkerManager, _overview_target_level
from challenge_review.wsi_cache import PersistentWsiCache


def test_persistent_cache_survives_restart_and_protects_pinned_entries(tmp_path):
    cache = PersistentWsiCache(tmp_path / "cache", max_bytes=100, trim_bytes=60)
    source = "/readonly/example.isyntax"
    pinned = cache.thumbnail_path("priority", source)
    old_tile = cache.tile_path("ordinary", source, 1, 0, 0)
    new_tile = cache.tile_path("ordinary", source, 1, 1, 0)

    cache.write_bytes(pinned, b"p" * 30, "priority", source, "thumbnail", pinned=True)
    cache.write_bytes(old_tile, b"a" * 40, "ordinary", source, "tile")
    cache.write_bytes(new_tile, b"b" * 40, "ordinary", source, "tile")

    reopened = PersistentWsiCache(tmp_path / "cache", max_bytes=100, trim_bytes=60)
    assert reopened.read_bytes(pinned) == b"p" * 30
    assert reopened.stats()["bytes"] <= 60
    assert not old_tile.exists()
    assert not new_tile.exists()


def test_warm_queue_is_persistent_coalesced_and_resumable(tmp_path):
    cache = PersistentWsiCache(tmp_path / "cache")
    source = "/readonly/example.isyntax"
    assert cache.enqueue_warm("slide", source, priority=20) == "queued"
    assert cache.enqueue_warm("slide", source, priority=0, pinned=True) == "queued"
    job = cache.claim_next_warm()
    assert job["slide_id"] == "slide"
    assert job["priority"] == 0
    assert job["pinned"] == 1
    assert cache.warm_status("slide", source) == "warming"

    cache.reset_interrupted_warms()
    assert cache.warm_status("slide", source) == "queued"


def test_overview_level_is_limited_to_4096_pixels():
    metadata = {
        "width": 200_000,
        "height": 100_000,
        "level_count": 19,
        "tile_size": 254,
    }
    target = _overview_target_level(metadata)
    scale = 2 ** (metadata["level_count"] - 1 - target)
    assert max(metadata["width"], metadata["height"]) / scale <= 4096
    assert max(metadata["width"], metadata["height"]) / (scale / 2) > 4096


def test_isyntax_worker_restarts_after_native_process_failure():
    manager = ISyntaxWorkerManager(timeout=10)
    try:
        with pytest.raises(RuntimeError, match="restarted"):
            manager.call("_test_crash", "unused")
        assert manager.status()["restarts"] == 1
        assert manager.call("_test_echo", "unused", "alive") == "alive"
    finally:
        manager.close()
