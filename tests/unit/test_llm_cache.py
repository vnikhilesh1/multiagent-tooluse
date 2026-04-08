"""Unit tests for LLM cache."""

import tempfile
import threading
from datetime import datetime
from pathlib import Path

import pytest

from src.llm import LLMCache, LLMResponse, CacheEntry, CacheStats


class TestCacheEntry:
    """Tests for CacheEntry model."""

    def test_create_entry(self):
        """Test creating a cache entry."""
        entry = CacheEntry(
            content="Hello!",
            model="claude-sonnet-4-20250514",
            usage={"input_tokens": 10, "output_tokens": 5},
            stop_reason="end_turn",
            created_at=datetime.utcnow(),
            prompt_hash="abc123",
        )
        assert entry.content == "Hello!"
        assert entry.model == "claude-sonnet-4-20250514"
        assert entry.hit_count == 0

    def test_entry_serialization(self):
        """Test entry can be serialized to JSON."""
        entry = CacheEntry(
            content="Test",
            model="test-model",
            usage={"input_tokens": 1, "output_tokens": 1},
            created_at=datetime.utcnow(),
            prompt_hash="hash123",
        )
        json_str = entry.model_dump_json()
        assert "Test" in json_str
        assert "test-model" in json_str

    def test_entry_defaults(self):
        """Test entry has correct defaults."""
        entry = CacheEntry(
            content="Test",
            model="test-model",
            usage={},
            created_at=datetime.utcnow(),
            prompt_hash="hash123",
        )
        assert entry.hit_count == 0
        assert entry.last_accessed is None
        assert entry.stop_reason is None
        assert entry.raw_response is None


class TestCacheStats:
    """Tests for CacheStats model."""

    def test_create_stats(self):
        """Test creating stats with defaults."""
        stats = CacheStats()
        assert stats.memory_hits == 0
        assert stats.disk_hits == 0
        assert stats.misses == 0
        assert stats.memory_size == 0
        assert stats.disk_entries == 0

    def test_stats_with_values(self):
        """Test creating stats with values."""
        stats = CacheStats(
            memory_hits=10,
            disk_hits=5,
            misses=3,
            memory_size=100,
            disk_entries=50,
        )
        assert stats.memory_hits == 10
        assert stats.disk_hits == 5


class TestLLMCacheInit:
    """Tests for LLMCache initialization."""

    def test_init_creates_directory(self):
        """Test cache creates directory on init."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "test_cache"
            cache = LLMCache(cache_dir=cache_dir)
            assert cache_dir.exists()

    def test_init_disabled(self):
        """Test cache can be initialized disabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = LLMCache(cache_dir=Path(tmpdir), enabled=False)
            assert not cache.enabled

    def test_init_with_max_entries(self):
        """Test cache accepts max_memory_entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = LLMCache(cache_dir=Path(tmpdir), max_memory_entries=100)
            assert cache._max_memory_entries == 100

    def test_enabled_property_setter(self):
        """Test enabled property can be changed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = LLMCache(cache_dir=Path(tmpdir), enabled=False)
            assert not cache.enabled
            cache.enabled = True
            assert cache.enabled


class TestComputeHash:
    """Tests for hash computation."""

    def test_same_input_same_hash(self):
        """Test same inputs produce same hash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = LLMCache(cache_dir=Path(tmpdir))
            hash1 = cache._compute_hash("Hello", "model-1")
            hash2 = cache._compute_hash("Hello", "model-1")
            assert hash1 == hash2

    def test_different_prompt_different_hash(self):
        """Test different prompts produce different hashes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = LLMCache(cache_dir=Path(tmpdir))
            hash1 = cache._compute_hash("Hello", "model-1")
            hash2 = cache._compute_hash("Goodbye", "model-1")
            assert hash1 != hash2

    def test_different_model_different_hash(self):
        """Test different models produce different hashes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = LLMCache(cache_dir=Path(tmpdir))
            hash1 = cache._compute_hash("Hello", "model-1")
            hash2 = cache._compute_hash("Hello", "model-2")
            assert hash1 != hash2

    def test_hash_includes_system(self):
        """Test system prompt affects hash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = LLMCache(cache_dir=Path(tmpdir))
            hash1 = cache._compute_hash("Hello", "model-1", system="Be helpful")
            hash2 = cache._compute_hash("Hello", "model-1", system="Be brief")
            assert hash1 != hash2

    def test_hash_includes_temperature(self):
        """Test temperature affects hash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = LLMCache(cache_dir=Path(tmpdir))
            hash1 = cache._compute_hash("Hello", "model-1", temperature=0.0)
            hash2 = cache._compute_hash("Hello", "model-1", temperature=1.0)
            assert hash1 != hash2

    def test_hash_is_md5_format(self):
        """Test hash is 32-character hex string."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = LLMCache(cache_dir=Path(tmpdir))
            hash_val = cache._compute_hash("Hello", "model-1")
            assert len(hash_val) == 32
            assert all(c in "0123456789abcdef" for c in hash_val)

    def test_hash_includes_tools(self):
        """Test tools affect hash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = LLMCache(cache_dir=Path(tmpdir))
            tools1 = [{"name": "tool1", "description": "desc1"}]
            tools2 = [{"name": "tool2", "description": "desc2"}]
            hash1 = cache._compute_hash("Hello", "model-1", tools=tools1)
            hash2 = cache._compute_hash("Hello", "model-1", tools=tools2)
            assert hash1 != hash2

    def test_hash_includes_messages(self):
        """Test messages affect hash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = LLMCache(cache_dir=Path(tmpdir))
            messages1 = [{"role": "user", "content": "Hi"}]
            messages2 = [{"role": "user", "content": "Hello"}]
            hash1 = cache._compute_hash("", "model-1", messages=messages1)
            hash2 = cache._compute_hash("", "model-1", messages=messages2)
            assert hash1 != hash2


class TestGetSet:
    """Tests for get/set operations."""

    def test_set_and_get(self):
        """Test basic set and get."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = LLMCache(cache_dir=Path(tmpdir))

            response = LLMResponse(
                content="Hello!",
                model="test-model",
                usage={"input_tokens": 10, "output_tokens": 5},
            )

            cache.set(prompt="Hi", model="test-model", response=response)
            cached = cache.get(prompt="Hi", model="test-model")

            assert cached is not None
            assert cached.content == "Hello!"

    def test_set_returns_cache_key(self):
        """Test set returns the cache key."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = LLMCache(cache_dir=Path(tmpdir))

            response = LLMResponse(
                content="Hello!",
                model="test-model",
                usage={},
            )

            cache_key = cache.set(prompt="Hi", model="test-model", response=response)
            assert isinstance(cache_key, str)
            assert len(cache_key) == 32

    def test_get_miss(self):
        """Test get returns None for missing entry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = LLMCache(cache_dir=Path(tmpdir))
            cached = cache.get(prompt="Unknown", model="test-model")
            assert cached is None

    def test_disabled_cache_returns_none(self):
        """Test disabled cache always returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = LLMCache(cache_dir=Path(tmpdir), enabled=False)

            response = LLMResponse(
                content="Hello!",
                model="test-model",
                usage={},
            )

            cache.set(prompt="Hi", model="test-model", response=response)
            cached = cache.get(prompt="Hi", model="test-model")

            assert cached is None

    def test_disk_persistence(self):
        """Test cache persists to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)

            # Create cache and add entry
            cache1 = LLMCache(cache_dir=cache_dir)
            response = LLMResponse(
                content="Persisted!",
                model="test-model",
                usage={},
            )
            cache1.set(prompt="Test", model="test-model", response=response)

            # Create new cache instance (simulating restart)
            cache2 = LLMCache(cache_dir=cache_dir)
            cached = cache2.get(prompt="Test", model="test-model")

            assert cached is not None
            assert cached.content == "Persisted!"

    def test_get_increments_hit_count(self):
        """Test get increments hit count."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = LLMCache(cache_dir=Path(tmpdir))

            response = LLMResponse(
                content="Test",
                model="test-model",
                usage={},
            )

            cache.set(prompt="Test", model="test-model", response=response)

            cached1 = cache.get(prompt="Test", model="test-model")
            assert cached1.hit_count == 1

            cached2 = cache.get(prompt="Test", model="test-model")
            assert cached2.hit_count == 2


class TestInvalidate:
    """Tests for cache invalidation."""

    def test_invalidate_removes_entry(self):
        """Test invalidate removes entry from both caches."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = LLMCache(cache_dir=Path(tmpdir))

            response = LLMResponse(
                content="To delete",
                model="test-model",
                usage={},
            )

            cache_key = cache.set(
                prompt="Delete me", model="test-model", response=response
            )

            # Verify it exists
            assert cache.get(prompt="Delete me", model="test-model") is not None

            # Invalidate
            result = cache.invalidate(cache_key)
            assert result is True

            # Verify it's gone
            assert cache.get(prompt="Delete me", model="test-model") is None

    def test_invalidate_nonexistent(self):
        """Test invalidate returns False for missing entry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = LLMCache(cache_dir=Path(tmpdir))
            result = cache.invalidate("nonexistent_hash")
            assert result is False

    def test_invalidate_removes_from_disk(self):
        """Test invalidate removes entry from disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            cache = LLMCache(cache_dir=cache_dir)

            response = LLMResponse(
                content="Test",
                model="test-model",
                usage={},
            )

            cache_key = cache.set(prompt="Test", model="test-model", response=response)
            disk_path = cache._get_disk_path(cache_key)
            assert disk_path.exists()

            cache.invalidate(cache_key)
            assert not disk_path.exists()


class TestClear:
    """Tests for cache clearing."""

    def test_clear_removes_all(self):
        """Test clear removes all entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = LLMCache(cache_dir=Path(tmpdir))

            # Add multiple entries
            for i in range(5):
                response = LLMResponse(
                    content=f"Response {i}",
                    model="test-model",
                    usage={},
                )
                cache.set(prompt=f"Prompt {i}", model="test-model", response=response)

            # Clear
            cache.clear()

            # Verify all gone
            for i in range(5):
                assert cache.get(prompt=f"Prompt {i}", model="test-model") is None

    def test_clear_resets_stats(self):
        """Test clear resets statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = LLMCache(cache_dir=Path(tmpdir))

            response = LLMResponse(
                content="Test",
                model="test-model",
                usage={},
            )

            cache.set(prompt="Test", model="test-model", response=response)
            cache.get(prompt="Test", model="test-model")
            cache.get(prompt="Missing", model="test-model")

            cache.clear()
            stats = cache.get_stats()

            assert stats.memory_hits == 0
            assert stats.misses == 0


class TestStats:
    """Tests for cache statistics."""

    def test_stats_tracking(self):
        """Test cache tracks hits and misses."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = LLMCache(cache_dir=Path(tmpdir))

            response = LLMResponse(
                content="Test",
                model="test-model",
                usage={},
            )

            # Miss
            cache.get(prompt="Unknown", model="test-model")

            # Set and hit
            cache.set(prompt="Known", model="test-model", response=response)
            cache.get(prompt="Known", model="test-model")
            cache.get(prompt="Known", model="test-model")

            stats = cache.get_stats()
            assert stats.misses >= 1
            assert stats.memory_hits >= 1

    def test_stats_memory_size(self):
        """Test stats tracks memory size."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = LLMCache(cache_dir=Path(tmpdir))

            for i in range(3):
                response = LLMResponse(
                    content=f"Response {i}",
                    model="test-model",
                    usage={},
                )
                cache.set(prompt=f"Prompt {i}", model="test-model", response=response)

            stats = cache.get_stats()
            assert stats.memory_size == 3


class TestLRUEviction:
    """Tests for LRU eviction."""

    def test_eviction_when_full(self):
        """Test oldest entry is evicted when cache is full."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = LLMCache(cache_dir=Path(tmpdir), max_memory_entries=3)

            # Add entries up to limit
            for i in range(3):
                response = LLMResponse(
                    content=f"Response {i}",
                    model="test-model",
                    usage={},
                )
                cache.set(prompt=f"Prompt {i}", model="test-model", response=response)

            # Access prompt 0 to make it recently used
            cache.get(prompt="Prompt 0", model="test-model")

            # Add one more (should evict prompt 1, the least recently accessed)
            response = LLMResponse(
                content="Response 3",
                model="test-model",
                usage={},
            )
            cache.set(prompt="Prompt 3", model="test-model", response=response)

            # Check memory size is still at limit
            stats = cache.get_stats()
            assert stats.memory_size <= 3

    def test_evicted_entry_still_on_disk(self):
        """Test evicted entries are still available on disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = LLMCache(cache_dir=Path(tmpdir), max_memory_entries=2)

            # Add 3 entries (will evict first)
            for i in range(3):
                response = LLMResponse(
                    content=f"Response {i}",
                    model="test-model",
                    usage={},
                )
                cache.set(prompt=f"Prompt {i}", model="test-model", response=response)

            # First entry should be evicted from memory but still on disk
            cached = cache.get(prompt="Prompt 0", model="test-model")
            assert cached is not None
            assert cached.content == "Response 0"


class TestDiskPath:
    """Tests for disk path generation."""

    def test_disk_path_structure(self):
        """Test disk path uses two-level directory structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = LLMCache(cache_dir=Path(tmpdir))
            cache_key = "abcdef1234567890abcdef1234567890"
            disk_path = cache._get_disk_path(cache_key)

            # Should be cache_dir/ab/abcdef...json
            assert disk_path.parent.name == "ab"
            assert disk_path.name == f"{cache_key}.json"


class TestThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_access(self):
        """Test cache handles concurrent access."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = LLMCache(cache_dir=Path(tmpdir))
            errors = []

            def worker(worker_id):
                try:
                    for i in range(10):
                        response = LLMResponse(
                            content=f"Response {worker_id}-{i}",
                            model="test-model",
                            usage={},
                        )
                        cache.set(
                            prompt=f"Prompt {worker_id}-{i}",
                            model="test-model",
                            response=response,
                        )
                        cache.get(
                            prompt=f"Prompt {worker_id}-{i}",
                            model="test-model",
                        )
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(errors) == 0, f"Errors occurred: {errors}"

    def test_concurrent_invalidate(self):
        """Test concurrent invalidation is safe."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = LLMCache(cache_dir=Path(tmpdir))
            errors = []

            # Pre-populate cache
            response = LLMResponse(
                content="Test",
                model="test-model",
                usage={},
            )
            cache_key = cache.set(prompt="Test", model="test-model", response=response)

            def worker():
                try:
                    cache.invalidate(cache_key)
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=worker) for _ in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(errors) == 0


class TestCreateCacheFromConfig:
    """Tests for create_cache_from_config helper."""

    def test_creates_cache_from_config(self):
        """Test cache creation from config object."""
        from unittest.mock import MagicMock
        from src.llm.cache import create_cache_from_config

        # Create a mock config
        mock_config = MagicMock()
        mock_config.cache.directory = Path("/tmp/test_cache")
        mock_config.cache.enabled = True

        cache = create_cache_from_config(mock_config)

        assert cache._cache_dir == Path("/tmp/test_cache/llm_responses")
        assert cache.enabled is True

    def test_creates_disabled_cache(self):
        """Test disabled cache creation from config."""
        from unittest.mock import MagicMock
        from src.llm.cache import create_cache_from_config

        mock_config = MagicMock()
        mock_config.cache.directory = Path("/tmp/test_cache")
        mock_config.cache.enabled = False

        cache = create_cache_from_config(mock_config)

        assert cache.enabled is False
