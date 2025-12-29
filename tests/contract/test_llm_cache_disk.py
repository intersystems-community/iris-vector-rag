"""
Contract tests for DiskCacheBackend.
"""

import pytest
import shutil
from pathlib import Path
from iris_vector_rag.common.llm_cache_disk import DiskCacheBackend

@pytest.fixture
def temp_cache_dir():
    cache_dir = Path("./.test_cache_disk")
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    yield str(cache_dir)
    if cache_dir.exists():
        shutil.rmtree(cache_dir)

def test_disk_cache_set_get(temp_cache_dir):
    """Verify basic set/get functionality."""
    backend = DiskCacheBackend(cache_dir=temp_cache_dir)
    
    key = "test_key"
    value = {"text": "hello world", "model": "gpt-4"}
    
    backend.set(key, value)
    
    # Verify file exists
    assert (Path(temp_cache_dir) / f"{key}.json").exists()
    
    # Retrieve
    retrieved = backend.get(key)
    assert retrieved == value

def test_disk_cache_expiration(temp_cache_dir):
    """Verify TTL handling."""
    import time
    backend = DiskCacheBackend(cache_dir=temp_cache_dir, ttl_seconds=1)
    
    key = "exp_key"
    backend.set(key, "data")
    
    assert backend.get(key) == "data"
    
    # Wait for expiration
    time.sleep(1.1)
    
    assert backend.get(key) is None
    assert not (Path(temp_cache_dir) / f"{key}.json").exists()

def test_disk_cache_clear(temp_cache_dir):
    """Verify clearing entire cache."""
    backend = DiskCacheBackend(cache_dir=temp_cache_dir)
    backend.set("k1", "v1")
    backend.set("k2", "v2")
    
    assert len(list(Path(temp_cache_dir).glob("*.json"))) == 2
    
    backend.clear()
    
    assert len(list(Path(temp_cache_dir).glob("*.json"))) == 0
