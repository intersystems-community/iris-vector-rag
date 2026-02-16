from __future__ import annotations

from contextlib import contextmanager
import time


@contextmanager
def assert_duration(max_seconds: float, label: str = "operation"):
    """Assert that a block completes within max_seconds."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    assert elapsed <= max_seconds, (
        f"{label} took {elapsed:.3f}s, exceeds {max_seconds:.3f}s"
    )
