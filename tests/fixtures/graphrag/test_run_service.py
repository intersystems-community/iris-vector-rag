"""Test run management service for GraphRAG fixtures."""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Dict, Any


class TestRunService:
    """Service for managing test run metadata and results."""

    def __init__(self) -> None:
        self._runs: Dict[str, Dict[str, Any]] = {}

    def start_run(self, run_data: Dict[str, Any]) -> Dict[str, Any]:
        """Start a new test run and return run metadata."""
        run_id = run_data.get("run_id") or f"run-{uuid.uuid4()}"
        start_time = datetime.now(timezone.utc).isoformat()
        start_timestamp = time.time()

        run = {
            **run_data,
            "run_id": run_id,
            "start_time": start_time,
            "_start_timestamp": start_timestamp,
            "results": [],
        }
        self._runs[run_id] = run
        return run

    def update_run(self, run_id: str, update_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update a test run with completion data."""
        run = self._runs.get(run_id, {"run_id": run_id, "results": []})
        run.update(update_data)

        start_timestamp = run.get("_start_timestamp", time.time())
        duration_seconds = max(1, int(time.time() - start_timestamp))
        run["duration_seconds"] = duration_seconds

        self._runs[run_id] = run
        return run

    def add_result(self, run_id: str, result_data: Dict[str, Any]) -> Dict[str, Any]:
        """Add an individual test result to a run."""
        run = self._runs.setdefault(run_id, {"run_id": run_id, "results": []})
        run["results"].append(result_data)
        return result_data
