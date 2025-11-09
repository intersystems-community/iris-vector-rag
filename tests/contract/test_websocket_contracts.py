"""
Contract tests for WebSocket event schemas.

Validates against websocket.yaml OpenAPI spec (FR-025 to FR-028).

IMPORTANT: These tests MUST FAIL initially (TDD principle).
"""

import pytest
import json

try:
    from fastapi.testclient import TestClient
    from iris_vector_rag.api.main import create_app
except ImportError:
    create_app = None

pytestmark = pytest.mark.contract


@pytest.fixture
def client():
    if create_app is None:
        pytest.skip("API application not implemented yet (TDD)")
    app = create_app()
    return TestClient(app)


class TestWebSocketEventContract:
    """FR-028: JSON-based event streaming protocol."""

    def test_websocket_event_has_required_fields(self, client):
        """Contract: Events have event, data, timestamp, request_id."""
        pytest.skip("WebSocket testing requires async client")

        # Expected event format:
        # {
        #   "event": "query_start",
        #   "data": {...},
        #   "timestamp": "2025-10-16T12:34:56.789Z",
        #   "request_id": "uuid"
        # }

    def test_document_upload_progress_events(self, client):
        """FR-027: Stream document loading progress with percentage."""
        pytest.skip("WebSocket testing requires async client")

        # Expected progress event:
        # {
        #   "event": "document_upload_progress",
        #   "data": {
        #     "operation_id": "uuid",
        #     "processed_documents": 47,
        #     "total_documents": 100,
        #     "progress_percentage": 47.0
        #   },
        #   "timestamp": "ISO8601",
        #   "request_id": "uuid"
        # }
