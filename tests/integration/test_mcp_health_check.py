"""Integration test: the health check reports what it measured, against a live IRIS.

Replaces the earlier assertions on hard-coded values (ADR 0002; VM-06). The target is
the session's `iris_target` (an ephemeral container unless IRIS_CONTAINER_NAME pins
one), exported as IRIS_* so the server's connection resolver finds it.
"""

import pytest

from tests.support.mcp_tools import McpTools


@pytest.mark.integration
@pytest.mark.mcp
class TestMCPHealthCheck:
    @pytest.mark.asyncio
    async def test_reports_a_measured_connection(self, iris_target):
        result = await McpTools().health_check()

        assert result["database"]["connected"] is True, result["database"]
        assert result["database"]["response_time_ms"] >= 0
        assert result["status"] in ("healthy", "degraded")

    @pytest.mark.asyncio
    async def test_details_name_the_five_pipelines(self, iris_target):
        result = await McpTools().health_check(include_details=True)

        assert set(result["pipelines"]) == {
            "basic",
            "basic_rerank",
            "crag",
            "graphrag",
            "pylate_colbert",
        }
        for name, state in result["pipelines"].items():
            assert isinstance(state["constructible"], bool), name
            assert state["constructible"] or state["error"], name

    @pytest.mark.asyncio
    async def test_an_unreachable_iris_is_unavailable_not_healthy(
        self, iris_target, monkeypatch
    ):
        monkeypatch.delenv("IRIS_CONTAINER_NAME", raising=False)
        monkeypatch.setenv("IRIS_HOST", "127.0.0.1")
        monkeypatch.setenv("IRIS_PORT", "1")

        result = await McpTools().health_check()

        assert result["status"] == "unavailable"
        assert result["database"]["connected"] is False
        assert result["database"]["error"]
