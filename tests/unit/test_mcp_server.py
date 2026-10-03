"""The MCP server, driven through the SDK's in-memory client. No IRIS needed.

ADR 0002: Python over stdio, six tools, no authentication, errors raised and never faked.
"""

import json
from contextlib import asynccontextmanager

import pytest

pytest.importorskip("mcp", reason="the `mcp` extra provides the official SDK")

from mcp.shared.memory import create_connected_server_and_client_session  # noqa: E402

SIX_TOOLS = {
    "rag_basic",
    "rag_basic_rerank",
    "rag_crag",
    "rag_graphrag",
    "rag_pylate_colbert",
    "rag_health_check",
}


class FakePipeline:
    def __init__(self, fail_with=None):
        self.calls = []
        self.fail_with = fail_with

    def query(self, query, **params):
        self.calls.append((query, params))
        if self.fail_with is not None:
            raise self.fail_with
        return {
            "answer": "a fake answer",
            "retrieved_documents": [{"doc_id": "d1", "content": "c", "score": 0.9}],
            "sources": ["s1"],
            "execution_time": 0.01,
        }


def _registry(pipeline):
    from iris_vector_rag.mcp.technique_handlers import (
        TechniqueHandler,
        TechniqueHandlerRegistry,
    )

    registry = TechniqueHandlerRegistry()
    for name in registry.list_techniques():
        registry.register_handler(name, TechniqueHandler(name, lambda: pipeline))
    return registry


@asynccontextmanager
async def _client(pipeline=None, **server_kwargs):
    from iris_vector_rag.mcp.server import create_server

    async def iris_up():
        return 1.0

    server_kwargs.setdefault("iris_probe", iris_up)
    server = create_server(
        registry=_registry(pipeline or FakePipeline()), **server_kwargs
    )
    async with create_connected_server_and_client_session(server) as session:
        yield session


def _json(result):
    return json.loads(result.content[0].text)


async def test_lists_exactly_the_six_tools():
    async with _client() as client:
        tools = (await client.list_tools()).tools
    assert {t.name for t in tools} == SIX_TOOLS


async def test_every_input_schema_rejects_unknown_keys():
    async with _client() as client:
        tools = (await client.list_tools()).tools
    for tool in tools:
        assert tool.inputSchema.get("additionalProperties") is False, tool.name
        assert tool.description, tool.name


async def test_rag_basic_returns_the_pipeline_result_as_json():
    pipeline = FakePipeline()
    async with _client(pipeline) as client:
        result = await client.call_tool("rag_basic", {"query": "what?", "top_k": 3})
    assert not result.isError
    body = _json(result)
    assert body["answer"] == "a fake answer"
    assert body["retrieved_documents"][0]["doc_id"] == "d1"
    assert pipeline.calls[0][0] == "what?"
    assert pipeline.calls[0][1]["top_k"] == 3


async def test_unknown_argument_is_rejected_and_never_reaches_the_pipeline():
    pipeline = FakePipeline()
    async with _client(pipeline) as client:
        result = await client.call_tool("rag_basic", {"query": "q", "evil": 1})
    assert result.isError
    assert pipeline.calls == []


async def test_missing_query_is_an_error():
    async with _client() as client:
        result = await client.call_tool("rag_basic", {})
    assert result.isError


async def test_out_of_range_value_is_an_error_saying_why():
    async with _client() as client:
        result = await client.call_tool("rag_basic", {"query": "q", "top_k": 10_000})
    assert result.isError
    assert "top_k" in result.content[0].text


async def test_unknown_tool_is_an_error():
    async with _client() as client:
        result = await client.call_tool("rag_metrics", {})
    assert result.isError


async def test_pipeline_failure_is_an_error_that_does_not_echo_internals():
    secret = "password=hunter2 at /srv/internal/path"
    async with _client(FakePipeline(fail_with=RuntimeError(secret))) as client:
        result = await client.call_tool("rag_basic", {"query": "q"})
    assert result.isError
    text = result.content[0].text
    assert "hunter2" not in text and "/srv/internal" not in text
    assert "RuntimeError" in text


async def test_unimportable_pipeline_fails_the_call_not_the_listing():
    from iris_vector_rag.mcp.technique_handlers import TechniqueHandler

    def broken():
        raise ImportError("pylate is not installed")

    from iris_vector_rag.mcp.server import create_server

    registry = _registry(FakePipeline())
    registry.register_handler(
        "pylate_colbert", TechniqueHandler("pylate_colbert", broken)
    )

    async def iris_up():
        return 1.0

    server = create_server(registry=registry, iris_probe=iris_up)
    async with create_connected_server_and_client_session(server) as client:
        assert "rag_pylate_colbert" in {
            t.name for t in (await client.list_tools()).tools
        }
        result = await client.call_tool("rag_pylate_colbert", {"query": "q"})
    assert result.isError
    assert "ImportError" in result.content[0].text


async def test_health_check_reports_an_unreachable_iris_as_unavailable():
    async def iris_down():
        raise ConnectionError("no route to host")

    async def embedder_ok():
        return 384

    async with _client(iris_probe=iris_down, embedder_probe=embedder_ok) as client:
        result = await client.call_tool("rag_health_check", {})
    assert not result.isError
    body = _json(result)
    assert body["status"] == "unavailable"
    assert body["database"]["connected"] is False
    assert "ConnectionError" in body["database"]["error"]


async def test_health_check_invents_no_values():
    async def iris_ok():
        return 7.5

    async def embedder_ok():
        return 384

    async with _client(iris_probe=iris_ok, embedder_probe=embedder_ok) as client:
        body = _json(
            await client.call_tool("rag_health_check", {"include_details": True})
        )
    assert body["status"] == "healthy"
    assert body["database"] == {"connected": True, "response_time_ms": 7.5}
    assert body["embedder"] == {"loadable": True, "dimension": 384}
    assert set(body["pipelines"]) == {
        "basic",
        "basic_rerank",
        "crag",
        "graphrag",
        "pylate_colbert",
    }
    assert all(p["constructible"] is True for p in body["pipelines"].values())
    for invented in ("performance_metrics", "connection_pool_usage", "error_rate"):
        assert invented not in json.dumps(body)


async def test_health_check_degrades_when_one_pipeline_cannot_be_constructed():
    from iris_vector_rag.mcp.technique_handlers import TechniqueHandler

    def broken():
        raise ImportError("pylate is not installed")

    async def ok():
        return 1

    from iris_vector_rag.mcp.server import create_server

    registry = _registry(FakePipeline())
    registry.register_handler(
        "pylate_colbert", TechniqueHandler("pylate_colbert", broken)
    )
    server = create_server(registry=registry, iris_probe=ok, embedder_probe=ok)
    async with create_connected_server_and_client_session(server) as client:
        body = _json(
            await client.call_tool("rag_health_check", {"include_details": True})
        )
    assert body["status"] == "degraded"
    assert body["pipelines"]["pylate_colbert"]["constructible"] is False
    assert "ImportError" in body["pipelines"]["pylate_colbert"]["error"]
    assert body["pipelines"]["basic"]["constructible"] is True


async def test_default_iris_probe_uses_the_connection_settings_resolver(monkeypatch):
    """The probe connects where the resolver says, and raises when nothing listens."""
    from iris_vector_rag.mcp.health import probe_iris

    monkeypatch.delenv("IRIS_CONTAINER_NAME", raising=False)
    monkeypatch.setenv("IRIS_HOST", "127.0.0.1")
    monkeypatch.setenv("IRIS_PORT", "1")  # nothing listens on port 1
    with pytest.raises(Exception):
        await probe_iris()


async def test_a_query_with_iris_down_is_an_error_and_never_runs_the_pipeline():
    async def iris_down():
        raise ConnectionError("port 1 not reachable")

    pipeline = FakePipeline()
    async with _client(pipeline, iris_probe=iris_down) as client:
        result = await client.call_tool("rag_basic", {"query": "q"})
    assert result.isError
    assert "IRIS is not reachable" in result.content[0].text
    assert pipeline.calls == []
