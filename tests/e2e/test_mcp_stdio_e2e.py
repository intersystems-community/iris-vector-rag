"""End to end: the MCP server as a real subprocess over stdio, against a live IRIS.

One test per user story of ADR 0002:

1. an agent starts the server and discovers the six tools;
2. it asks a question and gets documents back from IRIS;
3. it asks for health and gets measured facts (and the truth when IRIS is down);
4. it sends bad input and gets an error, not a crash or a guess.

The server connects where the connection-settings resolver says, so the test hands it
`iris_target`'s IRIS_* variables; it never names a container. An ephemeral container is
started unless IRIS_CONTAINER_NAME pins one (see tests/conftest.py).
"""

import json
import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("mcp", reason="the `mcp` extra provides the official SDK")

from mcp import ClientSession, StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

pytestmark = [pytest.mark.e2e, pytest.mark.mcp]


# The SDK's client starts the server with a minimal environment. The answer step of the
# pipelines needs the LLM key, and offline model settings must reach the server too.
_PASS_THROUGH = ("OPENAI_API_KEY", "HF_HUB_OFFLINE", "HF_HOME", "TRANSFORMERS_OFFLINE")


class _Server:
    def __init__(self, env):
        env = {**{k: os.environ[k] for k in _PASS_THROUGH if k in os.environ}, **env}
        self.params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "iris_vector_rag.mcp"],
            cwd=str(ROOT),
            env=env,
        )

    async def call(self, tool, arguments):
        async with stdio_client(self.params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await session.call_tool(tool, arguments)

    async def tools(self):
        async with stdio_client(self.params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return {t.name: t for t in (await session.list_tools()).tools}


@pytest.fixture
def server(iris_target):
    # No IRIS_CONTAINER_NAME: the target is spelled out, so the server resolves to it.
    return _Server(iris_target.environment())


@pytest.fixture
def documents_in_iris(iris_target):
    from iris_vector_rag.core.models import Document
    from iris_vector_rag.pipelines.basic import BasicRAGPipeline

    BasicRAGPipeline().load_documents(
        documents=[
            Document(
                id="mcp-e2e-1",
                page_content="Diabetes symptoms include increased thirst, frequent "
                "urination, extreme fatigue and blurred vision.",
                metadata={"source": "mcp-e2e.txt"},
            ),
            Document(
                id="mcp-e2e-2",
                page_content="IRIS provides native vector search with HNSW indexing.",
                metadata={"source": "mcp-e2e.txt"},
            ),
        ]
    )


async def test_story_1_agent_discovers_six_tools(server):
    tools = await server.tools()
    assert set(tools) == {
        "rag_basic",
        "rag_basic_rerank",
        "rag_crag",
        "rag_graphrag",
        "rag_pylate_colbert",
        "rag_health_check",
    }
    assert all(t.inputSchema["additionalProperties"] is False for t in tools.values())


async def test_story_2_agent_asks_and_gets_documents_from_iris(
    server, documents_in_iris
):
    result = await server.call(
        "rag_basic", {"query": "What are the symptoms of diabetes?", "top_k": 2}
    )
    assert not result.isError, result.content[0].text
    body = json.loads(result.content[0].text)
    assert isinstance(body["answer"], str)
    contents = " ".join(d["content"] for d in body["retrieved_documents"])
    assert "thirst" in contents, body["retrieved_documents"]


async def test_story_3_health_is_measured(server):
    result = await server.call("rag_health_check", {"include_details": True})
    assert not result.isError, result.content[0].text
    body = json.loads(result.content[0].text)
    assert body["database"]["connected"] is True, body["database"]
    assert body["database"]["response_time_ms"] >= 0
    assert set(body["pipelines"]) >= {"basic", "crag"}


async def test_story_3_health_tells_the_truth_when_iris_is_down(iris_target):
    env = {**iris_target.environment(), "IRIS_HOST": "127.0.0.1", "IRIS_PORT": "1"}
    result = await _Server(env).call("rag_health_check", {})
    assert not result.isError
    body = json.loads(result.content[0].text)
    assert body["status"] == "unavailable"
    assert body["database"]["connected"] is False and body["database"]["error"]


async def test_story_4_bad_input_is_an_error_not_a_guess(server):
    unknown = await server.call("rag_basic", {"query": "q", "injected": "x"})
    assert unknown.isError and "injected" in unknown.content[0].text
    out_of_range = await server.call("rag_basic", {"query": "q", "top_k": 0})
    assert out_of_range.isError and "top_k" in out_of_range.content[0].text
    no_tool = await server.call("rag_metrics", {})
    assert no_tool.isError


async def test_story_4_a_query_against_a_dead_iris_fails_loudly(iris_target):
    env = {**iris_target.environment(), "IRIS_HOST": "127.0.0.1", "IRIS_PORT": "1"}
    result = await _Server(env).call("rag_basic", {"query": "anything"})
    assert result.isError
    assert "answer" not in result.content[0].text
