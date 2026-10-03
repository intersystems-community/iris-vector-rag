"""Drive the real MCP server in-process, with the calls the MCP tests make.

One server (and so one registry, with its pipelines) per instance; every call goes
through the SDK's client session, so the tool schemas, argument validation and error
handling under test are the ones a stdio client meets.
"""

import json
from typing import Any, Dict, List

from mcp.shared.memory import create_connected_server_and_client_session

from iris_vector_rag.mcp.server import create_server


class McpTools:
    def __init__(self, **server_kwargs: Any) -> None:
        self._server = create_server(**server_kwargs)

    async def call(self, tool: str, arguments: Dict[str, Any]):
        async with create_connected_server_and_client_session(self._server) as client:
            return await client.call_tool(tool, arguments)

    async def invoke_technique(
        self, technique: str, query: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """``{"success": bool, "result": dict | None, "error": str | None}``."""
        result = await self.call(f"rag_{technique}", {"query": query, **params})
        text = result.content[0].text
        if result.isError:
            return {"success": False, "result": None, "error": text}
        return {"success": True, "result": json.loads(text), "error": None}

    async def health_check(self, include_details: bool = False) -> Dict[str, Any]:
        result = await self.call(
            "rag_health_check", {"include_details": include_details}
        )
        assert not result.isError, result.content[0].text
        return json.loads(result.content[0].text)

    async def list_tools(self) -> List[Dict[str, Any]]:
        async with create_connected_server_and_client_session(self._server) as client:
            return [t.model_dump() for t in (await client.list_tools()).tools]
