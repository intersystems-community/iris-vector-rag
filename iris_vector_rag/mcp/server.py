"""The iris-vector-rag MCP server: official Python SDK, stdio transport, six tools.

ADR 0002. Stdio means the process boundary is the trust boundary, so there is no
authentication. The IRIS connection comes from the connection-settings resolver
(``IRIS_HOST``, ``IRIS_PORT``, ``IRIS_NAMESPACE``, ``IRIS_USERNAME``, ``IRIS_PASSWORD``,
``IRIS_CONTAINER_NAME``) through the pipelines' own configuration. A failure is
returned to the client as a tool error; nothing is faked.
"""

import json
import logging
import sys
from typing import Any, Dict, List, Optional

import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

from iris_vector_rag.mcp import tool_schemas
from iris_vector_rag.mcp.health import (
    EmbedderProbe,
    IrisProbe,
    check_health,
    probe_iris,
)
from iris_vector_rag.mcp.technique_handlers import TechniqueHandlerRegistry
from iris_vector_rag.mcp.validation import ValidationError

logger = logging.getLogger(__name__)


class IrisUnavailable(Exception):
    """IRIS cannot be reached. The message is safe to return to the client."""


SERVER_NAME = "iris-vector-rag"
HEALTH_TOOL = "rag_health_check"
_PREFIX = "rag_"


def _text(payload: Dict[str, Any]) -> List[types.TextContent]:
    return [types.TextContent(type="text", text=json.dumps(payload, default=str))]


def create_server(
    registry: Optional[TechniqueHandlerRegistry] = None,
    *,
    iris_probe: Optional[IrisProbe] = None,
    embedder_probe: Optional[EmbedderProbe] = None,
) -> Server:
    """Build the server. The probes and registry are injectable for tests."""
    registry = registry or TechniqueHandlerRegistry()
    server: Server = Server(SERVER_NAME)

    @server.list_tools()
    async def list_tools() -> List[types.Tool]:
        return [
            types.Tool(
                name=name,
                description=schema["description"],
                inputSchema=schema["inputSchema"],
            )
            for name, schema in tool_schemas.get_all_schemas().items()
        ]

    @server.call_tool(validate_input=False)  # validated below, with field names
    async def call_tool(
        name: str, arguments: Dict[str, Any]
    ) -> List[types.TextContent]:
        arguments = arguments or {}
        try:
            if tool_schemas.get_schema(name) is None:
                raise ValidationError("name", name, f"Unknown tool: {name}")
            params = tool_schemas.validate_params(name, arguments)

            if name == HEALTH_TOOL:
                report = await check_health(
                    registry,
                    include_details=params["include_details"],
                    iris_probe=iris_probe,
                    embedder_probe=embedder_probe,
                )
                return _text(report)

            # Some pipelines log a retrieval failure and still return an answer. An
            # agent must get an error for a dead IRIS, so check before running one.
            try:
                await (iris_probe or probe_iris)()
            except Exception as exc:
                raise IrisUnavailable(f"IRIS is not reachable: {exc}"[:300]) from None

            handler = registry.get_handler(name[len(_PREFIX) :])
            query = params.pop("query")
            return _text(await handler.execute(query, params))
        except ValidationError as exc:
            # Our own message about the caller's input: safe to return.
            raise ValueError(str(exc)) from None
        except IrisUnavailable as exc:
            logger.error("Tool %s: %s", name, exc)
            raise RuntimeError(str(exc)) from None
        except Exception as exc:
            logger.exception("Tool %s failed", name)
            raise RuntimeError(
                f"{name} failed with {type(exc).__name__}; see the server log on stderr"
            ) from None

    return server


async def serve() -> None:
    """Serve on stdin/stdout until the client disconnects."""
    server = create_server()
    async with stdio_server() as (read_stream, write_stream):
        # stdout now carries the protocol; a stray print() from a library must not.
        sys.stdout = sys.stderr
        await server.run(
            read_stream, write_stream, server.create_initialization_options()
        )
