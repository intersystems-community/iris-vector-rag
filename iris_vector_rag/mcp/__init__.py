"""MCP server for iris-vector-rag (stdio; see ``iris_vector_rag.mcp.server``).

Importing this package does not import the MCP SDK; only the server does.
"""

from iris_vector_rag.mcp.technique_handlers import (
    TechniqueHandler,
    TechniqueHandlerRegistry,
)

__all__ = ["TechniqueHandler", "TechniqueHandlerRegistry"]
