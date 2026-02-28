"""MCP implementation for iris-vector-rag."""

from iris_vector_rag.mcp.bridge import MCPBridge
from iris_vector_rag.mcp.iris_llm_handler import (
    IrisLLMToolHandler,
    register_iris_toolset,
)
from iris_vector_rag.mcp.technique_handlers import (
    TechniqueHandler,
    TechniqueHandlerRegistry,
)

__all__ = [
    "MCPBridge",
    "TechniqueHandler",
    "TechniqueHandlerRegistry",
    "IrisLLMToolHandler",
    "register_iris_toolset",
]
