"""
Python MCP Bridge Interface Contract.

This module defines the interface for the Python bridge layer that connects
the Node.js MCP server to the RAG pipelines. This is a contract (TDD approach)
- implementations should satisfy these interfaces.

Feature: Complete MCP Tools Implementation
Branch: 043-complete-mcp-tools
"""

from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
from datetime import datetime


class IMCPBridge(ABC):
    """Interface for Python MCP bridge."""

    @abstractmethod
    async def invoke_technique(
        self,
        technique: str,
        query: str,
        params: Dict[str, Any],
        api_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Invoke a RAG technique.

        Args:
            technique: Pipeline name (basic, basic_rerank, crag, graphrag, pylate_colbert, iris_global_graphrag)
            query: User's question or search query
            params: Pipeline-specific parameters (top_k, confidence_threshold, etc.)
            api_key: Optional API key for authentication (required if auth_mode='api_key')

        Returns:
            {
                "success": bool,
                "result": {
                    "answer": str,
                    "retrieved_documents": List[Dict],
                    "sources": List[str],
                    "metadata": Dict,
                    "performance": {
                        "execution_time_ms": int,
                        "retrieval_time_ms": int,
                        "generation_time_ms": int,
                        "tokens_used": int
                    }
                },
                "error": Optional[str]
            }

        Raises:
            No exceptions - all errors returned in response dict
        """
        pass

    @abstractmethod
    async def get_available_techniques(self) -> List[str]:
        """
        Get list of available RAG techniques.

        Returns:
            List of technique names (e.g., ["basic", "crag", "graphrag", ...])
        """
        pass

    @abstractmethod
    async def health_check(
        self,
        include_details: bool = False,
        include_performance_metrics: bool = True
    ) -> Dict[str, Any]:
        """
        Check health of all pipelines and database.

        Args:
            include_details: Include detailed metrics for each pipeline
            include_performance_metrics: Include recent performance statistics

        Returns:
            {
                "status": "healthy" | "degraded" | "unavailable",
                "timestamp": str (ISO 8601),
                "pipelines": {
                    "basic": {"status": "healthy", "last_success": str, "error_rate": float},
                    "crag": {...},
                    ...
                },
                "database": {
                    "connected": bool,
                    "response_time_ms": int,
                    "connection_pool_usage": str
                },
                "performance_metrics": {
                    "average_response_time_ms": int,
                    "p95_response_time_ms": int,
                    "error_rate": float,
                    "queries_per_minute": float
                } if include_performance_metrics else None
            }
        """
        pass

    @abstractmethod
    async def get_metrics(
        self,
        time_range: str = "1h",
        technique_filter: Optional[List[str]] = None,
        include_error_details: bool = False
    ) -> Dict[str, Any]:
        """
        Retrieve performance metrics and usage statistics.

        Args:
            time_range: Time range for metrics (5m, 15m, 1h, 6h, 24h, 7d)
            technique_filter: Filter metrics by specific techniques (None = all)
            include_error_details: Include detailed error information

        Returns:
            {
                "time_range": str,
                "total_queries": int,
                "successful_queries": int,
                "failed_queries": int,
                "average_response_time_ms": int,
                "p95_response_time_ms": int,
                "p99_response_time_ms": int,
                "technique_usage": {
                    "basic": int,
                    "crag": int,
                    ...
                },
                "error_breakdown": {...} if include_error_details else None
            }
        """
        pass


class ITechniqueHandler(ABC):
    """Interface for individual technique handlers."""

    @abstractmethod
    async def execute(
        self,
        query: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute the technique with given query and parameters.

        Args:
            query: User's question
            params: Technique-specific parameters

        Returns:
            {
                "answer": str,
                "retrieved_documents": List[Dict],
                "sources": List[str],
                "metadata": Dict,
                "performance": Dict
            }
        """
        pass

    @abstractmethod
    def validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate parameters against technique's schema.

        Args:
            params: Parameters to validate

        Returns:
            Validated parameters with defaults applied

        Raises:
            ValidationError: If params are invalid
        """
        pass

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """
        Check health of this technique.

        Returns:
            {
                "status": "healthy" | "degraded" | "unavailable",
                "last_success": datetime,
                "error_rate": float
            }
        """
        pass


class IMCPServerManager(ABC):
    """Interface for MCP server lifecycle management."""

    @abstractmethod
    async def start_server(self) -> Dict[str, Any]:
        """
        Start the MCP server.

        Returns:
            {
                "success": bool,
                "server_info": {
                    "name": str,
                    "version": str,
                    "transport": str,
                    "enabled_techniques": List[str]
                },
                "error": Optional[str]
            }
        """
        pass

    @abstractmethod
    async def stop_server(self) -> Dict[str, Any]:
        """
        Stop the MCP server.

        Returns:
            {"success": bool, "error": Optional[str]}
        """
        pass

    @abstractmethod
    async def handle_tool_call(
        self,
        tool_name: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Handle an MCP tool call.

        Args:
            tool_name: Tool to invoke (e.g., "rag_basic", "rag_health_check")
            params: Tool parameters

        Returns:
            {
                "success": bool,
                "result": Dict[str, Any],
                "error": Optional[str]
            }
        """
        pass

    @abstractmethod
    async def list_tools(self) -> List[Dict[str, Any]]:
        """
        List all available MCP tools.

        Returns:
            List of tool definitions with schemas
        """
        pass

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """
        Check server health.

        Returns:
            {
                "status": "healthy" | "degraded" | "unavailable",
                "techniques_available": int,
                "active_connections": int,
                "max_connections": int
            }
        """
        pass


class IAuthService(ABC):
    """Interface for authentication service."""

    @abstractmethod
    async def validate_api_key(self, api_key: str) -> Dict[str, Any]:
        """
        Validate an API key.

        Args:
            api_key: API key to validate

        Returns:
            {
                "valid": bool,
                "key_id": Optional[str],
                "permissions": Optional[List[str]],
                "tier": Optional[str]
            }
        """
        pass


class ValidationError(Exception):
    """Raised when parameter validation fails."""

    def __init__(self, field: str, value: Any, message: str):
        self.field = field
        self.value = value
        self.message = message
        super().__init__(f"Validation error on field '{field}': {message}")


class MCPError(Exception):
    """Base exception for MCP-related errors."""

    def __init__(self, code: str, message: str, data: Optional[Dict[str, Any]] = None):
        self.code = code
        self.message = message
        self.data = data or {}
        super().__init__(message)


# Type hints for common data structures
ToolSchema = Dict[str, Any]
ToolRequest = Dict[str, Any]
ToolResponse = Dict[str, Any]
HealthStatus = Dict[str, Any]
PerformanceMetrics = Dict[str, Any]
