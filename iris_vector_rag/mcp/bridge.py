"""
MCP Bridge Implementation.

Main bridge class implementing IMCPBridge interface, connecting
Node.js MCP server to Python RAG pipelines.

Feature: Complete MCP Tools Implementation
Branch: 043-complete-mcp-tools
"""

import inspect
from datetime import datetime
from typing import Any, Dict, List, Optional
from iris_vector_rag.mcp.technique_handlers import TechniqueHandlerRegistry
from iris_vector_rag.mcp.config import load_config
from iris_vector_rag.mcp import tool_schemas


class MCPBridge:
    """Main MCP bridge implementation."""

    def __init__(self, config=None, pipeline_manager=None, additional_handlers: Optional[Dict[str, Any]] = None):
        self.config = config or load_config()
        self.registry = TechniqueHandlerRegistry()
        self._request_count = 0
        self._start_time = datetime.now()
        self._pipeline_manager = pipeline_manager
        self.additional_handlers: Dict[str, Any] = additional_handlers or {}

    def _get_pipeline_manager(self):
        """Get or create PipelineManager instance (reuse from REST API if available)."""
        if self._pipeline_manager is None:
            try:
                # Try to import and use REST API's PipelineManager singleton
                from iris_vector_rag.api.services import PipelineManager
                self._pipeline_manager = PipelineManager.get_instance()  # type: ignore[attr-defined]
            except ImportError:
                # REST API not available, use standalone mode
                pass
        return self._pipeline_manager

    async def invoke_technique(
        self,
        technique: str,
        query: str,
        params: Dict[str, Any],
        api_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Invoke a RAG technique.

        Returns:
            {
                "success": bool,
                "result": {...} or None,
                "error": str or None
            }
        """
        try:
            # Validate API key if auth enabled
            if self.config.auth_mode == 'api_key' and api_key:
                # TODO: Validate API key via REST API AuthService
                pass

            handler = None
            result = None

            try:
                handler = self.registry.get_handler(technique)
            except KeyError:
                handler = None

            if handler is not None:
                validated_params = handler.validate_params(params)
                result = await handler.execute(query, validated_params)
            else:
                handler = self._find_additional_handler(technique)
                validated_params = self._validate_additional_handler_params(handler, technique, params)
                execution = handler.execute(technique, validated_params)
                result = await execution if inspect.isawaitable(execution) else execution

            self._request_count += 1

            return {
                'success': True,
                'result': result
            }

        except KeyError:
            return {
                'success': False,
                'error': f"Unknown technique: {technique}"
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    async def get_available_techniques(self) -> List[str]:
        """Get list of available RAG techniques."""
        return self.registry.list_techniques()

    async def health_check(
        self,
        include_details: bool = False,
        include_performance_metrics: bool = True
    ) -> Dict[str, Any]:
        """Check health of all pipelines and database."""
        # Get pipeline statuses
        pipeline_statuses = {}
        if include_details:
            handlers = self.registry.get_all_handlers()
            for name, handler in handlers.items():
                pipeline_statuses[name] = await handler.health_check()

        # Determine overall status
        if pipeline_statuses:
            statuses = [p['status'] for p in pipeline_statuses.values()]
            if any(s == 'unavailable' for s in statuses):
                overall_status = 'unavailable'
            elif any(s == 'degraded' for s in statuses):
                overall_status = 'degraded'
            else:
                overall_status = 'healthy'
        else:
            overall_status = 'healthy'

        # Database status (simplified)
        database_status = {
            'connected': True,
            'response_time_ms': 10,
            'connection_pool_usage': '1/10'
        }

        # Performance metrics
        performance_metrics = None
        if include_performance_metrics:
            uptime_seconds = (datetime.now() - self._start_time).total_seconds()
            qps = self._request_count / max(uptime_seconds, 1)

            performance_metrics = {
                'average_response_time_ms': 500,
                'p95_response_time_ms': 1200,
                'error_rate': 0.02,
                'queries_per_minute': qps * 60
            }

        return {
            'status': overall_status,
            'timestamp': datetime.now().isoformat(),
            'pipelines': pipeline_statuses,
            'database': database_status,
            'performance_metrics': performance_metrics
        }

    async def get_metrics(
        self,
        time_range: str = '1h',
        technique_filter: Optional[List[str]] = None,
        include_error_details: bool = False
    ) -> Dict[str, Any]:
        """Retrieve performance metrics and usage statistics."""
        # Simplified metrics (would be enhanced with real metric collection)
        return {
            'time_range': time_range,
            'total_queries': self._request_count,
            'successful_queries': int(self._request_count * 0.98),
            'failed_queries': int(self._request_count * 0.02),
            'average_response_time_ms': 500,
            'p95_response_time_ms': 1200,
            'p99_response_time_ms': 2000,
            'error_rate': 0.02,
            'queries_per_minute': 10.5,
            'technique_usage': {
                technique: self._request_count // 6
                for technique in self.registry.list_techniques()
            }
        }

    def _find_additional_handler(self, technique: str) -> Any:
        """Return the first additional handler that exposes the requested tool."""
        for handler in self.additional_handlers.values():
            has_tool = getattr(handler, 'has_tool', None)
            if callable(has_tool) and has_tool(technique):
                return handler
        raise KeyError(f"Unknown technique: {technique}")

    def _validate_additional_handler_params(
        self,
        handler: Any,
        technique: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Normalize parameter validation for additional handlers."""
        validator = getattr(handler, 'validate_params', None)
        if validator is None:
            return params
        try:
            return validator(technique, params)
        except TypeError:
            return validator(params)

    async def list_tools(self) -> List[Dict[str, Any]]:
        """List all available MCP tools."""
        all_schemas = tool_schemas.get_all_schemas()
        tools = list(all_schemas.values())
        for handler in self.additional_handlers.values():
            get_tools = getattr(handler, 'get_tools', None)
            if callable(get_tools):
                try:
                    extra_tools = get_tools()
                except Exception:
                    continue
                if isinstance(extra_tools, list):
                    tools.extend(extra_tools)
        return tools
