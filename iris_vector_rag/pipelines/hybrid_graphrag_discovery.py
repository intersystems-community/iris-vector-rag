"""
Graph Core Discovery Module

Provides secure discovery of iris_vector_graph modules without local path fallbacks.
"""

import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class GraphCoreDiscovery:
    """Handles discovery and import of iris_vector_graph modules safely."""

    def __init__(self, config_manager: Optional[Any] = None):
        self.config_manager = config_manager
        self._import_cache: Dict[str, Any] = {}

    def import_graph_core_modules(self) -> Dict[str, Any]:
        """Import iris_vector_graph modules or raise actionable ImportError."""
        message = (
            "HybridGraphRAG requires iris-vector-graph package. "
            "Install with: pip install rag-templates[hybrid-graphrag]"
        )

        if os.environ.get("FORCE_IRIS_VECTOR_GRAPH_IMPORT_ERROR") == "1":
            raise ImportError(message)

        try:
            import importlib
            import sys

            sys.modules.pop("iris_vector_graph", None)
            module = importlib.import_module("iris_vector_graph")
            fusion_module = importlib.import_module("iris_vector_graph.fusion")

            def _get_attr(name: str):
                return getattr(module, name, type(name, (), {}))

            hybrid_fusion = getattr(fusion_module, "HybridSearchFusion", None)
            if hybrid_fusion is None:
                hybrid_fusion = _get_attr("HybridSearchFusion")

            modules = {
                "IRISGraphEngine": _get_attr("IRISGraphEngine"),
                "HybridSearchFusion": hybrid_fusion,
                "TextSearchEngine": _get_attr("TextSearchEngine"),
                "VectorOptimizer": _get_attr("VectorOptimizer"),
            }

            logger.info("Successfully imported iris_vector_graph modules")
            self._import_cache["modules"] = modules
            return modules
        except ImportError as exc:
            raise ImportError(message) from exc

    def get_connection_config(self) -> Dict[str, Any]:
        """Return IRIS connection configuration from environment variables."""
        return {
            "host": os.environ.get("IRIS_HOST", "localhost"),
            "port": int(os.environ.get("IRIS_PORT", "1974")),
            "namespace": os.environ.get("IRIS_NAMESPACE", "USER"),
            "username": os.environ.get("IRIS_USERNAME", "SuperUser"),
            "password": os.environ.get("IRIS_PASSWORD", "SYS"),
        }

    def validate_connection_config(
        self, connection_config: Dict[str, Any]
    ) -> tuple[bool, list[str]]:
        """Validate required IRIS connection fields are present."""
        required_fields = ["host", "port", "namespace", "username", "password"]
        missing = [
            field
            for field in required_fields
            if not connection_config.get(field)
        ]
        return len(missing) == 0, missing
