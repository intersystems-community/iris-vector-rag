"""
Graph Core Discovery Module - Simplified Version

Requires iris-vector-graph package to be installed.
No fallbacks, no local path discovery - just direct imports.
"""

import logging
import os
from typing import Any, Dict

logger = logging.getLogger(__name__)


class GraphCoreDiscovery:
    """Handles import of iris_graph_core modules from iris-vector-graph package."""

    def __init__(self, config_manager=None):
        self.config_manager = config_manager
        self._modules = None

    def import_graph_core_modules(self) -> Dict[str, Any]:
        """
        Import graph core modules from iris-vector-graph package.

        This simplified version requires iris-vector-graph to be installed.
        No fallbacks to local development paths.

        Returns:
            Dictionary of imported modules

        Raises:
            ImportError: If iris-vector-graph is not installed
        """
        if self._modules is not None:
            return self._modules

        try:
            from iris_vector_graph_core.engine import IRISGraphEngine
            from iris_vector_graph_core.fusion import HybridSearchFusion
            from iris_vector_graph_core.text_search import TextSearchEngine
            from iris_vector_graph_core.vector_utils import VectorOptimizer

            self._modules = {
                "IRISGraphEngine": IRISGraphEngine,
                "HybridSearchFusion": HybridSearchFusion,
                "TextSearchEngine": TextSearchEngine,
                "VectorOptimizer": VectorOptimizer,
            }

            logger.info("Successfully imported iris_graph_core modules from iris-vector-graph package")
            return self._modules

        except ImportError as e:
            logger.error(
                "iris-vector-graph package is required for HybridGraphRAG. "
                "Install with: pip install rag-templates[hybrid-graphrag]"
            )
            raise ImportError(
                "HybridGraphRAG requires iris-vector-graph package. "
                "Install with: pip install rag-templates[hybrid-graphrag]"
            ) from e

    def get_connection_config(self) -> Dict[str, Any]:
        """Get IRIS connection configuration from ConfigurationManager and environment."""
        connection_config = {}

        # Get database configuration from ConfigurationManager
        if self.config_manager:
            db_config = self.config_manager.get("database:iris", {})

            # Merge config file settings
            if "host" in db_config:
                connection_config["host"] = db_config["host"]
            if "port" in db_config:
                connection_config["port"] = int(db_config["port"])
            if "namespace" in db_config:
                connection_config["namespace"] = db_config["namespace"]
            if "username" in db_config:
                connection_config["username"] = db_config["username"]
            if "password" in db_config:
                connection_config["password"] = db_config["password"]

        # Environment variables override config file
        if "IRIS_HOST" in os.environ:
            connection_config["host"] = os.environ["IRIS_HOST"]
        if "IRIS_PORT" in os.environ:
            connection_config["port"] = int(os.environ["IRIS_PORT"])
        if "IRIS_NAMESPACE" in os.environ:
            connection_config["namespace"] = os.environ["IRIS_NAMESPACE"]
        if "IRIS_USER" in os.environ:
            connection_config["username"] = os.environ["IRIS_USER"]
        if "IRIS_PASSWORD" in os.environ:
            connection_config["password"] = os.environ["IRIS_PASSWORD"]

        # Apply defaults if still missing
        connection_config.setdefault("host", "localhost")
        connection_config.setdefault("port", 1972)
        connection_config.setdefault("namespace", "USER")
        connection_config.setdefault("username", "_SYSTEM")
        connection_config.setdefault("password", "SYS")

        return connection_config
