"""
Graph Core Discovery Module

Provides secure, config-driven discovery of iris_vector_graph modules without
hard-coded paths or unsafe sys.path modifications.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class GraphCoreDiscovery:
    """Handles discovery and import of iris_vector_graph modules safely."""

    def __init__(self, config_manager=None):
        self.config_manager = config_manager
        self._discovery_cache = None
        self._import_cache = {}

    def discover_graph_core_path(self) -> Optional[Path]:
        """
        Discover iris_vector_graph path using config-first approach.

        Priority order:
        1. Configuration manager setting
        2. Environment variable GRAPH_CORE_PATH
        3. Sibling directory search (secure patterns only)

        Returns:
            Path to graph core directory or None if not found
        """
        if self._discovery_cache is not None:
            return self._discovery_cache

        # Priority 1: Configuration manager
        if self.config_manager:
            graph_core_config = self.config_manager.get("integrations:graph_core", {})
            config_path = graph_core_config.get("path")
            if config_path:
                config_path = Path(config_path).expanduser().resolve()
                if self._validate_graph_core_path(config_path):
                    self._discovery_cache = config_path
                    logger.info(f"Found graph core via config: {config_path}")
                    return config_path
                else:
                    logger.warning(f"Config path invalid: {config_path}")

        # Priority 2: Environment variable
        env_path = os.environ.get("GRAPH_CORE_PATH")
        if env_path:
            env_path = Path(env_path).expanduser().resolve()
            if self._validate_graph_core_path(env_path):
                self._discovery_cache = env_path
                logger.info(f"Found graph core via GRAPH_CORE_PATH: {env_path}")
                return env_path
            else:
                logger.warning(f"GRAPH_CORE_PATH invalid: {env_path}")

        # Priority 3: Sibling directory search (secure patterns only)
        sibling_path = self._search_sibling_directories()
        if sibling_path:
            self._discovery_cache = sibling_path
            logger.info(f"Found graph core via sibling search: {sibling_path}")
            return sibling_path

        # Not found
        self._discovery_cache = None
        logger.info("Graph core not found - hybrid features will be disabled")
        return None

    def _validate_graph_core_path(self, path: Path) -> bool:
        """Validate that a path contains iris_vector_graph package."""
        if not path.exists() or not path.is_dir():
            return False

        # Check for iris_vector_graph package directory (top-level module)
        package_dir = path / "iris_vector_graph"
        return package_dir.exists() and package_dir.is_dir()

    def _search_sibling_directories(self) -> Optional[Path]:
        """Search sibling directories for graph core (secure patterns only)."""
        current_dir = Path(__file__).parent.parent.parent.parent

        # Only search for known, safe directory patterns
        safe_patterns = [
            "iris-vector-graph",
            "iris-graph",
            "ai-graph",
            "graph-ai",
            "iris_vector_graph",
            "iris_graph",
        ]

        for pattern in safe_patterns:
            candidate = current_dir / pattern
            if self._validate_graph_core_path(candidate):
                return candidate

        return None

    def import_graph_core_modules(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Safely import graph core modules from iris-vector-graph package.

        First checks for the iris-vector-graph package (installed via extras),
        then falls back to local graph core path discovery for development.

        Returns:
            Tuple of (success: bool, modules: dict)
        """
        if "modules" in self._import_cache:
            return self._import_cache["success"], self._import_cache["modules"]

        modules = {}

        # First, try to import from iris-vector-graph package (preferred)
        try:
            from iris_vector_graph import IRISGraphEngine
            from iris_vector_graph import HybridSearchFusion
            from iris_vector_graph import TextSearchEngine
            from iris_vector_graph import VectorOptimizer

            modules = {
                "IRISGraphEngine": IRISGraphEngine,
                "HybridSearchFusion": HybridSearchFusion,
                "TextSearchEngine": TextSearchEngine,
                "VectorOptimizer": VectorOptimizer,
            }

            logger.info(
                "Successfully imported iris_vector_graph from iris-vector-graph package"
            )
            self._import_cache = {"success": True, "modules": modules}
            return True, modules

        except ImportError:
            logger.debug(
                "iris-vector-graph package not available, trying local graph core path"
            )

        # Fall back to local graph core path discovery (for development)
        graph_core_path = self.discover_graph_core_path()

        if not graph_core_path:
            self._log_dependency_help()
            self._import_cache = {"success": False, "modules": {}}
            return False, {}

        # Try local graph core path with sys.path modification
        original_path = sys.path.copy()
        try:
            if str(graph_core_path) not in sys.path:
                sys.path.insert(0, str(graph_core_path))

            # Try iris_vector_graph package from local path
            try:
                from iris_vector_graph import IRISGraphEngine
                from iris_vector_graph import HybridSearchFusion
                from iris_vector_graph import TextSearchEngine
                from iris_vector_graph import VectorOptimizer

                modules = {
                    "IRISGraphEngine": IRISGraphEngine,
                    "HybridSearchFusion": HybridSearchFusion,
                    "TextSearchEngine": TextSearchEngine,
                    "VectorOptimizer": VectorOptimizer,
                    "package_name": "iris_vector_graph",
                }
                logger.info(
                    "Successfully imported iris_vector_graph from local path"
                )
                self._import_cache = {"success": True, "modules": modules}
                return True, modules

            except ImportError as e:
                logger.warning(
                    f"Failed to import from local path: {e}"
                )
                self._import_cache = {"success": False, "modules": {}}
                return False, {}

        finally:
            sys.path[:] = original_path

    def _log_dependency_help(self):
        """Log helpful information about installing HybridGraphRAG dependencies."""
        logger.warning(
            "HybridGraphRAG requires the iris-vector-graph package for advanced features."
        )
        logger.info(
            "To enable HybridGraphRAG, install with: pip install rag-templates[hybrid-graphrag]"
        )
        logger.info(
            "This will install iris-vector-graph and enable 50x performance improvements."
        )
        logger.info("Continuing with standard GraphRAG capabilities only.")

    def get_connection_config(self) -> Dict[str, Any]:
        """Get secure IRIS connection configuration from ConfigurationManager."""
        if not self.config_manager:
            return {}

        # Get database configuration from ConfigurationManager
        db_config = self.config_manager.get("database:iris", {})

        # Build secure connection config (no defaults for security)
        connection_config = {}

        # Only use environment variables if explicitly set (no insecure defaults)
        if "host" in db_config or "IRIS_HOST" in os.environ:
            connection_config["host"] = db_config.get("host") or os.environ.get(
                "IRIS_HOST"
            )

        if "port" in db_config or "IRIS_PORT" in os.environ:
            try:
                port_cfg = db_config.get("port")
                if port_cfg is not None:
                    connection_config["port"] = int(port_cfg)
                else:
                    port_env = os.environ.get("IRIS_PORT")
                    if port_env:
                        connection_config["port"] = int(port_env)
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid IRIS port value: {e}")

        if "namespace" in db_config or "IRIS_NAMESPACE" in os.environ:
            connection_config["namespace"] = db_config.get(
                "namespace"
            ) or os.environ.get("IRIS_NAMESPACE")

        if "username" in db_config or "IRIS_USER" in os.environ:
            connection_config["username"] = db_config.get("username") or os.environ.get(
                "IRIS_USER"
            )

        if "password" in db_config or "IRIS_PASSWORD" in os.environ:
            connection_config["password"] = db_config.get("password") or os.environ.get(
                "IRIS_PASSWORD"
            )

        return connection_config

    def validate_connection_config(self, config: Dict[str, Any]) -> Tuple[bool, list]:
        """
        Validate that all required connection parameters are present.

        Returns:
            Tuple of (is_valid: bool, missing_params: list)
        """
        required_params = ["host", "port", "namespace", "username", "password"]
        missing = [
            param
            for param in required_params
            if param not in config or config[param] is None
        ]
        return len(missing) == 0, missing
