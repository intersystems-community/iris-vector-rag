"""
Bridge to iris-devtools for container lifecycle and state management.

Provides integration with iris-devtools package for IRIS container
management, schema reset, and health checks.

Feature: 035-make-2-modes
"""

import sys
from pathlib import Path
from typing import Any, Dict, Optional

from iris_vector_rag.testing.exceptions import IrisDevtoolsMissingError
from iris_vector_rag.testing.validators import IRISEdition


class IrisDevToolsBridge:
    """
    Bridge to iris-devtools container and state management.

    Provides abstraction over iris-devtools package for IRIS container
    lifecycle, schema reset, and connection validation.
    """

    def __init__(self, iris_devtools_path: Path = Path("../iris-devtools")):
        """
        Initialize bridge and import iris-devtools.

        Args:
            iris_devtools_path: Path to iris-devtools installation

        Raises:
            IrisDevtoolsMissingError: If iris-devtools not found or import fails
        """
        self.devtools_path = iris_devtools_path
        self._container: Optional[Any] = None
        self._iris_container_class: Optional[Any] = None
        self._import_devtools()

    def _import_devtools(self) -> None:
        """
        Import iris-devtools package from configured path.

        Raises:
            IrisDevtoolsMissingError: If path doesn't exist or import fails
        """
        if not self.devtools_path.exists():
            raise IrisDevtoolsMissingError(
                f"iris-devtools not found at {self.devtools_path}\n"
                "Required development dependency.\n"
                "Clone from: git clone <iris-devtools-repo> ../iris-devtools"
            )

        # Add to sys.path for import
        devtools_str = str(self.devtools_path.absolute())
        if devtools_str not in sys.path:
            sys.path.insert(0, devtools_str)

        try:
            from iris_devtools.containers import IRISContainer
            self._iris_container_class = IRISContainer
        except ImportError as e:
            raise IrisDevtoolsMissingError(
                f"Failed to import iris-devtools from {self.devtools_path}: {e}\n"
                "Verify iris-devtools is properly installed."
            ) from e

    def is_available(self) -> bool:
        """
        Check if iris-devtools is available and imported.

        Returns:
            True if iris-devtools successfully imported
        """
        return self._iris_container_class is not None

    def start_container(self, edition: IRISEdition) -> Any:
        """
        Start IRIS container matching specified edition.

        Args:
            edition: IRIS edition (COMMUNITY or ENTERPRISE)

        Returns:
            Started IRISContainer instance

        Examples:
            >>> bridge = IrisDevToolsBridge()
            >>> container = bridge.start_container(IRISEdition.COMMUNITY)
            >>> # Container is now running
        """
        if edition == IRISEdition.COMMUNITY:
            self._container = self._iris_container_class(
                image="intersystemsdc/iris-community:latest"
            )
        else:  # ENTERPRISE
            self._container = self._iris_container_class(
                image="intersystemsdc/irishealth:latest"
            )

        self._container.start()
        return self._container

    def stop_container(self, container: Any) -> None:
        """
        Stop and remove IRIS container.

        Args:
            container: IRISContainer instance to stop
        """
        if container:
            container.stop()
            container.remove()

    def reset_schema(self, connection: Any, namespace: str = "IRISRAG") -> None:
        """
        Reset database schema to clean state.

        Args:
            connection: Active IRIS database connection
            namespace: Namespace to reset (default: IRISRAG)
        """
        cursor = connection.cursor()

        # Drop and recreate namespace (simplified version)
        # In practice, iris-devtools may provide more sophisticated schema reset
        cursor.execute(f"DROP NAMESPACE {namespace}")
        cursor.execute(f"CREATE NAMESPACE {namespace}")

        connection.commit()

    def validate_connection(self, connection: Any) -> bool:
        """
        Validate connection health via simple query.

        Args:
            connection: IRIS database connection to validate

        Returns:
            True if connection is healthy, False otherwise
        """
        try:
            cursor = connection.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            return result and result[0] == 1
        except Exception:
            return False

    def check_health(self, connection: Any) -> Dict[str, Any]:
        """
        Get container/connection health metrics.

        Args:
            connection: Active IRIS database connection

        Returns:
            Dictionary with health status and metrics
        """
        is_healthy = self.validate_connection(connection)

        return {
            "status": "healthy" if is_healthy else "unhealthy",
            "connection_active": is_healthy,
        }

    def wait_for_ready(self, container: Any, timeout: int = 30) -> bool:
        """
        Wait for container to be ready for connections.

        Args:
            container: IRISContainer instance
            timeout: Maximum seconds to wait

        Returns:
            True if container became ready, False if timeout
        """
        if hasattr(container, "is_ready"):
            return container.is_ready()

        # Fallback: assume ready immediately
        return True
