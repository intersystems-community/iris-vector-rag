# tests/utils/container_lifecycle_manager.py

import logging
from typing import Dict, Any
from .container_manager import ContainerManager
from .compose_file_tracker import ComposeFileTracker


class ContainerLifecycleManager:
    """Orchestrates container lifecycle with compose file change detection."""
    
    def __init__(self, compose_file: str = None):
        """
        Initialize ContainerLifecycleManager.
        
        Args:
            compose_file: Path to docker-compose file. If None, uses COMPOSE_FILE
                         environment variable or defaults to "docker-compose.yml"
        """
        self.logger = logging.getLogger(__name__)
        self.tracker = ComposeFileTracker()
        self.container_manager = ContainerManager(compose_file)
        
    def ensure_correct_container_running(self) -> Dict[str, Any]:
        """
        Ensure the correct container is running based on current COMPOSE_FILE.
        
        Returns:
            Connection parameters for the running container.
        """
        # Always re-initialize the container manager to get the latest compose file
        current_compose_file = self.tracker.get_current_compose_file()
        self.container_manager = ContainerManager(current_compose_file)

        # Check if compose file has changed
        if self.tracker.has_compose_file_changed():
            self.logger.info("Compose file changed, restarting container with new configuration")
            self._restart_with_new_compose_file()
        else:
            # Start container if not running (normal case)
            if not self.container_manager.is_iris_running():
                self.logger.info("Starting IRIS container")
                self.container_manager.start_iris()
        
        # Wait for container to be healthy
        self.container_manager.wait_for_health()
        
        return self.container_manager.get_connection_params()
    
    def _restart_with_new_compose_file(self) -> None:
        """Restart container with new compose file configuration."""
        try:
            # Update container manager with current compose file
            current_compose_file = self.tracker.get_current_compose_file()
            self.container_manager = ContainerManager(current_compose_file)
            
            # Stop any existing container and start with new config
            if self.container_manager.is_iris_running():
                self.logger.info("Stopping existing container before restart")
                self.container_manager.stop_iris()
            
            self.logger.info(f"Starting container with new compose file: {current_compose_file}")
            self.container_manager.start_iris()
            
        except Exception as e:
            self.logger.error(f"Failed to restart container with new compose file: {e}")
            raise
    
    def get_connection_params(self) -> Dict[str, Any]:
        """Get connection parameters for the current container."""
        return self.container_manager.get_connection_params()
    
    def is_container_running(self) -> bool:
        """Check if container is currently running."""
        return self.container_manager.is_iris_running()
    
    def cleanup(self) -> None:
        """Cleanup resources (optional, for explicit cleanup)."""
        # Could be extended to clean up state files if needed
        pass