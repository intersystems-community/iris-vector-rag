# tests/utils/container_manager.py

import subprocess
import time
import logging
import os

class ContainerManager:
    """Enhanced Docker container management for IRIS with lifecycle support."""

    def __init__(self, compose_file: str = None):
        """
        Initialize ContainerManager with configurable compose file.
        
        Args:
            compose_file: Path to docker-compose file. If None, uses COMPOSE_FILE
                         environment variable or defaults to "docker-compose.yml"
        """
        self.compose_file = compose_file or os.getenv("COMPOSE_FILE", "docker-compose.yml")
        self.logger = logging.getLogger(__name__)

        # Dynamically determine service and container names from compose file
        self.service_name, self.container_name = self._get_service_and_container_names()
    
    def is_iris_running(self) -> bool:
        """Check if IRIS container is running and healthy."""
        return self.is_container_healthy()
    
    def start_iris(self) -> None:
        """Start IRIS container using Docker Compose."""
        if self.is_iris_running():
            self.logger.info("IRIS container already running")
            return
        
        self.logger.info(f"Starting IRIS container with compose file: {self.compose_file}")
        subprocess.run([
            "docker-compose", "-f", self.compose_file, "up", "-d", self.service_name
        ], check=True)
        self.logger.info("Docker-compose start command executed successfully")
    
    def stop_iris(self) -> None:
        """Stop IRIS container using Docker Compose."""
        self.logger.info(f"Stopping IRIS container with compose file: {self.compose_file}")
        subprocess.run([
            "docker-compose", "-f", self.compose_file, "down"
        ], check=True)
        self.logger.info("Docker-compose stop command executed successfully")
    
    def restart_iris(self) -> None:
        """Restart IRIS container using Docker Compose."""
        self.logger.info(f"Restarting IRIS container with compose file: {self.compose_file}")
        self.stop_iris()
        time.sleep(2)  # Brief pause between stop and start
        self.start_iris()
    
    def wait_for_health(self, timeout: int = 120) -> None:
        """Wait for IRIS container to be healthy."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            logging.info("Waiting for health check...")
            is_healthy = self.is_container_healthy()
            logging.info(f"Health check result: {'healthy' if is_healthy else 'not healthy'}")
            if is_healthy:
                logging.info("IRIS container is healthy")
                return
            
            time.sleep(5)
        
        raise TimeoutError("IRIS container failed to become healthy")
    
    def is_container_healthy(self) -> bool:
        """Check if IRIS container is healthy."""
        try:
            result = subprocess.run([
                "docker", "inspect", "--format", "{{.State.Health.Status}}",
                self.container_name
            ], capture_output=True, text=True, check=True)
            
            return "healthy" in result.stdout
        except subprocess.CalledProcessError:
            return False
    
    def get_connection_params(self) -> dict:
        """Get connection parameters for IRIS."""
        return {
            "hostname": "localhost",
            "port": 1972,
            "namespace": "USER",
            "username": "SuperUser",
            "password": "SYS"
        }

    def _get_service_and_container_names(self):
        """Parse docker-compose file to get service and container names."""
        import yaml
        try:
            with open(self.compose_file, 'r') as f:
                compose_data = yaml.safe_load(f)
            
            services = compose_data.get('services', {})
            if not services:
                raise ValueError("No services found in compose file")
            
            # Get the first service (assuming one IRIS service per file)
            service_name = next(iter(services))
            service_data = services[service_name]
            
            container_name = service_data.get('container_name')
            if not container_name:
                raise ValueError(f"container_name not defined for service {service_name}")
                
            return service_name, container_name
            
        except Exception as e:
            self.logger.error(f"Failed to parse {self.compose_file}: {e}")
            # Fallback to default names
            return "iris_db", "iris_db_rag_standalone_community"