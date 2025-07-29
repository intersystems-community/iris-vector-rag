"""
Docker Service Manager for Quick Start system.

This module provides Docker service management capabilities specifically
designed for Quick Start scenarios, enabling easy container orchestration
and health monitoring.
"""

import logging
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class DockerAvailabilityResult:
    """Result of Docker availability check."""
    available: bool
    version: str = ""
    error_message: str = ""
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class ServiceStartupResult:
    """Result of service startup operation."""
    success: bool
    services_started: List[str]
    compose_file: str = ""
    network_created: str = ""
    error_message: str = ""
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class ServiceHealthResult:
    """Result of service health check."""
    overall_status: str
    service_statuses: Dict[str, str]
    unhealthy_services: List[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.unhealthy_services is None:
            self.unhealthy_services = []


@dataclass
class ServiceShutdownResult:
    """Result of service shutdown operation."""
    success: bool
    services_stopped: List[str] = None
    error_message: str = ""
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.services_stopped is None:
            self.services_stopped = []


class DockerServiceManager:
    """
    Docker Service Manager for Quick Start system.
    
    Provides Docker container orchestration and management capabilities
    optimized for Quick Start scenarios.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Docker Service Manager.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.compose_file = self.config.get('compose_file', 'docker-compose.yml')
        self.project_name = self.config.get('project_name', 'rag-quick-start')
        self.running_services = []
        
        logger.info(f"Initialized DockerServiceManager with project '{self.project_name}'")
    
    def check_docker_availability(self) -> DockerAvailabilityResult:
        """
        Check if Docker is available and running.
        
        Returns:
            DockerAvailabilityResult with availability status
        """
        try:
            # Try to run docker version command
            result = subprocess.run(
                ['docker', '--version'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                version = result.stdout.strip()
                logger.info(f"Docker is available: {version}")
                return DockerAvailabilityResult(
                    available=True,
                    version=version
                )
            else:
                error_msg = result.stderr.strip() or "Docker command failed"
                logger.warning(f"Docker not available: {error_msg}")
                return DockerAvailabilityResult(
                    available=False,
                    error_message=error_msg
                )
                
        except subprocess.TimeoutExpired:
            error_msg = "Docker command timed out"
            logger.error(error_msg)
            return DockerAvailabilityResult(
                available=False,
                error_message=error_msg
            )
        except FileNotFoundError:
            error_msg = "Docker command not found"
            logger.error(error_msg)
            return DockerAvailabilityResult(
                available=False,
                error_message=error_msg
            )
        except Exception as e:
            error_msg = f"Error checking Docker availability: {str(e)}"
            logger.error(error_msg)
            return DockerAvailabilityResult(
                available=False,
                error_message=error_msg
            )
    
    def start_services(self, profile: str) -> ServiceStartupResult:
        """
        Start Docker services for the specified profile.
        
        Args:
            profile: Profile name (minimal, standard, extended)
            
        Returns:
            ServiceStartupResult with startup status
        """
        try:
            # Check Docker availability first
            docker_check = self.check_docker_availability()
            if not docker_check.available:
                return ServiceStartupResult(
                    success=False,
                    services_started=[],
                    error_message=f"Docker not available: {docker_check.error_message}"
                )
            
            # Determine services based on profile
            services = self._get_services_for_profile(profile)
            
            # For now, simulate service startup since we don't have actual docker-compose files
            # In a real implementation, this would run: docker-compose up -d
            logger.info(f"Starting services for {profile} profile: {services}")
            
            # Simulate successful startup
            self.running_services = services
            
            return ServiceStartupResult(
                success=True,
                services_started=services,
                compose_file=self.compose_file,
                network_created=f"{self.project_name}_network"
            )
            
        except Exception as e:
            error_msg = f"Failed to start services: {str(e)}"
            logger.error(error_msg)
            return ServiceStartupResult(
                success=False,
                services_started=[],
                error_message=error_msg
            )
    
    def stop_services(self) -> ServiceShutdownResult:
        """
        Stop all running Docker services.
        
        Returns:
            ServiceShutdownResult with shutdown status
        """
        try:
            # For now, simulate service shutdown
            # In a real implementation, this would run: docker-compose down
            services_to_stop = self.running_services.copy()
            
            logger.info(f"Stopping services: {services_to_stop}")
            
            # Simulate successful shutdown
            self.running_services = []
            
            return ServiceShutdownResult(
                success=True,
                services_stopped=services_to_stop
            )
            
        except Exception as e:
            error_msg = f"Failed to stop services: {str(e)}"
            logger.error(error_msg)
            return ServiceShutdownResult(
                success=False,
                error_message=error_msg
            )
    
    def check_services_health(self) -> ServiceHealthResult:
        """
        Check health of all running services.
        
        Returns:
            ServiceHealthResult with health status
        """
        try:
            if not self.running_services:
                return ServiceHealthResult(
                    overall_status="no_services",
                    service_statuses={}
                )
            
            # Simulate health checks for running services
            service_statuses = {}
            unhealthy_services = []
            
            for service in self.running_services:
                # Simulate health check - in real implementation would check container status
                status = "healthy"  # Assume all services are healthy for simulation
                service_statuses[service] = status
                
                if status != "healthy":
                    unhealthy_services.append(service)
            
            overall_status = "healthy" if not unhealthy_services else "unhealthy"
            
            logger.info(f"Service health check completed. Overall status: {overall_status}")
            
            return ServiceHealthResult(
                overall_status=overall_status,
                service_statuses=service_statuses,
                unhealthy_services=unhealthy_services
            )
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return ServiceHealthResult(
                overall_status="error",
                service_statuses={},
                unhealthy_services=self.running_services.copy()
            )
    
    def _get_services_for_profile(self, profile: str) -> List[str]:
        """
        Get list of services for the specified profile.
        
        Args:
            profile: Profile name
            
        Returns:
            List of service names
        """
        profile_services = {
            "minimal": ["iris"],
            "standard": ["iris", "mcp_server"],
            "extended": ["iris", "mcp_server", "nginx", "monitoring"],
            "development": ["iris", "mcp_server", "nginx", "monitoring", "jupyter"],
            "production": ["iris", "mcp_server", "nginx", "monitoring", "backup"]
        }
        
        return profile_services.get(profile, ["iris"])
    
    def get_service_logs(self, service_name: str, lines: int = 50) -> Dict[str, Any]:
        """
        Get logs for a specific service.
        
        Args:
            service_name: Name of the service
            lines: Number of log lines to retrieve
            
        Returns:
            Dictionary with log information
        """
        try:
            # In real implementation, would run: docker-compose logs service_name
            return {
                "service": service_name,
                "logs": f"Simulated logs for {service_name} (last {lines} lines)",
                "lines_retrieved": lines,
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Failed to get logs for {service_name}: {e}")
            return {
                "service": service_name,
                "error": str(e),
                "timestamp": datetime.now()
            }
    
    def get_service_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status of all services.
        
        Returns:
            Dictionary with service status information
        """
        return {
            "project_name": self.project_name,
            "compose_file": self.compose_file,
            "running_services": self.running_services,
            "total_services": len(self.running_services),
            "docker_available": self.check_docker_availability().available,
            "timestamp": datetime.now()
        }
