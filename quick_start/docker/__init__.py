"""
Docker-compose integration package for Quick Start system.

This package provides comprehensive Docker-compose integration that enables
containerized Quick Start environments with seamless integration to the
existing setup system.

Modules:
- compose_generator: Docker-compose file generation
- container_config: Container configuration management
- service_manager: Docker service management and operations
- volume_manager: Volume and network management
- templates: Docker-compose template system
"""

from .compose_generator import DockerComposeGenerator
from .container_config import ContainerConfigManager
from .service_manager import DockerServiceManager
from .volume_manager import VolumeManager

__all__ = [
    'DockerComposeGenerator',
    'ContainerConfigManager', 
    'DockerServiceManager',
    'VolumeManager'
]