"""
MCP Bridge Configuration.

Loads configuration from environment variables and YAML files
for both authentication modes and deployment modes.

Feature: Complete MCP Tools Implementation
Branch: 043-complete-mcp-tools
"""

import os
from typing import Optional, Literal
from dataclasses import dataclass


@dataclass
class MCPConfig:
    """MCP bridge configuration."""

    # Authentication
    auth_mode: Literal['api_key', 'none'] = 'none'

    # Deployment
    deployment_mode: Literal['standalone', 'integrated'] = 'integrated'

    # Server
    python_bridge_host: str = 'localhost'
    python_bridge_port: int = 8001

    # Connection limits
    max_connections: int = 5

    # Environment
    environment: str = 'development'


def load_config() -> MCPConfig:
    """Load MCP configuration from environment variables."""
    return MCPConfig(
        auth_mode=os.getenv('MCP_AUTH_MODE', 'none'),  # type: ignore
        deployment_mode=os.getenv('MCP_DEPLOYMENT_MODE', 'integrated'),  # type: ignore
        python_bridge_host=os.getenv('MCP_BRIDGE_HOST', 'localhost'),
        python_bridge_port=int(os.getenv('MCP_BRIDGE_PORT', '8001')),
        max_connections=int(os.getenv('MCP_MAX_CONNECTIONS', '5')),
        environment=os.getenv('MCP_ENVIRONMENT', 'development')
    )
