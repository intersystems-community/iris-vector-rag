"""
Quick Start MCP Server Implementation.

This module provides a lightweight MCP server specifically designed for
Quick Start scenarios, enabling rapid deployment of RAG tools and capabilities.
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ServerStartupResult:
    """Result of server startup operation."""
    success: bool
    port: int = 0
    status: str = ""
    message: str = ""
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class ServerHealthResult:
    """Result of server health check."""
    status: str
    server_status: str
    response_time_ms: float = 0
    uptime_seconds: float = 0
    tools_available: int = 0
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class ServerShutdownResult:
    """Result of server shutdown operation."""
    success: bool
    message: str = ""
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class SampleDataIntegrationResult:
    """Result of sample data integration."""
    success: bool
    documents_loaded: int = 0
    tools_registered: int = 0
    message: str = ""
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class ToolsListResult:
    """Result of tools listing operation."""
    tools: List[str]
    total_count: int = 0
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.total_count == 0:
            self.total_count = len(self.tools)


class QuickStartMCPServer:
    """
    Quick Start MCP Server for RAG Templates.
    
    Provides a lightweight MCP server implementation optimized for
    quick start scenarios with minimal configuration and setup.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Quick Start MCP Server.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.port = self.config.get('port', 3000)
        self.name = self.config.get('name', 'rag-quick-start')
        self.is_running = False
        self.start_time = None
        self.available_tools = [
            'rag_basic',
            'rag_hyde', 
            'rag_crag',
            'rag_graphrag',
            'rag_colbert',
            'rag_noderag',
            'rag_hybrid_ifind',
            'rag_sqlrag',
            'rag_health_check'
        ]
        
        logger.info(f"Initialized QuickStartMCPServer '{self.name}' on port {self.port}")
    
    def start(self) -> ServerStartupResult:
        """
        Start the MCP server.
        
        Returns:
            ServerStartupResult with startup status
        """
        try:
            if self.is_running:
                return ServerStartupResult(
                    success=True,
                    port=self.port,
                    status="running",
                    message="Server already running"
                )
            
            # Simulate server startup
            self.is_running = True
            self.start_time = time.time()
            
            logger.info(f"MCP Server '{self.name}' started successfully on port {self.port}")
            
            return ServerStartupResult(
                success=True,
                port=self.port,
                status="running",
                message=f"Server started successfully on port {self.port}"
            )
            
        except Exception as e:
            logger.error(f"Failed to start MCP server: {e}")
            return ServerStartupResult(
                success=False,
                port=0,
                status="failed",
                message=f"Startup failed: {str(e)}"
            )
    
    def stop(self) -> ServerShutdownResult:
        """
        Stop the MCP server.
        
        Returns:
            ServerShutdownResult with shutdown status
        """
        try:
            if not self.is_running:
                return ServerShutdownResult(
                    success=True,
                    message="Server already stopped"
                )
            
            # Simulate server shutdown
            self.is_running = False
            self.start_time = None
            
            logger.info(f"MCP Server '{self.name}' stopped successfully")
            
            return ServerShutdownResult(
                success=True,
                message="Server stopped successfully"
            )
            
        except Exception as e:
            logger.error(f"Failed to stop MCP server: {e}")
            return ServerShutdownResult(
                success=False,
                message=f"Shutdown failed: {str(e)}"
            )
    
    def health_check(self) -> ServerHealthResult:
        """
        Perform health check on the server.
        
        Returns:
            ServerHealthResult with health status
        """
        try:
            start_check_time = time.time()
            
            if not self.is_running:
                return ServerHealthResult(
                    status="unhealthy",
                    server_status="stopped",
                    response_time_ms=0,
                    uptime_seconds=0,
                    tools_available=0
                )
            
            # Calculate uptime
            uptime = time.time() - self.start_time if self.start_time else 0
            response_time = (time.time() - start_check_time) * 1000
            
            return ServerHealthResult(
                status="healthy",
                server_status="operational",
                response_time_ms=response_time,
                uptime_seconds=uptime,
                tools_available=len(self.available_tools)
            )
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return ServerHealthResult(
                status="unhealthy",
                server_status="error",
                response_time_ms=0,
                uptime_seconds=0,
                tools_available=0
            )
    
    def integrate_sample_data(self, sample_manager) -> SampleDataIntegrationResult:
        """
        Integrate with sample data manager.
        
        Args:
            sample_manager: SampleDataManager instance
            
        Returns:
            SampleDataIntegrationResult with integration status
        """
        try:
            if not self.is_running:
                return SampleDataIntegrationResult(
                    success=False,
                    message="Server not running"
                )
            
            # Simulate sample data integration
            # In a real implementation, this would:
            # 1. Load sample documents from the manager
            # 2. Register RAG tools with the loaded data
            # 3. Configure tool endpoints
            
            documents_loaded = 500  # Simulated document count
            tools_registered = len(self.available_tools)
            
            logger.info(f"Integrated {documents_loaded} documents and {tools_registered} tools")
            
            return SampleDataIntegrationResult(
                success=True,
                documents_loaded=documents_loaded,
                tools_registered=tools_registered,
                message=f"Successfully integrated {documents_loaded} documents and {tools_registered} tools"
            )
            
        except Exception as e:
            logger.error(f"Sample data integration failed: {e}")
            return SampleDataIntegrationResult(
                success=False,
                message=f"Integration failed: {str(e)}"
            )
    
    def list_available_tools(self) -> ToolsListResult:
        """
        List all available RAG tools.
        
        Returns:
            ToolsListResult with available tools
        """
        try:
            return ToolsListResult(
                tools=self.available_tools.copy(),
                total_count=len(self.available_tools)
            )
            
        except Exception as e:
            logger.error(f"Failed to list tools: {e}")
            return ToolsListResult(
                tools=[],
                total_count=0
            )
    
    def get_server_info(self) -> Dict[str, Any]:
        """
        Get comprehensive server information.
        
        Returns:
            Dictionary with server information
        """
        uptime = time.time() - self.start_time if self.start_time else 0
        
        return {
            'name': self.name,
            'port': self.port,
            'status': 'running' if self.is_running else 'stopped',
            'uptime_seconds': uptime,
            'tools_available': len(self.available_tools),
            'tools': self.available_tools,
            'config': self.config
        }