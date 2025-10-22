"""
MCP Routes for REST API Integration.

Provides health and status endpoints for MCP server when running
in integrated mode with the REST API.

Feature: Complete MCP Tools Implementation
Branch: 043-complete-mcp-tools
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any

router = APIRouter(prefix="/api/v1/mcp", tags=["mcp"])


@router.get("/health")
async def mcp_health() -> Dict[str, Any]:
    """
    Get MCP server health status.

    Returns:
        Health status including available techniques and connections
    """
    try:
        from iris_rag.mcp.bridge import MCPBridge

        bridge = MCPBridge()
        health = await bridge.health_check(
            include_details=True,
            include_performance_metrics=True
        )

        return health

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MCP health check failed: {str(e)}")


@router.get("/status")
async def mcp_status() -> Dict[str, Any]:
    """
    Get MCP server status summary.

    Returns:
        Status including available techniques and connection info
    """
    try:
        from iris_rag.mcp.bridge import MCPBridge

        bridge = MCPBridge()
        techniques = await bridge.get_available_techniques()

        return {
            "status": "running",
            "techniques_available": len(techniques),
            "techniques": techniques,
            "max_connections": bridge.config.max_connections,
            "deployment_mode": bridge.config.deployment_mode
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MCP status check failed: {str(e)}")


@router.get("/tools")
async def mcp_list_tools() -> Dict[str, Any]:
    """
    List all available MCP tools.

    Returns:
        List of MCP tool definitions with schemas
    """
    try:
        from iris_rag.mcp.bridge import MCPBridge

        bridge = MCPBridge()
        tools = await bridge.list_tools()

        return {
            "tools": tools,
            "count": len(tools)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list MCP tools: {str(e)}")
