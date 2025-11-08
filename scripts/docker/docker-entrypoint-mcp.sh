#!/bin/bash
# Docker entrypoint script for MCP server
#
# Supports two deployment modes:
# - MODE=standalone: Start Python bridge + Node.js MCP server
# - MODE=integrated: Start REST API with embedded MCP bridge
#
# Feature: Complete MCP Tools Implementation
# Branch: 043-complete-mcp-tools

set -e

echo "==================================================================="
echo "IRIS RAG MCP Server - Docker Container"
echo "==================================================================="
echo "MODE: ${MODE}"
echo "TRANSPORT: ${MCP_TRANSPORT}"
echo "IRIS_HOST: ${IRIS_HOST}:${IRIS_PORT}"
echo "NAMESPACE: ${IRIS_NAMESPACE}"
echo "MAX_CONNECTIONS: ${MAX_CONNECTIONS}"
echo "AUTH_MODE: ${AUTH_MODE}"
echo "==================================================================="

# Wait for IRIS database to be ready
echo "Waiting for IRIS database at ${IRIS_HOST}:${IRIS_PORT}..."
timeout=60
counter=0
until python3 -c "import iris; conn = iris.connect('${IRIS_HOST}', ${IRIS_PORT}, '${IRIS_NAMESPACE}', '${IRIS_USERNAME}', '${IRIS_PASSWORD}'); conn.close()" 2>/dev/null; do
    counter=$((counter + 1))
    if [ $counter -gt $timeout ]; then
        echo "ERROR: IRIS database not ready after ${timeout} seconds"
        exit 1
    fi
    echo "  Waiting for IRIS... (${counter}s)"
    sleep 1
done
echo "✓ IRIS database is ready"

# Execute based on deployment mode
if [ "$MODE" = "standalone" ]; then
    echo ""
    echo "Starting MCP Server in STANDALONE mode..."
    echo "  - Python Bridge: http://0.0.0.0:${PYTHON_BRIDGE_PORT}"
    if [ "$MCP_TRANSPORT" = "http" ] || [ "$MCP_TRANSPORT" = "both" ]; then
        echo "  - MCP Server (HTTP/SSE): http://0.0.0.0:${MCP_HTTP_PORT}"
    fi
    if [ "$MCP_TRANSPORT" = "stdio" ] || [ "$MCP_TRANSPORT" = "both" ]; then
        echo "  - MCP Server (stdio): connected"
    fi
    echo ""

    # Start Python FastAPI bridge in background
    echo "Starting Python MCP Bridge..."
    uvicorn iris_rag.mcp.bridge:app \
        --host 0.0.0.0 \
        --port ${PYTHON_BRIDGE_PORT} \
        --log-level info &

    PYTHON_PID=$!
    echo "  Python Bridge started (PID: ${PYTHON_PID})"

    # Wait for Python bridge to be ready
    echo "  Waiting for Python bridge to be ready..."
    sleep 5

    # Start Node.js MCP server in foreground
    echo "Starting Node.js MCP Server..."
    cd /app/nodejs
    exec node dist/mcp/cli.js

elif [ "$MODE" = "integrated" ]; then
    echo ""
    echo "Starting MCP Server in INTEGRATED mode..."
    echo "  - REST API + MCP: http://0.0.0.0:8000"
    echo "  - API Docs: http://0.0.0.0:8000/docs"
    echo "  - MCP Health: http://0.0.0.0:8000/api/v1/mcp/health"
    echo ""

    # Start REST API with MCP integration
    exec uvicorn iris_rag.api.main:app \
        --host 0.0.0.0 \
        --port 8000 \
        --log-level info

else
    echo "ERROR: Unknown MODE='${MODE}'"
    echo "Valid modes: standalone, integrated"
    exit 1
fi
