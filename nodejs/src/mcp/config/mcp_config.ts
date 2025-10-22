/**
 * MCP Server Configuration.
 *
 * Loads configuration from environment variables.
 *
 * Feature: Complete MCP Tools Implementation
 * Branch: 043-complete-mcp-tools
 */

export interface MCPServerConfig {
  name: string;
  description: string;
  version: string;
  enabled_techniques: string[];
  transport: 'stdio' | 'http' | 'both';
  http_port?: number;
  auth_mode: 'api_key' | 'none';
  max_connections: number;
  python_bridge_url: string;
  environment?: string;
}

export function loadConfig(): MCPServerConfig {
  return {
    name: process.env.MCP_SERVER_NAME || 'iris-rag-mcp-server',
    description: process.env.MCP_SERVER_DESCRIPTION || 'MCP server for IRIS RAG pipelines',
    version: process.env.MCP_SERVER_VERSION || '1.0.0',
    enabled_techniques: process.env.MCP_ENABLED_TECHNIQUES?.split(',') || [
      'basic', 'basic_rerank', 'crag', 'graphrag', 'pylate_colbert', 'iris_global_graphrag'
    ],
    transport: (process.env.MCP_TRANSPORT as any) || 'stdio',
    http_port: parseInt(process.env.MCP_HTTP_PORT || '3000'),
    auth_mode: (process.env.MCP_AUTH_MODE as any) || 'none',
    max_connections: parseInt(process.env.MCP_MAX_CONNECTIONS || '5'),
    python_bridge_url: process.env.MCP_PYTHON_BRIDGE_URL || 'http://localhost:8001',
    environment: process.env.MCP_ENVIRONMENT || 'development'
  };
}
