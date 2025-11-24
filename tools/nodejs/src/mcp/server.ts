/**
 * Main MCP Server.
 *
 * Orchestrates all components (transports, tools, Python bridge).
 *
 * Feature: Complete MCP Tools Implementation
 * Branch: 043-complete-mcp-tools
 */

import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from '@modelcontextprotocol/sdk/types.js';
import { PythonBridgeClient } from './python_bridge_client.js';
import { RAGToolsManager } from './rag_tools.js';
import { ConnectionManager } from './connection_manager.js';
import { MCPServerConfig } from './config/mcp_config.js';
import { StdioTransport } from './transport/stdio_transport.js';
import { HttpSseTransport } from './transport/http_sse_transport.js';

export class MCPServer {
  private server: Server;
  private config: MCPServerConfig;
  private pythonBridge: PythonBridgeClient;
  private toolsManager: RAGToolsManager;
  private connectionManager: ConnectionManager;
  private transports: Array<StdioTransport | HttpSseTransport>;

  constructor(config: MCPServerConfig) {
    this.config = config;
    this.pythonBridge = new PythonBridgeClient(config.python_bridge_url);
    this.toolsManager = new RAGToolsManager();
    this.connectionManager = new ConnectionManager(config.max_connections);
    this.transports = [];

    // Initialize MCP SDK server
    this.server = new Server(
      {
        name: config.name,
        version: config.version,
      },
      {
        capabilities: {
          tools: {},
        },
      }
    );

    this.setupHandlers();
  }

  private setupHandlers(): void {
    // List tools handler
    this.server.setRequestHandler(ListToolsRequestSchema, async () => {
      const tools = this.toolsManager.createTools();
      return { tools };
    });

    // Call tool handler
    this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
      const { name: toolName, arguments: toolArgs } = request.params;

      try {
        // Validate parameters
        const validatedParams = this.toolsManager.validateParams(
          toolName,
          toolArgs || {}
        );

        // Extract query from params
        const query = (toolArgs as any)?.query || '';

        // Determine technique name (remove 'rag_' prefix)
        const technique = toolName.replace('rag_', '');

        // Invoke Python bridge
        const result = await this.pythonBridge.invokeTechnique(
          technique,
          query,
          validatedParams
        );

        if (result.success && result.result) {
          return {
            content: [
              {
                type: 'text',
                text: JSON.stringify(result.result, null, 2),
              },
            ],
          };
        } else {
          return {
            content: [
              {
                type: 'text',
                text: `Error: ${result.error}`,
              },
            ],
            isError: true,
          };
        }
      } catch (error: any) {
        return {
          content: [
            {
              type: 'text',
              text: `Error: ${error.message}`,
            },
          ],
          isError: true,
        };
      }
    });
  }

  async start(): Promise<void> {
    console.error(`Starting MCP Server: ${this.config.name} v${this.config.version}`);
    console.error(`Transport mode: ${this.config.transport}`);
    console.error(`Max connections: ${this.config.max_connections}`);

    // Start appropriate transports
    if (this.config.transport === 'stdio' || this.config.transport === 'both') {
      const stdioTransport = new StdioTransport(this.server);
      await stdioTransport.start();
      this.transports.push(stdioTransport);
    }

    if (this.config.transport === 'http' || this.config.transport === 'both') {
      const httpTransport = new HttpSseTransport(this.server, this.config.http_port);
      await httpTransport.start();
      this.transports.push(httpTransport);
    }

    console.error('MCP Server is ready');
  }

  async stop(): Promise<void> {
    for (const transport of this.transports) {
      await transport.stop();
    }
    console.error('MCP Server stopped');
  }

  async healthCheck(): Promise<Record<string, any>> {
    const health = await this.pythonBridge.healthCheck(true, true);
    return {
      status: health.status,
      techniques_available: this.config.enabled_techniques.length,
      active_connections: this.connectionManager.getActiveConnectionCount(),
      max_connections: this.config.max_connections,
    };
  }
}
