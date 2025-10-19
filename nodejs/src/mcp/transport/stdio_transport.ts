/**
 * Stdio Transport for MCP Server.
 *
 * Implements JSON-RPC over stdin/stdout for local Claude Code integration.
 *
 * Feature: Complete MCP Tools Implementation
 * Branch: 043-complete-mcp-tools
 */

import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';

export class StdioTransport {
  private server: Server;
  private transport?: StdioServerTransport;

  constructor(server: Server) {
    this.server = server;
  }

  async start(): Promise<void> {
    this.transport = new StdioServerTransport();
    await this.server.connect(this.transport);
    console.error('MCP Server started on stdio transport');
  }

  async stop(): Promise<void> {
    if (this.transport) {
      await this.transport.close();
    }
  }

  getName(): string {
    return 'stdio';
  }
}
