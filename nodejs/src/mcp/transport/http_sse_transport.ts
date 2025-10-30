/**
 * HTTP/SSE Transport for MCP Server.
 *
 * Implements Server-Sent Events transport for remote MCP clients.
 *
 * Feature: Complete MCP Tools Implementation
 * Branch: 043-complete-mcp-tools
 */

import express, { Express } from 'express';
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { SSEServerTransport } from '@modelcontextprotocol/sdk/server/sse.js';

export class HttpSseTransport {
  private server: Server;
  private app: Express;
  private httpPort: number;
  private httpServer?: any;

  constructor(server: Server, port: number = 3000) {
    this.server = server;
    this.httpPort = port;
    this.app = express();
  }

  async start(): Promise<void> {
    this.app.use(express.json());

    // SSE endpoint
    this.app.get('/sse', async (_req, res) => {
      const transport = new SSEServerTransport('/messages', res);
      await this.server.connect(transport);
    });

    // Messages endpoint
    this.app.post('/messages', async (_req, res) => {
      // Handle incoming messages
      res.status(200).send();
    });

    return new Promise((resolve) => {
      this.httpServer = this.app.listen(this.httpPort, () => {
        console.error(`MCP Server started on HTTP/SSE transport (port ${this.httpPort})`);
        resolve();
      });
    });
  }

  async stop(): Promise<void> {
    if (this.httpServer) {
      return new Promise((resolve) => {
        this.httpServer.close(() => resolve());
      });
    }
  }

  getName(): string {
    return 'http-sse';
  }
}
