#!/usr/bin/env node
/**
 * MCP Server CLI.
 *
 * Command-line interface for starting the MCP server in standalone mode.
 *
 * Feature: Complete MCP Tools Implementation
 * Branch: 043-complete-mcp-tools
 */

import { MCPServer } from './server.js';
import { loadConfig } from './config/mcp_config.js';

async function main() {
  const config = loadConfig();

  const server = new MCPServer(config);

  // Handle shutdown signals
  process.on('SIGINT', async () => {
    console.error('\nReceived SIGINT, shutting down gracefully...');
    await server.stop();
    process.exit(0);
  });

  process.on('SIGTERM', async () => {
    console.error('\nReceived SIGTERM, shutting down gracefully...');
    await server.stop();
    process.exit(0);
  });

  try {
    await server.start();
  } catch (error: any) {
    console.error(`Failed to start MCP server: ${error.message}`);
    process.exit(1);
  }
}

main().catch((error) => {
  console.error(`Fatal error: ${error.message}`);
  process.exit(1);
});
