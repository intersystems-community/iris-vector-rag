/**
 * RAG Tools Manager.
 *
 * Manages MCP tool definitions loaded from Python bridge.
 *
 * Feature: Complete MCP Tools Implementation
 * Branch: 043-complete-mcp-tools
 */

import toolSchemas from '../../../specs/043-complete-mcp-tools/contracts/mcp_tool_schema.json';

export interface MCPTool {
  name: string;
  description: string;
  inputSchema: Record<string, any>;
}

export class RAGToolsManager {
  private tools: Map<string, MCPTool>;

  constructor() {
    this.tools = new Map();
    this.loadTools();
  }

  private loadTools(): void {
    for (const tool of toolSchemas.tools) {
      this.tools.set(tool.name, tool as MCPTool);
    }
  }

  createTools(): MCPTool[] {
    return Array.from(this.tools.values());
  }

  getTool(toolName: string): MCPTool | null {
    return this.tools.get(toolName) || null;
  }

  validateParams(toolName: string, params: Record<string, any>): Record<string, any> {
    const tool = this.getTool(toolName);
    if (!tool) {
      throw new Error(`Unknown tool: ${toolName}`);
    }

    const schema = tool.inputSchema;
    const validated = { ...params };

    // Apply defaults
    for (const [key, propSchema] of Object.entries(schema.properties || {})) {
      if (!(key in validated) && 'default' in (propSchema as any)) {
        validated[key] = (propSchema as any).default;
      }
    }

    return validated;
  }

  getToolSchemas(): Record<string, Record<string, any>> {
    const schemas: Record<string, Record<string, any>> = {};
    for (const [name, tool] of this.tools.entries()) {
      schemas[name] = tool.inputSchema;
    }
    return schemas;
  }
}
