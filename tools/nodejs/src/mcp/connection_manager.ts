/**
 * Connection Manager.
 *
 * Manages client connections and enforces max 5 concurrent connection limit.
 *
 * Feature: Complete MCP Tools Implementation
 * Branch: 043-complete-mcp-tools
 */

export class ConnectionManager {
  private activeConnections: Set<string>;
  private maxConnections: number;

  constructor(maxConnections: number = 5) {
    this.activeConnections = new Set();
    this.maxConnections = maxConnections;
  }

  registerConnection(connectionId: string): boolean {
    if (this.isAtMaxConnections()) {
      return false;
    }
    this.activeConnections.add(connectionId);
    return true;
  }

  unregisterConnection(connectionId: string): void {
    this.activeConnections.delete(connectionId);
  }

  getActiveConnectionCount(): number {
    return this.activeConnections.size;
  }

  isAtMaxConnections(): boolean {
    return this.activeConnections.size >= this.maxConnections;
  }
}
