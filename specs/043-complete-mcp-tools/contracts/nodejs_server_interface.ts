/**
 * Node.js MCP Server Interface Contract.
 *
 * This module defines TypeScript interfaces for the Node.js MCP server
 * that handles protocol communication with MCP clients (Claude Code).
 *
 * Feature: Complete MCP Tools Implementation
 * Branch: 043-complete-mcp-tools
 */

/**
 * MCP Server Configuration
 */
export interface MCPServerConfig {
  /** Server name (shown in Claude Code) */
  name: string;

  /** Human-readable server description */
  description: string;

  /** Server version (semantic versioning) */
  version: string;

  /** List of enabled RAG techniques */
  enabled_techniques: string[];

  /** Transport mechanism (stdio, http, or both) */
  transport: 'stdio' | 'http' | 'both';

  /** HTTP port (only used if transport includes 'http') */
  http_port?: number;

  /** Authentication mode */
  auth_mode: 'api_key' | 'none';

  /** Maximum concurrent client connections */
  max_connections: number;

  /** Python bridge base URL */
  python_bridge_url: string;

  /** Environment (development, staging, production) */
  environment?: string;
}

/**
 * MCP Tool Definition
 */
export interface MCPTool {
  /** Tool name (e.g., "rag_basic") */
  name: string;

  /** Human-readable description */
  description: string;

  /** JSON Schema for tool parameters */
  inputSchema: Record<string, any>;
}

/**
 * MCP Request Message
 */
export interface MCPRequest {
  /** Request ID for tracing */
  id: string;

  /** Method name (e.g., "tools/call") */
  method: string;

  /** Request parameters */
  params: Record<string, any>;
}

/**
 * MCP Response Message
 */
export interface MCPResponse {
  /** Request ID (matches request) */
  id: string;

  /** Success result (mutually exclusive with error) */
  result?: Record<string, any>;

  /** Error details (mutually exclusive with result) */
  error?: MCPError;
}

/**
 * MCP Error
 */
export interface MCPError {
  /** Error code (e.g., "invalid_params", "internal_error") */
  code: string;

  /** Human-readable error message */
  message: string;

  /** Additional error context */
  data?: Record<string, any>;
}

/**
 * Health Status
 */
export interface HealthStatus {
  /** Overall health status */
  status: 'healthy' | 'degraded' | 'unavailable';

  /** Number of techniques available */
  techniques_available: number;

  /** Number of active client connections */
  active_connections: number;

  /** Maximum allowed connections */
  max_connections: number;

  /** Timestamp of health check */
  timestamp: string;
}

/**
 * Transport Interface
 *
 * Abstraction over stdio and HTTP/SSE transports.
 */
export interface ITransport {
  /**
   * Start the transport layer.
   */
  start(): Promise<void>;

  /**
   * Stop the transport layer.
   */
  stop(): Promise<void>;

  /**
   * Register handler for incoming messages.
   *
   * @param handler - Function to handle incoming MCP messages
   */
  onMessage(handler: (message: MCPRequest) => Promise<MCPResponse>): void;

  /**
   * Send a message to the client.
   *
   * @param message - MCP response to send
   */
  sendMessage(message: MCPResponse): Promise<void>;

  /**
   * Get transport name.
   */
  getName(): string;
}

/**
 * Python Bridge Client Interface
 *
 * Client for communicating with Python MCP bridge via HTTP.
 */
export interface IPythonBridgeClient {
  /**
   * Invoke a RAG technique.
   *
   * @param technique - Pipeline name (basic, crag, etc.)
   * @param query - User's question
   * @param params - Pipeline-specific parameters
   * @param apiKey - Optional API key for authentication
   * @returns Technique execution result
   */
  invokeTechnique(
    technique: string,
    query: string,
    params: Record<string, any>,
    apiKey?: string
  ): Promise<TechniqueResult>;

  /**
   * Get list of available techniques.
   *
   * @returns Array of technique names
   */
  listTechniques(): Promise<string[]>;

  /**
   * Check health of Python bridge and RAG pipelines.
   *
   * @param includeDetails - Include detailed metrics
   * @param includePerformanceMetrics - Include performance stats
   * @returns Health status
   */
  healthCheck(
    includeDetails?: boolean,
    includePerformanceMetrics?: boolean
  ): Promise<BridgeHealthStatus>;

  /**
   * Get performance metrics.
   *
   * @param timeRange - Time range (5m, 15m, 1h, etc.)
   * @param techniqueFilter - Filter by specific techniques
   * @param includeErrorDetails - Include error details
   * @returns Performance metrics
   */
  getMetrics(
    timeRange?: string,
    techniqueFilter?: string[],
    includeErrorDetails?: boolean
  ): Promise<PerformanceMetrics>;
}

/**
 * Technique Execution Result
 */
export interface TechniqueResult {
  /** Success flag */
  success: boolean;

  /** Result data (present if success=true) */
  result?: {
    answer: string;
    retrieved_documents: Document[];
    sources: string[];
    metadata: Record<string, any>;
    performance: {
      execution_time_ms: number;
      retrieval_time_ms: number;
      generation_time_ms: number;
      tokens_used: number;
    };
  };

  /** Error message (present if success=false) */
  error?: string;
}

/**
 * Retrieved Document
 */
export interface Document {
  /** Document ID */
  doc_id: string;

  /** Document content */
  content: string;

  /** Relevance score */
  score: number;

  /** Document metadata */
  metadata: Record<string, any>;
}

/**
 * Bridge Health Status
 */
export interface BridgeHealthStatus {
  /** Overall status */
  status: 'healthy' | 'degraded' | 'unavailable';

  /** Timestamp */
  timestamp: string;

  /** Individual pipeline statuses */
  pipelines: Record<string, PipelineStatus>;

  /** Database status */
  database: DatabaseStatus;

  /** Performance metrics (optional) */
  performance_metrics?: PerformanceMetrics;
}

/**
 * Pipeline Status
 */
export interface PipelineStatus {
  /** Pipeline health status */
  status: 'healthy' | 'degraded' | 'unavailable';

  /** Last successful query timestamp */
  last_success: string;

  /** Error rate (0.0-1.0) */
  error_rate: number;
}

/**
 * Database Status
 */
export interface DatabaseStatus {
  /** Connection status */
  connected: boolean;

  /** Ping latency in milliseconds */
  response_time_ms: number;

  /** Connection pool usage (e.g., "5/20") */
  connection_pool_usage: string;
}

/**
 * Performance Metrics
 */
export interface PerformanceMetrics {
  /** Time range for metrics */
  time_range?: string;

  /** Total queries */
  total_queries?: number;

  /** Successful queries */
  successful_queries?: number;

  /** Failed queries */
  failed_queries?: number;

  /** Average response time (p50) */
  average_response_time_ms: number;

  /** 95th percentile response time */
  p95_response_time_ms: number;

  /** 99th percentile response time */
  p99_response_time_ms?: number;

  /** Error rate (0.0-1.0) */
  error_rate: number;

  /** Queries per minute */
  queries_per_minute: number;

  /** Technique usage breakdown */
  technique_usage?: Record<string, number>;

  /** Error breakdown (optional) */
  error_breakdown?: Record<string, number>;
}

/**
 * RAG Tools Manager Interface
 *
 * Manages MCP tool registration and execution.
 */
export interface IRAGToolsManager {
  /**
   * Create MCP tool definitions from available techniques.
   *
   * @returns Array of MCP tool definitions
   */
  createTools(): MCPTool[];

  /**
   * Get tool definition by name.
   *
   * @param toolName - Tool name (e.g., "rag_basic")
   * @returns Tool definition or null if not found
   */
  getTool(toolName: string): MCPTool | null;

  /**
   * Validate tool parameters against schema.
   *
   * @param toolName - Tool name
   * @param params - Parameters to validate
   * @returns Validated parameters with defaults applied
   * @throws ValidationError if params are invalid
   */
  validateParams(toolName: string, params: Record<string, any>): Record<string, any>;

  /**
   * Get all tool schemas.
   *
   * @returns Map of tool names to schemas
   */
  getToolSchemas(): Record<string, Record<string, any>>;
}

/**
 * MCP Server Interface
 *
 * Main server managing client connections and tool execution.
 */
export interface IMCPServer {
  /**
   * Start the MCP server.
   *
   * @returns Server start result
   */
  start(): Promise<{ success: boolean; server_info: ServerInfo; error?: string }>;

  /**
   * Stop the MCP server.
   *
   * @returns Server stop result
   */
  stop(): Promise<{ success: boolean; error?: string }>;

  /**
   * Handle an MCP tool call.
   *
   * @param toolName - Tool to invoke
   * @param params - Tool parameters
   * @returns Tool execution result
   */
  handleToolCall(toolName: string, params: Record<string, any>): Promise<MCPResponse>;

  /**
   * List all available tools.
   *
   * @returns Array of tool definitions
   */
  listTools(): Promise<MCPTool[]>;

  /**
   * Check server health.
   *
   * @returns Health status
   */
  healthCheck(): Promise<HealthStatus>;
}

/**
 * Server Info
 */
export interface ServerInfo {
  /** Server name */
  name: string;

  /** Server version */
  version: string;

  /** Active transport */
  transport: string;

  /** Enabled techniques */
  enabled_techniques: string[];
}

/**
 * Connection Manager Interface
 *
 * Manages client connections and enforces connection limits.
 */
export interface IConnectionManager {
  /**
   * Register a new client connection.
   *
   * @param connectionId - Unique connection ID
   * @returns True if connection accepted, false if limit exceeded
   */
  registerConnection(connectionId: string): boolean;

  /**
   * Unregister a client connection.
   *
   * @param connectionId - Connection ID to remove
   */
  unregisterConnection(connectionId: string): void;

  /**
   * Get number of active connections.
   *
   * @returns Active connection count
   */
  getActiveConnectionCount(): number;

  /**
   * Check if connection limit is reached.
   *
   * @returns True if at max connections
   */
  isAtMaxConnections(): boolean;
}
