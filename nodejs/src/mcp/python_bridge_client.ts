/**
 * Python Bridge Client.
 *
 * HTTP client for communicating with Python MCP bridge.
 *
 * Feature: Complete MCP Tools Implementation
 * Branch: 043-complete-mcp-tools
 */

import axios, { AxiosInstance } from 'axios';

export interface TechniqueResult {
  success: boolean;
  result?: {
    answer: string;
    retrieved_documents: any[];
    sources: string[];
    metadata: Record<string, any>;
    performance: {
      execution_time_ms: number;
      retrieval_time_ms: number;
      generation_time_ms: number;
      tokens_used: number;
    };
  };
  error?: string;
}

export interface HealthStatus {
  status: 'healthy' | 'degraded' | 'unavailable';
  timestamp: string;
  pipelines: Record<string, any>;
  database: Record<string, any>;
  performance_metrics?: Record<string, any>;
}

export class PythonBridgeClient {
  private client: AxiosInstance;

  constructor(baseUrl: string = 'http://localhost:8001') {
    this.client = axios.create({
      baseURL: baseUrl,
      timeout: 30000,
      headers: { 'Content-Type': 'application/json' }
    });
  }

  async invokeTechnique(
    technique: string,
    query: string,
    params: Record<string, any>,
    apiKey?: string
  ): Promise<TechniqueResult> {
    try {
      const response = await this.client.post('/mcp/invoke_technique', {
        technique,
        query,
        params,
        api_key: apiKey
      });
      return response.data;
    } catch (error: any) {
      return {
        success: false,
        error: error.message || 'Unknown error'
      };
    }
  }

  async listTechniques(): Promise<string[]> {
    try {
      const response = await this.client.get('/mcp/list_techniques');
      return response.data;
    } catch (error) {
      return [];
    }
  }

  async healthCheck(
    includeDetails: boolean = false,
    includePerformanceMetrics: boolean = true
  ): Promise<HealthStatus> {
    const response = await this.client.get('/mcp/health_check', {
      params: {
        include_details: includeDetails,
        include_performance_metrics: includePerformanceMetrics
      }
    });
    return response.data;
  }

  async getMetrics(
    timeRange: string = '1h',
    techniqueFilter?: string[],
    includeErrorDetails: boolean = false
  ): Promise<Record<string, any>> {
    const response = await this.client.get('/mcp/metrics', {
      params: {
        time_range: timeRange,
        technique_filter: techniqueFilter?.join(','),
        include_error_details: includeErrorDetails
      }
    });
    return response.data;
  }
}
