# Python-Node.js Bridge Architecture

## 1. Overview

This document defines the architecture for the Python-Node.js bridge interface that enables seamless communication between the Node.js MCP server layer and the Python RAG core. The bridge provides process management, load balancing, error handling, and performance optimization for enterprise-scale RAG operations.

## 2. Bridge Architecture Principles

### 2.1 Design Principles

- **Process Isolation**: Python processes run in isolated environments for stability and security
- **Asynchronous Communication**: Non-blocking message passing with streaming support
- **Load Balancing**: Intelligent distribution of requests across Python worker processes
- **Fault Tolerance**: Automatic recovery from process failures and resource exhaustion
- **Performance Optimization**: Connection pooling, caching, and resource management

### 2.2 Communication Patterns

- **Request-Response**: Standard synchronous operations with timeout handling
- **Streaming**: Real-time data streaming for large responses
- **Pub-Sub**: Event-driven notifications for monitoring and health checks
- **Batch Processing**: Efficient handling of multiple requests

## 3. Bridge Architecture Overview

### 3.1 High-Level Bridge Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    PYTHON-NODE.JS BRIDGE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Node.js Layer                                                  │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                Bridge Manager                               │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │ │
│  │  │ Process     │  │ Load        │  │ Message     │         │ │
│  │  │ Pool Mgr    │  │ Balancer    │  │ Router      │         │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘         │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                │                                │
│                                ▼                                │
│  Communication Layer                                            │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                JSON-RPC Protocol                            │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │ │
│  │  │ Serializer  │  │ Transport   │  │ Error       │         │ │
│  │  │ & Validator │  │ Manager     │  │ Handler     │         │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘         │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                │                                │
│                                ▼                                │
│  Python Worker Processes                                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ Worker 1    │  │ Worker 2    │  │ Worker N    │             │
│  │             │  │             │  │             │             │
│  │ • Pipeline  │  │ • Pipeline  │  │ • Pipeline  │             │
│  │   Registry  │  │   Registry  │  │   Registry  │             │
│  │ • Vector    │  │ • Vector    │  │ • Vector    │             │
│  │   Store     │  │   Store     │  │   Store     │             │
│  │ • Config    │  │ • Config    │  │ • Config    │             │
│  │   Manager   │  │   Manager   │  │   Manager   │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Process Management Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   PROCESS MANAGEMENT                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Process Pool Manager                                           │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ • Worker process lifecycle management                       │ │
│  │ • Health monitoring and auto-restart                        │ │
│  │ • Resource allocation and limits                            │ │
│  │ • Graceful shutdown and cleanup                             │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                │                                │
│                                ▼                                │
│  Worker Process Pool                                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ Worker 1    │  │ Worker 2    │  │ Worker N    │             │
│  │ Status:     │  │ Status:     │  │ Status:     │             │
│  │ • Ready     │  │ • Busy      │  │ • Starting  │             │
│  │ • CPU: 45%  │  │ • CPU: 78%  │  │ • CPU: 12%  │             │
│  │ • Mem: 2.1GB│  │ • Mem: 3.4GB│  │ • Mem: 0.8GB│             │
│  │ • Requests: │  │ • Requests: │  │ • Requests: │             │
│  │   1,247     │  │   892       │  │   0         │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
│                                │                                │
│                                ▼                                │
│  Load Balancer                                                  │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ • Round-robin with health awareness                         │ │
│  │ • Technique-specific routing                                │ │
│  │ • Request queuing and backpressure                          │ │
│  │ • Performance-based worker selection                        │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## 4. Communication Protocol

### 4.1 JSON-RPC Protocol Specification

```typescript
// Request Format
interface BridgeRequest {
  jsonrpc: "2.0";
  method: string;
  params: {
    technique: string;
    query: string;
    options?: Record<string, any>;
    technique_params?: Record<string, any>;
  };
  id: string | number;
  metadata?: {
    request_id: string;
    timestamp: string;
    priority: 'low' | 'normal' | 'high';
    timeout_ms: number;
  };
}

// Response Format
interface BridgeResponse {
  jsonrpc: "2.0";
  result?: {
    success: boolean;
    technique: string;
    query: string;
    answer: string;
    retrieved_documents: any[];
    performance: {
      total_time_ms: number;
      retrieval_time_ms: number;
      generation_time_ms: number;
    };
    metadata: Record<string, any>;
  };
  error?: {
    code: number;
    message: string;
    data?: any;
  };
  id: string | number;
}

// Streaming Response Format
interface StreamingResponse {
  jsonrpc: "2.0";
  method: "stream_chunk";
  params: {
    request_id: string;
    chunk_type: 'partial_answer' | 'document' | 'metadata';
    data: any;
    is_final: boolean;
  };
}
```

### 4.2 Method Specifications

```typescript
// Core RAG Methods
interface BridgeMethods {
  // Pipeline execution
  'rag.execute': (params: RAGExecuteParams) => Promise<RAGResponse>;
  'rag.stream': (params: RAGExecuteParams) => AsyncIterator<StreamChunk>;
  
  // Pipeline management
  'pipeline.initialize': (technique: string, config: any) => Promise<string>;
  'pipeline.destroy': (pipeline_id: string) => Promise<void>;
  'pipeline.list': () => Promise<string[]>;
  
  // Health and monitoring
  'health.check': () => Promise<HealthStatus>;
  'health.detailed': () => Promise<DetailedHealthStatus>;
  'metrics.get': (time_range?: string) => Promise<PerformanceMetrics>;
  
  // Configuration
  'config.update': (config: any) => Promise<void>;
  'config.validate': (config: any) => Promise<ValidationResult>;
  'config.reload': () => Promise<void>;
  
  // Utility methods
  'system.info': () => Promise<SystemInfo>;
  'cache.clear': (cache_type?: string) => Promise<void>;
}
```

## 5. Node.js Bridge Implementation

### 5.1 Bridge Manager

```typescript
class PythonBridgeManager {
  private processPool: ProcessPool;
  private loadBalancer: LoadBalancer;
  private messageRouter: MessageRouter;
  private healthMonitor: HealthMonitor;
  private config: BridgeConfig;
  
  constructor(config: BridgeConfig) {
    this.config = config;
    this.processPool = new ProcessPool(config.pool);
    this.loadBalancer = new LoadBalancer(config.loadBalancer);
    this.messageRouter = new MessageRouter();
    this.healthMonitor = new HealthMonitor(config.health);
  }
  
  async initialize(): Promise<void> {
    // Initialize process pool
    await this.processPool.initialize();
    
    // Start health monitoring
    await this.healthMonitor.start();
    
    // Setup message routing
    this.setupMessageRouting();
    
    console.log('Python bridge initialized successfully');
  }
  
  async execute(method: string, params: any): Promise<any> {
    const requestId = this.generateRequestId();
    const startTime = Date.now();
    
    try {
      // Select worker process
      const worker = await this.loadBalancer.selectWorker(method, params);
      
      // Route message to worker
      const response = await this.messageRouter.send(worker, {
        jsonrpc: '2.0',
        method,
        params,
        id: requestId,
        metadata: {
          request_id: requestId,
          timestamp: new Date().toISOString(),
          timeout_ms: this.config.timeout
        }
      });
      
      // Record metrics
      this.recordMetrics(method, Date.now() - startTime, true);
      
      return response.result;
      
    } catch (error) {
      this.recordMetrics(method, Date.now() - startTime, false, error);
      throw error;
    }
  }
  
  async stream(method: string, params: any): Promise<AsyncIterator<any>> {
    const worker = await this.loadBalancer.selectWorker(method, params);
    return this.messageRouter.stream(worker, method, params);
  }
}
```

### 5.2 Process Pool Management

```typescript
class ProcessPool {
  private workers: Map<string, WorkerProcess>;
  private config: PoolConfig;
  private healthChecker: ProcessHealthChecker;
  
  constructor(config: PoolConfig) {
    this.config = config;
    this.workers = new Map();
    this.healthChecker = new ProcessHealthChecker();
  }
  
  async initialize(): Promise<void> {
    // Create initial worker processes
    for (let i = 0; i < this.config.minWorkers; i++) {
      await this.createWorker();
    }
    
    // Start health monitoring
    this.startHealthMonitoring();
  }
  
  private async createWorker(): Promise<WorkerProcess> {
    const workerId = `worker_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    const worker = new WorkerProcess({
      id: workerId,
      pythonPath: this.config.pythonPath,
      scriptPath: this.config.bridgeScript,
      env: {
        ...process.env,
        WORKER_ID: workerId,
        PROJECT_ROOT: this.config.projectRoot
      },
      timeout: this.config.workerTimeout,
      maxMemory: this.config.maxMemoryPerWorker
    });
    
    await worker.start();
    this.workers.set(workerId, worker);
    
    return worker;
  }
  
  async getAvailableWorker(): Promise<WorkerProcess> {
    // Find least busy worker
    let bestWorker: WorkerProcess | null = null;
    let lowestLoad = Infinity;
    
    for (const worker of this.workers.values()) {
      if (worker.isHealthy() && worker.getLoad() < lowestLoad) {
        bestWorker = worker;
        lowestLoad = worker.getLoad();
      }
    }
    
    if (!bestWorker) {
      // Create new worker if under max limit
      if (this.workers.size < this.config.maxWorkers) {
        bestWorker = await this.createWorker();
      } else {
        throw new Error('No available workers and max limit reached');
      }
    }
    
    return bestWorker;
  }
  
  private startHealthMonitoring(): void {
    setInterval(async () => {
      for (const [workerId, worker] of this.workers) {
        const health = await this.healthChecker.check(worker);
        
        if (!health.isHealthy) {
          console.warn(`Worker ${workerId} unhealthy, restarting...`);
          await this.restartWorker(workerId);
        }
      }
    }, this.config.healthCheckInterval);
  }
  
  private async restartWorker(workerId: string): Promise<void> {
    const worker = this.workers.get(workerId);
    if (worker) {
      await worker.stop();
      this.workers.delete(workerId);
      await this.createWorker();
    }
  }
}
```

### 5.3 Load Balancer

```typescript
class LoadBalancer {
  private strategy: LoadBalancingStrategy;
  private techniqueRouting: Map<string, string[]>;
  private requestQueue: RequestQueue;
  
  constructor(config: LoadBalancerConfig) {
    this.strategy = this.createStrategy(config.strategy);
    this.techniqueRouting = new Map(config.techniqueRouting);
    this.requestQueue = new RequestQueue(config.queueSize);
  }
  
  async selectWorker(method: string, params: any): Promise<WorkerProcess> {
    const technique = this.extractTechnique(method, params);
    const availableWorkers = await this.getAvailableWorkers(technique);
    
    if (availableWorkers.length === 0) {
      // Queue request if no workers available
      return await this.requestQueue.wait(technique);
    }
    
    return this.strategy.select(availableWorkers, method, params);
  }
  
  private async getAvailableWorkers(technique?: string): Promise<WorkerProcess[]> {
    const allWorkers = await this.processPool.getHealthyWorkers();
    
    if (!technique) {
      return allWorkers;
    }
    
    // Filter workers that support the technique
    const supportedWorkers = this.techniqueRouting.get(technique);
    if (supportedWorkers) {
      return allWorkers.filter(w => supportedWorkers.includes(w.id));
    }
    
    return allWorkers;
  }
  
  private createStrategy(strategyName: string): LoadBalancingStrategy {
    switch (strategyName) {
      case 'round_robin':
        return new RoundRobinStrategy();
      case 'least_connections':
        return new LeastConnectionsStrategy();
      case 'performance_based':
        return new PerformanceBasedStrategy();
      default:
        return new RoundRobinStrategy();
    }
  }
}
```

## 6. Python Worker Implementation

### 6.1 Worker Process Structure

```python
# iris_rag/bridge/worker.py
import asyncio
import json
import sys
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from ..pipelines.registry import PipelineRegistry
from ..pipelines.factory import PipelineFactory
from ..config.manager import ConfigurationManager
from ..monitoring.performance_monitor import PerformanceMonitor

class BridgeWorker:
    """Python worker process for handling RAG requests from Node.js bridge."""
    
    def __init__(self, worker_id: str):
        self.worker_id = worker_id
        self.config_manager = ConfigurationManager()
        self.performance_monitor = PerformanceMonitor(self.config_manager)
        
        # Initialize pipeline components
        self.pipeline_factory = PipelineFactory(self.config_manager)
        self.pipeline_registry = PipelineRegistry(self.pipeline_factory)
        
        # Request handling
        self.request_handlers = self._setup_handlers()
        self.active_requests: Dict[str, Any] = {}
        
        # Health status
        self.start_time = datetime.now()
        self.request_count = 0
        self.error_count = 0
        
    def _setup_handlers(self) -> Dict[str, callable]:
        """Setup method handlers for JSON-RPC requests."""
        return {
            'rag.execute': self.handle_rag_execute,
            'rag.stream': self.handle_rag_stream,
            'pipeline.initialize': self.handle_pipeline_initialize,
            'pipeline.destroy': self.handle_pipeline_destroy,
            'pipeline.list': self.handle_pipeline_list,
            'health.check': self.handle_health_check,
            'health.detailed': self.handle_health_detailed,
            'metrics.get': self.handle_metrics_get,
            'config.update': self.handle_config_update,
            'config.validate': self.handle_config_validate,
            'system.info': self.handle_system_info,
            'cache.clear': self.handle_cache_clear
        }
    
    async def start(self):
        """Start the worker process and begin listening for requests."""
        try:
            # Initialize pipeline registry
            await self.pipeline_registry.register_pipelines()
            
            # Start performance monitoring
            self.performance_monitor.start_monitoring()
            
            # Begin request processing loop
            await self.process_requests()
            
        except Exception as e:
            logging.error(f"Worker {self.worker_id} failed to start: {e}")
            sys.exit(1)
    
    async def process_requests(self):
        """Main request processing loop."""
        while True:
            try:
                # Read request from stdin
                line = await self._read_line()
                if not line:
                    break
                
                request = json.loads(line)
                response = await self.handle_request(request)
                
                # Send response to stdout
                await self._write_response(response)
                
            except Exception as e:
                logging.error(f"Error processing request: {e}")
                error_response = self._create_error_response(
                    None, -32603, f"Internal error: {str(e)}"
                )
                await self._write_response(error_response)
    
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a single JSON-RPC request."""
        request_id = request.get('id')
        method = request.get('method')
        params = request.get('params', {})
        
        self.request_count += 1
        start_time = datetime.now()
        
        try:
            # Validate request format
            if not self._validate_request(request):
                return self._create_error_response(
                    request_id, -32600, "Invalid Request"
                )
            
            # Get handler for method
            handler = self.request_handlers.get(method)
            if not handler:
                return self._create_error_response(
                    request_id, -32601, f"Method not found: {method}"
                )
            
            # Execute handler
            result = await handler(params)
            
            # Record performance metrics
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            self.performance_monitor.record_metric(
                'request_execution_time', execution_time, 'ms',
                {'method': method, 'worker_id': self.worker_id}
            )
            
            return {
                'jsonrpc': '2.0',
                'result': result,
                'id': request_id
            }
            
        except Exception as e:
            self.error_count += 1
            logging.error(f"Error handling request {request_id}: {e}")
            
            return self._create_error_response(
                request_id, -32603, f"Internal error: {str(e)}"
            )
    
    async def handle_rag_execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle RAG execution request."""
        technique = params.get('technique')
        query = params.get('query')
        options = params.get('options', {})
        technique_params = params.get('technique_params', {})
        
        if not technique or not query:
            raise ValueError("Missing required parameters: technique and query")
        
        # Get pipeline instance
        pipeline = self.pipeline_registry.get_pipeline(technique)
        if not pipeline:
            raise ValueError(f"Pipeline not found: {technique}")
        
        # Execute pipeline
        start_time = datetime.now()
        result = await pipeline.execute(query, options, technique_params)
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Add performance metadata
        result['performance']['total_time_ms'] = execution_time
        result['metadata']['worker_id'] = self.worker_id
        result['metadata']['timestamp'] = datetime.now().isoformat()
        
        return result
    
    async def handle_health_check(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle health check request."""
        return {
            'status': 'healthy',
            'worker_id': self.worker_id,
            'uptime_seconds': (datetime.now() - self.start_time).total_seconds(),
            'request_count': self.request_count,
            'error_count': self.error_count,
            'error_rate': self.error_count / max(self.request_count, 1),
            'active_requests': len(self.active_requests),
            'memory_usage': self._get_memory_usage(),
            'pipelines_loaded': len(self.pipeline_registry.list_pipeline_names())
        }
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                'rss_mb': memory_info.rss / 1024 / 1024,
                'vms_mb': memory_info.vms / 1024 / 1024,
                'percent': process.memory_percent()
            }
        except ImportError:
            return {'error': 'psutil not available'}

# Entry point for worker process
if __name__ == '__main__':
    import sys
    
    worker_id = sys.argv[1] if len(sys.argv) > 1 else 'default'
    worker = BridgeWorker(worker_id)
    
    asyncio.run(worker.start())
```

## 7. Error Handling and Recovery

### 7.1 Error Classification

```typescript
enum BridgeErrorCode {
  // Communication errors
  WORKER_UNAVAILABLE = 1001,
  COMMUNICATION_TIMEOUT = 1002,
  SERIALIZATION_ERROR = 1003,
  
  // Process errors
  WORKER_CRASHED = 2001,
  WORKER_OVERLOADED = 2002,
  RESOURCE_EXHAUSTED = 2003,
  
  // Pipeline errors
  PIPELINE_NOT_FOUND = 3001,
  PIPELINE_INITIALIZATION_FAILED = 3002,
  PIPELINE_EXECUTION_ERROR = 3003,
  
  // Configuration errors
  INVALID_CONFIGURATION = 4001,
  MISSING_DEPENDENCIES = 4002,
  
  // System errors
  DATABASE_CONNECTION_ERROR = 5001,
  INSUFFICIENT_RESOURCES = 5002
}

class BridgeErrorHandler {
  static handleError(error: Error, context: ErrorContext): BridgeError {
    const errorCode = this.classifyError(error);
    const recovery = this.getRecoveryStrategy(errorCode);
    
    return new BridgeError(errorCode, error.message, context, recovery);
  }
  
  static async executeRecovery(bridgeError: BridgeError): Promise<boolean> {
    switch (bridgeError.recovery) {
      case RecoveryStrategy.RESTART_WORKER:
        return await this.restartWorker(bridgeError.context.workerId);
      
      case RecoveryStrategy.RETRY_REQUEST:
        return await this.retryRequest(bridgeError.context.request);
      
      case RecoveryStrategy.FALLBACK_TECHNIQUE:
        return await this.fallbackTechnique(bridgeError.context);
      
      default:
        return false;
    }
  }
}
```

## 8. Performance Optimization

### 8.1 Caching Strategy

```typescript
class BridgeCache {
  private embeddingCache: LRUCache<string, number[]>;
  private responseCache: LRUCache<string, any>;
  private pipelineCache: Map<string, any>;
  
  constructor(config: CacheConfig) {
    this.embeddingCache = new LRUCache({
      max: config.embeddingCacheSize,
      ttl: config.embeddingCacheTTL
    });
    
    this.responseCache = new LRUCache({
      max: config.responseCacheSize,
      ttl: config.responseCacheTTL
    });
    
    this.pipelineCache = new Map();
  }
  
  getCachedEmbedding(text: string): number[] | null {
    return this.embeddingCache.get(this.hashText(text));
  }
  
  setCachedEmbedding(text: string, embedding: number[]): void {
    this.embeddingCache.set(this.hashText(text), embedding);
  }
  
  getCachedResponse(query: string, technique: string, params: any): any | null {
    const key = this.createResponseKey(query, technique, params);
    return this.responseCache.get(key);
  }
  
  setCachedResponse(query: string, technique: string, params: any, response: any): void {
    const key = this.createResponseKey(query, technique, params);
    this.responseCache.set(key, response);
  }
}
```

### 8.2 Connection Pooling

```typescript
class ConnectionPool {
  private connections: Map<string, WorkerConnection>;
  private config: PoolConfig;
  
  async getConnection(workerId: string): Promise<WorkerConnection> {
    let connection = this.connections.get(workerId);
    
    if (!connection || !connection.isAlive()) {
      connection = await this.createConnection(workerId);
      this.connections.set(workerId, connection);
    }
    
    return connection;
  }
  
  private async createConnection(workerId: string): Promise<WorkerConnection> {
    return new WorkerConnection({
      workerId,
      keepAlive: true,
      timeout: this.config.connectionTimeout,
      maxRetries: this.config.maxRetries
    });
  }
}
```

This Python-Node.js bridge architecture provides a robust, scalable foundation for seamless communication between the MCP server layer and the Python RAG core, with comprehensive error handling, performance optimization, and enterprise-scale process management.