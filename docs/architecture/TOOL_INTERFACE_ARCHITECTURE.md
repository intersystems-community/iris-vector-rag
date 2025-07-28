# Tool Interface Architecture

## 1. Overview

This document defines the modular tool interface architecture for the IRIS RAG MCP server, providing standardized, extensible interfaces for all 8 RAG techniques with consistent parameter validation, error handling, and performance monitoring.

## 2. Tool Interface Design Principles

### 2.1 Modular Design Patterns

- **Interface Segregation**: Each tool implements a focused, single-responsibility interface
- **Dependency Injection**: Tools receive dependencies through constructor injection
- **Strategy Pattern**: Technique-specific logic encapsulated in strategy implementations
- **Template Method**: Common workflow with technique-specific customization points
- **Factory Pattern**: Tool creation and configuration management

### 2.2 Standardization Requirements

- **Consistent Input/Output Schemas**: All tools follow the same base schema structure
- **Uniform Error Handling**: Standardized error codes and response formats
- **Performance Monitoring**: Built-in metrics collection for all operations
- **Parameter Validation**: JSON schema validation with technique-specific extensions

## 3. Base Tool Interface

### 3.1 Core Tool Interface

```typescript
interface IRAGTool {
  // Tool metadata
  readonly name: string;
  readonly description: string;
  readonly version: string;
  readonly inputSchema: JSONSchema;
  readonly outputSchema: JSONSchema;
  
  // Core operations
  execute(params: ToolParameters): Promise<ToolResponse>;
  validate(params: ToolParameters): ValidationResult;
  getSchema(): ToolSchema;
  
  // Lifecycle management
  initialize(config: ToolConfig): Promise<void>;
  cleanup(): Promise<void>;
  
  // Health and monitoring
  healthCheck(): Promise<HealthStatus>;
  getMetrics(): PerformanceMetrics;
}
```

### 3.2 Standard Parameter Structure

```typescript
interface ToolParameters {
  // Required parameters
  query: string;
  
  // Common optional parameters
  options?: {
    top_k?: number;
    temperature?: number;
    max_tokens?: number;
    include_sources?: boolean;
    min_similarity?: number;
  };
  
  // Technique-specific parameters
  technique_params?: Record<string, any>;
  
  // Metadata
  request_id?: string;
  user_context?: Record<string, any>;
}
```

### 3.3 Standard Response Structure

```typescript
interface ToolResponse {
  // Status
  success: boolean;
  technique: string;
  request_id?: string;
  
  // Core response data
  query: string;
  answer: string;
  retrieved_documents: RetrievedDocument[];
  
  // Performance metrics
  performance: {
    total_time_ms: number;
    retrieval_time_ms: number;
    generation_time_ms: number;
    documents_searched: number;
  };
  
  // Metadata
  metadata: {
    timestamp: string;
    model_info: {
      embedding_model: string;
      llm_model: string;
    };
    technique_specific?: Record<string, any>;
  };
  
  // Error information (if applicable)
  error?: {
    code: string;
    message: string;
    details?: Record<string, any>;
  };
}
```

## 4. Tool Implementation Architecture

### 4.1 Abstract Base Tool

```typescript
abstract class BaseRAGTool implements IRAGTool {
  protected config: ToolConfig;
  protected performanceMonitor: PerformanceMonitor;
  protected validator: ParameterValidator;
  protected logger: Logger;
  
  constructor(
    config: ToolConfig,
    performanceMonitor: PerformanceMonitor,
    validator: ParameterValidator,
    logger: Logger
  ) {
    this.config = config;
    this.performanceMonitor = performanceMonitor;
    this.validator = validator;
    this.logger = logger;
  }
  
  // Template method pattern
  async execute(params: ToolParameters): Promise<ToolResponse> {
    const startTime = Date.now();
    const requestId = params.request_id || this.generateRequestId();
    
    try {
      // 1. Validate parameters
      const validationResult = this.validate(params);
      if (!validationResult.isValid) {
        throw new ValidationError(validationResult.errors);
      }
      
      // 2. Pre-processing hook
      const processedParams = await this.preProcess(params);
      
      // 3. Execute technique-specific logic
      const result = await this.executeInternal(processedParams);
      
      // 4. Post-processing hook
      const finalResult = await this.postProcess(result, processedParams);
      
      // 5. Record metrics
      this.recordMetrics(requestId, Date.now() - startTime, true);
      
      return this.formatResponse(finalResult, requestId, Date.now() - startTime);
      
    } catch (error) {
      this.recordMetrics(requestId, Date.now() - startTime, false, error);
      return this.formatErrorResponse(error, requestId);
    }
  }
  
  // Abstract methods for technique-specific implementation
  protected abstract executeInternal(params: ToolParameters): Promise<any>;
  protected abstract getTechniqueSpecificSchema(): JSONSchema;
  
  // Hook methods with default implementations
  protected async preProcess(params: ToolParameters): Promise<ToolParameters> {
    return params;
  }
  
  protected async postProcess(result: any, params: ToolParameters): Promise<any> {
    return result;
  }
  
  // Common utility methods
  protected generateRequestId(): string {
    return `${this.name}_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }
  
  protected recordMetrics(requestId: string, duration: number, success: boolean, error?: Error): void {
    this.performanceMonitor.recordToolExecution({
      toolName: this.name,
      requestId,
      duration,
      success,
      error: error?.message
    });
  }
}
```

### 4.2 Tool Factory Pattern

```typescript
interface ToolFactory {
  createTool(technique: string, config: ToolConfig): IRAGTool;
  getSupportedTechniques(): string[];
  getToolSchema(technique: string): ToolSchema;
}

class RAGToolFactory implements ToolFactory {
  private toolRegistry: Map<string, ToolConstructor>;
  private configManager: ConfigurationManager;
  private performanceMonitor: PerformanceMonitor;
  
  constructor(
    configManager: ConfigurationManager,
    performanceMonitor: PerformanceMonitor
  ) {
    this.configManager = configManager;
    this.performanceMonitor = performanceMonitor;
    this.toolRegistry = new Map();
    this.registerTools();
  }
  
  private registerTools(): void {
    this.toolRegistry.set('basic', BasicRAGTool);
    this.toolRegistry.set('crag', CRAGTool);
    this.toolRegistry.set('hyde', HyDETool);
    this.toolRegistry.set('graphrag', GraphRAGTool);
    this.toolRegistry.set('hybrid_ifind', HybridIFindTool);
    this.toolRegistry.set('colbert', ColBERTTool);
    this.toolRegistry.set('noderag', NodeRAGTool);
    this.toolRegistry.set('sqlrag', SQLRAGTool);
  }
  
  createTool(technique: string, config: ToolConfig): IRAGTool {
    const ToolClass = this.toolRegistry.get(technique);
    if (!ToolClass) {
      throw new Error(`Unsupported technique: ${technique}`);
    }
    
    const validator = new ParameterValidator(this.getToolSchema(technique));
    const logger = new Logger(`tool:${technique}`);
    
    return new ToolClass(config, this.performanceMonitor, validator, logger);
  }
}
```

## 5. Individual Tool Specifications

### 5.1 BasicRAG Tool

```typescript
class BasicRAGTool extends BaseRAGTool {
  readonly name = 'rag_basic';
  readonly description = 'Standard retrieval-augmented generation with vector similarity search';
  readonly version = '1.0.0';
  
  protected getTechniqueSpecificSchema(): JSONSchema {
    return {
      type: 'object',
      properties: {},
      additionalProperties: false
    };
  }
  
  protected async executeInternal(params: ToolParameters): Promise<any> {
    const { query, options = {} } = params;
    
    // 1. Initialize pipeline
    const pipeline = await this.getPipeline('basic');
    
    // 2. Execute query with performance tracking
    const retrievalStart = Date.now();
    const retrievedDocs = await pipeline.retrieve(query, options.top_k || 5);
    const retrievalTime = Date.now() - retrievalStart;
    
    const generationStart = Date.now();
    const answer = await pipeline.generate(query, retrievedDocs, options);
    const generationTime = Date.now() - generationStart;
    
    return {
      query,
      answer,
      retrieved_documents: retrievedDocs,
      performance: {
        retrieval_time_ms: retrievalTime,
        generation_time_ms: generationTime,
        documents_searched: retrievedDocs.length
      }
    };
  }
}
```

### 5.2 CRAG Tool

```typescript
class CRAGTool extends BaseRAGTool {
  readonly name = 'rag_crag';
  readonly description = 'Corrective RAG with retrieval quality evaluation and correction';
  readonly version = '1.0.0';
  
  protected getTechniqueSpecificSchema(): JSONSchema {
    return {
      type: 'object',
      properties: {
        confidence_threshold: {
          type: 'number',
          minimum: 0.0,
          maximum: 1.0,
          default: 0.8
        },
        enable_web_search: {
          type: 'boolean',
          default: false
        },
        correction_strategy: {
          type: 'string',
          enum: ['rewrite', 'expand', 'filter'],
          default: 'rewrite'
        }
      },
      additionalProperties: false
    };
  }
  
  protected async executeInternal(params: ToolParameters): Promise<any> {
    const { query, options = {}, technique_params = {} } = params;
    const confidenceThreshold = technique_params.confidence_threshold || 0.8;
    
    // 1. Initial retrieval
    const pipeline = await this.getPipeline('crag');
    const initialDocs = await pipeline.retrieve(query, options.top_k || 5);
    
    // 2. Evaluate retrieval quality
    const confidence = await pipeline.evaluateRetrievalQuality(query, initialDocs);
    
    let finalDocs = initialDocs;
    let correctionApplied = false;
    
    // 3. Apply correction if needed
    if (confidence < confidenceThreshold) {
      finalDocs = await this.applyCorrectionStrategy(
        query, 
        initialDocs, 
        technique_params.correction_strategy || 'rewrite'
      );
      correctionApplied = true;
    }
    
    // 4. Generate answer
    const answer = await pipeline.generate(query, finalDocs, options);
    
    return {
      query,
      answer,
      retrieved_documents: finalDocs,
      metadata: {
        technique_specific: {
          initial_confidence: confidence,
          correction_applied: correctionApplied,
          correction_strategy: technique_params.correction_strategy
        }
      }
    };
  }
}
```

### 5.3 ColBERT Tool

```typescript
class ColBERTTool extends BaseRAGTool {
  readonly name = 'rag_colbert';
  readonly description = 'Late interaction retrieval with token-level matching';
  readonly version = '1.0.0';
  
  protected getTechniqueSpecificSchema(): JSONSchema {
    return {
      type: 'object',
      properties: {
        max_query_length: {
          type: 'integer',
          minimum: 32,
          maximum: 512,
          default: 256
        },
        interaction_threshold: {
          type: 'number',
          minimum: 0.0,
          maximum: 1.0,
          default: 0.5
        },
        enable_query_expansion: {
          type: 'boolean',
          default: false
        },
        compression_ratio: {
          type: 'number',
          minimum: 0.1,
          maximum: 1.0,
          default: 0.8
        }
      },
      additionalProperties: false
    };
  }
  
  protected async executeInternal(params: ToolParameters): Promise<any> {
    const { query, options = {}, technique_params = {} } = params;
    
    // 1. Query preprocessing and tokenization
    const processedQuery = await this.preprocessQuery(
      query, 
      technique_params.max_query_length || 256
    );
    
    // 2. ColBERT-specific retrieval with late interaction
    const pipeline = await this.getPipeline('colbert');
    const retrievedDocs = await pipeline.lateInteractionRetrieve(
      processedQuery,
      {
        top_k: options.top_k || 5,
        interaction_threshold: technique_params.interaction_threshold || 0.5,
        compression_ratio: technique_params.compression_ratio || 0.8
      }
    );
    
    // 3. Generate answer with token-level context
    const answer = await pipeline.generateWithTokenContext(
      processedQuery,
      retrievedDocs,
      options
    );
    
    return {
      query,
      answer,
      retrieved_documents: retrievedDocs,
      metadata: {
        technique_specific: {
          processed_query_tokens: processedQuery.tokens.length,
          interaction_scores: retrievedDocs.map(doc => doc.interaction_score),
          compression_applied: technique_params.compression_ratio < 1.0
        }
      }
    };
  }
}
```

## 6. Tool Registry and Management

### 6.1 Tool Registry

```typescript
class ToolRegistry {
  private tools: Map<string, IRAGTool>;
  private factory: ToolFactory;
  private config: ConfigurationManager;
  
  constructor(factory: ToolFactory, config: ConfigurationManager) {
    this.tools = new Map();
    this.factory = factory;
    this.config = config;
  }
  
  async initializeTools(): Promise<void> {
    const enabledTechniques = this.config.get('enabled_techniques', [
      'basic', 'crag', 'hyde', 'graphrag', 'hybrid_ifind', 
      'colbert', 'noderag', 'sqlrag'
    ]);
    
    for (const technique of enabledTechniques) {
      try {
        const toolConfig = this.config.get(`tools.${technique}`, {});
        const tool = this.factory.createTool(technique, toolConfig);
        await tool.initialize(toolConfig);
        this.tools.set(technique, tool);
      } catch (error) {
        console.error(`Failed to initialize tool ${technique}:`, error);
      }
    }
  }
  
  getTool(technique: string): IRAGTool | undefined {
    return this.tools.get(technique);
  }
  
  getAvailableTools(): string[] {
    return Array.from(this.tools.keys());
  }
  
  async healthCheck(): Promise<Record<string, HealthStatus>> {
    const results: Record<string, HealthStatus> = {};
    
    for (const [technique, tool] of this.tools) {
      try {
        results[technique] = await tool.healthCheck();
      } catch (error) {
        results[technique] = {
          status: 'unhealthy',
          error: error.message
        };
      }
    }
    
    return results;
  }
}
```

### 6.2 Tool Router

```typescript
class ToolRouter {
  private registry: ToolRegistry;
  private loadBalancer: LoadBalancer;
  private rateLimiter: RateLimiter;
  
  constructor(
    registry: ToolRegistry,
    loadBalancer: LoadBalancer,
    rateLimiter: RateLimiter
  ) {
    this.registry = registry;
    this.loadBalancer = loadBalancer;
    this.rateLimiter = rateLimiter;
  }
  
  async routeRequest(toolName: string, params: ToolParameters): Promise<ToolResponse> {
    // 1. Rate limiting
    await this.rateLimiter.checkLimit(params.request_id || 'anonymous');
    
    // 2. Get tool instance
    const tool = this.registry.getTool(toolName);
    if (!tool) {
      throw new Error(`Tool not found: ${toolName}`);
    }
    
    // 3. Load balancing (for distributed deployments)
    const selectedInstance = await this.loadBalancer.selectInstance(toolName);
    
    // 4. Execute request
    return await tool.execute(params);
  }
  
  getToolSchemas(): Record<string, ToolSchema> {
    const schemas: Record<string, ToolSchema> = {};
    
    for (const toolName of this.registry.getAvailableTools()) {
      const tool = this.registry.getTool(toolName);
      if (tool) {
        schemas[toolName] = tool.getSchema();
      }
    }
    
    return schemas;
  }
}
```

## 7. Error Handling and Validation

### 7.1 Parameter Validation

```typescript
class ParameterValidator {
  private schema: JSONSchema;
  private ajv: Ajv;
  
  constructor(schema: JSONSchema) {
    this.schema = schema;
    this.ajv = new Ajv({ allErrors: true });
  }
  
  validate(params: ToolParameters): ValidationResult {
    const isValid = this.ajv.validate(this.schema, params);
    
    return {
      isValid: !!isValid,
      errors: this.ajv.errors || [],
      sanitizedParams: isValid ? this.sanitizeParams(params) : null
    };
  }
  
  private sanitizeParams(params: ToolParameters): ToolParameters {
    // Apply sanitization rules
    return {
      ...params,
      query: this.sanitizeQuery(params.query),
      options: this.sanitizeOptions(params.options),
      technique_params: this.sanitizeTechniqueParams(params.technique_params)
    };
  }
}
```

### 7.2 Error Response Formatting

```typescript
class ErrorHandler {
  static formatError(error: Error, requestId?: string): ToolResponse {
    const errorCode = this.getErrorCode(error);
    const errorMessage = this.sanitizeErrorMessage(error.message);
    
    return {
      success: false,
      technique: 'unknown',
      request_id: requestId,
      query: '',
      answer: '',
      retrieved_documents: [],
      performance: {
        total_time_ms: 0,
        retrieval_time_ms: 0,
        generation_time_ms: 0,
        documents_searched: 0
      },
      metadata: {
        timestamp: new Date().toISOString(),
        model_info: {
          embedding_model: 'unknown',
          llm_model: 'unknown'
        }
      },
      error: {
        code: errorCode,
        message: errorMessage,
        details: this.getErrorDetails(error)
      }
    };
  }
  
  private static getErrorCode(error: Error): string {
    if (error instanceof ValidationError) return 'VALIDATION_ERROR';
    if (error instanceof TimeoutError) return 'TIMEOUT_ERROR';
    if (error instanceof DatabaseError) return 'DATABASE_ERROR';
    if (error instanceof ModelError) return 'MODEL_ERROR';
    return 'INTERNAL_ERROR';
  }
}
```

## 8. Performance Monitoring Integration

### 8.1 Tool Performance Metrics

```typescript
interface ToolMetrics {
  toolName: string;
  requestCount: number;
  averageResponseTime: number;
  successRate: number;
  errorRate: number;
  p95ResponseTime: number;
  p99ResponseTime: number;
  lastExecutionTime: string;
}

class ToolPerformanceMonitor {
  private metrics: Map<string, ToolMetrics>;
  private metricsCollector: MetricsCollector;
  
  recordToolExecution(data: {
    toolName: string;
    requestId: string;
    duration: number;
    success: boolean;
    error?: string;
  }): void {
    this.metricsCollector.record({
      metric: 'tool_execution_time',
      value: data.duration,
      tags: {
        tool: data.toolName,
        success: data.success.toString()
      }
    });
    
    this.updateToolMetrics(data);
  }
  
  getToolMetrics(toolName: string): ToolMetrics | undefined {
    return this.metrics.get(toolName);
  }
  
  getAllMetrics(): Record<string, ToolMetrics> {
    return Object.fromEntries(this.metrics);
  }
}
```

This tool interface architecture provides a robust, modular foundation for implementing all 8 RAG techniques with consistent interfaces, comprehensive error handling, and built-in performance monitoring.