/**
 * RAG Tools Manager for MCP Server
 * 
 * This module provides the Node.js side of the MCP server implementation,
 * managing tool registration and execution for all 8 RAG techniques.
 * 
 * Integrates with:
 * - objectscript/mcp_bridge.py (Python bridge functions)
 * - iris_rag/mcp/server_manager.py (server management)
 * - iris_rag/mcp/technique_handlers.py (technique registry)
 */

import { spawn } from 'child_process';
import path from 'path';
import fs from 'fs/promises';
import { fileURLToPath } from 'url';

// ES6 equivalent of __dirname
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

/**
 * RAG Tools Manager
 * 
 * Manages the registration and execution of RAG technique tools
 * for the MCP server, providing a bridge between Node.js MCP
 * infrastructure and Python RAG implementations.
 */
class RAGToolsManager {
    constructor() {
        this.logger = console; // Use console for now, can be replaced with proper logger
        this.tools = new Map();
        this.pythonBridgePath = null;
        this.initialized = false;
        
        // Initialize paths
        this._initializePaths();
        
        // Register tools
        this._registerTools();
    }
    
    /**
     * Initialize file paths for Python bridge integration.
     */
    _initializePaths() {
        // Determine project root (assuming we're in nodejs/src/mcp/rag_tools/)
        const projectRoot = path.resolve(__dirname, '../../../../');
        this.pythonBridgePath = path.join(projectRoot, 'objectscript', 'mcp_bridge.py');
        
        this.logger.log(`Initialized paths - Python bridge: ${this.pythonBridgePath}`);
    }
    
    /**
     * Register all RAG technique tools.
     */
    _registerTools() {
        // Define tool configurations for all 8 RAG techniques with comprehensive descriptions
        const toolConfigs = [
            {
                name: 'rag_basic',
                description: 'Execute basic RAG with vector similarity search. This is the foundational RAG approach that retrieves relevant documents using vector similarity and generates answers using an LLM.',
                technique: 'basic'
            },
            {
                name: 'rag_crag',
                description: 'Execute Corrective RAG (CRAG) with retrieval quality evaluation. CRAG improves upon basic RAG by evaluating retrieval quality and applying correction strategies when retrieved documents are not relevant enough.',
                technique: 'crag'
            },
            {
                name: 'rag_hyde',
                description: 'Execute HyDE (Hypothetical Document Embeddings) RAG. HyDE generates a hypothetical answer to the query first, then uses that hypothetical answer\'s embedding to retrieve more relevant documents.',
                technique: 'hyde'
            },
            {
                name: 'rag_graphrag',
                description: 'Execute Graph RAG with entity relationship traversal. GraphRAG builds a knowledge graph from documents and uses graph traversal to find related entities and concepts for more comprehensive retrieval.',
                technique: 'graphrag'
            },
            {
                name: 'rag_hybrid_ifind',
                description: 'Execute Hybrid iFind RAG combining vector and keyword search. This approach combines semantic vector similarity with traditional keyword matching for more comprehensive retrieval.',
                technique: 'hybrid_ifind'
            },
            {
                name: 'rag_colbert',
                description: 'Execute ColBERT RAG with late interaction retrieval. ColBERT uses fine-grained token-level interactions between queries and documents for more precise relevance scoring.',
                technique: 'colbert'
            },
            {
                name: 'rag_noderag',
                description: 'Execute Node RAG with hierarchical document structure. NodeRAG organizes documents in a hierarchical tree structure and retrieves information by traversing document nodes.',
                technique: 'noderag'
            },
            {
                name: 'rag_sqlrag',
                description: 'Execute SQL RAG with database-driven retrieval. SQLRAG uses structured database queries to retrieve relevant information before generating answers.',
                technique: 'sqlrag'
            }
        ];
        
        // Register each tool
        for (const config of toolConfigs) {
            this.tools.set(config.name, {
                ...config,
                schema: this._generateToolSchema(config),
                handler: this._createToolHandler(config)
            });
        }
        
        // Add utility tools
        this._registerUtilityTools();
        
        this.logger.log(`Registered ${this.tools.size} RAG tools`);
        this.initialized = true;
    }
    
    /**
     * Register utility tools for server management.
     */
    _registerUtilityTools() {
        // Health check tool
        this.tools.set('rag_health_check', {
            name: 'rag_health_check',
            description: 'Check health status of RAG system components including database connectivity, embedding models, LLM availability, and vector store status.',
            technique: 'health',
            schema: {
                type: 'object',
                properties: {
                    include_details: {
                        type: 'boolean',
                        default: false,
                        description: 'Include detailed component status information in the health check response, such as version numbers, connection details, and performance metrics.'
                    }
                }
            },
            handler: this._createHealthCheckHandler()
        });
        
        // List techniques tool
        this.tools.set('rag_list_techniques', {
            name: 'rag_list_techniques',
            description: 'List all available RAG techniques with their descriptions and capabilities. Useful for discovering what RAG methods are available and their current status.',
            technique: 'list',
            schema: {
                type: 'object',
                properties: {},
                description: 'No parameters required. Returns a comprehensive list of all implemented RAG techniques.'
            },
            handler: this._createListTechniquesHandler()
        });
        
        // Performance metrics tool
        this.tools.set('rag_performance_metrics', {
            name: 'rag_performance_metrics',
            description: 'Get performance metrics for RAG techniques including execution times, retrieval quality scores, and system resource usage.',
            technique: 'metrics',
            schema: {
                type: 'object',
                properties: {
                    technique: {
                        type: 'string',
                        description: 'Specific RAG technique to get metrics for (e.g., "basic", "crag", "hyde"). Leave empty to get metrics for all techniques.'
                    },
                    time_range: {
                        type: 'string',
                        enum: ['1h', '24h', '7d', '30d'],
                        default: '24h',
                        description: 'Time range for performance metrics: "1h" (last hour), "24h" (last day), "7d" (last week), "30d" (last month)'
                    }
                }
            },
            handler: this._createMetricsHandler()
        });
    }
    
    /**
     * Generate tool schema for a RAG technique.
     */
    _generateToolSchema(config) {
        const baseSchema = {
            type: 'object',
            properties: {
                query: {
                    type: 'string',
                    description: 'User query text to search for and generate an answer about. This is the main input that drives the RAG pipeline.',
                    minLength: 1,
                    maxLength: 2048
                },
                options: {
                    type: 'object',
                    description: 'Configuration options for the RAG pipeline execution. These parameters control how the retrieval and generation process behaves.',
                    properties: {
                        top_k: {
                            type: 'integer',
                            minimum: 1,
                            maximum: 50,
                            default: 5,
                            description: 'Number of most relevant documents to retrieve from the vector store. Higher values provide more context but may introduce noise.'
                        },
                        temperature: {
                            type: 'number',
                            minimum: 0.0,
                            maximum: 2.0,
                            default: 0.7,
                            description: 'LLM generation temperature controlling randomness in answer generation. 0.0 = deterministic, 1.0 = balanced creativity, 2.0 = very creative/random.'
                        },
                        max_tokens: {
                            type: 'integer',
                            minimum: 50,
                            maximum: 4096,
                            default: 1024,
                            description: 'Maximum number of tokens in the generated response. Controls the length of the answer.'
                        },
                        include_sources: {
                            type: 'boolean',
                            default: true,
                            description: 'Whether to include source document references and metadata in the response. Useful for citation and verification.'
                        }
                    }
                }
            },
            required: ['query']
        };
        
        // Add technique-specific parameters
        if (config.technique === 'crag') {
            baseSchema.properties.technique_params = {
                type: 'object',
                properties: {
                    confidence_threshold: {
                        type: 'number',
                        minimum: 0.0,
                        maximum: 1.0,
                        default: 0.8,
                        description: 'Threshold for retrieval confidence'
                    },
                    correction_strategy: {
                        type: 'string',
                        enum: ['rewrite', 'expand', 'filter'],
                        default: 'rewrite',
                        description: 'Strategy for correcting poor retrievals'
                    }
                }
            };
        } else if (config.technique === 'colbert') {
            baseSchema.properties.technique_params = {
                type: 'object',
                properties: {
                    max_query_length: {
                        type: 'integer',
                        minimum: 32,
                        maximum: 512,
                        default: 256,
                        description: 'Maximum query length in tokens'
                    },
                    interaction_threshold: {
                        type: 'number',
                        minimum: 0.0,
                        maximum: 1.0,
                        default: 0.5,
                        description: 'Token interaction threshold'
                    }
                }
            };
        } else if (config.technique === 'graphrag') {
            baseSchema.properties.technique_params = {
                type: 'object',
                properties: {
                    max_hops: {
                        type: 'integer',
                        minimum: 1,
                        maximum: 5,
                        default: 2,
                        description: 'Maximum graph traversal hops'
                    },
                    entity_threshold: {
                        type: 'number',
                        minimum: 0.0,
                        maximum: 1.0,
                        default: 0.7,
                        description: 'Entity extraction confidence threshold'
                    }
                }
            };
        } else if (config.technique === 'hybrid_ifind') {
            baseSchema.properties.technique_params = {
                type: 'object',
                properties: {
                    vector_weight: {
                        type: 'number',
                        minimum: 0.0,
                        maximum: 1.0,
                        default: 0.7,
                        description: 'Weight for vector similarity scores'
                    },
                    keyword_weight: {
                        type: 'number',
                        minimum: 0.0,
                        maximum: 1.0,
                        default: 0.3,
                        description: 'Weight for keyword matching scores'
                    }
                }
            };
        }
        
        return baseSchema;
    }
    
    /**
     * Create tool handler for a RAG technique.
     */
    _createToolHandler(config) {
        return async (parameters) => {
            try {
                this.logger.log(`Executing ${config.name} with parameters:`, JSON.stringify(parameters, null, 2));
                
                // Validate parameters
                this._validateParameters(parameters, config.schema);
                
                // Execute via Python bridge
                const result = await this._executePythonBridge(config.technique, parameters);
                
                this.logger.log(`${config.name} execution completed successfully`);
                return result;
                
            } catch (error) {
                this.logger.error(`Error executing ${config.name}:`, error);
                throw error;
            }
        };
    }
    
    /**
     * Create health check handler.
     */
    _createHealthCheckHandler() {
        return async (parameters) => {
            try {
                const includeDetails = parameters.include_details || false;
                
                // Execute health check via Python bridge
                const result = await this._executePythonBridge('health_check', { include_details: includeDetails });
                
                return result;
                
            } catch (error) {
                this.logger.error('Error executing health check:', error);
                throw error;
            }
        };
    }
    
    /**
     * Create list techniques handler.
     */
    _createListTechniquesHandler() {
        return async (parameters) => {
            try {
                // Execute list techniques via Python bridge
                const result = await this._executePythonBridge('list_techniques', {});
                
                return result;
                
            } catch (error) {
                this.logger.error('Error listing techniques:', error);
                throw error;
            }
        };
    }
    
    /**
     * Create performance metrics handler.
     */
    _createMetricsHandler() {
        return async (parameters) => {
            try {
                const technique = parameters.technique || null;
                const timeRange = parameters.time_range || '24h';
                
                // Execute metrics via Python bridge
                const result = await this._executePythonBridge('performance_metrics', {
                    technique: technique,
                    time_range: timeRange
                });
                
                return result;
                
            } catch (error) {
                this.logger.error('Error getting performance metrics:', error);
                throw error;
            }
        };
    }
    
    /**
     * Execute Python bridge function.
     */
    async _executePythonBridge(technique, parameters) {
        return new Promise((resolve, reject) => {
            // Prepare command
            const command = 'uv';
            const args = ['run', 'python', this.pythonBridgePath, technique, JSON.stringify(parameters)];
            
            this.logger.log(`Executing Python bridge: ${command} ${args.join(' ')}`);
            
            // Spawn Python process
            const pythonProcess = spawn(command, args, {
                cwd: path.dirname(this.pythonBridgePath),
                stdio: ['pipe', 'pipe', 'pipe']
            });
            
            let stdout = '';
            let stderr = '';
            
            pythonProcess.stdout.on('data', (data) => {
                stdout += data.toString();
            });
            
            pythonProcess.stderr.on('data', (data) => {
                stderr += data.toString();
            });
            
            pythonProcess.on('close', (code) => {
                if (code === 0) {
                    try {
                        // Parse JSON response
                        const result = JSON.parse(stdout);
                        resolve(result);
                    } catch (parseError) {
                        reject(new Error(`Failed to parse Python bridge response: ${parseError.message}\nOutput: ${stdout}`));
                    }
                } else {
                    reject(new Error(`Python bridge execution failed with code ${code}\nStderr: ${stderr}\nStdout: ${stdout}`));
                }
            });
            
            pythonProcess.on('error', (error) => {
                reject(new Error(`Failed to spawn Python bridge process: ${error.message}`));
            });
        });
    }
    
    /**
     * Validate parameters against schema.
     */
    _validateParameters(parameters, schema) {
        // Basic validation - in production, use a proper JSON schema validator
        if (schema && schema.required) {
            for (const requiredField of schema.required) {
                if (!(requiredField in parameters)) {
                    throw new Error(`Missing required parameter: ${requiredField}`);
                }
            }
        }
        
        // Validate query if present
        if (parameters.query) {
            if (typeof parameters.query !== 'string') {
                throw new Error('Query must be a string');
            }
            if (parameters.query.length === 0) {
                throw new Error('Query cannot be empty');
            }
            if (parameters.query.length > 2048) {
                throw new Error('Query too long (max 2048 characters)');
            }
        }
        
        // Validate options if present
        if (parameters.options) {
            const options = parameters.options;
            
            if (options.top_k !== undefined) {
                if (!Number.isInteger(options.top_k) || options.top_k < 1 || options.top_k > 50) {
                    throw new Error('top_k must be an integer between 1 and 50');
                }
            }
            
            if (options.temperature !== undefined) {
                if (typeof options.temperature !== 'number' || options.temperature < 0 || options.temperature > 2) {
                    throw new Error('temperature must be a number between 0 and 2');
                }
            }
            
            if (options.max_tokens !== undefined) {
                if (!Number.isInteger(options.max_tokens) || options.max_tokens < 50 || options.max_tokens > 4096) {
                    throw new Error('max_tokens must be an integer between 50 and 4096');
                }
            }
        }
    }
    
    /**
     * Get all registered tools.
     */
    getTools() {
        const tools = [];
        for (const [name, tool] of this.tools) {
            tools.push({
                name: tool.name,
                description: tool.description,
                inputSchema: tool.schema
            });
        }
        return tools;
    }
    
    /**
     * Get tool by name.
     */
    getTool(name) {
        return this.tools.get(name);
    }
    
    /**
     * Execute tool by name.
     */
    async executeTool(name, parameters) {
        const tool = this.tools.get(name);
        if (!tool) {
            throw new Error(`Unknown tool: ${name}`);
        }
        
        return await tool.handler(parameters);
    }
    
    /**
     * Check if manager is initialized.
     */
    isInitialized() {
        return this.initialized;
    }
    
    /**
     * Get manager status.
     */
    getStatus() {
        return {
            initialized: this.initialized,
            total_tools: this.tools.size,
            python_bridge_path: this.pythonBridgePath,
            available_techniques: Array.from(this.tools.keys()).filter(name => name.startsWith('rag_') && !name.includes('_health_') && !name.includes('_list_') && !name.includes('_performance_'))
        };
    }
}

export { RAGToolsManager };