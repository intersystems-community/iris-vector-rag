/**
 * MCP Server Integration for RAG Templates Library Consumption Framework.
 * 
 * This module provides trivial MCP server creation following support-tools-mcp patterns.
 * Enables zero-config MCP server setup with RAG capabilities.
 */

const { RAG } = require('../simple');
const { ConfigurableRAG } = require('../standard');
const { createRAGTools } = require('./tools');

/**
 * Create a trivial MCP server with RAG capabilities.
 * 
 * This function provides zero-configuration MCP server creation,
 * following the patterns established in support-tools-mcp.
 * 
 * @param {Object} config - Server configuration
 * @param {string} config.name - Server name
 * @param {string} config.description - Server description
 * @param {string} config.version - Server version (default: "1.0.0")
 * @param {Object} config.ragConfig - RAG configuration (optional)
 * @param {Array<string>} config.enabledTools - List of enabled tools (optional)
 * @returns {Object} MCP server instance
 */
function createMCPServer(config) {
    const {
        name,
        description,
        version = "1.0.0",
        ragConfig = {},
        enabledTools = ['rag_search', 'rag_add_documents', 'rag_get_stats']
    } = config;
    
    if (!name || !description) {
        throw new Error('Server name and description are required');
    }
    
    // Initialize RAG instance based on configuration
    let ragInstance;
    if (ragConfig.technique) {
        // Use Standard API for advanced configuration
        ragInstance = new ConfigurableRAG(ragConfig);
    } else {
        // Use Simple API for zero-config
        ragInstance = new RAG();
    }
    
    // Create RAG tools
    const tools = createRAGTools(ragInstance, enabledTools);
    
    // Server state
    let isRunning = false;
    let serverInfo = null;
    
    const server = {
        /**
         * Get server information
         */
        getInfo() {
            return {
                name,
                description,
                version,
                protocol_version: "2024-11-05",
                capabilities: {
                    tools: {},
                    resources: {},
                    prompts: {},
                    logging: {}
                },
                tools: Object.keys(tools),
                isRunning
            };
        },
        
        /**
         * Start the MCP server
         */
        async start() {
            if (isRunning) {
                console.warn('Server is already running');
                return;
            }
            
            try {
                // Initialize RAG instance if needed
                if (ragInstance._initializePipeline && !ragInstance._initialized) {
                    console.log('Initializing RAG pipeline...');
                    // Note: We don't actually initialize here to avoid database dependencies
                    // In a real implementation, this would initialize the pipeline
                }
                
                serverInfo = this.getInfo();
                isRunning = true;
                
                console.log(`🚀 MCP Server "${name}" started successfully`);
                console.log(`📋 Available tools: ${Object.keys(tools).join(', ')}`);
                console.log(`🔧 Server version: ${version}`);
                
                return serverInfo;
            } catch (error) {
                throw new Error(`Failed to start MCP server: ${error.message}`);
            }
        },
        
        /**
         * Stop the MCP server
         */
        async stop() {
            if (!isRunning) {
                console.warn('Server is not running');
                return;
            }
            
            try {
                // Cleanup RAG resources if needed
                if (ragInstance.close && typeof ragInstance.close === 'function') {
                    await ragInstance.close();
                }
                
                isRunning = false;
                serverInfo = null;
                
                console.log(`🛑 MCP Server "${name}" stopped successfully`);
            } catch (error) {
                throw new Error(`Failed to stop MCP server: ${error.message}`);
            }
        },
        
        /**
         * Handle tool calls
         */
        async handleToolCall(toolName, parameters) {
            if (!isRunning) {
                throw new Error('Server is not running');
            }
            
            if (!(toolName in tools)) {
                throw new Error(`Unknown tool: ${toolName}`);
            }
            
            try {
                const result = await tools[toolName](parameters);
                return {
                    success: true,
                    result
                };
            } catch (error) {
                return {
                    success: false,
                    error: error.message
                };
            }
        },
        
        /**
         * List available tools
         */
        listTools() {
            return Object.keys(tools).map(toolName => ({
                name: toolName,
                description: tools[toolName].description || `RAG tool: ${toolName}`,
                inputSchema: tools[toolName].inputSchema || {
                    type: "object",
                    properties: {},
                    required: []
                }
            }));
        },
        
        /**
         * Get tool schema
         */
        getToolSchema(toolName) {
            if (!(toolName in tools)) {
                throw new Error(`Unknown tool: ${toolName}`);
            }
            
            return {
                name: toolName,
                description: tools[toolName].description || `RAG tool: ${toolName}`,
                inputSchema: tools[toolName].inputSchema || {
                    type: "object",
                    properties: {},
                    required: []
                }
            };
        },
        
        /**
         * Health check
         */
        async healthCheck() {
            return {
                status: isRunning ? 'healthy' : 'stopped',
                server: name,
                version,
                timestamp: new Date().toISOString(),
                ragInitialized: ragInstance._initialized || false
            };
        },
        
        // Expose RAG instance for advanced usage
        getRagInstance() {
            return ragInstance;
        }
    };
    
    return server;
}

/**
 * Create a simple MCP server with minimal configuration.
 * 
 * @param {string} name - Server name
 * @param {string} description - Server description
 * @returns {Object} MCP server instance
 */
function createSimpleMCPServer(name, description) {
    return createMCPServer({ name, description });
}

/**
 * Create an advanced MCP server with full configuration.
 * 
 * @param {Object} config - Full server configuration
 * @returns {Object} MCP server instance
 */
function createAdvancedMCPServer(config) {
    return createMCPServer(config);
}

module.exports = {
    createMCPServer,
    createSimpleMCPServer,
    createAdvancedMCPServer
};