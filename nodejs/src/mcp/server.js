/**
 * MCP Server for IRIS RAG System
 * 
 * Main entry point for the Model Context Protocol server that provides
 * access to all 8 RAG techniques through a standardized interface.
 * 
 * This server integrates:
 * - Node.js MCP infrastructure
 * - Python RAG pipeline implementations
 * - IRIS database backend
 * - Performance monitoring and health checks
 */

import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { CallToolRequestSchema, ListToolsRequestSchema } from '@modelcontextprotocol/sdk/types.js';
import { RAGToolsManager } from './rag_tools/index.js';

/**
 * IRIS RAG MCP Server
 * 
 * Provides Model Context Protocol interface for all RAG techniques,
 * enabling integration with AI assistants and other MCP clients.
 */
class IRISRAGMCPServer {
    constructor() {
        this.server = new Server(
            {
                name: 'iris-rag-server',
                version: '1.0.0',
                description: 'MCP Server for IRIS RAG System with 8 RAG techniques'
            },
            {
                capabilities: {
                    tools: {}
                }
            }
        );
        
        this.ragToolsManager = new RAGToolsManager();
        this.logger = console; // Use console for now, can be replaced with proper logger
        
        this._setupHandlers();
    }
    
    /**
     * Setup MCP server handlers.
     */
    _setupHandlers() {
        // List tools handler
        this.server.setRequestHandler(ListToolsRequestSchema, async () => {
            try {
                const tools = this.ragToolsManager.getTools();
                this.logger.log(`Listing ${tools.length} available tools`);
                return { tools };
            } catch (error) {
                this.logger.error('Error listing tools:', error);
                throw error;
            }
        });
        
        // Call tool handler
        this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
            try {
                const { name, arguments: args } = request.params;
                
                this.logger.log(`Executing tool: ${name}`);
                this.logger.log(`Arguments:`, JSON.stringify(args, null, 2));
                
                // Execute tool via RAG tools manager
                const result = await this.ragToolsManager.executeTool(name, args || {});
                
                this.logger.log(`Tool ${name} executed successfully`);
                
                // Format response according to MCP specification
                return {
                    content: [
                        {
                            type: 'text',
                            text: this._formatToolResult(name, result)
                        }
                    ]
                };
                
            } catch (error) {
                this.logger.error(`Error executing tool ${request.params.name}:`, error);
                
                // Return error response
                return {
                    content: [
                        {
                            type: 'text',
                            text: `Error executing ${request.params.name}: ${error.message}`
                        }
                    ],
                    isError: true
                };
            }
        });
        
        // Error handler
        this.server.onerror = (error) => {
            this.logger.error('MCP Server error:', error);
        };
        
        this.logger.log('MCP server handlers configured');
    }
    
    /**
     * Format tool result for MCP response.
     */
    _formatToolResult(toolName, result) {
        try {
            // Handle different result types
            if (typeof result === 'string') {
                return result;
            }
            
            if (typeof result === 'object' && result !== null) {
                // Format RAG technique results
                if (result.technique && result.query && result.answer) {
                    return this._formatRAGResult(result);
                }
                
                // Format utility tool results
                if (toolName.includes('health_check')) {
                    return this._formatHealthCheckResult(result);
                } else if (toolName.includes('list_techniques')) {
                    return this._formatListTechniquesResult(result);
                } else if (toolName.includes('performance_metrics')) {
                    return this._formatMetricsResult(result);
                }
                
                // Default JSON formatting
                return JSON.stringify(result, null, 2);
            }
            
            return String(result);
            
        } catch (error) {
            this.logger.error('Error formatting tool result:', error);
            return `Result formatting error: ${error.message}`;
        }
    }
    
    /**
     * Format RAG technique result.
     */
    _formatRAGResult(result) {
        const sections = [];
        
        // Header
        sections.push(`# ${result.technique.toUpperCase()} RAG Result`);
        sections.push('');
        
        // Query
        sections.push(`**Query:** ${result.query}`);
        sections.push('');
        
        // Answer
        sections.push(`**Answer:**`);
        sections.push(result.answer);
        sections.push('');
        
        // Retrieved documents
        if (result.retrieved_documents && result.retrieved_documents.length > 0) {
            sections.push(`**Retrieved Documents (${result.retrieved_documents.length}):**`);
            result.retrieved_documents.forEach((doc, index) => {
                sections.push(`${index + 1}. **Score:** ${doc.score || 'N/A'}`);
                if (doc.source) {
                    sections.push(`   **Source:** ${doc.source}`);
                }
                if (doc.content) {
                    const preview = doc.content.length > 200 
                        ? doc.content.substring(0, 200) + '...'
                        : doc.content;
                    sections.push(`   **Content:** ${preview}`);
                }
                sections.push('');
            });
        }
        
        // Performance metrics
        if (result.performance) {
            sections.push(`**Performance:**`);
            sections.push(`- Execution Time: ${result.performance.execution_time_ms?.toFixed(2) || 'N/A'}ms`);
            sections.push(`- Timestamp: ${result.performance.timestamp || 'N/A'}`);
            sections.push('');
        }
        
        // Metadata
        if (result.metadata && Object.keys(result.metadata).length > 0) {
            sections.push(`**Metadata:**`);
            sections.push('```json');
            sections.push(JSON.stringify(result.metadata, null, 2));
            sections.push('```');
        }
        
        return sections.join('\n');
    }
    
    /**
     * Format health check result.
     */
    _formatHealthCheckResult(result) {
        const sections = [];
        
        sections.push('# RAG System Health Check');
        sections.push('');
        
        if (result.status) {
            sections.push(`**Overall Status:** ${result.status}`);
            sections.push('');
        }
        
        if (result.components) {
            sections.push('**Component Status:**');
            for (const [component, status] of Object.entries(result.components)) {
                sections.push(`- ${component}: ${status}`);
            }
            sections.push('');
        }
        
        if (result.details) {
            sections.push('**Details:**');
            sections.push('```json');
            sections.push(JSON.stringify(result.details, null, 2));
            sections.push('```');
        }
        
        return sections.join('\n');
    }
    
    /**
     * Format list techniques result.
     */
    _formatListTechniquesResult(result) {
        const sections = [];
        
        sections.push('# Available RAG Techniques');
        sections.push('');
        
        if (result.techniques) {
            result.techniques.forEach((technique, index) => {
                sections.push(`${index + 1}. **${technique.name}**`);
                if (technique.description) {
                    sections.push(`   ${technique.description}`);
                }
                if (technique.enabled !== undefined) {
                    sections.push(`   Status: ${technique.enabled ? 'Enabled' : 'Disabled'}`);
                }
                sections.push('');
            });
        }
        
        if (result.total_count) {
            sections.push(`**Total Techniques:** ${result.total_count}`);
        }
        
        return sections.join('\n');
    }
    
    /**
     * Format performance metrics result.
     */
    _formatMetricsResult(result) {
        const sections = [];
        
        sections.push('# Performance Metrics');
        sections.push('');
        
        if (result.technique) {
            sections.push(`**Technique:** ${result.technique}`);
            sections.push('');
        }
        
        if (result.time_range) {
            sections.push(`**Time Range:** ${result.time_range}`);
            sections.push('');
        }
        
        if (result.metrics) {
            sections.push('**Metrics:**');
            for (const [metric, value] of Object.entries(result.metrics)) {
                sections.push(`- ${metric}: ${value}`);
            }
            sections.push('');
        }
        
        if (result.details) {
            sections.push('**Details:**');
            sections.push('```json');
            sections.push(JSON.stringify(result.details, null, 2));
            sections.push('```');
        }
        
        return sections.join('\n');
    }
    
    /**
     * Start the MCP server.
     */
    async start() {
        try {
            // Check if RAG tools manager is initialized
            if (!this.ragToolsManager.isInitialized()) {
                throw new Error('RAG Tools Manager failed to initialize');
            }
            
            this.logger.log('Starting IRIS RAG MCP Server...');
            
            // Create transport
            const transport = new StdioServerTransport();
            
            // Connect server to transport
            await this.server.connect(transport);
            
            this.logger.log('IRIS RAG MCP Server started successfully');
            this.logger.log(`Available tools: ${this.ragToolsManager.getTools().length}`);
            
            // Log server status
            const status = this.ragToolsManager.getStatus();
            this.logger.log('Server status:', JSON.stringify(status, null, 2));
            
        } catch (error) {
            this.logger.error('Failed to start MCP server:', error);
            throw error;
        }
    }
    
    /**
     * Stop the MCP server.
     */
    async stop() {
        try {
            this.logger.log('Stopping IRIS RAG MCP Server...');
            
            // Close server connection
            await this.server.close();
            
            this.logger.log('IRIS RAG MCP Server stopped');
            
        } catch (error) {
            this.logger.error('Error stopping MCP server:', error);
            throw error;
        }
    }
    
    /**
     * Get server status.
     */
    getStatus() {
        return {
            server_name: 'iris-rag-server',
            version: '1.0.0',
            rag_tools_manager: this.ragToolsManager.getStatus(),
            uptime: process.uptime()
        };
    }
}

// Main execution
async function main() {
    const server = new IRISRAGMCPServer();
    
    // Handle process signals
    process.on('SIGINT', async () => {
        console.log('Received SIGINT, shutting down gracefully...');
        await server.stop();
        process.exit(0);
    });
    
    process.on('SIGTERM', async () => {
        console.log('Received SIGTERM, shutting down gracefully...');
        await server.stop();
        process.exit(0);
    });
    
    // Start server
    try {
        await server.start();
    } catch (error) {
        console.error('Failed to start server:', error);
        process.exit(1);
    }
}

// Export for testing
export { IRISRAGMCPServer };

// Run if this is the main module
if (import.meta.url === `file://${process.argv[1]}`) {
    main().catch((error) => {
        console.error('Unhandled error:', error);
        process.exit(1);
    });
}