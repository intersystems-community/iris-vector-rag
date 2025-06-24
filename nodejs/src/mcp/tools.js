/**
 * MCP Tools for RAG Templates Library Consumption Framework.
 *
 * This module provides RAG tool definitions with JSON schema validation
 * following support-tools-mcp patterns and conventions.
 */

const { spawn } = require('child_process');
const path = require('path');

/**
 * Call Python bridge function via child process.
 *
 * @param {string} functionName - Name of the Python function to call
 * @param {string} query - Query parameter to pass to the function
 * @returns {Promise<Object>} Result from Python function
 */
async function callPythonBridge(functionName, query) {
    return new Promise((resolve, reject) => {
        // Get the project root directory (assuming we're in nodejs/src/mcp/)
        const projectRoot = path.resolve(__dirname, '../../..');
        const pythonScript = path.join(projectRoot, 'objectscript', 'python_bridge.py');
        
        // Spawn Python process to call the bridge function
        const pythonProcess = spawn('python3', ['-c', `
import sys
import os
sys.path.insert(0, '${projectRoot}')
from objectscript.python_bridge import ${functionName}
result = ${functionName}('${query.replace(/'/g, "\\'")}')
print(result)
        `], {
            cwd: projectRoot,
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
            if (code !== 0) {
                reject(new Error(`Python process exited with code ${code}: ${stderr}`));
                return;
            }
            
            try {
                // Parse the JSON response from Python
                const result = JSON.parse(stdout.trim());
                resolve(result);
            } catch (error) {
                reject(new Error(`Failed to parse Python response: ${error.message}. Output: ${stdout}`));
            }
        });
        
        pythonProcess.on('error', (error) => {
            reject(new Error(`Failed to spawn Python process: ${error.message}`));
        });
    });
}

/**
 * Create RAG tools for MCP integration.
 * 
 * @param {Object} ragInstance - RAG instance (Simple or Standard API)
 * @param {Array<string>} enabledTools - List of enabled tool names
 * @returns {Object} Object containing tool functions with schemas
 */
function createRAGTools(ragInstance, enabledTools = []) {
    const tools = {};
    
    // Define all available tools
    const availableTools = {
        rag_search: {
            description: "Search for relevant documents using RAG",
            inputSchema: {
                type: "object",
                properties: {
                    query: {
                        type: "string",
                        description: "Search query text"
                    },
                    max_results: {
                        type: "integer",
                        description: "Maximum number of results to return",
                        minimum: 1,
                        maximum: 50,
                        default: 5
                    },
                    min_similarity: {
                        type: "number",
                        description: "Minimum similarity threshold (0.0-1.0)",
                        minimum: 0,
                        maximum: 1
                    },
                    include_sources: {
                        type: "boolean",
                        description: "Whether to include source information",
                        default: false
                    }
                },
                required: ["query"],
                additionalProperties: false
            },
            async handler(params) {
                const { query, max_results = 5, min_similarity, include_sources = false } = params;
                
                if (!query || typeof query !== 'string') {
                    throw new Error('Query parameter is required and must be a string');
                }
                
                try {
                    const options = {
                        top_k: max_results,
                        min_similarity,
                        include_sources
                    };
                    
                    const result = await ragInstance.query(query, options);
                    
                    return {
                        query,
                        answer: typeof result === 'string' ? result : result.answer,
                        sources: typeof result === 'object' ? result.sources : undefined,
                        metadata: typeof result === 'object' ? result.metadata : undefined
                    };
                } catch (error) {
                    throw new Error(`RAG search failed: ${error.message}`);
                }
            }
        },
        
        rag_add_documents: {
            description: "Add documents to the RAG knowledge base",
            inputSchema: {
                type: "object",
                properties: {
                    documents: {
                        type: "array",
                        description: "Array of documents to add",
                        items: {
                            oneOf: [
                                {
                                    type: "string",
                                    description: "Document text content"
                                },
                                {
                                    type: "object",
                                    properties: {
                                        content: {
                                            type: "string",
                                            description: "Document text content"
                                        },
                                        title: {
                                            type: "string",
                                            description: "Document title"
                                        },
                                        metadata: {
                                            type: "object",
                                            description: "Additional document metadata"
                                        }
                                    },
                                    required: ["content"],
                                    additionalProperties: true
                                }
                            ]
                        },
                        minItems: 1,
                        maxItems: 100
                    },
                    chunk_documents: {
                        type: "boolean",
                        description: "Whether to chunk documents automatically",
                        default: true
                    },
                    generate_embeddings: {
                        type: "boolean",
                        description: "Whether to generate embeddings",
                        default: true
                    }
                },
                required: ["documents"],
                additionalProperties: false
            },
            async handler(params) {
                const { documents, chunk_documents = true, generate_embeddings = true } = params;
                
                if (!Array.isArray(documents) || documents.length === 0) {
                    throw new Error('Documents parameter must be a non-empty array');
                }
                
                try {
                    const options = {
                        chunk_documents,
                        generate_embeddings
                    };
                    
                    await ragInstance.addDocuments(documents, options);
                    
                    return {
                        success: true,
                        message: `Successfully added ${documents.length} documents to knowledge base`,
                        documents_added: documents.length
                    };
                } catch (error) {
                    throw new Error(`Failed to add documents: ${error.message}`);
                }
            }
        },
        
        rag_get_stats: {
            description: "Get statistics about the RAG knowledge base",
            inputSchema: {
                type: "object",
                properties: {},
                additionalProperties: false
            },
            async handler(params) {
                try {
                    let documentCount = 0;
                    
                    // Try to get document count if method exists
                    if (ragInstance.getDocumentCount && typeof ragInstance.getDocumentCount === 'function') {
                        documentCount = await ragInstance.getDocumentCount();
                    }
                    
                    return {
                        success: true,
                        statistics: {
                            total_documents: documentCount,
                            rag_type: ragInstance.constructor.name,
                            initialized: ragInstance._initialized || false
                        }
                    };
                } catch (error) {
                    throw new Error(`Failed to get statistics: ${error.message}`);
                }
            }
        },
        
        rag_clear_knowledge_base: {
            description: "Clear all documents from the RAG knowledge base",
            inputSchema: {
                type: "object",
                properties: {
                    confirm: {
                        type: "boolean",
                        description: "Confirmation that you want to clear all data",
                        default: false
                    }
                },
                required: ["confirm"],
                additionalProperties: false
            },
            async handler(params) {
                const { confirm = false } = params;
                
                if (!confirm) {
                    throw new Error('Confirmation required to clear knowledge base. Set confirm=true');
                }
                
                try {
                    if (ragInstance.clearKnowledgeBase && typeof ragInstance.clearKnowledgeBase === 'function') {
                        await ragInstance.clearKnowledgeBase();
                        return {
                            success: true,
                            message: "Knowledge base cleared successfully"
                        };
                    } else {
                        throw new Error('Clear knowledge base operation not supported by this RAG instance');
                    }
                } catch (error) {
                    throw new Error(`Failed to clear knowledge base: ${error.message}`);
                }
            }
        },
        
        rag_get_config: {
            description: "Get RAG configuration settings",
            inputSchema: {
                type: "object",
                properties: {
                    key: {
                        type: "string",
                        description: "Specific configuration key to retrieve (optional)"
                    }
                },
                additionalProperties: false
            },
            async handler(params) {
                const { key } = params;
                
                try {
                    if (ragInstance.getConfig && typeof ragInstance.getConfig === 'function') {
                        if (key) {
                            const value = ragInstance.getConfig(key);
                            return {
                                success: true,
                                key,
                                value
                            };
                        } else {
                            // Return basic configuration info
                            return {
                                success: true,
                                config: {
                                    rag_type: ragInstance.constructor.name,
                                    initialized: ragInstance._initialized || false,
                                    technique: ragInstance._technique || 'basic'
                                }
                            };
                        }
                    } else {
                        throw new Error('Get configuration operation not supported by this RAG instance');
                    }
                } catch (error) {
                    throw new Error(`Failed to get configuration: ${error.message}`);
                }
            }
        },
        
        iris_sql_search: {
            description: "Execute and rewrite SQL queries for InterSystems IRIS",
            inputSchema: {
                type: "object",
                properties: {
                    query: {
                        type: "string",
                        description: "SQL query to rewrite and execute"
                    }
                },
                required: ["query"],
                additionalProperties: false
            },
            async handler(params) {
                const { query } = params;
                
                if (!query || typeof query !== 'string') {
                    throw new Error('Query parameter is required and must be a string');
                }
                
                try {
                    // Call Python bridge function via child process
                    const result = await callPythonBridge('invoke_iris_sql_search', query);
                    return result;
                } catch (error) {
                    throw new Error(`IRIS SQL search failed: ${error.message}`);
                }
            }
        }
    };
    
    // Add enabled tools to the tools object
    for (const toolName of enabledTools) {
        if (toolName in availableTools) {
            const tool = availableTools[toolName];
            tools[toolName] = tool.handler;
            tools[toolName].description = tool.description;
            tools[toolName].inputSchema = tool.inputSchema;
        } else {
            console.warn(`Unknown tool requested: ${toolName}`);
        }
    }
    
    return tools;
}

/**
 * Get the schema for a specific tool.
 *
 * @param {string} toolName - Name of the tool
 * @returns {Object} Tool schema
 */
function getToolSchema(toolName) {
    const availableTools = {
        rag_search: {
            name: "rag_search",
            description: "Search for relevant documents using RAG",
            inputSchema: {
                type: "object",
                properties: {
                    query: {
                        type: "string",
                        description: "Search query text"
                    },
                    max_results: {
                        type: "integer",
                        description: "Maximum number of results to return",
                        minimum: 1,
                        maximum: 50,
                        default: 5
                    },
                    min_similarity: {
                        type: "number",
                        description: "Minimum similarity threshold (0.0-1.0)",
                        minimum: 0,
                        maximum: 1
                    },
                    include_sources: {
                        type: "boolean",
                        description: "Whether to include source information",
                        default: false
                    }
                },
                required: ["query"],
                additionalProperties: false
            }
        },
        iris_sql_search: {
            name: "iris_sql_search",
            description: "Execute and rewrite SQL queries for InterSystems IRIS",
            inputSchema: {
                type: "object",
                properties: {
                    query: {
                        type: "string",
                        description: "SQL query to rewrite and execute"
                    }
                },
                required: ["query"],
                additionalProperties: false
            }
        }
        // Add other tool schemas as needed
    };
    
    return availableTools[toolName] || null;
}

/**
 * List all available tool names.
 * 
 * @returns {Array<string>} Array of available tool names
 */
function listAvailableTools() {
    return [
        'rag_search',
        'rag_add_documents',
        'rag_get_stats',
        'rag_clear_knowledge_base',
        'rag_get_config'
    ];
}

module.exports = {
    createRAGTools,
    getToolSchema,
    listAvailableTools
};