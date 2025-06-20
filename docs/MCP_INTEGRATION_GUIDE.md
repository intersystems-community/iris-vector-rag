# MCP Integration Guide

A comprehensive guide for creating Model Context Protocol (MCP) servers with rag-templates, enabling trivial integration with Claude Desktop and other MCP clients.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Zero-Config Server Creation](#zero-config-server-creation)
3. [Advanced MCP Server Customization](#advanced-mcp-server-customization)
4. [Tool Configuration and Routing](#tool-configuration-and-routing)
5. [Integration with Existing MCP Ecosystems](#integration-with-existing-mcp-ecosystems)
6. [Production Deployment](#production-deployment)
7. [Troubleshooting](#troubleshooting)

## Quick Start

### Installation

#### JavaScript/Node.js (Primary MCP Support)
```bash
npm install @rag-templates/core @rag-templates/mcp
```

#### Python (Experimental MCP Support)
```bash
pip install rag-templates[mcp]
```

### 30-Second MCP Server

#### JavaScript
```javascript
import { createMCPServer } from '@rag-templates/mcp';

// Dead simple MCP server
const server = createMCPServer({
    name: "my-rag-server",
    description: "RAG-powered knowledge assistant"
});

server.start();
console.log("🚀 MCP server running!");
```

#### Python
```python
from rag_templates.mcp import create_mcp_server

# Dead simple MCP server
server = create_mcp_server(
    name="my-rag-server",
    description="RAG-powered knowledge assistant"
)

server.start()
print("🚀 MCP server running!")
```

## Zero-Config Server Creation

The framework provides zero-configuration MCP server creation that works immediately with sensible defaults.

### Basic Server Creation

#### JavaScript
```javascript
import { createMCPServer } from '@rag-templates/mcp';

// Minimal configuration - uses all defaults
const server = createMCPServer({
    name: "knowledge-assistant",
    description: "Company knowledge base assistant"
});

// Server automatically includes these tools:
// - rag_search: Search the knowledge base
// - rag_add_documents: Add documents to knowledge base
// - rag_get_stats: Get knowledge base statistics

await server.start();
```

#### Python
```python
from rag_templates.mcp import create_mcp_server

# Minimal configuration - uses all defaults
server = create_mcp_server(
    name="knowledge-assistant",
    description="Company knowledge base assistant"
)

# Server automatically includes these tools:
# - rag_search: Search the knowledge base
# - rag_add_documents: Add documents to knowledge base
# - rag_get_stats: Get knowledge base statistics

server.start()
```

### Claude Desktop Integration

#### 1. Create MCP Server File

```javascript
// knowledge-server.js
import { createMCPServer } from '@rag-templates/mcp';

const server = createMCPServer({
    name: "company-knowledge",
    description: "Search and query company documentation",
    version: "1.0.0"
});

server.start();
```

#### 2. Configure Claude Desktop

Add to your Claude Desktop configuration file:

**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "company-knowledge": {
      "command": "node",
      "args": ["knowledge-server.js"],
      "env": {
        "IRIS_HOST": "localhost",
        "IRIS_PORT": "52773",
        "IRIS_USERNAME": "demo",
        "IRIS_PASSWORD": "demo"
      }
    }
  }
}
```

#### 3. Restart Claude Desktop

The server will automatically appear in Claude Desktop with available tools.

### Environment-Based Configuration

Following MCP best practices, use environment variables for configuration:

```javascript
// server.js
import { createMCPServer } from '@rag-templates/mcp';

const server = createMCPServer({
    name: process.env.MCP_SERVER_NAME || "rag-assistant",
    description: process.env.MCP_SERVER_DESCRIPTION || "RAG-powered assistant",
    ragConfig: {
        technique: process.env.RAG_TECHNIQUE || 'basic',
        database: {
            host: process.env.IRIS_HOST || 'localhost',
            port: parseInt(process.env.IRIS_PORT || '52773'),
            username: process.env.IRIS_USERNAME || 'demo',
            password: process.env.IRIS_PASSWORD || 'demo',
            namespace: process.env.IRIS_NAMESPACE || 'MCP_RAG'
        },
        llm: {
            provider: process.env.LLM_PROVIDER || 'openai',
            apiKey: process.env.OPENAI_API_KEY,
            model: process.env.LLM_MODEL || 'gpt-4o-mini'
        }
    }
});

server.start();
```

## Advanced MCP Server Customization

### Custom RAG Configuration

#### JavaScript
```javascript
import { createMCPServer, RAG } from '@rag-templates/mcp';

// Create custom RAG instance
const rag = new RAG({
    technique: 'colbert',
    llmProvider: 'anthropic',
    llmConfig: {
        model: 'claude-3-sonnet',
        temperature: 0.1
    },
    embeddingModel: 'text-embedding-3-large',
    database: {
        host: process.env.IRIS_HOST,
        port: parseInt(process.env.IRIS_PORT),
        namespace: 'ADVANCED_RAG'
    }
});

// Create server with custom RAG
const server = createMCPServer({
    name: "advanced-rag-server",
    description: "Advanced RAG with ColBERT technique",
    rag: rag,
    tools: [
        'rag_search',
        'rag_add_documents', 
        'rag_get_stats',
        'rag_clear_cache'
    ]
});

server.start();
```

#### Python
```python
from rag_templates.mcp import create_mcp_server
from rag_templates import ConfigurableRAG

# Create custom RAG instance
rag = ConfigurableRAG({
    'technique': 'colbert',
    'llm_provider': 'anthropic',
    'llm_config': {
        'model': 'claude-3-sonnet',
        'temperature': 0.1
    },
    'embedding_model': 'text-embedding-3-large',
    'database': {
        'host': os.getenv('IRIS_HOST'),
        'port': int(os.getenv('IRIS_PORT')),
        'namespace': 'ADVANCED_RAG'
    }
})

# Create server with custom RAG
server = create_mcp_server(
    name="advanced-rag-server",
    description="Advanced RAG with ColBERT technique",
    rag=rag,
    tools=[
        'rag_search',
        'rag_add_documents', 
        'rag_get_stats',
        'rag_clear_cache'
    ]
)

server.start()
```

### Custom Tool Definitions

#### JavaScript
```javascript
import { createMCPServer } from '@rag-templates/mcp';

const server = createMCPServer({
    name: "custom-rag-server",
    description: "RAG server with custom tools",
    tools: [
        {
            name: "search_knowledge",
            description: "Search the knowledge base with advanced options",
            inputSchema: {
                type: 'object',
                properties: {
                    query: {
                        type: 'string',
                        description: 'Search query'
                    },
                    maxResults: {
                        type: 'integer',
                        minimum: 1,
                        maximum: 50,
                        default: 5,
                        description: 'Maximum number of results'
                    },
                    minSimilarity: {
                        type: 'number',
                        minimum: 0,
                        maximum: 1,
                        default: 0.7,
                        description: 'Minimum similarity threshold'
                    },
                    includeMetadata: {
                        type: 'boolean',
                        default: false,
                        description: 'Include document metadata in results'
                    }
                },
                required: ['query'],
                additionalProperties: false
            },
            handler: async (args, rag) => {
                const result = await rag.query(args.query, {
                    maxResults: args.maxResults,
                    minSimilarity: args.minSimilarity,
                    includeSources: args.includeMetadata
                });
                
                return {
                    answer: result.answer,
                    confidence: result.confidence,
                    sources: args.includeMetadata ? result.sources : undefined,
                    processingTime: result.metadata?.processingTime
                };
            }
        },
        {
            name: "add_knowledge",
            description: "Add documents to the knowledge base",
            inputSchema: {
                type: 'object',
                properties: {
                    documents: {
                        type: 'array',
                        items: {
                            type: 'object',
                            properties: {
                                content: { type: 'string' },
                                title: { type: 'string' },
                                source: { type: 'string' },
                                metadata: { type: 'object' }
                            },
                            required: ['content']
                        },
                        description: 'Documents to add'
                    },
                    category: {
                        type: 'string',
                        description: 'Document category for organization'
                    }
                },
                required: ['documents'],
                additionalProperties: false
            },
            handler: async (args, rag) => {
                // Add category to each document
                const documentsWithCategory = args.documents.map(doc => ({
                    ...doc,
                    metadata: {
                        ...doc.metadata,
                        category: args.category
                    }
                }));
                
                await rag.addDocuments(documentsWithCategory);
                
                return {
                    success: true,
                    documentsAdded: args.documents.length,
                    category: args.category
                };
            }
        }
    ]
});

server.start();
```

### Multi-Technique Server

Create a server that supports multiple RAG techniques:

#### JavaScript
```javascript
import { createMCPServer, ConfigurableRAG } from '@rag-templates/mcp';

// Create multiple RAG instances
const ragInstances = {
    basic: new ConfigurableRAG({ technique: 'basic' }),
    colbert: new ConfigurableRAG({ technique: 'colbert' }),
    hyde: new ConfigurableRAG({ technique: 'hyde' }),
    graphrag: new ConfigurableRAG({ technique: 'graphrag' })
};

const server = createMCPServer({
    name: "multi-technique-rag",
    description: "RAG server supporting multiple techniques",
    tools: [
        {
            name: "search_with_technique",
            description: "Search using a specific RAG technique",
            inputSchema: {
                type: 'object',
                properties: {
                    query: { type: 'string', description: 'Search query' },
                    technique: {
                        type: 'string',
                        enum: ['basic', 'colbert', 'hyde', 'graphrag'],
                        default: 'basic',
                        description: 'RAG technique to use'
                    },
                    maxResults: {
                        type: 'integer',
                        minimum: 1,
                        maximum: 20,
                        default: 5
                    }
                },
                required: ['query'],
                additionalProperties: false
            },
            handler: async (args) => {
                const rag = ragInstances[args.technique];
                const result = await rag.query(args.query, {
                    maxResults: args.maxResults,
                    includeSources: true
                });
                
                return {
                    answer: result.answer,
                    technique: args.technique,
                    confidence: result.confidence,
                    sources: result.sources?.length || 0,
                    processingTime: result.metadata?.processingTime
                };
            }
        },
        {
            name: "compare_techniques",
            description: "Compare results across multiple RAG techniques",
            inputSchema: {
                type: 'object',
                properties: {
                    query: { type: 'string', description: 'Search query' },
                    techniques: {
                        type: 'array',
                        items: {
                            type: 'string',
                            enum: ['basic', 'colbert', 'hyde', 'graphrag']
                        },
                        default: ['basic', 'colbert'],
                        description: 'Techniques to compare'
                    }
                },
                required: ['query'],
                additionalProperties: false
            },
            handler: async (args) => {
                const results = {};
                
                for (const technique of args.techniques) {
                    const rag = ragInstances[technique];
                    const result = await rag.query(args.query, {
                        maxResults: 3,
                        includeSources: true
                    });
                    
                    results[technique] = {
                        answer: result.answer,
                        confidence: result.confidence,
                        sources: result.sources?.length || 0,
                        processingTime: result.metadata?.processingTime
                    };
                }
                
                return {
                    query: args.query,
                    results: results,
                    comparison: {
                        bestConfidence: Math.max(...Object.values(results).map(r => r.confidence)),
                        fastestTechnique: Object.entries(results).reduce((a, b) => 
                            a[1].processingTime < b[1].processingTime ? a : b)[0]
                    }
                };
            }
        }
    ]
});

server.start();
```

## Tool Configuration and Routing

### Built-in Tools

The framework provides several built-in tools:

| Tool Name | Description | Input Schema |
|-----------|-------------|--------------|
| `rag_search` | Search the knowledge base | `{query: string, maxResults?: number}` |
| `rag_add_documents` | Add documents to knowledge base | `{documents: Document[]}` |
| `rag_get_stats` | Get knowledge base statistics | `{}` |
| `rag_clear_cache` | Clear response cache | `{}` |
| `rag_list_techniques` | List available RAG techniques | `{}` |
| `rag_switch_technique` | Switch RAG technique | `{technique: string}` |

### Tool Routing

#### JavaScript
```javascript
import { createMCPServer } from '@rag-templates/mcp';

const server = createMCPServer({
    name: "routed-rag-server",
    description: "RAG server with custom routing",
    toolRouter: {
        // Route based on query content
        routeByContent: true,
        routes: [
            {
                pattern: /code|programming|function|class/i,
                technique: 'colbert',  // Use ColBERT for code queries
                maxResults: 10
            },
            {
                pattern: /research|paper|study|analysis/i,
                technique: 'hyde',     // Use HyDE for research queries
                maxResults: 15
            },
            {
                pattern: /graph|relationship|connection/i,
                technique: 'graphrag', // Use GraphRAG for relationship queries
                maxResults: 8
            }
        ],
        defaultTechnique: 'basic'
    },
    tools: ['rag_search', 'rag_add_documents', 'rag_get_stats']
});

server.start();
```

### Custom Tool Middleware

#### JavaScript
```javascript
import { createMCPServer } from '@rag-templates/mcp';

const server = createMCPServer({
    name: "middleware-rag-server",
    description: "RAG server with custom middleware",
    middleware: [
        // Logging middleware
        async (toolName, args, next) => {
            console.log(`[${new Date().toISOString()}] Tool: ${toolName}, Args:`, args);
            const start = Date.now();
            
            try {
                const result = await next();
                const duration = Date.now() - start;
                console.log(`[${new Date().toISOString()}] Tool completed in ${duration}ms`);
                return result;
            } catch (error) {
                console.error(`[${new Date().toISOString()}] Tool failed:`, error);
                throw error;
            }
        },
        
        // Rate limiting middleware
        async (toolName, args, next) => {
            const rateLimiter = getRateLimiter(); // Your rate limiter
            
            if (!await rateLimiter.checkLimit(toolName)) {
                throw new Error('Rate limit exceeded');
            }
            
            return next();
        },
        
        // Input validation middleware
        async (toolName, args, next) => {
            if (toolName === 'rag_search' && (!args.query || args.query.length < 3)) {
                throw new Error('Query must be at least 3 characters long');
            }
            
            return next();
        }
    ],
    tools: ['rag_search', 'rag_add_documents']
});

server.start();
```

## Integration with Existing MCP Ecosystems

### Claude Desktop Integration

#### Complete Example

```javascript
// claude-rag-server.js
import { createMCPServer } from '@rag-templates/mcp';
import { ConfigurableRAG } from '@rag-templates/core';
import fs from 'fs/promises';
import path from 'path';

// Initialize RAG with company-specific configuration
const rag = new ConfigurableRAG({
    technique: 'colbert',
    database: {
        host: process.env.IRIS_HOST || 'localhost',
        port: parseInt(process.env.IRIS_PORT || '52773'),
        namespace: 'COMPANY_KNOWLEDGE'
    },
    llm: {
        provider: 'openai',
        apiKey: process.env.OPENAI_API_KEY,
        model: 'gpt-4o-mini'
    }
});

// Load company documents on startup
async function loadCompanyDocs() {
    const docsDir = process.env.DOCS_DIRECTORY || './company-docs';
    
    try {
        const files = await fs.readdir(docsDir);
        const documents = [];
        
        for (const file of files) {
            if (file.endsWith('.txt') || file.endsWith('.md')) {
                const content = await fs.readFile(path.join(docsDir, file), 'utf8');
                documents.push({
                    content,
                    title: file,
                    source: file,
                    metadata: {
                        type: 'company_doc',
                        lastModified: new Date().toISOString()
                    }
                });
            }
        }
        
        if (documents.length > 0) {
            await rag.addDocuments(documents);
            console.log(`📚 Loaded ${documents.length} company documents`);
        }
    } catch (error) {
        console.warn('Could not load company documents:', error.message);
    }
}

// Create MCP server
const server = createMCPServer({
    name: "company-knowledge",
    description: "Company knowledge base and documentation assistant",
    version: "1.0.0",
    rag: rag,
    tools: [
        {
            name: "search_company_docs",
            description: "Search company documentation and knowledge base",
            inputSchema: {
                type: 'object',
                properties: {
                    query: {
                        type: 'string',
                        description: 'Search query for company documentation'
                    },
                    department: {
                        type: 'string',
                        enum: ['engineering', 'hr', 'finance', 'marketing', 'all'],
                        default: 'all',
                        description: 'Filter by department'
                    },
                    maxResults: {
                        type: 'integer',
                        minimum: 1,
                        maximum: 20,
                        default: 5
                    }
                },
                required: ['query'],
                additionalProperties: false
            },
            handler: async (args) => {
                const result = await rag.query(args.query, {
                    maxResults: args.maxResults,
                    includeSources: true,
                    sourceFilter: args.department !== 'all' ? args.department : undefined
                });
                
                return {
                    answer: result.answer,
                    confidence: result.confidence,
                    sources: result.sources?.map(s => ({
                        title: s.title,
                        source: s.source,
                        relevance: s.similarity
                    })) || [],
                    department: args.department
                };
            }
        },
        {
            name: "add_company_doc",
            description: "Add a new document to the company knowledge base",
            inputSchema: {
                type: 'object',
                properties: {
                    content: {
                        type: 'string',
                        description: 'Document content'
                    },
                    title: {
                        type: 'string',
                        description: 'Document title'
                    },
                    department: {
                        type: 'string',
                        enum: ['engineering', 'hr', 'finance', 'marketing'],
                        description: 'Department this document belongs to'
                    },
                    tags: {
                        type: 'array',
                        items: { type: 'string' },
                        description: 'Tags for categorization'
                    }
                },
                required: ['content', 'title', 'department'],
                additionalProperties: false
            },
            handler: async (args) => {
                await rag.addDocuments([{
                    content: args.content,
                    title: args.title,
                    metadata: {
                        department: args.department,
                        tags: args.tags || [],
                        addedAt: new Date().toISOString(),
                        type: 'user_added'
                    }
                }]);
                
                return {
                    success: true,
                    message: `Document "${args.title}" added to ${args.department} knowledge base`
                };
            }
        }
    ]
});

// Start server and load documents
async function start() {
    await loadCompanyDocs();
    await server.start();
    console.log('🚀 Company knowledge MCP server is running');
}

start().catch(console.error);
```

#### Claude Desktop Configuration

```json
{
  "mcpServers": {
    "company-knowledge": {
      "command": "node",
      "args": ["claude-rag-server.js"],
      "env": {
        "IRIS_HOST": "company-iris.internal",
        "IRIS_PORT": "52773",
        "IRIS_USERNAME": "claude_mcp",
        "IRIS_PASSWORD": "secure_password",
        "OPENAI_API_KEY": "sk-...",
        "DOCS_DIRECTORY": "/path/to/company/docs"
      }
    }
  }
}
```

### Integration with Other MCP Clients

#### Generic MCP Client

```javascript
// generic-mcp-client.js
import { MCPClient } from '@modelcontextprotocol/client';
import { StdioTransport } from '@modelcontextprotocol/transport-stdio';

async function connectToRAGServer() {
    const transport = new StdioTransport({
        command: 'node',
        args: ['rag-server.js']
    });
    
    const client = new MCPClient({
        name: "rag-client",
        version: "1.0.0"
    });
    
    await client.connect(transport);
    
    // List available tools
    const tools = await client.listTools();
    console.log('Available tools:', tools);
    
    // Use RAG search tool
    const searchResult = await client.callTool({
        name: 'rag_search',
        arguments: {
            query: 'What is machine learning?',
            maxResults: 5
        }
    });
    
    console.log('Search result:', searchResult);
    
    await client.disconnect();
}

connectToRAGServer().catch(console.error);
```

## Production Deployment

### Docker Deployment

#### Dockerfile
```dockerfile
FROM node:18-alpine

WORKDIR /app

# Copy package files
COPY package*.json ./
RUN npm ci --only=production

# Copy application code
COPY . .

# Create non-root user
RUN addgroup -g 1001 -S nodejs
RUN adduser -S mcp -u 1001
USER mcp

# Expose MCP port (if using TCP transport)
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD node health-check.js

CMD ["node", "mcp-server.js"]
```

#### docker-compose.yml
```yaml
version: '3.8'

services:
  rag-mcp-server:
    build: .
    environment:
      - IRIS_HOST=iris-db
      - IRIS_PORT=52773
      - IRIS_USERNAME=mcp_user
      - IRIS_PASSWORD=${IRIS_PASSWORD}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - NODE_ENV=production
    volumes:
      - ./docs:/app/docs:ro
    depends_on:
      - iris-db
    restart: unless-stopped
    
  iris-db:
    image: intersystemsdc/iris-community:latest
    environment:
      - ISC_PASSWORD=${IRIS_PASSWORD}
    ports:
      - "52773:52773"
    volumes:
      - iris_data:/usr/irissys/mgr
    restart: unless-stopped

volumes:
  iris_data:
```

### Process Management

#### PM2 Configuration
```javascript
// ecosystem.config.js
module.exports = {
  apps: [{
    name: 'rag-mcp-server',
    script: 'mcp-server.js',
    instances: 1,
    autorestart: true,
    watch: false,
    max_memory_restart: '1G',
    env: {
      NODE_ENV: 'production',
      IRIS_HOST: 'localhost',
      IRIS_PORT: '52773'
    },
    error_file: './logs/err.log',
    out_file: './logs/out.log',
    log_file: './logs/combined.log',
    time: true
  }]
};
```

#### Systemd Service
```ini
# /etc/systemd/system/rag-mcp-server.service
[Unit]
Description=RAG MCP Server
After=network.target

[Service]
Type=simple
User=mcp
WorkingDirectory=/opt/rag-mcp-server
ExecStart=/usr/bin/node mcp-server.js
Restart=always
RestartSec=10
Environment=NODE_ENV=production
Environment=IRIS_HOST=localhost
Environment=IRIS_PORT=52773
EnvironmentFile=/opt/rag-mcp-server/.env

[Install]
WantedBy=multi-user.target
```

### Monitoring and Logging

#### Health Check Endpoint
```javascript
// health-check.js
import { createMCPServer } from '@rag-templates/mcp';

async function healthCheck() {
    try {
        // Test RAG functionality
        const rag = new RAG();
        await rag.query("health check", { maxResults: 1 });
        
        console.log('Health check passed');
        process.exit(0);
    } catch (error) {
        console.error('Health check failed:', error);
        process.exit(1);
    }
}

healthCheck();
```

#### Structured Logging
```javascript
// logger.js
import winston from 'winston';

export const logger = winston.createLogger({
    level: process.env.LOG_LEVEL || 'info',
    format: winston.format.combine(
        winston.format.timestamp(),
        winston.format.errors({ stack: true }),
        winston.format.json()
    ),
    defaultMeta: { service: 'rag-mcp-server' },
    transports: [
        new winston.transports.File({ filename: 'logs/error.log', level: 'error' }),
        new winston.transports.File({ filename: 'logs/combined.log' }),
        new winston.transports.Console({
            format: winston.format.simple()
        })
    ]
});

// Usage in MCP server
import { logger } from './logger.js';

const server = createMCPServer({
    name: "production-rag-server",
    description: "Production RAG MCP server",
    middleware: [
        async (toolName, args, next) => {
            logger.info('Tool called', { toolName, args });
            const start = Date.now();
            
            try {
                const result = await next();
                logger.info('Tool completed', { 
                    toolName, 
                    duration: Date.now() - start 
                });
                return result;
            } catch (error) {
                logger.error('Tool failed', { 
                    toolName, 
                    error: error.message,
                    stack: error.stack 
                });
                throw error;
            }
        }
    ]
});
```

## Troubleshooting

### Common Issues

#### 1. MCP Server Not Starting

**Problem**: Server fails to start with connection errors

**Solutions**:
```javascript
// Add connection testing
import { createMCPServer } from '@rag-templates/mcp';

const server = createMCPServer({
    name: "debug-server",
    description: "Debug MCP server",
    onStartup: async (rag) => {
        // Test database connection
        try {
            await rag.query("test", { maxResults: 1 });
            console.log('✅ Database connection successful');
        } catch (error) {
            console.error('❌ Database connection failed:', error);
            throw error;
        }
    }
});
```

#### 2. Claude Desktop Not Detecting Server

**Problem**: Server doesn't appear in Claude Desktop

**Solutions**:
1. Check Claude Desktop configuration file syntax
2. Verify server executable permissions
3. Check environment variables
4. Review server logs

```bash
# Test server manually
node mcp-server.js

# Check Claude Desktop logs (macOS)
tail -f ~/Library/Logs/Claude/claude.log
```

#### 3. Tool Schema Validation Errors

**Problem**: MCP tools fail with schema validation errors

**Solutions**:
```javascript
// Ensure strict schema compliance
const tools = [
    {
        name: "search_docs",
        description: "Search documentation",
        inputSchema: {
            type: 'object',
            properties: {
                query: { type: 'string' }
            },
            required: ['query'],
            additionalProperties: false  // Important for MCP compliance
        },
        handler: async (args) => {
            // Validate inputs
            if (!args.query || typeof args.query !== 'string') {
                throw new Error('Invalid query parameter');
            }
            
            // Process request
            return { result: "success" };
        }
    }
];
```

### Debug Mode

#### JavaScript
```javascript
import { createMCPServer } from '@rag-templates/mcp';

const server = createMCPServer({
    name: "debug-rag-server",
    description: "Debug RAG MCP server",
    debug: true