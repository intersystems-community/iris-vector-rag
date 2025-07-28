# IRIS RAG MCP Server Implementation Guide

## Overview

This guide shows how to leverage the **existing JavaScript infrastructure** to create MCP servers for all 8 RAG techniques with minimal additional work. The project already has most of the pieces needed!

## Existing Infrastructure Analysis

### ✅ What's Already Built

1. **Complete JavaScript RAG API** ([`nodejs/src/`](../nodejs/src/))
   - [`simple.js`](../nodejs/src/simple.js) - Zero-config RAG class
   - [`standard.js`](../nodejs/src/standard.js) - Configurable RAG with technique selection
   - [`config-manager.js`](../nodejs/src/config-manager.js) - Environment-based configuration

2. **MCP Server Framework** ([`nodejs/src/mcp/`](../nodejs/src/mcp/))
   - [`server.js`](../nodejs/src/mcp/server.js) - Complete MCP server with `createMCPServer()` function
   - [`tools.js`](../nodejs/src/mcp/tools.js) - Tool definitions with JSON schemas
   - Python bridge integration via child processes

3. **Python Bridge** ([`objectscript/python_bridge.py`](../objectscript/python_bridge.py))
   - Direct function calls to Python RAG pipelines
   - JSON-based communication protocol

### 🔧 What Needs Extension

The existing infrastructure only needs **minor extensions** to support all 8 RAG techniques:

1. **Add technique-specific tools** to [`tools.js`](../nodejs/src/mcp/tools.js)
2. **Extend ConfigurableRAG** to support all techniques
3. **Add technique parameter validation**

## Implementation Strategy

### Phase 1: Extend Existing Tools (1-2 hours)

The current [`tools.js`](../nodejs/src/mcp/tools.js) has a generic `rag_search` tool. We need to add technique-specific tools:

```javascript
// Add to nodejs/src/mcp/tools.js
const availableTools = {
    // Existing tools...
    rag_search: { /* existing */ },
    
    // NEW: Technique-specific tools
    rag_basic: {
        description: "Basic RAG with vector similarity search",
        inputSchema: { /* from MCP_TOOL_SCHEMAS.json */ },
        async handler(params) {
            const ragInstance = new ConfigurableRAG({ technique: 'basic' });
            return await ragInstance.query(params.query, params.options);
        }
    },
    
    rag_crag: {
        description: "Corrective RAG with retrieval evaluation",
        inputSchema: { /* from MCP_TOOL_SCHEMAS.json */ },
        async handler(params) {
            const ragInstance = new ConfigurableRAG({ 
                technique: 'crag',
                technique_params: params.technique_params 
            });
            return await ragInstance.query(params.query, params.options);
        }
    },
    
    // ... repeat for all 8 techniques
};
```

### Phase 2: Extend ConfigurableRAG (2-3 hours)

The existing [`standard.js`](../nodejs/src/standard.js) needs technique selection logic:

```javascript
// Extend nodejs/src/standard.js
class ConfigurableRAG {
    constructor(config = {}) {
        this.technique = config.technique || 'basic';
        this.techniqueParams = config.technique_params || {};
        
        // Technique-specific initialization
        switch (this.technique) {
            case 'basic':
                this._initializeBasicRAG(config);
                break;
            case 'crag':
                this._initializeCRAG(config);
                break;
            case 'hyde':
                this._initializeHyDE(config);
                break;
            // ... other techniques
        }
    }
    
    async query(query, options = {}) {
        // Delegate to technique-specific implementation
        return await this._executeQuery(query, options);
    }
}
```

### Phase 3: Add Parameter Validation (1 hour)

Use the schemas from [`MCP_TOOL_SCHEMAS.json`](MCP_TOOL_SCHEMAS.json):

```javascript
// Add to nodejs/src/mcp/tools.js
const toolSchemas = require('../../docs/MCP_TOOL_SCHEMAS.json');

function validateParameters(toolName, params) {
    const schema = toolSchemas.tools[toolName];
    if (!schema) throw new Error(`Unknown tool: ${toolName}`);
    
    // Use existing JSON schema validation
    return validateAgainstSchema(params, schema.inputSchema);
}
```

## Trivial MCP Server Creation

With the existing infrastructure, creating an MCP server for all RAG techniques becomes:

```javascript
// create-iris-rag-server.js
const { createMCPServer } = require('@rag-templates/mcp');

const server = createMCPServer({
    name: "iris-rag-server",
    description: "IRIS RAG server with 8 techniques",
    ragConfig: {
        // Environment-based configuration (already implemented)
        iris: {
            host: process.env.IRIS_HOST,
            port: process.env.IRIS_PORT,
            username: process.env.IRIS_USERNAME,
            password: process.env.IRIS_PASSWORD
        }
    },
    enabledTools: [
        'rag_basic', 'rag_crag', 'rag_hyde', 'rag_graphrag',
        'rag_hybrid_ifind', 'rag_colbert', 'rag_noderag', 'rag_sqlrag',
        'rag_health_check', 'rag_metrics'
    ]
});

server.start();
```

## Leveraging Existing Python Bridge

The existing Python bridge in [`tools.js`](../nodejs/src/mcp/tools.js) already shows how to call Python functions:

```javascript
// Existing pattern in tools.js (lines 18-67)
async function callPythonBridge(functionName, query) {
    return new Promise((resolve, reject) => {
        const pythonProcess = spawn('python3', ['-c', `
import sys
sys.path.insert(0, '${projectRoot}')
from objectscript.python_bridge import ${functionName}
result = ${functionName}('${query}')
print(result)
        `]);
        // ... handle response
    });
}
```

We just need to extend this pattern for each RAG technique:

```javascript
// New functions to add
async function callBasicRAG(query, options) {
    return callPythonBridge('execute_basic_rag', JSON.stringify({query, options}));
}

async function callCRAG(query, options) {
    return callPythonBridge('execute_crag', JSON.stringify({query, options}));
}

// ... etc for all techniques
```

## Required Python Bridge Extensions

Add these functions to [`objectscript/python_bridge.py`](../objectscript/python_bridge.py):

```python
def execute_basic_rag(params_json):
    """Execute Basic RAG pipeline"""
    params = json.loads(params_json)
    pipeline = get_pipeline('basic')
    result = pipeline.execute(params['query'], **params.get('options', {}))
    return json.dumps(format_standard_response('basic', result))

def execute_crag(params_json):
    """Execute CRAG pipeline"""
    params = json.loads(params_json)
    pipeline = get_pipeline('crag')
    result = pipeline.execute(params['query'], **params.get('options', {}))
    return json.dumps(format_standard_response('crag', result))

# ... repeat for all 8 techniques
```

## Implementation Checklist

### ✅ Already Complete
- [x] MCP server framework (`createMCPServer` function)
- [x] Tool registration system with JSON schemas
- [x] Python bridge communication
- [x] Environment-based configuration
- [x] Health checks and basic monitoring
- [x] Simple and Standard RAG APIs

### 🔧 Minimal Extensions Needed

#### 1. Tool Definitions (2-3 hours)
- [ ] Add 8 technique-specific tools to [`tools.js`](../nodejs/src/mcp/tools.js)
- [ ] Import schemas from [`MCP_TOOL_SCHEMAS.json`](MCP_TOOL_SCHEMAS.json)
- [ ] Add parameter validation using existing patterns

#### 2. Python Bridge Functions (2-3 hours)
- [ ] Add 8 technique execution functions to [`python_bridge.py`](../objectscript/python_bridge.py)
- [ ] Implement standard response formatting
- [ ] Add error handling and logging

#### 3. ConfigurableRAG Extensions (1-2 hours)
- [ ] Add technique selection logic to [`standard.js`](../nodejs/src/standard.js)
- [ ] Implement technique-specific parameter handling
- [ ] Add technique validation

#### 4. Testing Integration (1-2 hours)
- [ ] Extend existing MCP tests in [`test_javascript_simple_api_phase3.py`](../tests/test_javascript_simple_api_phase3.py)
- [ ] Add technique-specific test cases
- [ ] Verify real data integration (1000+ docs)

## Total Implementation Time: 6-10 hours

This is dramatically less than building from scratch because:

1. **MCP Protocol Handling**: Already implemented
2. **Tool Registration**: Already implemented  
3. **Python Communication**: Already implemented
4. **Configuration Management**: Already implemented
5. **Error Handling**: Already implemented
6. **Testing Framework**: Already exists

## Example Usage After Implementation

```javascript
// Basic usage (already works)
const { createMCPServer } = require('./nodejs/src/mcp/server');

const server = createMCPServer({
    name: "iris-rag-server",
    description: "IRIS RAG with 8 techniques"
});

await server.start();

// Use any technique
const result = await server.handleToolCall('rag_crag', {
    query: "What are the effects of COVID-19?",
    options: { top_k: 5 },
    technique_params: { 
        confidence_threshold: 0.8,
        correction_strategy: "rewrite" 
    }
});
```

## Conclusion

The existing JavaScript infrastructure makes MCP server creation **trivial**. Instead of building a complex Python-Node.js bridge from scratch, we can:

1. **Extend existing tools** with technique-specific handlers
2. **Add Python bridge functions** using the existing pattern
3. **Leverage existing configuration** and error handling
4. **Use existing test framework** for validation

This approach delivers a production-ready MCP server for all 8 RAG techniques with minimal development effort, building on the substantial work already completed.