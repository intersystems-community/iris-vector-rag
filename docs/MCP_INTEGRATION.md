# Model Context Protocol (MCP) Integration

**Purpose**: Expose iris-vector-rag pipelines as MCP tools for Claude Desktop and other MCP clients
**Status**: Production-Ready
**Protocol Version**: MCP 1.0

## Overview

The Model Context Protocol (MCP) allows iris-vector-rag pipelines to be used directly within Claude Desktop and other MCP-compatible applications. This enables conversational RAG workflows where Claude can query your document collections during conversations.

**Key Benefits**:
- 🔌 **Zero-code integration** - No API endpoints to maintain
- 💬 **Conversational RAG** - Claude queries your docs during chats
- 🚀 **All pipelines supported** - basic, basic_rerank, crag, graphrag, multi_query_rrf, pylate_colbert
- 🔒 **Local execution** - Documents stay on your machine
- ⚡ **Fast** - Direct Python execution, no HTTP overhead

## Quick Start

### 1. Start MCP Server

```bash
# From repository root
python -m iris_vector_rag.mcp

# Server starts on localhost
# Available tools automatically registered
```

### 2. Configure Claude Desktop

Add to your Claude Desktop configuration (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "iris-rag": {
      "command": "python",
      "args": ["-m", "iris_vector_rag.mcp"],
      "env": {
        "OPENAI_API_KEY": "your-openai-api-key",
        "IRIS_HOST": "localhost",
        "IRIS_PORT": "1972",
        "IRIS_NAMESPACE": "USER",
        "IRIS_USER": "_SYSTEM",
        "IRIS_PASSWORD": "SYS"
      }
    }
  }
}
```

### 3. Use in Claude Desktop

In a Claude conversation:

```
You: What does my medical documentation say about diabetes treatment?

Claude: Let me search your documents...
[Uses rag_basic tool automatically]

Claude: Based on your medical documentation, diabetes treatment typically involves...
```

## Available MCP Tools

The MCP server exposes the following tools:

### rag_basic
**Purpose**: Standard vector similarity search
**Best For**: General Q&A, getting started

```json
{
  "name": "rag_basic",
  "description": "Query documents using basic vector similarity search",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "The question to answer"
      },
      "top_k": {
        "type": "number",
        "description": "Number of documents to retrieve (default: 5)"
      }
    },
    "required": ["query"]
  }
}
```

### rag_basic_rerank
**Purpose**: Vector search with cross-encoder reranking
**Best For**: Higher accuracy requirements

```json
{
  "name": "rag_basic_rerank",
  "description": "Query documents with reranking for improved precision",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": {"type": "string"},
      "top_k": {"type": "number"},
      "candidate_k": {"type": "number", "description": "Candidates before reranking (default: 50)"}
    },
    "required": ["query"]
  }
}
```

### rag_crag
**Purpose**: Self-correcting RAG with web search fallback
**Best For**: Current events, fact-checking

```json
{
  "name": "rag_crag",
  "description": "Corrective RAG with self-evaluation and web search fallback",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": {"type": "string"},
      "top_k": {"type": "number"}
    },
    "required": ["query"]
  }
}
```

### rag_graphrag
**Purpose**: Hybrid search with knowledge graphs
**Best For**: Entity relationships, complex queries

```json
{
  "name": "rag_graphrag",
  "description": "Query using vector, text, and knowledge graph retrieval",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": {"type": "string"},
      "top_k": {"type": "number"}
    },
    "required": ["query"]
  }
}
```

### rag_multi_query_rrf
**Purpose**: Multiple query variations with fusion
**Best For**: Ambiguous queries, exploratory search

```json
{
  "name": "rag_multi_query_rrf",
  "description": "Generate query variations and fuse results",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": {"type": "string"},
      "top_k": {"type": "number"},
      "num_variations": {"type": "number", "description": "Query variations to generate (default: 3)"}
    },
    "required": ["query"]
  }
}
```

### rag_pylate_colbert
**Purpose**: Token-level late interaction retrieval
**Best For**: Long documents, fine-grained matching

```json
{
  "name": "rag_pylate_colbert",
  "description": "ColBERT-based retrieval with late interaction",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": {"type": "string"},
      "top_k": {"type": "number"}
    },
    "required": ["query"]
  }
}
```

### health_check
**Purpose**: Verify MCP server and database connectivity

```json
{
  "name": "health_check",
  "description": "Check MCP server and database health",
  "inputSchema": {
    "type": "object",
    "properties": {}
  }
}
```

### list_tools
**Purpose**: List all available RAG tools

```json
{
  "name": "list_tools",
  "description": "List all available RAG pipeline tools",
  "inputSchema": {
    "type": "object",
    "properties": {}
  }
}
```

## Configuration

### Environment Variables

Set these in the `env` section of your Claude Desktop config:

| Variable | Required | Description | Default |
|----------|---------|-------------|---------|
| `OPENAI_API_KEY` | Yes | OpenAI API key for LLM | - |
| `ANTHROPIC_API_KEY` | No | Anthropic API key (alternative to OpenAI) | - |
| `IRIS_HOST` | Yes | IRIS database host | localhost |
| `IRIS_PORT` | Yes | IRIS database port | 1972 |
| `IRIS_NAMESPACE` | Yes | IRIS namespace | USER |
| `IRIS_USER` | Yes | IRIS username | _SYSTEM |
| `IRIS_PASSWORD` | Yes | IRIS password | SYS |
| `TAVILY_API_KEY` | No | Tavily API key (for CRAG web search) | - |

### Custom Pipeline Configuration

To use custom pipeline configurations with MCP:

1. Create pipeline config file: `config/pipelines.yaml`
2. Set environment variable: `PIPELINE_CONFIG_PATH=config/pipelines.yaml`
3. Restart MCP server

Example `config/pipelines.yaml`:
```yaml
basic:
  top_k: 10
  chunk_size: 512

graphrag:
  enable_entity_extraction: true
  entity_types: ["Disease", "Medication", "Symptom"]
```

## Usage Examples

### Example 1: Basic Query

**Claude Desktop Conversation**:
```
User: What does my documentation say about diabetes?

Claude: Let me search your documents...
[Calls rag_basic with query="diabetes"]

Response: Based on your medical documentation, diabetes is a metabolic disorder...
[Shows sources: medical_text.pdf page 45, diabetes_guide.pdf page 12]
```

### Example 2: Complex Entity Query

**Claude Desktop Conversation**:
```
User: What medications interact with metformin?

Claude: Let me use the knowledge graph to find medication interactions...
[Calls rag_graphrag with query="metformin interactions"]

Response: According to your knowledge base, metformin has the following interactions:
1. Insulin - May cause hypoglycemia
2. Alcohol - Increases risk of lactic acidosis
...
[Shows entity relationships and sources]
```

### Example 3: Current Events with CRAG

**Claude Desktop Conversation**:
```
User: What's the latest guidance on diabetes treatment?

Claude: Let me check your docs and search for recent updates...
[Calls rag_crag with query="latest diabetes treatment guidance"]
[CRAG evaluates: Internal docs are 2 years old]
[CRAG triggers: Web search fallback]

Response: Your internal documents mention standard treatment, but recent research from 2024 shows...
[Combines internal docs + web search results]
```

## Advanced Features

### Pipeline Selection

Claude automatically selects the appropriate pipeline based on query characteristics:

- **Entity-heavy queries** → Uses `rag_graphrag`
- **Time-sensitive queries** → Uses `rag_crag` (web fallback)
- **Ambiguous queries** → Uses `rag_multi_query_rrf`
- **Default** → Uses `rag_basic`

You can also explicitly request a specific pipeline:

```
User: Use graphrag to find drug interactions for metformin

Claude: [Explicitly calls rag_graphrag]
```

### Multi-Turn Conversations

MCP supports multi-turn RAG conversations:

```
User: What is diabetes?
Claude: [Calls rag_basic] Diabetes is a metabolic disorder...

User: What are the treatment options?
Claude: [Calls rag_basic with context] Based on the previous answer, treatment options include...

User: What about side effects of metformin?
Claude: [Calls rag_graphrag for drug info] Metformin side effects include...
```

### Response Streaming

MCP supports streaming responses for real-time updates:

```python
# MCP server automatically streams
# Claude Desktop displays results incrementally
```

## Testing MCP Integration

### Test MCP Server

```bash
# 1. Start MCP server
python -m iris_vector_rag.mcp

# 2. Test health check
curl -X POST http://localhost:8000/mcp/health

# Expected response:
{
  "status": "healthy",
  "database": "connected",
  "tools": 8,
  "version": "1.0"
}
```

### Test Tool Invocation

```python
# test_mcp.py
import json
from iris_vector_rag.mcp import MCPServer

server = MCPServer()

# Test rag_basic
result = server.call_tool(
    name="rag_basic",
    arguments={"query": "What is diabetes?", "top_k": 5}
)

print(json.dumps(result, indent=2))
```

### Test in Claude Desktop

1. Open Claude Desktop
2. Start new conversation
3. Type: "Can you search my documents?"
4. Claude should show available RAG tools
5. Ask a question about your documents
6. Claude should query via MCP and return results

## Troubleshooting

### MCP Server Won't Start

**Symptom**: `python -m iris_vector_rag.mcp` fails

**Solutions**:
1. Check Python version: `python --version` (requires 3.11+)
2. Verify installation: `pip list | grep iris-vector-rag`
3. Check database connection: `ping <IRIS_HOST>`
4. Review logs: `tail -f logs/mcp_server.log`

### Claude Desktop Can't Connect

**Symptom**: Claude shows "Tool unavailable" error

**Solutions**:
1. Verify MCP server is running: `curl http://localhost:8000/mcp/health`
2. Check Claude Desktop config path: `~/Library/Application Support/Claude/claude_desktop_config.json`
3. Restart Claude Desktop after config changes
4. Check environment variables are set correctly

### Tools Return Empty Results

**Symptom**: MCP tools execute but return no documents

**Solutions**:
1. Verify documents are loaded: `SELECT COUNT(*) FROM documents`
2. Check embeddings exist: `SELECT COUNT(*) FROM embeddings`
3. Test pipeline directly: `python -c "from iris_vector_rag import create_pipeline; pipeline = create_pipeline('basic'); print(pipeline.query('test'))"`
4. Review document loading: Ensure `pipeline.load_documents()` completed successfully

### Slow Response Times

**Symptom**: MCP queries take >5 seconds

**Solutions**:
1. Use faster pipeline: Switch from `graphrag` to `basic`
2. Reduce top_k: `top_k=3` instead of `top_k=10`
3. Use smaller LLM: `gpt-3.5-turbo` instead of `gpt-4`
4. Enable IRIS EMBEDDING: Model caching speeds up vectorization

### Web Search Fallback Not Working (CRAG)

**Symptom**: `rag_crag` doesn't trigger web search

**Solutions**:
1. Set Tavily API key: `export TAVILY_API_KEY=your-key`
2. Check relevance threshold: May be too low (increase to 0.7)
3. Verify CRAG configuration: `cat config/pipelines.yaml`
4. Review logs for evaluation results

## Architecture

### MCP Server Architecture

```
Claude Desktop
    ↓ (MCP protocol)
MCP Server (Python)
    ↓ (function call)
iris_vector_rag.create_pipeline()
    ↓ (SQL queries)
IRIS Database
    ↓ (vector search)
Documents + Embeddings
    ↓ (results)
Claude Desktop
```

### Tool Registration

Tools are automatically registered on server startup:

```python
# iris_vector_rag/mcp/server.py
def register_tools():
    tools = [
        create_rag_tool("basic"),
        create_rag_tool("basic_rerank"),
        create_rag_tool("crag"),
        create_rag_tool("graphrag"),
        create_rag_tool("multi_query_rrf"),
        create_rag_tool("pylate_colbert"),
        create_health_check_tool(),
        create_list_tools_tool(),
    ]
    return tools
```

### Request Flow

1. **User query** → Claude Desktop
2. **Tool selection** → Claude chooses appropriate RAG tool
3. **MCP request** → Sent to MCP server
4. **Pipeline execution** → iris_vector_rag processes query
5. **Results** → Returned via MCP protocol
6. **Response generation** → Claude synthesizes answer
7. **Display** → Results shown to user

## Security Considerations

### Local Execution

- MCP server runs locally (localhost only by default)
- No data sent to external services (except LLM API for generation)
- Documents stay on your machine

### API Key Management

**Best Practice**: Use environment variables, not hardcoded keys

```json
{
  "mcpServers": {
    "iris-rag": {
      "env": {
        "OPENAI_API_KEY": "${OPENAI_API_KEY}",  // Read from shell env
      }
    }
  }
}
```

Set in shell:
```bash
export OPENAI_API_KEY=your-key
```

### Database Access

- MCP server requires IRIS database credentials
- Use read-only user if only querying (not updating)
- Restrict network access to IRIS (localhost or VPN only)

## Performance Tuning

### Connection Pooling

MCP server uses connection pooling for IRIS:

```python
# config/mcp_config.yaml
database:
  pool_size: 10
  max_overflow: 5
```

### Caching

Enable result caching for repeated queries:

```python
# config/mcp_config.yaml
caching:
  enabled: true
  ttl: 3600  # 1 hour
  max_entries: 1000
```

### Concurrent Requests

MCP server supports concurrent tool invocations:

```python
# config/mcp_config.yaml
server:
  max_concurrent_requests: 10
```

## Production Deployment

### Systemd Service (Linux)

Create `/etc/systemd/system/iris-rag-mcp.service`:

```ini
[Unit]
Description=IRIS RAG MCP Server
After=network.target

[Service]
Type=simple
User=iris-rag
WorkingDirectory=/opt/iris-vector-rag
Environment="OPENAI_API_KEY=your-key"
Environment="IRIS_HOST=localhost"
ExecStart=/usr/bin/python3 -m iris_vector_rag.mcp
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable iris-rag-mcp
sudo systemctl start iris-rag-mcp
sudo systemctl status iris-rag-mcp
```

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "-m", "iris_vector_rag.mcp"]
```

```bash
docker build -t iris-rag-mcp .
docker run -d -p 8000:8000 \
  -e OPENAI_API_KEY=your-key \
  -e IRIS_HOST=host.docker.internal \
  iris-rag-mcp
```

## See Also

- [User Guide](USER_GUIDE.md) - Complete iris-vector-rag usage guide
- [API Reference](API_REFERENCE.md) - Full API documentation
- [Pipeline Guide](PIPELINE_GUIDE.md) - Pipeline selection guide
- [Model Context Protocol Specification](https://spec.modelcontextprotocol.io/) - Official MCP docs
