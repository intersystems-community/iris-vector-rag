# RAG Templates - Enterprise RAG Framework

**Production-ready RAG applications with InterSystems IRIS.** Zero-configuration APIs, enterprise-grade architecture, and seamless LangChain integration.

## 🚀 Quick Start

### Python - Zero Configuration
```python
from rag_templates import RAG

# Works immediately - no setup required
rag = RAG()
rag.add_documents(["Your documents here"])
answer = rag.query("What is machine learning?")
print(answer)
```

### JavaScript - Zero Configuration  
```javascript
import { RAG } from '@rag-templates/core';

const rag = new RAG();
await rag.addDocuments(["Your documents here"]);
const answer = await rag.query("What is machine learning?");
console.log(answer);
```

### ObjectScript Integration
```objectscript
// Direct IRIS integration via Embedded Python
Set bridge = ##class(RAG.PythonBridge).%New()
Set result = bridge.Query("What is machine learning?", "basic")
Write result.answer
```

## 🏗️ Core Architecture

### Schema Manager
Centralized schema management with automatic migration support:
- **Universal dimension authority** for all vector tables
- **Automatic schema detection** and migration
- **Customizable table names** and field configurations
- **Version tracking** and rollback capabilities

### IRISVectorStore Interface
LangChain-compatible vector store with enterprise features:
```python
from rag_templates.storage import IRISVectorStore

# Drop-in LangChain replacement
vector_store = IRISVectorStore(connection_manager, config_manager)
retriever = vector_store.as_retriever(search_kwargs={"k": 5})
```

### Configuration System
Environment-aware configuration with validation:
```python
from rag_templates.config import ConfigurationManager

config = ConfigurationManager()
# Supports RAG_DATABASE__IRIS__HOST env vars
# Automatic YAML loading with schema validation
```

## 🛠️ Available RAG Techniques

| Technique | Description | Best For | Status |
|-----------|-------------|----------|---------|
| **basic** | Standard vector similarity | General purpose, fast queries | ✅ Production |
| **colbert** | Token-level embeddings with MaxSim | High precision retrieval | ✅ Production* |
| **crag** | Corrective RAG with self-correction | Accuracy-critical applications | ✅ Production |
| **hyde** | Hypothetical Document Embeddings | Complex reasoning tasks | ✅ Production |
| **graphrag** | Graph-based knowledge retrieval | Structured knowledge bases | ✅ Production |
| **hybrid_ifind** | Multi-modal search combination | Enterprise search | ✅ Production |
| **noderag** | Node-based structured retrieval | Hierarchical data | ✅ Production |

*ColBERT: Includes experimental [Pylate integration](https://github.com/lightonai/pylate) with pluggable backend support (`native`/`pylate`).

## 🎯 Developer Experience

### Three-Tier API Design

**Simple API** - Zero configuration for prototypes:
```python
rag = RAG()  # Works immediately
```

**Standard API** - Production configuration:
```python
rag = ConfigurableRAG({
    'technique': 'colbert',
    'llm_provider': 'openai'
})
```

**Enterprise API** - Full control:
```python
config = ConfigManager.from_file('enterprise-config.yaml')
rag = ConfigurableRAG(config)
```

### Key Configuration Options

| Variable | Description | Default |
|----------|-------------|---------|
| `RAG_DATABASE__IRIS__HOST` | IRIS database host | `localhost` |
| `RAG_EMBEDDING__MODEL` | Embedding model name | `all-MiniLM-L6-v2` |
| `RAG_LLM__PROVIDER` | LLM provider | `openai` |
| `RAG_TECHNIQUE` | RAG technique to use | `basic` |

## 🔗 MCP Integration

The Model Context Protocol (MCP) integration allows you to easily deploy and manage RAG services as "microservices". This design enables flexible deployment across various environments and seamless integration with existing enterprise systems.

### Creating MCP Servers

Create MCP servers in minutes:
```javascript
import { createMCPServer } from '@rag-templates/mcp';

const server = createMCPServer({
    name: "knowledge-server",
    description: "Company knowledge base",
    ragConfig: { technique: 'graphrag' }
});

server.start();
```

### IRIS SQL Tool Integration

The IRIS SQL tool provides direct SQL access and advanced vector search capabilities within your MCP-deployed RAG services. It allows for efficient data manipulation and retrieval directly from InterSystems IRIS databases, leveraging its powerful SQL and vector functionalities. This integration streamlines data management for RAG applications, enabling complex queries and high-performance vector lookups.

For detailed setup and usage, refer to the [MCP Integration Guide](docs/MCP_INTEGRATION_GUIDE.md).

## 📚 Documentation

| Guide | Description |
|-------|-------------|
| **[📖 User Guide](docs/USER_GUIDE.md)** | Complete usage guide and best practices |
| **[🔗 MCP Integration Guide](docs/MCP_INTEGRATION_GUIDE.md)** | Multi-Cloud Platform integration, MCP server creation, and IRIS SQL tool usage |
| **[📋 Documentation](docs/README.md)** | Additional documentation and guides |

## ✅ Verification

```bash
# Quick setup and validation
make setup-env && make install
make validate-iris-rag && make test-unit

# Full end-to-end testing with 1000+ documents
make load-1000 && make test-1000

# Performance benchmarking
make test-ragas-1000-enhanced
```

## 🌟 Key Features

- **Zero Configuration**: Production-ready defaults, works immediately
- **Enterprise Architecture**: Schema management, migrations, monitoring
- **LangChain Compatible**: Drop-in replacement for existing workflows  
- **Multi-Language**: Python, JavaScript, and ObjectScript support
- **MCP-First Design**: Trivial MCP server creation
- **Advanced RAG**: 7+ sophisticated retrieval techniques
- **Performance Optimized**: Built-in caching and optimization

## 📚 References & Research

### RAG Technique Papers & Implementations

| Technique | Original Paper | Key Repository | Additional Resources |
|-----------|---------------|----------------|---------------------|
| **Basic RAG** | [Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401) | [Facebook Research](https://github.com/facebookresearch/RAG) | [LangChain RAG](https://python.langchain.com/docs/tutorials/rag/) |
| **ColBERT** | [ColBERT: Efficient and Effective Passage Retrieval](https://arxiv.org/abs/2004.12832) | [Stanford ColBERT](https://github.com/stanford-futuredata/ColBERT) | [Pylate Integration](https://github.com/lightonai/pylate) |
| **CRAG** | [Corrective Retrieval Augmented Generation](https://arxiv.org/abs/2401.15884) | [CRAG Implementation](https://github.com/HuskyInSalt/CRAG) | [LangGraph CRAG](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_crag/) |
| **HyDE** | [Precise Zero-Shot Dense Retrieval](https://arxiv.org/abs/2212.10496) | [HyDE Official](https://github.com/texttron/hyde) | [LangChain HyDE](https://python.langchain.com/docs/how_to/hyde/) |
| **GraphRAG** | [From Local to Global: A Graph RAG Approach](https://arxiv.org/abs/2404.16130) | [Microsoft GraphRAG](https://github.com/microsoft/graphrag) | [Neo4j GraphRAG](https://github.com/neo4j/neo4j-graphrag-python) |
| **NodeRAG** | [Hierarchical Text Retrieval](https://arxiv.org/abs/2310.20501) | [NodeRAG Implementation](https://github.com/microsoft/noderag) | [Hierarchical Retrieval](https://python.langchain.com/docs/how_to/parent_document_retriever/) |

### Core Technologies

- **Vector Databases**: [InterSystems IRIS Vector Search](https://docs.intersystems.com/iris20241/csp/docbook/DocBook.UI.Page.cls?KEY=GSQL_vecsearch)
- **Embeddings**: [Sentence Transformers](https://github.com/UKPLab/sentence-transformers), [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)
- **LLM Integration**: [LangChain](https://github.com/langchain-ai/langchain), [OpenAI API](https://platform.openai.com/docs/api-reference)
- **Evaluation**: [RAGAS Framework](https://github.com/explodinggradients/ragas)

## 🤝 Contributing

We welcome contributions! See our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

**Transform your RAG development from complex to enterprise-ready. Start building in minutes, scale to production.**
