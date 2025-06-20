# RAG Templates - Library Consumption Framework

A **dead-simple** library for building Retrieval Augmented Generation (RAG) applications with InterSystems IRIS. Transform from complex setup to immediate productivity with zero-configuration APIs.

## 🚀 Quick Start

### Python - Zero Configuration
```python
from rag_templates import RAG

# Dead simple - works out of the box
rag = RAG()
rag.add_documents(["Your documents here"])
answer = rag.query("What is machine learning?")
print(answer)
```

### JavaScript - Zero Configuration
```javascript
import { RAG } from '@rag-templates/core';

// Dead simple - works out of the box
const rag = new RAG();
await rag.addDocuments(["Your documents here"]);
const answer = await rag.query("What is machine learning?");
console.log(answer);
```

### MCP Server - Zero Configuration
```javascript
import { createMCPServer } from '@rag-templates/mcp';

// Trivial MCP server creation
const server = createMCPServer({
    name: "my-rag-server",
    description: "RAG-powered MCP server"
});

server.start();
```

## 📦 Installation

### Python
```bash
# Install uv (Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install rag-templates
pip install rag-templates
```

### JavaScript/Node.js
```bash
npm install @rag-templates/core
# Optional: MCP integration
npm install @rag-templates/mcp
```

## 🛠️ Interactive Setup (Cross-Language CLI)

RAG Templates provides interactive setup wizards for both Python and JavaScript environments:

### Unified Setup Script
```bash
# Auto-detects environment and launches appropriate CLI
./rag-setup                    # Full interactive setup
./rag-setup --ifind-only      # IFind optimization only
./rag-setup --python          # Force Python CLI
./rag-setup --nodejs          # Force Node.js CLI
```

### Python CLI
```bash
python -m iris_rag.cli reconcile setup   # Full setup wizard
```

### JavaScript CLI
```bash
npm run setup                  # Full setup wizard
npm run setup:ifind           # IFind optimization only
node src/cli/setup-wizard.js  # Direct execution
```

### What the Setup Wizard Configures:
- **IRIS Database Connection**: Interactive connection configuration and testing
- **RAG Pipeline Types**: Choose from 7 pipeline architectures (Vector, ColBERT, GraphRAG, etc.)
- **IFind Optimization**: Full-text search with 70% storage reduction
- **Cross-Language Support**: Works seamlessly with both Python and JavaScript
- **Enterprise Features**: Overlay architecture for existing IRIS content

## ✅ Post-Installation Verification

After installation, verify everything works correctly:

### Environment Setup and Verification
```bash
# 1. Set up environment with uv and install dependencies
make setup-env  # Verifies uv is installed
make install    # Uses uv sync to install all dependencies

# 2. Verify package installation
make validate-iris-rag

# 3. Test database connectivity
make test-dbapi

# 4. Run basic functionality tests
make test-unit
```

### Full End-to-End Validation
```bash
# 1. Load test data (1000+ documents)
make load-1000

# 2. Run comprehensive E2E tests
make test-1000

# 3. Validate all RAG techniques (uv handles environment automatically)
uv run pytest tests/test_*_e2e.py -v
```

### Performance Testing
```bash
# Run RAGAs evaluation with real data
make test-ragas-1000-enhanced

# Benchmark all techniques
make eval-all-ragas-1000
```

### Comprehensive System Test Workup
```bash
# Run comprehensive system test workup across all categories
make test-system-workup

# Run with verbose output for detailed debugging
make test-system-workup-verbose
```

**📋 For detailed test categorization and troubleshooting, see [Existing Tests Guide](docs/EXISTING_TESTS_GUIDE.md)**

## 🎯 Library Consumption Philosophy

This framework transforms RAG from **complex setup** to **dead-simple consumption**:

- **Zero Configuration**: Works immediately with sensible defaults
- **Progressive Complexity**: Simple → Standard → Enterprise APIs
- **Language Parity**: Python and JavaScript feature equivalence
- **MCP-First Design**: Trivial MCP server creation
- **Enterprise Ready**: Scales from prototypes to production

## 📚 Documentation Hub

| Guide | Description |
|-------|-------------|
| **[📖 Library Consumption Guide](docs/LIBRARY_CONSUMPTION_GUIDE.md)** | Complete guide for consuming rag-templates as a library |
| **[🔗 MCP Integration Guide](docs/MCP_INTEGRATION_GUIDE.md)** | Step-by-step MCP server creation |
| **[📋 API Reference](docs/API_REFERENCE.md)** | Complete API documentation (Python & JavaScript) |
| **[🧪 Existing Tests Guide](docs/EXISTING_TESTS_GUIDE.md)** | Test categorization and validation commands |
| **[🔄 Migration Guide](docs/MIGRATION_GUIDE.md)** | Migrate from complex setup to Simple API |
| **[💡 Examples](docs/EXAMPLES.md)** | Comprehensive usage examples |
| **[🔧 Troubleshooting](docs/TROUBLESHOOTING.md)** | Common issues and solutions |

## 🏗️ Three-Tier API Design

### 1. Simple API (Zero Configuration)

**Perfect for**: Prototypes, demos, learning, simple applications

#### Python
```python
from rag_templates import RAG

rag = RAG()  # Zero config
rag.add_documents(["Document 1", "Document 2"])
answer = rag.query("Your question")
```

#### JavaScript
```javascript
import { RAG } from '@rag-templates/core';

const rag = new RAG();  // Zero config
await rag.addDocuments(["Document 1", "Document 2"]);
const answer = await rag.query("Your question");
```

### 2. Standard API (Basic Configuration)

**Perfect for**: Production applications, technique selection, custom configuration

#### Python
```python
from rag_templates import ConfigurableRAG

rag = ConfigurableRAG({
    'technique': 'colbert',
    'llm_provider': 'openai',
    'embedding_model': 'text-embedding-3-small'
})

result = rag.query("Question", {
    'max_results': 5,
    'include_sources': True
})
```

#### JavaScript
```javascript
import { ConfigurableRAG } from '@rag-templates/core';

const rag = new ConfigurableRAG({
    technique: 'colbert',
    llmProvider: 'openai',
    embeddingModel: 'text-embedding-3-small'
});

const result = await rag.query("Question", {
    maxResults: 5,
    includeSources: true
});
```

### 3. Enterprise API (Full Control)

**Perfect for**: Enterprise deployments, advanced features, custom pipelines

#### Python
```python
from rag_templates import ConfigurableRAG
from rag_templates.config import ConfigManager

config = ConfigManager.from_file('enterprise-config.yaml')
rag = ConfigurableRAG(config)

result = rag.query("Complex query", {
    'pipeline_config': {
        'caching': True,
        'monitoring': True,
        'reconciliation': True
    }
})
```

## 🔗 MCP Integration

### Simple MCP Server
```javascript
import { createMCPServer } from '@rag-templates/mcp';

const server = createMCPServer({
    name: "knowledge-server",
    description: "Company knowledge base"
});

server.start();
```

### Advanced MCP Server
```javascript
import { createMCPServer, RAG } from '@rag-templates/mcp';

const rag = new RAG({
    technique: 'graphrag',
    dataSource: './knowledge-base'
});

const server = createMCPServer({
    name: "advanced-rag-server",
    rag: rag,
    tools: ['search_knowledge', 'add_documents', 'get_stats']
});

server.start();
```

## 🛠️ Available RAG Techniques

| Technique | Description | Best For |
|-----------|-------------|----------|
| **basic** | Standard vector similarity | General purpose, fast queries |
| **colbert** | Token-level embeddings | High precision retrieval |
| **crag** | Corrective RAG with self-correction | Accuracy-critical applications |
| **hyde** | Hypothetical Document Embeddings | Complex reasoning tasks |
| **graphrag** | Graph-based knowledge retrieval | Structured knowledge bases |
| **hybrid_ifind** | Multi-modal search combination | Enterprise search |
| **noderag** | Node-based structured retrieval | Hierarchical data |

## 🌟 Key Features

### Dead-Simple Philosophy
- **Zero Setup**: No complex configuration files
- **Immediate Productivity**: Working RAG in 3 lines of code
- **Sensible Defaults**: Production-ready out of the box
- **Progressive Enhancement**: Add complexity only when needed

### Language Parity
- **Python & JavaScript**: Feature-equivalent APIs
- **Consistent Patterns**: Same concepts across languages
- **Cross-Platform**: Works everywhere Python/Node.js runs

### MCP-First Design
- **Trivial Server Creation**: MCP servers in minutes
- **Zero-Config Tools**: Automatic tool generation
- **Environment-Based Config**: No hardcoded secrets
- **Production Ready**: Docker, logging, error handling

### Enterprise Features
- **Advanced RAG Techniques**: 7+ sophisticated implementations
- **Caching & Performance**: Built-in optimization
- **Monitoring & Observability**: Production metrics
- **Security**: Secure parameter binding, input validation

## 📈 Migration Path

### From Complex Setup → Simple API

**Before (Complex)**:
```python
# 50+ lines of setup code
from iris_rag.pipelines.factory import create_pipeline
from common.utils import get_llm_func
from common.iris_connector import get_iris_connection

pipeline = create_pipeline(
    pipeline_type="basic",
    llm_func=get_llm_func(),
    external_connection=get_iris_connection()
)
result = pipeline.run("What is ML?", top_k=5)
```

**After (Simple)**:
```python
# 3 lines of code
from rag_templates import RAG

rag = RAG()
answer = rag.query("What is ML?")
```

## 🔧 Configuration

### Environment Variables
```bash
# Database (optional - uses defaults)
export IRIS_HOST=localhost
export IRIS_PORT=52773
export IRIS_USERNAME=demo
export IRIS_PASSWORD=demo

# LLM Provider (optional - uses defaults)
export OPENAI_API_KEY=your-key
export LLM_MODEL=gpt-4o-mini
```

### Configuration Files (Optional)
```yaml
# simple-config.yaml
technique: "colbert"
llm_provider: "anthropic"
embedding_model: "text-embedding-3-large"
data_source: "./documents"
```

## 🚀 Examples

### Basic Document Q&A
```python
from rag_templates import RAG

rag = RAG()
rag.add_documents([
    "Machine learning is a subset of AI.",
    "Deep learning uses neural networks.",
    "NLP enables language understanding."
])

answer = rag.query("What is machine learning?")
print(answer)  # "Machine learning is a subset of AI..."
```

### Advanced Technique Selection
```python
from rag_templates import ConfigurableRAG

rag = ConfigurableRAG({
    'technique': 'colbert',
    'llm_provider': 'openai',
    'max_results': 10
})

result = rag.query("Explain neural networks", {
    'include_sources': True,
    'min_similarity': 0.8
})

print(f"Answer: {result.answer}")
print(f"Sources: {len(result.sources)}")
```

### MCP Server for Claude Desktop
```javascript
// server.js
import { createMCPServer } from '@rag-templates/mcp';

const server = createMCPServer({
    name: "company-knowledge",
    description: "Company knowledge base search",
    ragConfig: {
        technique: 'graphrag',
        dataSource: './company-docs'
    }
});

server.start();
```

```json
// claude_desktop_config.json
{
  "mcpServers": {
    "company-knowledge": {
      "command": "node",
      "args": ["server.js"]
    }
  }
}
```

## 🤝 Contributing

We welcome contributions! See our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/your-org/rag-templates/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/rag-templates/discussions)
- **Email**: support@your-org.com

---

**Transform your RAG development from complex to dead-simple. Start building in minutes, not hours.**
