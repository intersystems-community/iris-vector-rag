# Enterprise RAG Framework Documentation

Welcome to the Enterprise RAG Framework documentation. This directory contains comprehensive guides for understanding, configuring, and working with our production-ready RAG (Retrieval-Augmented Generation) system built on InterSystems IRIS.

## 🚀 Quick Start

**Get started in minutes with our new one-command setup:**

- **🎯 New Users**: Start with [`QUICK_START_GUIDE.md`](QUICK_START_GUIDE.md) for one-command setup with intelligent profiles
- **📖 Detailed Usage**: See [`USER_GUIDE.md`](USER_GUIDE.md) for comprehensive usage guide and examples
- **👨‍💻 Developers**: See [`DEVELOPER_GUIDE.md`](DEVELOPER_GUIDE.md) for development setup and Quick Start extension

### Quick Commands
```bash
make quick-start-minimal    # Development setup (50 docs, 2GB RAM, ~5 min)
make quick-start-standard   # Production setup (500 docs, 4GB RAM, ~15 min)
make quick-start-extended   # Enterprise setup (5000 docs, 8GB RAM, ~30 min)
make quick-start           # Interactive wizard with profile selection
```

## 📚 Documentation Structure

### 🎯 Getting Started
| Document | Description | Audience |
|----------|-------------|----------|
| **[`QUICK_START_GUIDE.md`](QUICK_START_GUIDE.md)** | **NEW!** One-command setup with intelligent profiles | All Users |
| [`USER_GUIDE.md`](USER_GUIDE.md) | Comprehensive usage guide and best practices | End Users |
| [`EXAMPLES.md`](EXAMPLES.md) | Real-world examples and use cases | All Users |

### 🔧 Development & Configuration
| Document | Description | Audience |
|----------|-------------|----------|
| [`DEVELOPER_GUIDE.md`](DEVELOPER_GUIDE.md) | Developer onboarding, Quick Start extension, and contribution guide | Developers |
| [`CONFIGURATION.md`](CONFIGURATION.md) | Configuration system, Quick Start templates, and environment setup | Developers, DevOps |
| [`API_REFERENCE.md`](API_REFERENCE.md) | Complete API documentation for Python and JavaScript | Developers |

### 🏗️ Architecture & Integration
| Document | Description | Audience |
|----------|-------------|----------|
| [`MCP_INTEGRATION_GUIDE.md`](MCP_INTEGRATION_GUIDE.md) | Multi-Cloud Platform integration and MCP server creation | Architects, DevOps |
| [`LIBRARY_CONSUMPTION_GUIDE.md`](LIBRARY_CONSUMPTION_GUIDE.md) | Library consumption framework and patterns | Developers |
| [`LIBRARY_CONSUMPTION_FRAMEWORK_ARCHITECTURE.md`](LIBRARY_CONSUMPTION_FRAMEWORK_ARCHITECTURE.md) | Framework architecture and design patterns | Architects |

### 🔄 Operations & Maintenance
| Document | Description | Audience |
|----------|-------------|----------|
| [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) | Common issues and solutions | All Users |
| [`MIGRATION_GUIDE.md`](MIGRATION_GUIDE.md) | Migration strategies and upgrade paths | DevOps |
| [`REPOSITORY_SYNC.md`](REPOSITORY_SYNC.md) | Repository synchronization between internal and public repositories | Maintainers |

### 📋 Reference & Testing
| Document | Description | Audience |
|----------|-------------|----------|
| [`EXISTING_TESTS_GUIDE.md`](EXISTING_TESTS_GUIDE.md) | Testing strategy and test execution | Developers, QA |
| [`PIPELINE_CHUNKING_ARCHITECTURE_REFACTOR.md`](PIPELINE_CHUNKING_ARCHITECTURE_REFACTOR.md) | Pipeline architecture and chunking strategies | Architects |
| [`PIPELINE_MIGRATION_STRATEGY.md`](PIPELINE_MIGRATION_STRATEGY.md) | Pipeline migration and upgrade strategies | DevOps |

### 📁 Organized Directories
- **[`architecture/`](architecture/)** - System architecture diagrams and specifications
- **[`design/`](design/)** - Design documents and technical specifications
- **[`guides/`](guides/)** - Step-by-step guides and tutorials
- **[`reference/`](reference/)** - API references and technical documentation
- **[`project_governance/`](project_governance/)** - Project governance and status reports

## RAG Techniques Implemented

This framework implements multiple advanced RAG techniques:

| Technique | Status | Description |
|-----------|--------|-------------|
| **Basic RAG** | ✅ Production | Traditional retrieval-augmented generation with semantic search |
| **ColBERT** | ✅ Production | Contextualized late interaction over BERT for fine-grained retrieval |
| **NodeRAG** | ✅ Production | Node-based retrieval with graph structures |
| **GraphRAG** | ✅ Production | Graph-based retrieval and reasoning for complex queries |
| **HyDE** | ✅ Production | Hypothetical document embeddings for improved retrieval |
| **CRAG** | ✅ Production | Corrective retrieval-augmented generation with self-correction |
| **Hybrid IFind** | ✅ Production | Hybrid information finding combining multiple search approaches |

## References

### RAG Techniques

Each RAG technique implemented in this framework is based on established research and implementations:

#### Basic RAG
- [RAG: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) - Original RAG paper by Facebook AI Research

#### ColBERT
- [ColBERT: Efficient and Effective Passage Retrieval](https://arxiv.org/abs/2004.12832) - Original ColBERT paper
- [Stanford ColBERT](https://github.com/stanford-futuredata/ColBERT) - Official implementation

#### NodeRAG
- [Hierarchical Text Retrieval - NodeRAG](https://arxiv.org/abs/2310.20501) - NodeRAG research paper
- [NodeRAG Implementation](https://github.com/microsoft/noderag) - Microsoft's official implementation

#### GraphRAG
- [From Local to Global: A Graph RAG Approach](https://arxiv.org/abs/2404.16130) - Microsoft GraphRAG paper
- [Microsoft GraphRAG](https://github.com/microsoft/graphrag) - Official Microsoft implementation
- [Neo4j GraphRAG](https://github.com/neo4j/neo4j-graphrag-python) - Neo4j's GraphRAG implementation

#### HyDE
- [Precise Zero-Shot Dense Retrieval - HyDE](https://arxiv.org/abs/2212.10496) - Original HyDE paper
- [HyDE Official](https://github.com/texttron/hyde) - Official implementation

#### CRAG
- [Corrective Retrieval Augmented Generation](https://arxiv.org/abs/2401.15884) - CRAG research paper
- [CRAG Implementation](https://github.com/HuskyInSalt/CRAG) - Community implementation

#### Hybrid IFind
- [InterSystems IFind Documentation](https://github.com/intersystems/iris-ifind) - InterSystems IFind implementation and documentation

*Note: Hybrid IFind is a single, unified pipeline that combines IRIS IFind keyword search with vector similarity search using Reciprocal Rank Fusion (RRF) for optimal retrieval performance. This integrated approach leverages both semantic understanding through vector embeddings and precise keyword matching through IRIS's native IFind capabilities.*

## Architecture Overview

The framework follows a modular, enterprise-ready architecture:

- **🏗️ Schema Manager** - Automated database schema management and migrations
- **🔧 Configuration System** - Environment-aware configuration with validation
- **🗄️ IRISVectorStore** - LangChain-compatible vector store interface
- **🐍 ObjectScript Integration** - Native IRIS integration via Embedded Python
- **🔄 Pipeline Framework** - Unified interface for all RAG techniques
- **📊 Evaluation Framework** - Comprehensive testing and benchmarking

## Key Features

### 🎯 Enterprise Ready
- Production-tested with 50K+ documents
- Automated schema management and migrations
- Environment-aware configuration system
- Comprehensive error handling and logging

### 🔧 Developer Focused
- Simple, intuitive APIs for rapid development
- Extensive documentation and examples
- Test-driven development with real data validation
- Modular design with clean separation of concerns

### 📊 Advanced Capabilities
- Multiple embedding backends (Sentence Transformers, OpenAI, Hugging Face)
- Intelligent chunking strategies (recursive, semantic, adaptive, hybrid)
- LangChain compatibility via IRISVectorStore interface
- Native ObjectScript integration for advanced IRIS features

## Getting Started

1. **Installation**: Follow the setup instructions in the main [README.md](../README.md)
2. **Basic Usage**: Start with [USER_GUIDE.md](USER_GUIDE.md) for your first RAG pipeline
3. **Development**: See the main README for architecture and development setup

## Available Documentation

- [USER_GUIDE.md](USER_GUIDE.md) - Comprehensive user guide with examples
- [README.md](README.md) - This documentation index

For complete documentation, architecture details, and API references, see the main [README.md](../README.md).

## Contributing

This framework follows test-driven development principles:

1. **RED**: Write failing tests first
2. **GREEN**: Implement minimum code to pass tests
3. **REFACTOR**: Clean up code while keeping tests passing

All contributions must:
- Follow clean architecture principles
- Include comprehensive test coverage
- Use configuration abstractions
- Maintain backward compatibility