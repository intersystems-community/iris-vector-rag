# Enterprise RAG Framework Documentation

Welcome to the Enterprise RAG Framework documentation. This directory contains comprehensive guides for understanding, configuring, and working with our production-ready RAG (Retrieval-Augmented Generation) system built on InterSystems IRIS.

## Quick Start

- **New Users**: Start with [`USER_GUIDE.md`](USER_GUIDE.md) for basic usage and getting started
- **Developers**: See the main [README.md](../README.md) for development setup and architecture overview

## Available Documentation

### Core Documentation
- [`USER_GUIDE.md`](USER_GUIDE.md) - Getting started guide for end users
- [`README.md`](README.md) - This documentation index

### Development & Operations
- [`REPOSITORY_SYNC.md`](REPOSITORY_SYNC.md) - Repository synchronization between internal and public repositories
- [`CONFIGURATION.md`](CONFIGURATION.md) - Configuration system and environment setup
- [`DEVELOPER_GUIDE.md`](DEVELOPER_GUIDE.md) - Developer onboarding and contribution guide

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