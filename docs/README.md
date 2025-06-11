# RAG Templates Documentation

Welcome to the RAG Templates project documentation. This directory contains all essential documentation for understanding, configuring, and working with the RAG (Retrieval-Augmented Generation) system.

## Quick Start

- **New Users**: Start with [`USER_GUIDE.md`](USER_GUIDE.md)
- **Developers**: See [`DEVELOPER_GUIDE.md`](DEVELOPER_GUIDE.md)
- **Configuration**: All configuration options in [`CONFIGURATION.md`](CONFIGURATION.md)
- **API Reference**: Complete API documentation in [`API_REFERENCE.md`](API_REFERENCE.md)

## Documentation Structure

### Core Documentation
- [`USER_GUIDE.md`](USER_GUIDE.md) - Getting started guide for end users
- [`DEVELOPER_GUIDE.md`](DEVELOPER_GUIDE.md) - Development setup and contribution guide
- [`CONFIGURATION.md`](CONFIGURATION.md) - Complete configuration reference
- [`API_REFERENCE.md`](API_REFERENCE.md) - API documentation and examples

### Operational Guides ([`guides/`](guides/))
- [`DEPLOYMENT_GUIDE.md`](guides/DEPLOYMENT_GUIDE.md) - Production deployment instructions
- [`PERFORMANCE_GUIDE.md`](guides/PERFORMANCE_GUIDE.md) - Performance optimization and tuning
- [`SECURITY_GUIDE.md`](guides/SECURITY_GUIDE.md) - Security best practices and configuration
- [`DOCKER_TROUBLESHOOTING_GUIDE.md`](guides/DOCKER_TROUBLESHOOTING_GUIDE.md) - Docker-specific troubleshooting
- [`BRANCH_DEPLOYMENT_CHECKLIST.md`](guides/BRANCH_DEPLOYMENT_CHECKLIST.md) - Pre-deployment verification
- [`COMMIT_MESSAGE.md`](guides/COMMIT_MESSAGE.md) - Git commit message standards

### Technical Reference ([`reference/`](reference/))
- [`CHUNKING_STRATEGY_AND_USAGE.md`](reference/CHUNKING_STRATEGY_AND_USAGE.md) - Document chunking strategies
- [`IRIS_SQL_VECTOR_OPERATIONS.md`](reference/IRIS_SQL_VECTOR_OPERATIONS.md) - IRIS vector database operations
- [`MONITORING_SYSTEM.md`](reference/MONITORING_SYSTEM.md) - System monitoring and observability

## RAG Techniques Implemented

This project implements multiple RAG techniques:

- **Basic RAG** - Traditional retrieval-augmented generation
- **ColBERT** - Contextualized late interaction over BERT
- **NodeRAG** - Node-based retrieval with graph structures
- **GraphRAG** - Graph-based retrieval and reasoning
- **HyDE** - Hypothetical document embeddings
- **CRAG** - Corrective retrieval-augmented generation
- **Hybrid IFind** - Hybrid information finding approach

## Configuration Overview

The system supports multiple configuration approaches:

1. **YAML Configuration Files** - Primary configuration method
2. **Environment Variables** - Runtime overrides with `RAG_` prefix
3. **CLI Arguments** - Command-line parameter passing
4. **Reconciliation Framework** - Automated configuration management

See [`CONFIGURATION.md`](CONFIGURATION.md) for complete details.

## Architecture

The system follows a modular architecture with:

- **Pipeline Framework** - Unified interface for all RAG techniques
- **Configuration Management** - Centralized configuration handling
- **Database Integration** - IRIS vector database with optimized operations
- **Evaluation Framework** - Comprehensive testing and benchmarking
- **CLI Interface** - Command-line tools for operation and management

## Historical Documentation

Historical documentation, status reports, and legacy implementations have been archived in `../archive/archived_documentation/` to maintain project history while keeping current documentation focused and navigable.

## Getting Help

1. Check the relevant guide in this documentation
2. Review the troubleshooting sections in operational guides
3. Consult the API reference for implementation details
4. Check archived documentation for historical context

## Contributing

See [`DEVELOPER_GUIDE.md`](DEVELOPER_GUIDE.md) for development setup, coding standards, and contribution guidelines.