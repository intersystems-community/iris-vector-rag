# RAG Templates Documentation

Welcome to the RAG Templates project documentation. This directory contains comprehensive documentation for understanding, configuring, and working with the RAG (Retrieval-Augmented Generation) system built on InterSystems IRIS.

## Quick Start

- **New Users**: Start with [`USER_GUIDE.md`](USER_GUIDE.md) for basic usage and getting started
- **Developers**: See [`DEVELOPER_GUIDE.md`](DEVELOPER_GUIDE.md) for development setup and architecture
- **Configuration**: Complete configuration reference in [`CONFIGURATION.md`](CONFIGURATION.md)
- **API Reference**: Full API documentation in [`API_REFERENCE.md`](API_REFERENCE.md)

## Documentation Structure

### Core Documentation
- [`USER_GUIDE.md`](USER_GUIDE.md) - Getting started guide for end users
- [`DEVELOPER_GUIDE.md`](DEVELOPER_GUIDE.md) - Development setup, architecture, and contribution guide
- [`CONFIGURATION.md`](CONFIGURATION.md) - Complete configuration reference including CLI usage and reconciliation framework
- [`API_REFERENCE.md`](API_REFERENCE.md) - API documentation and examples

### Operational Guides ([`guides/`](guides/))
- [`DEPLOYMENT_GUIDE.md`](guides/DEPLOYMENT_GUIDE.md) - Production deployment instructions and strategies
- [`PERFORMANCE_GUIDE.md`](guides/PERFORMANCE_GUIDE.md) - Performance optimization and tuning recommendations
- [`SECURITY_GUIDE.md`](guides/SECURITY_GUIDE.md) - Security best practices and configuration
- [`DOCKER_TROUBLESHOOTING_GUIDE.md`](guides/DOCKER_TROUBLESHOOTING_GUIDE.md) - Docker-specific troubleshooting and solutions
- [`BRANCH_DEPLOYMENT_CHECKLIST.md`](guides/BRANCH_DEPLOYMENT_CHECKLIST.md) - Pre-deployment verification checklist
- [`COMMIT_MESSAGE.md`](guides/COMMIT_MESSAGE.md) - Git commit message standards and conventions

### Technical Reference ([`reference/`](reference/))
- [`CHUNKING_STRATEGY_AND_USAGE.md`](reference/CHUNKING_STRATEGY_AND_USAGE.md) - Document chunking strategies and implementation
- [`IRIS_SQL_VECTOR_OPERATIONS.md`](reference/IRIS_SQL_VECTOR_OPERATIONS.md) - IRIS vector database operations and SQL reference
- [`MONITORING_SYSTEM.md`](reference/MONITORING_SYSTEM.md) - System monitoring and observability setup
- [`KNOWN_ISSUES.md`](reference/KNOWN_ISSUES.md) - Known issues and their workarounds
- [`DAEMON_MODE_TESTING_SUMMARY.md`](reference/DAEMON_MODE_TESTING_SUMMARY.md) - Daemon mode testing results and analysis

### Design Documentation ([`design/`](design/))
- [`COMPREHENSIVE_GENERALIZED_RECONCILIATION_DESIGN.md`](design/COMPREHENSIVE_GENERALIZED_RECONCILIATION_DESIGN.md) - Comprehensive design for generalized reconciliation system
- [`RECONCILIATION_REFACTORING_PROPOSAL.md`](design/RECONCILIATION_REFACTORING_PROPOSAL.md) - Proposal for reconciliation system refactoring

### Project Governance ([`project_governance/`](project_governance/))
- [`BACKLOG.md`](project_governance/BACKLOG.md) - Project backlog and task tracking
- [`DOCS_CONTENT_REFINEMENT_SPEC.md`](project_governance/DOCS_CONTENT_REFINEMENT_SPEC.md) - Documentation content refinement specifications
- [`LLM_CACHE_PROJECT_COMPLETION_REPORT.md`](project_governance/LLM_CACHE_PROJECT_COMPLETION_REPORT.md) - LLM cache project completion report
- [`MERGE_REFACTOR_BRANCH_TO_MAIN_SPEC.md`](project_governance/MERGE_REFACTOR_BRANCH_TO_MAIN_SPEC.md) - Merge and refactor branch specifications
- [`PROJECT_STRUCTURE_REFINEMENT_SPEC.md`](project_governance/PROJECT_STRUCTURE_REFINEMENT_SPEC.md) - Project structure refinement specifications
- Project completion notes and milestone documentation

#### Status Reports ([`project_governance/status_reports/`](project_governance/status_reports/))
- [`PROJECT_STATUS_REPORT_2025-06-11.md`](project_governance/status_reports/PROJECT_STATUS_REPORT_2025-06-11.md) - Project status report for June 11, 2025

## RAG Techniques Implemented

This project implements multiple advanced RAG techniques:

- **Basic RAG** - Traditional retrieval-augmented generation with semantic search
- **ColBERT** - Contextualized late interaction over BERT for fine-grained retrieval
- **NodeRAG** - Node-based retrieval with graph structures
- **GraphRAG** - Graph-based retrieval and reasoning for complex queries
- **HyDE** - Hypothetical document embeddings for improved retrieval
- **CRAG** - Corrective retrieval-augmented generation with self-correction
- **Hybrid IFind** - Hybrid information finding combining multiple search approaches

## Configuration Overview

The system supports flexible configuration through multiple approaches:

1. **YAML Configuration Files** - Primary configuration method with structured settings
2. **Environment Variables** - Runtime overrides with `RAG_` prefix for deployment flexibility
3. **CLI Arguments** - Command-line parameter passing for operational tasks
4. **Dynamic Pipeline Configuration** - Config-driven pipeline loading from [`config/pipelines.yaml`](../config/pipelines.yaml)
5. **Reconciliation Framework** - Automated configuration management and state healing

See [`CONFIGURATION.md`](CONFIGURATION.md) for complete configuration details and examples.

## Architecture Overview

The system follows a modular, enterprise-ready architecture:

- **Pipeline Framework** - Unified interface for all RAG techniques with dependency injection
- **Configuration Management** - Centralized configuration handling with validation
- **Database Integration** - IRIS vector database with optimized HNSW indexing
- **Evaluation Framework** - Comprehensive testing and benchmarking with RAGAS integration
- **CLI Interface** - Command-line tools for operation and management
- **LLM Caching** - Intelligent response caching for cost optimization
- **Reconciliation System** - Desired-state configuration management

## Key Features

### 🎯 Core Capabilities
- Multiple RAG techniques with unified interface
- Dynamic pipeline loading and configuration
- Production-ready architecture with comprehensive error handling
- Native IRIS integration with vector search optimization
- Personal Assistant adapter for existing integrations

### 🔧 Technical Excellence
- Full type safety with comprehensive validation
- Modular design with clean separation of concerns
- Extensive test coverage with real data validation
- Enterprise-scale performance (50K+ documents)
- Security-first design with parameter binding

### 📊 Advanced Features
- Multiple embedding backends (Sentence Transformers, OpenAI, Hugging Face)
- Intelligent chunking strategies (recursive, semantic, adaptive, hybrid)
- LLM response caching with IRIS backend
- Comprehensive monitoring and health checks
- RAGAS integration for quality assessment
- Data integrity and reconciliation framework

## Getting Started

1. **Installation**: Follow the setup instructions in the main [`README.md`](../README.md)
2. **Basic Usage**: Start with [`USER_GUIDE.md`](USER_GUIDE.md) for your first RAG pipeline
3. **Development**: See [`DEVELOPER_GUIDE.md`](DEVELOPER_GUIDE.md) for architecture and development setup
4. **Configuration**: Customize your setup using [`CONFIGURATION.md`](CONFIGURATION.md)
5. **Deployment**: Use [`guides/DEPLOYMENT_GUIDE.md`](guides/DEPLOYMENT_GUIDE.md) for production deployment

## CLI Tools

The project includes comprehensive command-line interfaces:

- **[`ragctl`](../ragctl)** - Main CLI tool for RAG operations and management
- **Python Module CLI** - Direct module execution via `python -m iris_rag.cli`
- **Make Commands** - Development and testing automation via [`Makefile`](../Makefile)

See [`CONFIGURATION.md`](CONFIGURATION.md) for detailed CLI usage and examples.

## Historical Documentation

Historical documentation, status reports, and legacy implementations have been archived in [`../archive/archived_documentation/`](../archive/archived_documentation/) to maintain project history while keeping current documentation focused and navigable.

## Getting Help

1. **Start Here**: Check the relevant guide in this documentation structure
2. **Troubleshooting**: Review troubleshooting sections in operational guides
3. **API Details**: Consult [`API_REFERENCE.md`](API_REFERENCE.md) for implementation specifics
4. **Historical Context**: Check archived documentation for project evolution
5. **Configuration Issues**: See [`CONFIGURATION.md`](CONFIGURATION.md) for setup problems
6. **Performance**: Use [`guides/PERFORMANCE_GUIDE.md`](guides/PERFORMANCE_GUIDE.md) for optimization

## Contributing

See [`DEVELOPER_GUIDE.md`](DEVELOPER_GUIDE.md) for:
- Development environment setup
- Architecture overview and design principles
- Coding standards and best practices
- Testing requirements and TDD workflow
- Contribution guidelines and pull request process

## Documentation Navigation Tips

- **For Beginners**: [`USER_GUIDE.md`](USER_GUIDE.md) → [`CONFIGURATION.md`](CONFIGURATION.md) → [`guides/DEPLOYMENT_GUIDE.md`](guides/DEPLOYMENT_GUIDE.md)
- **For Developers**: [`DEVELOPER_GUIDE.md`](DEVELOPER_GUIDE.md) → [`API_REFERENCE.md`](API_REFERENCE.md) → [`reference/`](reference/) technical docs
- **For Operations**: [`guides/DEPLOYMENT_GUIDE.md`](guides/DEPLOYMENT_GUIDE.md) → [`guides/PERFORMANCE_GUIDE.md`](guides/PERFORMANCE_GUIDE.md) → [`guides/SECURITY_GUIDE.md`](guides/SECURITY_GUIDE.md)
- **For Troubleshooting**: [`guides/DOCKER_TROUBLESHOOTING_GUIDE.md`](guides/DOCKER_TROUBLESHOOTING_GUIDE.md) → [`reference/MONITORING_SYSTEM.md`](reference/MONITORING_SYSTEM.md)