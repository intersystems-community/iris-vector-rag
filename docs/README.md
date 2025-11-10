# iris-vector-rag Documentation

**Welcome to the complete documentation for iris-vector-rag** - Enterprise-grade RAG framework powered by InterSystems IRIS.

This index provides organized access to all documentation by topic and user journey.

## Getting Started

**New to iris-vector-rag?** Start here:

| Document | Description | Audience |
|----------|-------------|----------|
| **[User Guide](USER_GUIDE.md)** | Complete installation, configuration, and usage guide | All users |
| **[API Reference](API_REFERENCE.md)** | Detailed API documentation for all pipelines | Developers |
| **[Pipeline Guide](PIPELINE_GUIDE.md)** | Pipeline selection guide - which pipeline for your use case | Developers |

## Advanced Topics

**Deep dives into specific features:**

| Document | Description |
|----------|-------------|
| **[IRIS EMBEDDING Guide](IRIS_EMBEDDING_GUIDE.md)** | Auto-vectorization with model caching, multi-field embeddings |
| **[MCP Integration](MCP_INTEGRATION.md)** | Model Context Protocol setup for Claude Desktop |
| **[IRIS Global GraphRAG](IRIS_GLOBAL_GRAPHRAG_INTEGRATION.md)** | Academic papers with 3D visualization |
| **[IRIS Graph Core Integration](IRIS_GRAPH_CORE_INTEGRATION.md)** | Graph-based retrieval integration |

## Development & Contributing

**For contributors and developers:**

| Document | Description |
|----------|-------------|
| **[Contributing Guide](CONTRIBUTING.md)** | Development setup, testing, pull requests |
| **[Custom Reranker Guide](guides/custom_reranker_guide.md)** | Build custom reranking models |
| **[Example Enhancement Guide](EXAMPLE_ENHANCEMENT_GUIDE.md)** | Extend examples and demos |
| **[Test Setup](TEST_SETUP.md)** | Testing environment configuration |

## Architecture & Design

**System design and architecture documentation:**

| Document | Description |
|----------|-------------|
| **[Comprehensive Architecture Overview](architecture/COMPREHENSIVE_ARCHITECTURE_OVERVIEW.md)** | Complete system architecture |
| **[ColBERT Component Interfaces](architecture/colbert_component_interfaces.md)** | ColBERT integration design |
| **[ColBERT Modular Design](architecture/colbert_modular_design.md)** | Modular ColBERT architecture |
| **[GraphRAG Entity Extraction](architecture/graphrag_entity_extraction_integration.md)** | Entity extraction system |
| **[GraphRAG Service Interfaces](architecture/graphrag_service_interfaces.md)** | GraphRAG service layer |
| **[RAG Memory Architecture](architecture/rag_memory_component_architecture.md)** | Memory component design |
| **[RAG Templates Adapter](architecture/rag_templates_adapter_architecture.md)** | Adapter pattern implementation |

## Testing & Quality

**Test coverage, execution, and quality reports:**

| Document | Description |
|----------|-------------|
| **[E2E Test Coverage Report](testing/E2E_TEST_COVERAGE_REPORT.md)** | End-to-end test coverage |
| **[E2E Test Results](testing/E2E_TEST_RESULTS_SUMMARY.md)** | Test execution results |
| **[Test Execution Guide](testing/TEST_EXECUTION_GUIDE.md)** | How to run tests |
| **[E2E Test Strategy](testing/E2E_TEST_STRATEGY.md)** | Testing approach and methodology |
| **[TDD Compliance](testing/tdd-compliance.md)** | Test-driven development practices |
| **[Coverage Warnings](testing/coverage-warnings.md)** | Coverage gaps and improvements |
| **[Error Messages](testing/error-messages.md)** | Common errors and solutions |

## Migration & Deployment

**Production deployment and migration guides:**

| Document | Description |
|----------|-------------|
| **[Production Readiness Assessment](PRODUCTION_READINESS_ASSESSMENT.md)** | Production checklist |
| **[Pipeline Migration Strategy](PIPELINE_MIGRATION_STRATEGY.md)** | Migrating between pipelines |
| **[Public Repository Sync](PUBLIC_REPOSITORY_SYNC.md)** | Open-source repository management |

## Project Status & Planning

**Roadmaps, progress tracking, and governance:**

| Document | Description |
|----------|-------------|
| **[Unified Project Roadmap](UNIFIED_PROJECT_ROADMAP.md)** | Long-term project roadmap |
| **[Implementation Progress](development/IMPLEMENTATION_PROGRESS.md)** | Current implementation status |
| **[Implementation Status](development/IMPLEMENTATION_STATUS.md)** | Detailed feature status |
| **[Completion Roadmap](IMPLEMENTATION_COMPLETION_ROADMAP.md)** | Path to completion |
| **[Project Completion Report](PROJECT_COMPLETION_REPORT_VALIDATED.md)** | Validated completion analysis |

## Bug Reports & Troubleshooting

**Known issues and workarounds:**

| Document | Description |
|----------|-------------|
| **[IRIS SQL Vector Parameterization Bug](IRIS_SQL_VECTOR_PARAMETERIZATION_BUG_REPORT.md)** | Known IRIS vector query bug |
| **[IRIS Vector SQL Parameterization Repro](reports/IRIS_VECTOR_SQL_PARAMETERIZATION_REPRO.md)** | Bug reproduction steps |
| **[Schema Migration Loop Bug](development/BUG_REPORT_SCHEMA_MIGRATION_LOOP.md)** | Schema migration issue |
| **[E2E Schema Compatibility Fix](E2E_SCHEMA_COMPATIBILITY_FIX.md)** | Schema compatibility resolution |

## Benchmarking & Evaluation

**Performance benchmarks and RAGAS evaluation:**

| Document | Description |
|----------|-------------|
| **[Benchmarking Consolidation Guide](BENCHMARKING_CONSOLIDATION_GUIDE.md)** | Unified benchmarking approach |
| **[PyLate RAGAS Evaluation](PYLATE_RAGAS_EVALUATION.md)** | PyLate ColBERT evaluation results |
| **[PyLate ColBERT Testing](PYLATE_COLBERT_TESTING.md)** | ColBERT testing methodology |

## Design Documents

**Feature design and specification documents:**

| Document | Description |
|----------|-------------|
| **[Comprehensive Reconciliation Design](design/COMPREHENSIVE_GENERALIZED_RECONCILIATION_DESIGN.md)** | Reconciliation system design |
| **[Pseudocode Report Generator](specifications/07_pseudocode_report_generator.md)** | Code generation specifications |

## Archived Documentation

**Historical documents (no longer active):**

| Document | Description |
|----------|-------------|
| **[Graph RAG Templates Roadmap](archived/GRAPH_RAG_TEMPLATES_ROADMAP_ARCHIVED_20250914.md)** | Archived GraphRAG roadmap |

## Quick Navigation

### By User Journey

**Evaluator** (trying out iris-vector-rag):
1. [User Guide](USER_GUIDE.md) - Installation
2. [Pipeline Guide](PIPELINE_GUIDE.md) - Which pipeline to use
3. [API Reference](API_REFERENCE.md) - API details

**Developer** (building with iris-vector-rag):
1. [API Reference](API_REFERENCE.md) - Complete API
2. [IRIS EMBEDDING Guide](IRIS_EMBEDDING_GUIDE.md) - Auto-vectorization
3. [MCP Integration](MCP_INTEGRATION.md) - Claude Desktop integration
4. [Contributing Guide](CONTRIBUTING.md) - Development setup

**Contributor** (improving iris-vector-rag):
1. [Contributing Guide](CONTRIBUTING.md) - Setup and standards
2. [Test Execution Guide](testing/TEST_EXECUTION_GUIDE.md) - Run tests
3. [Architecture Overview](architecture/COMPREHENSIVE_ARCHITECTURE_OVERVIEW.md) - System design

**Enterprise Architect** (evaluating for production):
1. [Production Readiness Assessment](PRODUCTION_READINESS_ASSESSMENT.md) - Deployment checklist
2. [Architecture Overview](architecture/COMPREHENSIVE_ARCHITECTURE_OVERVIEW.md) - System design
3. [E2E Test Coverage](testing/E2E_TEST_COVERAGE_REPORT.md) - Quality metrics

### By Topic

**RAG Pipelines**:
- [Pipeline Guide](PIPELINE_GUIDE.md) - Selection guide
- [User Guide](USER_GUIDE.md) - Usage examples
- [API Reference](API_REFERENCE.md) - API documentation

**IRIS Integration**:
- [IRIS EMBEDDING Guide](IRIS_EMBEDDING_GUIDE.md) - Auto-vectorization
- [IRIS Graph Core](IRIS_GRAPH_CORE_INTEGRATION.md) - Graph integration
- [IRIS Global GraphRAG](IRIS_GLOBAL_GRAPHRAG_INTEGRATION.md) - Advanced GraphRAG

**Advanced Features**:
- [MCP Integration](MCP_INTEGRATION.md) - Model Context Protocol
- [Custom Reranker Guide](guides/custom_reranker_guide.md) - Custom models
- [Ontology Integration](ontology_integration_guide.md) - Domain ontologies

**Production & Quality**:
- [Production Readiness](PRODUCTION_READINESS_ASSESSMENT.md) - Deployment checklist
- [E2E Test Coverage](testing/E2E_TEST_COVERAGE_REPORT.md) - Test metrics
- [Benchmarking Guide](BENCHMARKING_CONSOLIDATION_GUIDE.md) - Performance evaluation

## Contributing to Documentation

Found a documentation issue? Want to improve these docs?

1. Check [Contributing Guide](CONTRIBUTING.md) for documentation standards
2. Submit a pull request with your improvements
3. Tag documentation changes with `docs:` prefix in commit message

## Support

- **Documentation Issues**: [GitHub Issues](https://github.com/intersystems-community/iris-vector-rag-private/issues)
- **Questions**: [GitHub Discussions](https://github.com/intersystems-community/iris-vector-rag-private/discussions)
- **Enterprise Support**: [InterSystems Support](https://www.intersystems.com/support/)

---

**Last Updated**: 2025-11-09
**Documentation Version**: 1.0
