# RAG-Templates Documentation Index

**Last Updated**: 2025-10-08

## 📚 Getting Started

| Document | Description | Audience |
|----------|-------------|----------|
| [README.md](README.md) | Project overview, quick start, API summary | Everyone |
| [USER_GUIDE.md](USER_GUIDE.md) | Step-by-step installation and usage guide | New users |
| [CLAUDE.md](CLAUDE.md) | Development commands and architecture for Claude Code | AI assistants |

## 📖 API Documentation

| Document | Description | Audience |
|----------|-------------|----------|
| [docs/API_REFERENCE.md](docs/API_REFERENCE.md) | Complete API documentation with examples | Developers |
| [TEST_VALIDATION_SUMMARY.md](TEST_VALIDATION_SUMMARY.md) | Testing framework validation (100% pass rate) | QA, Developers |

## 🏗️ Architecture & Integration

| Document | Description | Audience |
|----------|-------------|----------|
| [docs/VALIDATED_ARCHITECTURE_SUMMARY.md](docs/VALIDATED_ARCHITECTURE_SUMMARY.md) | Service boundaries and performance | Architects |
| [docs/INTEGRATION_HANDOFF_GUIDE.md](docs/INTEGRATION_HANDOFF_GUIDE.md) | How to integrate rag-templates | Integrators |
| [docs/PROJECT_COMPLETION_REPORT_VALIDATED.md](docs/PROJECT_COMPLETION_REPORT_VALIDATED.md) | Validated achievements and metrics | Stakeholders |
| [docs/PRODUCTION_READINESS_ASSESSMENT.md](docs/PRODUCTION_READINESS_ASSESSMENT.md) | Deployment requirements | DevOps |

## 🧪 Testing Documentation

| Document | Description | Audience |
|----------|-------------|----------|
| [docs/testing/tdd-compliance.md](docs/testing/tdd-compliance.md) | TDD workflow and compliance checking | Developers |
| [docs/testing/error-messages.md](docs/testing/error-messages.md) | Error message best practices | Developers |
| [docs/testing/coverage-warnings.md](docs/testing/coverage-warnings.md) | Coverage monitoring without build failures | QA |
| [docs/testing/E2E_TEST_STRATEGY.md](docs/testing/E2E_TEST_STRATEGY.md) | End-to-end testing strategy | QA |
| [docs/testing/TEST_EXECUTION_GUIDE.md](docs/testing/TEST_EXECUTION_GUIDE.md) | How to run tests | Developers |

## 🚀 Advanced Features

| Document | Description | Audience |
|----------|-------------|----------|
| [docs/IRIS_GRAPH_CORE_INTEGRATION.md](docs/IRIS_GRAPH_CORE_INTEGRATION.md) | HybridGraphRAG with RRF fusion | Advanced users |
| [docs/IRIS_GLOBAL_GRAPHRAG_INTEGRATION.md](docs/IRIS_GLOBAL_GRAPHRAG_INTEGRATION.md) | Academic papers with 3D visualization | Researchers |
| [docs/PYLATE_COLBERT_TESTING.md](docs/PYLATE_COLBERT_TESTING.md) | PyLate ColBERT testing guide | Developers |

## 🛠️ Development Guides

| Document | Description | Audience |
|----------|-------------|----------|
| [CONTRIBUTING.md](CONTRIBUTING.md) | How to contribute to the project | Contributors |
| [docs/PIPELINE_MIGRATION_STRATEGY.md](docs/PIPELINE_MIGRATION_STRATEGY.md) | Migrating between pipeline versions | Maintainers |
| [docs/BENCHMARKING_CONSOLIDATION_GUIDE.md](docs/BENCHMARKING_CONSOLIDATION_GUIDE.md) | Benchmarking and performance testing | Performance engineers |

## 📊 Status & Progress

| Document | Description | Audience |
|----------|-------------|----------|
| [STATUS.md](STATUS.md) | Current project status | Everyone |
| [PROGRESS.md](PROGRESS.md) | Development progress tracking | Team |
| [TODO.md](TODO.md) | Outstanding tasks | Team |
| [CHANGELOG.md](CHANGELOG.md) | Version history and changes | Everyone |

## 🔍 Specialized Topics

| Document | Description | Audience |
|----------|-------------|----------|
| [docs/PIPELINE_CHUNKING_ARCHITECTURE_REFACTOR.md](docs/PIPELINE_CHUNKING_ARCHITECTURE_REFACTOR.md) | Chunking architecture details | Advanced developers |
| [docs/ontology_integration_guide.md](docs/ontology_integration_guide.md) | Ontology integration patterns | Knowledge engineers |
| [docs/IRIS_SQL_VECTOR_PARAMETERIZATION_BUG_REPORT.md](docs/IRIS_SQL_VECTOR_PARAMETERIZATION_BUG_REPORT.md) | Known issues and workarounds | Developers |

## 📁 Archive

These documents are historical references:

- [docs/UNIFIED_PROJECT_ROADMAP.md](docs/UNIFIED_PROJECT_ROADMAP.md) - Original project roadmap
- [docs/PHASE_3_4_COMPLETION_SUMMARY.md](docs/PHASE_3_4_COMPLETION_SUMMARY.md) - Phase 3-4 summary
- [docs/IMPLEMENTATION_COMPLETION_ROADMAP.md](docs/IMPLEMENTATION_COMPLETION_ROADMAP.md) - Implementation roadmap
- [docs/EXAMPLE_ENHANCEMENT_GUIDE.md](docs/EXAMPLE_ENHANCEMENT_GUIDE.md) - Example scripts guide

## 🎯 Quick Navigation

**I want to...**

- **Get started quickly** → [README.md](README.md) Quick Start section
- **Learn the API** → [docs/API_REFERENCE.md](docs/API_REFERENCE.md)
- **Set up my environment** → [USER_GUIDE.md](USER_GUIDE.md)
- **Integrate rag-templates** → [docs/INTEGRATION_HANDOFF_GUIDE.md](docs/INTEGRATION_HANDOFF_GUIDE.md)
- **Run tests** → [docs/testing/TEST_EXECUTION_GUIDE.md](docs/testing/TEST_EXECUTION_GUIDE.md)
- **Understand the architecture** → [docs/VALIDATED_ARCHITECTURE_SUMMARY.md](docs/VALIDATED_ARCHITECTURE_SUMMARY.md)
- **Deploy to production** → [docs/PRODUCTION_READINESS_ASSESSMENT.md](docs/PRODUCTION_READINESS_ASSESSMENT.md)
- **Use advanced features** → [docs/IRIS_GRAPH_CORE_INTEGRATION.md](docs/IRIS_GRAPH_CORE_INTEGRATION.md)

## 📦 Pipeline-Specific Documentation

### BasicRAG & BasicRAGReranking
- Core functionality: [docs/API_REFERENCE.md](docs/API_REFERENCE.md)
- Usage examples: [USER_GUIDE.md](USER_GUIDE.md)

### CRAG (Corrective RAG)
- API documentation: [docs/API_REFERENCE.md](docs/API_REFERENCE.md)
- Implementation: [iris_rag/pipelines/crag.py](iris_rag/pipelines/crag.py)

### HybridGraphRAG
- Complete guide: [docs/IRIS_GRAPH_CORE_INTEGRATION.md](docs/IRIS_GRAPH_CORE_INTEGRATION.md)
- RRF fusion details: [docs/VALIDATED_ARCHITECTURE_SUMMARY.md](docs/VALIDATED_ARCHITECTURE_SUMMARY.md)

### PyLateColBERT
- Testing guide: [docs/PYLATE_COLBERT_TESTING.md](docs/PYLATE_COLBERT_TESTING.md)
- API reference: [docs/API_REFERENCE.md](docs/API_REFERENCE.md)

### IRIS-Global-GraphRAG
- Full documentation: [docs/IRIS_GLOBAL_GRAPHRAG_INTEGRATION.md](docs/IRIS_GLOBAL_GRAPHRAG_INTEGRATION.md)
- Academic use cases: Research papers with 3D visualization

## 🔗 External Resources

- [InterSystems IRIS Documentation](https://docs.intersystems.com/iris20241/csp/docbook/)
- [LangChain Documentation](https://python.langchain.com/)
- [RAGAS Evaluation Framework](https://docs.ragas.io/)
- [Sentence Transformers](https://www.sbert.net/)

## 📝 Document Maintenance

- All documentation follows Markdown formatting
- Code examples use Python syntax highlighting
- Links are relative paths within the repository
- Last updated dates at top of documents
- Version information where applicable

---

**Need help?** Start with [README.md](README.md) or [USER_GUIDE.md](USER_GUIDE.md)
