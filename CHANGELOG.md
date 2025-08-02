# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Requirements-driven orchestrator architecture for elegant automatic pipeline setup
- Unified Query() API architecture for consistent pipeline interfaces
- Basic reranking pipeline with cross-encoder support
- Comprehensive TDD validation for orchestrator architecture
- Pipeline development guide with best practices and anti-patterns
- Public repository synchronization infrastructure
- Enterprise-grade documentation structure

### Changed
- **BREAKING**: All pipelines now use unified `query()` method as the primary interface
- Vector store ID column handling improved for better database compatibility
- Pipeline registration system enhanced with requirements validation
- Development workflow standardized with SPARC methodology

### Fixed
- Chunking ID collision issues in vector store operations
- IDENTITY column compatibility with InterSystems IRIS
- Vector search TypeError in document processing
- Basic rerank pipeline registration and factory integration

### Deprecated
- Pipeline `execute()` and `run()` methods (use `query()` instead)

### Security
- Comprehensive filtering for public repository synchronization
- Exclusion of internal content, secrets, and sensitive data from public releases

## [0.1.0] - 2024-12-01

### Added
- Initial release of RAG Templates library
- Three-tier API design (Simple, Standard, Enterprise)
- Support for 7 RAG techniques: Basic, ColBERT, CRAG, GraphRAG, HyDE, HybridIFind, NodeRAG
- InterSystems IRIS vector database integration
- JavaScript/Node.js API support
- Docker containerization
- Comprehensive test suite with real PMC document validation
- Performance benchmarking framework
- RAGAS evaluation integration

### Changed
- N/A (Initial release)

### Fixed
- N/A (Initial release)

---

## Release Versioning Strategy

This project follows [Semantic Versioning](https://semver.org/):

- **MAJOR** version for incompatible API changes
- **MINOR** version for backwards-compatible functionality additions  
- **PATCH** version for backwards-compatible bug fixes

### Version Tags
- Development releases: `X.Y.Z-dev.N`
- Release candidates: `X.Y.Z-rc.N`
- Stable releases: `X.Y.Z`

### Release Process
1. Update CHANGELOG.md with release notes
2. Update version in pyproject.toml
3. Create release tag: `git tag -a vX.Y.Z -m "Release vX.Y.Z"`
4. Sync to public repository
5. Create GitHub release with highlights