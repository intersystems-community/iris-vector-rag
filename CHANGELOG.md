# Changelog

## [0.5.2] - 2025-11-12

### Added - Cloud Configuration Flexibility (Feature 058)
- **Environment Variable Support**: Configure IRIS connection via environment variables
  - `IRIS_HOST`, `IRIS_PORT`, `IRIS_USERNAME`, `IRIS_PASSWORD`, `IRIS_NAMESPACE`
  - `VECTOR_DIMENSION` (128-8192), `TABLE_SCHEMA` for cloud deployments
- **12-Factor App Configuration**: Priority order (env > config > defaults)
- **Configuration Source Tracking**: Audit trail showing where each value originated
- **Vector Dimension Flexibility**: Support 128-8192 dimensions for different embedding models
  - 384: SentenceTransformers (default)
  - 1024: NVIDIA NIM, Cohere
  - 1536: OpenAI ada-002
  - 3072: OpenAI text-embedding-3-large
- **Table Schema Configuration**: Configurable schema prefix via `TABLE_SCHEMA` env var
  - AWS: SQLUser schema requirement
  - Azure: RAG schema (default)
  - Local: RAG schema (default)
- **Validation Framework**: Preflight validation for vector dimensions and namespaces
  - `VectorDimensionValidator`: Prevents data corruption from dimension mismatches
  - `NamespaceValidator`: Validates namespace permissions
  - `PreflightValidator`: Orchestrates all validation checks
- **Configuration Entities**: Strongly-typed configuration models
  - `ConnectionConfiguration`, `VectorConfiguration`, `TableConfiguration`
  - `CloudConfiguration` with `.validate()` method
- **Deployment Examples**: Production-ready configuration templates
  - `config/examples/aws.yaml` - AWS IRIS (%SYS namespace, SQLUser schema)
  - `config/examples/azure.yaml` - Azure IRIS (USER namespace, Azure Key Vault)
  - `config/examples/local.yaml` - Local development (Docker-ready)
  - `config/examples/README.md` - Comprehensive deployment guide
- **Password Masking**: Automatic password masking in configuration logs (`***MASKED***`)

### Changed
- `ConfigurationManager.get_cloud_config()`: New method for cloud deployment configuration
- Configuration priority system ensures 100% backward compatibility
- All existing APIs continue to work unchanged (v0.4.x compatible)

### Fixed
- Cloud deployment configuration now properly respects environment variables
- Vector dimension validation prevents data corruption from configuration errors

### Documentation
- Added cloud deployment examples for AWS, Azure, and local environments
- Security best practices for secret management (AWS Secrets Manager, Azure Key Vault)
- Configuration troubleshooting guide with common errors and solutions

### Technical Details
- **Files Added**: 7 new files (~1,500 lines)
  - `iris_vector_rag/config/entities.py` (380 lines)
  - `iris_vector_rag/config/validators.py` (420 lines)
  - Configuration examples and documentation (700 lines)
- **Files Modified**: 2 files
  - `iris_vector_rag/config/manager.py` - Added `get_cloud_config()` method
  - Contract test suite - 18/22 tests passing (82%)
- **Test Coverage**: 18 contract tests passing, all unit tests passing
- **Zero Breaking Changes**: 100% backward compatible with v0.5.1

## [0.5.1] - 2025-11-09

### Fixed
- **CRITICAL**: Fixed packaging error in 0.5.0 where `common` module was installed at top-level of site-packages instead of inside `iris_vector_rag`
- Rebuilt package from clean git state to ensure correct directory structure
- Note: 0.5.0 should not be used - please upgrade to 0.5.1

## [0.5.0] - 2025-11-09 [YANKED - DO NOT USE]

### Changed - BREAKING
- **BREAKING**: Moved `common` module inside `iris_vector_rag` package to resolve namespace conflicts
  - Old: `from common.iris_dbapi_connector import X`
  - New: `from iris_vector_rag.common.iris_dbapi_connector import X`
  - Fixes: ModuleNotFoundError in environments with conflicting `common` packages
  - Impact: Only affects external code directly importing from `common` (rare - not documented as public API)
  - Normal usage via `ConnectionManager` and other public APIs requires no changes

### Fixed
- Fixed critical import error: `ModuleNotFoundError: No module named 'common.iris_dbapi_connector'`
- Resolved namespace conflict causing HippoRAG2 pipeline entity extraction to hang
- Updated 96 import statements across 52 files throughout the codebase
- Package now correctly includes common utilities at `iris_vector_rag.common`

### Technical Details
- Moved `common/` directory to `iris_vector_rag/common/`
- Updated pyproject.toml package configuration
- All 24 common module files now properly namespaced
- Zero performance impact - import paths only change
- Tested with 6 contract tests and full test suite
