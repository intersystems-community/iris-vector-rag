# Changelog

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
