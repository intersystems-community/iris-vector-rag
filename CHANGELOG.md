# Changelog

## [0.5.14] - 2025-11-24

### Added
- **DSPy Optimization Support** for entity extraction (Feature 063)
  - New `optimized_program_path` parameter in `OntologyAwareEntityExtractor.__init__()`
  - Load pre-trained DSPy programs for 31.8% F1 improvement (0.294 → 0.387)
  - Library-first design with clean API parameter (not environment variable)
  - Graceful fallback with clear logging when optimization files unavailable
  - Zero breaking changes (backward compatible optional parameter)

### Testing
- 8/8 contract tests passing for DSPy optimization feature
- 17/17 related entity extraction tests passing
- 7 integration tests created using iris-devtester v1.5.0
- All transformer import errors resolved (torch/torchvision compatibility)

### Documentation
- Usage examples in `examples/optimized_dspy_entity_extraction.py`
- Complete test results in `docs/features/063-dspy-optimization/`
- Updated STATUS.md, PROGRESS.md, TODO.md with Feature 063 completion

### Dependencies
- Updated iris-devtester to v1.5.0 from PyPI (significantly faster)
- Resolved torch/torchvision compatibility (2.4.0/0.19.0)

### Chore
- Removed tools/nodejs/node_modules from git tracking (build artifacts)

## [0.5.10] - 2025-11-23

### Documentation
- Updated CHANGELOG to explicitly document UV environment compatibility fix from v0.5.6
- Clarified that the `intersystems-irispython` UV incompatibility fix (Issue #5) is already included

### Notes
- No code changes - this release updates documentation only
- All fixes from v0.5.9 are included
- UV compatibility fix was already present in v0.5.6 (commit 478d3f1b) and carried forward

## [0.5.9] - 2025-11-23

### Added
- **Custom Metadata Filtering** for multi-tenant RAG deployments (Feature 051 User Story 1)
  - Configure custom filter keys via `storage.iris.custom_filter_keys` in YAML config
  - Extends default 17 metadata fields with enterprise-specific fields (tenant_id, security_level, department, etc.)
  - Multi-tenant isolation with tenant-based document filtering
  - Security classification filtering (confidential, public, etc.)
  - Departmental access control
  - MetadataFilterManager class for centralized validation

### Fixed
- **UV Environment Compatibility** - Fixed intersystems-irispython import failures in UV isolated environments (v0.5.6)
  - Replaced hardcoded `import intersystems_iris.dbapi._DBAPI as iris` with UV-compatible fallback logic
  - `get_iris_dbapi_connection()` now uses `_get_iris_dbapi_module()` with robust path resolution
  - Resolves ImportError in UV's isolated environment even when package is installed
- **IRIS JSON Filtering** - Implemented LIKE-based pattern matching workaround for IRIS Community Edition
  - Handles both `"key":"value"` and `"key": "value"` JSON formats (with/without space)
  - Works around missing JSON_VALUE() and JSON_EXTRACT() SQL functions in Community Edition
- **UV/pytest iris Module Import** - Fixed intersystems-irispython module caching during pytest collection
  - Modified `tests/conftest.py` to prioritize .venv site-packages before pytest collection
  - Resolves import issues for ALL tests using IRIS database connections
- **similarity_search Parameter** - Added support for both `filter` and `metadata_filter` parameter names for backward compatibility

### Security
- SQL injection prevention via field name validation (regex + dangerous pattern checks)
- Values safely escaped with SQL quote doubling
- Tenant isolation verified via E2E integration tests
- Whitelist enforcement for default + custom metadata fields

### Testing
- 13/13 contract tests passing (100% success rate)
  - 8 unit/contract tests for custom field configuration
  - 4 E2E integration tests with real IRIS database
  - 1 MetadataFilterManager unit test
- Multi-tenant isolation verified with cross-tenant queries
- LIKE-based JSON filtering tested against real IRIS database

### Performance
- Filter validation: O(n) where n = number of filter keys (typically < 10)
- SQL pattern matching: LIKE on serialized JSON (acceptable for metadata queries)
- Test execution: 3.60 seconds for full test suite
