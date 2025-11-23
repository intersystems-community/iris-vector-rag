# Changelog

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
